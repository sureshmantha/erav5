from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .audit import audit_run, replay_interval
from .batches import build_batch
from .checkpoint import CheckpointManager
from .corpus import load_documents
from .evidence import build_evidence, write_evidence_md
from .firewall import EvaluationFirewall, EvaluationFirewallError
from .hashing import sha256_file, sha256_json
from .io import read_json, read_jsonl, write_json
from .ledgers import Ledgers
from .logging_utils import EventLogger
from .mixture import compile_schedule
from .model import initialize_model
from .opus import compute_gradient_proxy_scores
from .packing import build_sample_catalog, validate_sample
from .shards import build_shards, load_records_for_manifest, validate_manifest
from .tokenizer import FrozenByteTokenizer
from .training import SimulatedCrash, TrainingStats, train_range


def _prepare_dirs(repo_root:Path):
    artifact=repo_root/"submission_artifacts"; workspace=repo_root/".demo_work"
    if artifact.exists(): shutil.rmtree(artifact)
    if workspace.exists(): shutil.rmtree(workspace)
    for p in [artifact/"manifests",artifact/"ledgers",artifact/"checkpoints",artifact/"shards",workspace]: p.mkdir(parents=True,exist_ok=True)
    return artifact,workspace


def _copy_manifests(src:Path,dst:Path):
    for p in src.glob("*.json"):
        shutil.copy2(p,dst/p.name)


def _cold_rebuild_catalog(artifact:Path, tokenizer:FrozenByteTokenizer, seq_len:int) -> tuple[list,dict,str]:
    """Reconstruct the executable sample catalog from persisted immutable artifacts only."""
    disk_manifests=[read_json(p) for p in sorted((artifact/"manifests").glob("*.manifest.json"))]
    records_by_lane=defaultdict(list)
    for manifest in disk_manifests:
        ok,errors=validate_manifest(manifest,artifact/"shards",tokenizer.tokenizer_hash)
        if not ok:
            raise RuntimeError(f"cold catalog manifest validation failed {manifest['shard_id']}: {errors}")
        if manifest["split"]!="train":
            continue
        EvaluationFirewall.assert_manifest_trainable(manifest)
        for record in load_records_for_manifest(manifest,artifact/"shards"):
            EvaluationFirewall.assert_record_trainable(record)
            records_by_lane[manifest["lane"]].append(record)
    samples,catalog=build_sample_catalog(records_by_lane,seq_len,tokenizer)
    errors={sample.sample_id:validate_sample(sample,tokenizer) for sample in samples}
    bad={k:v for k,v in errors.items() if v}
    if bad:
        raise RuntimeError(f"cold catalog packing validation errors: {bad}")
    catalog_hash=sha256_json([sample.recipe() for sample in sorted(samples,key=lambda x:x.sample_id)])
    return samples,catalog,catalog_hash


def run_demo(repo_root:Path, *, run_tests:bool=True) -> dict:
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    artifact,workspace=_prepare_dirs(repo_root)
    logger=EventLogger(artifact/"run.log")
    logger.info("V5 Training Data Execution System demonstration started")

    source_paths=[repo_root/"run_demo.py",repo_root/"configs/mixture_policy.json",repo_root/"data/documents.jsonl"]+sorted((repo_root/"v5exec").glob("*.py"))+sorted((repo_root/"tests").glob("*.py"))
    code_hashes={str(path.relative_to(repo_root)):sha256_file(path) for path in source_paths}
    code_manifest={"schema_version":1,"files":code_hashes,"aggregate_sha256":sha256_json(code_hashes)}
    write_json(artifact/"manifests/code_manifest.json",code_manifest)
    write_json(artifact/"manifests/environment.json",{
        "schema_version":1,"python_version":sys.version,"torch_version":torch.__version__,"platform":platform.platform(),
        "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"code_manifest_sha256":code_manifest["aggregate_sha256"],
    })
    logger.pass_("execution_code_hash_recorded",code_manifest["aggregate_sha256"][:16])

    tokenizer=FrozenByteTokenizer()
    docs=load_documents(repo_root/"data/documents.jsonl")
    if not tokenizer.verify_roundtrip([d.get("text","") for d in docs if "text" in d]): raise RuntimeError("tokenizer roundtrip failed")
    write_json(artifact/"manifests/tokenizer_manifest.json",{"spec":tokenizer.spec(),"tokenizer_hash":tokenizer.tokenizer_hash,"frozen":True})
    logger.pass_("tokenizer_hash_verified",tokenizer.tokenizer_hash[:16])

    shard_manifest_work=workspace/"manifests"
    manifests=build_shards(docs,tokenizer,artifact/"shards",shard_manifest_work)
    logger.info(f"shards created count={len(manifests)}")
    for m in manifests:
        ok,errors=validate_manifest(m,artifact/"shards",tokenizer.tokenizer_hash)
        if not ok: raise RuntimeError(f"manifest validation failed {m['shard_id']}: {errors}")
    _copy_manifests(shard_manifest_work,artifact/"manifests")
    logger.info("manifests validated")
    logger.pass_("immutable_shard_hashes_verified",f"count={len(manifests)}")

    # Explicitly try to feed both evaluation and validation data into training and require rejection.
    for blocked_split, pass_name in [("eval","eval_shard_blocked"),("validation","validation_shard_blocked")]:
        blocked_manifest=next(m for m in manifests if m["split"]==blocked_split)
        try:
            EvaluationFirewall.assert_manifest_trainable(blocked_manifest)
            raise RuntimeError(f"{blocked_split} firewall failed open")
        except EvaluationFirewallError:
            logger.info(f"{'evaluation' if blocked_split=='eval' else 'validation'} data blocked shard={blocked_manifest['shard_id']}")
            logger.pass_(pass_name,blocked_manifest["shard_id"])

    records_by_lane=defaultdict(list)
    train_manifests=[]
    for m in manifests:
        if m["split"]!="train": continue
        EvaluationFirewall.assert_manifest_trainable(m); train_manifests.append(m)
        for r in load_records_for_manifest(m,artifact/"shards"):
            EvaluationFirewall.assert_record_trainable(r); records_by_lane[m["lane"]].append(r)
    seq_len=64
    samples,catalog=build_sample_catalog(records_by_lane,seq_len,tokenizer)
    sample_errors={s.sample_id:validate_sample(s,tokenizer) for s in samples}
    bad={k:v for k,v in sample_errors.items() if v}
    if bad: raise RuntimeError(f"packing validation errors {bad}")
    catalog_manifest={"seq_len":seq_len,"samples":[s.recipe() for s in sorted(samples,key=lambda x:x.sample_id)]}
    catalog_manifest["catalog_sha256"]=sha256_json(catalog_manifest["samples"])
    write_json(artifact/"manifests/sample_catalog.json",catalog_manifest)
    packing_policies={}
    packing_examples={}
    for policy_name in sorted({s.policy for s in samples}):
        ps=[s for s in samples if s.policy==policy_name]
        capacity=sum(len(s.input_ids) for s in ps)
        nonpad=sum(sum(1 for t in s.input_ids if t!=tokenizer.PAD_ID) for s in ps)
        useful=sum(sum(s.loss_mask) for s in ps)
        packing_policies[policy_name]={
            "sample_count":len(ps),"capacity_tokens":capacity,"nonpad_tokens":nonpad,"loss_bearing_tokens":useful,
            "packing_utilization":nonpad/capacity,"useful_loss_utilization":useful/capacity,
        }
        e=ps[0]
        packing_examples[policy_name]={
            "sample_id":e.sample_id,"sample_hash":e.sample_hash,"source_spans":e.source_spans,
            "input_ids_hash":sha256_json(e.input_ids),"loss_mask_hash":sha256_json(e.loss_mask),
            "attention_mask_hash":sha256_json(e.attention_mask),"position_ids_hash":sha256_json(e.position_ids),
            "segment_ids_hash":sha256_json(e.segment_ids),
        }
    write_json(artifact/"packing_report.json",{
        "schema_version":1,"sequence_length":seq_len,"all_samples_valid":not bad,"validation_error_count":len(bad),
        "policy_count":len(packing_policies),"policies":packing_policies,"example_proofs":packing_examples,
    })
    logger.info(f"batches packed sample_catalog={len(samples)} policies={sorted(set(s.policy for s in samples))}")
    logger.pass_("packing_masks_verified",f"samples={len(samples)}")

    policy=read_json(repo_root/"configs/mixture_policy.json")
    opus_proxy_start=time.perf_counter()
    proxy_alignment=compute_gradient_proxy_scores(samples,tokenizer.VOCAB_SIZE,seq_len,seed=20260802)
    opus_proxy_seconds=time.perf_counter()-opus_proxy_start
    write_json(artifact/"manifests/opus_proxy_manifest.json",proxy_alignment)
    logger.info(f"OPUS gradient proxy computed candidates={len(proxy_alignment['scores'])} seconds={opus_proxy_seconds:.4f} hash={proxy_alignment['scores_sha256'][:16]}")
    logger.pass_("opus_gradient_proxy_verified",proxy_alignment["scores_sha256"][:16])
    plan,decisions,stage_proofs=compile_schedule(policy,samples,slots_per_stage=32,batch_size=4,seed=1667,proxy_alignment=proxy_alignment)
    write_json(artifact/"manifests/mixture_plan.json",plan)
    logger.info(f"mixture compiled stages={len(stage_proofs)} batches={len(plan['batches'])} plan_hash={plan['plan_sha256'][:16]}")
    if not all(p["floor_compliant"] for p in stage_proofs): raise RuntimeError("protected floor violation")
    logger.pass_("protected_floors_verified")

    ledgers=Ledgers(artifact/"ledgers")
    for d in decisions: ledgers.append_opus(d)
    statuses=Counter(d["final_decision"] for d in decisions)
    logger.info(f"OPUS decisions recorded {dict(statuses)}")
    for required in ["accepted","rejected","deferred","floor_override"]:
        if statuses[required]<=0: raise RuntimeError(f"missing OPUS decision class {required}")
    logger.pass_("opus_all_decision_classes_recorded",str(dict(statuses)))

    model=initialize_model(tokenizer.VOCAB_SIZE,seq_len,seed=20260801)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=0.01)
    ckpt=CheckpointManager(artifact/"checkpoints")
    stats=TrainingStats(); run_id="v5-demo-main"
    crash_step=16
    expected_next=None
    wall_start=time.perf_counter()
    try:
        train_range(run_id=run_id,model=model,optimizer=optimizer,plan=plan,catalog=catalog,tokenizer=tokenizer,
                    ledgers=ledgers,checkpoints=ckpt,start_batch=0,end_batch=len(plan["batches"]),checkpoint_every=8,
                    stats=stats,logger=logger,crash_after_step=crash_step)
    except SimulatedCrash as e:
        expected_next=e.expected_next_batch_id
        logger.info("crash simulated and caught by demo supervisor")

    if expected_next is None: raise RuntimeError("deliberate crash did not occur")
    resume_start=time.perf_counter()
    latest=ckpt.latest("main")
    model2=initialize_model(tokenizer.VOCAB_SIZE,seq_len,seed=20260801)
    optimizer2=torch.optim.AdamW(model2.parameters(),lr=2e-3,weight_decay=0.01)
    payload=ckpt.load(latest,model2,optimizer2)
    # Ledger offsets are part of checkpoint identity; fail closed on drift.
    current_offsets=ledgers.offsets()
    for key in ["consumption","learning","token_learning"]:
        if current_offsets[key]!=payload["ledger_offsets"][key]:
            raise RuntimeError(f"ledger offset mismatch on resume: {key}")
    # Simulate process memory loss: reconstruct loader state only from persisted manifests/shards.
    _,resume_catalog,resume_catalog_hash=_cold_rebuild_catalog(artifact,tokenizer,seq_len)
    catalog_hash_match=(resume_catalog_hash==catalog_manifest["catalog_sha256"])
    if not catalog_hash_match:
        raise RuntimeError("cold-resume sample catalog hash mismatch")
    resumed_entry=plan["batches"][payload["next_batch_index"]]
    resumed_batch=build_batch(resumed_entry,resume_catalog,tokenizer.PAD_ID)
    matched=(resumed_batch.batch_id==expected_next)
    resume_latency=time.perf_counter()-resume_start
    if not matched: raise RuntimeError("resume next batch mismatch")
    logger.info(f"run resumed checkpoint_step={payload['global_step']} next={resumed_batch.batch_id}")
    logger.pass_("resume_next_batch_matched",resumed_batch.batch_id)
    resume_proof={"checkpoint_step":payload["global_step"],"expected_next_batch_id":expected_next,"resumed_next_batch_id":resumed_batch.batch_id,"matched":matched,"resume_latency_seconds":resume_latency,"ledger_offsets":payload["ledger_offsets"],"cold_catalog_hash":resume_catalog_hash,"expected_catalog_hash":catalog_manifest["catalog_sha256"],"catalog_hash_match":catalog_hash_match}
    write_json(artifact/"resume_report.json",resume_proof)

    train_range(run_id=run_id,model=model2,optimizer=optimizer2,plan=plan,catalog=resume_catalog,tokenizer=tokenizer,
                ledgers=ledgers,checkpoints=ckpt,start_batch=payload["next_batch_index"],end_batch=len(plan["batches"]),
                checkpoint_every=8,stats=stats,logger=logger,crash_after_step=None,parent_checkpoint_hash=latest["checkpoint_sha256"])

    consumption=ledgers.consumption_rows()
    _,replay_catalog,replay_catalog_hash=_cold_rebuild_catalog(artifact,tokenizer,seq_len)
    replay_proof=replay_interval(plan,replay_catalog,consumption,tokenizer,5,10)
    replay_proof["cold_catalog_hash"]=replay_catalog_hash
    replay_proof["expected_catalog_hash"]=catalog_manifest["catalog_sha256"]
    replay_proof["catalog_hash_match"]=(replay_catalog_hash==catalog_manifest["catalog_sha256"])
    replay_proof["match"]=bool(replay_proof["match"] and replay_proof["catalog_hash_match"])
    write_json(artifact/"replay_report.json",replay_proof)
    logger.info(f"historical stream replayed steps=5..10 match={replay_proof['match']}")
    if not replay_proof["match"]: raise RuntimeError("replay mismatch")
    logger.pass_("replay_hash_matched",replay_proof["interval_hash"][:16])

    # Fork from step 8: same historical next batch, independent optimizer policy and ledgers.
    fork_meta=ckpt.at_step(8,"main")
    fork_model=initialize_model(tokenizer.VOCAB_SIZE,seq_len,seed=20260801)
    fork_opt=torch.optim.AdamW(fork_model.parameters(),lr=2e-3,weight_decay=0.01)
    fork_payload=ckpt.load(fork_meta,fork_model,fork_opt)
    # Preserve checkpoint optimizer state, then make one explicit branch-only hyperparameter change.
    for group in fork_opt.param_groups:
        group["lr"]=1e-3
    fork_ledgers=Ledgers(artifact/"ledgers/fork")
    fork_stats=TrainingStats()
    fork_entry=plan["batches"][fork_payload["next_batch_index"]]
    historical_expected=plan["batches"][8]["batch_id"]
    fork_batch=build_batch(fork_entry,replay_catalog,tokenizer.PAD_ID)
    fork_pass=fork_batch.batch_id==historical_expected
    train_range(run_id="v5-demo-fork",model=fork_model,optimizer=fork_opt,plan=plan,catalog=replay_catalog,tokenizer=tokenizer,
                ledgers=fork_ledgers,checkpoints=ckpt,start_batch=8,end_batch=9,checkpoint_every=1,stats=fork_stats,
                logger=logger,parent_checkpoint_hash=fork_meta["checkpoint_sha256"],branch="fork_lr_half")
    fork_proof={"pass":fork_pass,"parent_checkpoint":fork_meta["path"],"parent_checkpoint_sha256":fork_meta["checkpoint_sha256"],"fork_first_batch_id":fork_batch.batch_id,"historical_expected_batch_id":historical_expected,"branch":"fork_lr_half","branch_learning_rate":1e-3}
    write_json(artifact/"fork_report.json",fork_proof)
    logger.info(f"branch forked parent_step=8 first_batch={fork_batch.batch_id}")
    logger.pass_("fork_lineage_verified")

    measured=stats.train_seconds
    policy_perf={}
    for policy_name in sorted({s.policy for s in samples}):
        ps=[s for s in samples if s.policy==policy_name]
        cap=sum(len(s.input_ids) for s in ps); nonpad=sum(sum(1 for t in s.input_ids if t!=tokenizer.PAD_ID) for s in ps); useful=sum(sum(s.loss_mask) for s in ps)
        policy_perf[policy_name]={"catalog_samples":len(ps),"packing_utilization":nonpad/cap,"useful_loss_utilization":useful/cap}
    wall_elapsed=time.perf_counter()-wall_start
    performance={
        "schema_version":1,"sequence_length":seq_len,"global_batch_samples":4,"optimizer_steps":stats.optimizer_steps,
        "raw_nonpad_tokens":stats.raw_tokens,"useful_loss_bearing_tokens":stats.loss_tokens,"capacity_tokens":stats.capacity_tokens,
        "packing_utilization":stats.raw_tokens/stats.capacity_tokens,
        "useful_loss_utilization":stats.loss_tokens/stats.capacity_tokens,
        "measured_training_seconds":measured,"measured_loader_seconds":stats.loader_seconds,"wall_seconds":wall_elapsed,
        "raw_nonpad_tokens_per_second":stats.raw_tokens/measured,
        "useful_loss_bearing_tokens_per_second":stats.loss_tokens/measured,
        "end_to_end_useful_loss_bearing_tokens_per_second":stats.loss_tokens/wall_elapsed,
        "resume_latency_seconds":resume_latency,"opus_proxy_seconds":opus_proxy_seconds,
        "opus_decision_counts":dict(statuses),"packing_by_policy":policy_perf,
        "formulae":{"packing_utilization":"raw_nonpad_tokens / capacity_tokens","useful_loss_bearing_tokens_per_second":"useful_loss_bearing_tokens / measured_training_seconds","end_to_end_useful_loss_bearing_tokens_per_second":"useful_loss_bearing_tokens / wall_seconds"},
    }
    write_json(artifact/"performance.json",performance)
    logger.info(f"performance measured packing_utilization={performance['packing_utilization']:.4f} useful_loss_tokens_per_s={performance['useful_loss_bearing_tokens_per_second']:.2f}")

    expected_ids=[x["batch_id"] for x in plan["batches"]]
    audit=audit_run(artifact_root=artifact,plan=plan,catalog=catalog,tokenizer=tokenizer,manifests=manifests,shard_dir=artifact/"shards",
                    expected_batch_ids=expected_ids,resume_proof=resume_proof,replay_proof=replay_proof,fork_proof=fork_proof,performance=performance)
    write_json(artifact/"audit_report.json",audit)
    write_json(artifact/"mixture_report.json",{"stages":audit["mixture_stage_report"]})
    evidence=build_evidence(audit,artifact,resume_proof,replay_proof,fork_proof)
    write_json(artifact/"evidence.json",evidence); write_evidence_md(artifact/"evidence.md",evidence)
    logger.info("audit completed")
    if evidence["overall_result"]!="PASS":
        logger.fail("audit_completed",json.dumps(evidence["requirements"],sort_keys=True)); raise RuntimeError("one or more evidence requirements failed")
    logger.pass_("audit_completed","overall=PASS")

    if run_tests:
        env=dict(os.environ); env["V5_REPO_ROOT"]=str(repo_root)
        result=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=repo_root,env=env,text=True,capture_output=True)
        with (artifact/"run.log").open("a",encoding="utf-8") as f:
            f.write("\n===== AUTOMATED TESTS =====\n"+result.stdout+result.stderr+"===== END TESTS =====\n")
        if result.returncode!=0:
            print(result.stdout); print(result.stderr,file=sys.stderr); raise RuntimeError("automated tests failed")
        logger.pass_("automated_tests_passed")

    logger.info("V5 demonstration complete")
    return {"artifact_root":str(artifact),"evidence":evidence,"performance":performance}
