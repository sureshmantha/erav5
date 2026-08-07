from __future__ import annotations

from collections import Counter
from pathlib import Path

from .batches import build_batch
from .hashing import sha256_json
from .io import read_json, read_jsonl
from .shards import validate_manifest, load_records_for_manifest


def replay_interval(plan:dict,catalog:dict,consumption_rows:list[dict],tokenizer,start_step:int,end_step:int)->dict:
    original={r["global_step"]:r for r in consumption_rows}
    items=[]; ok=True
    for step in range(start_step,end_step+1):
        entry=plan["batches"][step-1]
        batch=build_batch(entry,catalog,tokenizer.PAD_ID)
        row=original[step]
        match=(batch.batch_id==row["batch_id"] and batch.batch_hash==row["batch_hash"] and batch.source_spans==row["source_spans"] and batch.sample_ids==row["sample_ids"])
        ok=ok and match
        items.append({"global_step":step,"batch_id":batch.batch_id,"replay_batch_hash":batch.batch_hash,
                      "original_batch_hash":row["batch_hash"],"source_spans_match":batch.source_spans==row["source_spans"],"match":match})
    return {"start_step":start_step,"end_step":end_step,"match":ok,"items":items,"interval_hash":sha256_json(items)}


def audit_run(*,artifact_root:Path,plan:dict,catalog:dict,tokenizer,manifests:list[dict],shard_dir:Path,expected_batch_ids:list[str],resume_proof:dict,replay_proof:dict,fork_proof:dict,performance:dict)->dict:
    consumption=read_jsonl(artifact_root/"ledgers/consumption.jsonl")
    learning=read_jsonl(artifact_root/"ledgers/learning.jsonl")
    token_learning=read_jsonl(artifact_root/"ledgers/token_learning.jsonl")
    opus=read_jsonl(artifact_root/"ledgers/opus_decisions.jsonl")
    manifest_checks=[validate_manifest(m,shard_dir,tokenizer.tokenizer_hash)[0] for m in manifests]
    actual_ids=[r["batch_id"] for r in consumption]
    no_skip_repeat=(actual_ids==expected_batch_ids and len(actual_ids)==len(set((r["global_step"],r["batch_id"]) for r in consumption)))
    train_ids=set(); blocked_ids=set()
    for m in manifests:
        ids={r["record_id"] for r in load_records_for_manifest(m,shard_dir)}
        (train_ids if m["split"]=="train" else blocked_ids).update(ids)
    consumed_ids={span["record_id"] for r in consumption for spans in r["source_spans"] for span in spans}
    eval_in_training=bool(consumed_ids & blocked_ids) or not consumed_ids.issubset(train_ids)
    stage_token_report=[]
    token_floor_ok=True
    for proof in plan["stage_proofs"]:
        stage=proof["stage"]; counts=Counter(); total=0
        for row in consumption:
            if row["stage"]!=stage: continue
            for lane,n in row["lane_token_counts"].items(): counts[lane]+=n; total+=n
        actual={lane:(counts.get(lane,0)/total if total else 0.0) for lane in proof["weights"]}
        stage_ok=all(actual.get(lane,0.0)+1e-12>=floor for lane,floor in proof["floors"].items())
        token_floor_ok=token_floor_ok and stage_ok
        stage_token_report.append({"stage":stage,"planned_weights":proof["weights"],"protected_floors":proof["floors"],"actual_nonpad_token_shares":actual,"token_floor_compliant":stage_ok})
    floor_ok=all(p["floor_compliant"] for p in plan["stage_proofs"]) and token_floor_ok
    opus_statuses={r["final_decision"] for r in opus}
    opus_proxy_path=artifact_root/"manifests/opus_proxy_manifest.json"
    opus_proxy=read_json(opus_proxy_path) if opus_proxy_path.exists() else {}
    opus_gradient_ok=(
        bool(opus_proxy.get("probe_model_hash"))
        and bool(opus_proxy.get("scores_sha256"))
        and len(opus_proxy.get("scores",{}))>0
        and len(opus)>0
        and all(isinstance(r.get("gradient_alignment"),(int,float)) and r.get("gold_proxy_sample_id") for r in opus)
    )
    learning_linked=(len(learning)==len(consumption) and all(l["batch_hash"]==c["batch_hash"] and l["batch_id"]==c["batch_id"] for l,c in zip(learning,consumption)) and len(token_learning)>0)
    packing_ok=all(0<r["packing_utilization"]<=1 and 0<=r["useful_loss_utilization"]<=r["packing_utilization"] for r in consumption)
    throughput_reconstruct=abs(performance["useful_loss_bearing_tokens_per_second"]-(performance["useful_loss_bearing_tokens"]/performance["measured_training_seconds"]))<1e-6 if performance["measured_training_seconds"]>0 else False
    checkpoint_index=read_json(artifact_root/"checkpoints/index.json")
    checkpoint_binding=all(c["next_batch_index"]==c["global_step"] and c["plan_hash"]==plan["plan_sha256"] and c["tokenizer_hash"]==tokenizer.tokenizer_hash for c in checkpoint_index["checkpoints"])
    return {
        "tokenizer_integrity": all(manifest_checks) and all(m["tokenizer_hash"]==tokenizer.tokenizer_hash for m in manifests),
        "immutable_shards": all(m["content_sha256"]==m["rebuild_sha256"] for m in manifests),
        "evaluation_firewall": (not eval_in_training),
        "packing_correctness": packing_ok,
        "mixture_compliance": floor_ok,
        "opus_audit_trail": {"pass": {"accepted","rejected","deferred","floor_override"}.issubset(opus_statuses) and opus_gradient_ok,"statuses":sorted(opus_statuses),"gradient_proxy_verified":opus_gradient_ok,"proxy_scores_sha256":opus_proxy.get("scores_sha256")},
        "consumption_learning_ledgers": learning_linked,
        "checkpoint_ledger_binding": checkpoint_binding and bool(resume_proof.get("ledger_offsets")),
        "crash_recovery": resume_proof.get("matched",False) and resume_proof.get("catalog_hash_match",False) and no_skip_repeat,
        "replay": replay_proof.get("match",False) and replay_proof.get("catalog_hash_match",False),
        "fork": fork_proof.get("pass",False),
        "throughput": throughput_reconstruct and performance["packing_utilization"]>0 and performance["useful_loss_bearing_tokens_per_second"]>0,
        "no_skipped_or_repeated_batches": no_skip_repeat,
        "mixture_stage_report": stage_token_report,
        "counts":{"consumption_rows":len(consumption),"learning_rows":len(learning),"token_learning_rows":len(token_learning),"opus_rows":len(opus)},
    }
