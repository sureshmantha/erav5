from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_json


def build_evidence(audit:dict,artifact_root:Path,resume_proof:dict,replay_proof:dict,fork_proof:dict)->dict:
    def item(passed:bool,evidence:list[str],details:dict|None=None):
        d={"result":"PASS" if passed else "FAIL","evidence":evidence}
        if details: d["details"]=details
        return d
    req={
        "Tokenizer integrity":item(audit["tokenizer_integrity"],["manifests/tokenizer_manifest.json","manifests/*.manifest.json"]),
        "Immutable shards and manifests":item(audit["immutable_shards"],["shards/*.jsonl","manifests/*.manifest.json"]),
        "Evaluation / validation firewall":item(audit["evaluation_firewall"],["run.log","manifests/eval__*.manifest.json","manifests/validation__*.manifest.json"]),
        "Packing correctness":item(audit["packing_correctness"],["packing_report.json","performance.json","ledgers/consumption.jsonl"]),
        "Mixture compliance":item(audit["mixture_compliance"],["manifests/mixture_plan.json","mixture_report.json"]),
        "OPUS audit trail":item(audit["opus_audit_trail"]["pass"],["manifests/opus_proxy_manifest.json","ledgers/opus_decisions.jsonl"],{"statuses":audit["opus_audit_trail"]["statuses"],"gradient_proxy_verified":audit["opus_audit_trail"].get("gradient_proxy_verified"),"proxy_scores_sha256":audit["opus_audit_trail"].get("proxy_scores_sha256")}),
        "Learning trace":item(audit["consumption_learning_ledgers"],["ledgers/consumption.jsonl","ledgers/learning.jsonl","ledgers/token_learning.jsonl"]),
        "Checkpoint / ledger binding":item(audit["checkpoint_ledger_binding"],["checkpoints/index.json","resume_report.json"],{"ledger_offsets":resume_proof.get("ledger_offsets")}),
        "Crash recovery":item(audit["crash_recovery"],["checkpoints/index.json","run.log"],resume_proof),
        "Replay":item(audit["replay"],["replay_report.json","ledgers/consumption.jsonl"],{"interval_hash":replay_proof.get("interval_hash")}),
        "Fork":item(audit["fork"],["fork_report.json","checkpoints/index.json"],fork_proof),
        "Throughput":item(audit["throughput"],["performance.json","ledgers/consumption.jsonl"]),
        "No skipped/repeated batches":item(audit["no_skipped_or_repeated_batches"],["ledgers/consumption.jsonl","manifests/mixture_plan.json"]),
    }
    overall=all(v["result"]=="PASS" for v in req.values())
    return {"schema_version":1,"overall_result":"PASS" if overall else "FAIL","requirements":req,"audit_counts":audit["counts"]}


def write_evidence_md(path:Path,evidence:dict)->None:
    lines=["# V5 Training Data Execution Evidence","",f"**Overall:** {evidence['overall_result']}","", "| Requirement | Result | Evidence |","|---|---|---|"]
    for name,row in evidence["requirements"].items():
        lines.append(f"| {name} | **{row['result']}** | {', '.join(row['evidence'])} |")
    lines += ["","> This file is generated from runtime audits. Results are not hardcoded.",""]
    path.write_text("\n".join(lines),encoding="utf-8")
