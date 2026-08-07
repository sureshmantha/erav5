# V5 Training Data Execution Evidence

**Overall:** PASS

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | **PASS** | manifests/tokenizer_manifest.json, manifests/*.manifest.json |
| Immutable shards and manifests | **PASS** | shards/*.jsonl, manifests/*.manifest.json |
| Evaluation / validation firewall | **PASS** | run.log, manifests/eval__*.manifest.json, manifests/validation__*.manifest.json |
| Packing correctness | **PASS** | packing_report.json, performance.json, ledgers/consumption.jsonl |
| Mixture compliance | **PASS** | manifests/mixture_plan.json, mixture_report.json |
| OPUS audit trail | **PASS** | manifests/opus_proxy_manifest.json, ledgers/opus_decisions.jsonl |
| Learning trace | **PASS** | ledgers/consumption.jsonl, ledgers/learning.jsonl, ledgers/token_learning.jsonl |
| Checkpoint / ledger binding | **PASS** | checkpoints/index.json, resume_report.json |
| Crash recovery | **PASS** | checkpoints/index.json, run.log |
| Replay | **PASS** | replay_report.json, ledgers/consumption.jsonl |
| Fork | **PASS** | fork_report.json, checkpoints/index.json |
| Throughput | **PASS** | performance.json, ledgers/consumption.jsonl |
| No skipped/repeated batches | **PASS** | ledgers/consumption.jsonl, manifests/mixture_plan.json |

> This file is generated from runtime audits. Results are not hardcoded.
