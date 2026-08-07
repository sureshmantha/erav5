# Assignment Traceability Matrix

This file maps every grading requirement to executable code, generated proof, and automated verification. The evidence files are produced by `python run_demo.py`; they are not static declarations.

| Assignment requirement | Implementation | Generated evidence | Automated verification |
|---|---|---|---|
| End-to-end `documents -> ... -> audit` | `v5exec/demo.py` orchestrates every subsystem | `submission_artifacts/run.log`, `evidence.json` | integration evidence test |
| Immutable tokenized shards | `v5exec/shards.py` canonical JSONL + deterministic rebuild + file lock | `shards/*.jsonl`, `manifests/*.manifest.json` | tamper-detection test |
| Frozen tokenizer + hashes | `v5exec/tokenizer.py` | `manifests/tokenizer_manifest.json`; tokenizer hash in each shard/checkpoint/ledger | tokenizer round-trip/hash test |
| Execution identity | `v5exec/demo.py` hashes executable code/config/fixture and records environment | `manifests/code_manifest.json`, `manifests/environment.json` | regenerated on every run |
| Different packing policies | `v5exec/packing.py` | `packing_report.json`, `manifests/sample_catalog.json` | four-policy test |
| Loss mask / attention mask / position IDs | `v5exec/packing.py`, `v5exec/batches.py` | packing proof hashes + consumption mask hashes | causal/block-mask, agentic-loss, position-reset test |
| Curriculum S0-S5 + lane weights | `configs/mixture_policy.json`, `v5exec/mixture.py` | `manifests/mixture_plan.json`, `mixture_report.json` | deterministic plan test |
| Protected floors | `v5exec/mixture.py`; final token-share audit in `v5exec/audit.py` | stage proofs + planned-vs-actual token shares | floor test + runtime audit |
| Evaluation + validation firewall | `v5exec/firewall.py`, admission checks and final source-ID audit | blocked events in `run.log`; eval/validation manifests; `audit_report.json` | fail-closed test for both splits |
| OPUS acceptance/rejection/deferral | `v5exec/opus.py`, `v5exec/mixture.py` | `manifests/opus_proxy_manifest.json`, `ledgers/opus_decisions.jsonl` | real-autograd proxy + all decision classes test |
| Protected-floor OPUS override | `v5exec/mixture.py` | `floor_override` decision records with reason and source IDs | schedule/OPUS test |
| Consumption ledger | `v5exec/training.py`, `v5exec/ledgers.py` | `ledgers/consumption.jsonl` | final full-plan equality audit |
| Learning ledger | `v5exec/training.py`, `v5exec/ledgers.py` | `ledgers/learning.jsonl` | batch ID/hash join audit |
| Token/sample loss tracking | `v5exec/model.py`, `v5exec/training.py` | sample losses in learning ledger; `ledgers/token_learning.jsonl` | learning-trace audit |
| Checkpoint tied to ledger offsets | `v5exec/checkpoint.py` | checkpoint files/manifests/index; `resume_report.json` | checkpoint binding test |
| Deliberate crash | `v5exec/training.py`, `v5exec/demo.py` | crash event after durable step 16 in `run.log` | integration test |
| Exact resume | `v5exec/demo.py` cold-rebuilds catalog from persisted shards and validates ledger offsets | `resume_report.json` | exact expected/resumed batch + catalog hash test |
| No skipped/repeated batches | `v5exec/audit.py` compares complete planned and consumed streams | `audit_report.json`, `evidence.json` | integration test |
| Historical replay | `v5exec/demo.py`, `v5exec/audit.py` cold-rebuild catalog then reconstruct steps 5-10 | `replay_report.json` | IDs, hashes, sample IDs and source spans must match |
| Fork earlier checkpoint | `v5exec/demo.py` restores model+optimizer from main step 8 then changes branch LR only | `fork_report.json`, fork ledgers/checkpoint | integration test |
| Packing utilization | training and catalog counters | `packing_report.json`, `performance.json` | reconstructible formula audit |
| Useful loss-bearing tokens/sec | timed real tiny-model training | `performance.json` | reconstructible formula audit |
| Generated execution log | `v5exec/logging_utils.py` | `run.log` | required events emitted by real operations |
| Machine-readable evidence bundle | `v5exec/evidence.py` consumes runtime audit booleans | `evidence.json`, `evidence.md` | overall PASS required before demo exits successfully |
| Automated tests | `tests/test_invariants.py` | test section appended to `run.log` | 9 tests executed by `run_demo.py` |

## Important scope boundary

The execution demo deliberately scales down **model size, tokenizer vocabulary, context length, corpus size and stage duration**. It does not change the production constitution in `Bharat_V5_40B_Mixture_Curriculum_README.md`: the 40B/6T policy, S0-S5 lane weights, protected floors, Indic strategy, late reserve, and 1B/3B proxy gates remain production decisions. The demo proves that those decisions can be represented, executed, recovered and audited correctly.
