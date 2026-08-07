# Bharat V5 Training Data Execution System

A small, complete, reproducible reference implementation of the **V5 training-data execution path**:

```text
documents
  -> frozen tokenizer
  -> immutable tokenized shards + manifests
  -> evaluation firewall
  -> curriculum / mixture + protected floors
  -> OPUS-style gradient-alignment selection decisions
  -> data-type-aware packing
  -> deterministic batches
  -> tiny real language-model training
  -> consumption + learning + token-loss ledgers
  -> checkpoint bound to ledger offsets
  -> deliberate crash
  -> exact resume
  -> historical replay
  -> checkpoint fork
  -> audit + performance + evidence bundle
```

The objective is **data-system correctness**, not model scale. The demo uses a 260-token UTF-8 byte tokenizer, 64-token sequences, and a tiny CPU PyTorch causal LM so a reviewer can regenerate every artifact locally. The production policy remains the 40B/6T Bharat V5 mixture specification in [`docs/Bharat_V5_40B_Mixture_Curriculum_README.md`](docs/Bharat_V5_40B_Mixture_Curriculum_README.md).

## One command

```bash
python run_demo.py
```

Requirements: Python 3.10+ and PyTorch (`pip install -r requirements.txt`). The command deletes and regenerates `submission_artifacts/`, runs the complete demonstration, deliberately crashes at a durable checkpoint, resumes, replays, forks, audits the run, and executes the automated test suite.

Expected final line:

```text
Overall evidence: PASS
```

## Architecture and design decisions

| Subsystem | Implementation | Invariant proved |
|---|---|---|
| Frozen tokenizer | `v5exec/tokenizer.py` | Same vocabulary/spec hash for every stage and shard; exact UTF-8 round trip |
| Immutable shards | `v5exec/shards.py` | Canonical JSONL, content SHA-256, deterministic rebuild SHA-256, read-only shard files |
| Manifests | `v5exec/shards.py` | Split, lane, source hashes, cleaning hash, license/provenance, tokenizer hash, gate status |
| Firewall | `v5exec/firewall.py` | Non-`train` manifests and records fail closed; eval/validation IDs are audited after training |
| Packing | `v5exec/packing.py` | Best-fit text, greedy code, structure-preserving agent traces, contiguous long-context chunks |
| Masks | `v5exec/packing.py` | Causal/block attention, role-aware loss masking, PAD masking, position reset at isolated boundaries |
| Mixture | `v5exec/mixture.py` | S0–S5 weights projected from the 40B plan; protected floors enforced before batching |
| OPUS trail | `v5exec/opus.py`, `v5exec/mixture.py` | Frozen tiny-probe LM-head gradient cosine against per-lane golden proxies, then runtime-derived `accepted`, `deferred`, `rejected`, and `floor_override` decisions |
| Batches | `v5exec/batches.py` | Batch IDs/hashes bind sample hashes, token tensors, masks, positions and source spans |
| Training | `v5exec/model.py`, `training.py` | Real next-token cross-entropy, sample loss, token loss/perplexity and gradient norm |
| Ledgers | `v5exec/ledgers.py` | Append-only consumption, learning, token-learning, OPUS and fork ledgers |
| Checkpoint | `v5exec/checkpoint.py` | Model + optimizer + RNG + next-batch cursor + ledger offsets + plan/tokenizer hashes |
| Resume | `v5exec/demo.py` | The first post-crash batch ID is compared to the pre-crash expected next batch |
| Replay | `v5exec/audit.py`, `v5exec/demo.py` | Catalog is rebuilt from persisted shards; historical batches are reconstructed and IDs, source spans and SHA-256 hashes must match |
| Fork | `v5exec/demo.py` | Restore an earlier model/optimizer checkpoint, then change only branch LR and train independently |
| Audit/evidence | `v5exec/audit.py`, `evidence.py` | Evidence is computed from generated manifests/ledgers/reports, never hardcoded; execution code/environment hashes are also recorded |

### Packing policies

1. **General / Indic / multilingual / math / science / reasoning — best fit.** Documents are packed tightly, but each document/chunk receives a separate segment ID. The attention matrix is causal **and block-isolated**, preventing an unrelated packed document from becoming context for another.
2. **Code — greedy structure-aware packing.** Code chunks are filled quickly in deterministic file order; chunk/file boundaries remain attention-isolated.
3. **Agentic — structure preserving.** One trace occupies one sample. User text and tool observations are visible as context but carry zero loss; assistant planning, tool calls, revisions and final answers carry loss.
4. **Long context — one document, contiguous spans.** Long examples are chunked into contiguous source-token spans and are never mixed with unrelated documents.

### Determinism

The stream does **not** depend on Python's process-randomized hash or a mutable dataloader RNG. Ordering is derived from SHA-256 (`stable_u64`) over explicit seeds and immutable IDs. A batch hash includes the complete token tensor, loss mask, attention mask, position IDs and source spans. Replay therefore verifies the data that actually reached the model, not only a random seed.

### OPUS-style selector

The demonstration implements the assignment's acceptance/rejection/deferral/override contract without claiming to reproduce any unpublished production selector. It first freezes a tiny probe language model and chooses the highest-quality sample in each capability lane as that lane's golden proxy. For every candidate, `v5exec/opus.py` computes the **actual LM-head gradient** and its cosine alignment with the lane proxy gradient. The selector then combines that measured alignment with quality, token novelty, curriculum difficulty fit and replay pressure. Thresholds produce `accepted`, `deferred` and `rejected`; when selection would cross a protected lane floor, the final decision becomes `floor_override`. The probe model hash, golden sample IDs, every gradient-alignment score and a scores SHA-256 are written to `submission_artifacts/manifests/opus_proxy_manifest.json`; every decision also records its proxy, score, reason, lane, stage, sample, shard IDs and occurrence.

### Crash, resume and exactly-once consumption

A training step is committed in this order:

```text
build deterministic batch
-> forward/backward/optimizer step
-> append consumption + learning + token-learning records
-> periodically save atomic checkpoint with exact ledger offsets and next batch cursor
```

The demo crashes **immediately after step 16 is durably checkpointed and before step 17 is consumed**. Resume verifies the checkpoint hash, tokenizer hash, plan hash and every ledger offset, **cold-rebuilds the sample catalog from the persisted read-only shards/manifests**, rebuilds the batch at the saved cursor, and proves it equals the expected pre-crash next batch. The final audit then compares all consumed batch IDs to the complete plan to detect any skip or repeat.

## Generated submission artifacts

`python run_demo.py` creates:

```text
submission_artifacts/
  run.log
  evidence.json
  evidence.md
  audit_report.json
  mixture_report.json
  resume_report.json
  replay_report.json
  fork_report.json
  packing_report.json
  performance.json
  shards/
  manifests/
    code_manifest.json
    environment.json
  ledgers/
    consumption.jsonl
    learning.jsonl
    token_learning.jsonl
    opus_decisions.jsonl
    fork/
  checkpoints/
```

The required evidence table is generated into `evidence.md`. `performance.json` reports both **packing utilization** and **useful loss-bearing tokens/second**, with formulas and underlying counts so the values can be reconstructed.

## Automated tests

The one-command demo runs them automatically. They can also be run separately:

```bash
python -m unittest discover -s tests -v
```

Tests cover tokenizer determinism, shard tamper detection, evaluation/validation firewall, packing/mask/position correctness, mixture-floor and OPUS behavior, deterministic batch reconstruction, checkpoint cursor/ledger binding, and integration evidence.

## What is production policy vs. demo machinery?

The **production policy** is still the 40B-class, 6T-token plan: 131,072-token Brahmic-safe tokenizer; S0–S5 curriculum; ~20% India/Indic; explicit code/agentic/reasoning/long-context lanes; protected floors; and the quarantined late anneal/cooldown reserve. This repository intentionally shrinks only the model, tokenizer and token counts needed to exercise the execution machinery. It does **not** reinterpret the production percentages as model-quality claims.

See [`docs/ASSIGNMENT_TRACEABILITY.md`](docs/ASSIGNMENT_TRACEABILITY.md) for a criterion-by-criterion implementation map.
