# Recoverability and V5 stage-specific scale — explicit contracts

> **REV4 MODEL SCOPE (2026-10-27):** ERA V5 is a new three-model capstone programme; Session 15 did not disclose final sizes. Do not assume a 1T or 40B V5 target; the 40B case was Session 3 homework and V4's 118.67B is historical evidence. V5 stage-specific tensor/optimizer inventories from G5 determine GPU, checkpoint and transport sizing. Consult `00_TRAINER_INTENT_V5_VS_V4.md` and `07_CONFIG_DRIVEN_MODEL_STAGE_CAPACITY.md`. Existing design controls and Week 1 CPU tests remain applicable independent of model size.


## Three different guarantees

1. **Committed-progress durability (RPO = 0 for acknowledged checkpoints)**: every checkpoint acknowledged as durable after successful all-rank upload and manifest/pointer commit must be readable after losing any one defined failure domain. Local NVMe cannot meet a lost-node durability requirement.
2. **Exact deterministic replay**: after restoring an earlier checkpoint in the **same topology, software and numerical implementation**, the identical batch stream, RNG and optimizer updates reproduce all future loss/weight hashes, unless a kernel/collective is nondeterministic. This does **not** mean zero recomputation.
3. **No recomputation of already executed steps**: preserving a committed update on node failure requires *synchronously durable state for every step* before acknowledging that step (or a correctly replicated surviving hot state). It is not ensured by a 30-minute checkpoint interval. Under arbitrary simultaneous correlated failures or network partitions, a universal guarantee is impossible without stated failure and availability assumptions.

## Options independently

**A. Periodic durable checkpoints + replay**: every C minutes, snapshot weights, optimizer, RNG, scheduler, dataset cursor, state IDs; recover to last remote committed generation and replay intervening steps deterministically. Cost: async stage and write, remote checkpoint traffic; expected recomputation ~C/2 for uniformly distributed failures, worst case ~C. Excellent baseline; cannot promise zero rework.

**B. Durable per-step state**: each optimizer step emits a **consistent** distributed state transition and may be acknowledged only after its recovery state is durably replicated outside the failure domain. The exact state delta still includes changed optimizer moments and weights, typically O(trainable parameters); for Adam, almost all trained parameters may change every step. The data ledger is tiny in comparison. Requires append-only generation/manifest, ordered writer epochs, fsync/remote durability, and flow control. Increases latency and bandwidth substantially. Benchmark; do not assert feasible for the eventual V5 model without measured hardware evidence.

**C. Hot standby replica**: synchronized second training replica or sufficiently complete active optimizer-state mirror in a separate failure domain, with coordinated commit index and leader fencing. Allows low recovery latency from a *single* primary failure if the mirror remains available and state transfer is current. May approximately double parameter-state memory and transmit massive per-step state or execute mirrored compute. It does not guarantee preservation against correlated failures without durable logging.

**D. Combined**: periodic full distributed checkpoints + small durable *data/progress ledger* + hot state replication or per-step durable state log, with fencing. This can offer RPO 0 for *acknowledged* steps under a declared one-node failure model and dramatically reduce rework; otherwise the scheme may offer only exact replay. No free lunch: pay in GPU/network/disk bandwidth, synchronized wait, replica footprint and operational complexity.

## Scale model — stage-specific arithmetic, NOT a benchmark

**No finalized V5 sizes are publicly established in the reviewed instructor transcript.** The proper workload model is an input tuple `(P_stored, P_trainable, P_active, weight/optimizer dtypes, quantized frozen bytes, DP/TP/PP/EP/CP degrees, seq_len, tokens_per_step, target durable checkpoint interval)`. The 14 B/parameter rule applies only to **trainable** BF16 + FP32-master + FP32-Adam state (2+4+4+4 B), not all frozen/quantized MoE base weights. Checkpoint size is determined by actual serialized tensors, optimizer layout and non-tensor state.

The 2026 LightningLM V4 reference had **118.67B stored, 5.93B active** in its 120B release, with a quantized expert base and trained adapters; those do not imply a V5 checkpoint size. The instructor says V5 trains three new models, sizes undisclosed. The user-requested 1T thought experiment is retired as a planning baseline.

A stage snapshot has measured payload `S` bytes and measured sustainable storage bandwidth `R` bytes/s; transfer-only lower bound is `S/R`, excluding all verification, data movement, tail latency and contention. To durably persist a *full* snapshot per optimizer step taking `t` seconds requires average payload bandwidth at least `S/t`; that can be prohibitively expensive. Use **measurements**, not GPU peak bandwidth.

## Recommendation tiers

**Gate 1 (31 Oct)**: implement A reliably and rigorously (CPU model, new-process identical loss/optimizer/dataset IDs, tamper rejection); design hooks for B/C; do NOT claim zero-recompute. Integrate exit/requeue contract with Slurm.

**Gates 2–3**: measure save-stage p50/p95/p99, durable commit latency, load/restart latency, checkpoint size, MTBF, network/utilization impact, data replay, and GPU loss under actual topology. Compare B/C by expected useful tokens per GPU-hour. Require hardware-owner agreement on spare resources and failure model.

**Gate 4**: authorize a bounded SLO (e.g. zero lost *committed checkpoints*, deterministic replay for fixed topology and ≤C minutes re-execution), or if B/C pass measured tests, tighter RPO. Keep RTO separately measured. Continuous exact bitwise GPU replay remains conditional on kernel/collective determinism.

## Stateful replay acceptance data

Include `run_id`, `attempt_id`, `writer_epoch`, `checkpoint_generation`, model + optimizer (+ FP32 master if used), scheduler/scaler, accumulation boundary, LR config, RNG: Python/NumPy/torch CPU/all CUDA devices/all generators, deterministic sampler and workers, exact source shard hashes, packed batch/sample/token IDs, data ledger transaction ID, consumed token count, routing/expert parallel maps, TP/PP/DP/EP/CP topology and global rank map, tokenizer config/version, git SHA/image digest, torch/CUDA/NCCL/compiler/kernels, training and loss dtype, validation probe with packed input hash, and post-restore first-M-step expected hashes. Manifests hold metadata only and must not contain secrets. Use the same inference/training mode and *numerical* kernel for bitwise probes; changing batch, mesh or hardware voids the exact-bitwise guarantee until proven anew.

## Mathematical checkpoint optimization

A heuristic often called Young/Daly minimizes expected waste; naive `sqrt(2 * checkpoint * MTBF)` assumes particular failure/overhead conditions. When snapshots are asynchronous, the relevant cost is **total critical-path stall and throughput degradation**, not just the host copy. Background upload contention, staging buffers and failure-domain persistence affect the optimum. Measure and fit; do not promise a 30-minute optimal interval.


---


## Session 15 and 16 extensions — recovery and goodput

### A. Separate deterministic reproduction from an intervention branch

Session 15 distinguishes a **batch trigger** from the **model state** at which it becomes harmful. Required debugging record: `(checkpoint_generation, committed_ledger_step, ordered_batch_ids, packing_mask_hash, model_config_hash, parameter/optimizer state digest, gradient norm, loss, intervention_id)`. Restarting the **same branch** must reproduce that tuple and subsequent losses on a fixed deterministic CPU run. Deliberately skipping a batch or reducing LR creates a **new branch with a recorded decision**; it cannot honestly be described as replaying the original run unchanged.

### B. Three failure guarantees remain independent

- **Committed progress durability:** S3/object or independent-domain commit acknowledgments survive a specified failure. A local snapshot is not remotely committed merely because it was saved.
- **Exact replay:** under a pinned computation and data topology, restored continuation agrees in loss bit patterns, optimizer and parameter digests, RNG, scheduler and ordered training batches. Per-step equality must be tested; one probe loss is insufficient.
- **Zero recomputation:** no previously computed steps reexecuted. Session 15’s checkpoint/rollback examples explicitly tolerate lost work since last snapshot, so they do **not** support this stronger guarantee. Evaluate per-step durable journal or hot-replica design as a separate optional optimization, with bandwidth and failure-domain cost disclosed.

### C. Goodput capacity math revised by Session 16

Measure training-step MFU on active fault-free training windows; calculate unique-token goodput over the **whole allocated window**. If your MFU denominator already includes downtime, multiplying by goodput again double counts. Report `committed_unique_tokens`, `physically_executed_tokens`, `replayed_tokens`, `checkpoint_stall_time`, `recovery_time`, `evaluation_delay`, and `GPU-hours`. V5 checkpoint size must be derived from its selected stage: stored/frozen/active/trainable parameters and real serialized state. No final size is assumed. Do not budget a reported 16× optimization until independently implemented and measured on fixed-quality baselines.

### D. Recovery runbook branches

`HW_FAULT`: fence attempt, do not snapshot dead GPU, restore remote commit. `NUMERICAL`: detect and hold; inspect last model state and batch, then authorized LR/skip/rollback **new branch**. `STORAGE_CORRUPTION`: reject checkpoint and verify older immutable generation. `EVAL_ANOMALY`: open diagnostic incident; a quantized evaluation derivative is not a state-equality oracle. Every branch outputs an auditable decision record and explicit RPO/RTO/replay count.
