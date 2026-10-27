# G7 Checkpoint and exact-replay design v0.3

> **REV4 MODEL SCOPE (2026-10-27):** ERA V5 is a new three-model capstone programme; Session 15 did not disclose final sizes. Do not assume a 1T or 40B V5 target; the 40B case was Session 3 homework and V4's 118.67B is historical evidence. V5 stage-specific tensor/optimizer inventories from G5 determine GPU, checkpoint and transport sizing. Consult `00_TRAINER_INTENT_V5_VS_V4.md` and `07_CONFIG_DRIVEN_MODEL_STAGE_CAPACITY.md`. Existing design controls and Week 1 CPU tests remain applicable independent of model size.


**Status: proposed design, not validated on a distributed GPU cluster.** This refines the attached checkpoint draft and ADR 0001; it does not claim that the accompanying CPU *reference simulator* implements distributed DCP, S3 uploads, ZeRO, or production restart.

## Consistency contract

**A checkpoint is an immutable, coherent optimizer-step boundary snapshot.** Acknowledgment requires every expected rank's shard, metadata, data-commit cursor and manifest to be durably readable in independent failure-domain storage. *No lost committed checkpoint* is the Week-1 hard guarantee. **Zero recomputation** is NOT implemented by periodic snapshots. Exact continuation is proven only by deterministic A/B replay on a fixed topology and implementation.

If interrupting halfway through accumulation, either wait until accumulation completes (preferred), or record *all* partially accumulated gradients, microbatch index, partially consumed data, loss-scaler and RNG for every rank; do not silently discard this state while claiming exact replay.

## Immutable generation and fencing

A checkpoint identifier `(run_id, writer_epoch, global_step, generation_uuid)` is unique. Each rank writes only into its own immutable generation prefix to avoid overwrite. The **coordinator may seal only after all expected rank outputs are present**. Workers write a validated rank receipt `{rank, state_bytes, tensor_metadata_hash, file_sha256, data_commit_id, step, epoch}`. The coordinator hashes all files and checks rank/count/topology; emits a manifest. Only after the manifest itself is verifiably durable can coordinator CAS-update `latest` using S3's `PutObject If-Match` and the previous ETag. Lease epoch and monotonic committed step are checked to reject stale/incomparable writers. On 409/412/conflict, abort/promote through reconciliation, not blind overwrite.

This is an *application-level distributed commit protocol*, not an atomic multi-object S3 transaction; crash consistency relies on the final immutable manifest + pointer rule. A directory `rename` on one local filesystem may be atomic; across S3 objects and nodes it is **not**. The local fast checkpoint is a cache until promoted remotely; losing its node might lose it.

## Save state contract

**Per-rank**: model and optimizer shards, master weights if used, local RNG torch/CUDA/NumPy/Python, data-loader worker state or stateless per-sample RNG counters, sampler cursor/packing position, microstep boundary, rank/mesh map, local expert mapping, adaptive precision/scaler details, forward/optimizer checksums. **Global**: LR scheduler, step, committed token/sample IDs, ledger offset, tokenizer/dataset manifest hash, training config, compile flags, matmul precision policy, CUDA/NCCL versions, container digest, loss probe input checksum and ordered expected future batch IDs. Secrets and raw sensitive data never enter manifest.

Checkpoint bytes reflect *actual* model and optimizer state, not a universal 14 B/param rule. MoE inactive experts still hold weights/optimizer state. Record optimizer tensor count and dtype, snapshot compression codec if lossless, and frozen parameter group separately.

## GPU-aware staging

Use pinned CPU buffers and an asynchronous stager **only as allowed by the installed PyTorch version**. `torch.distributed.checkpoint.async_save` and the newer DefaultStager have different synchronization requirements. If staging is truly async, optimizer must **not mutate the source** until staging completion is guaranteed; writing/upload can then proceed in the background. Track `stall_ms`, `staging_ms`, `write_ms`, `remote_commit_ms`, `bytes`, `CPU RAM peak`, `GPU bandwidth throughput impact`. A bounded queue (normally 1 outstanding) prevents unbounded RAM accumulation. At model scale, planned host RAM and scratch NVMe must hold checkpoint buffers; otherwise throttle.

TorchTitan's newer `torch_checkpointing` is a candidate for performance evaluation versus DCP; benchmark under G5's chosen stack before changing format. No framework is accepted merely for benchmark headline figures.

## Restore protocol

1. Read durable current `latest` (pointer includes generation, epoch, step, manifest SHA) with consistent read; check run ID, monotonic step, writer term.
2. Fetch immutable manifest and validate expected rank count/topology and **all hashes** (including DCP metadata, per-rank non-tensor state).
3. Check signed/pinned software stack, tensor schemas, optimizer param IDs, dtype/layout, dataset/tokenizer snapshot, expert and TP/PP/EP/DP/CP mapping.
4. Construct target model + optimizer in required topology and **load strictly**. DCP can reshard compatible tensors, but cross-topology optimizer/expert mapping may require a tested conversion, not an automatic assumption.
5. Restore RNG, scheduler/scaler, progress ledger and exactly the *next* consumed batch IDs; validate probe in the same compute mode/dtype.
6. Test M-step deterministic equivalence of loss `.hex()`/bit-pattern, parameter+optimizer tensor hash, batch IDs and LR/RNG; fail closed upon mismatch for 'exact' mode.
7. A changed world size, GPU architecture or kernel implementation creates a **new non-bitwise-comparable branch** unless proved otherwise; do not silently claim identical continuation.
8. Emit `checkpoint.restored` event with recovered generation and the highest durable progress, replay count, elapsed RTO and potential rework.

**Rollback:** if latest is corrupt, reject, alert, and choose older independently verified checkpoint through an auditable recovery state machine. Never delete or rewrite the corrupt generation during investigation. Verify that data ledger and restored optimizer step remain coherent.

## Negative test matrix

C1: K+M straight versus K save, *new process*, M resume, every step bit-equal and state hashes identical (fixed layout 1 CPU rank).
C2: kill after shard writes but before durable manifest -> previous `latest` unchanged.
C3: corrupt byte in shard or metadata -> checksum rejection; previous retained.
C4: stale writer attempts to publish older generation -> CAS/lease rejection.
C5: unknown or missing model/optimizer key -> strict failure.
C6: missing RNG/scheduler/data sampler -> deterministic test detects mismatch.
C7: stop during gradient accumulation -> either safe delayed boundary or fully recorded partial state.
C8: fail during S3 upload, HEAD/GET transient errors -> do not promote.
C9: 4→1 process DCP restore for supported optimizer layout; numerical probe equality; future bit-equality explicitly **not promised**.
C10: simulate node loss and S3 unavailable -> RTO/RPO measured against declared policy; no stale-local-only checkpoint considered durable.
C11: corruption after manifest before latest -> recovery fails closed.
C12: test with random seeds, GPU SDC fault fixtures, and live rank kill week 4.

**Gate 1 executable code:** `ckpt/ckpt.py` implements a *CPU-only*, single-process snapshot, integrity checks, atomic `latest` update, full optimizer/RNG/cursor restore, and comparison in a fresh process. The program is an acceptance harness for checkpoint semantics only; it is **not** a replacement for production DCP distributed checkpointing.

## Sources

- https://docs.pytorch.org/docs/main/distributed/checkpoint.html
- https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html
- https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html
- https://pytorch.org/blog/torchtitan-update-new-sota-models-performance-at-scale-titanrl/


---


## Session 15 checkpoint hardening and Session 16 capacity changes

### New data and branch-consistency contract

An optimizer-step checkpoint must durably bind `(run_id, branch_id, attempt_epoch, step, committed_token_cursor, ordered_data_batch_digest, tokenizer/mixture/packing/mask_hash, dataset_manifest_hash, replay_policy_version, skip_list_decision_id)` to its model and optimizer state. Saving the RNG without the **data consumption ledger and skip policy** is incomplete. A rollback after discovering a bad-batch/weak-state interaction is a **new traceable branch**, not a modification of the old immutable checkpoint.

### Numerical stability and checkpoint quality

Save a finite-state/optimizer canary and grad-norm telemetry. Do not overwrite known-good checkpoint for NaN, Inf or verified divergence. Gradient-norm alerts are diagnostic signals; their sensitivity and false-alarm rate must be measured. Never infer the cause of a spike solely from input text; compare the *same batch* at different verified model generations when authorized. Model interventions (clip/KQ norm, warmup, LR changes, freezing) belong to Team 5 policy and create versioned config hashes when applied.

### Asynchronous save: precise PyTorch qualification

`torch.distributed.checkpoint.async_save` historically stages from GPU to host synchronously then writes asynchronously. In newer PyTorch versions `DefaultStager`/`StagingOptions(use_async_staging=True)` can overlap device-to-host staging and computation but require careful synchronization, state lifetime and version tests. Do **not** state that all GPU→host copying is always free or that the only cost is a fixed stall. Source: https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html (PyTorch tutorial updated 2026-02-03). Benchmark true step-time interference, host/RAM usage and throughput.

### Preflight checkpoint transfer test (explicit instructor request)

Before any trained model exists, instantiate **random weights with Team 5's exact tensor shapes and optimizer state**, including frozen vs trainable groups and sharded ownership. Measure device→CPU stage; immutable local write+fsync; file checksums; upload/copy; conditional remote commit; cold download; strict load; probe; M-step replay. Report p50/p95 and total source/target bytes. The estimate 14 B/param is only valid for the declared fully-trainable BF16+FP32 master+Adam state example. For frozen quantized experts or adapters use actual tensor/optimizer byte inventories.

### New acceptance tests

`C-S15-1`: save/restart reproduces next N batch IDs and masks as well as loss. `C-S15-2`: authorized skip and LR intervention reproduces after a *branch* restart; old branch remains immutable. `C-S15-3`: running numerical alarm does not corrupt durable latest. `C-S15-4`: an evaluation job cannot observe a partially uploaded checkpoint. `C-S15-5`: inject delayed writer from old attempt; conditional pointer update is rejected. `C-S16-1`: random-weight final-shape checkpoint performance report includes real bytes/durations. `C-S16-2`: actual installed DCP version and optimizer reshard compatibility are tested instead of assumed.
