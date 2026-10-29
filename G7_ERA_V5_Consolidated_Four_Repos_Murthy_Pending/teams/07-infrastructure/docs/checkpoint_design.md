# G7 Week 1 — Checkpoint, exact CPU replay and distributed production plan

**Status:** Week 1 CPU reference implementation and tests; proposed production DCP architecture.
**Reference:** colleague checkpoint ADR 0001 and `checkpoint_design.md`, ERA V5 Session 6 ledger, Sessions 10–14 numerical/sharding, Session 15 stability, Session 16 goodput and instructor Gate 1.

## 1. Precision of the goal

Three distinct guarantees must not be conflated:

1. **Committed-work durability:** every checkpoint promoted to `latest` is completely materialized, checked and retained in the declared failure domain. A crashed staging writer never changes the committed pointer.
2. **Bit-equal continuation:** the sequence of future batch IDs, per-step loss bits, parameter tensors, optimizer state, RNG stream and scheduler state matches a continuous run with fixed topology/software/precision. A matching probe loss alone is insufficient.
3. **Literal zero recomputation:** no repeated work across failures. This is **not promised or implemented** by interval checkpointing; it requires sufficiently frequent replicated durable transitions / standby and an explicit consistency protocol, with significant overhead.

The Week 1 acceptance is **#2 on CPU**. Week 2 must prove valid recovery on the actual approved distributed trainer and NVIDIA devices. World-size and kernel changes can preserve *model values* without guaranteeing bit-identical future gradients.

## 2. Exactly what a checkpoint must contain

At an **optimizer-step boundary** (all ranks have applied the update, scheduler advanced and gradients cleared):

| Category | State | Why |
|---|---|---|
| Model | full state key schema and tensor dtype/shape; sharded owners for production | No random initialization, no missing parameters |
| Optimizer | state tensors (Adam moments, step), parameter-group membership, FP32 masters if configured | Momentum and future updates must match |
| Precision | BF16/FP16/FP8 policy, dynamic scaler state if present | Subsequent math / loss scaling |
| Scheduling | LR scheduler, step count, tokens seen, curriculum/mixture phase | Exact hyperparameters |
| RNG | Python, NumPy, Torch CPU and relevant CUDA per-device/per-rank RNG | Dropout and other randomness |
| Data | immutable dataset/tokenizer/packing manifest digest, global ledger cursor, sampler state, ordered next-batch IDs, skip-list or decision ID | Reuse precisely the intended training tokens |
| Execution | config hash, git SHA, image digest, library/kernel versions, DP/EP/TP/PP mesh, expected rank count | Prevent silent runtime drift |
| Recovery | run/branch/attempt epoch, checkpoint generation, manifest checksum and predecessor | Auditable rollback and stale-writer fencing |

**Boundary condition:** the CPU fixture `ckpt.py` saves after `optimizer.step()` and `scheduler.step()` with `zero_grad(set_to_none=True)` and a deterministic batch cursor. It is **single rank, single process and CPU**. It intentionally does not implement production model sharding, live S3 or a per-rank ledger.

## 3. Implemented CPU checkpoint transaction

`ckpt/ckpt.py` persists a **trusted local** PyTorch tensor container with `torch.save`, `torch.load(weights_only=True)`, and data free of custom Python objects. JSON manifest contains format, step, cursor, config fingerprint, payload length and SHA-256. Each save uses a new generation path:

```
checkpoint-root/
  step_00000004_<random-id>/
    state.pt
    MANIFEST.json
  latest.json                  # one atomic pointer updated only AFTER sealing
  step_00000006_<id>.tmp/     # incomplete; explicitly ignored
```

1. Capture exact coherent CPU state at optimizer boundary; write into a **new** `.tmp` generation.
2. Flush+fsync the state file and the manifest, fsync temporary directory.
3. `os.replace` the temporary directory to a uniquely named immutable generation on the **same filesystem**; fsync the parent directory.
4. Write and fsync a new pointer; `os.replace` `latest.json`; fsync root directory.
5. Restore reads pointer, verifies manifest identity SHA-256, payload SHA-256 and byte size, applies strict model shape/dtype/schema verification, then restores optimizer, scheduler and RNG. It rejects config drift.
6. An incomplete temp directory is never accepted as `latest`.

This establishes local **single-writer** crash-consistency properties within normal POSIX filesystem durability assumptions. It is **not** an S3 multi-object atomic commit, an authenticated signature, an inter-node transaction or a guarantee against dishonest malicious modification of both state and manifest. Avoid loading checkpoints from untrusted sources.

## 4. Executable acceptance proof

Command: `bash tests/run_cpu_tests.sh`; tests launch `ckpt/ckpt.py` in separate OS processes and compare:

- Continuous 12 optimizer steps versus first 5 steps → save → exit → **fresh process** → restore → remaining 7.
- Every resumed **loss bit pattern** (`struct.pack('!d', loss.item()).hex()`), LR bit pattern and sampled batch ID.
- Final model + optimizer canonical tensor digest and Torch RNG digest.
- Corrupt-byte detection (load refuses), interrupted save prior to seal (pointer unchanged), and config mismatch refusal.
- All checks are *deterministic CPU fixture proofs only*. Neither drop-in distributed DCP nor CUDA bitwise continuation is asserted.

## 5. Proposed real-distributed DCP transaction (NOT in CPU ckpt.py)

Use [`torch.distributed.checkpoint`](https://docs.pytorch.org/docs/main/distributed.checkpoint.html) where the approved Team 5 trainer has proven integration. DCP saves sharded tensors from participating ranks and supports compatible load-time resharding. It **does not replace** a durable coordinator protocol. Proposed distributed generation `(run_id, attempt_epoch, global_step, generation_uuid)`:

```
step boundary agreed (rank 0 broadcast)
  -> all ranks snapshot frozen state / block mutation until staging complete
  -> each rank writes its immutable shard + non-tensor state + SHA-256 receipt
  -> coordinator validates expected rank set, tensor metadata, ledger commit & hashes
  -> immutable manifest uploaded and verified in remote durable storage
  -> conditional CAS latest update, rejecting old attempt_epoch/stale writer
  -> emit checkpoint.committed only after remote acknowledgement
```

**Asynchronous caveat:** DCP `async_save` often stages CPU copies synchronously unless a supported async stager is configured. Overlap, pinned-memory pressure, upload interference, rank synchronization and model mutation must be *measured*. Never claim only device→host copy stalls until proven by profiling. S3 pointer promotion must use a tested conditional update plus writer-epoch fencing; no ordinary S3 object listing can prove distributed transaction completion.

## 6. Restore and data-ledger integrity

1. Read immutable manifest reachable from latest committed pointer (not highest directory name).
2. Verify content hash, byte count and rank completeness; detect missing/corrupt files early.
3. Validate config, topology, model and optimizer keys, dtype/shape, data manifest, tokenizer, curriculum and software stack.
4. Load into the approved framework/mesh, reshard only along **supported tested** dimensions; strict names/parameter mapping required.
5. Restore per-rank RNG, scheduler/scaler, *exact next batch IDs*, data ledger and auditable skip-list branch decision.
6. Recompute fixed probe then **M subsequent steps**; compare losses and state digests where fixed topology permits exactness.
7. Emit verified resume event with committed cursor, epoch, branch and any recomputation. On any failure, **stop**, do not quietly relax to tolerance.

If a bad batch or NaN requires skipping tokens or changing LR, continue as a **new branch**, not as identical replay. If partial microbatch accumulation must be checkpointed, capture gradients and the microbatch cursor; otherwise save only at a completed optimizer-step boundary.

## 7. Model-stage capacity and goodput

ERA V5 has trainer-selected model stages; **do not size using an assumed 1-trillion-parameter model**. For each approved config:

`checkpoint_bytes = sum(actual serialized tensor nbytes, including optimizer masters/moments and frozen-base representations) + metadata/indices`.

Measure per-node staging memory, local scratch, save stall, local write throughput, remote upload/commit, restore download, reconfiguration and the resulting useful-token goodput. Distinguish peak GPU FLOPs, model FLOPs utilization and **goodput** (fraction of resource time yielding unique committed training progress). Tune checkpoint frequency by measured failure hazard and stall cost; formulas such as Young's are starting approximations, not an SLA.

## 8. Gate matrix and explicit limitations

| Test | Week 1 | Week 2+ |
|---|---|---|
| CPU exact save/new-process resume, all losses bits equal | Executed in tests | CI regression |
| CPU corrupt byte and interrupted temp dir | Executed in tests | CI regression |
| 2/4 rank sharded DCP across fresh process | **Not implemented** | Required integration benchmark |
| S3 generation commit, conditional latest, stale-writer fence | **Design only** | Implement and fail-inject |
| Real GPU checkpoint and restore | **Not executed** | Required |
| World-size change and optimizer state remapping | **Not validated** | Supported layouts only |
| Requeue from actual Slurm termination | CPU lab script available, **not run** | Integration proof |
| Zero recomputation | **Not guaranteed** | Separate proposal if justified by costs |

**Sources:** [DCP](https://docs.pytorch.org/docs/main/distributed.checkpoint.html), [PyTorch asynchronous checkpointing](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html), [AWS S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html), LightningLM https://lightninglm.theschoolofai.in/.
