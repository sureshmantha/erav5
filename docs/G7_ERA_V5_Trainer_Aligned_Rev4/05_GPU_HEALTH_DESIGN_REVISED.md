# G7 Health, Xid and NCCL design v0.3

> **REV4 MODEL SCOPE (2026-10-27):** ERA V5 is a new three-model capstone programme; Session 15 did not disclose final sizes. Do not assume a 1T or 40B V5 target; the 40B case was Session 3 homework and V4's 118.67B is historical evidence. V5 stage-specific tensor/optimizer inventories from G5 determine GPU, checkpoint and transport sizing. Consult `00_TRAINER_INTENT_V5_VS_V4.md` and `07_CONFIG_DRIVEN_MODEL_STAGE_CAPACITY.md`. Existing design controls and Week 1 CPU tests remain applicable independent of model size.


## Two independent modes

**Laptop / CI (CPU-only)**: missing NVIDIA tools => `skipped`, not `pass`. Run 2-rank torch distributed **Gloo** correctness test. This validates JSON reporting and distributed software only, and **does not** satisfy the GPU NCCL gate.

**GPU / production**: GPU count and UUID inventory vs scheduler allocation; NVML/nvidia-smi; DCGM as available; Xid kernel log; ECC/remap; clock throttling and node/network state. Required uninspectable GPU checks => `warn` or `fail` depending environment policy; a GPU node may not be admitted to mainrun from warning-only evidence without an explicit, auditable operator override.

## Prolog and deep checks must not collide

- **Prolog (fast, passive)**: driver presence, GPU identity/count, quick ECC/remap, recent Xid, DCGM health as suitable. No long `dcgmi diag` or intrusive NCCL saturation while other jobs run. Emit JSON with stopwatch duration and precise tool availability.
- **Exclusive offline diagnostic allocation**: `dcgmi diag` level appropriate to generation (and GPU idle requirement), `nccl-tests all_reduce_perf` one-node and inter-node matrix, connectivity and EFA/NVLink tests. Run before admitting new nodes or during remediation; not under an unrelated live training job.
- **During training**: persistent DCGM exporter/G8 telemetry plus Xid detection and `ProcessGroupNCCL` flight recorder on hangs; sampling budget documented. Operator-assisted remediation and proof of re-entry.

## Xid semantics

Retain original log line hash, monotonic time, UTC time, boot ID, node ID, GPU UUID, PCI bus and Xid, app/job ID if correlatable. Incremental journal cursor, dedup and past incident retention required. NVIDIA's Xid catalog distinguishes **94 contained** (restart affected app; GPU may continue until convenient reset) from **95 uncontained** (reset/recover GPU). The independent NVIDIA Health Validation workflow may mark both unavailable until remediated. Make severity, requested workflow and actual `DRAIN` policy separate fields; do not assert every Xid maps to fatal permanent hardware failure.

**Do not compute arbitrary fixed temperature/busBW thresholds** before getting H100/B200 baselines. Different GPU generations, switches, cable layouts and EFA placement have different expected limits. Automatic return from drained state only after remediation plus quick and deep checks.

## Proposed health contract (provisional pending Team 8's schema on 29 Oct)

```
{
 "schema":"v5-health/1-provisional", "node":"...", "generated_at":"RFC3339",
 "mode":"local|prolog|full", "expected_gpu_count":null,
 "checks":[{"id":"gpu.inventory", "status":"pass|warn|fail|skipped", "reason":"...", "metrics":{}, "required":true}],
 "verdict":"pass|warn|fail", "gpu_admission":"allow|deny|manual_review", "drain_reason":null,
 "provenance":{"tools":{},"boot_id":"..."}, "duration_ms":0
}
```

Version and align to Team 8 rather than force them to use these fields. A missing tool must retain `skipped`; an overall verdict must not claim pass while required GPU safety checks are unverified.

## All-reduce acceptance

From `torchrun` with 2+ ranks on *real NVIDIA GPUs*: initialize `nccl`, rank values are exactly representable integers, reduce sum and check expected = `N(N+1)/2`. Exercise 1MiB–256MiB sizes after a quick small sanity test; report bytes, p50/p95 timing, algBW, busBW only as defined by the benchmark and algorithm used, link topology, GPU UUID, NCCL + CUDA version, configuration, allocation, baseline. Run `nccl-tests all_reduce_perf` separately for official bus bandwidth output. With no GPU, use Gloo and mark clearly `cpu_gloo`; never call it NCCL.

## Tests before approval

- H1: missing GPU tools -> report correct `skipped`, CPU mode `manual_review/deny` for GPU admission; syntactically valid JSON.
- H2: fixture Xid 13, 79, 94, 95, unknown + duplicate -> parse and policy/boot/time behavior tested.
- H3: simulate nvidia-smi failure/malformed output -> no fabricated pass; diagnostics recorded.
- H4: real single GPU driver, ECC/DCGM readings and UUID match, JSON valid.
- H5: 2 GPU or multi-node NCCL correctness, peer failure timeout, flight recorder event.
- H6: injected fail -> queue automatically drains assigned test node; corrected healthy state re-admitted only after remediation.
- H7: G8 dashboard consumes JSON, verifies event schema and timestamps.

## Sources

- https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html
- https://docs.nvidia.com/health-validation/reference/checks/xid
- https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html


---


## Session 15 stability monitoring and Session 16 performance extensions

### Health verdicts require separate namespaces

Use independent output fields `hardware_admission`, `communication_validation`, `numerical_training_health`, `data_replay_integrity`, `checkpoint_integrity`, and `telemetry_integrity`. **Do not make a GPU Xid diagnosis from a loss spike**; conversely a finite loss does not prove healthy hardware. Missing kernel-log permission remains `skipped`/`unknown`, never PASS. The error-action lookup remains NVIDIA-version-specific; capture raw Xid and source message, GPU UUID, PCI BDF and time window.

### Continuous monitoring contract with Team 8

Track per-rank gradient norms, update/weight statistics, loss and held-out loss, LR and scaler, token throughput, collective timing, expert usage/fraction inactive, number of zero-grad/frozen param groups, GPU peak allocation, ECC remap/Xid state, NVLink/InfiniBand, CPU/host-memory pressure, checkpoint stage/upload time and recovery duration. Session 15 points out that expert collapse and gradient spikes can precede evident model-quality failures; the monitors must be calibrated to this run. An exponentially weighted z-score is a **candidate** detector, not a universally validated threshold. Preserve incident evidence and raw series; render downsampled overview series with min/max to avoid erasing spikes.

### Fault policy examples

- `Xid 79`: usually severe lost-GPU access; evaluate node drain and job-level recovery. Do not assume checkpoint can be saved on that dead GPU.
- `Xid 31/43`: investigate application/driver context before treating as hardware-only permanent failure.
- `DCGM tool missing on CPU/Colab`: report `skipped`, never advertise H100 hardware readiness.
- `collective timeout`: record all ranks, transport, topology, NCCL debug/flight recorder if configured; differentiate a poisoned rank from a slow network link.
- `grad norm spike`: generate a **numerical** diagnostic event with batch ID and checkpoint generation. No automatic Slurm node drain.

### New tests

`H-S15-1`: same Xid code outside allowed lookback is not represented as new incident. `H-S15-2`: unreadable kernel logs ⇒ source unavailable, not pass. `H-S15-3`: synthetic grad spike ⇒ numerical channel only. `H-S15-4`: old-attempt telemetry cannot overwrite current-attempt status. `H-S15-5`: mock large time series downsample preserves min/max spike. `H-S16-1`: measure GPU health/NCCL on actual topology in Week 2; do not invent benchmark throughput in Week 1.
