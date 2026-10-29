# G7 Week 1 — Hardware health, Xid scanning, NCCL collective verification

**Status:** implemented CPU-safe JSON and fixture tests; real physical NVIDIA GPU admission and NCCL **pending hardware**. Inputs: teammate ADR 0003, Session 15 fault prevention, Team 8 JSON schema still to be ratified.

## 1. Health is evidence, not a label

Four statuses: `pass` (check executed and met criteria), `warn` (investigate), `fail` (defined hard condition), `skipped` (not executed or unsupported; **never silently pass**). Admission is a separate boolean. All scripts emit JSON, with an explicit source and failure reason; stdout is machine-readable, diagnostic stderr is not interpreted as healthy.

## 2. Implemented files

- `health/gpu_health.sh` is the requested shell command and delegates to `gpu_health.py` to emit **a single valid JSON object**.
- `health/gpu_health.py`: collects `nvidia-smi` GPU inventory (model, UUID, memory, temperature); optional volatile uncorrectable ECC; scoped kernel Xids; quick prolog or full diagnostics; optional `dcgmi diag -r 1`; NVLink status and real PyTorch NCCL correctness when multiple GPUs exist. Every unavailable tool is `skipped` with reason.
- `health/xid_scan.py`: `journalctl -k` with last-30-min default for live nodes; `dmesg` fallback available explicitly, but **no time filter**; synthetic `--source file --file ...` for unit tests. It emits code/BDF/severity/counts, not unchecked raw user data.
- `health/nccl_probe.py`: launched with `torchrun` and actual ranks; exact numerical all-reduce sum for ranks 1..N and several message sizes. Supports **Gloo CPU explicitly** for Week 1 proof. Actual `NCCL` requires CUDA and two or more GPUs.
- `health/launch.sh`: preflight gate; refuses command execution when health verdict is `warn` or `fail` (unless run in `--report-only` mode). Does not itself perform Slurm requeue or drain; those require permissioned site glue.

## 3. JSON contract

```
{"schema":"v5-health/1", "host":"...", "timestamp_utc":"...",
 "mode":"prolog", "expected_gpus":8, "verdict":"pass|warn|fail",
 "admit_training":false, "checks":[{"name":"xid","status":"skipped",
   "required":true,"reason":"journalctl unavailable", "metrics":{}}],
 "drain_reason":null}
```

The source of truth for Team 8 integration is its final approved JSON contract (due 29 Oct). G7's current contract is **provisional** and should be versioned when merged. A `warn` is deliberately not admissible by `launch.sh`; instructor can define explicit manual overrides with audit later.

## 4. Xid classification

Conservative admission classifier using NVIDIA catalog; not a universal node repair order:

- Proposed hard fail for observed 48, 64, 74, 79, 95, 119, 120 (consult vendor and policy before GPU reset or permanent drain).
- Warn/investigate for 13, 31, 43, 45, 63, 94. Xid 94 is **contained**; it is not the same as Xid 95. Xid 43 may reflect an application failure while the GPU remains healthy.
- Unknown code → warn and preserve evidence. Unavailable kernel-log access → `skipped`, never `pass`.
- Live query from journalctl is windowed; dmesg uses full buffer and may include historical resolved faults.

NVIDIA's current GPU-specific catalog supersedes any static mapping. When in doubt, record but **avoid destructive hardware actions** and escalate to the operator.

## 5. NCCL methodology

1. Reserve the entire intended test GPU group; don't overlap NCCL stress testing with live production training.
2. Verify rank to GPU index mapping, `CUDA_VISIBLE_DEVICES`, GPUs on the allocated node, driver/CUDA/NCCL compatibility and topology.
3. Run `torchrun` across N ranks, each contributing a known constant. Verify every element equals `N(N+1)/2` after all-reduce; enforce a timeout.
4. For performance, use NVIDIA `nccl-tests/all_reduce_perf` on the *same allocation*, capture raw logs, chosen message sizes, iterations, algbw and busbw. The included PyTorch correctness probe is **not a trustworthy peak bandwidth benchmark** and does not substitute for nccl-tests.
5. In multi-node deployment verify InfiniBand/EFA, rank rendezvous and collective fabric timeouts; record injected broken rank/link results.

CPU Gloo correctness is a **code-path check**, not evidence that a GPU NCCL collective passed. Week 1 with no GPU may emit `skipped` and must say so.

## 6. Example commands

```bash
bash health/gpu_health.sh --mode prolog --require-gpu --expected-gpus 8
python3 health/xid_scan.py --source file --file health/fixtures/xids_sample.log
# Optional Mac CPU-only smoke, not a GPU benchmark:
torchrun --standalone --nproc-per-node=2 health/nccl_probe.py --backend gloo --sizes 1024,65536
# Real GPU on dedicated allocated node (requires >= 2 GPUs):
bash health/launch.sh --full --require-gpu --expected-gpus 8 --report-only
```

## 7. Safety and operational considerations

- `dcgmi diag` can be disruptive; use full mode only in a maintenance/allocated exclusive window. Prolog is fast/passive.
- Shell `dmesg | grep xid` is insufficient: permission failures and no matching events can both produce nonzero grep status.
- GPU health acceptance must specify expected **physical devices** vs MIG partitions, GPU UUID, allocations and environment; `nvidia-smi` may show all physical GPUs, not only the CUDA-visible set.
- Temperature and NVLink thresholds depend on SKU/topology; **do not invent** universal thresholds.
- Slurm node drain must be performed by the privileged control plane, after authorized policy evaluation; diagnostic script should not run `scontrol` as an ordinary training user.
- Data plane faults, loss spikes, expert imbalance, NaN/Inf and data-ledger anomalies belong to Team 5/8 telemetry, not falsely categorized as a GPU hardware fault.

**Sources:** [NVIDIA Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html), [NVIDIA NCCL Tests](https://github.com/NVIDIA/nccl-tests), [NVIDIA DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/), [PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html).
