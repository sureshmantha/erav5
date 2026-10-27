# ERA V5 G7 — Week 1 Acceptance / Evidence Register

**Evidence labels:** `PASS_LOCAL` (run in this environment), `READY_UNEXECUTED` (artifact supplied but not run here), `PENDING_HARDWARE`, `PENDING_OWNER`. No combined 100% score until every instructor-mandated item is actually tested on the appropriate platform.

| Criterion | Implementation | Test/evidence | Current disposition |
|---|---|---|---|
| Queue design | `docs/queue_design.md` | design linked to ADR 0002 | READY_FOR_REVIEW |
| Slurm in Docker, two **Slurm** accounts | `slurm-docker/compose.yaml`, init scripts, two compute workers | `run_lab.sh`, `test_accounts.sh`: two permitted, over-limit rejection, unauthorized rejection, sacct | **BLOCKED_ON_USER_MAC** in original version; hotfix delivered, Docker/Slurm rerun required (missing here) |
| Slurm policy exits | `RequeueExit=75`, `RequeueExitHold=1`, QoS limits | `test_requeue.sh` pending real lab | READY_UNEXECUTED |
| Health scripts | `health/launch.sh`, `gpu_health.sh`, `xid_scan.py` | JSON and Xid fixture unit tests | PASS_LOCAL (synthetic fixtures) |
| Missing tools `skipped` | `health/gpu_health.py`, `xid_scan.py` | missing fixture and unavailable tool tests | PASS_LOCAL |
| CPU two-rank all-reduce | `health/nccl_probe.py` Gloo mode | `test_collective.py` | PASS_LOCAL_HOTFIX (Gloo verified in assistant Linux container; macOS rerun required, **not NCCL**) |
| GPU health JSON on one GPU | `health/gpu_health.py` | Actual `nvidia-smi` capture | **PENDING_HARDWARE** |
| Multi-GPU NCCL all-reduce | `health/nccl_probe.py` NCCL mode | CUDA GPU logs | **PENDING_HARDWARE** |
| Checkpoint design | `docs/checkpoint_design.md` | reviewable contract + production boundaries | READY_FOR_REVIEW |
| CPU save/new-process restore bit-equal each step | `ckpt/ckpt.py` | `tests/test_checkpoint.py`: 12 uninterrupted vs 5+7 | **PASS_LOCAL** |
| CPU corrupted payload refusal | `ckpt/ckpt.py` integrity manifest | `test_corrupt_byte_fails_closed` | PASS_LOCAL |
| CPU interrupted save cannot publish | atomic generation & pointer | `test_midwrite_does_not_publish_latest` | PASS_LOCAL |
| Actual DCP distributed sharded save | docs design only | not implemented in CPU fixture | PENDING_WEEK2 |
| No raw corpus/secrets committed | `.gitignore`, synthetic fixtures only | repo diff review | LOCAL_SOURCE_REVIEW_REQUIRED |
| `gitleaks detect --source .` zero findings | `scripts/pre_push.sh` | must run in user's Git checkout | **PENDING_TOOL** (not installed here) |
| Reviewed GitHub PR | tracked Git flow in README | actual PR number and reviewer | **PENDING_OWNER** |

## Mandatory user-side commands on machine with Docker / real GPU

1. `bash tests/run_cpu_tests.sh` — confirm all tests locally.
2. `cd slurm-docker && ./scripts/run_lab.sh`; verify output and copy `sacctmgr`, `sacct`, rejection logs to a public-safe PR.
3. If GPUs allocated: `bash health/gpu_health.sh --require-gpu --expected-gpus <ACTUAL_COUNT>` and `bash health/launch.sh --full ... --report-only`; confirm JSON and NCCL correctness, vendor baseline separately.
4. At Git repo root: `gitleaks detect --source .`; record tool version, exit code and redacted output.
5. Open a reviewed PR; never include `.env`, credentials, sample Parquet pack, arbitrary logs containing identifiers or cloud secrets.

## Mac feedback and remedial validation (2026-10-27)

The user actually ran the original bundle and reported 12/13 successful CPU tests in a temporary NumPy-1.26-compatible environment, one Gloo timeout, and Slurm container failures involving MUNGE access, topology, and cgroup/DBus. These observations **supersede** the original local-only evidence when describing Gate 1 readiness on the user's machine. This hotfix addresses the known issues but does not yet prove a successful Mac Slurm runtime. Subsequent users should archive the exact command, exit code, and redacted logs as proof.

`IgnoreSystemd=yes` and `SlurmdParameters=config_overrides` make this a **CPU accounting / placement policy exercise**. G7 must still demonstrate production cgroup/GRES isolation on its real GPU nodes in Week 2.
