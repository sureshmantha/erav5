# ERA V5 Team 7 — Repository source ledger (audit revision A)

## Sources received and inspected

| Label | Uploaded archive | Relevant subtree / root | Evidence scope |
|---|---|---|---|
| P | `pradeep_era-v5-lab-infrastructure.zip` | `teams/07-infrastructure/submission/` plus `support/` | 4 proposed ADRs (checkpoint, Slurm, GPU health, cost), queue/health/checkpoint modules, simulation, metrics, tests, Docker files |
| L | `mallik_era-v5-lab-07-infrastructure.zip` | `teams/07-infrastructure/{health,checkpoint,queue}/` | DCP checkpoint, restart demo, Slurm Docker lab, GPU/NCCL health files; no formal numbered ADRs located |
| X | `max_queue-main.zip` | repository root `queue_design.md`, `cluster/` | 15+ queue design decisions and installer/tests. Explicitly **not** a complete G7 health/checkpoint stack or Week 1 Docker lab. |
| S | `suresh_era-v5-lab-infrastructure.zip` | `teams/07-infrastructure/week1-gate/` | 2-account Docker demo, fail-closed JSON health, CPU replay, synthetic Xid, hotfix/tests; separate proposed design docs |
| M | `murthy_Team7-infrastructure-main.zip` | **UNAVAILABLE** | User asked to add this as fifth. Exact-name search of visible attachments and Library did not return bytes; **no files/ADRs/code from M were reviewed or merged.** |

`FOUR_REPOS_FILE_INVENTORY.csv` records relative paths, sizes and SHA-256 checksums of all *available* infrastructure-related files. **The project is a four-source implementation with an explicit fifth-source hold, NOT an audited five-source final release.** All four available repositories are user-provided code, not externally authenticated commits. No Git repository history, authorship, issue tracker, fork parent, license or production deployment was verified.

## User's Week 1 acceptance contract (not inferred from frameworks)

1. `docs/queue_design.md` and **actual** Slurm-in-Docker test with **two Slurm accounts**, limits enforced.
2. `health/launch.sh`, `gpu_health.sh`, `xid_scan.py` and GPU NCCL test, with valid JSON and unavailable tools marked `skipped`; no fabricated GPU results.
3. `docs/checkpoint_design.md` + `ckpt/ckpt.py`, CPU save/terminate/fresh-process restore with **bitwise identical subsequent loss**.
4. Git secrets scan (`gitleaks detect --source .`) and reviewed pull request.

## Technical North Stars — read the caveat

**DeepSpeed is a training engine, not a replacement for Slurm, DCGM, NCCL or a distributed checkpoint commit coordinator.** It has documented design choices and APIs, but does not publish one official all-inclusive "ADR repository" covering this complete infrastructure lifecycle. `01_ADR_COVERAGE_AND_GAP_ANALYSIS.md` therefore contains **our proposed ADR taxonomy, derived from framework capabilities**, not official DeepSpeed ADR numbers.

- DeepSpeed training, ZeRO, checkpointing, inference: https://deepspeed.readthedocs.io/en/latest/
- DeepSpeed checkpoint API: https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
- TorchTitan training + composable parallelism: https://github.com/pytorch/torchtitan
- TorchTitan checkpointing: https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md
- Megatron Core/NeMo distributed checkpointing: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/checkpoints/dist_ckpt.html
- Slurm: https://slurm.schedmd.com/documentation.html
- NCCL tests: https://github.com/NVIDIA/nccl-tests

## Audit integrity policy

Every statement is one of: `SOURCE_PRESENT`, `DESIGN`, `STATIC_CHECK`, `CPU_EXECUTED`, `RUNTIME_UNTESTED`, `GPU_UNTESTED`, `CLOUD_UNTESTED`, `SOURCE_MISSING`. Synthetic logs may only prove the parser or simulator. A completed course gate is a decision by instructor/lead based on actual evidence; no source-code review can substitute for it.
