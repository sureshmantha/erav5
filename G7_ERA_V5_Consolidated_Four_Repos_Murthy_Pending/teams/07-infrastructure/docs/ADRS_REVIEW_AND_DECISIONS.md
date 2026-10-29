# Alignment with teammates' proposed ADRs — G7 Week 1

**Source files provided by team:** `0001-checkpoint-format-and-commit.md`, `0002-job-queue-slurm.md`, `0003-node-health-checks.md`, `checkpoint_design.md`, `README-2.md`, plus follow-up health/launcher prototype. These documents were **Proposed** at review time. Our implementation should go through team lead and instructor approval; no claim is made that teammates approved this independently authored submission.

| ADR | Adopted elements | Changed/clarified | Remaining integration |
|---|---|---|---|
| 0001 DCP + commit | Per-rank sharding target, immutable manifest, checksums, full state, probe validation, async staging objective | Gate 1 *CPU* code implements a verified single-process local transaction; DCP, S3 CAS, writer fencing are explicitly **not implemented**. Probe equality not sufficient: M-step new-process exact replay | Team 5 trainer optimizer state and mesh; Team 1/2 ledger; remote CAS, per-rank receipts, staged-memory benchmarks |
| 0002 Slurm | `mainrun` and `teams` accounts, SlurmDBD, QoS, fairshare and retry exit contract | Actual two Slurm account associations; two CPU Docker workers. `RequeueExit=75` explicit; `--signal` walltime vs preemption distinguished. Production scheduler choice remains provisional | Real Docker run and evidence, fairshare/preempt tests, GPU GRES/cgroups, cloud IAM, controller HA |
| 0003 health | JSON results, skipped tools, Xid catalog, DCGM/NCCL, fail-to-drain concept | Corrected Xid 31/43/45/94 classifications; unreadable kernel log cannot report PASS; default lightweight prolog and exclusive full diagnostics; CPU Gloo never mislabeled NCCL | Team 8 schema, node operator drain privileges, GPU SKU baselines, actual CUDA/NCCL logs |

## Review decision

Accept these *engineering directions* as proposed, but **do not merge as fully Gate 1 complete** without real Docker account-limit logs, any-required local GPU evidence, CI/test logs, and Gitleaks exit 0.

Additional source context: Session 15 emphasizes silent corruption, numerical guardrails and wrong-checkpoint failures; Session 16 emphasizes useful training progress and hardware time. ERA V5 model size should be set by Team 5's real config, **not by a hypothetical 1T checkpoint budget**.
