# G7 Queue design v0.3 — for review, not production deployment

> **REV4 MODEL SCOPE (2026-10-27):** ERA V5 is a new three-model capstone programme; Session 15 did not disclose final sizes. Do not assume a 1T or 40B V5 target; the 40B case was Session 3 homework and V4's 118.67B is historical evidence. V5 stage-specific tensor/optimizer inventories from G5 determine GPU, checkpoint and transport sizing. Consult `00_TRAINER_INTENT_V5_VS_V4.md` and `07_CONFIG_DRIVEN_MODEL_STAGE_CAPACITY.md`. Existing design controls and Week 1 CPU tests remain applicable independent of model size.


**Objectives:** two independently accountable Slurm accounts (not two AWS accounts), protected long training, fair bounded experiments, robust failure classification, a reliable scheduler/control plane, and auditable automatic restart. Gate 1 is a *CPU Docker* validation. The Docker fixture must not claim that GRES/NVML/NCCL is production-validated.

## Decision: keep Slurm, defer cloud provisioning choice

The attached ADR 0002 chooses self-managed EC2 Slurm. Re-evaluate against AWS ParallelCluster (AWS-supported cluster orchestration with Slurm, EFA integrations, shared FS and autoscaling); the choice requires ownership of slurmctld/slurmdbd/DB/munge upgrades, backups and security. Keep the choice `Proposed` until TSAI confirms cloud ops ownership, acceptable control-plane RTO, scale, image lifecycle and cost. The job submission API should stay standard `sbatch` in either case.

## Control plane / data plane

- **Scheduler:** pinned versioned `slurmctld`; HA controller for failover where supported; `slurmdbd`+database snapshots and recovery drill. A backup controller alone is not a DB HA design.
- **Compute:** node image pinned by digest; GPU driver+CUDA+NCCL compatibility matrix; MUNGE keys secured; GRES/GPU mapping verified; instances in dedicated private networking, IAM role (no long-lived keys), EFA-ready placement if required; no GPU node accessible to students via direct SSH.
- **Security:** least-privilege job submitter; launcher receives allowed account and validated immutable artifact SHA; signed image digest, no shell interpolation of untrusted user commands. Keep secrets in instance roles / secret manager, never in Slurm scripts or git.
- **Isolation:** `task/cgroup`, `proctrack/cgroup`, appropriate cgroup v2 support, GPU GRES isolation and accounting. Verify hardware/container compatibility.
- **Topology:** jobs request whole nodes for mainrun (`--exclusive`) and consistent GPU mesh; spare replacement capacity optional and budget-dependent; rack/network failure domains explicit.

## Associations and resource policy

Create **Slurm** accounts `mainrun` and `teams` (with children `arch`, `perf`, `eval`, `quant`, etc), associating authorized submitter users. `mainrun` QoS highest priority/exclusive ownership and permits all agreed main-run GPU resources; `teams` subaccounts get explicit `GrpTRES`, `MaxTRES`, time caps, concurrent job caps and fairshare. `AccountingStorageEnforce=associations,limits,qos`; verify no-account, unauthorized account and over-limit submissions **reject**. Fairshare weights, decay, preemption and billable GPU-hours are separate tests; showing two accepted jobs is insufficient.

**Preemption semantics:** `sbatch --signal=USR1@120` is a warning before the *time limit*, not a universal QoS preemption signal. Study and test `PreemptParameters=send_user_signal` + QoS `GraceTime` + `PreemptMode=REQUEUE` in the installed Slurm version. Slurm may send SIGTERM immediately for preemption and SIGKILL at grace end; the application cannot assume it always has 120 seconds. For a training job, mainrun should normally not be preemptible. Benchmarked checkpoint time must fit the observed grace or restart from durable snapshot.

**Exit states:** `0` success, `75` safe retry **only if save protocol reached a defined committed threshold**, `1` application fatal -> hold/escalate. Configure `RequeueExit=75` explicitly; `--requeue` alone is not the same thing. `RequeueExitHold=1` is a proposed policy requiring test (it will mark held as special exit); do not assume a custom epilog can safely mutate its own running job. Validate `SLURM_RESTART_COUNT`, backoff, max attempt count, unknown exits and node-failure behavior.

**Lease/fencing:** scheduler allocation is **not** enough to prevent two processes restoring the same run and both writing its `latest`. Every writer obtains `(run_id, epoch)` lease; all checkpoint commits and data-ledger acknowledgments reject stale epochs. Kill/wedged ranks must not commit once lease expires. Prefer immutable attempt-unique checkpoint generations and CAS updates of `latest` on durable store.

## Workflows

`SUBMIT → ADMISSION (account/limits/artifact/topology) → PROLOG (passive health) → PREPARE (verify checkpoint, environment, data cursor) → RUN (structured events) → SAVING / DRAIN / PREEMPT / COMPLETE → ATTEMPT-FENCED COMMIT → REQUEUE or FINAL`.

- Errors separated: transient node/NCCL crash: quiesce/requeue after backend teardown; data corruption/NaN: hold for review; deterministic misconfiguration: fatal; interrupted checkpoint: fall back **only to a previously durable generation**; bad GPU: drain and remediation.
- Avoid busy retry storm: bounded retries, increasing backoff, circuit breaker, a durable audit event for every attempt and the changed topology. Operational SLOs: queue placement p95, time-to-start p95, RTO p95, abandoned GPUs, unbilled GPU-hours, node replacement p95.
- Hook Team 8 JSON statuses; Team 5 config/parallelism contract; Team 6 NCCL baseline; Teams 9/12 subscribe only to committed model checkpoints.

## Week-1 Docker validation — executable acceptance specification, not asserted passed

A real test must run a pinned Docker Slurm deployment with controller, slurmdbd, DB, MUNGE and **at least two simulated CPU compute nodes**. Make `slurm.conf`/`sacctmgr` bootstrapping idempotent, show image digests. Run all steps in CI with captured command and exact output:

| Case | Procedure | Required evidence |
|---|---|---|
| Q1 | `sacctmgr` create `mainrun`, `teams`+child associations and QoS limits | `sacctmgr show ...` contains both and exact limits |
| Q2 | submit under each authorized account | distinct job IDs, completed `sacct` account usage |
| Q3 | submit unauthorized association and exceed `MaxTRES` | job rejected/held with expected reason |
| Q4 | run competing CPU jobs on 2 nodes; inspect `sshare` and priority changes | same-load fairness direction measured, no starvation |
| Q5 | normal job running, mainrun submitted, preemption configured | actual caught signal and correct requeue; `GraceTime` behavior observed |
| Q6 | process saves and exits 75 | requeued, `SLURM_RESTART_COUNT` increments, resumes last committed state |
| Q7 | process exits 1 | held/not automatically looping, alert emitted |
| Q8 | health fixture returns critical | node becomes `DRAIN`, job moved to healthy node after permitted retry |
| Q9 | controller restart and DB snapshot recovery | queue metadata, accounting and job ownership retained |

Do not invoke real `slurmctld` outside a separated test VM/namespace or claim `Docker Compose` tests without Docker. The environment used to prepare this review has **no Docker or Slurm binaries**; these acceptance tests remain pending.

## Source references

- https://slurm.schedmd.com/sbatch.html (`--signal`, `--account`, `--requeue`)
- https://slurm.schedmd.com/slurm.conf.html (`RequeueExit`, `AccountingStorageEnforce`)
- https://slurm.schedmd.com/preempt.html (`GraceTime` and QoS)
- https://docs.aws.amazon.com/parallelcluster/latest/ug/what-is-aws-parallelcluster.html


---


## Additional queue requirements from Sessions 15 and 16

1. **Three service classes, not only a priority number:** reserve main-run GPUs; use bounded/noisy-neighbor-isolated `debug`/CI and checkpoint/evaluation jobs; prioritize mandatory checkpoint verification and held-out eval without starving long-running training. Allocation quotas/Slurm accounts remain enforced.
2. **Anomaly is not hardware:** G8 numerical alarms must not issue Slurm `DRAIN`. A hardware diagnosis or operator-approved fault policy may drain nodes. Distinguish node `DRAIN` from training attempt `HOLD`.
3. **Single writer and fencing:** a requeued job must obtain new attempt epoch before it can publish checkpoints; an old attempt may finish a delayed upload but cannot advance durable `latest`.
4. **Prove recovery semantics:** test successful exit, fatal exit, configured requeue exit, time-limit warning, QoS preemption, and process-group teardown separately. `--signal` does not substitute for correct Slurm preemption policy or `RequeueExit` setup.
5. **Prevent evaluation congestion:** commit event triggers separate read-only eval. Backlog must be bounded, checkpoint generation is immutable, and mandatory milestone evals are not skipped. Monitor evaluation lag.
6. **Capacity planning is based on goodput:** cost accounting distinguishes queued, allocated, training, checkpoint-stall, replay and recovery GPU-hours. Record actual successful unique loss tokens; GPU busy is not equivalent to useful training.
7. **Capstone dates:** use authoritative CAP plan/current instructor messages for Gate 1 Oct 31, Gate 2 Nov 14, Gate 3 Nov 21, Gate 4 Nov 28. The duplicate S16 transcript has inconsistent holiday wording: never shift dates solely on that text.

### New acceptance cases

`Q-S15-1`: injected numerical NaN ⇒ trainer held, not node drained. `Q-S15-2`: Xid fixture matching approved node policy ⇒ node drains and jobs leave affected allocation. `Q-S15-3`: same job requeues twice; stale attempt rejected by checkpoint publisher. `Q-S15-4`: checkpoint commit triggers eval but does not block optimizer steps. `Q-S16-1`: accounting reconciles allocated/replay/useful GPU-hours.
