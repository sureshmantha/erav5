# G7 Week 1 — Queue design, Slurm-in-Docker acceptance

**Status:** candidate for instructor review; the CPU container integration tests **have not been executed** in this build environment (Docker and Slurm binaries unavailable).
**Scope:** ERA V5 training and side workloads, aligned with trainer-selected Team 5 configurations; **no fixed model size or 1T assumption**.
**Source:** G7 ADR 0002 (Proposed), Session 15 reliability and Session 16 compute/goodput, instructor Gate 1 requirements.

## 1. First principles and guarantees

An allocation must establish *who may use resources, what they may use, when, for how long, how usage is accounted, and which recovery mechanism is called*. The scheduler is **not** the checkpoint engine. Two **Slurm accounting accounts**, `mainrun` and `teams`, are required, rather than two AWS accounts or two VMs.

- `mainrun`: instructor-approved primary training; protected priority; whole-node exclusive allocation in GPU production; authorized submitter only.
- `teams`: small bounded experimental jobs. The CPU-lab QoS `g7_team` has **MaxTRESPerJob=cpu=1**, **MaxWall=00:05:00**, **MaxSubmitJobsPerUser=2**. The CPU lab intentionally does not claim production GPU caps.
- All jobs use `AccountingStorageEnforce=associations,limits,qos`. Reject accounts without an authorized association; reject jobs exceeding limits. Record job ID, account, QoS, requested TRES, state and restart count.
- Fairshare is configured and must be **measured over time**, not assumed because both accounts exist. `PriorityWeightFairshare=1000`, `PriorityWeightQOS=10000` are **lab starting values**, not production-tuned values.

### Reference system

```
                         instructor/job gateway
                                 |
                          sbatch / squeue
                                 |
                      +-----------------------+
                      | slurmctld (controller)|
                      +----+--------------+---+
                           |              |
             slurmdbd + MariaDB      Slurmd on nodes
               associations          CPU lab: compute1/2
               usage, QoS            PROD: pinned GPU image,
               audit                 cgroups, NCCL, EFA/NVLink
                           |
         committed checkpoint manager + data ledger
                           |
                     recovery controller
```

## 2. Platform decision

Slurm is the common **scheduler contract**. The teammate ADR recommends self-managed Slurm on EC2, while AWS ParallelCluster/PCS can manage parts of the cluster lifecycle. Self-managed operation requires controller failover planning, DB backup, MUNGE key rotation, private networking, multi-AZ storage strategy, patching and cluster-level RBAC. Cloud choice remains **Proposed** pending instructor AWS topology/ownership confirmation. For the Week 1 proof use Docker Compose with Slurm, slurmdbd, MariaDB, MUNGE and two CPU nodes.

## 3. CPU Docker test topology

`slurm-docker/` contains:

- `compose.yaml`: MariaDB, slurmdbd, slurmctld, two `slurmd` CPU workers, and a one-time MUNGE key generator.
- `conf/slurm.conf`: cluster `v5lab`, CPU-only node inventory, QoS preemption, association and limit enforcement, `RequeueExit=75`, `RequeueExitHold=1`.
- `scripts/init_accounts.sh`: idempotent Slurm accounts, QoS and user association creation.
- `scripts/test_accounts.sh`: **executed** assertions (when Docker is available) for two valid accounts, CPU over-limit denial, unauthorized account denial, and accounting records.
- `scripts/test_requeue.sh`: optional retry test for exit 75; must be independently run and inspected.

**Important:** a CPU container cannot validate NVIDIA GRES, GPU cgroups, EFA, NVLink, torch NCCL or AWS launch permissions. `task/none` and privileged worker containers are a *lab accommodation* only. Production must use cgroup v2-compatible `task/cgroup`, `proctrack/cgroup`, `gres.conf` verified by `slurmd -G`, IAM role and driver/device isolation. Do not transplant `slurm.conf` unchanged to GPU production; carry over the **policy concepts**, adjusting actual hardware/TRES configuration.

## 4. Running the two-account lab

Prerequisites: a Linux-capable Docker Engine / Docker Desktop with Compose V2, memory available for MariaDB and two workers, internet for image build. This **build script is provided but is unexecuted in this environment**. The commands are designed to fail rather than declare success when an acceptance assertion fails.

```bash
cd teams/07-infrastructure/slurm-docker
./scripts/init_env.sh                 # makes .env (gitignored), no credentials printed
./scripts/run_lab.sh                  # build/start, initialize accounts, run assertions
# Direct inspection:
docker compose exec controller sacctmgr show assoc format=account,user,qos
docker compose exec controller sacctmgr show qos format=name,priority,maxtresperjob,maxwall
docker compose exec controller sinfo -Nel
docker compose exec controller sshare -al
# Optional retry verification:
docker compose exec controller /lab/scripts/test_requeue.sh
# Copy recorded controller evidence to your workspace, then manually inspect it:
docker compose cp controller:/tmp/g7-slurm-evidence ./evidence-from-docker
# Stop after tests:
docker compose down
```

The lab's database passwords are generated in local `.env`; the MUNGE key is generated once in a Docker volume, never committed. Do **not** upload the `.env`, volume, Parquet corpus, cloud keys or exported private logs to a public PR. For production, use AWS IAM roles/secret manager and a documented bootstrap/rotation mechanism.

### Acceptance assertions

| ID | Action | Mandatory evidence |
|---|---|---|
| Q1 | Initialize cluster and accounts | `sacctmgr` output shows `mainrun` and `teams`, correct QoS names and limits |
| Q2 | Submit `labmain` to mainrun and `labteam` to teams | Two different job IDs and `sacct` account attribution |
| Q3 | Team user submits `--cpus-per-task=2` under CPU cap of 1 | `sbatch` nonzero exit, captured rejection reason |
| Q4 | Team user submits to unauthorized mainrun account | `sbatch` nonzero exit, captured rejection reason |
| Q5 | Run exit=75 test | Show the job was requeued, `Restarts` changed, subsequent completion |
| Q6 | Mainrun preempts eligible teams work | Explicitly record signal, grace and requeue; **NOT proven by current test script** |
| Q7 | Fair-share convergence | `sshare` and `sprio` after workload; **NOT proven by current test script** |
| Q8 | Health-driven drain | `scontrol update NodeName=... State=DRAIN Reason=...` with authorized controller integration; **NOT proven by CPU unit tests** |

Do not mark the entire instructor queue criterion **PASS** until Q1–Q4 execute. Q5–Q8 are valuable hardening demonstrations, not falsely claimed complete.

## 5. Correct training launch contract

One Slurm task/`torchrun` **agent** per node, not 8 Slurm tasks each launching 8 workers. On an 8-GPU node: `--ntasks-per-node=1` and `--nproc-per-node=8` yields 8 local workers. For N nodes, Slurm launches N agents and torchrun spawns 8N workers. Set a job-unique rendezvous ID and a verified free port. `slurm-docker/examples/train_gpu.sbatch` is a **disabled-until-configured GPU reference**, not an accepted job. It must be integrated with Team 5 trainer configuration, signal-to-checkpoint acknowledgement and scheduler retries.

- Slurm `--signal=B:USR1@180` warns the **batch shell** before *wall-time*. It is not a universal preemption callback.
- QoS preemption uses separately configured `PreemptMode=REQUEUE`, allowed `Preempt` QoS and `GraceTime`; observe actual Slurm behavior in the installed version.
- `--requeue` allows retry. `RequeueExit=75` is the explicit policy for exit 75. A fatal numerical anomaly must **not** cause an infinite retry loop; `RequeueExitHold=1` is configured as a proposed hold policy and requires an integration test.
- At recovery, use last **durable** checkpoint and an attempt-epoch fencing token; terminate old ranks before a new writer may promote `latest`. Training launch must not accept a dirty temporary checkpoint.

## 6. Failure classes and recovery authority

| Event | Owner | Default response |
|---|---|---|
| Node gone, driver fault, NCCL hang | Node health + Slurm | Fence attempt, drain affected node if warranted, retry from committed generation |
| Graceful impending walltime | Trainer + checkpoint + Slurm | Save to durable threshold if time permits, exit=75 / requeue |
| Checkpoint corruption | Checkpoint manager | Verify failure, immutable quarantine, previous independently validated generation |
| NaN or unrecoverable shape mismatch | Trainer / Team 5 | Stop + hold for investigation; **never** autosave contaminated state |
| Bad batch or policy-driven skip | Data team + trainer | Explicit branch and ledger decision; never silently skip and call replay exact |

**Never promise literal zero recomputation** from periodic checkpointing. A job may repeat steps since the last remote durable commit. Publish `RPO` (committed step/token progress lost: target zero) separately from recomputation and `RTO`.

## 7. Evidence, threat model and open dependencies

The CPU Compose lab validates Slurm's *accounting and basic policies* only. Real GPU production additionally requires: node images pinned by digest; CPU/GPU binding; topology `nvidia-smi topo -m`; full NCCL test under allocation; EFA network (if AWS), monitored distributed hangs; checkpoint bandwidth; and protected shared storage. Inputs needed: Team 5 config hash and parallel mesh, Team 8 JSON schema, instructor cluster administrator's scheduler and IAM decisions.

**Primary documentation:** [Slurm Accounting](https://slurm.schedmd.com/accounting.html), [Resource Limits](https://slurm.schedmd.com/resource_limits.html), [QoS](https://slurm.schedmd.com/qos.html), [Preemption](https://slurm.schedmd.com/preempt.html), [slurm.conf](https://slurm.schedmd.com/slurm.conf.html), [sbatch](https://slurm.schedmd.com/sbatch.html).
