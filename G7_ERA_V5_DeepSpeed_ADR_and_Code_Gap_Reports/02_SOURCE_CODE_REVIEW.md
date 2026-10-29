# ERA V5 G7 — Multi-repository source-code review and consolidation decision

**Scope:** four accessible archives; Murthy file unavailable. Source locations and hashes in `FOUR_REPOS_FILE_INVENTORY.csv`. Findings are *code inspection* unless explicitly marked test executed.

## A. Comparison by actual implementation

| Workstream | Pradeep (P) | Mallik (L) | Max (X) | Suresh (S) | Unified decision |
|---|---|---|---|---|---|
| Formal ADRs | **0001–0004** comprehensive proposals | `checkpoint/DESIGN.md`, `health/TESTING.md`, Docker README; no numbered ADR found | `queue_design.md` with inline ADR-style decisions; queue only | proposed docs, gate tracker; earlier P ADRs integrated | Keep P originals in `adr/`, add a **separate** 30-area coverage register rather than claiming DS official ADRs |
| Queue breadth | Queue wrapper, `slurm/` production site templates, SlurmDBD, cloud assumptions | CPU Docker two nodes, sim health/restart | Deepest installer/config/runnable policy test suite, focused on EC2 and isolated team nodes | CPU Compose lab, MUNGE/DB init, two-account policy + Mac hotfix | Week 1 CPU Compose lab + X policy insights + P ops patterns, real runtime pending |
| Queue production assumption | QoS preemption for protected main-run; potential spare node | Requeue, two simulated workers | **No inter-team preemption**; train/eval isolated nodes, real EC2 as Week 1 test | CPU Docker for instructor Week 1 | Policy is **conditional**; do not quietly merge mutually exclusive strategies. Default protected mainrun, `NO_PREEMPT` fallback where scarce nodes/unknown grace; instructor chooses |
| Account behavior | Slurm associations and team hierarchy | Docker account setup | Per-user/per-account admission and denied limits tests | `mainrun`/`teams` plus **DenyOnLimit** hotfix | Verify authorization, account limits and actual `sacct` for Docker; `DenyOnLimit` for immediate submit rejection, pending also valid if explicitly tested |
| GPU launch | Slurm srun launcher, rank/NCCL env + failure exit contract | One torchrun agent/node; flight recorder env | DeepSpeed `deepspeed` launcher for single node | GPU sbatch reference with one torchrun agent/node | One launcher agent/node; Team 5 engine may call `deepspeed.initialize` inside torchrun; no 8x over-subscription |
| Checkpoint backend | Distributed DCP + S3/store receipts, logical state hash, environment checks | DCP + per-rank state + `COMPLETE`, CPU self-test | Queue includes smaller model+optimizer save/restore for demo | Single-process immutable local gen + manifest hash + exact new-process CPU | Keep S for Week 1 correctness; validate L's DCP on two CPU ranks; integrate P's durable distributed commit separately; **never** claim DeepSpeed ZeRO states automatically load via DCP |
| State completeness | Per-rank RNG, cfg, samples, skip/branch, model+optimizer+LR | RNG, adam, data cursor, frozen params | model, optim and trainer metadata in queue demo | Python/NumPy/Torch RNG + Adam/scheduler/cursor, checksum | Manifest schema must be Team 5 + Teams 1/2 agreed, strict state load, never accept random weights silently |
| Faults | Comprehensive reason taxonomy 0/75/78, health/metrics | rank/node kill demos, tests | robust queue and retry tests | corruption + partial-write + config drift | Separate transient retry, permanent hold, node drain, stale-writer prevention, immutable incident ID |
| GPU health | Deep mode + periodic DCGM, ECC, thermal, Xid, clock, NIC | nvidia-smi, Xid, NCCL, flight recorder, synthetic Docker | out of scope | fail-closed JSON + `skipped` states, CUDA NCCL path | P's check inventory and modes + S truthful JSON/fixtures + L flight recorder; avoid vendor-unsupported global thresholds |
| NCCL | multi-rank allreduce and workload tests | GPU multi-node collective benchmark script | queue design mentions NCCL; not executed | CPU Gloo math probe and CUDA NCCL path | retain both correctness and independent `nccl-tests` performance; two CPU ranks ≠ GPU validation |
| Cost/metrics | **ADR 0004**, 74-metric catalogue, simulation, price assumptions | limited | detailed placement/usage design | acceptance evidence status | Take event schema, cost categories and metrics from P but **prices/thresholds remain assumptions** |
| Security | operator roles and event governance in docs | fixtures, Slurm Docker | installer requires privileged real EC2 config | local .env ignored, pre-push gitleaks, source checks | never commit keys, munge secrets, private pack, raw telemetry; require actual gitleaks execution |
| Demonstrated validation | P includes extensive support tests and prior simulation artifacts; results not independently replayed here | Mallik `checkpoint/ckpt.py` self-test **passed in earlier local audit** | installer tests require real EC2 Slurm | 4 checkpoint CPU cases passed in this audit; Gloo timed out on current environment rerun | Ship fresh per-test evidence; no blanket verification based on printed `PASS` strings |

## B. Strengths worth retaining

**Pradeep:** Most complete *operations model*: structured reason/exit contracts, checkpoint receipt and store components, separate cost/alerts ADR, simulation and test support. This design should inform production integrations, but its simulated metrics are not measured ERA V5 GPU goodput. Do not assume `async_save` eliminates GPU→host stalls or S3 object uploads are a multi-object atomic transaction. Some modules are wired to specific paths/services and cannot be copied into `health/` without adaptation.

**Mallik:** Most concrete DCP and multi-rank correctness experimentation; `COMPLETE` commit ideas; clear checkpoint data-cursor and frozen-weight controls; Slurm launch uses **one torchrun agent per node**, avoiding 8× workers. Caveats: `COMPLETE` without verified per-object manifest/checksum does not prove distant S3 availability; a rank-local optimizer shard cannot automatically be replicated to new world size by selecting rank 0's file; model identical tensor values after load do not guarantee identical subsequent collective reduction order. Model-size assumptions (e.g. 30B MoE) must remain provisional.

**Max:** Strongest queue-specific design discipline: account/user/partition policies, installer render/verification, limit/requeue tests, controller backups and architecture options. It assumes dedicated train/eval EC2 nodes and explicitly excludes the **instructor-mandated Week 1 Docker Slurm test**. Its no-preempt design conflicts with P's preempting QoS design, so keep as explicit scenario, not a merged setting. `team7config`/installer should only be run after dry-run review; it modifies privileged services.

**Suresh:** Easiest local acceptance path and negative CPU tests, honest hardware `skipped`, self-contained Docker lab and reviewer-facing state table; corrected MUNGE socket traversal, virtual CPU topology and DenyOnLimit after user Mac feedback. Shortcomings: prior Docker/Slurm integration failed on user Mac and the hotfix is **not yet proven** under a rerun; S3 and multi-node commit remain proposals; an existing Gloo test timed out in the current review environment; never use static tests as proof of enforced Slurm limits.

**Murthy:** Not assessed. The archive named by the user was **not accessible as bytes**. No conclusions, ratings or file merges can honestly be made for this fifth source.

## C. Critical bugs / risky behaviors across implementation families

1. **Conflicting Slurm policy:** `PreemptMode=REQUEUE` versus isolated no-preempt partitions; choose after knowing spare resources, QoS, grace/kill behavior and checkpoint time. Default main-run protection is not evidence that cooperative preemption works.
2. **Immediate rejection semantics:** without QoS `Flags=DenyOnLimit`, an over-cap request might be pending rather than `sbatch` rejected; CI must test the real installed version and assert the exact intended contract.
3. **MUNGE directory and container simulation:** Docker hosts may have cgroup v2/systemd and virtual CPU topology constraints. A working Docker lab is **not** evidence of GPU GRES/cgroup isolation in production.
4. **Oversubscription:** `srun -n8 torchrun --nproc-per-node=8` on one eight-GPU node can spawn 64 workers. One `torchrun` agent per node is the selected pattern; `srun` with one task per GPU is a **different** alternative for `nccl-tests` binaries.
5. **Health false PASS:** unreadable dmesg/journal, missing DCGM/NCCL, or stale Xid should be recorded explicitly; don't collapse `skipped` into `pass`.
6. **Wrong Xid severity:** Xid 31/43 are often application-related; Xid 94 contained and Xid 95 uncontained differ. Run actual NVIDIA recovery policy per SKU/driver before draining.
7. **Checkpoint false assurance:** all ranks writing files is not an acknowledged distributed checkpoint until complete, durable remote shard verification and fenced latest-pointer commit. Local `os.replace` does not atomically commit an S3 directory.
8. **Silent model misload:** always verify full parameter/optimizer key schema, dtype and shape, provenance, data ordering and actual next-step loss; not only model `load_state_dict(strict=False)` or a `COMPLETE` marker.
9. **Requeue isn’t state restore:** Slurm can reschedule a shell, not restore Adam/RNG/ledger; the trainer must select latest *committed* state and validate before performing more updates.
10. **Replayed tokens ≠ unique committed tokens:** track separate counters and use denominators carefully in MFU/goodput and dollar calculations.
11. **Cost conclusions vary with region and instance price:** the repo includes sample P6 pricing, but no real account-specific quote; treat all dollar figures as modeling assumptions, not market facts.
12. **No secure operation without evidence:** do not run install scripts as root on real AWS without reviewing rendered configs, backup/rollback, credentials handling, storage paths and Slurm version.

## D. Unified implementation boundaries

- Instructor-facing code lives under `teams/07-infrastructure/`, not an arbitrary `week1-gate/` folder.
- Root-level `health/{launch.sh,gpu_health.sh,xid_scan.py}` and `ckpt/ckpt.py` are definitive Week 1 entrypoints.
- `slurm-docker/` is the CPU account/policy test; `integrations/` is a **future DeepSpeed interface**, not a tested GPU ZeRO implementation.
- `experimental/dcp/` includes the actual two-rank CPU DTensor DCP save/load experiment and Mallik provenance; its separate test is not a DeepSpeed optimizer restore.
- `reference/01_ADR_COVERAGE_AND_GAP_ANALYSIS.md` is the revised decision/gap register. Proposed teammate ADRs remain Proposed.
- **No Murthy code was integrated.** Integrate after file access, with its own provenance rows and tests.
