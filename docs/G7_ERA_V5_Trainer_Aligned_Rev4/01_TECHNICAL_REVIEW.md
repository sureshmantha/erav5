# G7 design review — 26 October 2026

> **REV4 MODEL SCOPE (2026-10-27):** ERA V5 is a new three-model capstone programme; Session 15 did not disclose final sizes. Do not assume a 1T or 40B V5 target; the 40B case was Session 3 homework and V4's 118.67B is historical evidence. V5 stage-specific tensor/optimizer inventories from G5 determine GPU, checkpoint and transport sizing. Consult `00_TRAINER_INTENT_V5_VS_V4.md` and `07_CONFIG_DRIVEN_MODEL_STAGE_CAPACITY.md`. Existing design controls and Week 1 CPU tests remain applicable independent of model size.


**Status: changes required; not a production approval.** Reviewed attached `checkpoint_design.md`, ADRs 0001–0003, `README-2.md`. The linked Gemini discussion and GitHub branch could not be retrieved; no assessment of their actual contents is claimed. V4 implementation and Session 15 empirical claims in the drafts have not been independently verified here. The source draft files are preserved unmodified in `source_drafts/`.

## Decision register

| ID | Severity | Where | Verified observation | Required change / acceptance evidence |
|---|---|---|---|---|
| CK-01 | BLOCKER | Checkpoint design §1, §5 | Saves only periodically, with admitted rollback to last committed checkpoint. Does not guarantee zero recomputation. | Define recovery guarantees separately; no impossible SLA. Introduce durable progress fence and optional every-step commit or replicated hot state. |
| CK-02 | BLOCKER | §5, §8 | Local `latest` and remote `latest` have different durability domains; emergency exit after local seal may leave nothing accessible on a replacement node. | Define durable commit only after storage survives loss of node and failure domain; a replacement must never trust local-only state. |
| CK-03 | BLOCKER | §3, §9 | "Gradients not saved" breaks exact restoration when stopping mid-gradient accumulation. | Only checkpoint *post-optimizer-step* or save gradient accumulation buffers and microstep/gradient scaler. Test both. |
| CK-04 | BLOCKER | §5 | `async_save` overlaps background write, but staging might block or race optimizer mutations; 'only stalls for copy' is not always true. | Pin version and stager mode; explicitly wait on staging-completion before modifying source weights; wait on upload-completion before marking committed. Benchmark both. |
| CK-05 | BLOCKER | §4–§6 | DCP load-time resharding is supported but not a blanket guarantee for optimizer, expert parallelism, or altered TP/PP mesh. | Prove compatibility for model + optimizer + expert placement; record mesh and optimizer format; on world-size change require new validation branch. |
| CK-06 | HIGH | §4 | 14B/parameter assumes one BF16 copy + FP32 master and Adam's two FP32 moments; actual architecture may use different optimizer/master/scaler/frozen states. | Derive size from actual G5 parameter groups; count **all stored parameters**, not only active MoE parameters. |
| CK-07 | HIGH | §5 | Multi-rank writes and S3 pointer update lack writer-epoch fencing and pointer CAS. | Immutable generation IDs, per-rank DONE, quorum/all-rank verification, SHA, writer lease/term, S3 conditional `If-Match` pointer update. |
| CK-08 | HIGH | §6, §9 | One matching held-out probe loss does not establish that optimizer/RNG/data order are identical. | Uninterrupted vs resumed trajectories across K+M steps, exact parameter and optimizer hashes, next-batch IDs, scheduler/RNG equality; negative tests. |
| CK-09 | HIGH | §7 | Using only stall time in Young's formula ignores bandwidth contention, remote durability lag and probability of local loss. | Use observed incremental *critical-path* overhead and time to durable recovery, with incident and storage models; optimize expected goodput. |
| CK-10 | HIGH | §8 | Three-copy retention and `1.2 × size` NVMe rule are not enough for a large V5 stage without in-flight staging, local failure domains, protected checkpoints. | Capacity plan with staging, snapshot peaks, 2 committed generations, uploads, scratch space, plus S3 retention & restore rehearsals. |
| CK-11 | HIGH | §3 | A skip-list on rollback risks silently dropping legitimate samples; `ledger_offset` must agree with optimizer progress. | Durable idempotent consumption/commit ledger; exact consumed token IDs and replay semantics. Never make a new skip list merely to avoid recomputation. |
| QU-01 | BLOCKER | ADR 0002 restart | Exit code 75 is not inherently a Slurm requeue operation. | Test `RequeueExit=75` and `RequeueExitHold=1` (or explicit requeue/hold operator) with `sacct` and `SLURM_RESTART_COUNT`. |
| QU-02 | BLOCKER | ADR 0002 preempt | `--signal=USR1@120` is a time-limit signal; QoS preemption signaling is separate. | Configure/test `PreemptParameters=send_user_signal`, GraceTime, signal handler, remaining hard-kill window and reservation behavior. |
| QU-03 | HIGH | ADR 0002 cloud | Self-managed Slurm wins control at significant operational burden; no quantified comparison to supported ParallelCluster. | Keep choice *provisional*: run an AWS ParallelCluster vs self-managed operational scorecard with TSAI Lab owner of final decision. |
| QU-04 | HIGH | ADR 0002 two accounts | "Two accounts" should mean two Slurm accounts (not AWS IAM accounts). Limits are not in the ADR. | Define associations, allowed partitions, GrpTRES/MaxTRES/MaxWall, max jobs, priorities; show rejected unauthorized jobs, accounting, preemption. |
| QU-05 | HIGH | ADR 0002 ops | Single head plus backup controller leaves slurmdbd/MariaDB disaster recovery unclear. | Backups, restore RTO, leader fencing, reproducible infrastructure/config, Slurm version pinning, credentials/munge protection, least-privileged IAM. |
| HE-01 | BLOCKER | ADR 0003 JSON policy | "skipped" handling plus `launch.sh` 'not fail' admits a GPU launch with required diagnostics unavailable. | Distinguish local CPU simulation from production; strict gating for mandatory GPU checks in production; schema version and fail-closed policy agreed with G8. |
| HE-02 | HIGH | ADR 0003 Xid | Xid mapping groups contained (94) and uncontained (95) together; new NVIDIA guidance is more nuanced. | Keep event severity, reset action, isolation state separately; use driver/GPU-specific NVIDIA Xid guidance and policy-reviewed mappings. |
| HE-03 | HIGH | ADR 0003 scan | Parsing 'since boot' every time can rediscover ancient incidents and repeatedly drain. | Store journal cursor/boot ID, incident ID, timestamps, per-GPU UUID and deduplication; reset only with operator-approved post-remediation tests. |
| HE-04 | HIGH | ADR 0003 NCCL | Running an all-reduce sweep in Slurm prolog on an occupied GPU node can collide with jobs. `busbw` formula is topology-specific. | Separate quick passive prolog from **exclusive** multi-rank/multi-node NCCL diagnostic job; report time, message sizes, algBW and official nccl-tests busBW without overgeneralization. |
| HE-05 | HIGH | ADR 0003 correctness | Low bus bandwidth isn't by itself proof of bad GPU without hardware-specific baselines. | Baseline per GPU generation, node topology and interconnect; thresholds require observed distribution and known control; run injection tests. |

## Strengths retained

- ADR template and *Proposed* discipline; self-managed Slurm evaluated against alternatives.
- DCP shards, per-file SHA-256 and immutable manifest approach; separate upload/restore verification.
- Strong emphasis on data-cursor and RNG restoration, negative tests for corruption, forced death and strict state loading.
- Machine-readable node-health contract, skipped-tool distinction, real NCCL correctness aspiration.
- Explicit responsibilities for G5 architecture, G8 monitoring, G9 evaluation, G12 serving.

## Not verified

- Gemini shared discussion contents; Pradeep's current GitHub branch; V4 original code/incident history; Session 15 figures and incident root causes; existence/behavior of any `ckpt.py`, Docker Compose or health script not attached; final GPU count/type, actual Slurm deployment, AWS cluster/network limits; customer-specific RTO/RPO budget.

## Approval criteria

**Week 1 conditional sign-off** needs: revised ADRs accepted by lead/instructor; runnable Slurm-in-Docker account/policy demo evidence; pure CPU bit-exact uninterrupted-versus-restored test in a *new process*; tamper-and-partial-write tests; health JSON with skipped unavailable tools; an actual GPU NCCL test must be explicitly marked pending until GPU availability. Never label local CPU/Gloo success NCCL success. `gitleaks` 0 findings is a separate required verification; it was not run here.

**Week 2–4 production sign-off** requires pinned hardware/software, measured checkpoint stall/commit/recovery, GPU/NCCL diagnostics, full multi-node failure injection, data/optimizer parity evidence, topology-dependent throughput, network/security and node-loss recovery tests.


---


## Session 15/16 follow-up review — 26 October 2026 (review revision 3)

**New primary evidence:** Session 15 (stability, debugging and live monitoring), Session 16 / identical capstone transcript (scaling, goodput and gates). **Acceptance decision remains “changes required”; no hardware or cluster tests were performed by reviewing these transcripts.**

### Findings added to the severity register

| ID | Severity | Finding | Concrete fix and proof |
|---|---|---|---|
| S15-G7-01 | P0 integration | Replay currently lacks a cross-team **deterministic batch contract** | Persist ordered batch IDs, packing/masking hash, sampler/mixture version and skip decisions; prove in fresh-process continuation |
| S15-G7-02 | P1 | Restore verification by one fixed probe cannot detect all optimizer/RNG/ledger divergences | Verify N steps, state digests, LR, token IDs and per-step loss bit patterns on CPU |
| S15-G7-03 | P1 | Hardware and numerical failure categories can get conflated | Separate Xid/hardware responses from gradient-spike/NaN responses; no automatic GPU drain for numerical issues |
| S15-G7-04 | P1 | Evaluation can trail behind new checkpoint generation | Trigger from immutable committed generation, prioritize read-only evaluation, track backlog and latency |
| S15-G7-05 | P1 | Simulator and stale/old-run data could falsely appear authoritative | Every event records attempt/generation + clock and source; UI labels synthetic/CPU/GPU/cluster proof |
| S15-G7-06 | P1 | No explicit checkpoint copy-time dry run on actual model shape | Randomly initialize final configured model, benchmark staging/local write/remote commit/load stages separately |
| S16-G7-07 | P1 | MFU and goodput may be double-counted in budget estimation | Define denominators and unique-token goodput; report wall-time components and measured pilot |
| S16-G7-08 | P2 | 16× compute shortcut asserted without executable recipe | Zero credit in production capacity calculations until evidence submitted |
| S16-G7-09 | P1 | 14 B/param projected onto an arbitrary all-trainable size without frozen/trainable breakdown | Inventory actual full/frozen/active/master/optimizer state; calculate before capacity commitment |
| S16-G7-10 | P1 | Spoken schedule references are ambiguous | Capstone plan/current instructor gate deadlines override informal holiday statements |

**What changed in verdict?** The team's original direction is reinforced: Slurm + real JSON health + sharded checkpoint/ledger correctness. Session 15 substantially increases required integration with Team 8 (alert/event truth), Teams 1/2 (sampler ledger), Teams 9/12 (checkpoint-triggered evaluation), and G5 (safe numerical intervention). Neither transcript proves any supplier-specific 1T throughput or that literal zero recomputation is achievable.
