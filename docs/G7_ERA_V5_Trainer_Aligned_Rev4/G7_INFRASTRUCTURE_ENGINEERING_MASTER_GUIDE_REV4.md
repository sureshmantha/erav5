# ERA V5 G7 — Infrastructure Engineering Master Guide

**Version:** 2026-10-27, revision 4 — trainer-aligned ERA V5 objective correction · **Audience:** Module 7 team, other capstone teams, instructors and technical reviewers  
**Coverage:** first principles; cloud scheduler/platform comparison; Slurm two-account laboratory; GPU readiness; NVIDIA Xid and NCCL; training launch; distributed checkpoint transactions; deterministic replay; G5-driven per-model/stage capacity and recovery.  
**Status:** independent design/research guide, **not** hardware acceptance, an approved instructor standard, a runbook validated on ERA V5 GPU nodes, or a claim that any specific V5 model size or GPU allocation has been approved.  
**Authoring method:** reviewed Sessions 2–16, the capstone transcript (text-identical to Session 16), team ADRs, instructor requirements, LightningLM public report and official vendor manuals. Session 1 remains unavailable as a verified transcript. See Sections 26–29 and the new source-traceability file for evidence and qualifications. Selected official references were freshly reverified on 26 October 2026; the bibliography includes additional links that must be validated in the deployment environment. Pin actual API/library versions before deployment.

> **Engineering principle:** Never confuse (1) intended behavior, (2) analytically predicted behavior, (3) behavior seen in a simulation, (4) behavior measured on a CPU, and (5) behavior verified on the *allocated GPU cluster*. Every approval claim must state which it is.

## REVISION 4 — scope correction from the instructor's V5 intent

**Previous assumption retired:** one trillion fully trainable parameters is not a documented ERA V5 target. Instructor: Session 15 says **three models whose sizes remain undisclosed**; Session 6 says do **not** initialize V5 from V4 weights; Session 14's 20–45T tokens and Session 16's 3B-active/6T example are **conditional ideas**, not adopted specifications. Session 3's 40B is a student homework planning scenario. V4's published 118.67B/5.93B-active LightningLM is reference evidence only. G7 should await real G5 configuration and size all stages from serialized tensor inventories. See new crosswalk and capacity docs alongside this guide.

## Contents

1. Evidence rules, scope, session lineage and what is unknown
2. Starting from first principles: what constitutes a correct distributed training step
3. Requirements and service-level objectives at four different levels
4. V5 model and stage configuration: trainer intent, real parameter accounting
5. Reference architecture and ownership boundaries
6. Platform comparison: AWS, Microsoft Azure, Google Cloud, NVIDIA, native Slurm, Kubernetes
7. Slurm internals: nodes, partitions, accounts, QoS, cgroups, security and scheduling
8. Two-account Slurm-in-Docker Week-1 proof and acceptance cases
9. Reliable launching: torchrun, srun, world size, rendezvous, signals, retries
10. GPU health philosophy and layered diagnostic design
11. NVIDIA Xid/SXid, ECC, DCGM and safe remediation
12. NCCL: distributed correctness, measurement, topology and failure analysis
13. Checkpointing fundamentals: what exactly must be saved
14. Transactional distributed checkpoint design with DCP and object storage
15. What bit-equal means and a rigorous continuation test protocol
16. Recovery models: no lost committed work, exact replay, zero recomputation
17. Failure injection, safety matrix and controlled recovery
18. Performance engineering, goodput, budgets and worked numerical examples
19. Monitoring, event schemas, incident response, security and auditability
20. Week-1 through Week-4 verification programme and acceptance gates
21. Fatal pitfalls and design review anti-patterns
22. Clarifications requiring TSAI Lab approval
23. Annotated primary references and industry reading sequence
24. Primary-source benchmark case studies and reproducible reporting
25. First-principles operational walkthrough and owner handoffs
26. NEW — Session 15 stability/incident-response integration
27. NEW — Session 16 capacity and compute-budget integration
28. NEW — Revised G7 interfaces, acceptance tests and decisions
29. NEW — Fact checking and transcript qualification register

---

## 1. Evidence rules, scope and ERA V5 lineage

### 1.1 Evidence levels

Use this label in architecture discussions, pull requests and dashboards:

| Level | Meaning | Example | May qualify Week 1? |
|---|---|---|---|
| `SOURCE` | Directly present in official docs, instructor instruction or paper | Slurm `RequeueExit` documentation | Supports design only |
| `ANALYTICAL` | Calculation from explicitly declared assumptions | configuration-specific checkpoint bytes from actual optimizer tensors | Supports capacity hypothesis |
| `PROPOSED` | A decision not approved or executed | Conditional S3 `latest` pointer | Supports ADR only |
| `SIMULATED` | Scripted output or model of hardware | Dashboard Xid 79 button | UI demonstration only |
| `LOCAL_EXECUTED` | Actual CPU/Docker process with captured output | Gloo 2-rank all-reduce, CPU replay | Yes, relevant items |
| `GPU_EXECUTED` | Actual named GPU, driver, command, logs and timestamp | `all_reduce_perf` on the allocated node | Later hardware gate |
| `CLUSTER_PROVEN` | Fault tests on full parallel topology and storage | Loss replay after rack/node failure | Gate 3/4 |

**Never describe a simulated hardware number as measured.** Actual GPUs allocated to ERA V5, placement, storage, model configuration and measured throughput are **not established** by supplied files. Hardware matrices below are conditional planning examples.

### 1.2 What we can source from previous sessions

The accessible attachments contain raw source material for Sessions 2–16, including the **complete Session 15 source** (10 October 2026) and **Session 16** (24 October 2026). The file `ERA V5 Session capstone.docx` has exactly the same extracted paragraph text as `ERA V5 Session - 16.docx`; these are **one textual source, not independent corroboration**. Session 1 has not been supplied as a verified complete transcript. Instructor statements are treated as course requirements/context, not automatic independent confirmation of third-party benchmarks.

Relevant connections:

- **Sessions 2–5 — tokenizer, mixture, cleaning:** versioned training data, text identity, reproducible data sources and a held-out evaluation firewall matter to infrastructure because an identical GPU state fed *different tokens* will produce different learning.
- **Session 6 — training-data execution / consumption and learning ledgers:** the authoritative next batch must be persisted with model state. Batch ID, packed loss mask, token positions, mixture decisions and skip records are part of exact replay.
- **Sessions 7–9 — architecture, attention, loss and masking:** rebuilding a model under changed causal masks, loss denominators, token shifting or precision can change numerical values independently of checkpoint corruption.
- **Session 10 — gradients and accumulation:** a resume in the middle of gradient accumulation must either save partial gradients and accumulation counters or roll back to a clean optimizer-step boundary; microbatch weighting must remain unchanged.
- **Session 11 — Adam and scheduling:** weights alone are not sufficient. First and second optimizer moments, per-param step, loss scaler and learning-rate schedule must be restored exactly. Adam bias correction cannot be reconstructed only from LR and weights.
- **Session 12 — ZeRO-0/1/2/3:** sharded model and optimizer state must be reconstructed according to ownership/parallel layout; sharding storage does not automatically reduce all compute or activation memory.
- **Session 13 — tensor, pipeline, sequence, context parallelism:** rank-to-tensor mapping and communication/reduction order matter, especially when the world size changes; reversible training may alter activation memory versus recomputation trade-offs.
- **Session 14 — sparse MoE:** expert ownership, router state, active versus stored weights, frozen bases and trainable adapters determine actual checkpoint size and whether cross-layout loading is possible.
- **Session 15 — stability, divergence, monitoring and incident response:** deterministic batch replay, gradient-norm/spike telemetry, held-out evaluation, expert-load tracing, immutable rollback/skip decisions, and checkpoint-transfer benchmarking become explicit G7↔G8↔G1/G2 interfaces.
- **Session 16 / capstone briefing — scaling and capacity:** distinguish active versus stored parameters, MFU from goodput, planned versus achieved tokens, batch/LR assumptions, and the checkpoint/health/Slurm dependencies of training on 28 November. The capstone file repeats Session 16; it is not a separate independent source.

**LightningLM is a public experience report, not an independent full-scale benchmark reproduction.** Its 120B stage was a sparse MoE with a quantized expert base and trainable adapters. It reports a damaged step-3009 checkpoint leading to a restart from step 2909, as well as a converter that silently mismatched checkpoint keys. The report itself grades claims by artifact/checkpoint/log/transcript/recollection strength. Reference: https://lightninglm.theschoolofai.in/ . In our design, the lessons are (a) atomically commit, (b) strict key/dtype/shape checks, (c) run loss+state canaries, (d) record the exact release/harvest checkpoint identity, (e) never assume the current training step is durable.

### 1.3 Instructor's actual Module 7 Week-1 deliverables

- Write `teams/07-infrastructure/docs/queue_design.md` and demonstrate **Slurm in Docker with two *Slurm* accounts enforcing actual policies**. Accounts are not two AWS credentials or two VMs.
- Write `health/launch.sh`, `gpu_health.sh`, `xid_scan.py`; health emits valid JSON and accurately marks missing tools `skipped`; provide NCCL all-reduce validation where GPU hardware exists.
- Write `docs/checkpoint_design.md` and `ckpt/ckpt.py`; **a CPU save/new-process restore must provide bit-equal subsequent loss**.
- Keep secrets out of Git; run `gitleaks detect --source .` before push; publish evidence under reviewed PRs.
- Distinguish Week-1 laptop/CPU and Week-2 physical shared GPU-node tests; do not invent a real-node NCCL pass.

---

## 2. First principles: what happens during one training step?

A logically successful synchronous data-parallel optimization step has these events:

1. **Admission:** scheduler grants GPU/rank/CPU/memory/nodes for a specific immutable job specification.
2. **Environment:** driver/runtime/container/library/config/data digest and topology are checked; each process is assigned its local device.
3. **Acquire batch:** dataloader consumes a defined sequence of immutable `sample_id` / token-span / loss-mask records.
4. **Forward pass:** model and its stochastic operations produce logits and loss.
5. **Backward pass:** per-rank gradients computed; accumulation may span several microbatches.
6. **Collectives:** reduce-scatter/all-reduce/all-gather synchronize the intended mathematical global gradient, under a known data-parallel subgroup.
7. **Optimizer:** states (`m`, `v`, step counters and others) and master/model weights change once; LR scheduler advances under specified ordering.
8. **Commit boundary:** an optimizer step is considered *recoverably committed* only when the agreed durable state and ledger position have been persisted. In-memory completion is a weaker statement.
9. **Record evidence:** step ID, attempt/epoch, effective examples/tokens, model/optimizer version, numeric loss, grad norm, nonfinite flag, timing and rank-map topology are logged.
10. **Repeat or terminate:** scheduled save, preemption, fault or training completion can occur here or asynchronously and must be coordinated across ranks.

Let the full training state after optimizer step `k` be:

`S_k = (model weights, optimizer, LR/scaler, RNG streams, sampler/mixture/ledger, run config, parallel mapping, auxiliary learned state)`.

A deterministic next-step transition is mathematically `S_(k+1) = F(S_k, Batch_(k+1), Software, Hardware, CollectiveOrder)`. Equal checkpoint bytes are useful, but **full, exact replay additionally needs identical subsequent inputs and operations**. Floating-point addition is generally nonassociative: regrouping a reduction across 64 vs 56 GPUs may change low-order bits. A strict promise of bitwise identity across changed topology is not justified without additional constraints and proof.

### 2.1 Separate reliability layers

- **Scheduler reliability:** job gets a consistent allocation, restarts/holds when commanded.
- **Hardware readiness:** devices, interconnect and diagnostics work within acceptance policy.
- **Communication correctness:** collectives complete, produce correct values and satisfy measured network expectations.
- **Checkpoint durability:** an entire reconstructible state survives the specified failure domain.
- **Scientific correctness:** after restore, the same examples and optimizer sequence create the same results in the declared deterministic environment.

A single green “GPU ready” flag does not prove all five layers.

---

## 3. Requirements and SLOs: define the contract before selecting tools

Write an RFC with these separately measurable goals:

| Objective | Proposed measurable contract | Evidence |
|---|---|---|
| Admission correctness | Every job has authorized Slurm account/QoS, requested CPU/GPU/memory; unauthorized requests rejected | `sacctmgr`, `scontrol`, `squeue`, negative `sbatch` logs |
| Hardware safety | Required checks either pass or node is withheld; missing diagnostics never converted to `pass` | JSON schema fixtures + real `nvidia-smi`/DCGM |
| NCCL correctness | Correct output under known all-reduce and failure behavior | raw `nccl-tests`, configured timeout |
| Durable checkpoint | No `latest` points at incomplete or hash-mismatched generation | kill/corrupt/competing-writer tests |
| Exact CPU replay | Same hardware/software/layout and input order: every post-resume loss bit pattern and complete state digest matches | uninterrupted vs separate-process run |
| Production continuation | GPU version/precision and collective ordering fixed; loss verification at declared level | repeated single-/multi-node tests |
| No lost *committed* steps | State with committed receipt remains retrievable under defined node failures | storage+manifest+fencing faults |
| Zero recomputation | Distinct RPO = 0 target; only achievable with suitable per-step durability/replication/commit semantics under defined failures | fault matrix and cost study |
| Recovery-time objective | Define RTO percentile from failure detection to resumed useful training; do not invent until measured | incident event clocks |
| Goodput | measured useful loss-bearing tokens divided by billed elapsed accelerator time | scheduled/job runtime, consumptions ledger |
| Budget and tenant fairness | account-wise GPU-hours, GPU idle, pending time, aging and priority | `sacct`/`sreport` |

Before acceptance specify failure domain: *single training process*, *single GPU*, *one physical host*, *AZ/network segment*, *entire object-store region*, and *malicious operator* are not equivalent. RPO = 0 for a node loss does not imply RPO = 0 for object-store-region failure.

---

## 4. Trainer-aligned ERA V5: model/stage configuration and capacity

### 4.1 ERA V4 is the evidence baseline, not the V5 specification

The publicly documented V4 [LightningLM report](https://lightninglm.theschoolofai.in/) reports four grown stages, ending at **118,670,450,556 stored parameters and 5,927,559,036 active parameters per token**. It used trained adapters on a quantized expert base at the largest stage; the reported 2.26B expert-path adapter optimizer parameters are a much more useful clue to checkpoint economics than multiplying 118.67B by an all-trainable Adam-state constant. V4's nodes used eight GPUs across multiple generations, with measured token throughput. **These are historical V4 results, not V5 deployment targets.** V4 also documents damaged checkpoints and silent conversion misloads, which define mandatory negative tests.

ERA V5's instructor says **three models with sizes intentionally withheld** (Session 15), a new architecture rather than restoring V4 weights (Session 6), end-to-end training with fuller post-training and serving/release (Sessions 15/16 and [V5 programme](https://registrations.theschoolofai.in/courses/era-v5)). The 40B example in Session 3 is **an individual assignment**. The 20T–45T token ambitions in Session 14 are conditional on budget and an unverified efficiency technique. Therefore G7 cannot honestly specify total/active/trainable V5 parameters, number of GPU nodes, checkpoint bytes or performance targets until G5 and the instructor commit the actual stages and operator topology. Consult `00_TRAINER_INTENT_V5_VS_V4.md` and `07_CONFIG_DRIVEN_MODEL_STAGE_CAPACITY.md` for the complete source crosswalk and contract.

### 4.2 Calculate from separate stored, trainable, frozen and active weights

Let `P_stored` be all persistent model parameters, `P_train` the subset updated by the optimizer, `P_frozen=P_stored−P_train`, and `P_active` the weights participating in one token's compute. These are **different quantities** in MoE. With a *specific* BF16+FP32-master Adam representation (and clean optimizer-step boundary), an approximate state size is:

`S_checkpoint ≈ bytes_frozen_representation + 2*P_train + 4*P_train(master) + 8*P_train(Adam m/v) + quantization/optimizer metadata + nonparameter state`.

If a model saves its BF16 trainable weight, FP32 master copy and two FP32 moments, the trainable portion is roughly 14 bytes/trainable parameter. This is **not** 14 bytes × total MoE parameters when quantized/frozen experts or trained adapters are used. If an implementation does not carry an FP32 master, or holds additional optimizer states, adapt the formula. Peak device RAM also includes activation, gradients, gather and temporary buffers; file size is not device residency. Treat these as accounting formulas, *not performance measurements*.

### 4.3 Storage and network planning is conditional on the actual stage

For each of the trainer's selected V5 models/stages, G5 supplies a frozen tensor/optimizer inventory and G7 measures checkpoint bytes, sharding ownership, per-rank host staging, sustained node→NVMe→remote durable throughput, and restore. Transfer-time lower bound = `measured checkpoint payload bytes / measured sustained end-to-end bytes/s`, excluding verification and tail latency. Measure p50/p95/p99 and concurrent checkpoint/upload overhead. NVLink peak bandwidth cannot stand in for storage/upload bandwidth.

### 4.4 Stage boundaries change the meaning of exactness

Within an identical training stage, process restart should recover exact model+optimizer+RNG+ledger and produce bit-equal next steps under a declared deterministic environment. When V5 intentionally grows its architecture or changes quantization/optimizer layout, it is a **new stage with an explicit mapping/continuity test**, not a byte-identical restart. Record parent checkpoint, converter revision, mapping audit and held-out loss before and after the transformation; never silently load with `strict=False` and declare success.

### 4.5 Goodput and checkpoint intervals

Retain the fault/interval tradeoff from Sections 16 and 18, but supply measured stage-wise step time, stall, upload interference and recovery profile. `MFU × goodput` can double-count downtime if MFU was measured against the same end-to-end elapsed window; define measurement boundaries first. Report consumed, committed, unique and replayed tokens distinctly. Parameterized planning is valid before model sizes arrive; numeric GPU/cost commitments are not.

## 5. Architecture: the actual separation of concerns

```text
USER / TSAI SUBMISSION INTERFACE (no student cloud secrets)
      │      immutable run spec: code/image/config/data digests, account/QoS
      ▼
SLURM CONTROL PLANE ─── Accounting DB / audit ─── Reservations + budgets
      │ allocation, rank map, admission, priority, requeue
      ▼
NODE PROLOG / HEALTH ───── DCGM / NVML / Xid/SXid / fabric inventory
      │ PASS (required checks met)
      ▼
LAUNCH / TORCHRUN ──────── Model / data loader / RNG / collectives
      │                             │                  │
      │                             └── consumed batch ledger + optimizer state
      ▼
CHECKPOINT COORDINATOR ─── DCP per-rank staging/save to unique generation
      │ every expected rank receipt + checksum + manifest verification
      ▼
DURABLE COMMIT ──────────── immutable manifest + conditional latest pointer
      │ immutable checkpoint ID + run attempt / fencing epoch
      ▼
RECOVERY CONTROLLER ─────── Slurm requeue + new node allocation + strict restore
      │ record branch identity (exact or non-bitwise resharded)
      ▼
MONITORING / EVALUATION ─── read-only committed artifacts / metrics / alerts
```

**Owner boundaries:** Slurm owns allocation and job state; training runtime owns tensors and data-consumption state; checkpoint manager owns safe persistence; recovery coordinator owns which committed generation is eligible to resume; health system owns per-check evidence and quarantine recommendation; Team 8 owns shared event schema/alarms, with G7 producing source telemetry; Teams 9/12 consume only committed read-only checkpoint identities.

A single source of truth for `run_id`, `attempt_id`, `generation_id`, `committed_step`, `ledger_offset`, and `image_digest` prevents divergence between scripts.

---
## 6. Cloud and cluster platform comparison

**Principle:** Choose a workload control plane separately from a hardware/data plane. Slurm, Kubernetes, EFA/NVLink/NCCL, cloud IAM and distributed checkpoint engines solve different problems. A vendor-managed cluster does not remove the need to checkpoint complete data+optimizer state correctly.

| Platform | Native/recommended queue/orchestrator choices | Strong points | Trade-offs and critical checks |
|---|---|---|---|
| **AWS self-managed EC2** | Directly operated Slurm on GPU EC2 | Exact Slurm/QoS/prolog control; custom stacks | Operate slurmctld/slurmdbd/MUNGE/accounting DB, node lifecycle, EFA+FSx, AMI/image pinning, control-plane HA; high operational burden |
| **AWS ParallelCluster** | Provisioned Slurm cluster with AWS-managed deployment tooling | AWS-oriented fleet/queue/EFA/FSx integration, less manual bootstrap | Still operates Slurm and cloud costs; IaC upgrades, IAM and policy changes require testing; check supported instance/AMI/region and version |
| **AWS Parallel Computing Service (PCS)** | Managed Slurm cluster control plane | Offloads scheduler management while retaining Slurm job workflow; official 2026 user guide documents queues, compute-node groups, storage and multi-node jobs | Cloud managed-service charges and integration responsibilities remain; verify supported versions, networking, accounting, IAM, region and scheduler feature compatibility |
| **AWS SageMaker HyperPod (Slurm)** | HyperPod Slurm orchestration and auto-resume features | Built-in health checks, automatic node replacement and resume integration | Auto-resume restarts job from its entrypoint and user-saved checkpoint; does **not** provide the missing checkpoint nor bit-equal guarantee; feature availability/version/cluster conditions |
| **AWS SageMaker HyperPod EKS** | Kubernetes scheduling, task governance | Namespaced team quotas, priority, lending/borrowing, unified training/serving options | More Kubernetes expertise; topology/gang admission and RDMA/EFA integration; do not assume equivalent Slurm policy semantics |
| **Microsoft Azure CycleCloud Workspace for Slurm** | Deployed Slurm plus Azure network/storage/access | Slurm familiarity, HPC autoscaling, packaged environments | Ownership of customized configs and VM/storage/network remains; select InfiniBand-enabled GPU SKUs and validate drivers |
| **Azure CycleCloud custom** | Slurm, PBS, LSF etc. | Supports existing HPC scheduler patterns and custom node pools | Higher integration and ops work than preconfigured Workspace; check quotas, placement and container run model |
| **Google Cloud Cluster Toolkit** | Slurm on GCE with templated Terraform/Packer; alternative GKE/Kueue | Declarative HPC GPU cluster blueprints and customization | Need to validate chosen GPU shape/network/storage; manages deployment, not model-state recovery |
| **GKE/Kueue** | Kubernetes workload admission/quotas and gang-aware training integration | Kubernetes namespaces and shared quotas | Requires cluster/network operators, device plugins, training controllers; different policy/mental model from Slurm |
| **NVIDIA Base Command Manager** | Cluster manager integrating Slurm, monitoring and provisioning | Single operational control plane for multi-user GPU infrastructure | Licensing/deployment cost, version coupling, site integration; not automatically a managed cloud product |
| **Bare Slurm on institutional nodes** | Slurm | Transparency and full control, familiar HPC semantics | Requires security, health, quotas, accounting, HA and recovery engineering |

**Primary vendor sources:**
- AWS ParallelCluster: https://docs.aws.amazon.com/parallelcluster/latest/ug/Scheduling-v3.html ; EFA: https://docs.aws.amazon.com/parallelcluster/latest/ug/efa-v3.html ; FSx+EFA: https://docs.aws.amazon.com/parallelcluster/latest/ug/tutorial-efa-enabled-fsx-lustre.html
- AWS PCS managed Slurm: https://docs.aws.amazon.com/pcs/latest/userguide/getting-started.html ; AWS HyperPod auto-resume: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency-slurm-auto-resume.html
- Microsoft: https://learn.microsoft.com/en-us/azure/cyclecloud/overview-ccws ; https://learn.microsoft.com/en-us/azure/cyclecloud/concepts/plan-and-size-hpc-clusters
- Google Cluster Toolkit: https://docs.cloud.google.com/cluster-toolkit/docs ; https://docs.cloud.google.com/cluster-toolkit/docs/quickstarts/slurm-cluster
- NVIDIA BCM: https://docs.nvidia.com/base-command-manager/product-manuals/admin/workloadmanagement

### 6.1 Architectural decision for ERA V5

**Retain Slurm as the *interface* for the Week-1 test** because the instructor specifically requires Slurm in Docker with account-limit evidence. **Do not decide self-managed EC2 versus PCS/ParallelCluster/HyperPod without TSAI Lab's access policy, budget, controller availability and supported cluster product.** If the lab provisions Slurm already, overriding it with another controller would increase risk unnecessarily.

**Cloud network mechanics:** AWS EFA, NVIDIA NVLink and generic InfiniBand can all participate in different segments of a GPU training path, but they are not interchangeable. EFA requires supported instance types and placement/AZ considerations; intra-node NVSwitch uses a distinct connectivity graph. Kubernetes overlay network performance and host-network RDMA permissions must be verified. The point is to *measure* actual CUDA/NCCL communication using your actual nodes and process groups, not to rely on peak link specification.

### 6.2 Platform choice checklist (no inferred facts)

Ask the instructor/operator for: provider, region/AZ, GPU SKU/HBM/GPU count per node, number of nodes, tenancy/exclusivity, network adapter inventory, NVSwitch/Fabric Manager status, instance type, AMI/image, Slurm version, writable checkpoint storage/mounts and quotas, IAM roles, preemptibility, on-call and rollback powers. Only then can you calculate actual V5-stage feasibility and checkpoint throughput.

---

## 7. Slurm: mechanism, policies and proven failure modes

### 7.1 What Slurm actually owns

- `slurmctld`: scheduling controller, job/partition/node state and policy engine; should be HA and backed up.
- `slurmd`: worker daemon on each compute node, launches/manages allocated work.
- `slurmdbd` + SQL database: accounting associations, historical usage, QoS, fair share.
- MUNGE and OS/IAM-derived identities: authentication; avoid one shared privileged student login.
- `sacctmgr`: account, user association and QoS definitions.
- `sbatch`: submit noninteractive job specification; `srun`: launch job steps within an allocation; `salloc`: interactive allocation.
- `gres/gpu`, TRES and `task/cgroup`: resource representation, isolation and enforcement.
- `Prolog`/`Epilog`, health checker and node state: admission and cleanup hooks, not a substitute for in-job validation.

**Critical distinction:** A Slurm *account* is an accounting/policy grouping, not a separate EC2/Azure billing account. The Gate-1 two-account test should compare `mainrun` and `teams` and prove that limits, permissions and priority work. Configuring two user names without enforced limits is insufficient.

Official: https://slurm.schedmd.com/qos.html ; https://slurm.schedmd.com/gres.html ; https://slurm.schedmd.com/cgroups.html ; https://slurm.schedmd.com/fair_tree.html

### 7.2 Logical namespace vs compute placement

Define two broad account tiers and child team associations. Example hierarchy (proposal, not executed):

```text
root
├── mainrun    ── QoS=critical, partition=train, exclusive nodes, no opportunistic preemption
└── teams      ── QoS=normal or debug, account descendants perf/eval/quant/arch
                 enforced per-team GrpTRES / MaxJobs / MaxWall and fair-share
```

Choose limits after cluster capacity is known. **A priority value does not automatically reserve capacity.** A main run that uses all nodes leaves no spare nodes for short diagnostics, eval or quick recovery. Partition design and reservations need deliberate capacity/budget decisions.

Scheduling priorities should combine fair-share, age and QoS; document actual weights and `sprio` behavior. Priority is not a quota, and accounting reports are not admission policy. Use `AccountingStorageEnforce` with the association/limit/QoS rules required by your Slurm release.

**Isolation:** Configure `task/cgroup`, GRES device isolation, CPU core and memory confinement. Enforce GPU assignment (not merely visible GPU count). For MIG, device identities and monitoring limitations differ from whole-GPU allocations. Test negative admission with requests exceeding quotas.

### 7.3 Requeue and preemption are separate policies

`#SBATCH --requeue` **allows** requeue; it does not say which application exit codes cause it. `RequeueExit=75` in `slurm.conf` can drive requeue on that exit code, and `RequeueExitHold` can hold selected exits. `SLURM_RESTART_COUNT` identifies the number of requeues. See https://slurm.schedmd.com/slurm.conf.html.

`#SBATCH --signal=B:USR1@120` can tell the batch shell before its **time limit** (Slurm may send early). It is **not universally the signal for QoS preemption**; configure and test `PreemptType`, `PreemptMode` and the current Slurm version's preemption signaling, including `PreemptParameters=send_user_signal` when applicable. The signal must reach the training coordinator and all participating ranks through a documented protocol. See https://slurm.schedmd.com/sbatch.html and https://slurm.schedmd.com/qos.html.

The durable exit classification needs an explicit policy, e.g.:

| Exit / event | Behavior | Queue side | Training side |
|---|---|---|---|
| 0 | Completed | FINAL COMPLETE | final manifest and metrics |
| 75 (provisional) | Transient retry or requested controlled restart | requeue with cap/backoff | resume last durably committed checkpoint |
| 1 (provisional) | Deterministic fatal software/NaN/security error | HOLD, page, don't loop | preserve diagnostic bundle; don't overwrite good checkpoint |
| Hardware fault | Node unhealthy | drain or isolate according to vendor guidance | terminate process group; restore from committed generation |
| Time limit/preemption | grace save if deadline permits | requeue/hold per policy | bounded save, avoid deadlock/barrier |
| Controller failure | controller HA recovery | controller takeover | fencing and idempotent submission |

The mapping must be confirmed in your *specific* Slurm version, not hard-coded into comments alone.

### 7.4 Gang scheduling and the global job

A synchronous distributed training job needs *all required ranks* concurrently available; one slow/bad rank can stall the whole communicator. Prefer coherent allocations on local fast-topology islands where possible, avoid mixed GPU architecture within a collective unless tested, and reserve a genuinely useful failure-replacement capacity if the reliability/business case justifies it. Never ask the scheduler to use 56 ranks for a hard-coded 64-rank optimizer unless the model/runtime explicitly supports resharding/restart and accepts altered numerics.

### 7.5 Observability to collect from Slurm

- submission timestamp, eligibility time, pending reason, start time, end time;
- account/QoS, user/team, partition, job ID, attempt count, requested and allocated GPU TRES, wall time, GPU-hours;
- node list, GPU UUIDs, image/config hashes, partition and reservation IDs;
- preemption source, signal, grace period, exit code, requeue count;
- `squeue`, `sacct`, `sreport`, `scontrol show job/node`, `sprio`, `sshare`, controller/slurmd logs.

**Cost optimization:** queue wait, idle reservation, kernel performance and checkpoint overhead must be reported separately. Raising GPU utilization is harmful if it preempts the protected training job or starves failure recovery.

---

## 8. Slurm-in-Docker: Week-1 experiment is an integration proof, not cloud Slurm

### 8.1 Suggested local topology

```text
Docker Compose isolated network
 ├── slurm-controller (slurmctld)
 ├── accounting (slurmdbd + SQL storage)
 ├── worker-a (slurmd; CPU resources)
 ├── worker-b (slurmd; CPU resources)
 └── test-runner (same version client binaries, non-root submission identity)
```

Pin a distribution/Slurm version. Use a persistent SlurmDBD/accounting database and explicit MUNGE credentials for test only; don't commit real cluster MUNGE keys. The container topology proves policies and restart *logic*, **not** real GPU GRES, PCI topology, slurmd interactions with privileged host cgroups, EFA networking or NCCL.

### 8.2 Minimal test plan (positive and negative)

1. Boot services; assert `sinfo` lists two CPU compute nodes and `scontrol ping` sees controller.
2. `sacctmgr` create exactly defined test `mainrun` and `teams` accounts, distinct OS users/associations as required, and two explicit QoS policies.
3. Submit a valid `mainrun` job, a valid `teams` job, and an invalid/unauthorized account request. Record accepted job IDs and rejected stderr.
4. Submit jobs above the configured CPU/GPU-equivalent TRES or walltime cap; verify **rejection or pending due to policy**, not merely a logged message.
5. Force resource competition and inspect `sprio`, `sshare` and final order. Repeat after adjusting usage to demonstrate fair-share behavior as intended.
6. Run a dummy step-writing job that exits 75; verify Slurm requeues due to an *explicit* `RequeueExit` config, not manual resubmit. Confirm `SLURM_RESTART_COUNT` increments and run ID is stable.
7. Submit a failing exit-1 job; demonstrate it is held rather than endlessly retried.
8. Inject a mock node-health failure; record whether node is DRAIN and other jobs avoid it.
9. In a controlled high-priority job arrival, observe precise preemption signal, time window, grace and worker requeue semantics. Use **captured logs**, not assumed behavior.
10. Restart `slurmctld` and `slurmdbd`; verify test accounts, histories and queue state survive as expected.
11. Validate that creating or editing an account without authorization is refused, and jobs cannot use another team's paths/credentials.
12. Cleanly tear down; export `docker compose ps`, relevant logs, `sacct` and Slurm config hashes.

This is the specification for the test. **No claim that these commands passed is made until your actual Docker service outputs exist.**

### 8.3 Acceptance evidence directory

```text
evidence/week1/slurm-docker/
  versions.txt                # Slurm, image IDs/digests, host, Docker versions
  config-redacted/             # slurm.conf, cgroup.conf, allowed qos/account settings
  job-submissions.jsonl       # job IDs, account, qos, expected outcome
  sacct-output.txt
  scontrol-node-state.txt
  fairshare-before-after.txt
  preemption-signal-timeline.json
  requeue-attempts.jsonl
  test-results.xml
  sha256sums.txt
```

Config may contain private host details; publish sanitized test configuration only. Do not publish password material or signed access tokens.

---

## 9. `launch.sh`: correct process architecture and failure containment

### 9.1 Never multiply launcher processes unintentionally

An `srun` job step creates the number of tasks requested by Slurm. A `torchrun` **agent** on each node creates `--nproc-per-node` worker processes. If an eight-GPU node has `--ntasks-per-node=8` and *each* of those runs `torchrun --nproc-per-node=8`, the step may create 64 workers for eight devices: an obvious over-subscription defect.

For an eight-GPU Torchrun-per-node pattern, request one Slurm agent task per node, then eight workers per agent. PyTorch TorchTitan's multinode Slurm example demonstrates one launcher per node: https://github.com/pytorch/torchtitan . Slurm and PyTorch options vary by deployment; validate on 1 node and 2 nodes before creating production launch scripts.

### 9.2 Reference *pattern* for one-node Torchrun (adapt to site)

This block is deliberately an architectural example rather than an already-tested script:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=era-v5-train
#SBATCH --account=mainrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -Eeuo pipefail

# The production wrapper must validate the allocation and pinned image first.
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
: "${MASTER_PORT:?set an allocation-specific free rendezvous port}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Ensure exactly one agent per node, and each agent owns the locally allocated GPUs.
srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
  torchrun --nnodes="${SLURM_NNODES}" --nproc-per-node=8 \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-backend=c10d --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train.py --config configs/approved-v1.yaml
```

**Do not paste this as production `launch.sh` without adapting** Slurm task options, container runtime, device visibility, correct `--gres`/`--gpus-per-task`, rendezvous port allocator, fault signaling, EFA/NCCL settings, configuration file, logging directory, retries and checkpoint integration. Choosing `MASTER_PORT` requires an actual collision-free port policy; this example intentionally refuses to invent one. `--rdzv-id` must be stable/unique for the job attempt and must not alias an unrelated concurrent job.

### 9.3 Engineering constraints for launch

- Container/image digest, model config hash, tokenizer/data-manifest hash and expected checkpoint schema validated before trainers spawn.
- AuthZ: job account granted the right storage paths/roles; no static secrets in scripts.
- Device map: each local rank maps to exactly one assigned GPU, logs PCI UUID, logical CUDA index and NCCL ranks. One torchrun agent per node unless an explicitly different launch strategy is validated.
- CPU: NUMA/locality and pinned threads tuned for tokenization, I/O and dataloaders; oversubscription can defeat GPU utilization.
- Signals: shell/process-group propagation and all-rank save decision; no isolated rank stuck in a collective barrier.
- Restart: `latest` means last *committed* generation, not largest directory name or most recent local file.
- Crash loop: bounded retries and backoff; known deterministic fatal errors held for operator review.
- Training steps: deterministic boundary for durable checkpoint; no partial optimizer step labeled saved.
- Training completes only after final checkpoint and commit receipt where specified.
- Metric integrity: separate per-rank logs and one consensus event stream.

### 9.4 Frequent launcher failures

Port collision, hostname unresolvable, wrong node rank, bad NIC selected by NCCL, firewall/security-group failure, EFA plugin missing, incorrect `CUDA_VISIBLE_DEVICES`, mixed versions, duplicated agents, missing `MASTER_ADDR`, zombie old ranks after node loss, shared file lock contention, fake GPU health PASS, SIGUSR1 delivered only to batch shell but not trainer, job exits 75 but not requeued, repeated `srun` inside restart causing a duplicate training attempt.

**Mitigation:** run a launch matrix with 1 rank, 8 ranks single node, 16 ranks two nodes, preemption before/after optimizer update, simultaneous two independent jobs, corrupted checkpoint, and wrong config digest. Record actual process IDs and per-GPU UUIDs. For scale, add rank start-time skew, stragglers, Slurm-controller failover and network partitions.

---
## 10. GPU health: design an evidence-based safety system

**GPU health is not a single yes/no command.** It is a multi-layer hypothesis about whether a *particular GPU and interconnect topology* can execute a *particular workload* safely and fast enough. A successful `nvidia-smi` query proves driver visibility, not memory accuracy, RDMA connectivity or stable throughput.

### 10.1 Diagnostic modes

| Mode | When | Intended cost | Checks | Actions |
|---|---|---|---|---|
| INVENTORY | provisioning/maintenance | low | GPU UUID, SKU, HBM, PCI location, driver, firmware, visible MIG, NICs, Fabric Manager status, clock/power policy | reject incompatible node |
| PROLOG | immediately before allocation | seconds, non-contentious | mandated GPU count and UUID presence, recent critical fault since watermark, ECC counters, NVLink basic state, required agents | PASS/WARN/BLOCK; never run a destructive stress suite in general prolog |
| CONTINUOUS | during training | lightweight, rate-limited | DCGM/NVML telemetry, Xid/SXid events, ECC, GPU temp/power, link counters, NCCL RAS, per-rank progress/stragglers | alert, isolate or controlled stop based on severity |
| FULL_DIAGNOSTIC | maintenance window or isolated allocated node | minutes/longer | DCGM diagnostics, memory/copy tests, full NCCL size matrix, inter-node fabric path, GPU-to-NIC topology | quarantine/requalify before return to production |
| INCIDENT_FORENSICS | after error | bounded, non-destructive first | driver/kernel logs, event timeline, CUDA/NCCL traces, dump hashes, process group, NIC counters | causal diagnosis, page/hold/drain as justified |

DCGM has tests ranging from short checks to stressful memory/compute workloads. A deep diagnostic can interfere with production jobs; use an **exclusive maintenance allocation**, not a background prolog. https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/dcgm-diagnostics.html ; https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html

### 10.2 Mandatory admission invariants

1. Inventory matches expected GPU count and GPU identifiers; cluster operator sets expected count from real allocation.
2. Each assigned GPU's driver interface responds; device permissions/isolation verified; no unexpected GPU in the job namespace.
3. Uncorrectable ECC and recovery-action-required states interpreted according to NVIDIA guidance for hardware/driver generation.
4. Required telemetry source reachable. **Unreadable logs are `skipped`/`unknown`, not 'no errors'.**
5. NVLink/NVSwitch and fabric configuration match the *declared* node topology, including Fabric Manager where applicable.
6. If network test is mandatory for a performance-sensitive job, run it within a dedicated authorized allocation and store raw output.
7. Diagnostics do not silently change device state, reset another workload's GPU or kill an unrelated job.
8. The final verdict is derived from explicit policy, not a last `echo "healthy"` statement.

### 10.3 Health JSON event contract (PROPOSED; confirm schema with Team 8)

```json
{
  "schema": "era-v5/g7-health/v1",
  "run_id": "optional-run-id",
  "attempt_id": "optional-attempt-id",
  "node": "host-from-runtime",
  "observed_at_utc": "ISO-8601-with-offset",
  "mode": "prolog",
  "gpu_inventory": {"expected": 8, "observed": null, "uuid_list": []},
  "checks": [
    {"name": "nvidia_smi", "status": "skipped", "required": true,
     "reason": "binary not found", "value": null, "unit": null,
     "evidence_id": "log-or-object-id"}
  ],
  "verdict": "block",
  "recommended_action": "do-not-admit",
  "policy_version": "g7-policy-v1",
  "schema_version": 1
}
```

**Semantics:** `pass` means an executed check met its explicit criterion; `warn` means abnormal but policy may admit under operator decision; `fail` means executed and violated criterion; `skipped` means no conclusion because tool/permission/hardware unavailable. An aggregation into `pass` from *only skipped checks* is incorrect. On a developer laptop, required GPU tests can be `skipped` with **environment-specific admission=not-applicable**, not a real GPU readiness pass.

**Make stdout exactly one JSON object** (or JSONL with separate documented events) and send human-readable diagnostics to stderr/log storage, so parsers never have to sift informational Bash `echo` statements. Version the schema; validate with JSON Schema and cases for absent tools, malformed NVML, duplicate GPU UUID, negative/NaN metrics and timeouts.

### 10.4 Metrics and thresholds

- GPU inventory, HBM total/free, GPU clocks/SM utilization, temperature, power draw/limit, throttle reason, ECC volatile and persistent counters, remapped rows/retired pages, PCIe correctable/replay, NVLink link state, GPU UUID.
- NIC/RDMA: adapter firmware, link layer, port state, rate, drops/retransmits, topology locality and selected interface; have a separate NIC fault record from GPU Xids.
- Platform: CPU memory pressure, pinned-host staging availability, NVMe free space and wear/status, NTP/clock health.
- Training: progress heartbeat, last `optimizer_step` and ledger ID, nonfinite loss/grad, per-rank step time and trailing percentile, GPU idle due to data loader.
- NCCL: error/hang/backpressure and per-collective timing from supported diagnostics; no claim based on synthetic chart curves.

**Do not invent numeric alert thresholds.** Collect clean baselines under known device/SKU/driver/clock/GPU-count/collective-size conditions. Use NVIDIA-defined hard failures where applicable; other alert thresholds should be based on baseline distribution and measurement error. Temperature alone does not establish failure: correlate thermal throttling, clocks, power and performance.

---

## 11. Xid, SXid, ECC and correct recovery

### 11.1 Interpret the source and the event

- **Xid**: NVIDIA GPU driver error code emitted in host/kernel log and sometimes surfaced by DCGM/NCCL telemetry. It can arise from application, driver, PCIe, memory or physical device problems.
- **SXid**: switch/fabric-related events for relevant NVSwitch-generation systems, collected separately.
- A code is a *diagnostic indication*, **not always root cause**. Correlate `GPU UUID`, PCI bus ID, timestamp, node, driver, workload/attempt and preceding events.
- Driver and GPU generation matter to remediation; read the current official Xid catalog and GPU debug guidelines rather than inheriting a homemade fixed list.

Primary: https://docs.nvidia.com/deploy/xid-errors/introduction.html ; https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html ; https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/diag-errors.html

### 11.2 Minimum casebook (illustrative triage, not automatic universal policy)

| Xid | NVIDIA classification / usual meaning | Responsible policy action |
|---:|---|---|
| 13 | Graphics-engine exception | investigate kernel/application and context; escalate on repeated/unexplained failures |
| 31 | GPU MMU/page fault, often illegal access by application | analyze app with Compute Sanitizer; not proof of hardware failure by itself |
| 43 | Channel reset / application-level fault; GPU may remain healthy | terminate/repair application and assess context, not automatic node replacement |
| 45 | Preemptive removal of channel, often consequence of abort/another error | correlate nearby primary Xid; do not mislabel as ECC row-remap |
| 48 | Double-bit ECC | serious HBM/data-path integrity event; use platform guidance for isolation/reset |
| 63/64 | Page retirement / row-remap recording, generation/context dependent | correlate ECC and remap state, follow policy for specified GPU |
| 74 | NVLink error | investigate P2P/network path; maintenance may be required |
| 79 | GPU fell off PCIe bus | stop allocating affected node, diagnose bus/power/GPU; don't assume a safe in-job checkpoint is still possible |
| 92 | Excessive single-bit ECC interrupts | NOT the teammate script's “thermal throttle” description |
| 94 | Contained GPU memory error | affected application restart; follow NVIDIA reset guidance; not equivalent to uncontained fault |
| 95 | Uncontained memory error | more disruptive reset/isolation as recommended by NVIDIA |
| 119/120 | GSP firmware RPC timeout/error | driver/firmware/hardware triage and recommended recovery |

Example from official catalog: Xid 43 commonly indicates a user application failure while GPU remains healthy. Xid 45 can be produced by application abort. Xid 94 is contained; Xid 95 is uncontained. These distinguish *application restart*, *GPU reset*, *node cordon* and *repair service*; they cannot all map to “force physical power-cycle.”

### 11.3 Robust Xid parser design

**Input sources:** `journalctl -k --since ...` if permitted, `/dev/kmsg` via privileged service, curated log file fixture for CI; fallback to `dmesg` or known host logging path with provenance. Use precise `NVRM: Xid (PCI:...): <number>` recognition and separately parse switch events. Do not use `grep -i xid` as a complete safety policy.

**Implementation method:**
1. Record source attempted, required permissions, first/last timestamp, watermark and whether log access succeeded.
2. Parse each event into node/PCI/device/code/text and preserve raw log line in restricted evidence storage.
3. Deduplicate recurring *same event* with persistent offsets or unique event ID; do not silently drop repeats representing distinct incidents.
4. Enrich by official catalog version (not arbitrary dictionary strings).
5. Correlate other events in small windows and select a *recommendation*, not unreviewable hardware action.
6. Return structured `pass|warn|fail|skipped` per check, with `check_performed=true/false`.
7. If logs are inaccessible, return `skipped`/`unknown`; if mandatory for this host, aggregate to admission block.
8. Avoid unconditional root `dmesg`; deploy a least-privilege host agent with audited log access.
9. Test Xid 13,31,43,45,48,63/64,74,79,92,94/95,119/120, unknown code, malformed input, unreadable logs and two independent GPUs.
10. Prove that the drain action does not affect healthy nodes or incorrectly kill another job.

### 11.4 Remediation state machine

`HEALTHY → SUSPECT → QUARANTINED → DIAGNOSING → REPAIRED/REPLACED → REQUALIFY → HEALTHY`.

Separate *application* and *hardware* states. A recoverable application MMU fault can fail the job while leaving the GPU eligible for other jobs. An uncontained HBM/PCIe event may require node isolation and vendor-specific reset/repair before reuse. The controller must record **who** authorized destructive actions and how a node re-enters service after retesting.

### 11.5 Silent data corruption is a separate challenge

Xids/DCGM cannot detect every silent numerical error. Complement with fixed canary microbatches, weight/optimizer checksums, repeated computation on trusted isolated devices for selected windows, anomaly alarms, and log-based replay. Do not interpret a normal Xid scan as evidence that all loss tensors and gradients are correct.

---

## 12. NCCL: what is computed, where bandwidth comes from, what a test proves

### 12.1 First-principles collectives

For N ranks and equal-size vectors of S bytes:

- `all_reduce`: every rank receives elementwise sum/average of every rank's vectors; used for gradient synchronization.
- `reduce_scatter`: reduce all vectors, then give each rank its assigned disjoint slice; common in sharded gradient synchronization.
- `all_gather`: every rank receives a concatenation of slices; used for replicated state reconstruction.
- `all_to_all`: every rank sends a distinct slice to every rank, common in expert routing.
- `broadcast`: one rank sends a value to others.

Data parallelism with DDP/ZeRO makes repeated collectives essential; GPU-to-GPU traffic and latency depend on message size, algorithm, topology and network. Do not assume that 900 GB/s advertised H100 NVLink equals per-rank application all-reduce bandwidth.

### 12.2 Algorithm bandwidth vs bus bandwidth

`nccl-tests` reports `algbw` (payload / operation time) and `busbw` (a model for fabric utilization). For ring all-reduce with N ranks and large messages, the conventional aggregate data-volume factor is `2*(N-1)/N`; the NCCL test's bus-bandwidth interpretation depends on the collective and is intended to support comparison with hardware link capabilities. Consult primary measurement method: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md.

A measurement line only has meaning together with: GPU architecture/count, local versus inter-node rank placement, PCI/NVSwitch/EFA topology, datatype, operation (`sum/avg`), buffer bytes, warm-up, repetitions, CUDA/NCCL versions, process/GPU mapping, concurrent load, and raw test output.

**Correctness and performance are different assertions**. A successful exit code should be accompanied by the test's actual out-of-bounds and wrong-result summary plus measured times. Performance baselines must be measured repeatedly and thresholds stratified by topology. NVIDIA DCGM's NCCL test plugin has its own skip/fail conditions and currently documents single-node operation; multi-node tests require separately launched workloads. Sources: https://github.com/NVIDIA/nccl-tests ; https://docs.nvidia.com/datacenter/dcgm/latest/reference/diagnostics/plugins/nccl-tests.html .

### 12.3 Layered test sequence

**A. CPU/Gloo correctness test — Week 1**  
Two or four local processes perform a deterministic all-reduce. Check expected sum on *all* processes. This proves Python process-group logic and distributed orchestration basics only; it is **not NCCL**, no NVLink, no EFA and no GPU fabric performance evidence.

**B. CUDA single-GPU process test**  
Initialize NCCL on one GPU and run nontrivial tensors. Exercises runtime and device allocation, not inter-GPU network.

**C. Single-node 8-GPU NCCL**  
Run `all_reduce_perf` with correct GPU count, e.g. in an exclusive allocation `all_reduce_perf -b 8M -e 512M -f 2 -g 8` using the binary's appropriate one-process/multithread mode. The exact rank/threads/GPU mode is part of the evidence; test MPI-driven one-GPU-per-process as a separate path if that matches production. Check correctness, out-of-bounds, bus/algo bandwidth, GPU clocks/topology and reproducibility.

**D. Multi-node NCCL**  
Launch ranks across actual two+ nodes using the production launch route. Confirm EFA/InfiniBand/RDMA selected transport, NCCL plugin loading, PCIe ACS/IOMMU/P2P, routes, port/firewall, NIC errors and bandwidth. Repeat with mixed message sizes, rank counts and topology placements.

**E. Training-like collective replay**  
Use sizes and ordering representative of the real DP/TP/EP/PP groups. Measure small-message latency and end-to-end step overlap; all-reduce peak on 512 MiB alone cannot predict MoE all-to-all or pipeline performance.

**F. Fault tests**  
One rank disappears; one node disappears; network throttled; corruption in test buffer; one process is stuck; ensure bounded watchdog and clean failure rather than infinite waiting. Check Slurm drains only the right node and GPU job resumes from a verified checkpoint.

### 12.4 NCCL's own RAS system

Recent NCCL documentation describes an integrated Reliability, Availability, Serviceability subsystem available since NCCL 2.24, with `ncclras` command/status monitoring and diagnostics. It can identify dead/unresponsive ranks and collect communicator health. This is often a better *live in-job diagnostic* than firing heavyweight collective benchmarks during training; the exact features depend on the deployed NCCL version. Official: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html .

**Important:** NCCL RAS diagnostics do **not** exercise all data paths or guarantee complete cluster health. Use them alongside real communication tests, GPU telemetry and application progress evidence. Avoid exposing a RAS port outside trusted network boundaries.

### 12.5 Most common NCCL pitfalls

Wrong NIC binding; mixed IB/EFA settings; incompatible `libnccl`/driver/plugin; invalid PCIe ACS/IOMMU for GPUDirect; bad peer connectivity; unconfigured Fabric Manager; wrong rank-to-GPU mapping; oversubscribed nodes; transient thermal throttling; cold/warm run discrepancies; failure of 512 MiB test due to memory bounds or GPU occupancy; NCCL watchdog timeout from application straggler instead of bad network; reusing old rendezvous state; benchmarking while another job runs; treating DCGM skip as NCCL pass.

**Diagnose methodically:** verify GPU mapping → check NIC and PCI topology → inspect NCCL RAS/logs → run clean NCCL tests → compare baseline at identical settings → isolate physical node or application-level cause. Don't randomly set many NCCL environment variables at once: change one documented variable and record effect.

---
## 13. Checkpoint contents: exact state, not just model weight files

### 13.1 State inventory at a clean optimizer boundary

An exact resume is a statement about the **entire state transition**, not just the stored parameter tensors. The required inventory is framework/config dependent; at minimum audit:

| Category | Substate | Why a loss-only canary can miss it |
|---|---|---|
| Model | all trainable parameters, nonpersistent/persistent buffers, frozen bases, quantizer parameters/scales, adapters, router parameters, tied-weight aliases, recurrent memory if part of state | probe loss may be similar despite wrong hidden state or some unused experts |
| Optimizer | per-param Adam/momentum moments, step counters, param-group hyperparameters, FP32 masters if present, weight decay semantics, weight mapping | first restored forward does not use optimizer moments; next update diverges |
| Scheduling | LR scheduler, warmup phase, decay step, gradient scaler, EMA/swa and associated counters | next step uses wrong LR/scaler |
| Randomness | Python RNG, NumPy RNG, PyTorch CPU generator, CUDA generators for every actual device/rank, named per-worker generators, custom kernels' RNG streams | fixed eval probe may be deterministic while next dropout differs |
| Data | corpus manifests, mixture/pool state, sampler permutations/seeds, worker cursor, packed sequence IDs, token/loss masks, consumed-token ledger, committed batch ID | same initial probe but different next data |
| Distributed | total/group world sizes, TP/PP/DP/CP/EP mapping, expert ownership, pipeline microbatch schedule, gradient accumulation counter, collective/topology contract | changed layout may load state but not reproduce exact reduction order |
| Runtime | code/git hash, image digest, library/driver/kernel version, env variables, deterministic settings, model config and tokenizer digest | silently changed kernel or tokenizer changes semantics |
| Recovery metadata | `run_id`, immutable `checkpoint_id`, `attempt_id`, `writer_epoch`, step/tokens, parent/fork/branch, last durable ledger entry, uploader receipt | permits split-brain writer or double-consumption |
| Diagnostics | checkpoint size/checksum/manifest, last complete step, selected fixed input IDs, reference logit/loss hash, gradient/optimizer probe | helps prove integrity and catch silent conversion errors |

**Important special case: in-progress accumulation.** If checkpoints only occur after *optimizer.step()* and all gradients are zeroed at a known boundary, omitting transient partial gradients is valid. If save can happen midway through accumulation, you must save accumulation counter, partial gradients and their averaging weights, or deliberately roll back to previous boundary and replay microbatches.

**Training model vs inference export:** a model-only safetensors/merged weights export can be excellent for evaluation/deployment but is typically **not** a full optimizer-resumable checkpoint. TorchTitan distinguishes model-only export from full training state: https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md .

### 13.2 What DCP does and does not guarantee

PyTorch Distributed Checkpoint (`torch.distributed.checkpoint`, DCP) supports parallel writes and load-time resharding of compatible distributed state. `get_state_dict` / `set_state_dict` are useful canonical parameter-identity APIs; the optimizer's state and current parallelism must be compatible. `async_save` stages a *training-safe copy* and writes asynchronously. Exact API options may differ by installed PyTorch release and must be pinned. Official: https://docs.pytorch.org/docs/main/distributed/checkpoint.html ; https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html .

DCP **is a serialization/resharding engine, not a complete business transaction manager**. It does not automatically commit the separate data ledger, protect the S3 latest pointer against split-brain writers, guarantee that every object in a multi-rank generation was uploaded durably and verified, or guarantee bitwise continuation with a changed rank count.

### 13.3 Integrity is layered

1. Tensor semantic integrity: known canonical FQNs, shape, dtype, aliases, frozen/trainable mask and validation against model config.
2. File integrity: exact list, byte lengths and cryptographic digests; DCP metadata included.
3. Distributed completeness: every **expected rank and shard** for the declared layout represented; no partial generation.
4. Writer coordination: unique attempt/epoch prevents old rank workers from publishing after newer run has taken ownership.
5. Durable storage: acknowledge only after full confirmed upload/verification into a failure domain independent of the lost node(s).
6. Atomic *visibility* of generation: immutable manifest and single-key conditional pointer publication, or equivalent transactional metadata service.
7. Semantic replay: restore all state and run independent continuation/probe tests.

**`os.replace` alone is insufficient**: while it can replace a same-filesystem name atomically, crash durability requires appropriate `fsync`, and a distributed checkpoint contains many files across ranks and storage systems. S3 individual object operations are strongly consistent, but writing a dozen S3 objects is not an atomic multi-object transaction. S3 conditional writes can prevent stale writers overwriting a pointer: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html ; https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html .

---

## 14. Proposed distributed checkpoint transaction protocol

### 14.1 Objects and identities

Use `run_id` for the logical training line, `branch_id` for resume/fork/reshard variant, `attempt_id` for a concrete Slurm allocation, `writer_epoch` (monotonic fencing token) for exclusive checkpoint ownership, `global_optimizer_step` and `consumed_ledger_seq`, and immutable `generation_id` for the checkpoint candidate.

Proposed directory/object layout:

```text
runs/<run_id>/branches/<branch_id>/
   generations/<step>_<writer_epoch>_<generation_uuid>/
      dcp/                  # one or more rank-written files + DCP metadata
      auxiliary/            # RNG, scheduler, scaler, sampler and ledger state
      receipts/             # one signed/validated receipt per expected writer
      manifest.json         # complete logical state declaration and hashes
   latest                   # one compact versioned pointer to a committed generation
   history/                 # append-only audit/transition events
```

### 14.2 The two-phase *application* commit pattern

This is a design proposal and **is not implemented or hardware-proven in G7 yet**:

1. **Fence old attempts.** Recovery coordinator assigns `writer_epoch`. No attempt with a lower epoch may publish a new committed generation. The durable metadata authority must enforce this, not only an in-process comparison.
2. **Reach a coherent step boundary.** Rank 0 broadcasts a save decision, then ranks participate in the same decision; avoid a one-rank-only collective. If a rank is unhealthy or nonfinite guard fires, abort save/handle failure safely.
3. **Stage safely.** Capture tensors/auxiliary state at step `k` in an immutable snapshot (e.g., async DCP stage). Prevent source mutations during stage completion and avoid staging OOM by allocating measured host RAM budget.
4. **Write immutable candidate.** Unique generation prefix; every rank writes its expected shards, metadata and auxiliary state. Do not mutate last committed generation.
5. **Collect rank receipts.** Coordinator waits for every expected participant and validates shard inventory, lengths, checksums, dtype/shape and the *same* `(run_id, step, ledger offset, config digest, writer_epoch)`.
6. **Verify storage.** Check the DCP index and every uploaded file/size/content digest at durable storage, not only local NVMe. Enforce explicit failure and retry timeout.
7. **Publish immutable `manifest.json`.** Only after all receipts and integrity validation succeed. Manifest points to exact immutable objects, has generation schema/version, data ledger cursor, environment digests, tensor keyspace info and proof material.
8. **Advance one atomic pointer.** Conditional write `latest` from previous known ETag/version to new committed manifest reference, rejecting races/outdated epochs. Record the authoritative committed step and data cursor in the same logical decision; if separate stores, use an idempotent protocol that reconciles partial confirmation.
9. **Acknowledge durable progress.** Only now may recovery controller claim step `k` is protected under the defined failure domain. Emit `checkpoint.committed` and append audit event exactly once per generation.
10. **Garbage collect with protection.** Retain previous verified generations, rollback copies and pinned eval/export checkpoints. Deletion must not race with readers or active commits.

The exact consistency semantics of S3, EBS, FSx and the metadata service differ; document where the **linearization/commit point** is, and how to recover if client receives an ambiguous timeout after an object-store `PUT` succeeded.

### 14.3 Manifest minimum fields

```json
{
  "schema": "era-v5/g7-checkpoint/v1",
  "run_id": "...", "branch_id": "...", "attempt_id": "...",
  "writer_epoch": 17, "generation_id": "...", "optimizer_step": 10000,
  "loss_bearing_tokens_committed": 0,
  "consumption_ledger": {"manifest_digest": "sha256:...", "next_batch_id": "...", "cursor": "..."},
  "parallelism": {"dp": 1, "tp": 1, "pp": 1, "ep": 1, "cp": 1,
    "world_size": 1, "rank_map_digest": "sha256:..."},
  "software": {"git_sha": "...", "container_digest": "sha256:...",
    "torch_version": "...", "cuda_version": "...", "nccl_version": "..."},
  "components": [
    {"logical_name": "model_and_optimizer", "backend": "dcp",
     "index": "dcp/.metadata", "object_manifest_sha256": "..."}
  ],
  "expected_rank_receipts": 1,
  "all_object_sizes_sha256": "sha256:...",
  "checks": {"probe_input_digest": "sha256:...", "logit_digest": "sha256:...",
    "loss_bits": "hex-bit-pattern", "optimizer_state_digest": "sha256:..."},
  "commit": {"state": "SEALED", "created_at_utc": "ISO8601"}
}
```

All values above are **schema placeholders**, not real capstone data. Loss bits should be stored for a scalar of defined dtype, signedness and byte order, not a rounded decimal string. Use hash data for proof, not as a replacement for numeric test execution. State should include enough detail to restore exact next data batch.

### 14.4 Fallback and strict load

Restore code should:

1. Read latest **committed** pointer; resolve immutable manifest and note writer epoch.
2. Verify the manifest and every listed object cryptographically, including metadata and expected-rank count.
3. Reject unexpected checkpoint schema/config/tokenizer/model keyspace before allocating huge tensors.
4. Instantiate correct model in safe manner; load DCP model/optimizer with required canonical state names, reshard only if explicitly approved, track mapping.
5. Restore scheduler, scaler, RNG for relevant ranks, sampler, data ledger and special states.
6. Validate frozen hashes, parameter mapping, selected fixed probe and all expected optimizer state.
7. Resume at *next* batch after durable consumed cursor; record read attempt and deterministic/resharded branch status.
8. On invalid newest generation, fall back only to an independently **verified previously committed generation**; do not choose directory by lexicographic max and do not trust local NVMe after physical node loss without independent availability.
9. If world size changed, explicitly classify as non-bitwise-continuation until stronger proof exists. A probe in eval mode may match yet the first gradient reduction may differ.

**No automatic “fix up weights by name”** during recovery; a conversion between architectures must be a separately versioned, deterministic transform with before/after key/digest and loss equivalence tests. LightningLM's silent key mismatch is the instructive failure mode.

### 14.5 Async checkpoint design pitfalls

- Snapshot tensor references that are mutated by subsequent optimizer steps create nondeterministic mixed-step checkpoint. Staging must finish or use appropriate copy-on-write semantics.
- Background upload can lag behind multiple checkpoint intervals; enforce one in-flight policy or bounded pipeline with correct generations; do not silently drop required committed save.
- Node crash after local save but before S3 publication makes local checkpoint unavailable; it is *not globally durable*.
- Object-store list-order is not checkpoint order. Durable `latest` is a controlled transaction pointer, not max filename.
- Too few retained checkpoints means silent corruption may remove only good recovery copy.
- Final checkpoint export to Teams 9/12 must be read-only and pinned to manifest ID, never a mutable staging prefix.
- Checkpoint compression/deduplication/delta encoding requires end-to-end digest and reconstruction testing; ratio is model-specific.
- Saving a very large all-trainable model every step without a measured incremental/delta design may overwhelm the storage system; compute throughput and concurrency budgets first.

---

## 15. What does bit-equal loss actually prove?

### 15.1 Four different guarantees (never conflate)

**G0 — Readable:** file can be opened and tensor shapes appear plausible. Weakest.  
**G1 — Identical fixed forward:** model on a fixed input in a specified eval-mode/precision/device yields identical logits/loss *bit patterns*. Catches many wrong weights and mask errors but **not missing optimizer state or data cursor**.  
**G2 — Identical full state at checkpoint:** all relevant serialized tensors, optimizer, RNG and ledger are equal according to defined canonical comparison.  
**G3 — Identical future continuation:** after new-process restore, every next batch ID/mask, LR, loss bit pattern, gradient digest and parameter/optimizer state is identical for M optimizer steps. This is the proper **Week-1 CPU acceptance target**.  
**G4 — Same bitwise trajectory after topology/hardware/library change:** much stronger and generally **not promised** without special deterministic distributed algorithms, homogeneous hardware, fixed reduction tree and thorough validation.

**Scientific implication:** Recomputing a `probe_batch_loss` after load is a *good guard*, not a proof that the run resumed exactly.

### 15.2 Rigorous Week-1 CPU experiment

Fix and persist: PyTorch version; deterministic CPU mode; model architecture and init seed; exact synthetic/real token IDs and masks; a small AdamW model; LR schedule; num threads, multiprocessing start method; sampler sequence and independent RNG streams.

**Reference A:** new Python process executes optimizer steps `0..K+M-1` continuously. After every step (or just before step, consistently) write a canonical record: `batch_id`, input/target hash, loss dtype/raw bits, LR raw bits, gradient norm raw bits, optimizer state digest, weights digest and RNG digest.

**Candidate B:** separate process starts same initial model, executes `0..K-1`, reaches *clean optimizer-step boundary*, saves full checkpoint, and terminates. A **third fresh process** loads checkpoint and executes `K..K+M-1`.

Compare every B continuation record against A for the corresponding optimizer step. **Equality must be exact** (e.g. `float32.view(uint32)` or `float64.view(uint64)`); `torch.allclose` with a tolerance is *not* a bit-equal test. Compare nested state dicts canonically, accounting for stable tensor name/order/dtype/byte order. A cryptographic digest is useful but store enough per-variable diagnostics to debug mismatches.

Run a separate G1 fixed-probe test using a known held-out batch before and after save, preserving probe model mode/RNG state. Probe must not mutate the training RNG or advance dataloader. If it does, restore those states or use explicit isolated generators.

**Test negative cases:** delete one optimizer moment; rename a model key; change one tensor dtype; change LR schedule index; advance the data cursor; alter one RNG stream; tamper with one shard; kill between write and manifest; kill just before/after pointer commit; simulate checkpoint upload timeout; submit a stale writer. Each must either fail deterministically or branch with a clear non-equivalence declaration.

### 15.3 Multi-rank and GPU extensions

- Run G3 on CPU with Gloo at one rank, then at four ranks, with stable distributed sequence and reduction ordering.
- Run on one CUDA GPU, then 8 GPUs and 2+ nodes with exactly pinned GPU/RNG/collective schedule and deterministic algorithm settings. PyTorch `torch.use_deterministic_algorithms(True)` is necessary for some operations but explicitly **not sufficient** for cross-version/platform reproducibility. CUBLAS workspace configuration may be required for certain CUDA kernels.
- GPU Xid or recovery tests should not be carried out while trusting unverified in-memory state. Restore from independently durable committed generation.
- For resharding 64→56 ranks, separately test **model tensor equivalence** and **continued statistical/numerical convergence**; do not falsely assert bitwise replay just because `DCP.load` succeeded.
- For MoE, exercise both active and rarely selected experts, shared experts, routing/auxiliary load-balancing state, expert optimizer updates and frozen quantized base hashes.

Primary reference: https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html and https://docs.pytorch.org/docs/main/distributed/checkpoint.html .

### 15.4 Particularly tricky replay mechanisms

- *Dropout/activation recomputation:* activation checkpointing and reversible layers may rerun stochastic forward operations; RNG handling must preserve intended semantics.
- *Gradient accumulation:* off-by-one in accumulation denominator, loss masks, or partially accumulated gradients changes update.
- *Asynchronous dataloaders:* worker prefetch queues depend on scheduling; checkpoint explicit *consumption sequence* or deterministic counter-based access. Prefetched does not mean consumed.
- *MoE routing:* sparse expert choices may be sensitive to FP tie-breaking and device/collective order; random load balancing/residual ties need explicit handling.
- *Mixed precision:* TF32/BF16/FP8 kernel choice, FP32 optimizer masters, stochastic rounding, scaler and fused kernels can change behavior.
- *Dynamic shapes or compiler kernels:* compilation/cache state and autotuner choices may be nondeterministic or affect low-order bits; pin relevant settings.
- *NCCL reductions:* nonassociative FP operations and changed topology can change results without any software bug.
- *CUDA graphs:* address reuse and captured RNG/collectives have state implications; test the exact production configuration.
- *Evaluation call:* a fixed probe might consume RNG or change a module from train to eval without restoring mode; isolate probe execution.

**Reporting:** “bit-equal on tested CPU model and fixed layout” is a valid finding. “Bit-equal on all GPU clusters at all sizes” is not supported by local tests.

---

## 16. No lost committed work, exact replay, and literal zero recomputation

### 16.1 Definitions with one timeline

```
step 1000 durable checkpoint A (acknowledged)
       1001 ... 1019 complete optimizer updates in volatile memory
step 1020 begins, physical node fails
latest durable state = 1000
```

- **No lost durable/committed progress:** checkpoint A remains retrievable and can be resumed. Nothing committed is lost; steps 1001–1019 can require replay.
- **Exact replay:** run the same steps 1001–1019 again from A, same inputs and environment, get bit-identical post-step states (within a declared deterministic contract). **This deliberately recomputes.**
- **Zero recomputation:** restart at step 1020 without redoing any already completed step. Requires durable per-step state or an equivalent continuously maintained redundant execution/state service that survives the failure.

Thus periodic checkpointing + replay can meet the first two but **cannot meet all three under arbitrary abrupt loss of volatile memory**.

### 16.2 Options by guarantee and overhead

| Strategy | RPO / replay | Operational price | Scientific exactness |
|---|---|---|---|
| Periodic full checkpoints | Up to last interval rollback | storage bandwidth, stage stall, failure rollback | exact replay possible on fixed deterministic stack |
| Full checkpoints + input/decision log | Same rollback, deterministic inputs easier | durable ledger writes, audit service | improves replay but **log alone does not recover unpersisted weights/optimizer state** |
| Per-step durable full state | durable step RPO≈0 after ack | potentially enormous sustained IO, frequent synchronization | exact if complete state and atomic commit |
| Full checkpoint + verified incremental/delta per optimizer step | durable step RPO≈0 after ack | delta tracking, chain integrity, compaction, write amplification | requires mathematically sufficient deltas, correct optimizer + RNG + ledger reconstruction |
| Replicated hot standby with synchronous state updates | potentially RPO≈0 for specified single-failure domain after ack | duplicate hardware/communication and replica validation; still need remote durability | must prove no split brain and committed state parity |
| Dual execution with deterministic compare | may detect silent corruption | near-duplicate compute, extra network/coordination | not a zero-overhead fault cure |
| Periodic checkpoints + journal + hot replica | broad protection under multiple faults | highest implementation complexity, fencing/commit coordination | only claims validated for enumerated fault matrix |

**The key subtlety:** an optimizer updates a potentially very large state, with moments and possibly master weights. A “small log of next batch IDs and random seeds” is not a substitute for preserving the updated model/optimizer state if you refuse to recompute. For Adam, reconstructing exact new `m,v,weights` from a previous checkpoint requires either the full update information and deterministic application, which itself is computation, or a stored new state/delta. In a large sparse MoE, only selected expert state may change each step, but dense backbone states still change; the delta size and bandwidth must be **measured**.

### 16.3 Combined guarantee requires transaction definitions

To pursue **no lost committed steps + exact replay + zero recomputation**, define all these explicitly:

1. Step `k` is only *acknowledged committed* to the training orchestrator after sufficient state/delta replicas and ledger cursor are durable under the chosen failure model.
2. Process and scheduler attempt are **fenced** so an old or partitioned attempt cannot commit after failover.
3. Reconstructed state on recovery is tensor-identical, including optimizer and per-rank auxiliary state.
4. No batch is logically consumed twice: consumption ledger transaction is coordinated with model commit.
5. Cold restore path works without a hot replica; hot replica lag and failure paths are measured.
6. Multi-node consensus semantics (if used) survive acknowledged-writer crash and timeouts without two 'latest' authorities.
7. Training is allowed to acknowledge future optimizer steps only if replication durability has met policy, or speculative work is honestly reported as uncommitted/replayable.

**You cannot generally demand zero recomputation of uncommitted in-flight work after the only copy is destroyed.** Even per-step persistence will not protect computation currently in a forward/backward pass unless its intermediary state is also durable or replicated. Define zero recomputation as *completed and acknowledged optimizer steps*, not every partially executed floating-point operation.

### 16.4 Honest overhead accounting

At an assumed stage-specific checkpoint size `S` and step time `t`, durable full-state saving on every step requires at least `S/t` **useful sustained checkpoint throughput**, before replication and storage overhead. The required numeric bandwidth remains unknown until `S` and `t` are measured for the trainer-approved stage. This is an **analytical relation**, not an ERA V5 observed rate. Multiply bytes by replica count where copies cross separate storage systems; count network traffic, CPU hashing/staging RAM and checkpoint-consensus latency. A 20% delta ratio would be an *assumption*, not a measured fact. Without changed-state measurements do not propose delta compression as a proven solution.

**Recommendation (engineering judgement):**
- Gate-1 contract: verified durable committed checkpoints and exact separate-process CPU replay.
- Week-2/3: async DCP, independent durable pointer, GPU restore, measured checkpoint stall/throughput and bounded error recovery.
- Scale-sensitive advanced research: instrument changed optimizer-state bytes, benchmark periodic vs incremental/per-step vs hot standby, model failures and compute/cost goodput. Adopt zero-recompute scheme **only if it meets your RPO domain and measured economic target**.

---

## 17. Fault injection and recovery design, not just tests of successful save

A serious fault-injection programme perturbs the system at the **boundaries** where claims can become false. Each test must save preconditions, fault timing, expected invariant, actual output, and cleanup.

| ID | Injection | What must be proven | Stage |
|---|---|---|---|
| F01 | kill CPU training process immediately before snapshot | previous durable generation remains valid | Week 1 |
| F02 | kill during local candidate write | incomplete candidate not selected | Week 1 |
| F03 | kill after shard files but before manifest | `latest` unchanged; previous checkpoint usable | Week 1 |
| F04 | alter one checkpoint byte | hash verification aborts/fallback | Week 1 |
| F05 | remove one shard file | missing-rank/object detected | Week 1 |
| F06 | add an unexpected key or change tensor dtype | strict semantic validation rejects | Week 1 |
| F07 | change RNG only | G3 continuation mismatch detected | Week 1 |
| F08 | change sampler/ledger cursor only | next batch ID mismatch detected | Week 1 |
| F09 | change optimizer moments only | next optimizer update differs despite same fixed forward | Week 1 |
| F10 | two writers concurrently updating same `latest` | stale writer rejected/fenced | Week 1 conceptual, later S3 test |
| F11 | missing `nvidia-smi`, DCGM, `dmesg` | `skipped`/blocked appropriately; no false PASS | Week 1 |
| F12 | sample Xid 43 vs Xid 79 in fixture | application fault vs device fault policy differs | Week 1 |
| F13 | partial Slurm account credentials | unauthorized job denied | Week 1 |
| F14 | invalid QoS or quota breach | policy enforced, not merely logged | Week 1 |
| F15 | process exits 75 | explicit Slurm requeue with incremented attempt count | Week 1 |
| F16 | process exits 1 | held/no crash-loop | Week 1 |
| F17 | signal only batch shell | verify forwarding to coordinator/workers | Week 1–2 |
| F18 | GPU temperature/power skew | telemetry captures straggler; no invented cause | Week 2 |
| F19 | NCCL buffer correctness failure | blocking negative result, raw failed data summary | Week 2 |
| F20 | NCCL peer hang/timeout | bounded group termination, no infinite barrier | Week 2 |
| F21 | kill one GPU rank while others progress | group failure and new attempt; no mixed-step commit | Week 3–4 |
| F22 | remove whole node | requeue after healthy node replacement; full state restored | Week 3–4 |
| F23 | object-store `PUT` ambiguous timeout | reconcile idempotently and prove `latest` pointer correctness | Week 3 |
| F24 | reduce world size and reshard | restored tensor state valid; non-bitwise continuation labeled | Week 3–4 |
| F25 | corrupt newest generation | select prior proven generation only | Week 1 local / Week 3 GPU |
| F26 | data ingest slow or shard unavailable | no consumed-ledger commit for batch never used | Week 3 |
| F27 | network partition isolates old writer | fencing prevents split-brain commit | Week 4 |
| F28 | controller/slurmdbd restart during training | job state accounted/recovered | Week 3–4 |
| F29 | cluster reboot, all local NVMe lost | independent S3/remote durable generation usable | Week 4 |
| F30 | replay using changed kernel/image | deterministic contract detects version mismatch; only explicit new branch proceeds | Week 2–4 |

**For each test:** label either `PASS`, `FAIL`, `SKIPPED` or `NOT_RUN` and include evidence ID. `SKIPPED` cannot count as a pass. Timeouts and kill injection need safe, scoped isolated test jobs; don't perform disruptive faults on a shared production training node without explicit operator authorization and change window.

---
## 18. Performance engineering, measurement discipline and goodput

### 18.1 Define the measurement denominator

Three useful, noninterchangeable rates:

1. **Instantaneous training tokens/s:** tokens processed during healthy forward/backward/optimizer steps per elapsed compute time. State whether includes prompt tokens, masked tokens, padding, packing and gradients.
2. **Job-wall tokens/s:** unique loss-bearing tokens processed divided by wall clock from first launch to final save, including save/prefetch/launch failures and retries.
3. **Paid-accelerator goodput:** unique loss-bearing tokens durably accepted divided by sum of accelerator reservation time; includes idle allocations and failed/replayed GPU time.

If a run processes 1B tokens, spends 10h on successful execution, 2h on recovery/replay, and occupies 64 GPUs for all 12h (illustrative), then successful-step rate is `1B/10h`, job-wall rate `1B/12h`, and paid throughput is `1B/(64*12h)`. Do not quote them interchangeably. Treat replayed batches as *spent GPU time but not additional useful tokens*, lest an unreliable run falsely appear faster.

### 18.2 Useful reliability metrics

- Availability fraction, useful-step efficiency, scheduled time vs actual busy time.
- GPU effective utilization *and* MFU. MFU depends on a defined arithmetic estimator and peak FLOP/s denominator; meaningful only with compatible dtype, hardware and FLOP count. Session 10 CPU results did not establish GPU MFU.
- Mean/median/P95 checkpoint stage stall, local write, upload, manifest verification, restore fetch, reconstruction and first resumed useful step.
- RPO in **optimizer steps and committed loss-bearing tokens**; RTO in elapsed seconds, separately per fault class.
- Requeues per 1000 GPU-hours, hardware failures per node-hour, `DRAIN` false positive rate, Xid incidents by GPU/driver version.
- NCCL allreduce latency/bus bandwidth by `(collective size, gpu_count, rank placement, topology, dtype)`; collectives appearing on critical path in profiler.
- Queuing: mean/P95 wait by account, preemptions per experiment, GPU-hours per team, idle-but-billed fraction.
- Storage economics: bytes checkpointed per trained weight, S3 request/GB-month/retrieval, snapshots retained, network egress/cross-AZ and FSx throughput/cost; state specific vendor pricing date if using current rates.

### 18.3 Checkpoint frequency and goodput modeling

Let `T` be checkpoint interval, `C` be effective training blocking cost per successful save, `M` be measured MTBF of the specific distributed job/failure domain, `R` be measured restore/restart time. A simplified efficiency penalty is approximately:

`checkpoint overhead ≈ C/T`, `expected lost compute after failure ≈ T/(2M)` and `recovery overhead ≈ R/M`.

So `1 - overhead ≈ 1 - C/T - T/(2M) - R/M` for **small independent overheads**, ignoring queueing, uploads, overlapping saves, correlated failures and all nonstationarity. An optimum near `sqrt(2*C*M)` follows if the model assumptions hold and differentiability/in-flight-save constraints allow it. In asynchronous DCP, **acknowledged durability lag** may exceed staging stall. A 10-second stage plus 5-minute background upload does not make the latest step globally durable after 10 seconds. Therefore plot both `blocking_cost(T)` and `checkpoint_commit_lag(T)`.

**Example only:** if blocked staging C=30 seconds and effective job MTBF M=12 hours, the simple interval estimate is `sqrt(2*30*43200)=1609.97 s≈26.8 minutes`. This is a modeling example; do **not** say it is measured or optimal for ERA V5. If upload takes longer than interval, pipeline concurrency/backpressure changes the model.

### 18.4 Benchmark protocol before comparing designs

1. Fix the same model, dataset, tokenizer, effective batch, sequence length, optimizer, precision, parallel groups, GPU count, image and driver.
2. Warm up kernels/compilers and data cache according to a fixed documented schedule.
3. Measure step time and losses for enough steps to capture variability and exclude startup bias; report quantiles and confidence if possible.
4. For checkpoint modes, vary interval and storage tier; log stage stall, durable commit latency and recovery read time.
5. For scheduling, compare capacity packing, queue delays, preemption and main-run effective time, not only per-job raw speed.
6. For fault tolerance, inject the **same faults** at the same logical steps for competing implementations.
7. Repeat with at least two seeds/placements where resources permit, and report both per-run results and aggregate.
8. Keep evaluation independent; a changed dropout/randomness/rank-map is a distinct experiment, not a clean design A/B.
9. Stop and diagnose any data-identity or restore mismatch before optimizing throughput.
10. Publish raw command outputs, timing JSON, software/hardware versions, node count and profiler traces, including failures.

### 18.5 Research reference comparison (do not copy numbers as targets)

| Reference | Published scale/evidence | Transferable engineering lesson | Not implied |
|---|---|---|---|
| Meta Llama 3 (2024) | 405B dense; 16K GPU training; 466 interruptions in 54-day window and >90% effective training time | fault categorization, operations and verification need first-class engineering | same failure rate, GPU goodput or tolerance on ERA V5 |
| DeepSeek V3 (2024/25) | 671B stored sparse MoE, 37B activated, 14.8T training tokens, report 2.788M H800 GPU-hours | active/stored parameter accounting and cross-rank communication overlap | Reported GPU-hours belong to a different architecture/hardware; do not extrapolate to V5 |
| LightningLM (2026 source) | 118.67B reported saved MoE, approximately 5.93B active, single 8-GPU node using quantized frozen base/adapters | checkpoints, loss continuity, function-preserving growth, one-node memory trade-offs | full-precision training of 118B with Adam states on an ordinary 8×80GB node |
| PyTorch TorchTitan | public multi-parallelism trainer with distributed checkpointing and observability | reference-tested engineering patterns, compare configurations | compatible without inspecting version, model and state layout |
| NVIDIA NCCL/DCGM | tool and diagnostic standards; not a universal cluster speed score | benchmark both correctness and speed, vendor-recommended error triage | one arbitrary absolute `busbw` floor for all topologies |

References: https://ai.meta.com/research/publications/the-llama-3-herd-of-models ; https://arxiv.org/abs/2412.19437 ; https://lightninglm.theschoolofai.in ; https://github.com/pytorch/torchtitan .

---

## 19. Monitoring, dashboards, events, security and human operations

### 19.1 Operational event schema

Define common immutable identifiers and at least five event classes:

- `job.admitted`, `job.started`, `job.preempt_requested`, `job.requeued`, `job.failed`, `job.finished`.
- `health.check_completed`, `health.xid_observed`, `health.node_quarantined`, `health.node_requalified`.
- `checkpoint.stage_started`, `checkpoint.candidate_complete`, `checkpoint.committed`, `checkpoint.failed`, `checkpoint.restore_verified`.
- `training.step_committed`, `training.nonfinite`, `training.replay_started`, `training.replay_equivalence_failed`.
- `nccl.collective_failed`, `nccl.rank_unresponsive`, `nccl.baseline_outlier`.

Use `schema_version`, UTC timestamp, run/branch/attempt/job/node/GPU UUID, step/ledger cursor, severity, failure category, evidence URI/hash and redacted action. Explicitly distinguish **notification** from **automated action**, and an error event from its verified root cause.

### 19.2 Dashboard truth policy

A dashboard may have a simulated `Xid 79` button as a useful demo, but place `SYNTHETIC` prominently and source synthetic fixtures independently from production metrics. In a real dashboard:

- Link any green PASS to a saved JSON evidence record containing command, host and observed time.
- Link any throughput chart to raw NCCL logs and matching GPU+NCCL version.
- Show `UNKNOWN`/`SKIPPED` explicitly rather than silently omitting it.
- Expose `last_committed_step` separately from `latest_in_memory_step` and `latest_checkpoint_candidate_step`.
- Show RPO/replayable-work exposure and checkpoint upload lag, not just the latest filename.
- Keep checkpoint state as one canonical namespace; do not invent dashboard-side data.
- Correct the course timetable and label Week-1 CPU tests separately from Week-2 GPU tests.

### 19.3 Incident response runbook sequence

1. Detect: loss/step time, health event, scheduler error or NCCL RAS event.
2. Observe: freeze incident timeline, collect stack/config/topology IDs and representative logs. Avoid starting destructive DCGM tests while production shares devices.
3. Decide: application-only restart, node cordon/replace, reconfigure network, or stop and escalate.
4. Fence: prevent old attempts from writing new checkpoint pointers or consuming new data as acknowledged.
5. Select last verified generation: confirm manifest, hashes, environment and ledger cursor.
6. Restore: instantiate compatible layout, verify all tensors/auxiliary states and run loss probe.
7. Continue: run next exact logical data record; monitor first N steps and compare against deterministic expected output if available.
8. Report: RPO, RTO, replayed tokens, failed components, cost, root cause with confidence and follow-up remediation.
9. Requalify failed hardware with dedicated diagnostics and operator authorization before returning to normal queue.

### 19.4 Security and blast-radius controls

- Never commit AWS credentials, STS temporary keys, presigned S3 URLs, private datasets, MUNGE credentials, service-token JSON or model IP to public Git.
- `gitleaks detect --source .` as required by instructor; also use pre-commit/CI secret scanning and redact structured logs.
- Use cloud roles/identities scoped by account and path; read-only evaluation teams must not be able to overwrite training checkpoint generations or `latest`.
- Encrypt checkpoint state in transit and at rest using organization-approved primitives; separate metadata logs with minimal personal information.
- Version and pin container images; verify base dependencies, CUDA/NCCL versions, hashes and package origin.
- Restrict Slurm prolog and diagnostic scripts, especially privileged `dmesg`, `scontrol update state=drain`, GPU reset and DCGM diagnostics, to the operator boundary.
- No shell injection through untrusted Slurm job name/config/hostnames; validate paths and arguments, avoid `eval`.
- Retain a tamper-evident audit trail for checkpoint pointer changes and recovery branches; control deletion and retention policy.
- `torch.load` pickled checkpoint data can be unsafe if untrusted. Favor DCP-supported trusted state models, safe formats and approved loading rules; verify serializer/API compatibility rather than permitting arbitrary pickle execution.

---

## 20. A four-week, test-driven programme with explicit gates

### Week 1 — through Saturday 31 Oct: DESIGN + LOCAL PROOF

**Queue:** accepted ADR with platform alternatives, Slurm account hierarchy and access policy; Docker/Slurm demonstration with **two real Slurm accounts**, visible limits and failure tests.  
**Health:** versioned JSON schema aligned with Team 8 (format expected 29 Oct), no-GPU/permission-skip fixtures, mocked Xids including 31,43,45,79,94,95; CPU Gloo path only honestly labelled. `launch.sh` shell/static/process tests.  
**Checkpoint:** state inventory and commit ADR, trusted CPU checkpoint reference with `G3` bit-equal uninterrupted vs newly loaded process, corrupted and interrupted generation tests, strict loading errors; confirm Team 5 config v0 interface expected 27 Oct.  
**Security:** gitleaks zero findings evidence; all work PR-reviewed; do not publish data pack or credentials.

**What is not yet proven:** real NVIDIA GPUs, real NCCL, H100/Blackwell specs on the allocated node, cloud S3 commit behavior, multi-node DCP, live restart.

### Week 2 — 1–14 Nov: REAL NODE INTEGRATION

- Observe actual host/GPU/NIC/Slurm version and replace proposed topology with measured inventory.
- Validate nvidia-smi UUIDs, DCGM, Fabric Manager, GPU logs and interconnect health; choose vendor-recommended remediation.
- Run dedicated GPU/NCCL allreduce/point-to-point/alltoall baselines and record raw performance across size/placement configurations.
- Integrate Slurm allocation and launcher: 1 agent per node + expected workers/ranks; test restart after job kill on available node(s).
- Implement full-state sharded DCP and immutable commit on the actual storage; test env mismatch/strict restore, stage/upload/fetch timing.
- Restore initial model with exact experiment state. Mark any world-size change non-bitwise unless tested and proven otherwise.
- Draft `parallel_layout_v0.md` with Team 5 config and Team 6 kernel/runtime measurements; no hypothetical benchmark becomes final.

### Week 3 — 15–21 Nov: END-TO-END DRY RUN 1

- Freeze a tested image, code, data, tokenizer, model config and approved rank layout.
- Test queue fairness under concurrent evaluated/benchmark jobs without disrupting the main training allocation.
- Measure full training goodput, actual checkpoint durability lag, RPO, RTO, and restart behavior.
- Inject rank death/network lag within an isolated dry run; correlate Slurm, GPU, NCCL, training, checkpoint and ledger events.
- Validate read-only checkpoint consumer flow for evaluation/quantization and checkpoint retention/pinning.
- Publish `docs/runbook.md`, owner rota, escalation contacts and operator permissions; reject false PASS if tests skipped.

### Week 4 — 22–28 Nov: DRY RUN 2 + GO/NO-GO

- Execute approved controlled fault scenarios (one rank loss and one host drop), not uncontrolled destructive testing.
- Validate automatic requeue, fencing, immutable checkpoint selection, full state load, exact data position and scientific continuation invariants.
- Run vendor-approved deep diagnostics (`dcgmi diag` etc.) in a maintenance allocation and NCCL full suite.
- Confirm storage reserve, checkpoint retention, on-call, budget, observability, security, rollback/cold-start drill.
- Present evidence with Pass/Fail/Not run/Blocked for each acceptance criterion and signed instructor approval before training launch.

### 20.1 Week-1 acceptance checklist template

```
[ ] ADR queue design reviewed and approved by TSAI
[ ] Slurm two-account `sacctmgr` evidence with distinct policies
[ ] Unauthorized account/limit negative test PASS
[ ] Documented `RequeueExit` test PASS (if required by design)
[ ] Preemption signal test PASS (if enabled)
[ ] Health JSON against schema PASS; missing tools SKIPPED not PASS
[ ] Xid fixture tests demonstrate appropriate classifications
[ ] CPU Gloo collective result correct, explicitly not NCCL
[ ] Checkpoint full state on CPU + new process exact multi-step replay PASS
[ ] Corrupt/partial/stale generation refuses and falls back safely
[ ] Code + config hash + dataset ledger integration documented
[ ] Team 5/8 dependencies acknowledged or marked BLOCKED
[ ] gitleaks executed; zero findings
[ ] GitHub PR / logs / command versions / machine / timestamp documented
```

**Pass/fail integrity:** Local execution evidence is separate from review acceptance. An instructor can reject a correctly executed demo if it does not satisfy the published contract. No test is “passed” just because a script exists in a PR.

---

## 21. The 20 highest-risk pitfalls: root cause and avoidable control

| Pitfall | Why it happens | Countermeasure |
|---|---|---|
| 1. `srun torchrun` 64 workers for 8 GPUs | `--ntasks-per-node=8` times `torchrun -nproc=8` | one Torchrun agent per node; assert per-GPU UUID/rank uniqueness |
| 2. Exit 75 never triggers requeue | return code is not Slurm policy | configure `RequeueExit`, verify with live dummy job |
| 3. Time-limit `USR1` assumed preemption notification | different Slurm mechanics | separate preemption/grace configuration and signal testing |
| 4. `nvidia-smi` works, declared healthy | driver visibility ≠ full health | inventory+ECC+fault policy+fabric tests |
| 5. `dmesg` permission error counted as no Xids | empty list on exception | mark `skipped`, require host agent or block |
| 6. Xid 31/43 incorrectly drains hardware | simplistic all-critical table | use NVIDIA catalog, correlate workload |
| 7. Xid 45 or 92 mislabeled | copying inaccurate dictionary | pin catalog and fixture tests |
| 8. GPU check prints prose instead of JSON | shell scripts easiest for demos | schema-first JSON output + tests |
| 9. NCCL simulated throughput called measured | dashboard fixtures disconnected from tools | raw logs, hardware IDs and measured plot only |
| 10. Huge NCCL benchmark in every prolog | confuses readiness with stress test | light prolog; full isolated diagnostic allocation |
| 11. `torch.get_rng_state_all()` invoked | nonexistent API | CPU `torch.get_rng_state`, CUDA `torch.cuda.get_rng_state_all` as applicable |
| 12. Loss probe reported as bit-equal training | optimizer/data state not compared | multi-step fresh-process continuation + full state digest |
| 13. Checkpoint selected from highest step directory | incomplete uploads mistaken latest | immutable manifest + fenced committed pointer |
| 14. Multiple rank writers, no barrier/receipts | partial distributed save accepted | expected writer set + all-rank verification |
| 15. `os.replace` called durable S3 commit | different filesystems/transactions | fsync local, S3 object checks, conditional latest |
| 16. Async staging races with optimizer | mutable live tensors being serialized | training-safe copy, completion and memory bound |
| 17. Resume different data batch | dataloader prefetch vs consumed differs | authoritative consumption ledger cursor in checkpoint |
| 18. World-size change called bitwise exact | FP reduction order/layout changed | explicit reshard branch; separate equivalence criteria |
| 19. 14 bytes/parameter applied to sparse frozen MoE | confusing total, trainable and active | separate `P_total`, `P_train`, `P_active` accounting |
| 20. A numeric V5 model/hardware design approved before G5 stage spec and node topology known | extrapolating paper throughput/spec | conditional budget + real operator specs + bench |

---

## 22. Questions requiring TSAI Lab / operator approval

These must remain **unknown** until answered. Do not design as though answers exist.

1. What is the actual first shared GPU node SKU, count, HBM, OS, Slurm version, image/driver/NCCL version and network layout?
2. Is G7 allowed to operate Slurm controller (and DB), or only write job/scripts for an instructor-owned cluster?
3. Does “two accounts” mean `mainrun` and `teams` or names defined elsewhere in the repo? What limits and fairness are acceptance-tested?
4. Which failures does “no lost work” cover: one application crash, rank, GPU, node, rack, AZ, storage service? Are **replayed** steps acceptable?
5. Are RPO=0 / zero recomputation research goals, or part of Gate-1 acceptance? At what acknowledgment boundary?
6. Does Week-1 bit-equal mean same-batch forward only or an M-step CPU fresh-process continuation? We recommend the latter.
7. Which library owns training state: DeepSpeed ZeRO, native FSDP2/TorchTitan, Megatron-Core, custom? What are the final layouts and model/optimizer states?
8. How are consumption-ledger entries and batch IDs exposed from data teams, and who defines atomicity with checkpoint commit?
9. Does the cluster provide S3/FSx shared storage, NVMe-only scratch, IAM roles and immutable checkpoint permissions? Where is durable storage geographically?
10. What is Team 8's health event JSON schema/version and action authority? Is G7 allowed to automatically DRAIN a physical node?
11. What are approved `dcgmi diag`/GPU reset maintenance windows and which operator owns destructive actions?
12. How will evaluator/quantization teams select pinned checkpoints, and what is the retention/Glacier restore latency policy?
13. What retry/hold/alert limits are acceptable for crashes that occur repeatedly after a save?
14. What cloud budget and GPU availability constrain the main-run and spare-node strategy? Are spot instances permitted?
15. Who signs Gate-1 design decisions and then Gate-2 hardware configuration? Which member owns each PR and evidence bundle?

---

## 23. Annotated bibliography and reading sequence

**Reading order:** A newcomer should read (1) Slurm Concepts and QoS, (2) Torchrun launch semantics, (3) NVIDIA DCGM/Xid and NCCL tests, (4) DCP and TorchTitan, (5) S3 conditional commit, (6) Llama 3 and LightningLM operational evidence, and (7) cloud platform alternatives. Every reference is an actual public URL; none is an invented GitHub repo link for G7.

### 23.1 Cluster scheduling and job execution (official)

1. Slurm `sbatch`: https://slurm.schedmd.com/sbatch.html — batch execution, `--signal`, `--requeue`.
2. Slurm config: https://slurm.schedmd.com/slurm.conf.html — `RequeueExit`, controller and queue properties.
3. Slurm QoS: https://slurm.schedmd.com/qos.html — priorities, preemption and limits.
4. Slurm fair-tree: https://slurm.schedmd.com/fair_tree.html — hierarchical accounts and fair-share.
5. Slurm multifactor priority: https://slurm.schedmd.com/priority_multifactor.html — priority weights.
6. Slurm GRES: https://slurm.schedmd.com/gres.html — GPU scheduling/accounting.
7. Slurm cgroups: https://slurm.schedmd.com/cgroups.html — enforce resource/device allocation.
8. PyTorch TorchTitan: https://github.com/pytorch/torchtitan — distributed training, tested Slurm launcher example.
9. PyTorch torchrun: https://docs.pytorch.org/docs/stable/elastic/run.html — worker/agent process semantics.

### 23.2 Platform implementation options

10. AWS ParallelCluster scheduling: https://docs.aws.amazon.com/parallelcluster/latest/ug/Scheduling-v3.html — queue provisioning/config.
10a. AWS Parallel Computing Service: https://docs.aws.amazon.com/pcs/latest/userguide/getting-started.html — AWS-managed Slurm clusters and multi-node training jobs.
11. AWS ParallelCluster EFA: https://docs.aws.amazon.com/parallelcluster/latest/ug/efa-v3.html — fabric setup and supported instance validation.
12. AWS EFA+FSx Lustre tutorial: https://docs.aws.amazon.com/parallelcluster/latest/ug/tutorial-efa-enabled-fsx-lustre.html — actual shared storage/fabric integration.
13. AWS HyperPod auto-recovery: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency-slurm-auto-resume.html — managed node health/retry semantics.
14. Azure CycleCloud Slurm Workspace: https://learn.microsoft.com/en-us/azure/cyclecloud/overview-ccws — packaged Slurm environment.
15. Azure CycleCloud planning: https://learn.microsoft.com/en-us/azure/cyclecloud/concepts/plan-and-size-hpc-clusters — custom scheduler and topology decision.
16. Google Cluster Toolkit: https://docs.cloud.google.com/cluster-toolkit/docs — IaC for HPC/AI training.
17. Google Slurm quickstart: https://docs.cloud.google.com/cluster-toolkit/docs/quickstarts/slurm-cluster — practical cloud setup.
18. NVIDIA Base Command Manager: https://docs.nvidia.com/base-command-manager/product-manuals/admin/workloadmanagement — integrated WLM management.

### 23.3 GPU hardware, failure policy, collectives and troubleshooting

19. NVIDIA Xid introduction: https://docs.nvidia.com/deploy/xid-errors/introduction.html — interpretation and limitations.
20. NVIDIA Xid catalog: https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html — code-specific recovery recommendations.
21. NVIDIA DCGM diagnostics: https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html — workload stress and diagnostics.
22. NVIDIA DCGM diagnostic errors: https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/diag-errors.html — categorized severity and actions.
23. NVIDIA NCCL tests: https://github.com/NVIDIA/nccl-tests — executable collective benchmark arguments.
24. NCCL performance interpretation: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md — algbw/busbw formulas.
25. NVIDIA NCCL RAS: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html — distributed hangs and live-process checks.
26. NVIDIA DCGM NCCL plugin: https://docs.nvidia.com/datacenter/dcgm/latest/reference/diagnostics/plugins/nccl-tests.html — plugin prerequisites and pass/fail cases.

### 23.4 Checkpoints, determinism and transactional state

27. PyTorch DCP API: https://docs.pytorch.org/docs/main/distributed/checkpoint.html — sharded save/reshard and canonical state.
28. PyTorch DCP tutorial: https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html — practical examples.
29. PyTorch async DCP tutorial: https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html — stage vs write and concurrency.
30. TorchTitan checkpoint usage: https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md — full training versus model-only export.
31. PyTorch deterministic algorithms: https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html — guarantee limits.
32. PyTorch activation checkpointing: https://docs.pytorch.org/docs/main/utils/checkpoint.html — recomputation and RNG.
33. AWS S3 conditional writes: https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html — immutable/CAS pointer mechanics.
34. S3 consistency: https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html — per-key atomicity versus multi-object transaction.

### 23.5 Large-scale field reports and papers

35. LightningLM systems report: https://lightninglm.theschoolofai.in/ — reported 120B-class sparse MoE, failure modes, checkpoint and evidence grades.
36. Meta Llama 3 report: https://ai.meta.com/research/publications/the-llama-3-herd-of-models — training at 16K-GPU scale with quantified failure/recovery history.
37. Meta infrastructure reliability: https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/ — failure categorization and testing practice.
38. DeepSeek-V3 report: https://arxiv.org/abs/2412.19437 — 671B MoE efficiency, active vs total parameter examples.
39. DeepSpeed ZeRO original: https://arxiv.org/abs/1910.02054 — optimizer/gradient/parameter sharding design.
40. PyTorch TorchTitan code/benchmarks: https://github.com/pytorch/torchtitan — comparative performance and distributed training reference.
41. AWS ParallelCluster LLM operations field report: https://aws.amazon.com/blogs/hpc/part-1-managing-large-scale-llm-training-with-aws-parallelcluster/ — practical deployed HPC architecture and operational lessons.

### 23.6 Source precedence and validation warning

Official *current-version* vendor documentation takes precedence over unsupported tutorial statements about APIs and Xid definitions. Papers document what their authors observed on their systems; they do not guarantee equivalent cost, quality, or throughput under ERA V5's different topology. Instructor's acceptance requirements define *what G7 must deliver*, but do not establish that the team's proposed scripts are correctly implemented. Our local session experiment files support only their specific measured CPU/evaluation cases, never real allocated-node hardware performance.

---

## 24. Primary-source benchmark case studies and reproducible reporting

This section distinguishes **published measurements on other organizations' hardware** from values measured on the ERA V5 allocated GPU estate. No bandwidth, reliability or GPU cost reported here is an ERA V5 production measurement.

### 24.1 LightningLM: two concrete recovery failures

**Source:** [LightningLM 120B systems report](https://lightninglm.theschoolofai.in/), particularly its sections describing checkpoint corruption, state-preserving expansion, and failure analysis.

- Published report: 120B-class quantized-base sparse MoE trained in stages on one eight-GPU node; do **not** extrapolate full FP32 Adam state for all stored 120B parameters because the report describes trainable adapter state instead.
- Its pretraining run reached a logged step 3009 but its step-3009 checkpoint was unusable. A supervised run continued from **step 2909**, the latest sound checkpoint, demonstrating that *logged forward progress is not synonymous with durable committed progress*.
- During an expansion, target parameter names did not match source checkpoint names. Non-strict loading silently left target parameters randomly initialized while load returned successfully. Reported step-one loss changed from about 2.55 (correct load) to 2.95 (incorrect load). **One probe loss comparison is an additional canary, not complete future-step equivalence.**
- A background upload implemented as part of the training process reportedly degraded useful training throughput severely; separating upload from the critical path improved useful performance. **Do not adopt numerical throughput as an ERA V5 target without matching workload/hardware/runtime.**
- Required G7 controls: strict tensor identity and dtype/shape inventory, immutable checkpoint generations, protected last-known-good checkpoint, last-committed pointer, upload staging capacity, and canary/continuation tests.

### 24.2 Meta Llama 3: large synchronized training reliability

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) documents a 405B-parameter dense model and distributed large-scale training. The paper is an **observational benchmark** for fault-handling discipline, not an ERA V5 hardware performance guarantee.
- [Meta infrastructure reliability report, 22 July 2025](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/) describes fault classification, GPU/HBM/network-induced interruptions, and *silent data corruption* that can occur without an Xid or simple NaN flag.
- Implications for G7: nvidia-smi/DCGM/Xid tests are necessary but cannot prove scientific correctness. Add deterministic test workloads, gradient-norm/step traces, random shadow verification if budget allows, and quarantined host checks after suspicious divergence.

### 24.3 AWS: field-proven Slurm accounting at 240-GPU research allocation

- [AWS HPC field report dated 25 August 2026](https://aws.amazon.com/blogs/hpc/part-1-managing-large-scale-llm-training-with-aws-parallelcluster/) describes **30 p5en.48xlarge instances (240 H200 GPUs)** allocated over the broader programme, and an architecture using ParallelCluster, partition priorities, Slurm accounting limits, and FSx for Lustre. An example setup connected **up to 16 p5en nodes**; avoid conflating this deployment scope with all 30 provisioned instances.
- Notable controls: `AccountingStorageEnforce=limits,qos,associations`, distinct debug/normal/priority/exclusive partitions, explicit job-count and TRES policy, instance-placement and memory scheduling.
- **Important nuance:** AWS's published example shows certain configuration choices such as `JobExclusiveAllocation: false` for their multi-tenant use case. It is not a blanket recommendation for G7's synchronized long-running main training job, where exclusive node control may be preferable.

### 24.4 Cloud/platform compatibility evidence

- [AWS ParallelCluster EFA](https://docs.aws.amazon.com/parallelcluster/latest/ug/efa-v3.html): EFA enables low-latency OS-bypass networking on supported EC2 instances, recommends a placement group, and does **not** support cross-AZ EFA connectivity.
- [AWS Parallel Computing Service](https://docs.aws.amazon.com/pcs/latest/userguide/getting-started.html): managed Slurm service, distinct from provisioning a self-operated Slurm controller with ParallelCluster; evaluate as a separate operating model.
- [AWS HyperPod auto-resume](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency-slurm-auto-resume.html): managed node recovery and job restart *from the checkpoint supplied by the training workload*, not magically lossless in-step continuation.
- [Azure CycleCloud Workspace for Slurm](https://learn.microsoft.com/en-us/azure/cyclecloud/overview-ccws?view=cyclecloud-8): packaged Slurm infrastructure in customer's Azure tenant; infrastructure still needs operator-defined networking, identity, monitoring, and data durability.
- [Google Cluster Toolkit Slurm quickstart](https://docs.cloud.google.com/cluster-toolkit/docs/quickstarts/slurm-cluster): verified reusable cloud deployment pattern, not a substitute for distributed trainer state consistency.
- [NVIDIA Base Command Manager Slurm workload management](https://docs.nvidia.com/base-command-manager/product-manuals/admin/workloadmanagement): cluster provisioning/WLM operational system; determine support/licensing and who operates it before inclusion.

### 24.5 Repeatable GPU/NCCL benchmark reporting, without fake thresholds

Benchmark every *actual allocated topology* rather than reproducing attractive vendor graphics:

1. Log node IDs, GPU UUIDs, GPU/NIC PCI layout, driver, CUDA/NCCL versions, fabric configuration, thermal/power caps, cluster state, and code/image digests.
2. Run a correctness-only small all-reduce first. Verify expected reduction values and ensure ranks use their allocated GPUs.
3. Run [NVIDIA `nccl-tests`](https://github.com/NVIDIA/nccl-tests) with **pinned version and arguments**, small-to-large message sizes, warmups/repetitions and per-size observed latency. Distinguish intra-node, inter-node, TP, EP and other rank sets.
4. Report **algorithm bandwidth** (payload/time) and **bus bandwidth** as [defined by NVIDIA](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md), and provide raw command stdout/stderr. `busbw` is a normalized metric; it is not identical to wire throughput and is not a hard universal 400 GB/s admission target.
5. Profile p50/p95/p99 and run-to-run variance, test under well-defined isolation, and compare with a **per-topology empirically established** baseline. Label threshold values as `proposed`, `baseline-derived`, or `accepted`.
6. Separate a failed benchmark, unavailable benchmark, and partially completed benchmark. Never show `PASS` just because the binary existed or exited after a subset of tests.
7. If a cluster is dedicated to H100 today and B200 tomorrow, re-baseline after hardware/driver/network/kernel changes. Never inherit simulated H100 chart values.

### 24.6 Provenance table for any externally sourced numeric claim

| Required field | What must be recorded |
|---|---|
| Source class | Official manual, published paper, vendor experience report, own raw logs, analytical assumption or simulation |
| Exact reference | URL/DOI, section, publication/retrieval date, version/commit |
| Hardware scope | GPUs per node, HBM, node count, host/NIC/storage, GPU UUIDs if own test |
| Training scope | Stored vs active vs trainable params, batch, context, precision, optimizer, sharding layout |
| Execution scope | Number of steps/tokens, warmup, elapsed time, selection of run, failures |
| Metric | Numerator/denominator, units, decimal GB vs binary GiB, median/mean/p95 |
| Known limitations | Unknowns and what cannot be inferred for G7 |

---

## 25. First-principles operational walkthrough and owner handoffs
26. NEW — Session 15 stability/incident-response integration
27. NEW — Session 16 capacity and compute-budget integration
28. NEW — Revised G7 interfaces, acceptance tests and decisions
29. NEW — Fact checking and transcript qualification register

### 25.1 One successful training job — concrete, auditable sequence

**Before submission:** G5 produces the config hash and parallel-layout contract; data teams provide immutable manifest plus sampler/ledger API; G7 freezes launcher and checkpoint protocol; G8 defines event schema; instructors approve node access and storage roles. A mismatch is `BLOCKED`, not a skipped pass.

**Queue admission:** validate account association, QoS, CPU/GPU TRES and actual entitlement; Slurm queues until all required resources can be allocated together; job receives a unique `run_id` and immutable `attempt_id` and records the allocated node/rank map.

**Node preflight:** inspect GPU UUID count, driver/fabric/Xid visibility, recent critical events and required health agents. If diagnostics are inaccessible, emit `skipped` or `unknown` as specified by a versioned schema and admission policy. Full DCGM diagnostic and NCCL bandwidth sweep run in isolated diagnostic windows, not automatically as part of every job start.

**Trainer launch:** verify one Torchrun agent per allocated node for the chosen launch model, exactly one active rank mapped to each expected GPU, and one rendezvous per unique job attempt. Record all library/dataset/config hashes. A successful `torchrun` start by itself does not prove a correct global optimizer step.

**During training:** log batch IDs, rank-group membership, loss denominators, optimizer step, sampler ledger offset and data digests. Save **only at a globally agreed optimizer-step boundary** unless partial accumulated gradients and their denominator are also captured.

**Checkpoint transaction:** stage immutable tensor snapshots; per-rank DCP writes are completed and verified; every expected rank publishes a receipt; a coordinator validates full rank/layout ownership, all file checksums, environment identity and data cursor; the manifest is sealed; only then can an atomic conditional `latest` pointer advance. At no point should rank 0 gather the entire large, sharded model state by default.

**Recovery:** the controller fences old attempts, classifies the incident, tells Slurm to retry only if policy allows, reads only the last *committed* checkpoint generation, validates content and config, reconstructs full optimizer/scheduler/RNG/data state, checks a fixed loss probe and **multiple next-step transitions**, then resumes with an explicit replay/resharding classification.

**After completion:** the final generation must be durably committed; downstream evaluation/quantization can consume it read-only. The job records goodput and all recovery/queue/storage cost; an empty `latest` pointer or unverified generation must block publication.

### 25.2 Three failure timelines that a competent G7 review must understand

**Scenario A: failure before a checkpoint starts.** In-memory steps after the last committed generation are not durable. Normal policy is replay from that generation. Exact replay is feasible only under the fixed deterministic contract; zero recomputation cannot be claimed.

**Scenario B: failure during a multi-rank save.** Some shard files may exist. An older committed generation remains the only eligible restore point until the expected shard set, checksum/manifest and pointer CAS succeed. Stale attempts must not supersede a newer committed checkpoint.

**Scenario C: network partition after state commit but before writer acknowledgement.** The writer may not know whether the pointer advance succeeded. On restart, read the pointer and its immutable generation/attempt metadata; use conditional writes and idempotent transition checks instead of blindly saving a duplicate generation or overwriting progress.

### 25.3 What each team must demonstrate to G7

| Team | Contract supplied to G7 | Evidence needed |
|---|---|---|
| G5 Model/architecture | Full model + optimizer state inventory, model identity, parallel rank/tensor mapping, frozen/adapter state | strict save-load test on identical and remapped layout |
| G6 Kernels/performance | Runtime library versions, collective groups and latency/throughput telemetry | measured launch/NCCL/per-step and checkpoint overhead logs |
| G1/G2 Data | Deterministic batch IDs, ledger offsets, immutable tokenizer/packing/masking hashes | interrupted/replayed sequence equals uninterrupted batch order |
| G8 Monitoring | Health JSON, event names, alert ownership, labels | schema validation; ingest real negative/positive fixtures |
| G9 Evaluation | Snapshot consumption and reproducibility criteria | read-only eval from committed generation, no access to in-flight files |
| G12 Quantization | Supported export format and model identity | does not consume partial optimizer/shard data by accident |
| TSAI operations | Cluster inventory, storage, IAM, quotas, incident permissions | approved policy and actual Week-2 allocated-node logs |

### 25.4 Minimal lead's acceptance rubric

Do not merge a design as **operationally complete** until (1) the failure contract is measurable, (2) positive and negative tests are run, (3) evidence includes raw logs and versions, (4) all account/job/node/checkpoint identifiers line up, and (5) the instructor explicitly accepts the Gate-1 scope. The Week-1 gate can legitimately approve a **design and local CPU proof** while leaving actual NCCL/Slurm GPU and full V5 production-scale checkpoints unproven for Week 2 onward.

---

## Bottom-line decision summary

1. **Week 1:** treat Slurm-in-Docker as account/limit/requeue policy proof, Python/Gloo as a CPU communication proof, and CPU checkpoint replay as a deterministic-state proof. They are three independent evidence types.
2. **GPU readiness:** refuse to infer PASS from missing diagnostics or simulated UI; use NVIDIA's Xid catalog, DCGM and real NCCL data under the assigned topology.
3. **Checkpointing:** use DCP as the baseline serialization engine where compatible; wrap it with immutable rank-complete transactional commit, fencing, strict load and data-ledger coherence.
4. **Recovery:** guarantee no lost **committed** progress and exact replay at a defined fixed-layout test level first. Literal zero recomputation after physical failure is a distinct advanced RPO design requiring per-step durable state/deltas or sufficiently synchronized hot replication.
5. **V5 stage-specific scaling:** do not conflate stored, active and trained parameters. Publish capacity and throughput models as analytical until hardware, layout, optimizer and checkpoint measurements are provided.
6. **Leadership:** accept a PR on passing evidence and agreed contracts, not screenshot quality or descriptions of functions that code never executes. Reconcile all ADRs and team interfaces before treating any module as production-ready.



---

## 26. NEW — Session 15: stability, incident response, and monitoring contracts

**Source:** `ERA V5 Session - 15.docx`, transcript timestamp 2026-10-10 06:30 IST; see `SOURCE_TRACEABILITY_S15_S16.md` for anchors. **This is the instructor's account of prior training incidents and desired procedures, not independently reproduced measurements.**

### 26.1 Training health is multidimensional: compute + numerical + data + evaluation

The instructor distinguishes immediately visible failures (node loss, hangs, NaNs, loss spikes and imbalance) from silent failures (incorrect weights after resume, stale dashboards, wrong learning rate and unexpected data). For infrastructure this changes the meaning of `healthy`:

1. **Infrastructure healthy:** nodes, ranks, NICs, filesystems and links satisfy measured constraints.
2. **Training system healthy:** every optimizer step is executed exactly once in the committed progress ledger and is checkpointable.
3. **Numerically healthy:** finite loss/gradients/optimizer state; gradients, LR and update-to-weight ratio have plausible ranges for the *measured* run.
4. **Learning healthy:** held-out loss and agreed probes are consistent with the trajectory. A temporary training loss spike is not by itself proof of irreversible divergence.
5. **Observability healthy:** monitoring describes the *current* run, job attempt and checkpoint generation, not stale records.

**Rule:** Do not collapse these five into one untyped PASS. A passing `nvidia-smi` or NCCL test proves neither correct gradients nor a correct model restore. A numerical alert requires diagnosis; it is not an instruction to automatically drain otherwise healthy hardware.

### 26.2 G7 instrumentation API additions (coordinate with Team 8)

Every record needs `run_id`, `branch_id`, `attempt_id`, `global_step`, `tokens_committed`, `step_boundary`, `wall_time_utc`, `source_host`, `global_rank`, `mesh_config_hash`, `git_sha` and `config_hash`. A stream without these identities can make an old attempt look live. Include a **clock synchronization error estimate** if comparing events from multiple nodes.

Suggested events (design schema, pending Team 8 agreement):

| Event | Authoritative producer | Minimum payload | Operational consumer |
|---|---|---|---|
| `train.step.committed` | Trainer/ledger | step, batch IDs/hash, tokens, LR, loss, grad_norm, update-to-weight stats, duration | G8, G1/G2, G7 |
| `train.anomaly.detected` | Trainer/monitor | type, severity, window, threshold version, affected metrics, batch ID | On-call + incident service |
| `data.batch.manifest` | Data team | ordered sample IDs, shard offsets, packing/mask hash, tokenizer/mixture versions | Deterministic replay |
| `checkpoint.stage_completed` | Checkpointer | generation, all-rank receipts, D2H staging duration, staged-byte count | G7 |
| `checkpoint.committed` | Commit coordinator | immutable generation, exact ledger boundary, durable pointer, manifest checksum | Scheduler, eval, quant |
| `checkpoint.restore.verified` | Recovery process | step, restored state hashes, probe ID and match verdict | G7, G8 |
| `incident.branch.created` | Authorized controller | parent step/generation, reason, skip-list version, LR/config modifications, approver | Auditors + trainers |
| `eval.checkpoint.completed` | Evaluation | checkpoint generation, evaluation code/dataset hash, quantization scheme if any, metrics | Model team |

**Proposed minimal per-step numerical fields:** training objective loss(es); finite state; grad global norm; LR; optimizer step; update-to-weight ratio by parameter group; loss scaler for FP16, if present; expert selection counts and imbalance for MoE; key precision overflow counters when supported. **Do not assert that any one metric always predicts divergence.** Session 15 describes observed gradient-norm lead indicators, which are workload-specific hypotheses to calibrate.

### 26.3 Incident policy: detection is not an authorization to alter training

Implement a state machine `OBSERVE → SUSPECT → CONFIRM → PAUSE/QUARANTINE → DIAGNOSE → RECOVER_OR_BRANCH → VERIFY → RESUME`. Keep incident IDs and ordered evidence. Three paths are deliberately separate:

- **Hardware incident:** isolate affected node per current NVIDIA guidance; stop relevant ranks; fence abandoned attempt; recover from latest durable checkpoint. Never rely on the failing GPU to perform a last-minute snapshot after a GPU is already unavailable.
- **Numerical incident:** first inspect NaN/Inf, gradient behavior, exact data, precision, LR and model state; pause or rollback according to approved policy. Do not silently corrupt an otherwise good latest checkpoint with a NaN snapshot.
- **Suspected data-triggered incident:** record both batch identity *and* pre-batch model state. The instructor specifically notes that a batch may spike at a later model state but not an earlier one; blanket automatic deletion would be unjustified.

A rollback or batch exclusion creates an **explicit new branch**. Persist `(parent_checkpoint_id, decision_id, skipped_batch_ids/ranges, reason, authorizer, new_config_hash, timestamp)` and atomically connect it to the consumption ledger. The replay criterion is **same outcome for the same branch and decisions**, not the original branch's batch sequence after an intentional policy change. Do not retroactively edit a previously committed history.

### 26.4 Evaluations run on durable committed checkpoints

Session 15 requests fast held-out feedback and cheap evaluation copies. Do not block the training critical path on model conversion or benchmarking. Trigger evaluation from `checkpoint.committed` after verifying data integrity; a read-only worker consumes the immutable checkpoint, optionally producing a quantized evaluation derivative. Track that derivative's **quantization/config hash**: quantized evaluation is an approximate model-quality signal, **not a bit-equality proof of the original training weights**.

Set queue priorities and bounded evaluation backlog. If eval is slower than checkpoint generation, pick a documented sampling policy (e.g. protect mandatory milestone checkpoints and coalesce intermediate unstarted evaluations) without deleting committed training checkpoints. Publish held-out data manifest hash and leakage checks; never evaluate on data mistakenly included in training.

### 26.5 Time-series storage and dashboards

Session 15 warns that returning hundreds of thousands of points times dozens of plots to the browser is unworkable. Preserve full-resolution telemetry in authoritative storage with retention rules. Serve overview dashboards via downsampled series with **per-window min, max and count** as well as representative mean/percentile; this preserves sharp spikes that averaging alone hides. Support zoom for raw intervals and `attempt_id` filters. Never mix values from different branches after rollback without making branch boundaries visible.

### 26.6 G7 tests newly required by the Session 15 integration

- **S15-01 Replay batch identity:** recover in a new process; verify the next `N` ordered batch IDs, positions, packed masks and tokenizer/mixture hashes equal the original branch.
- **S15-02 Branch reproducibility:** inject numerical alert; authorize a skip window; persist decision; restart; verify identical skip-list semantics and batch sequence *for that branch*.
- **S15-03 Anomaly separation:** injected gradient spike must create numerical event and may pause job, but must **not automatically drain healthy GPU**.
- **S15-04 Stale dashboard:** a delayed event from an older `attempt_id` cannot become current run status.
- **S15-05 Eval attachment:** only `checkpoint.committed` triggers evaluation; metrics include committed generation and derivative quantization identity.
- **S15-06 Checkpoint load/copy benchmark:** allocate *randomly initialized model of final G5 topology*, measure state materialization, D2H, local write, object upload, verification and restore separately; no trained model is required for transfer and shape testing.
- **S15-07 State-machine alert quality:** tune detectors on a known trace and record false alarms, missed events, threshold versions and response time. Never claim a universal 7-sigma threshold.

## 27. NEW — Session 16: compute-budget, goodput and schedule integration

**Source:** `ERA V5 Session - 16.docx`, dated 2026-10-24; `ERA V5 Session capstone.docx` contains **identical extracted text**. This is one source reproduced in two files.

### 27.1 Compute planning is a contract, not a GPU-type preference

The instructor emphasizes model size, token target, batch size and LR; for infrastructure, the operative configuration must include both **total stored parameters** and **active trainable/compute parameters**. For sparse MoE and frozen/quantized bases these can differ dramatically. A rough planning approximation is

`training_flops ≈ 6 × active_parameter_count × training_tokens`, **only under declared model and FLOP-accounting assumptions**. Attention sequence-length terms, router/collective overhead, embeddings, long-context kernels, recomputation/reversibility and uneven expert activation can make actual FLOPs differ. Do not infer exact training time from this formula without a measured MFU and goodput on the target configuration.

**Source qualification:** The transcript suggests learning rate scales directly with batch size. That is a heuristic, not a general mathematical identity for AdamW, different loss normalization or different token mixtures. Team 5/6 should tune LR/batch jointly with controlled pilot runs. G7 must preserve the chosen settings and prohibit unapproved drift.

### 27.2 MFU and goodput: prevent denominator/double-counting mistakes

Adopt unambiguous definitions before using the planning equation:

- `MFU_during_training`: useful theoretical model FLOPs per **active training wall-second** divided by aggregated compatible peak FLOP/s for the selected precision.
- `goodput`: fraction of allocated wall-time spent delivering *new committed useful progress*, excluding interruption/recovery, replayed steps, startup and other nonproductive time.
- End-to-end usable throughput is `theoretical_peak × MFU_during_training × goodput / FLOPs_per_useful_token` as an **engineering approximation**, provided MFU is measured over periods that **exclude** the downtime counted by goodput. If MFU was computed over the *entire wall clock*, multiplying by goodput again **double counts downtime**.
- `tokens_per_second`: specify `computed_tokens/s`, `non-padding_loss_tokens/s`, and `durably_committed_unique_tokens/s`. These differ with packing, replay and failures. The denominator for the schedule is the third if the objective is unique usable progress.

Report `GPU-hours allocated`, `GPU-hours useful`, `checkpoint-stall-hours`, `replay_GPU-hours`, `queue-wait-hours`, `download/restore-hours`, `held-out evaluation lag`, and a clear total-cost basis. Session 16 uses goodput and MFU planning examples but they are not measured ERA V5 targets.

### 27.3 Training-scale dry runs

Once G5 provides actual model config and G6 provides kernels, perform the following **in this order**:

1. Calculate actual total/trainable/frozen parameter inventory, precision, optimizer, mesh ranks, forward activation envelopes and checkpoint payload via tensor introspection.
2. Start a small deterministic pilot on target accelerator type and interconnect. Collect warm-up-excluded median and p95 step duration, memory peak, successful tokens/s and fault-free `MFU_during_training`.
3. Benchmark checkpoint stages **on a randomly initialized instance** of the production shape first; log D2H staging, host RAM, NVMe, remote durable commit, restore and verify cost. This was explicitly requested in Session 15.
4. Run a **schedule calibration pilot** that includes periodic checkpoints, forced node/rank failure, recovery, eval triggers and log ingestion. Compute end-to-end unique-token goodput.
5. With Team 5/6, compute the feasibility envelope and sensitivity to GPU count, bandwidth, precision, failure rate and checkpoint cadence. Use conservative error bounds rather than a point estimate with false precision.
6. Only then adopt a checkpoint interval and production queue reservation. Reevaluate after Week-2 baseline and Week-3 dry run; do not canonize the instructor's example batch sizes or MFU numbers.

### 27.4 Instructor capstone gates vs conflicting transcript statements

Session 16 says the target training launch is **28 November** and mentions gates on **31 October, 14 November, 21 November and 28 November**. The transcript has inconsistent phrasing about holiday non-working days; the instructor's current CAP plan and latest Axiom message govern the operational calendar. Do not automatically shift a team deadline based on one colloquial statement in a spoken transcript. Week 1 remains a laptop/CPU design and local acceptance gate; a CUDA/NCCL physical-node pass is conditional on actual GPU access.

The transcript also discusses a proposed **16× compute/data advantage** without giving a complete reproducible algorithm in the provided source. **Treat this as an unverified research idea, not a capacity-planning credit or reason to reserve fewer GPUs.** Require concrete algorithm, baseline, quality-equal comparison, token counting and independently measured cost before accounting for it.

### 27.5 V5 stage-specific state inventory replaces an earlier hypothetical

The earlier 1T calculation was a sensitivity illustration, **not instructor intent**. V5 Session 15 states three models with sizes not disclosed; Session 6 rejects starting from V4 model weights. The official V5 programme focuses on a new frontier-scale MoE and its complete pretrain→posttrain→alignment→serve lifecycle. For each eventual V5 stage, derive `S_checkpoint = serialized bytes of every rank-owned tensor + non-tensor state + protocol overhead`; use `P_stored`, `P_trainable`, `P_active`, expert sparsity, base quantization and actual optimizer metadata separately. Compare measured stages only to like-for-like V4 logs, *not* to hypothetical 1T arithmetic. See the Rev4 crosswalk and config-driven model-stage capacity contract.

## 28. NEW — End-to-end G7 acceptance and owner integration

| Contract | G7 delivers | Dependency | Gate/proof |
|---|---|---|---|
| Queue | Slurm two **account** tiers with TRES/QoS, enforceable limits, real `sacct`/`sshare`, tested `RequeueExit` and preemption | Instructor's submit mechanism, cloud-admin privileges | Week-1 Docker job logs; no AWS keys in git |
| Launcher | Exactly one `torchrun` agent per node and expected worker/rank map; signals forward to checkpoint coordinator | G5 training entrypoint; G6 topology | Local CPU 2-rank; Week-2 CUDA allocation |
| GPU admission | JSON `pass/warn/fail/skipped`, bounded Xid window and source-access status, GPU IDs, DCGM where available | Team 8 schema; NVIDIA driver/hardware | CPU fixtures Week 1; real `nvidia-smi`/DCGM Week 2 |
| NCCL | Correct collective result, rank/group setup, timer, algbw/busbw and raw logs | Actual node/NIC topology | Gloo simulation Week 1; NCCL GPU Week 2 |
| Checkpoint | Whole coherent state incl. data ledger and branch decisions; immutable per-rank commit receipts and pointer fencing | G1/G2 data cursor; G5 optimizer/mesh; G6 DCP compatibility | Fresh-process CPU bit-equal Week 1; distributed restore Week 2 |
| Numerical recovery | Replay stable branch; anomaly/rollback creates auditable **new** branch | G8 monitor, G5 numerical policy | S15-01 through S15-07; dry runs |
| Capacity | Model-state tensor inventory, checkpoint bytes, throughput/stall and effective unique-token goodput | G5/G6 configuration, vendor node | Measured Week-2 baseline; Week-3/4 sensitivity/SLO |
| Evaluation | Trigger only after durable checkpoint; read-only evaluation copy, derivative config identity | G9/G12 | End-to-end event plus immutable checkpoint identity |
| Security/evidence | No credentials or private data in repo; scan, test transcript, config hash, provenance | All team PR contributors | `gitleaks` and reproducible test report |

**Owner discipline:** revise the six ADR/design documents in this package and open PRs independently; a fully formed design is not executable acceptance. Do not call a health simulator a real H100 check or label a CPU checkpoint test as a production-scale distributed restore.

## 29. NEW — Qualification register: transcript assertions vs verified engineering facts

| Statement | Status | Design consequence |
|---|---|---|
| S15 reports frequent interruptions and ~45-minute recovery in an example run | **Instructor report; exact external lab benchmark not independently verified here** | Design for measurable RTO, retries and recovery automation. Do not use as forecast MTBF. |
| S15 describes gradient norm rising before some loss spikes | **Instructor observation, not deterministic rule** | Instrument grad norms, calibrated detectors, delayed labels and incident investigation; don't always auto-stop. |
| S15 instructs exact batch reruns after rollback and revised skip lists | **Course requirement** | Bind data-ledger and branch policy to checkpoints; replay tests must compare batch IDs. |
| S15 asks fast checkpoint-triggered evaluation | **Course requirement** | Generate eval tasks from `checkpoint.committed`; isolate eval backlog and GPU budget. |
| S15 asks random-weight copies of final model shape to benchmark checkpoint speed | **Course requirement** | Benchmark D2H/NVMe/cloud/restore before trained model available. |
| S16 uses ~6ND planning | **First-order approximation, not exact for arbitrary MoE / attention** | Use analytical envelope then measured pilot; no peak-only runtime promises. |
| S16 suggests a direct learning-rate ↔ batch-size rule | **Unverified generalization** | Explicitly flag as heuristic; controlled sweep owned by G5/G6. |
| S16 mentions ~16× compute/data gain | **Insufficient method or evidence in uploaded transcript** | Do not include in production speed/price estimate until demonstrated. |
| S16's model examples / MFU and goodput numbers | **Reported teaching examples, not measured G7 values** | Define metric denominators; record exact hardware and measured loss-token throughput. |
| `capstone.docx` is another distinct briefing | **FALSE:** extracted text matches S16 exactly (checked SHA-256) | Cite once, not twice. |
| PyTorch DCP supports parallel save and resharding compatible states | **Official API**, subject to framework and optimizer layout | Pin PyTorch version; verify fixed mesh and topology changes separately. |
| PyTorch standard async save never blocks GPU→CPU | **FALSE for standard blocking-stage async_save**; newer stager may overlap stages under conditions | Benchmark and synchronize staged snapshots; consult version-specific docs. |

**Official API references (rechecked October 2026):** [DCP](https://docs.pytorch.org/docs/main/distributed/checkpoint.html), [Asynchronous DCP tutorial](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html); Slurm [`sbatch`](https://slurm.schedmd.com/sbatch.html), [`slurm.conf`](https://slurm.schedmd.com/slurm.conf.html), [`PreemptParameters`](https://slurm.schedmd.com/preempt.html); [NVIDIA Xid](https://docs.nvidia.com/deploy/xid-errors/index.html); [NCCL tests](https://github.com/NVIDIA/nccl-tests).
