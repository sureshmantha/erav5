# Comprehensive ADR coverage and DeepSpeed-aligned gap analysis

**Status:** Proposed G7 architecture decision register, October 29 2026. Audits **P=Pradeep, L=Mallik, X=Max, S=Suresh**; **M=Murthy not accessible**, NOT reviewed. More detailed file-level observations: [`02_SOURCE_CODE_REVIEW.md`](02_SOURCE_CODE_REVIEW.md).

> **Terminology:** These are *proposed G7 architectural decision topics*. Neither DeepSpeed nor the two comparison training frameworks publish a universal official 36-ADR list. We derive the subjects from documented APIs and system behavior; do not cite the IDs as official DeepSpeed ADRs. **DeepSpeed runs training, not Slurm nor GPU maintenance.** An aligned G7 system must compose DeepSpeed with Slurm, NVIDIA driver/DCGM/NCCL, durable storage and the ERA V5 data/monitoring contracts.

## Framework comparison and selection

| Function | DeepSpeed (North Star) | PyTorch TorchTitan / FSDP2 | NVIDIA Megatron Core | Consequence for G7 |
|---|---|---|---|---|
| Training engine | ZeRO stages, gradient accumulation, mixed precision, offload and MoE | Native FSDP2 + DTensor, FSDP/TP/PP/EP/CP and trainer recipes | TP/PP/EP/CP, distributed optimizer, optimized GPU kernels | Team 5 selects actual model/framework; ensure stable job and state contracts |
| Checkpoint writer | `engine.save_checkpoint` / `load_checkpoint`; all ranks participate; ZeRO optim states | DCP sharded and TorchTitan async checkpointer | Distributed checkpointing with topology-aware shard metadata | Do not assume one checkpoint format can load across all stacks |
| CPU proof | Can test trainer semantics locally where DS is installed, but DS not available here | DCP DTensor 2-rank CPU save/load is executable | Library/GPUs not available in audit | Preserve exact model+optimizer training CPU reference and DCP experiment separately |
| Rank placement | DS launcher or `torchrun`+`deepspeed.initialize` | `torchrun`, Slurm multi-node recipe | Typically `torchrun` / site launcher | Choose **one agent/node** for torchrun, or one direct rank/GPU for MPI tools |
| Scheduler | External Slurm/Kubernetes/AWS | External Slurm/Kubernetes/AWS | External Slurm/Kubernetes/AWS | Implement actual two-account Slurm Docker acceptance |
| Health/NCCL | Distributed communications, not hardware lifecycle | ProcessGroupNCCL + debugging tools | NCCL, Transformer Engine, NVIDIA resiliency add-ons | DCGM/Xid/NVML are separate independent operational gates |
| Resilience | Engine checkpoint API, ZeRO state | Checkpointer/elastic tooling | Distributed checkpoint + fault management | Our own manifest/remote commit/queue reconciliation requires engineering and tests |

## 36 decision topics and repository coverage

Legend: `ADR#` = design documented; `Prototype` = code exists but not necessarily validated; `Gap` = cannot establish from source; no column value implies production acceptance.

| # | Decision topic | Framework/industry reference | P | L | X | S | Target gap action | Pri |
|---:|---|---|---|---|---|---|---|---|
| 1 | Mission and SLO/RPO/RTO | DS checkpoint; Titan fault tolerance | ADR1/2 | DESIGN | Design | Docs | Specify failure domains, replay definition, queue RTO | P0 |
| 2 | Job admission and authenticated accounts | Slurm associations/QoS (external) | ADR2 | Docker | Detailed EC2 | Docker | Real Slurm Docker two-account proof; Murthy audit | P0 |
| 3 | QoS limits/fair-share/accounting | Slurm TRES, qos, fairshare | ADR2 | Docker | Detailed EC2 | DenyOnLimit | Measure rejection/pending, sacct GPU-hours and scheduling fairness | P0 |
| 4 | Scheduler controller/DB resilience | Slurm best practices | DB backup | Docker DB | Detailed EC2 | Docker DB | DR drills/DB backup, MUNGE rotation and controller fencing | P1 |
| 5 | Preemption and exit contracts | Slurm; DS trainer signal integration | 75/78 | Requeue | No preempt | 75/Hold | Resolve mutually exclusive policies; test SIGTERM, timeout and 75/78 | P0 |
| 6 | GPU resource isolation / GRES | Slurm GPU GRES and cgroups | Node configs | Mock GRES | EC2 GRES | CPU fake | Week2 real GPU cgroup, MIG awareness and per-task device isolation | P1 |
| 7 | Launcher world size and rank mapping | DeepSpeed launch + torchrun; Titan Slurm | mainrun.sbatch | torchrun 1/node | DS CLI single | torchrun 1/node | Assert exact world-size and one agent/node across ranks | P0 |
| 8 | Immutable run configuration | DS config JSON; Titan config registry | Config refs | Meta/hash | Installer schema | Config hash | Freeze configs; pin git/image/CUDA/driver/NCCL versions | P0 |
| 9 | DeepSpeed engine initialization | DeepSpeed engine/initialize | Proposed | Not integrated | Launcher DS | Not integrated | Team5 model+training script to call DS.initialize and verify runtime config | P1 |
| 10 | ZeRO stage 1/2/3 and optimizer sharding | DeepSpeed ZeRO | Design | DCP (different stack) | Training demo | CPU full | Choose stage after Team5 model+node memory profiling; no blanket 14B | P1 |
| 11 | ZeRO offload / Infinity | DeepSpeed ZeRO-Offload/Infinity | Some notes | Not tested | Not tested | Not tested | Quantify CPU/NVMe pressure, cache/stall, NUMA and I/O budgets | P2 |
| 12 | BF16/FP16/FP8 numeric policy | DS mixed precision; NVIDIA TE | Config | bf16 note | T4 fp16 note | CPU fp32 | Fix G5 precision, scaler, overflow, deterministic probes | P1 |
| 13 | Gradient accumulation and step boundary | DS gradient accumulation | Checkpoint docs | Boundary | Demo | CPU boundary | Test mid-microstep interrupt and decision to save grads or defer save | P0 |
| 14 | Global batch and loss normalization | DS training API; TorchTitan | Docs | Data ledger | Sample jobs | CPU sample IDs | Global loss-token weighting + DP subgroup correctness with padding masks | P1 |
| 15 | Tensor/pipeline/expert/context parallelism | Megatron Core; TorchTitan mesh | ADR1 | DCP/DTensor | Single node | Not tested | Team5 mesh contract incl. EP mapping/optimizer compatibility | P1 |
| 16 | Collective correctness and NCCL | NVIDIA nccl-tests; DS comm | allreduce | allreduce | Bench guidance | Gloo+NCCL | GPU all-reduce with exact values + nccl-tests alg/busbw topology | P0 |
| 17 | Collective hangs/flight recorder | PyTorch PG NCCL; NVIDIA resiliency | Some env | Flight recorder | Operational notes | Missing | Trace-buffer settings, timeout injection, rank anomaly and escalation | P1 |
| 18 | GPU inventory and ECC/Xid | NVIDIA Xid/DCGM | Rich modes | Scripts | Out of scope | JSON fixtures | Agree G8 JSON, Xid severity, access errors, sampled time window | P0 |
| 19 | NVLink/IB/EFA/topology baselines | NCCL tests, NCCL docs | Monitoring | NCCL | EC2 notes | Not measured | Separate passive prolog from exclusive benchmark, hardware baselines | P1 |
| 20 | DCGM continuous and maintenance diag | NVIDIA DCGM | Multiple modes | Tests | None | Full opt | No disruptive diagnostics on occupied GPU; mock tool errors | P1 |
| 21 | Sharded checkpoint format | DS save_checkpoint; Titan DCP; Megatron DCP | ADR1/DCP | DCP | Local torch save | CPU local | Converge on Team5 engine; DS Zero vs DCP interop adapter/format | P0 |
| 22 | Per-rank state inventory | DS save_checkpoint client_state; DCP | ADR1 broad | Per-rank RNG | Demo | Python/NumPy/Torch | Include full RNG, optimizer, sampler, LR, precision and skip decision | P0 |
| 23 | Atomic multi-rank commit | Outside DS engine: object-store CAS and receipts | Proposed S3 | COMPLETE | Not full | Local manifest | Implement rank receipts + remote integrity/CAS + stale epoch fencing | P0 |
| 24 | Checkpoint integrity and corruption | DS checkpoint API + production wrapper | Hashes | COMPLETE | Basic | SHA+negative | Remote SHA, metadata, late failure and rollback test | P0 |
| 25 | Async checkpoint staging | DS + DCP async options | Design | Future | Not integrated | Design | Measure staging overwrite hazards, host RAM, write stalls | P1 |
| 26 | Topology-changing restart | DCP resharding, Megatron dist_ckpt; DS limitations | Design/proofs | Claims | Single node | CPU fixed | Test model+optimizer reshard, mark GPU bitwise claims conditional | P1 |
| 27 | Strict restoration & no silent misload | DS load_checkpoint strict/format; TorchTitan | Probe+key checks | Frozen/hash | Demo | Strict schema | Compare next M losses and param/optimizer + input IDs across fresh processes | P0 |
| 28 | Data consumption ledger and audit branches | TorchTitan data Grain; G7 session6 | Ledger | Sample cursor | Data demo | Cursor | G1/G2 agreed ledger, exact batch masks, branch ID and skip policy | P0 |
| 29 | DeepSpeed compatibility at checkpoint boundary | DeepSpeed ZeRO stage semantics | Not verified | Not verified | DS launcher | Not integrated | All-rank save/load through real DS engine; do not claim DCP equals DS | P1 |
| 30 | High availability restart controller | Slurm requeue+DS restart | Queue wrapper | restart.sh | Installer | Requeue scripts | Node kill→fence→verify remote commit→restart→test next steps | P1 |
| 31 | Operational observability and alerts | DS monitor, Titan metrics, NVML | ADR4 74 metrics | Limited | Scheduler logs | Acceptance table | Event IDs, standardized time, cost/goodput dashboards with source tags | P1 |
| 32 | Performance metrics and goodput | DS flops profiler; Titan MFU | Cost model | Sparse | Queue accounting | Status | Measure unique committed tokens, not GPU busy; avoid double count | P1 |
| 33 | Evaluation/quantization consuming checkpoints | DS zero_to_fp32; Titan export | Docs | DCP conversion | eval.py | Not present | Read-only model export and immutable generation proof for G9/G12 | P2 |
| 34 | Secrets/authentication and supply chain | Slurm admin+cloud IAM; source scanning | Operational notes | Docker only | Root installer | gitleaks script | gitleaks executed, pinned images, least privilege, no keys in PR | P0 |
| 35 | End-to-end fault injection & runbooks | NVIDIA resiliency; TorchTitan debugging | Simulation | Demo | Recovery test | CPU negative | Actual Slurm+node/DCP/S3 fault injections, operator on-call | P1 |
| 36 | Evidence and provenance integrity | Testing standards | Support tests | Demo results | EC2 tests | CPU evidence | Show CPU/GPU/cluster flags, test commands, hashes, stdout+stderr and reviewer | P0 |

## Most material missing commitments

**1 — Week 1 block:** Actual Slurm Docker runtime acceptance. Config presence, simulated policy outcomes and healthy Docker logs are insufficient. Capture *two authorized submissions*, *denied forbidden account*, *over-cap rejected or blocked according to policy*, accounting, and exact tool versions. `DenyOnLimit` exists in S hotfix but remains unverified on the user's Mac.

**2 — Engine compatibility:** Team 5 must choose whether the production engine is DeepSpeed (ZeRO stage and optimizer), TorchTitan/FSDP, or Megatron. DeepSpeed checkpoint files must be saved/loaded using the matched engine and ZeRO semantics; DCP is **not** a universal interchange format, and the CPU DTensor DCP experiment proves only its separate DCP path.

**3 — Durable transaction beyond checkpoint writer:** a backend's `save_checkpoint` success does not, alone, prove multi-rank remote data survived node loss and can be atomically selected as latest. Design immutable generation IDs, complete expected-rank receipt set, checksum verification, writer epoch/lease fencing and a single conditional pointer advancement. Verify on an approved object store with injected failures.

**4 — Slurm scheduler policy conflict:** Max's no-preemption/partition isolation, versus Pradeep's mainrun QoS preemption and hot spare. The first-stage simulation may use `mainrun` protected and no forced preemption of irreplaceable training; production policy decided after spare capacity/MTBF/GraceTime measurement. Document the decision in a new accepted ADR.

**5 — Scientific replay:** same weights are insufficient: optimizer, scheduler, RNG, data cursor, skip decisions, masks, software and reduction topology must be considered. A fixed CPU new-process exact replay is established in local tests. Future GPU bitwise identity across different meshes must **not** be promised without an explicit reproducibility proof.

**6 — Hardware health:** `skipped` must be distinct from `pass`; Xid severity uses NVIDIA per-driver guidance; passive prolog should not run intrusive 512 MiB NCCL sweeps on an occupied GPU. Deep GPU diagnostics should execute in an exclusive maintenance allocation.

**7 — Evidence:** raw generated simulations, claimed user Mac passes or model benchmarks must never be promoted to actual Docker/Slurm/GPU evidence. Need reproducible commands, stdout/stderr, hashes and an authorized reviewer.

## Plan — prioritized and time bounded

### Gate 1 (Week 1, Oct 31): minimum defensible completion

1. **P0** Merge repo path and clean redundant folders; preserve original ADRs and conflicting policy decisions in the review register.
2. **P0** Execute Slurm Docker Compose on a local Docker machine and run policy tests Q1–Q4; fix portability/Slurm version only with actual logs. Record commands, job IDs, account/priority state, denied requests and `sacct` output.
3. **P0** Run exact CPU replay (12 uninterrupted vs save/restart 5+7) including loss bits, input IDs, model+optimizer digest, cursor; corrupt and incomplete save negative tests.
4. **P0** Validate health JSON on CPU, `skipped` missing tools, injected Xids, and CPU Gloo when available; **do not claim NCCL on CPU**.
5. **P0** Run `gitleaks detect --source .` and review PR against teacher's original directory paths.

### Gate 2 (Nov 1–14): GPU + DeepSpeed integration

- Team 5 selects actual training entrypoint, model/mesh, precision, DeepSpeed config and ZeRO stage; validate allocation, one-agent/node or native per-rank launcher, DeepSpeed engine save/load and optimizer state on approved node.
- Verify actual GPU ECC/UUID/Xid/NVLink/DCGM, real NCCL correctness and vendor `all_reduce_perf` raw bandwidth; pin topology and driver/NCCL versions.
- Instrument checkpoint stage times and implement remote durable receipts/manifest/pointer fencing. Inject job kill and wrong config/optimizer key.

### Gate 3 (Nov 15–21): dry run 1 and goodput

- Compare full rank kill vs node loss; verify complete checkpoint restarts, exact data IDs, no silent optimizer reinitialization.
- Goodput measured as *new unique committed training tokens* / allocated wall time; evaluate checkpoint cadence and queue partition policy with actual counters.
- Run numerical versus hardware incident policy, audit alerts, dashboard read-only committed checkpoint consumption.

### Gate 4 (Nov 22–28): dry run 2, operating handoff

- Planned rank/node failure, recovery and verification under real configuration. Confirm drain/return-to-service approval and deployment rollback.
- Publish failure runbook and evidence table with accepted risk exceptions. Instructor approves launch; model-specific production SLA is bound to actual nodes and configuration.

### Beyond Gate 4

Optional: ZeRO-3 offload/Infinity, multi-architecture checkpoint conversion, fault-tolerant per-step state replication, dynamic resource pool/spot policy, topology-changing numerical replay and formal RTO/RPO SLO. Choose only after measured goodput benefit.

## Pitfall avoidance and strong practices

| Pitfall | Prevention and proof |
|---|---|
| Accidental extra torchrun workers | Assert one torchrun agent/node, worker count exactly allocated GPUs |
| Over-limit job remains pending unexpectedly | Choose DenyOnLimit and assert actual installed Slurm semantics |
| Job requeued without training state | Trainer loads durable committed checkpoint; verify LR/optimizer/RNG/ledger next M steps |
| Mixed data from several attempts | `run_id`, `attempt_epoch`, `branch_id`, `checkpoint_generation` everywhere |
| Missing GPU tools reported as success | `skipped`, require checks tied to deployment mode, fail admission safely |
| Xid maps treated as universal hardware death | Vendor Xid + HBM/DCGM context and owner-approved drain decisions |
| Async checkpoint races weight mutations | Measure staging completion/memory, bound in-flight saves, freeze before writing |
| Uncommitted shard selected as latest | all-rank receipt verification, immutable manifest and conditional fenced latest pointer |
| Rank/world size changes promise exact bits | Qualify determinism only on pinned layout or supply actual independent proof |
| MFU × goodput double-count | Define active-train MFU denominator separately from allocated-wall goodput |
| Security disclosures in public PR | `.gitignore`, redact real logs, run gitleaks at repo root, do not commit credential env |

## Primary references (not claims of official ADR numbering)

- DeepSpeed: [docs](https://deepspeed.readthedocs.io/en/latest/), [checkpoint](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html), [ZeRO](https://deepspeed.readthedocs.io/en/latest/zero3.html)
- PyTorch: [TorchTitan](https://github.com/pytorch/torchtitan), [checkpointer](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md), [DCP API](https://docs.pytorch.org/docs/main/distributed.checkpoint.html)
- NVIDIA: [Megatron Core](https://developer.nvidia.com/cuda/cuda-x-libraries/megatron-core), [NeMo DCP](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/checkpoints/dist_ckpt.html), [Xid](https://docs.nvidia.com/deploy/xid-errors/index.html), [NCCL tests](https://github.com/NVIDIA/nccl-tests)
- Scheduling: [Slurm QoS](https://slurm.schedmd.com/qos.html), [Slurm account limits](https://slurm.schedmd.com/resource_limits.html), [Slurm preemption](https://slurm.schedmd.com/preempt.html), [Slurm Docker container guidance](https://slurm.schedmd.com/containers.html)
