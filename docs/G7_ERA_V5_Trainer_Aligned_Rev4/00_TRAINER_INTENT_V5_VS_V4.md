# ERA V5 vs ERA V4 — instructor intent, verified evidence and G7 consequences

**Revision:** 4 · 2026-10-27 · **Status:** source-grounded requirements crosswalk, **NOT** a frozen V5 model or hardware specification.

## Decision in one sentence

**Do not design G7 around a preselected 1T model.** Design for a V5 three-model programme and a versioned sequence of stage configurations whose sizes and hardware become authoritative only when Architecture (G5) and the instructor publish/approve them. V4 LightningLM is our *lessons-learned reference*, not the V5 weights or a binding V5 architecture.

## Evidence hierarchy

- **Trainer direct:** original Sessions 3, 6, 14, 15, 16; quotations below preserve spoken uncertainty. These are statements of intent, not production benchmarks.
- **Published V4 evidence:** [LightningLM systems/cookbook paper](https://lightninglm.theschoolofai.in/) and its [public training code](https://github.com/The-School-of-AI/LLM).
- **Published V5 programme:** [ERA V5 enrolment/course page](https://registrations.theschoolofai.in/courses/era-v5), describing the intended model-release and research workflow.
- **Operational requirements:** instructor's G7 Week 1 notice, capstone plan and latest Axiom messages. G5 config v0 (27 Oct) and frozen config (anticipated 14 Nov) are **external dependencies**, not documents verified here.
- **Unknown:** final V5 three model sizes, routed/active/fully trainable parameter counts, target token budgets, GPU type/count, optimizer/state format, and scheduled checkpoint interval. No values are fabricated for these.

## Source-grounded comparison

| Dimension | ERA V4: published or stated evidence | ERA V5: what instructor actually intended/said | G7 interpretation |
|---|---|---|---|
| Purpose | Public 0.1V model lineage; systems recipe on making constrained hardware productive | New lab-run frontier-scale MoE effort; pretraining, SFT, alignment, evaluation, serving, open code and research outputs | Platform must cover entire model lifecycle, not just a one-off pretraining job |
| Number/size of models | Four **growth stages** with released 1.78B dense, 4.96B MoE, 9.36B MoE, 118.67B MoE artifacts | Session 15 explicitly says **three models** and sizes are **secret** at that session | No 1T target, no inferred 40B/120B V5 endpoint; maintain independent `model_id` and `stage_id` |
| Starting point | V4 small trained dense seed grew into later stages | Session 6 explicitly says **not** starting V5 from V4 weights — different model “DNA” | Separate new V5 initialization and keyspaces from V4; V4 checkpoint converter only informs tests |
| Architecture | 120B model reports 460 routed experts, top-12, one shared expert; reversible backbone; quantized expert base with adapter updates | Instructor discusses novel/fresh architecture choices, MoE, reversibility, experts, parallelism; G5 owns decisions | Do not preselect topology/optimizer or checkpoint accounting without G5 tensor inventory |
| Parameters | 120B: 118,670,450,556 stored, 5,927,559,036 active per token; ~2.26B adapter params with optimizer state in TQP expert path (paper's report) | Final V5 total, active and trainable counts **not given** | `P_stored`, `P_active_per_token`, `P_trainable`, frozen/quantized layout are separate configuration fields |
| Token budget | Session 3 cites roughly 1.15T–1.18T corpus inventory; **do not equate corpus inventory to actually processed tokens**. V4 paper explicitly distinguishes design corpus from consumption | Session 14 proposes 20T/25T/possibly 45T **if budget allows**; 16× data/compute technique is discussed without published V5 validation | Express token schedule as planned/unique/actually consumed/replayed. Treat ambitions as scenarios, not allocations |
| 40B model | Not V4's final released size | Session 3 **40B** is an explicitly hypothetical individual homework/assignment, not an assigned capstone size | Never set G7 capacity to 40B on that basis |
| 3B active / 6T example | Not a V4 size assertion | Session 16 states *“If we decide that we have three billion active parameters…”* to demonstrate 2,000 tokens/active parameter; **conditional** | Use as arithmetic example only, not a binding V5 training-token target |
| Hardware | V4 growth documented on eight-GPU nodes: A100, H200, B200/B300 across stages, according to paper | V5 instructor discusses candidate hardware; shared V5 GPU node arrives Week 2 per CAP plan | Hardware discovery and benchmark decide topology; no assumed H100/B200/GB200 allocation |
| GPU operations | V4 paper reports checkpoint corruption, silent key-mismatch load, upload interference | G7 required to implement Slurm queue, health/NCCL, durable checkpoint, CPU bit-equal test, auto restart, two dry runs | Make negative/recovery tests gate blockers; not merely diagrams |
| Posttraining | Session 15 instructor says V4 posttraining was incomplete/time-limited | Session 15 explicitly says V5 posttraining is an objective | Support new job types and stage boundaries, not only training checkpoints |
| Review/launch | V4 report is a published past model lineage | CAP plan: Gate 1 Oct 31, Gate 2 Nov 14, dry runs Nov 21/28 and planned main training launch Nov 28 | Design first; actual numerical scale/timing only after G5 and measured GPU baselines |

### Attribution and precision notes

**Session 3**, near its conclusion, says to *assume* a 40B model for an individual assignment. The immediately following Chinchilla arithmetic in the spoken transcript is inconsistent (`40B × 20` is 800B, not 200B); it is not a production requirement. **Session 6** (~student question on starting with V4 weights) says no due to a different architecture. **Session 14** (~intro) conditionally mentions 20T–45T data and a proposed 16× approach. **Session 15** (~team formation/release) explicitly states three V5 models with sizes not yet disclosed and emphasizes posttraining. **Session 16** discusses capacity equations and external model examples; do not treat external figures such as 16,000 GB200 or 21% MFU as V5 GPU purchases or target MFU. The capstone-labeled DOCX is text-identical to Session 16 in the available extracted text.

The publicly available LightningLM paper describes a four-stage lineage and provides its own explicit evidence qualifiers. It reports 1.78B to 118.67B growth, production stage throughput by hardware, and 120B trained through a frozen quantized base plus trainable adapters. Its public page says the corpus design was larger than actual run consumption. Earlier publicity quotes about corpus size or training duration should not be substituted for token-ledger measurements.

## Trainer-aligned V5 success conditions (programme-level)

1. Implement **real** end-to-end training and release of the instructor-selected models, including pretraining and subsequent training stages.
2. Start from **new V5 architecture and trained initialization**, not an undeclared V4 checkpoint.
3. Validate MoE routing, numerical stability, data curriculum, tokenizer fidelity, expert load, held-out evaluation, and trainability.
4. Make the run **observable and recoverable**: no partially committed snapshot selectable, bitwise replay where deterministic, auditable exceptions for changed topology or intervention, and measured recovery/goodput.
5. Work on available hardware with **measured** profiles: distinguish advertised FLOP peaks from sustained token throughput, runtime MFU and whole-run goodput.
6. Publish traceable configurations, checksums, code/evidence, checkpoint lineage, and trained model artifacts, subject to instructor decisions.

### What this means for Week 1 G7

No part of Week 1 requires proving a trillion-parameter checkpoint. The instructor wants **queue design and two-account Slurm Docker test; JSON health/launch/Xid with missing tools marked skipped; NCCL test path; checkpoint design and bit-equal CPU restore; secret scanning and PR evidence**. Define typed interfaces to G5/G8 now. Size the real checkpoint service only after G5 supplies a model and optimizer tensor inventory. Preserve Stage 0 CPU contract tests independent of the eventual parameter scale.

## Links and artifacts of record

- V5 official programme: https://registrations.theschoolofai.in/courses/era-v5
- V4 systems paper: https://lightninglm.theschoolofai.in/
- V4 code/cookbook: https://github.com/The-School-of-AI/LLM
- Local archive references (no claims beyond text): `ERA V5 Session - 3.txt`, `ERA V5 Session 6(1).txt`, `ERA V5 Session 14.docx`, `ERA V5 Session - 15.docx`, `ERA V5 Session - 16.docx`, `ERA V5 Session capstone.docx`.

## Unresolved facts that G7 must request rather than assume

1. Three V5 model IDs, stages, scale ladder, parameter ownership and stage boundaries.
2. Actual G5 v0 and v1 JSON/YAML configs and manifests, including checkpoint model-state naming and precision.
3. Owned/frozen/trainable expert-base and optimizer parameter inventory, checkpoint type and tensor shapes.
4. Whether trainer intends 3 *independent* models, or one 3-stage grown model family — “three models” **does not by itself resolve this**.
5. Planned unique, effective, consumed and replay tokens by phase; known training-token schedule and optimizer steps.
6. GPU SKU/count, nodes, EFA/IB/NVLink topology, CPU RAM/NVMe, checkpoint durability location, and provider ownership.
7. Exact meaning of no lost committed work, planned failover tests, and allowable re-computation, plus zero-recompute performance budget.
