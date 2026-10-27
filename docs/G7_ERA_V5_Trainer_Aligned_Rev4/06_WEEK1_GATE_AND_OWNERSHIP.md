# Gate 1 evidence, ownership and integration handoffs

> **REV4 MODEL SCOPE (2026-10-27):** ERA V5 is a new three-model capstone programme; Session 15 did not disclose final sizes. Do not assume a 1T or 40B V5 target; the 40B case was Session 3 homework and V4's 118.67B is historical evidence. V5 stage-specific tensor/optimizer inventories from G5 determine GPU, checkpoint and transport sizing. Consult `00_TRAINER_INTENT_V5_VS_V4.md` and `07_CONFIG_DRIVEN_MODEL_STAGE_CAPACITY.md`. Existing design controls and Week 1 CPU tests remain applicable independent of model size.


Date: 26 Oct 2026. No personal diagnosis included in this review.

## Roles — proposals only, ask people to confirm on Axiom

| Workstream | Proposed current evidence owner | Status from chat (not assignment) | Next acceptance evidence |
|---|---|---|---|
| Architecture/ADR coordination | Pradeep has written first proposed ADRs | Drafts attached | Log instructor comments and ADR sign-off status |
| Checkpoint protocol+CPU code | Pratik volunteered | Not confirmed | CPU new-process replay, corruption, negative tests, reviewer sign-off |
| GPU health/Xid/NCCL | Satyanarayana reportedly has first cut | Code not attached here | JSON, fixtures, CPU/Gloo + real GPU pending |
| Queue/Slurm Docker | Unassigned after Nitin left | **Must name owner by Tue 27 Oct** | Two Slurm accounts + limits, preemption and requeue tests |
| Overall integration review | G7 lead | Responsible for accepted PR integration | Merge gate and open-risk register |

Do not publish this table as a roster until named people confirm ownership. Instructor requires work/decisions visible through GitHub PRs and Axiom; summarize every call in Axiom.

## 27 Oct / 29 Oct dependencies

- **27 October:** G5 architecture config v0. Need model/optimizer state shape, activation size, proposed parallelism and GPU memory target. Without it checkpoint sizing is a sensitivity analysis, **not** a measured capacity plan.
- **29 October:** G8 health JSON contract. Interim `v5-health/1-provisional` must be reconciled; no permanent JSON schema claim.
- G1/G2 dataset ledger/schema: step-to-data commit mapping, token IDs and replay semantics.
- G6: performance and NCCL diagnostics requirements.
- G9/G12: checkpoint consumer semantics and read-only manifest contract.
- TSAI Lab: compute count, exact GPU model, Slurm/ParallelCluster control-plane choice, cluster ownership, IAM roles, persistent storage, failure-domain/SLO definitions.

## Core 31 Oct gate — evidence packet

- Queue: `queue_design.md`, updated ADR, Docker Compose source and pinned images, `docker compose config` output, Slurm version, account/limit `sacctmgr` commands, admission/preemption/requeue/sacct logs, Docker test result.
- Health: `launch.sh`, `gpu_health.sh`, `xid_scan.py`, 1 valid JSON file from **real GPU** if available, skipped-tool proof on CPU, fixture assertions, NCCL all-reduce result from real GPU or **explicit exception/pending** due to week-2 hardware access. Gloo CPU is not NCCL.
- Checkpoint: `checkpoint_design.md`, `ckpt.py`, tests in distinct processes, loss hex values, final parameter/optimizer hashes, sampler input IDs, integrity error/fallback and partial-write fault injection results.
- Security: repository scope diff and `gitleaks detect --source .` result **run by the team**; no keys, signed URLs, parquet packs or personal information committed.
- Governance: issue/PR review by lead and instructor with list of accepted exceptions and explicit statuses.

## What this package does and does not prove

**Executed here on CPU**: new-process uninterrupted-vs-restored training losses exactly equal (hex), every per-step batch ID equal, full model+optimizer+scheduler+RNG state hash equal; SHA tampering denied; earlier checkpoint fallback; failed pre-seal generation does not change latest; Xid parsing fixture; JSON skipped checks; 2-process Gloo all-reduce correctness.

**NOT executed here**: Docker-based Slurm cluster, two-account enforcement or preemption, real NVIDIA GPU metrics, DCGM/ECC, NCCL GPU test, AWS S3 consistency, DCP sharded GPU saves, node failure, 1T state persistence or zero-recomputation guarantee, `gitleaks`. These are explicitly pending.

## Discussion prompts for instructor

1. Does 'no lost work' mean no loss of *durably committed checkpoint* or **no repeated optimizer steps**? The latter requires per-step state durability or replicated hot state, an entirely different budget.
2. Is bitwise replay required only for fixed world size/kernel stack, and is deterministic-equivalent/non-bitwise resharded replay allowed after rank loss?
3. Are the two Slurm accounts `mainrun` and `teams` in a *single AWS account*? What limits should be enforced?
4. Does the lab's submission service or Slurm own the requeue and restart loop? Which exit and signal contract is required?
5. Which exact GPU, count, region, networking, failure domains, S3/FSx design, and IAM roles are approved by TSAI Lab?
6. Is a GPU NCCL Gate-1 requirement waived until the week-2 shared node? If not, who provides approved Colab/Kaggle access and evidence?


---


## Source-informed Week-1 changes (Sessions 15/16/capstone)

**Priority is unchanged:** local queue design, Slurm-in-Docker with **two real Slurm accounts**, health/NCCL JSON fixture tests, checkpoint CPU bit-equal restore, gitleaks 0 findings. The Session 16 briefing explicitly describes Week 1 as lightweight preparatory work; full node work follows hardware allocation. The capstone attachment repeats Session 16 verbatim rather than adding separate requirements.

### Required handoffs and concrete owners (proposed; team lead confirms)

| Handoff | Provider → G7 | Acceptance evidence |
|---|---|---|
| Model state/tensor inventory | G5 → Checkpoint owner | Config v0 and checkpointable tensors including optimizer/master/frozen state |
| Canonical data-ledger cursor, batch IDs and mask hashes | G1/G2 → Checkpoint + replay owner | N identical batches after independent fresh-process CPU restart |
| Kernel and numerical policy | G5/G6 → Checkpoint + launcher | Pinned precision and RNG plan, no unapproved KQ/clip/LR changes |
| Health schema and alert/sampling rules | G8 → Health owner | JSON-schema fixtures, numerical vs hardware incident separation |
| Durable checkpoint events + eval queue | G7 → G9/G12 | Eval can only observe manifest-sealed committed generation |
| GPU topology and object-store/IAM details | TSAI operators → Queue + GPU owners | No credentials in Git; real node run in Week 2 |

### Added acceptance probes

1. CPU A/B trace: identical next N batch identifiers, LR, RNG and every continuation loss bit pattern.
2. Corruption and stale-writer fixtures: partial generation never advanced to `latest`.
3. Numerical incident: injected grad-norm/NaN anomaly creates diagnostic and **does not drain** a healthy mock node.
4. Pinned checkpoint generation automatically enqueues but does not block a stub held-out evaluation.
5. Security review: `gitleaks detect --source .`, manifest contains no secrets or raw sensitive data.
6. Benchmark plan: instantiate the actual final *untrained* model shape and optimizer to measure checkpoint stages on Week-2 GPU.
7. Real NCCL, DCGM and interconnect throughput explicitly remain **Week-2 evidence**, not Week-1 claims.

### Gate deadlines and escalation

27 Oct: personal diagnosis partial label milestone (tracked separately from G7); queue ownership and G5 config-v0 handoff. 29 Oct: G8 health JSON handoff. 31 Oct: Week-1 gate. Later gates: Nov 14, 21 and 28. If the spoken transcript's holiday phrasing conflicts with the CAP plan, request instructor confirmation; do not update shared schedules unilaterally. **Only the team lead/instructor can approve ownership and ADR status.**
