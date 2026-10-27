# Revision 4 change log — correct G7 scope to trainer's ERA V5 plan

Date: 2026-10-27. Source basis: original Sessions 3, 6, 14, 15, 16; published LightningLM V4 paper; official ERA V5 course statement; current G7 plan.

## The correction

**Removed** the 1T all-trainable scenario as the architecture's primary planning case. **Added** an explicit distinction between (a) V4's documented four-stage growth lineage, (b) V5's instructor-stated three-model intent whose sizes are undisclosed, (c) homework hypotheticals such as 40B, and (d) conditional data ambitions/efficiency claims from Sessions 14 and 16.

## Files

- `00_TRAINER_INTENT_V5_VS_V4.md` NEW: evidence-tiered comparison and unknowns.
- `07_CONFIG_DRIVEN_MODEL_STAGE_CAPACITY.md` NEW: required config and calculation procedures for G5-provided sizes.
- `G7_INFRASTRUCTURE_ENGINEERING_MASTER_GUIDE_REV4.md`: replaced Section 4 size assumption, added Rev4 scope note, revised Section 27.5, removed erroneous 1T labels from examples.
- `01–06_*.md`: scope header and cleanup of unsupported 1T projections; other Week 1 controls preserved.

## Unchanged guarantees

Two Slurm accounting accounts; real process rank maps; truthful GPU and NCCL health, JSON with `skipped`; CPU bit-equal replay; committed-checkpoint integrity/fencing; ledger state; negative fault tests; gitleaks scan. No GPU-node benchmark, Slurm Docker acceptance or production-scale checkpoint claim is implied by this documentation revision.

## Caveats

Session 15 mentions *three models*, but it does **not** state whether they are three independent models or consecutive scale stages. The exact V5 backbone, total/active/trainable parameters, planned tokens, and GPU topology remain open. The V4 paper also distinguishes a planned corpus inventory from actual token consumption. A claimed 16× efficiency concept was not evaluated for this revision.
