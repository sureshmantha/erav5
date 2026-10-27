# ERA V5 G7 — Trainer-aligned architecture correction (Revision 4)

Start with `00_TRAINER_INTENT_V5_VS_V4.md` → `07_CONFIG_DRIVEN_MODEL_STAGE_CAPACITY.md` → Master Guide Rev4 Section 4 → six revised workstream documents.

This pack **replaces the previous arbitrary 1-trillion-parameter sizing baseline** with source-backed V5 programme goals and an explicit G5-dependent capacity contract.

**Main conclusion:** V4's published 118.67B/5.93B-active LightningLM, with 4 growth stages, is a historical benchmark. Session 15 says V5 intends *three models with sizes withheld*. Session 6 rules out starting from V4 checkpoint weights. V5 model/hardware/tokens are not yet specified and must never be guessed.

This package is a **design update, not a cluster acceptance certificate**. Existing rev3 originals are unchanged in their prior version. Week 1 remains local/CPU design and demonstrable tests; CUDA/NCCL on allocated GPU nodes is for later validated runs.
