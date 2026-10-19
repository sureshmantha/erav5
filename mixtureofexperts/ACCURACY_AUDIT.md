# Session 14 — Accuracy and Source Audit

## Verified locally

1. Seven tests passed, including held-out record disjointness, causal attention, shifted byte targets, function-preserving conversion, unselected-expert gradients being absent, router usage, and initial loss close to uniform vocabulary.
2. SHA-256 of 300-row cleaned academic sample: `41f311e4db037900aac621b1850221c15d1c5431c49a0b14fdb554ca7b8b8e9c`.
3. Full measured run `artifacts/verified/results.json`: dense 60 updates, MoE 90 updates, CE 5.566130→3.376164→3.314345. Pre-conversion and immediate post-conversion CE agree to displayed precision. Maximum logit difference 1.192093e-6.
4. Independent dense continuation `artifacts/verified/controls.json`: 90 additional updates, 3.376164→3.314582; same checkpoint, batches, evaluation set and LR; both arms start with reset optimizer state.
5. All six experts received tokens in both layers; actual counts in `artifacts/verified/results.json`.
6. Active parameter count is a storage-independent static count of parameter elements referenced by the active forward *path* at a token. It is not a measured FLOP, activation byte, or real speed-up claim.

## Corrections made during development

- **Untrained output head:** An early pilot initialized tied embeddings with a standard-normal scale, causing loss ≫ log(vocab). Corrected to a small normal initialization and **reran the final 60+90 experiment**. Earlier pilot results in `artifacts/main/` are deliberately excluded from the final submission; this folder is not packaged.
- **Router load:** A pilot without proper initialization produced uneven routing. Final experiment reports actual utilization, does not assert ideal balance.
- **Schedule controls:** A wider three-arm auxiliary-loss sweep was interrupted by execution-time limits. It is **not presented as a completed result**. The finalized experiments include the full MoE run and a completed matched dense-only control.
- **Scope:** The instructor's Session 14 final assignment asks for a dense model converted to MoE with falling loss, allowing student choice of size/data. We implemented a small transformer to target its FFN, not 20M/50M from Session 13.

## Distinguish lecture from proven statements

- The lecture describes experts as often specializing in *language constructs* rather than topic labels. Our experiment did not measure learned specialization; no specific expert is labeled “coding”, “mathematics”, or “science.”
- The lecture explores multiple routing score types (softmax, sigmoid, softplus), shared experts, capacity limits and balance algorithms. Our run only measures **top-2 selected-logit softmax**, **no shared expert**, **dropless dispatch** and a documented load auxiliary.
- The lecture discusses modern MoE performance figures, hardware interconnect, and large-scale training. None are outcomes of this tiny CPU experiment. Published headline speedup numbers are not projected onto this configuration.
- `N×Σ f_i P_i` balances *assignments* and routing probabilities but does not guarantee equal expert counts or prevent all collapse scenarios.

## Exact sources

- Session 14 transcript supplied by user, **2026-09-26**, especially FFN replacement at approximately 00:05–00:15, router and weighted sum around 01:00–01:05, balancing around the later router section, and assignment at approximately 02:10.
- Sparse Upcycling (2022): https://arxiv.org/abs/2212.05055
- Switch Transformer (2021): https://arxiv.org/abs/2101.03961
- DeepSeekMoE (2024): https://arxiv.org/abs/2401.06066
- Data license notices in `README.md`.

## Claims deliberately NOT made

- No claim to an 20M LLM or 50M tokens (Session 13 requirement).
- No claim of real multi-GPU, expert parallelism or peak VRAM savings.
- No claim of higher quality versus the dense continuation, since the measured final validation difference is 0.000237 CE nats/token from one seed.
- No claim that model behavior or expert specialization generalizes outside the 300-sample educational dataset.
- No claim of a published GitHub repo until it exists.

## Notebook execution provenance

- `ERA_V5_S14_EXECUTED.ipynb` was produced by a **completed earlier top-to-bottom execution** (7 executed code cells, 0 code-cell errors). The CPU training checkpoints, fixed held-out results and plots from this same experiment are independently available in `artifacts/verified/`.
- After that successful execution, the unexecuted Colab template `ERA_V5_S14_Dense_to_MoE.ipynb` received **only a conditional-install safety guard** for missing standard Python dependencies. A second attempt to execute the revised template failed while starting the notebook kernel in this container; **there is no claim that the revised template was re-executed**. The training/evaluation source did not change between those two notebook versions.
