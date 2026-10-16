# ERA V5 • Session 10 — Loss, gradients, accumulation & precision

> **Execution status:** The notebook in this repository was executed end-to-end with **zero cell errors** in a CPU-only environment. GPU MFU is explicitly **N/A**. This is not a claim of a GPU speed test, model generalization, or production LLM readiness.

## Project files

| File | Purpose |
|---|---|
| `ERA_V5_S10_Gradients_Accumulation_MFU.ipynb` | Standalone, Colab-compatible notebook, no repo imports |
| `ERA_V5_S10_EXECUTED.ipynb` | Same notebook, with the local CPU reference outputs embedded |
| `session10_harness.py` | Plain Python / Jupytext-style executable source |
| `results.json` | Machine-produced measurements and 55 logged steps |
| `accumulation_loss_curves.png` | Correct versus deliberately wrong training comparison |
| `gradient_norm_and_loss.png` | Every-step gradient-norm monitoring versus fixed-corpus loss |
| `WRITEUP.md` | Concise graded-assignment answer including numeric results |
| `TRANSCRIPT_AUDIT.md` | Corrections and terminology grounded in Session 10 transcript |
| `requirements.txt` | Dependencies |

## Run in Google Colab

1. Open Colab → **File → Upload notebook** → select `ERA_V5_S10_Gradients_Accumulation_MFU.ipynb`.
2. Run **Runtime → Run all**. The notebook requires no network calls and installs nothing; Colab normally includes PyTorch, NumPy, and Matplotlib.
3. Observe the measured results, 55 logged optimizer updates, plots and assertions. The default CPU execution is intentional for deterministic cross-platform behavior.
4. To execute the *optional GPU benchmark*, choose a CUDA runtime and **manually provide the exact published dense FP32 GPU peak TFLOP/s** for the *GPU SKU actually allocated* as `GPU_PEAK_DENSE_TFLOPS_FP32`. Do not use a BF16, FP8, tensor-core or sparsity-assisted peak for this FP32 benchmark. The cell refuses to invent a peak value.
5. All generated files (`results.json`, PNGs) appear in the notebook working directory.

Also available: `python session10_harness.py` in a local environment with the requirements installed.

## Mathematical invariants / why these tests matter

- `L = CE(logits[:, :-1], tokens[:, 1:])`, ignoring padding targets.
- Manual derivative check: `dL/dw ≈ (L(w+ε)-L(w-ε))/(2ε)`, performed in `torch.float64` with ε=1e-5, no parameter update.
- For micro-batches with valid token counts `n_i`, the correct effective-batch CE is `Σ loss_sum_i / Σ n_i`. **Never** average independently reduced losses equally when `n_i` differ.
- A true optimizer update comprises all necessary forward/backward micro-batches, followed by *one* optimizer step, not a single backward call for the whole group.
- Pre-update global gradient norm is recorded on **every** step without clipping, so the observed signal is not hidden by clipping. Its change is not proof of an imminent loss spike.
- Simplified MFU numerator is `6 × nonembedding_parameter_count × useful_predicted_tokens/s`. This omits some attention/normalization/softmax costs and can be a poor approximation for a very small model. A true GPU MFU ratio needs a matching measured CUDA throughput and a sourced hardware peak for the computation dtype.

## Precision recommendation

The reference experiment trains in **FP32**. On appropriately supported accelerators, **BF16 compute with FP32 optimizer/master states** is the defensible default to test next (range similar to FP32, 7 fraction bits; much lower precision than FP32). FP8 E4M3FN needs appropriate scaling, kernel support, overflow/underflow safeguards and validation; no FP8 training run was performed here. Its *value representation* was independently validated, which is not the same as validating FP8 training.

## Reproducibility boundaries

- Fixed original 16-sentence toy corpus, fixed seed `20260829`, width 32, four heads and zero dropout. This is a **controlled, real optimizing loop**, not a general-purpose benchmark.
- Numerically deterministic within the reference environment. Minor differences across CPUs, PyTorch builds, kernel implementations or GPU environments may occur; exact timer-based performance is never deterministic.
- `torch.use_deterministic_algorithms(True)` protects against supported nondeterministic kernels, but is not a cross-hardware bitwise-reproduction guarantee.
- The CPU proxy denominator is a measured local FP32 matmul reference, **not vendor theoretical GPU FLOP/s**; it is deliberately not described as MFU.
- Both accumulation-loss plots evaluate the **same fixed training corpus**, not unseen validation data. These show optimization effects, not generalization performance.

## Sources and source-audit integrity

**Course:** `ERA V5 Session 10.docx`, Session 10, 2026-08-29, supplied with the assignment. Its account of gradient accumulation, precision and the rough 6N MFU estimate is used as course context. The separately documented arithmetic and wording corrections are in `TRANSCRIPT_AUDIT.md`.

**Primary documentation:**

- PyTorch floating data types: https://docs.pytorch.org/docs/stable/tensor_attributes.html
- PyTorch E4M3FN encoding (bias 7, 1/4/3 bits): https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_e4m3fn.h
- NVIDIA H100 architecture data and dense/sparse precision distinctions: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- PaLM large-scale MFU convention: https://arxiv.org/abs/2204.02311
- FP8 formats paper: https://arxiv.org/abs/2209.05433

No benchmark GPU peak, clock speed, model-quality gain or empirical bottleneck ranking was fabricated.
