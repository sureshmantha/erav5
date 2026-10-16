# Session 10 — Assignment write-up (measured CPU reference)

**Status:** All notebook cells ran successfully. This document records measured values, distinguishes estimates from measurements, and does not claim GPU MFU.

## 1. Every tensor shape

The notebook prints the complete forward tensor list and every trainable parameter shape, corresponding backward-gradient shape and AdamW optimizer-state tensor shape. In this tested configuration:

- `B=2` sequences, `T=5` tokens in the displayed short micro-batch; `C=32`, `H=4`, `Dh=8`, `V=120`.
- Tokens `[B,T]` = `(2,5)`; embedding `[B,T,C]` = `(2,5,32)`; Q/K/V `[B,H,T,Dh]` = `(2,4,5,8)`; scores `[B,H,T,T]` = `(2,4,5,5)`; logits `[B,T,V]` = `(2,5,120)`.
- Loss is scalar `[]`; gradients match their respective parameter tensor dimensions.

## 2. Backprop versus finite difference

- Parameter: `output_head weight[1,1]`, ε=1e-05 (`float64`).
- `backward()` derivative: **-0.000727148747**.
- Hand/finite-difference derivative: **-0.000727148697**.
- **Absolute difference 4.971e-11; relative difference 6.837e-08. PASS.**
- Central difference held every other weight fixed and did not step the optimizer.

## 3. Deliberately broken accumulation — actual training

- Micro-batch A: **8** valid next-token targets; micro-batch B: **34**.
- Correct weights: `8/(8+34)` and `34/(8+34)` = **0.190 / 0.810**.
- Wrong weighting uses **0.5 / 0.5**. That overweights shorter sequences.
- Full batch versus correct accumulated gradient, maximum component difference: **3.725e-08**.
- Full batch versus wrong accumulated gradient, maximum component difference: **0.258932**.
- After 55 AdamW steps, fixed training-corpus CE: correct **0.289159**; incorrect **0.315906**. The curves are in `accumulation_loss_curves.png`.
- This does **not** imply all workloads will show the same-sized difference.

## 4. Gradient norm before loss moves

- Every optimizer step records both the pre-update global L2 gradient norm and fixed-corpus loss.
- Predeclared observation threshold: absolute fixed-corpus CE change ≤ **0.005**, relative grad-norm change ≥ **10%**.
- Observed event: **step 24**; loss absolute change **0.004932105541229248**; gradient-norm relative change **34.30938369894329%**. This is a *threshold contrast*, not proof that the gradient foretold the next loss.
- Separate analytic fixed-weight probe at `w=0` with `(x=1,y=1)` versus `(x=4,y=1)` held MSE **1.0** and changed grad norm **2 → 8**, showing that identical loss need not mean identical sensitivity.

## 5. Measured compute / MFU

- Model: **16,672** total parameters, **12,256** non-embedding parameters by the declared convention.
- CPU-only benchmark: **42** valid predicted tokens/update, **25** updates, **36,019.11** useful tokens/s measured in **0.029151 s** (timing is hardware-dependent).
- Useful model arithmetic estimate `6 N × tokens/s`: **2.649 GFLOP/s**.
- Independently measured best local CPU FP32 matmul rate: **195.305 GFLOP/s**.
- Experimental **CPU reference ratio = 1.356%** — *NOT standard GPU MFU*.
- **GPU MFU: N/A.** The executed environment has no CUDA GPU. Thus neither a real MFU percentage nor a defensible quantified “distance to 40%” can be reported yet. The Colab CUDA cell will calculate it after receiving a verified dense FP32 FLOP/s denominator.
- Hypotheses, not tested attributions: small GEMMs, Python launch overhead, memory movement, softmax/layer norm, and optimizer work. Each requires independent profiling before claiming it explains a measured fraction of the gap.

## 6. Decimal 0.1 in three formats (round to nearest-even)

| Format | Sign / exponent / fraction | Hex | Stored value | Absolute error |
|---|---|---|---|---|
| FP32 | `0 / 01111011 / 10011001100110011001101` | `0x3DCCCCCD` | 0.10000000149011612 | 1.49011611383e-09 |
| BF16 | `0 / 01111011 / 1001101` | `0x3DCD` | 0.10009765625000000 | 9.765625e-05 |
| FP8 E4M3FN | `0 / 0011 / 101` | `0x1D` | 0.10156250000000000 | 0.0015625 |

`0.1 = 1.6 × 2^(-4)`; binary fractional digits repeat, and all three formats approximate it. Exponent bias is 127 for FP32/BF16, 7 for E4M3FN. FP8 decoded value is `1.625 × 2^(-4) = 0.1015625`.

**Training decision:** The completed CPU baseline used FP32. For compatible modern GPUs, test **BF16 computation with FP32 optimizer/master states** first; BF16 retains eight exponent bits and offers a wider range than FP16, though less mantissa precision than FP32. FP8 requires validated scaling and kernels and was *not* used in this training. It is not safe to infer training quality from a single 0.1 conversion.

## Additional audited corrections

- The Session 10 transcript briefly describes accumulated training as `X forwards + 1 backward`, but later corrects that to `X forwards + X backwards + 1 optimizer update`. The code implements the corrected account.
- The transcript's reported `9B params × 12k tokens/s × 6` corresponds to **648 TFLOP/s**, not “6,848 TFLOP/s.” `648 / 7,912 = 8.19%`, consistent with its stated 8.2% when 7,912 TFLOP/s is taken as the aggregate peak (roughly eight H100 dense BF16 tensor-core devices). These transcript figures are not measurements from this notebook.
