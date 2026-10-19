# ERA V5 — Session 11: results and interpretation

**Execution:** CPU-only PyTorch 2.10.0+cpu; actual measured experiment; original deterministic synthetic token grammar (not a web corpus); two seeds 23 and 47. All losses are held-out cross-entropy unless noted. To reproduce, run `python experiment.py` and `python -m unittest -v test_experiment.py`.

## 1. Adam, calculated independently by hand

Fixed scalar weight `w0=0.7`, gradients `[+0.4, -0.2, +0.1, 0.0, -0.3]`, `β1=0.9`, `β2=0.999`, `α=0.01`, `ε=1e-8`, weight decay **zero** for this check. Arithmetic is computed using Python `Decimal` at 55-digit precision, completely separately from the PyTorch optimizer. PyTorch `torch.optim.Adam` provides the cross-check in Float64. A negative update means **w increases** because `w = w − update`.

| Step | Gradient | m | v | m̂ | v̂ | α m̂/(√v̂+ε) | weight after step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.400 | +0.040000000 | 0.000160000 | +0.400000000 | 0.160000000 | +0.010000000 | 0.690000000 |
| 2 | -0.200 | +0.016000000 | 0.000199840 | +0.084210526 | 0.099969985 | +0.002663370 | 0.687336630 |
| 3 | +0.100 | +0.024400000 | 0.000209640 | +0.090036900 | 0.069949980 | +0.003404291 | 0.683932339 |
| 4 | +0.000 | +0.021960000 | 0.000209431 | +0.063855772 | 0.052436232 | +0.002788588 | 0.681143750 |
| 5 | -0.300 | -0.010236000 | 0.000299221 | -0.024995727 | 0.059964026 | -0.001020752 | 0.682164503 |

At each of all five steps, the manual value is checked against PyTorch for `m`, `v`, `m̂`, `v̂`, the update, and the resulting parameter. Maximum absolute mismatch across those **30 comparisons**: **1.457e-15**. The bias-corrected moments are reconstructed from PyTorch’s documented `exp_avg` and `exp_avg_sq` states.

**Formula:** `m_t=β1 m_(t−1)+(1−β1)g_t`; `v_t=β2 v_(t−1)+(1−β2)g_t²`; `m̂_t=m_t/(1−β1^t)`; `v̂_t=v_t/(1−β2^t)`; `w_t=w_(t−1)−α·m̂_t/(√v̂_t+ε)`.

## 2. Bias correction ablation

A deliberately controlled **constant** gradient `g=0.4` isolates the first-moment/second-moment zero-initialization bias. This is not a learned-model comparison. The graph `bias_correction_20_steps.png` shows the first 20 optimizer steps. With a persistent threshold defined as **relative difference in per-step update magnitude**, not difference between accumulated weights:

- Under **5%** for every subsequent step (through the checked step 16,000): **step 2,375**.
- Under **1%** for every subsequent step: **step 3,925**.
- At step 1, relative update discrepancy is **216.23%**; at step 20, **524.09%**.

**Critical:** Even once updates are close, accumulated *weight trajectories* can remain different. The phrase “difference stops mattering” needs an operational threshold; it does not mean exact equivalence. Bias correction is separate from decoupled weight decay (AdamW).

## 3. Layer update-to-weight ratios and warmup

Model has **six parameterized modules**: token embedding, learned positional embedding, normalization, bottleneck input, bottleneck output, output head. Each step logs `sqrt(Σ||θ_after−θ_before||²)/max(sqrt(Σ||θ_before||²),1e−12)` with norms calculated at pre-update values. See `layer_update_ratios.csv` (30 updates × 6 layers) and `update_to_weight_ratios.png`.

**Warmup ends at optimizer step 20** and **the scheduled LR no longer increases beginning with step 21** for WSD. The ratios **do not become constant at step 21**: the gradients, Adam moments, and parameter magnitudes still change. This is why the graph includes a vertical warmup boundary rather than claiming a flat ratio.

## 4. Cosine versus WSD, 300 planned / stop at step 200

Both models have **identical architecture and initial weights for a given seed** and consume exactly the same training sequences in the same order. Both use AdamW with fixed `β=(0.9,0.999)`, `eps=1e−8`, decoupled decay 0.01, warmup 20 updates, and **300-step planned horizon**. At 200 they are deliberately **stopped**, not trained to step 300.

Cosine: warmup→cosine annealing down to 10% peak at step 300. WSD: warmup→constant peak through step 240→linear decay to 10% at step 300. The WSD decay **has not started at step 200**, so this measures checkpoint quality at that time, *not* ultimate WSD merit.

### Independent pilot tuning: 120 steps, both seeds

Both schedules separately search the same six LR values. Tuning is selected on mean held-out loss at **step 120** (a limited pilot, not exhaustive schedule tuning):

| Peak LR | Cosine pilot loss | WSD pilot loss |
|---:|---:|---:|
| 0.0001 | 3.378975 | 3.351035 |
| 0.0003 | 2.950809 | 2.914924 |
| 0.001 | 2.235793 | 2.138345 |
| 0.003 | 1.267344 | 1.262407 |
| 0.01 | 1.288101 | 1.369054 |
| 0.03 | 1.703708 | 1.867834 |

### Evaluation at step 200 (two seeds)

| Schedule | Tuned peak LR | Seed 23 | Seed 47 | Mean held-out loss |
|---|---:|---:|---:|---:|
| cosine | 0.003 | 1.088080 | 1.075632 | 1.081856 |
| wsd | 0.003 | 1.174730 | 1.152276 | 1.163503 |

**Keep at the forced step-200 checkpoint:** **COSINE**, because its mean held-out loss is smaller in this experiment (**1.081856 vs 1.163503**). Do **not** claim cosine is inherently better at 300 steps: the experiment deliberately stops WSD *before decay*.

## 5. Width 256, 512, 1,024: controlled LR sweep

Same architecture, context 12, vocabulary 32, bottleneck 64, batch 16, 55 optimizer updates per trial, same warmup (20), constant LR after warmup, AdamW settings and exactly the same two initialization seeds and token batch sequence. Mean held-out cross-entropy is used to select the lowest **measured grid value**, not the continuous optimum.

| Model hidden width | LR with lowest mean held-out loss among tested values | Mean held-out loss |
|---:|---:|---:|
| 256 | 0.01 | 1.626736 |
| 512 | 0.003 | 1.794459 |
| 1024 | 0.003 | 1.739674 |

The figure `lr_sweep_widths.png` marks those three grid minima. Grid: `0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03`. The 256 optimum shifts from the 512/1024 optimum: **there is no evidence that these LRs transfer unchanged across all widths in this ordinary parameterization.**

**Width 4,096 provisional starting candidate:** **0.003**. **Confidence: VERY LOW**. This width **was not tested**. The architecture uses ordinary PyTorch initialization (standard parameterization), **not maximal-update parameterization (μP)**. Claiming μTransfer from this experiment would be scientifically incorrect. Proper next step is a 4,096-width sweep with stronger seeds and a documented μP implementation for genuine width-transfer study. This is a candidate for further testing, **not a validated optimum**.

## Limitations

- Tiny causal residual **token mixer**, not an attention-based Transformer or a pretrained model. Demonstrates optimization mechanics, not LLM quality.
- **Synthetic and deterministic** 32-symbol sequence grammar; 2,048 unique training sequences and 512 unique validation sequences, with full-sequence overlaps explicitly prevented. Rules shared intentionally; no real-world data claim.
- Training uses only 200/55 updates in corresponding experiments. Two seeds reduce but do not remove variance; pilot schedule tuning is only at step 120.
- Fixed optimizer families, decay, model depth, context and batch; no assertion of global optimality.
- CPU wall time is not a GPU throughput result.
- WSD comparison is specifically **early stopping at 200**, not post-decay convergence at 300.

## Sources and audit

- **Attached Session 11 transcript (5 Sep 2026):** assignment framing, Adam, LR warmup, WSD/cosine discussion, μP rationale.
- **PyTorch Adam official docs:** https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html — equations and optimizer states.
- **Adam paper:** Kingma & Ba (2014), https://arxiv.org/abs/1412.6980.
- **AdamW:** Loshchilov & Hutter (2017), https://arxiv.org/abs/1711.05101 — weight decay is not bias correction.
- **WSD:** MiniCPM (Hu et al., 9 Apr 2024), https://arxiv.org/abs/2404.06395; WSD analysis (Wen et al., 7 Oct 2024), https://arxiv.org/abs/2410.05192.
- **μP:** Yang et al., Tensor Programs V (2022), https://arxiv.org/abs/2203.03466 — requires the special parameterization, which this assignment's base model does not implement.
