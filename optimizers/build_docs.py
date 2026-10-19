from pathlib import Path
import json,csv
p=Path(__file__).parent
r=json.loads((p/'results.json').read_text())
a=list(csv.DictReader(open(p/'adam_hand_vs_pytorch.csv')))
rows='\n'.join(f"| {i['step']} | {float(i['gradient']):+.3f} | {float(i['m_hand']):+.9f} | {float(i['v_hand']):.9f} | {float(i['mhat_hand']):+.9f} | {float(i['vhat_hand']):.9f} | {float(i['update_hand']):+.9f} | {float(i['weight_hand']):.9f} |" for i in a)
s=r['schedules'];w=r['width_sweep'];adam=r['adam'];bias=r['bias_correction']
pilot='\n'.join(f"| {lr:g} | {s['pilot_mean_loss']['cosine'][str(lr)]:.6f} | {s['pilot_mean_loss']['wsd'][str(lr)]:.6f} |" for lr in r['experiment']['lr_grid'])
widths='\n'.join(f"| {width} | {w['minima_on_tested_grid'][str(width)]['lr']:g} | {w['minima_on_tested_grid'][str(width)]['mean_val_loss']:.6f} |" for width in (256,512,1024))
schedule='\n'.join(f"| {name} | {s['independent_best_lr'][name]:g} | {s['step200_val_loss_by_seed'][name]['23']:.6f} | {s['step200_val_loss_by_seed'][name]['47']:.6f} | {s['step200_mean_val_loss'][name]:.6f} |" for name in ('cosine','wsd'))
writeup=f'''# ERA V5 — Session 11: results and interpretation

**Execution:** CPU-only PyTorch {r['environment']['torch']}; actual measured experiment; original deterministic synthetic token grammar (not a web corpus); two seeds 23 and 47. All losses are held-out cross-entropy unless noted. To reproduce, run `python experiment.py` and `python -m unittest -v test_experiment.py`.

## 1. Adam, calculated independently by hand

Fixed scalar weight `w0=0.7`, gradients `[+0.4, -0.2, +0.1, 0.0, -0.3]`, `β1=0.9`, `β2=0.999`, `α=0.01`, `ε=1e-8`, weight decay **zero** for this check. Arithmetic is computed using Python `Decimal` at 55-digit precision, completely separately from the PyTorch optimizer. PyTorch `torch.optim.Adam` provides the cross-check in Float64. A negative update means **w increases** because `w = w − update`.

| Step | Gradient | m | v | m̂ | v̂ | α m̂/(√v̂+ε) | weight after step |
|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

At each of all five steps, the manual value is checked against PyTorch for `m`, `v`, `m̂`, `v̂`, the update, and the resulting parameter. Maximum absolute mismatch across those **30 comparisons**: **{adam['max_abs_error']:.3e}**. The bias-corrected moments are reconstructed from PyTorch’s documented `exp_avg` and `exp_avg_sq` states.

**Formula:** `m_t=β1 m_(t−1)+(1−β1)g_t`; `v_t=β2 v_(t−1)+(1−β2)g_t²`; `m̂_t=m_t/(1−β1^t)`; `v̂_t=v_t/(1−β2^t)`; `w_t=w_(t−1)−α·m̂_t/(√v̂_t+ε)`.

## 2. Bias correction ablation

A deliberately controlled **constant** gradient `g=0.4` isolates the first-moment/second-moment zero-initialization bias. This is not a learned-model comparison. The graph `bias_correction_20_steps.png` shows the first 20 optimizer steps. With a persistent threshold defined as **relative difference in per-step update magnitude**, not difference between accumulated weights:

- Under **5%** for every subsequent step (through the checked step 16,000): **step {bias['thresholds_for_persistent_relative_per_step_update_difference']['5%']:,}**.
- Under **1%** for every subsequent step: **step {bias['thresholds_for_persistent_relative_per_step_update_difference']['1%']:,}**.
- At step 1, relative update discrepancy is **{bias['step_1_relative_difference']*100:.2f}%**; at step 20, **{bias['step_20_relative_difference']*100:.2f}%**.

**Critical:** Even once updates are close, accumulated *weight trajectories* can remain different. The phrase “difference stops mattering” needs an operational threshold; it does not mean exact equivalence. Bias correction is separate from decoupled weight decay (AdamW).

## 3. Layer update-to-weight ratios and warmup

Model has **six parameterized modules**: token embedding, learned positional embedding, normalization, bottleneck input, bottleneck output, output head. Each step logs `sqrt(Σ||θ_after−θ_before||²)/max(sqrt(Σ||θ_before||²),1e−12)` with norms calculated at pre-update values. See `layer_update_ratios.csv` (30 updates × 6 layers) and `update_to_weight_ratios.png`.

**Warmup ends at optimizer step {r['warmup']['ends_at_optimizer_step']}** and **the scheduled LR no longer increases beginning with step {r['warmup']['lr_plateau_starts']}** for WSD. The ratios **do not become constant at step 21**: the gradients, Adam moments, and parameter magnitudes still change. This is why the graph includes a vertical warmup boundary rather than claiming a flat ratio.

## 4. Cosine versus WSD, 300 planned / stop at step 200

Both models have **identical architecture and initial weights for a given seed** and consume exactly the same training sequences in the same order. Both use AdamW with fixed `β=(0.9,0.999)`, `eps=1e−8`, decoupled decay 0.01, warmup 20 updates, and **300-step planned horizon**. At 200 they are deliberately **stopped**, not trained to step 300.

Cosine: warmup→cosine annealing down to 10% peak at step 300. WSD: warmup→constant peak through step 240→linear decay to 10% at step 300. The WSD decay **has not started at step 200**, so this measures checkpoint quality at that time, *not* ultimate WSD merit.

### Independent pilot tuning: 120 steps, both seeds

Both schedules separately search the same six LR values. Tuning is selected on mean held-out loss at **step 120** (a limited pilot, not exhaustive schedule tuning):

| Peak LR | Cosine pilot loss | WSD pilot loss |
|---:|---:|---:|
{pilot}

### Evaluation at step 200 (two seeds)

| Schedule | Tuned peak LR | Seed 23 | Seed 47 | Mean held-out loss |
|---|---:|---:|---:|---:|
{schedule}

**Keep at the forced step-200 checkpoint:** **{s['would_keep_at_step200'].upper()}**, because its mean held-out loss is smaller in this experiment (**{min(s['step200_mean_val_loss'].values()):.6f} vs {max(s['step200_mean_val_loss'].values()):.6f}**). Do **not** claim cosine is inherently better at 300 steps: the experiment deliberately stops WSD *before decay*.

## 5. Width 256, 512, 1,024: controlled LR sweep

Same architecture, context 12, vocabulary 32, bottleneck 64, batch 16, 55 optimizer updates per trial, same warmup (20), constant LR after warmup, AdamW settings and exactly the same two initialization seeds and token batch sequence. Mean held-out cross-entropy is used to select the lowest **measured grid value**, not the continuous optimum.

| Model hidden width | LR with lowest mean held-out loss among tested values | Mean held-out loss |
|---:|---:|---:|
{widths}

The figure `lr_sweep_widths.png` marks those three grid minima. Grid: `{', '.join(str(x) for x in r['experiment']['lr_grid'])}`. The 256 optimum shifts from the 512/1024 optimum: **there is no evidence that these LRs transfer unchanged across all widths in this ordinary parameterization.**

**Width 4,096 provisional starting candidate:** **{w['candidate_at_4096']:g}**. **Confidence: {w['confidence'].upper()}**. This width **was not tested**. The architecture uses ordinary PyTorch initialization (standard parameterization), **not maximal-update parameterization (μP)**. Claiming μTransfer from this experiment would be scientifically incorrect. Proper next step is a 4,096-width sweep with stronger seeds and a documented μP implementation for genuine width-transfer study. This is a candidate for further testing, **not a validated optimum**.

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
'''
(p/'WRITEUP.md').write_text(writeup)
readme=f'''# ERA V5 — Session 11 | Optimizer Laboratory

**Status: executed on CPU, reproducible, six tests passing.** Built from the attached Session 11 transcript and the exact five-part assignment. This is a demonstrator of optimizer mathematics, not a frontier model or proof of large-model scaling.

## Submission map

| Required element | Code | Evidence |
|---|---|---|
| 1. Adam by hand against PyTorch | `hand_adam()` | `adam_hand_vs_pytorch.csv`, notebook printed table, all 30 values asserted |
| 2. Bias correction off, first 20 + convergence threshold | `bias_experiment()` | `bias_20_steps.csv`, `bias_correction_20_steps.png`, `results.json` |
| 3. Per-layer update/weight ratios; warmup boundary | `run_model(track=True)` | `layer_update_ratios.csv`, `update_to_weight_ratios.png` |
| 4. Cosine & WSD 300 planned, stop 200; separately tune | `lr_at()`, `run_all()` | `schedule_tuning.csv`, `schedule_final_histories.csv`, `cosine_vs_wsd_stop200.png` |
| 5. Width sweep & 4,096 confidence | `run_all()` | `width_lr_sweep.csv`, `lr_sweep_widths.png`, `WRITEUP.md` |

## Run locally

Python 3.10+ recommended; PyTorch, NumPy, Matplotlib.

```bash
python -m pip install -r requirements.txt
python experiment.py
python -m unittest -v test_experiment.py
```

To run in Colab, upload `ERA_V5_S11_Optimizer_Lab.ipynb` and select **Runtime → Run all**. Notebook is **self-contained** and uses only preinstalled scientific packages; it prints findings and displays measured plots. No GPU, account, API key, downloaded dataset or internet access is required once packages are available. Runs in approximately **{r['elapsed_seconds']:.1f} seconds on this CPU environment**, excluding notebook startup overhead; other machines may vary.

All generated results are written to the current working directory in Colab or next to `experiment.py` locally (can override with `ERA_S11_OUT`). A shorter **sanity-only** run is `python experiment.py --fast`; those reduced-budget numbers are **not** the submission's reported measurements.

## Numerical definitions and reproduction protocol

- Fixed seed pairs 23/47; NumPy, Torch, Python RNG reseeded for every trial. No nondeterministic GPU kernels used (all measurements CPU).
- All LR trials use the **same per-step training-example index generator**. Training and held-out sets are produced from **different seeds with unique disjoint full sequences**, established by tests.
- Model has width-dependent token + position embeddings, one causal previous-token mixer, RMS? **No:** standard PyTorch LayerNorm and bottleneck (64) residual MLP, then separate vocabulary head. No Transformer self-attention and no μP; named truthfully in `experiment.py`.
- All losses are **cross entropy on held-out synthetic sequences**, fixed and reused across methods.
- Step is **one forward, backward and optimizer update**; full sequence input predicts next token `target[:,1:]`. No padding.
- Corrected Adam scalar test uses **zero** weight decay and Float64 PyTorch reference to isolate exactly the original Adam formula. The training experiments use **AdamW with decoupled weight decay 0.01**.
- Parameter group ratios use post-update difference divided by the pre-update parameter norm. Logs include the exact scheduled LR at every step.
- Warmup steps 1–20 reach peak LR; first full stable-rate step is 21. **Warmup ends, not the evolution of the update/weight ratio.**
- The scheduled WSD decay begins **at step 241**, after plateau through step 240. Thus the forced checkpoint at step 200 is before any WSD cooldown.
- Schedule peak LR is tuned **independently** across the six-grid values using both seeds over 120 updates, then fresh runs stop at 200. Both schedules use identical datasets, seeds, model size, optimizer, batch sizes and parameter initializations for their matching seeds.
- Width sweep holds bottleneck and all other design parameters fixed, uses 55 updates at width 256/512/1,024 across both seeds. The 4,096 answer is a **low-confidence untested hypothesis**, not a transferred optimum.

## Selected numerical conclusions

- Adam maximum discrepancy to PyTorch: **{adam['max_abs_error']:.3e}**.
- No-bias-correction *per-update* persistent 1% threshold (constant gradient, defined test): step **{bias['thresholds_for_persistent_relative_per_step_update_difference']['1%']:,}**.
- Cosine at step 200: **{s['step200_mean_val_loss']['cosine']:.6f}**, WSD: **{s['step200_mean_val_loss']['wsd']:.6f}** (mean over two seeds).
- LR grid minimum for width 256: **{w['minima_on_tested_grid']['256']['lr']}**; width 512: **{w['minima_on_tested_grid']['512']['lr']}**; width 1,024: **{w['minima_on_tested_grid']['1024']['lr']}**.
- Tentative LR at 4,096: **{w['candidate_at_4096']}**, **very low confidence**.

Read [WRITEUP.md](WRITEUP.md) for the full per-step hand calculation and limitations; [ACCURACY_AUDIT.md](ACCURACY_AUDIT.md) for discrepancies between transcript explanations and tested mathematical definitions.

## Original sources

1. PyTorch Adam documentation: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
2. Adam (2014): https://arxiv.org/abs/1412.6980
3. Decoupled Weight Decay Regularization (AdamW): https://arxiv.org/abs/1711.05101
4. MiniCPM and WSD: https://arxiv.org/abs/2404.06395
5. WSD river-valley interpretation: https://arxiv.org/abs/2410.05192
6. μTransfer (requires μP): https://arxiv.org/abs/2203.03466

## Auditability

All CSV files are generated by code rather than typed manually. `results.json` contains hardware, seeds, hyperparameters, pilot/final losses and the exact confidence qualifier. Use `python -m unittest -v test_experiment.py` to verify manual Adam parity, no future-token leakage, genuinely disjoint train/validation sequences, LR boundaries, reproducible initializations and valid ratios.

**GitHub status:** this is a GitHub-ready source bundle; no repository has been created or published automatically.
'''
# correct unneeded question about RMS
readme=readme.replace('one causal previous-token mixer, RMS? **No:** standard PyTorch LayerNorm','one causal previous-token mixer, standard PyTorch LayerNorm')
(p/'README.md').write_text(readme)
(p/'ACCURACY_AUDIT.md').write_text('''# Accuracy audit — Session 11

## Grounded in the transcript
The uploaded Session 11 transcript (5 Sep 2026) discusses Adam's moments, warmup, cosine, WSD and μP, and ends with this exact assignment. Numerical optimizer coefficients and schedule boundaries used here are explicit **experimental selections** rather than values claimed to be mandated by the transcript.

## Source comparison and corrections

1. **Bias correction vs AdamW:** Near the end the transcript describes bias correction as what AdamW does to avoid being affected by weight decay. These are distinct mechanisms. Adam corrects the zero-initialization biases in exponential moving averages (`1−β1^t` and `1−β2^t`); AdamW decouples weight decay from the gradient moments. Checked against official PyTorch Adam API and AdamW paper. This experiment does not silently repeat the conflation.
2. **Second moment is not physical acceleration nor a second derivative.** Adam's `v_t` is an exponential moving average of **squared gradients**. It does not compute `d²L/dw²` or acceleration in the calculus sense.
3. **Warmup end vs ratio stabilization:** A flat scheduled peak LR after step 20 does not guarantee the actual update-to-weight ratios flatten. All six ratios are logged explicitly.
4. **WSD at step 200:** With the experiment's 300 planned steps and decay at 240, there is *no WSD decay before the requested stop.* Comparing held-out losses at step 200 therefore does not justify claims about the performance of the completed WSD schedule.
5. **μP versus ordinary width scaling:** The transcript highlights μTransfer; the experiment uses ordinary PyTorch parameterization for the requested LR sweep. The 4,096 LR is explicitly unvalidated. Genuine μP claims require μP parameterization and a separate width transfer experiment.
6. **Dataset overlap:** A first test of an earlier synthetic generator caught identical train/validation full sequences. That test failed. The generator was corrected to enforce unique, disjoint sequences; all measurements were fully rerun. Only post-fix results are submitted.
7. **No unearned generalization:** Two seeds, 55/120/200 steps and synthetic 32-symbol grammar cannot establish a globally optimal optimizer or learning rate for production LLMs.

## Verification sources
- Official PyTorch Adam: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
- Adam (Kingma & Ba): https://arxiv.org/abs/1412.6980
- AdamW (Loshchilov & Hutter): https://arxiv.org/abs/1711.05101
- WSD MiniCPM: https://arxiv.org/abs/2404.06395
- μP Tensor Programs V: https://arxiv.org/abs/2203.03466
''')
(p/'requirements.txt').write_text('torch>=2.2,<3\nnumpy>=1.24,<3\nmatplotlib>=3.7,<4\nnbformat>=5.8,<6\n')
print('docs',[(f.name,len(f.read_bytes())) for f in p.glob('*.md')])
