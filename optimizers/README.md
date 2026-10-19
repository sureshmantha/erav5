# ERA V5 — Session 11 | Optimizer Laboratory

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

To run in Colab, upload `ERA_V5_S11_Optimizer_Lab.ipynb` and select **Runtime → Run all**. Notebook is **self-contained** and uses only preinstalled scientific packages; it prints findings and displays measured plots. No GPU, account, API key, downloaded dataset or internet access is required once packages are available. Runs in approximately **23.6 seconds on this CPU environment**, excluding notebook startup overhead; other machines may vary.

All generated results are written to the current working directory in Colab or next to `experiment.py` locally (can override with `ERA_S11_OUT`). A shorter **sanity-only** run is `python experiment.py --fast`; those reduced-budget numbers are **not** the submission's reported measurements.

## Numerical definitions and reproduction protocol

- Fixed seed pairs 23/47; NumPy, Torch, Python RNG reseeded for every trial. No nondeterministic GPU kernels used (all measurements CPU).
- All LR trials use the **same per-step training-example index generator**. Training and held-out sets are produced from **different seeds with unique disjoint full sequences**, established by tests.
- Model has width-dependent token + position embeddings, one causal previous-token mixer, standard PyTorch LayerNorm and bottleneck (64) residual MLP, then separate vocabulary head. No Transformer self-attention and no μP; named truthfully in `experiment.py`.
- All losses are **cross entropy on held-out synthetic sequences**, fixed and reused across methods.
- Step is **one forward, backward and optimizer update**; full sequence input predicts next token `target[:,1:]`. No padding.
- Corrected Adam scalar test uses **zero** weight decay and Float64 PyTorch reference to isolate exactly the original Adam formula. The training experiments use **AdamW with decoupled weight decay 0.01**.
- Parameter group ratios use post-update difference divided by the pre-update parameter norm. Logs include the exact scheduled LR at every step.
- Warmup steps 1–20 reach peak LR; first full stable-rate step is 21. **Warmup ends, not the evolution of the update/weight ratio.**
- The scheduled WSD decay begins **at step 241**, after plateau through step 240. Thus the forced checkpoint at step 200 is before any WSD cooldown.
- Schedule peak LR is tuned **independently** across the six-grid values using both seeds over 120 updates, then fresh runs stop at 200. Both schedules use identical datasets, seeds, model size, optimizer, batch sizes and parameter initializations for their matching seeds.
- Width sweep holds bottleneck and all other design parameters fixed, uses 55 updates at width 256/512/1,024 across both seeds. The 4,096 answer is a **low-confidence untested hypothesis**, not a transferred optimum.

## Selected numerical conclusions

- Adam maximum discrepancy to PyTorch: **1.457e-15**.
- No-bias-correction *per-update* persistent 1% threshold (constant gradient, defined test): step **3,925**.
- Cosine at step 200: **1.081856**, WSD: **1.163503** (mean over two seeds).
- LR grid minimum for width 256: **0.01**; width 512: **0.003**; width 1,024: **0.003**.
- Tentative LR at 4,096: **0.003**, **very low confidence**.

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
