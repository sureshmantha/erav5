import nbformat
from pathlib import Path
p=Path(__file__).parent
script=(p/'experiment.py').read_text();script=script[:script.index("if __name__=='__main__':")]
nb=nbformat.v4.new_notebook()
md=nbformat.v4.new_markdown_cell
code=nbformat.v4.new_code_cell
nb.cells=[
 md('''# ERA V5 — Session 11 | Optimizer Laboratory
### Adam • bias correction • update-to-weight • cosine vs WSD • LR by width

**Reproducible CPU notebook, grounded in the Session 11 assignment.** The notebook contains its entire executable source; no GitHub clone or internet dataset is needed. All plots and CSV tables are generated from execution rather than mock data.

**Scientific scope:** A small causal residual *token mixer* with synthetic 32-symbol sequences, not a Transformer/production LLM. The 4,096 LR is an explicitly low-confidence extrapolation; no μP is implemented. Hidden validation set has no overlapping full sequences with training.'''),
 md('''## Methodology / safeguards

- Hand calculations in `Decimal(precision=55)` vs independent `torch.optim.Adam(float64)` for **all six quantities at five steps**.
- Adam bias correction vs deliberate removal: fixed gradient isolates zero-state bias; report when **individual update size**, not parameter path, stays within 1%/5%.
- Layerwise ratio includes both weight and bias of each parameterized module, before/after snapshots on each update.
- 300-step *planned* schedule, physically **stop at 200**; WSD's cooldown only begins at 241, so it cannot be credited with unobserved improvements.
- Independently LR-tune both schedules with a 120-step two-seed pilot; compare both at step 200 with matched initializations, data, and settings.
- Width sweep covers 256, 512 and 1,024, six LR values, two seeds each; **held-out** cross-entropy selects tested-grid minima.
- Research choices (step counts, rates, seeds, model, data generator) are documented, not passed off as transcript facts.'''),
 md('## 1. Complete source: architecture, independent Adam, schedules and measured experiments'),
 code(script),
 md('## 2. Verify disjoint training/held-out data and causal behavior'),
 code('''print('TRAIN:',tuple(TRAIN.shape),'VALID:',tuple(VAL.shape))
assert not {tuple(x) for x in TRAIN.tolist()} & {tuple(x) for x in VAL.tolist()}
seed_all(23)
causal_model = CausalTokenModel(256)
x=TRAIN[:2,:-1].clone(); original=causal_model(x)
x[:,5:]=(x[:,5:]+13)%V
revised=causal_model(x)
assert torch.allclose(original[:,:5],revised[:,:5],rtol=1e-6,atol=1e-6)
print('PASS: no full-sequence duplicates; modifying future tokens does not change earlier logits')'''),
 md('## 3. Hand-computed Adam compared with `torch.optim.Adam`'),
 code('''adam_rows = hand_adam()
print('step | grad | m | v | m-hat | v-hat | signed update | weight')
for r in adam_rows:
 print(f"{r['step']:>4} | {r['gradient']:+.3f} | {r['m_hand']:+.9f} | {r['v_hand']:.9f} | {r['mhat_hand']:+.9f} | {r['vhat_hand']:.9f} | {r['update_hand']:+.9f} | {r['weight_hand']:.9f}")
print('PASS: all six columns × five steps verified independently')'''),
 md('''## 4. Run full experiment (may take tens of seconds on a CPU)

All measurements are logged and plots saved. `run_all()` deliberately regenerates outputs; no stale results are carried forward.'''),
 code('''summary=run_all(fast=False)
assert not summary['fast_smoke_test']
print('Planned 300, stopped at',summary['experiment']['stop_steps'])
print('Selected pilot LRs:',summary['schedules']['independent_best_lr'])
print('Held-out losses at 200:',summary['schedules']['step200_mean_val_loss'])
print('Width grid minima:',summary['width_sweep']['minima_on_tested_grid'])
print('4096 provisional LR:',summary['width_sweep']['candidate_at_4096'], 'confidence:',summary['width_sweep']['confidence'])'''),
 md('## 5. Bias correction, layerwise warmup ratios, schedule comparison, width grid'),
 code('''from IPython.display import Image, display
for img in ['bias_correction_20_steps.png','update_to_weight_ratios.png','cosine_vs_wsd_stop200.png','lr_sweep_widths.png']:
    print('Measured plot:',img)
    display(Image(filename=str(OUT/img),width=880))'''),
 md('''## 6. Interpret the numbers (without exaggeration)

- Bias correction is **not** AdamW weight decay. `m̂` and `v̂` correct zero-initialized EMAs; AdamW's decoupling is different.
- At **step 21**, scheduled WSD warmup stops increasing the learning rate. The update-to-weight ratios still vary with optimizer state and gradients.
- At step 200, WSD has **not** entered its planned decay. Keep whichever checkpoint has lower held-out loss **at 200**, without asserting that it wins at the full schedule horizon.
- Width 4,096 was **not run**. A provisional starting LR from ordinary PyTorch width scaling is not established μTransfer. Test it before use.

### Primary references

- [PyTorch Adam specification](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- [Adam](https://arxiv.org/abs/1412.6980) and [AdamW](https://arxiv.org/abs/1711.05101)
- [MiniCPM WSD](https://arxiv.org/abs/2404.06395) and [WSD analysis](https://arxiv.org/abs/2410.05192)
- [Tensor Programs V / μTransfer](https://arxiv.org/abs/2203.03466)

Refer to `README.md`, `WRITEUP.md`, and `ACCURACY_AUDIT.md` in the ZIP for complete detail and original code checks.''')
]
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
nbformat.write(nb,p/'ERA_V5_Session_11_Optimizer_Lab.ipynb')
print(len(nb.cells),'cells',len(script),'script characters')
