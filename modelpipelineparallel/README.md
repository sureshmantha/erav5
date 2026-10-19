# ERA V5 — Session 13 | Reversibility and a 20M-parameter LM

## Submission status (read first)

**Implementation + deterministic pilot validated. The mandatory three 50M-token GPU trainings have NOT been executed in this environment. The notebook must be executed on Google Colab GPU, and the results must be added before submission. No GitHub repository has been published.**

Only Session 13 transcript was attached in this turn; separate lesson notes were not visible. Relevant source transcript: `ERA V5 Session 13.docx`.

## Requirements ↔ Evidence

| Requirement | Status | Evidence |
|---|---|---|
| ~20M trainable parameters | PASS | Both modes have **19,804,800** tied-head parameters; `--test` |
| Original model 50M tokens | PENDING GPU | Notebook experiment A |
| Reversible model 50M at same batch | PENDING GPU | Notebook experiment B |
| Reversible model 50M at largest batch | PENDING GPU | Notebook experiments C and batch probe |
| Backward verification | PASS on CPU | inverse and gradient numerical tests |
| Final loss, tokens/s, peak GPU memory | PENDING GPU | Generated `summary.json` and comparison CSV after run |
| GitHub repo link | PENDING publication | ZIP is repository ready |

## Model and dataset

Both 19,804,800-parameter models use width 320, 16 transformer layers, 8 causal attention heads, 4× feed-forward expansion, learned position embeddings, and **tied input/output byte embeddings**, vocabulary 256. The original is a pre-layer-norm residual decoder; the reversible model uses two reversible streams with additive coupling and averages the outputs. They have equal parameter counts but not exactly identical computations, so the comparison is not a controlled activation-memory-only A/B test.

The full training notebook streams **roneneldan/TinyStories**, using UTF-8 bytes as integer tokens. This is a real-data experiment but a byte tokenizer is inefficient and does not imply subword token counts. Verify the dataset's current license/revision before use. Dataset link: https://huggingface.co/datasets/roneneldan/TinyStories

Each measured target position counts as one processed token. The last batch may slightly overshoot 50M; compare using `executed_tokens`, not the requested label. Batch rows have a one-byte target overlap within each row but are split in a simple deterministic fashion.

## What reversible means in THIS repository

Each block has two streams. Let `F` be layer-norm + causal attention and `G` be layer-norm + GELU feed-forward:

```
y1 = x1 + F(x2)
y2 = x2 + G(y1)
x2 = y2 - G(y1)
x1 = y1 - F(x2)
```

This is **Gomez-style additive coupling**, not Euler or midpoint numerical integration. Unlike normal activation checkpointing, its custom backward reconstructs inputs from outputs and reruns F/G to recover parameter derivatives. Recomputing costs time; it does **not** eliminate all memory (embeddings, output-head logits, gradients, optimizer state, attention workspace still cost memory). The positional and embedding operations are outside reversible core.

Transcript references Euler / midpoint. Those are NOT silently identified with the implementation. The assignment asks to report what worked; final comparative verdict requires running actual experiments on the target hardware. Do not claim the transcript's 30–40% slowdown necessarily holds for this implementation.

## Evidence from executed CPU smoke tests

```
python train.py --test
```

- Inverse maximum absolute difference **4.440892098500626e-16** in double precision.
- Custom backward vs normal PyTorch autograd maximum gradient difference **1.214306433183765e-17**.
- Parameter count each **19,804,800**.
- One CPU step per architecture succeeded with a random synthetic byte stream; see `pilot_results.json`. These are NOT final losses or comparable training results.

## Execute 3 full GPU runs

1. Open `ERA_V5_Session_13_20M_50M_Reversibility.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type → GPU**. Use a sufficiently large GPU and persistent Drive storage (ephemeral runtimes can end early).
3. Execute cells, confirm parameter-count and numerical tests.
4. Select baseline `BATCH` which actually fits. Keep **exactly the same batch** for baseline and first reversible run.
5. Run 50M target tokens baseline; separately run 50M reversible fixed batch.
6. Probe reversible maximum batch. Probe reports powers of two; notebook reserves headroom and uses a conservative high batch. For the literal maximum feasible batch, additionally binary-search between successful and failing values and rerun sustained steps before acceptance.
7. Run 50M target tokens reversible higher batch.
8. Export summary JSON files, full logs, plot and comparisons into the repo. Record GPU model, CUDA/PyTorch versions, peak allocated+reserved memory, interrupted runs, dataset revision, and wall-clock time.
9. Publish to GitHub and only then include its actual URL in the assignment submission.

Run a short synthetic smoke from CLI: `python train.py --data synthetic --mode reversible --tokens 32 --batch 2 --context 16 --max-steps 1 --output smoke`.

CLI full runs (after installing requirements and using GPU):

```
python train.py --data tinystories --mode baseline --tokens 50000000 --batch 4 --output results/baseline
python train.py --data tinystories --mode reversible --tokens 50000000 --batch 4 --output results/reversible_fixed
python train.py --data tinystories --mode reversible --find-batch --batch 4
# replace 8 with empirically verified batch value
python train.py --data tinystories --mode reversible --tokens 50000000 --batch 8 --output results/reversible_large
```

## Accuracy warnings and next improvements

- **No automatic checkpoint resume.** `latest.pt` stores weights/Adam/steps; resume should also restore exact stream state and RNG. A restarted command currently starts a fresh experiment. A Colab interruption does not count as 50M completed.
- Initial project does not perform held-out loss; comparison is of logged *training mini-batch loss only*. Add fixed held-out evaluation for strong model-quality claims.
- Three independent trainings match seeds but architectures differ. For scientific conclusions require multiple seeds, same effective token budget, optimizer schedule and dataset ordering. Different batch sizes imply different optimizer update counts; this is a material confounder.
- Full training is FP32 by default. CUDA BF16 mixed precision and kernel profiling are future extensions, not results.
- Maximum available batch depends on hardware, context length, kernel, precision and how much reserve headroom you require. Do not invent maximum.
- `torch.cuda.max_memory_allocated` measures allocated tensor peak, NOT GPU reserved peak or global system utilization.
- Byte-token training is deliberately simple; 50M bytes are 50M byte tokens, not 50M language-model BPE tokens.

## Personal understanding checkpoints — write these yourself before submitting

1. Why does ZeRO-3 fail to make the entire backward activations disappear, whereas reversible coupling can reconstruct some of them?
2. Show the algebra that reconstructs x1/x2. Which floating-point effects make it non-exact on hardware?
3. Why might reversible 50M tokens train slower despite using lower peak activation memory?
4. Why is training at a larger batch not an isolated study of reversibility alone?
5. If midpoint is mentioned in the transcript, why is it inaccurate to label this additive coupling as midpoint without demonstrating its specific equations?

## Sources

- Uploaded Session 13 transcript, 2026-09-19: distributed computation and reversibility assignment.
- Gomez et al. (2017), *The Reversible Residual Network: Backpropagation Without Storing Activations*: https://arxiv.org/abs/1707.04585
- Kitaev et al. (2020), *Reformer: The Efficient Transformer*: https://arxiv.org/abs/2001.04451
- Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- PyTorch causal scaled-dot-product attention: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
