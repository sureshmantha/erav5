# ERA V5 · Session 12 — ZeRO on 32 Virtual GPUs

**Assignment:** run a small real model on **32 logical GPU ranks**; simulate standard data parallelism (`ZeRO-0`), ZeRO stages 1, 2, and 3; compare memory and computation; submit an executable `.ipynb` with an explanatory README.

**Status:** all four stages executed; 7 automated tests passed. No physical GPU or distributed interconnect was used. Repository URL is **not yet published**. Everything can run on CPU or in Google Colab.

> **Honesty boundary:** This is an *educational ZeRO simulator* written in PyTorch. It uses real CPU tensor math, real autograd/backpropagation, and real shard ownership and Adam updates. Local `torch.cat` is used to **emulate** parameter all-gathers rather than NCCL or network transfers. It does **not** execute true distributed DeepSpeed/FSDP, does not measure GPU VRAM, and cannot estimate real GPU throughput from CPU timing. The mixed-precision memory and communication tables are analytical projections, not allocator telemetry.

## How to run

1. Download `ERA_V5_S12_ZeRO_32_Virtual_GPUs.ipynb`, open it in [Google Colab](https://colab.research.google.com/), and select `Runtime → Run all`. **CPU runtime is sufficient**. The notebook is standalone: it writes its own `src/` and `run_experiment.py` files from the embedded implementation.
2. Alternatively, clone the repository after publishing or extract the ZIP, then run:

   ```bash
   python -m pip install -r requirements.txt
   python run_experiment.py
   python -m unittest discover -s tests -v
   ```

3. Check `results/experiment.json` and `figures/` for fresh output. Both scripts fix CPU intra-op threads at one for inexpensive, predictable tiny tensor examples; threading cannot simulate physical GPU collectives.

### Files

| Path | Contents |
|---|---|
| `ERA_V5_S12_ZeRO_32_Virtual_GPUs.ipynb` | Standalone Colab notebook, executable top-to-bottom |
| `ERA_V5_S12_EXECUTED.ipynb` | Notebook executed in our CPU environment, complete cell outputs |
| `src/virtual_zero.py` | 32-rank state, real training, sharded optimizer, Stage-3 layer gather/rematerialize, math |
| `run_experiment.py` | Reproducible results and figures export |
| `tests/test_virtual_zero.py` | Seven automated tests, including PyTorch Adam equivalence |
| `figures/*.png` | Memory, modeled communication, computation, loss parity, 32-rank state map |
| `results/*.json` and `*.csv` | Machine-generated raw results |
| `ACCURACY_AUDIT.md` | Transcript-versus-source terminology and limitations |

## 1. My understanding — a reviewable explanation

*This first-person summary is a **draft for the student to check, explain aloud, and revise**. It must not be presented as independent human work without personal understanding and review.*

**Why data parallelism is expensive in memory:** I want each of 32 ranks to process different examples of the *same* model. Ordinary data-parallel training replicates the model parameters, gradients, and optimizer state on every rank. The weights are initially identical, the local gradients differ because local inputs differ, and a reduction of the 32 local gradients produces the globally correct update. Therefore identical updates keep model copies synchronized.

**What ZeRO actually removes:** The *optimizer state* and the *reduced gradient* do not need 32 redundant replicas. I can assign each rank a non-overlapping ownership slice and leave computation of the full forward/backward model on the rank's own input. Sharding changes storage and data movement; it does not make a dense forward/backward pass cost 1/32 as many matrix multiplications on a given rank.

**Stage 0:** Each rank stores the complete 16-bit parameters and gradients plus Adam master parameters/moments. Gradients are all-reduced, and **each rank** applies Adam to its entire parameter set. It is correct but repeats optimizer arithmetic and state across all 32 ranks.

**Stage 1:** Each rank still has the complete 16-bit weights and local gradients. But it owns only **1/32 of the Adam master weights, m and v**. It reduces the gradient shard for which it is responsible, updates that shard, and all-gathers the updated weights so each rank has the same complete weights again. It saves most of the 12-byte-per-parameter Adam state.

**Stage 2:** Parameters remain replicated, but *persistently stored reduced gradients* are now sharded with the optimizer. As backward computes gradients, reduce-scatter routes the relevant gradient pieces to their owners. Temporary backward tensors and communication buckets may still exist; sharded persistent state does not mean zero memory spikes.

**Stage 3:** Even 16-bit parameters become sharded. For a layer to compute, the relevant weights are temporarily gathered, used, then released/repartitioned. Backward may need to fetch them again, so communication increases. My code implements per-layer local gathers on 32 virtual ranks and **actually rematerializes** each layer's weights in its custom backward operation; gradients flow to the rank-owned original parameter fragments. The math is unchanged. At the end of each step, no rank persistently owns a full model.

**When I would choose:** Stage 1 for inexpensive optimizer-state savings when the rest of the model fits; Stage 2 if persistent gradients are the next bottleneck; Stage 3 when full parameters cannot fit or memory is the priority despite extra gathers. None is guaranteed fastest: network, buckets, overlap, memory limit, activations, and architecture determine actual performance.

**Important condition:** The equal results in this notebook prove *algorithmic equivalence on this small example*. They do not prove good scaling on 32 real GPUs. For a large run I would measure actual per-rank peak VRAM, NCCL byte counts, bandwidth, step time, MFU, stalls, and correctness over several batches/checkpoints.

## 2. Real experiment configuration

- **World size:** 32 logical, sequentially simulated ranks (IDs 0–31); **physical CUDA GPUs used: 0**.
- **Data:** fixed synthetic supervised example; 2 samples per rank = 64 globally; 16 input features; deterministic generated class targets; **not an LLM dataset**.
- **Model:** a small real two-layer MLP classifier, `16 → 32 → 8`, ReLU, cross-entropy; 808 trainable parameters.
- **Training:** 4 Adam optimizer updates, LR `0.008`, β₁ `0.9`, β₂ `0.999`, eps `1e-8`, decay `0`. Same starting parameters and inputs for all stages.
- **Execution precision:** FP64 for strict equivalence of PyTorch calculations. **Storage projection precision:** BF16 2B weights, BF16 2B gradients, FP32 master weights 4B + Adam m 4B + Adam v 4B. These are not the same measurement.
- **Determinism:** fixed torch random seeds `20260912` (weights) and `312` (dataset); single CPU thread. CPU runtimes can still vary by hardware and environment.

### Actual global cross-entropy, before each optimizer update

| Stage | Step 1 | Step 2 | Step 3 | Step 4 | Max final weight error versus Stage 0 |
|---|---:|---:|---:|---:|---:|
| 0 | 2.082058659 | 2.044493302 | 2.007219136 | 1.968615257 | 0 |
| 1 | 2.082058659 | 2.044493302 | 2.007219136 | 1.968615257 | < 8e-17 |
| 2 | 2.082058659 | 2.044493302 | 2.007219136 | 1.968615257 | < 8e-17 |
| 3 | 2.082058659 | 2.044493302 | 2.007219136 | 1.968615257 | < 8e-17 |

**What this demonstrates:** all modes preserve the global gradient and Adam updates to floating-point precision; the loss falls during actual computation. The graph overlays four trajectories, rather than inventing divergent accuracy improvements from memory sharding.

## 3. How persistent-state memory changes (32 ranks)

The example assumes mixed-precision training with **16 bytes / parameter / rank** without ZeRO:

- BF16 model parameter: 2 B/parameter
- BF16 gradient: 2 B/parameter
- FP32 master parameter: 4 B/parameter
- FP32 Adam first moment: 4 B/parameter
- FP32 Adam second moment: 4 B/parameter

For N=32 ranks, with P trainable parameters:

| Stage | Persistent model+gradient+Adam bytes per rank | Expression | Projected at 30B params (GiB / rank) | Reduction vs baseline |
|---|---:|---|---:|---:|
| 0 | 16.0000 P | `16 P` | **447.035** | 1.00× |
| 1 | 4.3750 P | `(4 + 12/N) P` | **122.236** | 3.66× |
| 2 | 2.4375 P | `(2 + 14/N) P` | **68.103** | 6.56× |
| 3 | 0.5000 P | `(16/N) P` | **13.970** | 32.00× |

**30B is an analytic projection, not an executed 30B model.** The 808-parameter model is the only real training run. Memory conversion is `GiB = bytes / 2^30`; e.g. `30e9 × 16 / 2^30 = 447.0348 GiB`.

These numbers omit activations, attention buffers, allocator fragmentation, communication staging, checkpoint state, temporary parameter gathers and sequence batch requirements. Stage 3's **minimum additional** largest one-layer BF16 materialization in our toy model is 544 × 2 = 1,088 bytes. On real models and in frameworks with prefetching the temporary/peak cost can be far higher; **13.97 GiB does not prove a 30B model fits inside a 14 GiB GPU**.

Averaging 808 parameters among 32 ranks creates shards of **25–26 elements**. The per-rank storage heatmap uses those exact uneven shard sizes, rather than claiming fractional physical tensor entries.

## 4. Compute and communication trade-off

### Arithmetic performed

| Work (per rank over same local samples) | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|---:|
| Dense model forward/backward arithmetic | 100% | 100% | 100% | 100% |
| Adam element updates owned by rank (cluster-wide sum unchanged) | 808 | avg. 25.25 | avg. 25.25 | avg. 25.25 |
| Full 16-bit parameters permanently resident | Yes | Yes | Yes | No |
| Additional parameter gather needed for layer execution | No | No | No | Yes |

Stage 3 may incur some extra CPU **rematerialization** work; it does not change the mathematical forward/backward gradient objective. The notebook separately shows measured CPU step times, which are simulator implementation times and are **not** GPU throughput/modeled network timings.

### A transparent illustrative network payload model

Let `S = 2 P` **bytes of BF16 model parameters**, `N=32`, and `f=(N−1)/N = 31/32`. Assuming a ring and no prefetch/overlap/bucket overhead:

- Reduce-scatter(S) sends ~`f S` bytes/rank.
- All-gather(S) sends ~`f S` bytes/rank.
- All-reduce(S) is typically reduce-scatter + all-gather, sending ~`2 f S` bytes/rank.
- Stage 0: gradient all-reduce → **`2 f S`**.
- Stage 1: gradient reduce-scatter to owner + updated parameter all-gather → **`2 f S`** (an *optimized* Stage 1 layout, not every implementation).
- Stage 2: gradient reduce-scatter + updated parameter all-gather → **`2 f S`**.
- Stage 3: reduced gradient shard + **one parameter gather each in forward and backward** → **`3 f S`**.

Thus Stage 3 sends **1.5×** the estimated Stage-0 ring *bytes* per rank, not necessarily 1.5× wall-clock time. For **30B parameters**, this gives stages 0–2 **116.25 GB/rank/step**, stage 3 **174.375 GB/rank/step**. These are simplified payload counts. The code has **no physical networking at all**; uses local tensors to implement the same data-flow semantics. Actual traffic depends on bucket order, how many times parameters are prefetched/gathered, retries, replication, topology, and collectives. Other ZeRO-1 implementations may use a full gradient all-reduce **plus** parameter all-gather, costing more.

## 5. Evidence that the simulation is functionally correct

The automated tests assert:

1. Exactly 32 disjoint, exhaustive ownership slices, sizes differ by at most one element.
2. All four stages produce the same 4-step global loss and final weights (tolerance `1e-12`).
3. Stage 3's per-layer gather produces the exact same forward logits as the fully assembled model.
4. Stage-specific **parameter, gradient, moment resident tensors** have correct ownership shapes.
5. Both the 8-rank classroom formulas and 32-rank project formulas match the explicit accounting.
6. Communication assumptions are internally consistent (stage-3 payload 1.5× stage-0).
7. Independent PyTorch `torch.optim.Adam` one-step update matches our custom Adam implementation.

Tests are executed with `python -m unittest discover -s tests -v`.

## 6. Five oral-defense questions to check personal understanding

**Q1. Why does ZeRO-1 reduce memory even when every rank still runs all layers?** Because optimizer parameters/moments are sharded while forward/backward still consume replicated compute weights.

**Q2. Why aren't ZeRO-1 and ZeRO-2 identical?** At Stage 1 each rank retains a full local gradient allocation; Stage 2 partitions persistent reduced gradients among owners, removing that residency.

**Q3. Why can Stage 3 be slower despite much lower resident memory?** We have to materialize the missing parameter shards for forward/backward. Communication latency, network bandwidth, synchronization and peak buffers can dominate.

**Q4. Why shouldn't we divide forward/backward FLOPs by 32 for Stage 3?** We are applying **data parallelism** across 32 different minibatches, not splitting a single dense matmul across 32 workers with tensor parallelism.

**Q5. Why do identical losses not prove full DeepSpeed equivalence?** Real implementations include mixed precision, asynchronous communication, interleaving, fused kernels, optimizer step details, checkpointing and much larger networks. This simulator validates the selected mathematical and ownership invariants only.

## 7. Reference list and transcript provenance

**Primary assignment/source**: attached *ERA V5 Session 12* transcript, dated September 12, 2026. The relevant material follows the instructor's discussion of data parallelism, all-reduce/reduce-scatter/all-gather, ZeRO-1/2/3, and the closing assignment. The transcription notes that it was computer-generated and may contain errors. Its 16-byte memory example was checked independently rather than accepted uncritically. **No separately attached “lesson notes” file was visible in this turn; only the Session 12 `.docx` was available.**

**Independent primary documentation:**

- Rajbhandari, Rasley, Ruwase & He (2019), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, https://arxiv.org/abs/1910.02054 (original ZeRO paper).
- DeepSpeed team, official ZeRO-3 documentation: https://deepspeed.readthedocs.io/en/stable/zero3.html (stage definitions, gather semantics).
- DeepSpeed team, configuration reference: https://www.deepspeed.ai/docs/config-json/ (stage 0–3 and tunables).
- PyTorch, distributed collective documentation: https://pytorch.org/docs/stable/distributed.html (understanding ring collectives and true distributed implementation — not invoked by this simulator).

## 8. What to do for real 32-GPU capstone training

Use actual `torch.distributed` with NCCL and 32 processes, or a measured DeepSpeed ZeRO/FSDP configuration; fix global effective batch and seed/sharding scheme, compare checkpoints/held-out loss, inspect `torch.cuda.max_memory_allocated` per rank, profile NCCL and step time, and tune bucket size, overlap and activation checkpointing. **The current notebook is an architectural understanding and correctness demonstration, not that performance trial.**

## 9. GitHub publication

Ready to commit, **not already published**. Proposed repository: `era-v5-session-12-zero-32-virtual-gpus`. Upload the complete project folder, then open the `.ipynb` file in Colab using GitHub import. Submit the actual repository URL after publishing; do not submit a guessed URL.
