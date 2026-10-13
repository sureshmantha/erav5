# Session 9 — Submission Write-up

**Experiment:** actual execution of a tiny decoder-only causal transformer in PyTorch. **Dataset:** deterministic, synthetic, 12 sentences (36 training examples after repetition). Seed 20260822; CPU execution. This is a correctness audit, not pretraining on a 10–20M-token dataset.

## Part 1 — Seven evidence checkpoints

| Requirement | Measured output / verification |
|---|---|
| 1. Every tensor shape | `tokens=[2, 20]`, `hidden=[2, 20, 48]`, `logits=[2, 20, 128]`; shifted logits `[2, 19, 128]`, targets `[2, 19]`. More intermediate shapes printed in notebook. `B` batch, `T` sequence, `D` hidden, `H` heads, `V` vocabulary. |
| 2. Verify shift | **PASS**: printed literal `INPUT → TARGET` token strings for every position. At index 0, `<bos> → the` rather than `<bos> → <bos>`. |
| 3. Padding excluded | **38 → 24** positions; **14** omitted. Loss 4.857075 before vs 4.853628 after PAD exclusion. |
| 4. Packed document boundary | **1 EOS→BOS transition excluded**. Loss on the same logits **4.857234 → 4.857613**; block-diagonal attention + local position reset yields **4.857832**. The direction of loss change depends on the boundary token's difficulty. |
| 5. Perplexity sanity | **128.2046** vs vocab **128**; `PPL/V=1.00160`. Pass (uniform-init check). |
| 6. Tied vs untied head | **43,440 tied**, **49,584 untied**, **6,144 parameters saved**. |
| 7. Full vs chunked CE peak memory | CPU process peak RSS: **353.90 MiB full**, **295.19 MiB chunked**, ratio **1.199× of absolute peaks**. Both losses and gradients match within tolerance. This is CPU RSS, not GPU VRAM. |

### Part 2 — Two independent next-token targets

| At step 100 | Cross entropy, nats/token |
|---|---:|
| `t+1` head | **0.213744** |
| `t+2` head | **0.429029** |
| Joint sum | **0.642773** |

At initialization the losses were 4.848046 (`t+1`) and 4.851737 (`t+2`). After 100 updates both decreased, but `t+2` remained larger on this specific training corpus. Predicting two tokens ahead leaves more uncertainty because the intermediate token has not been observed by that head. This explanation is an **interpretation of this experiment**, not a general theorem; loss ordering depends on dataset, capacity, initialization, masks and optimization. The total loss uses unit weights: `loss = loss_t1 + loss_t2`.

### Scope and non-claims

Results are from a **toy corpus** and **CPU** execution. The trainer does not establish generalization or the performance of a 20M-parameter model trained on FineWeb. The memory comparison isolates the output projection and loss/backward (not a complete transformer step). The actual GPU VRAM measurement must be obtained by running the same notebook on a Colab GPU.

**Correctness tests passed:** correct shift, padding masks, document boundary target mask, packed attention isolation, independent local positions, no future-information leak, `t+2` two-crossing guard, chunked loss and gradient equality.
