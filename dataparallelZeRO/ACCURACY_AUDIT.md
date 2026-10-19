# Source-and-assumption audit — ERA V5 Session 12

- Primary evidence: attached `ERA V5 Session 12.docx`, transcript dated 2026-09-12. Sections: state accounting, data parallelism, collectives, Stage 1/2/3, memory, assignment. Machine transcription may contain errors.
- Independent checks: original ZeRO paper (https://arxiv.org/abs/1910.02054); official DeepSpeed ZeRO documentation (https://deepspeed.readthedocs.io/en/stable/zero3.html); configuration reference (https://www.deepspeed.ai/docs/config-json/).
- **Terminology:** Standard data parallel is *ZeRO Stage 0*, not a separate optimization called `ZeRO` besides Stages 1–3. "Simulate ZeRO, ZeRO1, ZeRO2, ZeRO3" is operationalized as Stage 0, 1, 2, 3.
- **Corrections/clarifications to protect accuracy:**
  1. **Training batch size does not imply one full parameter-model copy per example.** A GPU normally uses the *same* weight tensors for the batch; activations and temporary tensors scale with the batch dimension. Transcript analogy that batch size 8 means 8 copies of the model is not literally correct standard PyTorch behavior.
  2. **ZeRO does not divide dense forward/backward FLOPs by 32.** Data ranks do the full model on distinct input samples; savings arise from sharded persistent optimizer/gradient/weight state. Stage 3 may *add* gather/rematerialization overhead.
  3. **16 B/parameter is conditional.** Assumes 2B low-precision params + 2B gradients + 4B FP32 master + 8B FP32 Adam moments. Other dtypes, optimizers, fused implementations, parameter ties and offloading change this formula.
  4. **ZeRO-3 state memory ≠ ZeRO-3 peak memory.** Gathered layer buffers, activations, transient gradients, communication buckets, allocator reserve and checkpointing cost are excluded from the ideal resident-state formula.
  5. **Layer-by-layer ownership is illustrative.** Real ZeRO Stage 3 may partition flattened parameter tensors with mixed layer boundaries and prefetch; the simulator partitions all 808 parameter elements contiguously into 32 uneven shards and assembles each layer from fragments.
  6. **Network traffic estimates are not timings.** They assume an optimized ring and a single Stage 3 all-gather each in forward and backward. Stage 1's optimal gradient RS + parameter AG is not universal; topology/overlap/collectives can alter costs.
  7. **Virtual device accounting is not CPU tensor allocator accounting.** Execution uses FP64 PyTorch tensors to test exactness; the 16-byte budget is a BF16/FP32 *analytical projection*. The code uses `torch.cat` to emulate an all-gather within one process.
  8. **ZeRO-3 CPU calculation overhead is not GPU speed.** Tiny tensor CPU runtimes are displayed to make the local model real, not presented as distributed speedup.
  9. **Gradient accumulation and tensor parallelism differ from ZeRO sharding.** This notebook fixes 2 samples per rank, aggregates 32-rank gradients and uses one optimizer update per global batch. It does not implement tensor parallelism or GPU pipeline scheduling.
  10. **No separate lesson notes available** from the attachments of this turn; transcript is the provided source. No unobserved lesson notes are represented as read.
- **Numerical reproducibility:** Native PyTorch Adam reference test; 7 tests; check `results/experiment.json` for exact values. Analytic 30B dataset is not executable on this runtime and is labeled as a scenario, not a measurement.
- **Honest publication status:** GitHub repository not yet created; provide the zip and publishing instructions rather than inventing a URL.
