# Session 10 — transcript accuracy audit

Course source: **ERA V5 Session 10**, transcript dated 2026-08-29 (user-supplied DOCX).

## Grounded concepts carried into code

- From `token → embedding → model → logits → shifted cross-entropy → backward → optimizer.step()`.
- Micro-batches accumulate gradient buffers without changing weights between their backward passes.
- Measuring and monitoring both gradient norms and loss, with distinct time indices.
- A rough *dense* training-compute heuristic `6 × parameters × training tokens` is **an estimate**, not a profiler count.
- FP32, BF16 and FP8 E4M3 each quantize 0.1 to different values.

## Transcript wording and arithmetic that must not be copied uncritically

1. **Accumulation backward count**: the speaker initially says there is one backward for several micro-batch forwards, then explicitly corrects that one optimizer update requires **multiple forward/backward micro-batch pairs**. Code follows corrected wording.
2. **Gradient is a derivative, not yet the weight update**: `p.grad` stores the derivative; optimizer transformations and learning rate determine the actual parameter change. The harness keeps them distinct.
3. **Unequal micro-batches**: a naive average of mean losses is not the gradient of a global average across all valid tokens. Code verifies the equivalence with an explicit full batch.
4. **Numerical example**: with N=9×10^9 and tokens/s=12,000, the 6N rule yields 648×10^12 FLOP/s. Dividing by a stated 7,912×10^12 FLOP/s aggregate peak gives 8.19%; the quoted “6,848 TFLOP/s” is inconsistent with that arithmetic. The original peak GPU setup is not independently benchmarked here.
5. **Gradient changes versus loss timing**: a changed gradient norm may occur without a large immediate scalar loss change. This alone cannot establish a predictive lead-lag relationship. Notebook makes this distinction.
6. **FP8 support**: the codec is E4M3FN with exponent bias seven. Not all FP8 encodings behave alike; E4M3FNUZ and E5M2 are different formats.
7. **FP8 hardware**: bit encoding can be checked on CPU. FP8 training efficacy, throughput and stability require a separate hardware validation campaign.

## References

- PyTorch tensor attributes: https://docs.pytorch.org/docs/stable/tensor_attributes.html
- PyTorch FP8 E4M3FN definition: https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_e4m3fn.h
- NVIDIA Hopper architecture specifications: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- FP8 floating-point formats research: https://arxiv.org/abs/2209.05433
