# Audit of claims and artifacts

**Verified:** 19,804,800 trainable parameters each variant; FP64 inverse error 4.44e-16; PyTorch backward equivalence error 1.21e-17; both models completed one CPU synthetic-byte training step.

**Not done:** 50M-token runs on Colab, hardware maximum-batch determination, final training losses, meaningful GPU peak allocation, performance comparison, GitHub publication. CPU pilot speed is not Colab/GPU throughput.

**Method distinction:** Implemented reversible residual additive coupling (Gomez et al.), not numerical midpoint or Euler. A custom autograd function actually reconstructs activations; calling torch checkpoint is not used.

**Source distinction:** Transcript describes several reversible integration ideas and a cohort-specific Lightning LM; their exact trained architecture, reproducing code and measurements were not supplied as inputs. This project does not claim to replicate those specific proprietary methods or scale results.
