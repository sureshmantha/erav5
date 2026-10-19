# Session 14 — Short submission answer

**Method.** Train an 86,592-parameter byte-level causal Transformer with two dense `64→128→64` FFNs on a 300-record cleaned Alpaca/CodeAlpaca/Dolly educational subset. After 60 AdamW updates, clone each learned FFN into six experts and initialize a top-2 router. Normalized weighted summation makes the conversion function-preserving. Train the converted 253,120-parameter MoE for 90 further updates on the same training split. Measure next-byte held-out cross-entropy on the same six fixed evaluation batches; do not include balancing auxiliary loss in the reported CE.

**Measured fixed held-out CE (nats/token):** 5.566130 untrained → **3.376164 dense** → 3.376164 immediately after conversion → **3.314345 MoE**. Maximum conversion logit error: **1.192093e−6**. The dense-only matched continuation reached **3.314582**: no meaningful performance advantage established. All 6 experts received held-out assignments in each MoE layer; routing is imperfectly balanced.

**Why it works.** At conversion, every expert is an exact FFN clone and selected routing weights sum to one, so outputs remain unchanged. Later, chosen experts get specialized gradient updates. **Cost:** 253,120 total / 120,512 active-path parameters per token; this toy MoE is not faster than the much smaller dense model. The program is CPU-based and performs real PyTorch forward/backward/optimizer steps. Seven automated checks pass.

**Submission:** Notebook `ERA_V5_S14_Dense_to_MoE.ipynb`, full `README.md`, `src/`, `tests/`, checkpoints, figures, machine-readable results. GitHub publication is pending; no invented link.
