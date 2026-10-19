# Accuracy audit — Session 11

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
