# Attention Atlas — ERA V5 Session 8

**A research-audited, single-screen, offline-capable, interactive attention chronology.**

![Static project; no user account is needed](https://img.shields.io/badge/dependencies-0-247b83) ![Milestones](https://img.shields.io/badge/milestones-32-315fc0) ![Date audit](https://img.shields.io/badge/dates-original_sources-edb649)

## Assignment submission

- **Question 1 / Live URL:** *Not deployed yet*. Publish `index.html` using Netlify Drop / Vercel after authorizing access, and replace this field with the actual verified URL.
- **Question 1 / GitHub repository:** *Not created yet*. Create a public GitHub repository, push this directory, and replace this field with the verified URL.
- **Question 2 / What does the timeline show?** See [ANSWER_QUESTION_2.md](ANSWER_QUESTION_2.md).

## Start locally

Double-click `index.html`. It is **fully self-contained**: CSS, JS, and all 32 milestone records are embedded; works without internet except links to research papers. You can also run `python3 -m http.server 8000` in this directory and open `http://localhost:8000`.

## Features

- **32 milestones dated in original release order**, searchable and filterable by *engineering trade-off* (category does not alter chronology).
- **An educational causal attention heatmap**, which changes when different conceptual families are selected. The matrix is deliberately *schematic*, not model-extracted data or a real benchmark.
- **Pros, cons, mechanism, recommendation, and direct primary-paper links** for every item.
- **KV-cache calculator** comparing MHA/GQA/MQA over 2K–1M contexts. Assumptions are visible: 32 decoder layers, 32 query heads, 128 head dimension, 2 bytes per element, no quantization, no batching. Storage = `2 × N × n_kv_heads × head_dim × bytes × layers`. In GiB, divide by 1024^3.
- **Dense N×N attention score entry counter.** Dense causal attention computes only legal positions, but quadratic scaling still holds.
- **Two-item compare view**, step-by-step timeline playback, year jump, clickable sources, keyboard left/right navigation.
- **Responsive layout and zero dependencies.** All display logic is locally executed.

## Date audit policy

1. The primary date is the **first public arXiv v1 submission** (UTC date), not the acceptance date, journal year, or most recent arXiv revision. Equal-day entries are tied, not claims about which idea was earlier that day.
2. Where arXiv is inappropriate, label **official product release** or **contemporary public community record** explicitly.
3. Some rows represent **position encodings, kernels, cache management, or memory systems**, rather than brand-new attention score functions. They are intentionally included because the assignment requires the *history of attention efficiency* and its adjacent engineering decisions. Category is explicitly shown on every card.
4. Do **not** present estimates, toy masks, or year-arc summaries as empirical measured values.
5. Citations in the dataset correspond to **original research**, not third-party summaries. NTK-aware scaling alone is a documented community exception.

### Complete source-to-date mapping

| # | First public date (UTC) | Milestone | Date source | Basis |
|---:|:---|:---|:---|:---|
| 01 | 2017-06-12 | Scaled dot-product attention | [Original paper](https://arxiv.org/abs/1706.03762) | arXiv v1 |
| 02 | 2017-06-12 | Sinusoidal positions | [Original paper](https://arxiv.org/abs/1706.03762) | arXiv v1 |
| 03 | 2017-06-12 | Learned absolute positions | [Original paper](https://arxiv.org/abs/1706.03762) | arXiv v1 |
| 04 | 2018-02-15 | Local neighborhood attention | [Original paper](https://arxiv.org/abs/1802.05751) | arXiv v1 |
| 05 | 2018-03-06 | Relative position representations | [Original paper](https://arxiv.org/abs/1803.02155) | arXiv v1 |
| 06 | 2019-04-23 | Sparse Transformer | [Original paper](https://arxiv.org/abs/1904.10509) | arXiv v1 |
| 07 | 2019-11-06 | Multi-query attention (MQA) | [Original paper](https://arxiv.org/abs/1911.02150) | arXiv v1 |
| 08 | 2019-11-13 | Compressive Transformer | [Original paper](https://arxiv.org/abs/1911.05507) | arXiv v1 |
| 09 | 2019-12-25 | Explicit top-k sparse attention | [Original paper](https://arxiv.org/abs/1912.11637) | arXiv v1 |
| 10 | 2020-04-10 | Sliding-window + global attention | [Original paper](https://arxiv.org/abs/2004.05150) | arXiv v1 |
| 11 | 2020-06-29 | Linear attention | [Original paper](https://arxiv.org/abs/2006.16236) | arXiv v1 |
| 12 | 2020-07-28 | BigBird | [Original paper](https://arxiv.org/abs/2007.14062) | arXiv v1 |
| 13 | 2020-09-30 | Performer (FAVOR+) | [Original paper](https://arxiv.org/abs/2009.14794) | arXiv v1 |
| 14 | 2021-04-20 | Rotary position embeddings (RoPE) | [Original paper](https://arxiv.org/abs/2104.09864) | arXiv v1 |
| 15 | 2021-08-27 | ALiBi | [Original paper](https://arxiv.org/abs/2108.12409) | arXiv v1 |
| 16 | 2022-05-27 | FlashAttention | [Original paper](https://arxiv.org/abs/2205.14135) | arXiv v1 |
| 17 | 2023-05-22 | Grouped-query attention (GQA) | [Original paper](https://arxiv.org/abs/2305.13245) | arXiv v1 |
| 18 | 2023-06-27 | Position Interpolation (PI) | [Original paper](https://arxiv.org/abs/2306.15595) | arXiv v1 |
| 19 | 2023-06-29 | NTK-aware RoPE scaling | [Original community post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) | Original Reddit post; date cross-checked 2023-06-29 |
| 20 | 2023-07-17 | RetNet retention | [Original paper](https://arxiv.org/abs/2307.08621) | arXiv v1 |
| 21 | 2023-07-17 | FlashAttention-2 | [Original paper](https://arxiv.org/abs/2307.08691) | arXiv v1 |
| 22 | 2023-08-31 | YaRN | [Original paper](https://arxiv.org/abs/2309.00071) | arXiv v1 |
| 23 | 2023-09-12 | PagedAttention (vLLM) | [Original paper](https://arxiv.org/abs/2309.06180) | arXiv v1 |
| 24 | 2023-09-29 | Attention sinks / StreamingLLM | [Original paper](https://arxiv.org/abs/2309.17453) | arXiv v1 |
| 25 | 2023-10-03 | Ring Attention | [Original paper](https://arxiv.org/abs/2310.01889) | arXiv v1 |
| 26 | 2024-05-07 | DeepSeek Multi-head Latent Attention (MLA) | [Original paper](https://arxiv.org/abs/2405.04434) | arXiv v1 |
| 27 | 2024-06-10 | DeltaNet / delta rule | [Original paper](https://arxiv.org/abs/2406.06484) | arXiv v1 |
| 28 | 2024-10-07 | Differential Transformer | [Original paper](https://arxiv.org/abs/2410.05258) | arXiv v1 |
| 29 | 2024-12-09 | Gated DeltaNet | [Original paper](https://arxiv.org/abs/2412.06464) | arXiv v1 |
| 30 | 2025-02-16 | Native Sparse Attention (NSA) | [Original paper](https://arxiv.org/abs/2502.11089) | arXiv v1 |
| 31 | 2025-09-29 | DeepSeek Sparse Attention (DSA) | [Official release](https://api-docs.deepseek.com/news/news250929/) | Official release 2025-09-29 |
| 32 | 2025-12-13 | DroPE — drop positional embeddings | [Original paper](https://arxiv.org/abs/2512.12167) | arXiv v1 |

## Significant attribution / correctness traps

- **Attention vs positions:** Sinusoidal and learned absolute position methods are both discussed in Vaswani et al.'s 2017 Transformer paper. They do not improve the underlying N² score cost. We anchor *these Transformer variants* in June 2017 without claiming the first-ever invented position embeddings.
- **Local attention vs Longformer:** Local-neighborhood Transformer attention is already described in 2018's Image Transformer (vision). Longformer's later 2020 local **plus global** pattern is a separate, more text-oriented milestone.
- **Sparse layout vs explicit top-k:** Sparse Transformer (2019-04) uses predetermined patterns; Explicit Sparse Transformer (2019-12) selects the strongest logits. Top-k selection may still require dense scores or an indexer. The Session 8 transcript's early **top-k** discussion is about **output sampling**, which is not the same thing as top-k *attention*.
- **MQA/GQA:** MQA (2019) reduces K/V to a single shared KV head. GQA (2023) retains separate **query heads** while sharing **K and V** per group. Queries are NOT shared within the group.
- **KV cache and N²:** The number of cached K/V entries in standard Transformer decoding scales as O(N); *pairwise attention scores/work* scale as O(N²) over a prefill sequence. GQA primarily reduces cached KV width, not the N² class of dense full-sequence attention.
- **FlashAttention:** Exact attention algorithm preserving mathematical softmax output with more efficient IO. Despite O(N) extra attention-memory footprint, its *arithmetic* is still O(N²).
- **NTK-aware scaling:** June 29, 2023 is supported by [a contemporaneous ExLLaMA issue](https://github.com/turboderp/exllama/issues/115) referencing [bloc97's original Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/). There is no invented arXiv publication for the original technique. A community timeline is inherently less formal than arXiv and should be labeled.
- **State-memory algorithms:** Linear attention and DeltaNet recurrent state have different long-range associative quality compared to exact softmax. **Delta rule** is a much older learning concept; June 10, 2024 is the scalable **DeltaNet paper**, not the invention of the underlying learning rule.
- **Attention sinks:** Stable generation for millions of streaming tokens is **not** equivalent to faithful random access to a million tokens of arbitrary historical content.
- **DeepSeek:** *MLA* was published **May 7, 2024** in DeepSeek-V2 and reduces KV representation/storage. *Native Sparse Attention (NSA)* appears **Feb 16, 2025** as a distinct DeepSeek research proposal involving compressed summaries, selected blocks and windows. *DeepSeek Sparse Attention (DSA)* was **released Sep 29, 2025** with V3.2-Exp, using lightning indexing and fine-grained token selection. They are not the same method and none should be given the others' date.
- **DroPE vs DRoPE:** The requested **DroPE** means *Dropping Positional Embeddings after training*, first posted **Dec 13, 2025**. **DRoPE**, *Directional Rotary Position Embedding* (2025-03), is a different work in agent trajectory modeling. Don't mix them.
- **Session 7 correction:** The uploaded Session 8 transcript as extracted does not make the particular Session 7 correction independently identifiable. The app checks each original paper rather than attributing a speculative correction to the instructor.

## Source material

- ERA V5 session 8 transcript (uploaded August 15, 2026 lecture), plus user-supplied sessions 2–7 for teaching context.
- [Session 8 YouTube video](https://www.youtube.com/watch?v=sMgWxXFAvHM). Video is given as a reference; the project uses the supplied transcript as its lecture source and does not claim independent verification of its entire video runtime.
- Each research row's link in the table above is a paper or an official, attributable public record.

## Rebuilding the single-file page

`build_data.py` generates `mechanisms.json` from curated records. `index.template.html` has three placeholders (`__DATA__`, `__CSS__`, `__JS__`). To update the offline HTML after editing data/CSS/JS, run:

```bash
python3 build_data.py
python3 build_site.py
python3 validate.py
```

## Deployment

**Netlify Drop (no command line required):** Sign into Netlify; deploy by dragging this project directory (containing `index.html`) into the site's deploy area. Copy the real `https://…netlify.app` URL after it is published. Set a custom site name if desired.

**Vercel:** Import a GitHub repo, choose **Other** / static framework (no build command), set project root to this folder, and deploy. Copy its actual Vercel URL.

**GitHub:** Create a new public repository (e.g. `attention-atlas-era-v5`). Then in this folder:

```bash
git init
git add index.html README.md ANSWER_QUESTION_2.md mechanisms.json styles.css app.js build_data.py build_site.py validate.py
 git commit -m "Build source-audited attention timeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/attention-atlas-era-v5.git
git push -u origin main
```

Do not submit placeholder links. Only submit live verified endpoints once published.

## Limitations

- Interactive teaching visualizations approximate *structure*, never run a real neural network. Some architectures (e.g. linear attention / retention) are represented conceptually, not as equivalent attention matrices.
- No benchmarking was run; this project makes no measured speed, perplexity, or fidelity claims of its own.
- Formula labels are abbreviated teaching formulas, not drop-in complete implementations.
- Novel designs can postdate the lecture; the selection is intentionally broader than the session list, with an explicit extra label.
