# Bharat V5-40B: Mixture and Curriculum Specification

> **Status:** Design complete; proxy experiments not yet executed  
> **Version:** 1.0 — 30 July 2026  
> **Decision owner:** Suresh Kumar Mantha  
> **Submission artifact:** GitHub `README.md`  
> **Scope:** Text-first pretraining mixture, curriculum, data gates, and 1B/3B proxy validation plan

---

## 1. Executive decision

This specification proposes a **40B-class dense, text-first foundation model** optimized for:

1. **Coding and software engineering**
2. **Long-horizon agentic work with tools and failure recovery**
3. **Controllable reasoning depth**
4. **India-first knowledge and high-quality capability across 22 Indic languages**
5. **Useful long-context behavior rather than a nominal context-window claim**
6. **Strong general knowledge, science, and mathematics without catastrophic specialization**

The production run is **not authorized by this document alone**. Every percentage below is a hypothesis. The mixture is promoted to full scale only after controlled **1B- and 3B-scale proxy experiments** meet pre-declared statistical and no-regression gates.

### Frozen top-level decisions

| Decision | Value |
|---|---:|
| Target model | 40B-class dense decoder-only transformer |
| Tokenizer | 131,072-token, Brahmic-safe, byte-fallback tokenizer |
| Total pretraining exposure | **6.00T tokens** |
| Main curriculum | **5.70T tokens (95%)** |
| Capability anneal | **240B tokens (4%)** |
| Final cooldown | **60B tokens (1%)** |
| Total protected late-stage reserve | **300B tokens (5%)** |
| India/Indic allocation | **1.188T realized exposure (19.8%; designed as 20%)** |
| Indic tiers | 35% verified, 25% unverified, 20% translated, 20% synthetic |
| Context curriculum | 4K → 8K → 16K → 32K → 64K → 128K pilot |
| Proxy runs | 1B screening, then 3B confirmation |
| Full-scale decision | Only after all hard gates pass |

### One-sentence hypothesis

> A staged 6T-token curriculum with a protected 20% India/Indic lane, 18% code, explicit agentic/reasoning/long-context slots, and a quarantined 5% high-quality anneal reserve will outperform a flat mixture at equal compute, while improving Indic, coding, and agentic capability without losing more than 1–1.5 points on general English capability.

---

## 2. What “competes with OpenAI, Claude, and Gemini” means

A 40B open model should **not** claim universal parity with much larger proprietary systems. As of July 2026, the public frontier includes GPT-5.6 Sol, Claude Opus 5, and Gemini 3.6 Flash/3.5. Their public claims emphasize the same capabilities this plan prioritizes: long-running coding agents, tool use, computer use, controllable effort, self-verification, and token efficiency.

Examples of the current target frontier:

- OpenAI reports GPT-5.6 Sol at **80** on the Coding Agent Index, **88.8%** on Terminal-Bench 2.1, **92.2%** on BrowseComp with multi-agent execution, and **62.6%** on OSWorld 2.0.
- Anthropic positions Claude Opus 5 around long-running, multi-step coding, root-cause analysis, careful iteration, and cost-efficient knowledge work.
- Google reports Gemini 3.6 Flash improvements in DeepSWE (**49% vs 37%**), MLE-Bench (**63.9% vs 49.7%**), and OSWorld-Verified (**83.0% vs 78.4%**) over Gemini 3.5 Flash.

Therefore, **competitive** has three precise meanings:

1. **Selected-lane competitiveness:** approach frontier behavior on coding, tool use, India/Indic tasks, and long-context tasks after controlling for harness and inference compute.
2. **Efficiency competitiveness:** maximize correct tasks per generated token, tool call, second, and ₹/request.
3. **Sovereign differentiation:** decisively outperform frontier general models on India-specific and Indic-language evaluations.

This document does **not** claim that the proposed model will match frontier scores before the experiments are run.

---

## 3. Evidence ledger

The following are the numerical observations that materially shape the plan. Reported values are treated as design evidence, not independently verified facts.

| Scale or number | Meaning for this plan |
|---|---|
| **131,072 tokenizer vocabulary** | Strong balance among English, Brahmic scripts, code, math, and embedding cost. This is the default production tokenizer size. |
| English fertility around **1.24** and Hindi around **1.64** in the tokenizer discussion | Token counts are not equivalent to words or information. Fertility gates must pass before mixture accounting is trusted. |
| A nominal **1B** model becoming roughly **1.8–2B** | Large vocabularies can materially increase embedding/output-head parameters. Every proxy must report both transformer-body and total parameter counts. |
| V4 seeded path **2B → 5B → 9B → 120B** | Small models are useful for testing data decisions, but a small model cannot be assumed to predict all large-model behavior. |
| Planning drift **1→2B, 3→5B, 5→8B, 70→120B** | Architecture and tokenizer taxes must be calculated before compute is booked. |
| V4 **1.15T collected**, roughly **200B trained**, with selector-based effective exposure discussed as much larger | Raw, cleaned, unique, sampled, and effective tokens are different accounting units and must never be mixed. |
| Course design range **2.4T–4T** | This plan intentionally chooses **6T** because it targets a 40B overtrained production model, not a compute-optimal research checkpoint. |
| **40B target** | Fixed model-class assumption inherited from the earlier assignment. |
| Transcript phrase “40B × 20 = 200B” | Arithmetic correction: **40B × 20 = 800B**, not 200B. This plan uses **800B** as the Chinchilla-style compute-optimal reference and 6T as deliberate overtraining. |
| **4K → 8K → 16K → 32K → 64K** | Long context must be learned through genuinely long examples and length-specific batches, not by cutting long documents into 4K fragments. |
| **Last 1%** as best-quality data | Implemented as a physically protected **60B-token cooldown**. |
| **1B and 3B proxies** | Required promotion gates, not optional demonstrations. |
| OPUS-style selector claim of large efficiency gain | A gradient-alignment selector is useful, but it must remain bounded by protected floors and be ablated rather than trusted by assumption. |
| MoE discussion of very low active fractions | Architecture efficiency is separate from capability-lane mixture. This plan remains dense so the mixture experiment is interpretable. |

### Accounting terms

To avoid wishful accounting, this specification uses four different quantities:

- **Raw tokens:** before cleaning.
- **Unique clean tokens:** after extraction, normalization, filtering, deduplication, PII removal, licensing, and decontamination.
- **Scheduled exposure tokens:** tokens actually sampled for training, including controlled replay.
- **Loss-bearing tokens:** tokens on which loss is calculated. In agentic traces, tool outputs and environment logs are context, not prediction targets.

---

## 4. Model and tokenizer constitution

### 4.1 Model

- Dense decoder-only transformer, approximately 40B total parameters.
- Modern attention and MLP components may change, but all proxy arms must use the same architecture.
- Tied input/output embeddings unless proxy evidence shows a meaningful quality gain from untying.
- Text-first. Multimodal training is outside this assignment and must not silently consume the text budget.

### 4.2 Tokenizer gates

A tokenizer is accepted only if it meets all of the following on a frozen audit corpus:

| Gate | Target |
|---|---:|
| Visible-text round trip | 100% |
| NFC/Brahmic-safe normalization round trip | 100% |
| English fertility | ≤ 1.25 tokens/word |
| Hindi and major Indic languages | ≤ 1.75 tokens/word |
| Remaining scheduled Indic languages | ≤ 2.20 tokens/word |
| HumanEval-like code | ≤ 0.34 tokens/character |
| Invalid byte or ghost-token rate | 0 |
| Special-token collision | 0 |

The **131,072-token** tokenizer is the production default. A 196K candidate may be evaluated only as a separate tokenizer experiment; it must not be changed inside a mixture ablation.

---

## 5. The 6T capability budget

Every training sample receives exactly **one primary accounting lane**, preventing double counting. It may also have secondary tags such as `indic`, `code`, `long`, `agentic`, or `reasoning`.

| Capability lane | Realized share | Scheduled exposure | Why it exists | Primary data inventory | Primary evaluations |
|---|---:|---:|---|---|---|
| General web and world knowledge | **27.82%** | **1.669T** | Language, common sense, rare facts, culture, broad recall | FineWeb, FineWeb-Edu, DCLM, curated Wikipedia, books, news, legal and government text | MMLU-Pro, ARC-C, HellaSwag, factuality set |
| India/Indic | **19.80%** | **1.188T** | Sovereignty, language competence, India-first knowledge | Sangraha/IndicLLMSuite, government and education corpora, Samanantar/BPCC, IndicAlign, licensed native sources, verified synthetic data | MILU, IndicGenBench, FLORES+, BharatLens-2K, BharatAgent-500 |
| Non-Indic multilingual | **2.50%** | **150B** | Cross-lingual transfer and international utility | Curated high-resource multilingual corpora | XNLI, FLORES+, multilingual QA |
| Code and software engineering | **18.13%** | **1.088T** | Generation, debugging, repository understanding, tests | The Stack v2, permissive repositories, docs, issue/patch pairs, unit tests, compiler feedback | HumanEval+, MBPP+, LiveCodeBench, SWE-bench |
| Mathematics and formal reasoning | **8.55%** | **513B** | Arithmetic, algebra, proofs, symbolic reasoning | OpenWebMath/FineMath-class corpora, Proof-Pile-style data, verified generated problems | GSM8K, MATH-500, AIME-style hidden set |
| Science and technical knowledge | **8.50%** | **510B** | Physics, chemistry, biology, engineering, technical explanation | FineWeb-Edu, DCLM technical subsets, licensed papers/books, structured STEM data | GPQA, MMLU-Pro STEM, internal technical QA |
| Agentic and tool use | **5.58%** | **335B** | Planning, function calls, recovery, multi-step execution | ToolBench, API/function corpora, terminal/browser trajectories, generated verified tool tasks | BFCL-style tests, ToolBench, Terminal-Bench mini, BharatAgent-500 |
| Explicit reasoning traces | **4.72%** | **283B** | Controllable low/medium/high/ultra effort | Verified solution traces, self-checking traces, proof and planning data | BBH, MATH-500, GPQA, effort calibration |
| Native long context | **4.40%** | **264B** | Retrieval, synthesis, memory, repository/document coherence | Full books, judgments, reports, repositories, long multi-document tasks | RULER, LongBench, custom 32K/64K/128K probes |
| **Total** | **100.00%** | **6.000T** |  |  |  |

### Why the mixture is not more specialized

The general lane remains near 28% because broad and obscure knowledge often exists only on the web. Reducing it below the 10% protected floor produces a narrow specialist that may code or call tools but cannot understand normal user requests. Conversely, a web-heavy mixture would allow English abundance to erase Indic, agentic, and long-reasoning capability.

---

## 6. India/Indic lane: explicit tier accounting

The 20% headline is not accepted without a tier split.

### 6.1 Tier budget

| Indic tier | Share of Indic lane | Share of total run | Scheduled exposure | Planning target for unique clean supply | Implied replay/generation pressure |
|---|---:|---:|---:|---:|---:|
| Verified native | **35%** | **7%** | **420B** | 120–160B | 2.6–3.5× controlled replay |
| Unverified native | **25%** | **5%** | **300B** | 180–260B | 1.15–1.67× |
| Translated | **20%** | **4%** | **240B** | 100–140B | 1.7–2.4× |
| Synthetic | **20%** | **4%** | **240B** | 120–180B | 1.3–2.0× |
| **Total** | **100%** | **20%** | **1.200T design target** | 520–740B | Explicitly requires replay and generation |

The stage schedule realizes **1.188T**, a 12B rounding difference from the 1.200T design target.

### 6.2 Tier definitions

- **Verified native:** original-language text from high-trust sources or human-audited shards; valid license; language and script verified; no unresolved PII, duplication, or contamination.
- **Unverified native:** original-language web/native text that passes automated quality, language, script, safety, and provenance gates but has not received human semantic verification.
- **Translated:** high-value English or cross-Indic material translated with source linkage, translation-quality estimation, round-trip checks, named-entity preservation, and human audits.
- **Synthetic:** generated India-centric knowledge, reasoning, code, agentic, education, public-service, and cultural tasks accepted only after independent verification.

### 6.3 Language protection

The scheduled 22-language set is:

Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, and Urdu.

Rules:

1. **Full-run minimum:** each language receives at least **20B scheduled exposures**.
2. **Rolling-window floor:** each language receives at least **0.30% of global tokens** in every rolling 100B-token window unless a data-quality gate blocks the lane.
3. **Script diversity:** 80% of the Indic lane is native-script text, 10% romanized/code-mixed text, and 10% India-centric English or cross-lingual material.
4. **Residual allocation after floors:**

\[
w_l \propto 0.40D_l + 0.25Q_l + 0.20G_l + 0.15R_l
\]

Where:

- \(D_l\): measured user and public-service demand
- \(Q_l\): verified unique-token supply
- \(G_l\): benchmark performance gap
- \(R_l\): low-resource uplift

No language receives a larger share merely because low-quality data is abundant.

### 6.4 Replay rules

- A verified document may be sampled at most **4×** over the full run.
- An unverified document may be sampled at most **2×**.
- Translated or synthetic items must have source lineage and may be sampled at most **3×**.
- Near-duplicate clusters share a single replay budget.
- Replay rate is reported by tier, language, domain, source, and difficulty band.

---

## 7. Curriculum schedule

### 7.1 Stage mixture

Percentages below sum to 100 in every stage.

| Stage | Tokens | Gen | Indic | X-ling | Code | Math | Sci | Agent | Reason | Long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 — Foundation | 0.60T | 50 | 18 | 4 | 10 | 6 | 6 | 1 | 1 | 4 |
| S1 — Broad competence | 2.10T | 38 | 20 | 3 | 15 | 7 | 8 | 2 | 2 | 5 |
| S2 — Capability build | 1.80T | 22 | 20 | 2 | 21 | 10 | 10 | 6 | 5 | 4 |
| S3 — Specialization | 1.20T | 12 | 20 | 2 | 23 | 10 | 8 | 12 | 9 | 4 |
| S4 — Capability anneal | 0.24T | 10 | 20 | 1 | 20 | 10 | 10 | 12 | 12 | 5 |
| S5 — Final cooldown | 0.06T | 12 | 20 | 1 | 18 | 10 | 10 | 10 | 14 | 5 |

### 7.2 Stage intent

- **S0 Foundation:** learn language, syntax, common facts, basic math/code, and token semantics. Agentic and reasoning traces remain small because the model cannot yet absorb long expert trajectories efficiently.
- **S1 Broad competence:** establish world knowledge and multilingual stability. Introduce educational code, STEM, simple tools, and 8K examples.
- **S2 Capability build:** increase code, math, science, agentic, and explicit reasoning. Move to 16K and 32K examples.
- **S3 Specialization:** train repository-level coding, multi-tool work, failure recovery, long reasoning, and difficult India-specific tasks.
- **S4 Capability anneal:** use only high-quality, high-signal data preserved from the main sampler. No new weak source is introduced.
- **S5 Final cooldown:** the best 1% of data. Reduce learning rate, freeze mixture changes, and train only on verified or strongly validated examples.

### 7.3 Smooth transitions

A stage boundary is a **blend window**, not a hard switch.

- Last 5% of stage \(S_i\): linearly mix in the first 5% of \(S_{i+1}\).
- First 5% of \(S_{i+1}\): retain a decaying share of \(S_i\).
- A transition is paused if gradient-norm p99 exceeds **1.5×** the previous rolling median or if validation loss does not recover within **200 optimizer steps**.

---

## 8. Protected always-on floors

An OPUS-style gradient-alignment selector may rank samples, but it is **not allowed to cross these floors**.

| Lane | Protected floor |
|---|---:|
| India/Indic | 18% at S0; 20% from S1 onward |
| Verified Indic | 5.5% of global tokens at S0; 6% thereafter |
| General/world knowledge | 10% |
| Code | 10% |
| Math + science combined | 10% |
| Agentic | 1% before 45% of training; 5% after 45% |
| Explicit reasoning | 1% before 45%; 4% after 45% |
| Native long context | 4% after long-context training begins |
| Non-Indic multilingual | 1% |
| Each Indic language | 0.30% in a rolling 100B-token window |

Selector rules:

1. The selector may change a lane by at most **±15% relative** in a 1B-token selection window.
2. Changes are exponentially smoothed over 5B tokens.
3. It may not select from data that failed a hard data gate.
4. It may not consume the 300B anneal reserve before S4.
5. It is re-evaluated periodically because the model’s deficits change during training.
6. A fixed-mixture fallback remains available if selector overhead reduces throughput by more than 5% or destabilizes training.

---

## 9. Difficulty and reasoning-length bands

### 9.1 Difficulty bands

| Band | Description | Concrete example |
|---|---|---|
| D0 — Foundational | Explicit, short, single-concept | Explain a Python `for` loop; identify the capital of a state; add two two-digit numbers |
| D1 — Intermediate | One abstraction or short chain | Fix a single Python function using its unit test; solve a linear equation; summarize a 4K policy document |
| D2 — Advanced | Multi-step, cross-concept | Debug a multi-file module; solve an olympiad-style problem; complete a 3–5 tool workflow; compare two 16K reports |
| D3 — Expert | Repository/document scale with recovery | Patch a real repository issue; produce a graduate proof; execute 8–15 tool calls with failed-call recovery; synthesize 32–64K evidence |
| D4 — Research/frontier | Novel or weakly specified, long horizon | Migrate a large codebase; investigate a scientific question; complete a 20+ tool workflow; reason over 64–128K with self-verification |

Target distribution by stage:

| Stage | D0 | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|---:|
| S0 | 45 | 40 | 13 | 2 | 0 |
| S1 | 25 | 40 | 25 | 9 | 1 |
| S2 | 10 | 30 | 35 | 20 | 5 |
| S3 | 5 | 15 | 35 | 32 | 13 |
| S4 | 2 | 8 | 30 | 40 | 20 |
| S5 | 0 | 5 | 25 | 45 | 25 |

### 9.2 Reasoning-effort bands

| Effort tag | Target reasoning budget | Intended behavior |
|---|---:|---|
| `instant` | 0–128 tokens | Direct answer when no long chain is needed |
| `low` | 129–512 | Short calculation, explanation, or plan |
| `medium` | 513–2,048 | Multi-step analysis with one verification pass |
| `high` | 2,049–8,192 | Difficult proof, code task, or multi-source investigation |
| `ultra` | 8,193–32,768 | Long-horizon planning, branching, verification, and recovery |

The model is **not** trained to equate verbosity with intelligence. Reward includes:

- correctness,
- successful verification,
- calibrated stopping,
- tokens-to-correct,
- tool-calls-to-correct,
- recovery after failure.

### 9.3 Agentic loss mask

For an agentic trajectory:

```text
user → assistant plan → tool call → tool observation → assistant revision → final answer
```

Loss is calculated on assistant planning, tool-call arguments, revisions, and final answers. It is **masked** on user text, arbitrary tool observations, compiler logs, browser pages, and environment output. The model must see those tokens as context without being trained to reproduce them.

---

## 10. Long-context curriculum

A claimed context window is accepted only if the model is trained and evaluated on coherent examples at that length.

| Consumed-token interval | Maximum trained context | Notes |
|---|---:|---|
| 0–10% | 4K | Fast foundation |
| 10–45% | 8K | Documents, short code files |
| 45–70% | 16K | Multi-document and multi-file tasks |
| 70–90% | 32K | Repository and long-report reasoning |
| 90–99% | 64K | Long-context specialization |
| Final 1% | 128K pilot | Only verified long examples; promotion requires a long-context gate |

Rules:

- All sequences in a batch have the same target length.
- Global tokens per batch remain approximately constant; sequence count falls as context grows.
- Long documents are not randomly cut if the task depends on distant information.
- At least 30% of long-context examples require information from the first and final quartiles.
- At least 20% are adversarial “needle plus distractor” tasks.
- Short-context evaluations run at every long-context transition to catch regressions.

---

## 11. Data inventory and supply realism

| Lane | Open supply anchor | Supply conclusion |
|---|---|---|
| General | FineWeb/FineWeb-Edu, DCLM | Sufficient unique supply; quality selection matters more than repetition |
| Indic | IndicLLMSuite/Sangraha: 251B tokens and 74.8M instruction-response pairs across 22 languages | The 1.2T exposure target cannot be met with unique verified native data; replay, translation, and synthetic generation are explicitly budgeted |
| Parallel/translated Indic | Samanantar/BPCC-class parallel corpora | Useful seed, but insufficient alone; high-value domains must be translated and verified |
| Code | The Stack v2: roughly 900B full-token scale, 600+ languages | Supply is large, but permissive licensing, language selection, repository deduplication, tests, and issue/patch structure are essential |
| Agentic | ToolBench and API/function datasets | Public token supply is small; real and generated verified trajectories are mandatory |
| Reasoning | Open math/proof corpora plus generated verified traces | Quantity can be generated; correctness verification is the bottleneck |
| Long context | Books, judgments, reports, code repositories | Raw length is available; coherent tasks and contamination-safe evaluation are scarce |

### Data starvation response

If a lane cannot meet its scheduled unique-token target:

1. Improve extraction and cleaning before increasing replay.
2. Add licensed or public sources with full provenance.
3. Translate only high-value, undersupplied domains.
4. Generate synthetic data with independent verifiers.
5. Replay within the declared caps.
6. Reduce the lane share if the above steps still cannot support it.

A share is never retained merely because the plan originally contained it.

---

## 12. Data gates before any training token is admitted

The training sampler only reads shards whose manifest reports `gate_status: PASS`.

### 12.1 Pipeline

1. **Extraction:** remove navigation, cookie banners, repeated footers, malformed markup, and non-content blocks.
2. **Brahmic-safe normalization:** Unicode NFC, script-aware joiners/halants, zero-width character rules, HTML unescape, whitespace normalization.
3. **Language and script ID:** document and segment level; explicit code-mix labeling.
4. **Quality filtering:** heuristic plus model-based educational/technical quality.
5. **Exact and near deduplication:** local, source-level, and global.
6. **PII and secret scrubbing:** email, phone, address, credentials, API keys, tokens, private identifiers.
7. **Benchmark decontamination:** exact n-grams, MinHash/shingles, semantic similarity, repository/date cutoffs, canaries.
8. **License and provenance validation.**
9. **Determinism replay:** rerun a sample of shards and require identical hashes.
10. **Manifest publication.**

### 12.2 Hard gates

| Gate | Requirement |
|---|---|
| License/provenance completeness | 100% |
| Manifest completeness | 100% |
| Credential/API-key leakage | 0 accepted instances |
| Benchmark exact-match contamination | 0 accepted instances |
| Visible-text corruption after normalization | 0 |
| Language-ID confidence | ≥0.85 monolingual; code-mix top-two sum ≥0.90 |
| Exact duplicate rate after global dedup | <0.1% audit estimate |
| Near-duplicate rate | <2% within a sampled audit |
| Deterministic shard hash | 100% match on rerun sample |
| Human audit | ≥500 documents per source/language/tier before promotion |

Each shard manifest contains source URI or source ID, license, contributor, crawl time, cleaning-code hash, raw hash, clean hash, language distribution, token count, quality distribution, PII report, contamination report, dedup cluster ID, tier, domain, difficulty, context length, and replay count.

---

## 13. Testable hypotheses

| ID | Hypothesis | Confirmation metric | Confirmation threshold | Refutation |
|---|---|---|---|---|
| H1 | Staged curriculum beats a flat order at equal cumulative mixture | Weighted capability composite | ≥2.0-point gain; lower 95% CI > 0 | Gain <0.5 or any hard regression |
| H2 | The 5% protected anneal/cooldown improves difficult tasks | D2–D4 reasoning + agentic average | ≥3 points and overall composite ≥1.5 | No gain or general/Indic loss >0.75 |
| H3 | 20% Indic is superior to 10% without unacceptable English loss | Indic macro average; English general score | Indic +8 points; English loss ≤1.5; code loss ≤1.0 | Indic gain <5 or English loss >1.5 |
| H4 | 35/25/20/20 Indic tiers beat a native-heavy 50/30/10/10 split | Indic benchmark macro, cultural hallucination, translationese | Macro +3; hallucination −10% relative; English loss ≤0.5 | CI overlaps zero or quality regressions |
| H5 | Protected floors prevent selector-induced forgetting | Minimum lane score across checkpoints | No lane falls >1 point from fixed-mixture arm | Any repeated floor-lane collapse |
| H6 | Long-context ramp creates real 32K/64K capability | RULER/LongBench at 32K and 64K | +5 points over non-ramped control; ≤1-point short-context loss | No gain or short-context regression |
| H7 | Reasoning-effort tags improve efficiency | Accuracy at fixed token budget and tokens-to-correct | ≥2 points or ≥15% fewer tokens at equal score | More tokens with no quality gain |
| H8 | Cleaning and deduplication improve data efficiency | Validation NLL and downstream score at fixed tokens | ≥2% relative NLL gain or ≥1.5-point composite gain | No measurable improvement |

---

## 14. Proxy experiments

### 14.1 Common controls

Every arm must use:

- the same tokenizer;
- the same model architecture and exact parameter count;
- the same optimizer, learning-rate schedule, batch-token budget, and initialization family;
- equal total training tokens and approximately equal FLOPs;
- frozen development and hidden test sets;
- identical decontamination rules;
- checkpoint evaluation at predeclared token counts;
- complete run manifests.

Report **transformer-body parameters, embedding parameters, and total parameters**. A “1B” label without this breakdown is rejected.

### 14.2 Phase A — 1B screening

Train each arm for **20B tokens** and evaluate at 5B, 10B, 15B, and 20B.

| Arm | Mixture/curriculum |
|---|---|
| A0 — Flat | Same cumulative lane shares as the proposal, randomly mixed from token 1; no reserved anneal |
| A1 — Curriculum | Fixed S0–S5 stage order; late data is not physically quarantined |
| A2 — Curriculum + reserve | Fixed stages plus a protected 5% anneal/cooldown reserve |
| A3 — Full policy | A2 plus protected floors and bounded OPUS-style gradient-alignment selection |

Budget: **80B total proxy tokens** for the four arms. Repeat the best two with a second seed if the first-seed difference is below 2 composite points.

#### 1B decision

A3 advances if:

- its composite score is at least **2 points** above A0;
- the lower 95% bootstrap confidence bound is positive;
- no primary lane regresses by more than **1 point** versus A0;
- selector overhead is ≤5%;
- transitions meet gradient-stability gates.

If A2 beats A3, use the fixed curriculum and reject the selector.

### 14.3 Phase B — 3B mixture confirmation

Use the winning curriculum policy from Phase A. Train each arm for **60B tokens**.

| Arm | Total Indic share | Indic tier split |
|---|---:|---|
| B0 — Low-Indic control | 10% | 50/30/10/10 |
| B1 — Native-heavy | 20% | 50/30/10/10 |
| B2 — Proposed | 20% | **35/25/20/20** |

Run one seed for all three arms. Repeat the best two with a second seed. Maximum planned budget: **300B proxy tokens**.

#### 3B decision

B2 is promoted only if all are true:

1. Indic macro score is ≥8 points above B0.
2. English general score is no more than 1.5 points below B0.
3. Code score is no more than 1 point below B0.
4. B2 is ≥3 points above B1 on the Indic macro or reduces cultural hallucination by ≥10% relative while maintaining score.
5. Long-context, agentic, and reasoning hard gates pass.
6. The lower 95% confidence bound of the primary composite versus B0 is positive.
7. Results reproduce across the two finalist seeds.

### 14.4 Evaluation suite

| Lane | 1B screening | 3B confirmation/full model |
|---|---|---|
| General | ARC-C, HellaSwag, MMLU subsets, held-out NLL | MMLU-Pro, ARC-C, broad factuality |
| Indic | MILU subsets, FLORES+, custom held-out QA | MILU, IndicGenBench, BharatLens-2K, BharatAgent-500, cultural hallucination |
| Code | HumanEval+, MBPP+ | LiveCodeBench, SWE-bench Lite/Verified, repository unit tests |
| Agentic | Function-call exactness, ToolBench subset | Terminal-Bench mini/full, BFCL-style suite, tool recovery |
| Math/reasoning | GSM8K, BBH subsets | MATH-500, GPQA, hidden AIME-style set |
| Long context | RULER 8K/16K | RULER/LongBench at 32K/64K/128K |
| Efficiency | tokens-to-correct, latency, tool calls | task success per ₹, token, second, and tool call |
| Integrity | contamination canaries, PII probes | hidden contamination set, memorization and provenance audit |

### 14.5 Statistical method

- Use paired evaluation items whenever possible.
- Binary outcomes: paired bootstrap and McNemar test.
- Continuous rubric scores: 10,000-sample paired bootstrap.
- Report mean, 95% confidence interval, and both seed values.
- Primary model selection uses a frozen development composite; the hidden test is run once after the decision.
- Public benchmark results are not used to tune mixture weights after the hidden test is opened.

### 14.6 Primary composite

\[
S = 0.20G + 0.20I + 0.18C + 0.15A + 0.10R + 0.08M + 0.06L + 0.03E
\]

Where:

- \(G\): general/world knowledge
- \(I\): Indic/India-first
- \(C\): code
- \(A\): agentic/tool use
- \(R\): explicit reasoning
- \(M\): math/science
- \(L\): long context
- \(E\): efficiency/calibration/integrity

All component scores are normalized against the A0 or B0 control. **Hard no-regression gates override the weighted score.**

---

## 15. Results ledger — intentionally blank until runs complete

Fabricated proxy numbers would invalidate the assignment. Results must be committed with run IDs, configuration hashes, logs, checkpoints, and evaluation artifacts.

| Experiment | Run ID | Tokens | Seed | Composite | Indic | Code | Agentic | General | Long | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | TBD | 20B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A1 | TBD | 20B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A2 | TBD | 20B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A3 | TBD | 20B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B0 | TBD | 60B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B1 | TBD | 60B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B2 | TBD | 60B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 16. Full-scale promotion and stop rules

### Promote to 40B only when

- data-gating threshold is met for every required lane;
- the 3B proposed arm passes all hard gates;
- two finalist seeds reproduce the direction of improvement;
- replay pressure remains within declared caps;
- anneal reserve is fully manifest-locked;
- tokenizer fertility and round-trip gates pass;
- training throughput and stability estimates fit the approved compute envelope.

### Stop or redesign when

- any benchmark contamination is discovered;
- a protected language or capability lane misses its floor for two selection windows;
- gradient norm or loss diverges across a stage transition;
- synthetic or translated data shows source leakage, translationese, or cultural hallucination above threshold;
- agentic reward can be hacked without completing the task;
- selector overhead exceeds 5% without a compensating quality gain;
- a lane requires replay above the declared cap;
- the 3B experiment fails the no-regression gates.

---

## 17. Post-training handoff — outside the 6T pretraining budget

The assignment focuses on pretraining mixture and curriculum. The following is a separate planning envelope and must not be counted as pretraining tokens.

| Stage | Planning scale | Objective |
|---|---:|---|
| Supervised fine-tuning | ~12B loss-bearing tokens | Instruction following, code editing, tool calls, concise answers, multilingual conversation |
| Preference optimization | ~3B paired/ranked tokens | Helpfulness, style, safety, calibration, India-first preference data |
| RL/RLVR | 50K–200K verified prompts/trajectories | Math/code verifiers, terminal/browser success, failure recovery, long-horizon completion |
| Safety and policy | Dedicated hidden sets | Misuse resistance, privacy, reversible action, human escalation |

Tool observations remain loss-masked unless the target task is explicitly to model that environment.

---

## 18. Main risks and mitigations

| Risk | Mitigation |
|---|---|
| 40B cannot match universal frontier systems | Claim selected-lane and efficiency competitiveness only; publish gaps |
| Indic share is exposure-heavy but unique-data-light | Disclose unique supply, replay, translation, and synthetic pressure |
| Synthetic data amplifies teacher errors | Independent verifier, multi-teacher disagreement, human audits, source grounding |
| Translated data creates translationese | Native-language perplexity and classifier, named-entity checks, human comparison |
| English/web abundance erases scarce lanes | Always-on floors and rolling-window audits |
| Over-specialization hurts broad knowledge | 10% general floor and hard no-regression gates |
| Agentic data trains logs rather than decisions | Assistant-only loss mask; environment output remains context |
| Reward hacking | Hidden tests, state verification, anti-shortcut tests, task-completion checks |
| Long context is only nominal | Train on coherent target-length examples and test distant dependency |
| Abrupt curriculum shifts destabilize loss | 5% overlap windows, gradient/loss stop gates |
| Benchmark leakage inflates results | Exact, near, semantic, canary, repository/date decontamination |
| Vocab tax breaks compute plan | Report body/embedding/total parameters before booking compute |
| Proxy does not transfer to 40B | Require both 1B and 3B evidence; treat transfer as probabilistic, not guaranteed |

---

## 19. Repository structure

```text
.
├── README.md
├── validate_plan.py
├── configs/
│   ├── tokenizer.yaml
│   ├── proxy_1b_a0.yaml
│   ├── proxy_1b_a1.yaml
│   ├── proxy_1b_a2.yaml
│   ├── proxy_1b_a3.yaml
│   ├── proxy_3b_b0.yaml
│   ├── proxy_3b_b1.yaml
│   └── proxy_3b_b2.yaml
├── manifests/
│   ├── data_sources.parquet
│   ├── benchmark_canaries.json
│   └── anneal_reserve.lock
├── evals/
│   ├── bharatlens_2k/
│   ├── bharatagent_500/
│   └── hidden/
└── results/
    ├── proxy_1b.md
    └── proxy_3b.md
```

Run the arithmetic validator:

```bash
python validate_plan.py
```

The command must exit non-zero if a stage does not sum to 100%, the total is not 6T, the reserve is not 5%, or a protected floor is violated.

---

## 20. Final defended position

The plan does not bet on one secret dataset. It treats data design as a controlled system:

- **benchmarks define the target;**
- **inventory defines what is feasible;**
- **unique supply and replay are separately reported;**
- **floors protect scarce capabilities;**
- **curriculum controls when difficulty enters;**
- **annealing protects the highest-value data;**
- **cleaning and provenance decide whether tokens are admissible;**
- **1B and 3B proxies decide whether the numbers survive contact with evidence.**

The strongest part of this submission is not the 20% Indic share or the 18% code share by itself. It is that each number has a dataset source, a supply constraint, a benchmark, a falsifiable threshold, and a rule for changing the plan when the evidence disagrees.

---

## References

### Current frontier references — July 2026 snapshot

- OpenAI, “GPT-5.6: Frontier intelligence that scales with your ambition”: https://openai.com/index/gpt-5-6/
- Anthropic, “Introducing Claude Opus 5”: https://www.anthropic.com/news/claude-opus-5
- Google, “Introducing Gemini 3.6 Flash, 3.5 Flash-Lite, and 3.5 Flash Cyber”: https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber/
- Google DeepMind, “Gemini 3.5: frontier intelligence with action”: https://deepmind.google/blog/gemini-3-5-frontier-intelligence-with-action/

### Data and methodology

- Hoffmann et al., “Training Compute-Optimal Large Language Models”: https://arxiv.org/abs/2203.15556
- DataComp-LM / DCLM: https://arxiv.org/abs/2406.11794
- FineWeb and FineWeb-Edu: https://arxiv.org/abs/2406.17557
- IndicLLMSuite / Sangraha / IndicAlign: https://arxiv.org/abs/2403.06350
- AI4Bharat IndicLLMSuite repository: https://github.com/AI4Bharat/IndicLLMSuite
- Samanantar: https://arxiv.org/abs/2104.05596
- The Stack v2: https://huggingface.co/datasets/bigcode/the-stack-v2
- ToolBench: https://arxiv.org/abs/2307.16789
- SWE-bench: https://arxiv.org/abs/2310.06770
- RULER: https://arxiv.org/abs/2404.06654
- LongBench: https://arxiv.org/abs/2308.14508

---

## Execution-system addendum — Session 6

> For the Session 6 submission, this addendum supersedes the illustrative repository layout in Section 19. The executable repository root is documented by the top-level `README.md`.

The mixture above is now treated as **policy input** to a separate execution system rather than as an informal sampling suggestion. The reference implementation in this repository adds the following hard execution invariants:

1. Every training token must originate from an immutable, content-addressed tokenized shard whose manifest binds the frozen tokenizer hash, cleaning-pipeline hash, source/provenance, split, lane and content SHA-256.
2. Evaluation and validation are rejected twice: first at manifest admission and again at record admission. A final audit proves that every consumed source record belongs to a training shard.
3. Packing is data-type aware: best-fit/block-isolated text packing, greedy/block-isolated code packing, structure-preserving agentic traces, and single-document contiguous long-context chunks.
4. Agentic loss is applied only to assistant planning/tool-call/revision/final tokens; user prompts and tool observations remain visible context with loss masked to zero.
5. Each deterministic batch is content addressed over tokens, loss mask, attention mask, position IDs and source-token spans. Python process hashing is never used as the replay primitive.
6. S0–S5 stage weights and protected floors are compiled into a machine-readable plan before training. OPUS-style selection uses a frozen tiny-probe LM-head gradient cosine against a per-lane golden proxy; the proxy model/hash/scores are manifest-locked, and selection may be overridden only to preserve declared floors.
7. The consumption ledger records what the model saw; the learning ledger records what came back from the model (batch/sample/token losses, perplexity and gradient norm). The two are joined by batch ID and batch hash.
8. Checkpoints bind model state, optimizer state, RNG state, mixture-plan hash, tokenizer hash, next-batch cursor and exact ledger offsets.
9. Resume fails closed on any hash or ledger-offset drift; it must prove the first reconstructed post-checkpoint batch is exactly the expected batch.
10. Replay reconstructs historical batch IDs, source spans and hashes from immutable artifacts. Forking restores an earlier checkpoint and starts a separate lineage without mutating the main run.
11. Performance reporting includes raw non-PAD tokens, useful loss-bearing tokens, packing utilization and useful loss-bearing tokens/second using reconstructible formulas.

The execution demo intentionally uses a frozen byte tokenizer and a tiny CPU model. These are **test doubles for scale only**; they do not replace the production 131,072-token tokenizer, 40B model, 6T exposure budget, lane allocations, Indic tier split, protected late reserve, or the 1B/3B proxy gates defined above.
