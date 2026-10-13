# Question 2 — What does the timeline reveal?

The dates reveal a **series of independent, repeated bottlenecks**, not one race to replace softmax. In 2017 the Transformer gave every token a powerful exact route to earlier tokens, but paid O(N²) to do so. As early as 2018–20, local and sparse designs tried to reduce *which tokens communicate* and linear models tried to replace token-level retrieval with a persistent state. The two approaches sacrificed different things: sparse attention loses inaccessible edges; recurrent states compress history and can lose detailed associative recall.

**The first important reversal comes in 2019:** MQA did not try to improve the attention scoring rule at all. It observed that *inference memory bandwidth* had become the constraint and aggressively shared K/V heads. GQA in 2023 was a partial retreat from that savings, restoring additional KV heads to reclaim quality. Later MLA in 2024 took a different route: compress the KV information into a learned latent instead of sharing one literal cache for everything.

**The second reversal starts in 2021–23:** RoPE improved relative positional representation, but the longer the context, the more positions beyond pretraining were a problem. Position Interpolation, NTK-aware scaling, and YaRN made it feasible to use *more* tokens, but none erased the hardware cost or guaranteed that a model could reliably recover an early sentence. StreamingLLM showed that an apparently unimportant first token could be essential to stable bounded-cache streaming, while the rest of old history might still be discarded.

**The most surprising point is that speed and approximation were uncoupled in 2022.** FlashAttention showed that careful GPU IO and tiling can speed *exact* softmax without reducing which keys a query can see. PagedAttention tackled cache allocation, not model expressivity. These two hardware/serving advances must not be confused with MQA, sparse attention, or linear attention.

**Then the field revisited what it first tried to avoid.** Modern large-context work combined compression and selective sparse retrieval: DeepSeek's MLA (May 7, 2024) attacks KV **storage**, whereas later DeepSeek Sparse Attention (September 29, 2025) attacks *which positions get computed*. Between them, DeltaNet and Gated DeltaNet revisited linear states but made memory writes and forgetting more selective. Rather than discard the dense baseline, present-day designs stack tricks along orthogonal axes: position, exactness, state, KV memory, compute, and kernel efficiency.

**What I would predict:** future designs will increasingly make *where to look* an inexpensive learned operation while preserving an exact, verifiable fallback for hard retrieval. The real metric will be quality at a fixed total serving budget—rather than advertised context length alone. This is a reasoned prediction, not a verified publication claim.

## Additional mechanisms not explicitly required

- **Image Transformer local-neighborhood attention — Feb 15, 2018:** [Parmar et al.](https://arxiv.org/abs/1802.05751). Early local attention before Longformer.
- **Relative Position Representations — Mar 6, 2018:** [Shaw et al.](https://arxiv.org/abs/1803.02155). Represent pairwise distance explicitly.
- **BigBird — Jul 28, 2020:** [Zaheer et al.](https://arxiv.org/abs/2007.14062). Local/random/global sparse links.
- **FlashAttention — May 27, 2022:** [Dao et al.](https://arxiv.org/abs/2205.14135). Exact attention with IO-aware tiling.
- **PagedAttention — Sep 12, 2023:** [Kwon et al.](https://arxiv.org/abs/2309.06180). Virtual-memory-style KV cache blocks.
- **Native Sparse Attention — Feb 16, 2025:** [Yuan et al.](https://arxiv.org/abs/2502.11089). Hierarchical sparse attention with compressed / selected / sliding branches.

## Essential corrections and caveats

- “Top-k” in the transcript refers to **token sampling** at output; top-k **attention selection** is an independent mechanism.
- The name **DroPE** refers to [dropping positional embeddings after pretraining](https://arxiv.org/abs/2512.12167), dated December 13, 2025; do not conflate it with the unrelated DRoPE positional trajectory paper.
- Million-token *streaming* does not automatically equal million-token *accurate recall*.
- The app does not claim to have been deployed or that its toy matrices are actual model measurements.
