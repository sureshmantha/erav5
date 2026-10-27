# Browser verification — ERA V5 Training Step Lab

The interactive UI was exercised with headless Chromium and Playwright using the complete standalone HTML supplied in this folder. Browser content was loaded into Playwright directly, because container navigation to `file:` or localhost URLs is blocked in the test runner; this does not affect the normal browser behavior of the delivered file.

**Observed passing assertions:**

- Initial baseline checkpoint `S0`, no executed stages.
- Completing optimizer stage 7 produces volatile `S1` while durable state remains `S0`.
- Completing commit stage 8 advances durable to `S1`.
- Pre-commit rank crash restores volatile to `S0` and increments re-executed optimizer steps.
- Checkpoint seal interruption rejects commit; retry succeeds without re-executing the optimizer step while in-memory state survives.
- Changed-topology recovery shows DP-rank change and warns `NOT ASSURED` for scientific bitwise replay.
- JavaScript FP32 `Math.fround` demonstrates two reduction groupings producing `0` versus `1`.
- No page JavaScript exceptions during test.
- Responsive widths `1440px`, `820px`, and `375px`: no horizontal document overflow.

**Limitations:** These are front-end behavior assertions. They **do not** establish actual Slurm, GPU/NCCL, checkpoint storage durability, or real scientific bit-equality, which are intentionally out of scope for this educational visualization.
