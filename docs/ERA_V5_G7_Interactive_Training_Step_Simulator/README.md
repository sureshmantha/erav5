# ERA V5 — The Anatomy of a Training Step

Single-file, offline-capable HTML/CSS/JavaScript teaching simulator for Module 7.

**Open:** `index.html` in a modern browser. No npm, CDN, GPU, or web service is required. Deploy the folder to Netlify or Vercel as a static site if desired.

## Interactive lessons

- Ten chronological phases from Slurm allocation and environment validation through batch consumption, forward/backward, collectives, optimizer, durable checkpoint, evidence and recovery.
- GPU rank lanes (2/4/8), selected microbatch accumulation (1/2/4), state inspector `S_k` and staged versus committed data ledger cursor.
- Scenario experiments: successful step, rank crash before commit, interrupted checkpoint seal, crash after durable commit, and changed topology on restore.
- Auto-run or manual step navigation, pause/reset, normal/slower/faster playback, independent five reliability-layer legend, and semantic event history.
- FP32 nonassociativity using `Math.fround` illustrates why changing collective order can break bitwise identity.

## Truthfulness / scope

**THIS IS A CONCEPTUAL SIMULATION.** The application never invokes Slurm, `nvidia-smi`, NVIDIA GPUs, NCCL or a real persistent object store. Rank cards, step/optimizer values, evidence events and checkpoint acknowledgments are illustrative. The five status cards say `ILLUSTRATED`, not `PASSED` or `CERTIFIED`. The fifth (scientific reproducibility) remains `NOT TESTED` because true bit-equal replay requires deterministic executable tests, not an animation.

`S_k = (weights, optimizer state, LR/scaler, RNG streams, sampler/ledger, run config, parallel mapping, auxiliary state)` is simplified visually. Correct recovery needs all state plus the same future inputs/operations. A durable checkpoint differs from an in-memory optimizer update. No literal zero-recomputation promise is made for periodic snapshots.

The pre-commit crash scenario demonstrates a lost volatile optimizer update and replay of a step. The checkpoint seal failure shows retry without optimizer recomputation **provided the ranks and in-memory state survive**. The post-commit crash demonstrates surviving committed state (conditional on real persistence). Changed topology demonstrates why fixed-order bitwise replay cannot be assumed.

## Sources and technical mapping

Course-provided 10-stage first-principles description and five independent reliability layers. Team 07 infrastructure design discussions on queue accounts, GPU/Xid diagnostics, and CPU checkpoint/new-process replay.

- [Slurm documentation](https://slurm.schedmd.com/documentation.html)
- [PyTorch Distributed](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [NVIDIA NCCL Tests](https://github.com/NVIDIA/nccl-tests)

### How to test manually

1. Open and press **Next event** 7 times. The volatile counter must change to `S1` but durable remain `S0`.
2. Advance once to stage 8. Durable becomes `S1`; ledger cursor also advances.
3. Reset, choose **Rank fails before commit**, advance to stage 8 and click **Restore and continue**. The volatile update returns to the durable state and the replay counter increases.
4. Choose **Checkpoint seal fails**, advance to stage 8 and click **Retry durable commit**. The optimizer update survives; sealing may be retried without an extra optimizer step.
5. Choose **Crash after commit**, advance to stage 10; restoration keeps the acknowledged `S1`.
6. Choose **Restore with changed topology**, induce crash/recover; the inspector must flag changed DP and scientific bit-equality is `NOT ASSURED`.
7. Inspect the reliability-layer cards; none claim real-world certification.
8. Resize browser to phone width and inspect accessibility, keyboard focus and overflow.
