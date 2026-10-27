# G7 config-driven model-stage and infrastructure capacity contract

**Scope:** trainer-selected V5 models, *unknown until G5 declares them*. Never turn illustrative V4 or external 1T numbers into V5 sizing defaults.

## 1. Model configuration interface

Every planned training run/phase supplies a versioned manifest (proposed schema; coordinate with G5/G1/G2):

```yaml
schema: g7-training-stage/1
programme: ERA-V5
model_id: <required-from-G5>
stage_id: <required-from-G5>
parent_checkpoint_id: null  # or immutable committed parent for approved growth
architecture_config_sha256: <required>
code_git_sha: <required>
container_digest: <required>
parameter_inventory:
  total_stored: <integer>
  trainable: <integer>
  frozen: <integer>
  active_per_token: <model-validated-estimate>
  base_weight_dtype: <dtype>
  optimizer_state_tensors: <name/dtype/shard-layout list>
  quantized_weight_bytes_and_metadata: <tensor inventory>
parallel_mesh:
  dp: <integer>
  tp: <integer>
  pp: <integer>
  ep: <integer>
  cp: <integer>
  nodes: <integer>
  gpus_per_node: <integer>
training:
  sequence_length: <integer>
  global_tokens_per_update: <integer>
  accumulation_steps: <integer>
  optimizer_name_and_config_sha256: <required>
  scheduler_config_sha256: <required>
  active_precision_and_kernel_hashes: <required>
data:
  tokenizer_digest: <required>
  curriculum_manifest_sha256: <required>
  sampler_and_ledger_protocol: <required>
checkpoint:
  serializer: <DCP/Megatron/other verified>
  stage_boundary: optimizer_step
  strict_resume_policy: <defined>
  durable_store: <IAM role reference, no credentials>
```

Angle-bracketed fields are **placeholders**, not discovered values. Machine-readable schema and CI validation should reject unknown required fields and impossible mesh products. GPU ranks used by the training job must satisfy the agreed parallel-degree topology, with documented exceptions such as hybrid sharding.

## 2. Calculate training-state and checkpoint capacity from actual tensors

For stored parameters, distinguish `P_stored = P_frozen + P_trainable`; `P_active` is the *compute path*, not the state that needs persistent storage. For conventional BF16 + FP32-master Adam **only**, at a clean optimizer boundary:

`checkpoint_bytes ≈ b_frozen*P_frozen + (b_train_weight + b_master + b_m + b_v)*P_trainable + quantizer_metadata + adapters_nonstandard + scheduler + RNG + ledger + sharding_metadata`.

A conventional unquantized choice might use `b_train_weight=2`, `b_master=4`, `b_m=b_v=4`; **do not multiply total MoE weights by 14** when the base is frozen or quantized and adapters are trained. The value `b_frozen` depends on exact frozen-weight serialization, e.g. compressed quantized representation and scales. A CPU/GPU memory envelope also adds `gradient + activations + comm buffers + allgather/inflight + allocator` — checkpoint file size is not peak HBM.

Use actual optimizer `state_dict` or distributed tensor inventory to measure, not only algebra. Maintain both decimal GB/TB and binary GiB/TiB.

## 3. Compute and token budget without double counting

Approximate planning compute for dense-like active paths: `C_approx ≈ 6 * P_active * tokens_executed`, with explicit correction for attention complexity, recurrence, routing, quantization, reversible recomputation and any fused kernels. At small sequence lengths this is still an approximation. Use profiling FLOPs if available.

- `tokens_plan`: planned useful corpus tokens (possibly repeated according to approved curriculum).
- `tokens_physical`: tokens actually executed by GPUs; includes rollbacks/replays.
- `tokens_committed`: tokens reflected in the latest durable model+ledger state.
- `tokens_unique_source`: distinct original corpus tokens; **different** from training exposures.
- `tokens_replayed = tokens_physical - logically_committed_progress`, carefully account for evaluation/probes separately.
- `step_token_count`: actual non-pad/non-masked loss tokens, not only `batch_size × context_length` unless fully dense.
- **MFU**: numerator/denominator and FLOP model must be explicitly specified; do not use GPU busy % as MFU.
- **Goodput**: define wall-clock and GPU-hour numerator; never multiply by a second downtime fraction if it is already included in measured wall throughput.

Session 16's 3B-active × 2,000-tokens/active-param = 6T is *one conditional exercise*, not a final V5 data contract. Session 14's 20T–45T range is conditional aspirational discussion; the 16× claim needs a validated method and quality-equal benchmark before becoming a budget credit.

## 4. Capacity decision procedure for each trainer-selected stage

1. Obtain G5's signed `model_id`/stage plan and parameter inventory; G1/G2 data ledger; G6 precision/parallel kernels; G8 event schema.
2. Make a stage-wise matrix of **stored**, **trainable**, **active/token**, **frozen**, and **checkpoint representation**. Ask whether this is a new model from scratch, a function-preserving growth from the V5 predecessor, or a post-training branch.
3. Build an untrained skeleton with the *actual tensor shapes* and a CPU distributed state inventory; never require the full model trained before benchmarking state size.
4. Probe GPU peak with candidate DP/TP/PP/EP/CP meshes. Establish communication correctness and sustained throughput; refuse to infer memory fit from shard totals alone.
5. Record save breakdown (device-to-host stall, NVMe write, remote durable commit, hash/manifest, restore/verify) and p50/p95/tail performance. Include overlap interference with useful training.
6. Compute checkpoint interval from measured critical-path overhead, failure rate, expected rework and remote durability lag. Keep exact replay and no lost *committed* checkpoint semantics separate from zero recomputation.
7. Run failover and resume on a pinned exact-topology lane; allow topology/optimizer/architectural stage changes only as **explicit new branches with different expected equality**.
8. Gate with loss/quality and run-level goodput; keep model quality as a hard constraint, never optimize purely for tokens/s.
9. Do not promise all three V5 models share checkpoint format, topology, optimizer or GPU hardware unless G5 explicitly validates the compatibility.

## 5. Two distinct checkpoint operations

**Ordinary restart:** same stage, identical model+optimizer+schedule+data state; require complete strict load and step-wise exact replay in the tested environment.

**Model growth/transition:** stage changes tensor shapes or model representation. This is an approved **conversion** with separate mapping tests and held-out loss continuity. You cannot claim identical raw tensor names, bytes or loss when architecture intentionally changes. Save immutable provenance `{parent_checkpoint, converter SHA, conversion spec, new stage ID, validation evidence}`. V4's step-3009 corruption and converter key-mismatch are distinct failure classes; test both.

## 6. Week 1 implementation first, then measured scale

Week 1: two-account Slurm test, JSON health with honest skipped, NCCL/Gloo explicit backend identity, CPU bit-equal loss and next-batch replay, failure injection, security scan. Week 2: real GPU/driver/topology identification, actual DCP/optimizer adapter integration, NCCL/DCGM, checkpoint throughput. Weeks 3–4: dry runs, rank/node failure, comm and checkpoint contention, Runbook and go/no-go. All final numerical budgets wait for actual G5 configuration.
