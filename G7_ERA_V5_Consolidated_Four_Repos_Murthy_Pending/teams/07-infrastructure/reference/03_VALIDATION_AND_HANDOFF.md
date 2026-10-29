# G7 unified submission — actual execution record and operator handoff

**Review timestamp:** 29 October 2026. **Source coverage:** four accessible ZIPs, fifth Murthy source **unavailable**. This is a reproducible *candidate submission*, **not** a production deployment or instructor-approved PR.

## Tests actually executed in the current container

| Acceptance | Result | Repro command | What it establishes |
|---|---|---|---|
| Combined Python unit suite | **PASS: 20 tests / 0 failures** | `python -m unittest discover -s tests -v` | CPU checkpoint/optimizer/RNG/ledger equivalence, Xid fixtures and fail-closed JSON, DeepSpeed adapter **mock**, static queue policy and 2-rank Gloo collective; see `evidence/UNIFIED_CPU_TESTS.log` |
| End-to-end CPU incident walkthrough | **PASS** | `python sim/cpu_recovery_walkthrough.py --out evidence/CPU_END_TO_END_SIMULATION.json` | Real CPU 12-step baseline and 5+7 checkpoint resume equality; scheduler/resource failure events are **synthetic** |
| Two-rank CPU DTensor DCP | **PASS** | `python experimental/dcp/run_dtensor_smoke.py` | Actual CPU sharded DCP save, process exit, new 2-rank load with tensor equality; **not** DS ZeRO states |
| DeepSpeed stage-2 config contract | **PASS_STATIC** | `python integrations/deepspeed_contract.py --config integrations/deepspeed_stage2.example.json` | Schema sanity and multiplication of DP ranks×microbatch×accumulation. No DeepSpeed installed/initialized |
| Compose YAML parse | **PASS_STATIC** | PyYAML `safe_load` | Six declared services only; doesn't validate Docker/Slurm runtime |
| Shell syntax | **PASS_STATIC** | `bash -n health/*.sh slurm-docker/scripts/*.sh ...` | Syntax, not runtime API validity |
| **Slurm Docker two-account actual enforcement** | **NOT RUN** | `cd slurm-docker && ./scripts/run_lab.sh` | Docker / Slurm binaries **unavailable** here; user must execute and attach sanitized logs |
| Real GPU inventory / DCGM / Xid | **NOT RUN** | `bash health/gpu_health.sh --mode prolog --require-gpu ...` | NVIDIA hardware **unavailable** |
| GPU NCCL all-reduce / vendor nccl-tests | **NOT RUN** | See `docs/health_design.md` | CPU Gloo is **not** NCCL |
| DeepSpeed ZeRO save/restart | **NOT RUN** | Team 5 DeepSpeed trainer integration | DeepSpeed not installed; config contract/adapter are the only local checks |
| Remote S3 durability / stale-writer CAS / node-loss recovery | **DESIGN ONLY** | Gate 2/3 integration tests | No cloud cluster or object store configured |
| Required `gitleaks detect --source .` | **NOT RUN** | `scripts/pre_push.sh` from Git repo | Gitleaks binary absent here; do not claim zero findings |
| Reviewed PR | **NOT CREATED** | `gh pr create ...` | Requires team submitter / lead acceptance |

## Local regressions found and fixed in this consolidation

- Initial unified test run stalled while Gloo collective test was exercised; a later bounded *standalone* Gloo test passed, and the complete 20-test rerun passed. Record failures/timeouts instead of quietly discarding them.
- **Three Xid-related tests failed** because the unified working tree omitted `health/fixtures/xids_sample.log`. Added a clearly synthetic fixture, reran and confirmed five health tests pass. This is a packaging regression, not a hardware result.
- Single-process checkpoint and optimizer replay preserved from the Suresh source. Supplemental CPU DTensor DCP round-trip separately implemented and run successfully.
- DeepSpeed adapter contract added with strict config-fingerprint and data-cursor checks; simulated engine tests are not a claim that actual DeepSpeed is installed.

## How to complete Week 1 acceptance

1. **Read** `reference/00_SOURCE_AND_SCOPE.md`, `01_ADR_COVERAGE_AND_GAP_ANALYSIS.md`, `02_SOURCE_CODE_REVIEW.md`. Decide owner of queue and whether mainrun preempts team jobs. Those decisions must be made on Axiom/GitHub, not silently inferred.
2. **Check Python** in the actual fork: `python -m unittest discover -s tests -v`. Preserve exact Python/PyTorch/NumPy versions and log (redact user paths/IDs if public).
3. **Run Docker Slurm** from `teams/07-infrastructure/slurm-docker`: `./scripts/init_env.sh && ./scripts/run_lab.sh`. Verify authorized accounts and QoS, denied unauthorized/over-limit requests and `sacct` attribution; capture real stdout and test exit code. A static read of `Flags=DenyOnLimit` doesn't prove scheduling.
4. **Approve JSON schema** with Team 8. If GPU present, capture `nvidia-smi` UUID/version, Xid, ECC/DCGM status, NCCL correctness; run `nccl-tests` in exclusive allocation for bus bandwidth and report actual hardware, message sizes and topology.
5. **Run secret scan** from root of user's Git fork: `gitleaks detect --source .`. The project may contain a locally generated `.env`; it must be gitignored and not included in any public PR. Only submit after 0 findings.
6. **Confirm the fifth source:** attach `murthy_Team7-infrastructure-main.zip` *as a file whose bytes are available*. Re-run its ADR, tests and source inventory before claiming a five-repo audit.
7. **Open one curated PR** under `teams/07-infrastructure/`, preserving teammate provenance. Protect PR from unnecessary source files beyond the assignment folder.

## Production verification checklist (NOT Week 1 code-complete)

- Team 5 approved exact model, ZeRO stage, distributed mesh and precision.
- DeepSpeed actual all-rank checkpoint save/load integration with optimizer/ZeRO stage, strict config and data ledger; no implicit parameter reinitialization.
- Remote durable generation with all-rank receipts, object integrity, conditional `latest` update and stale-writer fencing.
- Real Slurm requeue/signal behavior under node loss and QoS preemption/limits; resource isolation, telemetry and restoration checks.
- Failure-injection matrix: pre-seal save interrupt, corrupt shard, stale upload, bad optimizer key, CUDA/NCCL hang, Xid, node loss, changed world size and numerical NaN.
- Model quality and checkpoint goodput measured on actual hardware; no assumed per-step zero recomputation or bit equality across topology changes.

## Framework sources

[DeepSpeed checkpoint](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html), [ZeRO](https://deepspeed.readthedocs.io/en/latest/zero3.html), [TorchTitan checkpoint](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md), [Megatron DCP](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/checkpoints/dist_ckpt.html), [Slurm limits](https://slurm.schedmd.com/resource_limits.html).
