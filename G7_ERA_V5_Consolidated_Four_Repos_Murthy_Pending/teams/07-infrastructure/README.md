# ERA V5 — Team 07 Infrastructure | Week 1 Gate Submission

**State:** CPU code executed and tested; Slurm Docker and CUDA paths supplied but **not executed here**.
**Path:** `teams/07-infrastructure/week1-gate/` · **Milestone:** Gate 1 — 31 Oct 2026 · **Module lead:** Suresh Mantha.

## What this submission gives the instructor

1. `docs/queue_design.md`: evidence-driven scheduler architecture and two distinct Slurm accounting accounts, with a Docker Compose CPU lab and scripts to **assert policy enforcement** (not just account existence).
2. `slurm-docker/`: Slurm controller, `slurmdbd`, MariaDB, MUNGE and two simulated CPU nodes; `labmain` and `labteam` Unix identities mapped to authorized Slurm `mainrun` and `teams` accounts; denied requests and `sacct` accounting proof.
3. `health/launch.sh`, `gpu_health.sh`, `xid_scan.py`: valid JSON, explicit `skipped` for missing tools, Xid fixture classification and a separate actual NCCL collective path. A CPU Gloo path proves code correctness **without claiming GPU validation**.
4. `docs/checkpoint_design.md`, `ckpt/ckpt.py`: deterministic CPU-only checkpoint with optimizer/scheduler/RNG/cursor preservation; immutable local generations, manifest hashes, fail-closed restore and **new-process loss-bit equality** across all resumed steps.
5. `tests/`, `evidence/`, `docs/ACCEPTANCE_MATRIX.md` and `scripts/pre_push.sh`: reproducible proof, failures and security gate.

### Start here

```bash
cd teams/07-infrastructure/week1-gate
python3 --version
bash scripts/setup_local_venv.sh
bash tests/run_cpu_tests.sh
# Run a manual CPU checkpoint:
python3 ckpt/ckpt.py train --directory /tmp/g7-ckpt --steps 5 --save
python3 ckpt/ckpt.py inspect --directory /tmp/g7-ckpt
python3 ckpt/ckpt.py train --directory /tmp/g7-ckpt --steps 12 --resume
# Health fixture should FAIL (exit 2) and print a valid JSON object:
python3 health/gpu_health.py --require-gpu --expected-gpus 8 --xid-source file \
  --xid-file health/fixtures/xids_sample.log
```

**Dependencies:** Python 3.10+, PyTorch and NumPy for checkpoint and collective tests; Docker Engine and Compose V2 for Slurm lab. At Week 1, local CUDA is optional. See `scripts/requirements-cpu.txt` for minimum compatible versions. Use approved lab/Colab environments if a GPU is necessary.

### Slurm account lab (must run on a Docker-capable machine)

```bash
cd teams/07-infrastructure/week1-gate/slurm-docker
./scripts/init_env.sh
./scripts/run_lab.sh
# On failure inspect the specific failing service:
./scripts/diagnose_lab.sh
# A successful run_lab.sh automatically copies evidence into ./evidence-from-docker/
# Verify QoS and accounting:
docker compose exec controller sacctmgr show qos
docker compose exec controller sacctmgr show assoc
# Optional (independent) exit-75 integration:
docker compose exec controller /lab/scripts/test_requeue.sh
```

**Do not call the Slurm task completed until the real `run_lab.sh` exits 0.** Docker is not installed in the environment where this submission was assembled, so those run results remain outstanding.

### Hardware health checks

```bash
# Always a single JSON result; 0 pass / 1 warn / 2 fail.
bash health/gpu_health.sh --mode prolog --expected-gpus 8 --require-gpu
# Full diagnostics only on an exclusive GPU node (can be disruptive):
bash health/launch.sh --full --report-only --expected-gpus 8 --require-gpu
# CPU distributed correctness only: not NVIDIA NCCL!
torchrun --standalone --nproc-per-node=2 health/nccl_probe.py --backend gloo
```

### Integrity and security

- Model states are checkpointed **after** a completed optimizer step. Incomplete gradient accumulation is excluded from the CPU acceptance; production must be designed explicitly for it.
- `ckpt.py` is a **CPU proof**, not a production DCP replacement; no S3 transaction, cross-node consistency or literal zero-recomputation claim is made.
- All local fixture data is synthetic. No Parquet corpus, cloud keys, GPU diagnostics with private identifiers, `.env` passwords or secrets should enter the public repository.
- Before each push: `gitleaks detect --source .` from the **repository root**. `scripts/pre_push.sh` enforces presence of Gitleaks and a zero exit code; it does not report success if missing.

### GitHub contribution workflow

```bash
# Within your own fork of theschoolofai/era-v5-lab:
git checkout -b infra-g7-week1-<yourhandle>
# Copy the topic folder into teams/07-infrastructure/week1-gate/
git add teams/07-infrastructure/week1-gate
git diff --cached --check
./teams/07-infrastructure/week1-gate/scripts/pre_push.sh
git commit -m "infra: week1 queue health checkpoint CPU evidence"
git push -u origin infra-g7-week1-<yourhandle>
gh pr create --repo theschoolofai/era-v5-lab --fill
```

The command `gitleaks` and creation of a GitHub PR have **not** been executed by the assistant; the team member must run them with appropriate access and preserve the results.

### Boundaries, test coverage and cross-team dependencies

The source check and tests in this package do not demonstrate a verified real-GPU, network-fault, 4-rank DCP or AWS distributed checkpoint system. For next steps, see `docs/ACCEPTANCE_MATRIX.md` and design details. Team 5 supplies approved model/mesh/config v0; Team 8 owns final health JSON schema; Teams 1/2 own consumable data/ledger semantics. No 1T parameter assumption is made; trainer stated V5 involves three model stages/targets with dimensions still to be confirmed.

**Upstream design provenance:** teammate-proposed G7 ADR 0001/0002/0003, trainer Sessions 6, 10–16 and LightningLM experience report. Documents are reconciled in `docs/ADRS_REVIEW_AND_DECISIONS.md` without claiming formal ADR acceptance.

## 2026-10-27 macOS hotfix notes

This revision repairs four observed blockers from the Mac report:

- **PyTorch 2.0.1 + NumPy 2.4.6**: cannot use the NumPy tensor bridge. `scripts/setup_local_venv.sh` creates `.venv` with NumPy 1.x and checks compatibility. The harness prefers `.venv/bin/python`, and nested child processes use the **same** `sys.executable`. No global Anaconda environment change is required.
- **Gloo timeout**: CPU proof now launches two processes explicitly using `MASTER_ADDR=127.0.0.1`, a free port and the macOS `lo0` interface. `torchrun` remains available for actual NCCL training but is not required for the local CPU-only fixture.
- **MUNGE socket access**: `/run/munge` changed from 0700 to 0755 so the Slurm identity can connect; a non-root auth round trip is checked before services start.
- **Docker Desktop topology/systemd**: the teaching lab uses `SlurmdParameters=config_overrides`, explicit virtual CPU topology, cgroup v2's test-only `IgnoreSystemd=yes` and `JobAcctGatherType=jobacct_gather/none`. These are **test-only configuration choices**. They verify Slurm **scheduler policy**, not OS-level CPU/GPU isolation, and must NOT be copied into actual GPU cluster configuration. Privileged containers are unnecessary here.
- **Over-limit enforcement**: `Flags=DenyOnLimit` makes a request exceeding `MaxTRESPerJob=cpu=1` a submission-time rejection rather than a pending job.

The Docker changes have been **statically checked** but could not be run in this execution environment (Docker/Slurm unavailable). If still blocked on your machine, use `cd teams/07-infrastructure/week1-gate/slurm-docker && ./scripts/diagnose_lab.sh` and share the **redacted** service logs. Do not disclose `.env`, MUNGE keys or raw secrets. A nonzero `run_lab.sh` means the Slurm acceptance remains PENDING.

---

## Unified four-source audit + DeepSpeed North Star (29 Oct 2026)

**Important provenance:** This folder incorporates reviewed design and useful implementation patterns from the available Pradeep, Mallik, Max and Suresh archives. The user also requested `murthy_Team7-infrastructure-main.zip`, but its file bytes were not available in the conversation runtime; **Murthy has not been audited or merged**. See [`reference/00_SOURCE_AND_SCOPE.md`](reference/00_SOURCE_AND_SCOPE.md) and the four-repo SHA-256 file inventory. Do **not** describe this candidate as a completed five-repo audit.

**Start here:**

1. [`reference/01_ADR_COVERAGE_AND_GAP_ANALYSIS.md`](reference/01_ADR_COVERAGE_AND_GAP_ANALYSIS.md): 36 DeepSpeed/TorchTitan/Megatron-derived architecture decision topics, repo coverage, conflicts, and Week 1–4 roadmap. Topics are our proposed ADRs, **not official DeepSpeed numbered ADRs**.
2. [`reference/02_SOURCE_CODE_REVIEW.md`](reference/02_SOURCE_CODE_REVIEW.md): strengths, differences, weaknesses and merge decision per repository.
3. [`reference/03_VALIDATION_AND_HANDOFF.md`](reference/03_VALIDATION_AND_HANDOFF.md): exact 20-test suite result, distributed CPU DCP and CPU simulator results, honest missing hardware/Docker/DeepSpeed/gitleaks gates.

**New executable, CPU-safe extensions:**

```bash
python -m unittest discover -s tests -v
python sim/cpu_recovery_walkthrough.py --out evidence/CPU_END_TO_END_SIMULATION.json
python experimental/dcp/run_dtensor_smoke.py
python integrations/deepspeed_contract.py --config integrations/deepspeed_stage2.example.json
```

- `experimental/dcp/run_dtensor_smoke.py` saved and reloaded sharded **CPU DTensor** state using PyTorch DCP across **two separate 2-rank process groups**. This is a real DCP API proof, but **not** a DeepSpeed ZeRO optimizer checkpoint test.
- `integrations/deepspeed_contract.py` is a **shape/configuration guard and integration boundary** for Team 5's eventual actual DeepSpeed engine; only a mocked save/load contract and static config were tested. It does **not** claim DeepSpeed, CUDA or zero-offload works on this host.
- `sim/cpu_recovery_walkthrough.py` uses actual CPU training and checkpoint replay, while clearly labeling its queue/hardware events **synthetic**.
- Previous `slurm-docker/` remains the Week 1 two-account runtime acceptance path, which must actually pass on a Docker-capable computer; static config tests alone are insufficient.

**No assumed trillion-parameter training baseline.** Team 5 controls model size, optimizer, ZeRO stage, mesh and token plan. Checkpoint bytes, GPU counts, LR and goodput must be measured for each approved ERA V5 model/stage.

**Single-command CPU validation:** `bash scripts/validate_unified_cpu.sh`; it never asserts real Docker/Slurm or CUDA results.
