#!/usr/bin/env bash
# Reproducible local-only checks. Does not claim Docker/Slurm or NVIDIA GPU execution.
set -euo pipefail
BASE="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"
python3 -m unittest discover -s tests -v
python3 sim/cpu_recovery_walkthrough.py --out evidence/CPU_END_TO_END_SIMULATION.json
python3 experimental/dcp/run_dtensor_smoke.py
python3 integrations/deepspeed_contract.py --config integrations/deepspeed_stage2.example.json
python3 -m compileall -q ckpt health integrations experimental sim tests
bash -n health/*.sh slurm-docker/scripts/*.sh scripts/*.sh slurm-docker/examples/*.sbatch
printf '\nLOCAL CPU CHECKS PASS. Actual Docker Slurm and real GPU NCCL remain pending.\n'
