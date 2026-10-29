#!/usr/bin/env bash
# FUTURE GPU SITE TEMPLATE ONLY. Not executed in Week 1 Docker CPU environment.
# Assumes site has gpu partition, GRES, /training shared storage, Trainer G5 entrypoint,
# and approved checkpoint-on-preemption integration. Fail rather than run with placeholders.
#SBATCH --job-name=era-v5-train
#SBATCH --account=mainrun
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=era-v5-train-%j.log
set -euo pipefail
: "${V5_TRAIN_ENTRYPOINT:?Set reviewed training executable path from Team 5}"
: "${V5_CONFIG_PATH:?Set immutable Team 5 configuration path}"
: "${V5_NPROC_PER_NODE:?Set validated GPUs/node for actual topology}"
HERE="${V5_INFRA_HOME:?Set absolute path to teams/07-infrastructure}"
# Admission only. This does not implement checkpoint signal handling inside Team 5 trainer.
"$HERE/health/launch.sh" --prolog --expected-gpus "$V5_NPROC_PER_NODE" --require-gpu --report-only
HOST0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
: "${V5_RDZV_PORT:=29400}" # Site must guarantee port availability across concurrent jobs.
export V5_HOST0="$HOST0" V5_RDZV_PORT V5_CONFIG_PATH V5_NPROC_PER_NODE V5_TRAIN_ENTRYPOINT
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --export=ALL bash -c '
  exec torchrun --nnodes="${SLURM_NNODES}" \
    --nproc-per-node="${V5_NPROC_PER_NODE}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${V5_HOST0}:${V5_RDZV_PORT}" \
    "${V5_TRAIN_ENTRYPOINT}" --config "${V5_CONFIG_PATH}"
'
