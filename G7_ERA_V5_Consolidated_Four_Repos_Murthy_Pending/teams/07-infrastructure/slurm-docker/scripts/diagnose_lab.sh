#!/usr/bin/env bash
# Read-only diagnostic helper. Does not dump full Compose config / generated passwords.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
echo '=== Container states ==='
docker compose ps || true
echo '=== Slurm/MUNGE/DB status ==='
docker compose logs --tail=100 --no-color keygen db dbd controller compute1 compute2 || true
echo '=== Expected Slurm nodes ==='
docker compose exec -T controller sinfo -N -o '%N %T %E' || true
echo '=== MUNGE socket perms (controller) ==='
docker compose exec -T controller ls -ld /run/munge /run/munge/munge.socket.2 || true
echo '=== Distro Slurm + cgroup mode ==='
docker compose exec -T compute1 sh -c 'slurmd -V; cat /etc/slurm/cgroup.conf' || true
