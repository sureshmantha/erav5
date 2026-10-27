#!/usr/bin/env bash
# A zero exit status is evidence of actual Slurm-account enforcement.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
"$HERE/scripts/init_env.sh"
if ! docker compose version >/dev/null 2>&1; then
  echo 'Docker Compose V2 is required and must be running.' >&2; exit 3
fi
# Do NOT print docker compose config: it interpolates secrets from .env.
docker compose config --quiet
docker compose up --build --detach
ready=0
for i in $(seq 1 60); do
  nodes="$(docker compose exec -T controller sinfo -h -N -o '%N:%T' 2>/dev/null || true)"
  if printf '%s\n' "$nodes" | grep -Eq '^compute1:(idle|mixed|allocated)$' &&
     printf '%s\n' "$nodes" | grep -Eq '^compute2:(idle|mixed|allocated)$'; then
    ready=1
    break
  fi
  sleep 3
done
if ((ready!=1)); then
  echo 'ERROR: Both Slurm compute nodes did not become schedulable.' >&2
  "$HERE/scripts/diagnose_lab.sh" >&2 || true
  exit 1
fi
echo 'Controller and both simulated CPU nodes are available.'
docker compose exec -T controller /lab/scripts/init_accounts.sh
docker compose exec -T controller /lab/scripts/test_accounts.sh /tmp/g7-slurm-evidence
mkdir -p evidence-from-docker
docker compose cp controller:/tmp/g7-slurm-evidence/. ./evidence-from-docker/
echo 'PASS: real Slurm lab account policies. Artifacts: slurm-docker/evidence-from-docker (gitignored).'
echo 'Review/redact exported output before adding curated evidence to a public PR.'
