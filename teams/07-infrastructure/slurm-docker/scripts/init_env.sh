#!/usr/bin/env bash
# Generate lab-only DB passwords; do not commit .env or upload it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
if [[ -e .env ]]; then echo '.env exists; not overwriting'; exit 0; fi
umask 077
python3 - <<'PY' >.env
import secrets
print('SLURM_DB_ROOT_PASSWORD='+secrets.token_urlsafe(32))
print('SLURM_DB_PASSWORD='+secrets.token_urlsafe(32))
PY
chmod 600 .env
echo 'Generated local .env (gitignored); passwords not printed.'
