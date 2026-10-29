#!/usr/bin/env bash
set -euo pipefail
root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$root" ]]; then echo 'ERROR: must run in the git checkout' >&2; exit 2; fi
cd "$root"
if ! command -v gitleaks >/dev/null 2>&1; then
  echo 'ERROR: gitleaks is not installed. Scan NOT run; do not claim 0 findings.' >&2
  exit 3
fi
# Instructor specifically requests this detection mode (check CLI version compatibility).
gitleaks detect --source . --exit-code 1
printf 'PASS: gitleaks detect --source . returned 0\n'
