#!/usr/bin/env bash
# Team 7: structured health JSON on stdout. Exit 0=pass, 1=warn, 2=fail.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/gpu_health.py" "$@"
