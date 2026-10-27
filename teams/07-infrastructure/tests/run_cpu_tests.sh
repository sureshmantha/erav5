#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
PYTHON="${G7_PYTHON:-}"
if [[ -z "$PYTHON" && -x "$HERE/.venv/bin/python" ]]; then PYTHON="$HERE/.venv/bin/python"; fi
if [[ -z "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi
"$PYTHON" - <<'PY'
import sys
try:
    import numpy as np
    import torch
    x=torch.from_numpy(np.array([1.0],dtype=np.float32))
    assert x.numel()==1
except Exception as exc:
    sys.exit(f"ERROR: incompatible Python/PyTorch/NumPy: {exc}\n"
             "Run: bash scripts/setup_local_venv.sh (Python 3.11), then retry.\n"
             "If PyTorch is absent, install it using official platform instructions.")
print(f"Environment ready: Python={sys.version.split()[0]} torch={torch.__version__} numpy={np.__version__}")
PY
"$PYTHON" -m unittest discover -v -s tests -p 'test_*.py'
fixture_report="$(mktemp)"
trap 'rm -f "$fixture_report"' EXIT
set +e
"$PYTHON" health/gpu_health.py --mode prolog --xid-source file --xid-file health/fixtures/xids_sample.log > "$fixture_report"
result=$?
set -e
if [[ "$result" != 2 ]]; then echo "Expected injected-Xid failure exit 2; got $result" >&2; exit 1; fi
"$PYTHON" -m json.tool < "$fixture_report" >/dev/null
echo 'PASS: CPU tests and injected critical-Xid health JSON. NOT a CUDA/NCCL or Slurm result.'
