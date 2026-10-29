#!/usr/bin/env bash
# Mac-friendly bootstrap for torch 2.0.1 + NumPy 2.x Anaconda environments.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
PYTHON="${G7_BASE_PYTHON:-python3}"
"$PYTHON" - <<'PYVER'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    sys.exit('Use Python 3.10–3.12 for the NumPy 1.x compatibility environment (your Anaconda Python 3.11 is suitable).')
PYVER
"$PYTHON" -m venv --system-site-packages .venv
.venv/bin/python -m pip install --disable-pip-version-check 'numpy>=1.24,<2'
.venv/bin/python - <<'INNERPY'
import numpy as np
try:
    import torch
    y=torch.from_numpy(np.array([1,2],dtype=np.float32))
    print('READY torch=',torch.__version__,'numpy=',np.__version__,'torch bridge=',y.tolist())
except ImportError:
    print('PyTorch missing: install a supported build into .venv.')
    raise SystemExit(2)
INNERPY
echo 'Run: bash tests/run_cpu_tests.sh'
