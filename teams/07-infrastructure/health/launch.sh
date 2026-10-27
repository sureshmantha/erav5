#!/usr/bin/env bash
# G7 GPU training preflight. Never assumes missing diagnostics mean pass.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode=prolog; report_only=0; expected=0; require_gpu=0
xid_source=journalctl; xid_file=""
usage() { echo 'Usage: launch.sh [--prolog|--full] [--report-only] [--expected-gpus N] [--require-gpu] [--xid-source journalctl|dmesg|file] [--xid-file PATH] [-- command args...]' >&2; }
while (($#)); do
  case "$1" in
    --prolog) mode=prolog; shift;;
    --full) mode=full; shift;;
    --report-only) report_only=1; shift;;
    --expected-gpus) expected="$2"; shift 2;;
    --require-gpu) require_gpu=1; shift;;
    --xid-source) xid_source="$2"; shift 2;;
    --xid-file) xid_file="$2"; shift 2;;
    --) shift; break;;
    *) usage; exit 64;;
  esac
done
cmd=(python3 "$HERE/gpu_health.py" --mode "$mode" --expected-gpus "$expected" --xid-source "$xid_source")
((require_gpu)) && cmd+=(--require-gpu)
[[ -n "$xid_file" ]] && cmd+=(--xid-file "$xid_file")
status=0
"${cmd[@]}" || status=$?
if ((report_only)); then exit "$status"; fi
if ((status != 0)); then echo "Health not PASS (exit $status): training admission blocked" >&2; exit "$status"; fi
if (($# == 0)); then usage; exit 64; fi
exec "$@"
