#!/usr/bin/env bash
# Optional Slurm requeue integration. Must be executed on a running Docker Slurm lab.
set -euo pipefail
out="${1:-/tmp/g7-requeue-evidence}"; mkdir -p "$out"
job=$(su -s /bin/bash labteam -c 'sbatch --parsable --requeue --account=teams --qos=g7_team -p cpu -N1 -c1 --time=00:03:00 --wrap="if [ \"\${SLURM_RESTART_COUNT:-0}\" -eq 0 ]; then exit 75; fi; exit 0"')
[[ "$job" =~ ^[0-9]+$ ]]
echo "job_id=$job" >"$out/requeue_job.txt"
passed=0
for i in $(seq 1 90); do
  scontrol show job "$job" >"$out/job_last.txt" 2>&1 || true
  sacct -n -P -j "$job" -o JobID,State,ExitCode,Restarts >"$out/sacct_last.txt" 2>&1 || true
  if grep -qE '(COMPLETED|State=COMPLETED)' "$out/sacct_last.txt" "$out/job_last.txt" 2>/dev/null && \
     grep -Eq '(Restarts=1|Restarts=2|\|[12]$)' "$out/job_last.txt" "$out/sacct_last.txt"; then passed=1; break; fi
  sleep 2
done
if ((passed!=1)); then
  echo 'Requeue evidence not established; inspect job_last.txt / sacct_last.txt' >&2
  exit 1
fi
echo 'PASS: exit=75 caused scheduler requeue and subsequent completion'
