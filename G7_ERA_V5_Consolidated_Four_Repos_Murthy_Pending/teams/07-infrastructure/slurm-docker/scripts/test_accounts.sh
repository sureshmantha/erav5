#!/usr/bin/env bash
# Executed INSIDE controller; evidence only valid if script returns 0 and all assertions hold.
set -euo pipefail
out="${1:-/tmp/g7-slurm-evidence}"
mkdir -p "$out"
{
 echo '=== Slurm versions ==='; sinfo -V; scontrol show config | grep -E 'AccountingStorageEnforce|RequeueExit|PriorityType|PreemptMode'
 echo '=== Accounts ==='; sacctmgr -n -P show assoc format=account,user,qos
 echo '=== QOS ==='; sacctmgr -n -P show qos format=name,priority,maxtresperjob,maxwall,maxsubmitjobsperuser
 echo '=== Node availability ==='; sinfo -Nel
} >"$out/00-cluster.txt" 2>&1
# Apply su so the operating-system UID matches the Slurm association.
main_id="$(su -s /bin/bash labmain -c 'sbatch --parsable --account=mainrun --qos=g7_main -p cpu -N1 --cpus-per-task=2 --time=00:02:00 --wrap="hostname; sleep 2"')"
team_id="$(su -s /bin/bash labteam -c 'sbatch --parsable --account=teams --qos=g7_team -p cpu -N1 --cpus-per-task=1 --time=00:02:00 --wrap="hostname; sleep 2"')"
[[ "$main_id" =~ ^[0-9]+$ && "$team_id" =~ ^[0-9]+$ ]]
echo "mainrun_job=$main_id teams_job=$team_id" | tee "$out/01-authorized-jobs.txt"
# A team account explicitly cannot request 2 CPUs (QOS MaxTRESPerJob=cpu=1).
set +e
su -s /bin/bash labteam -c 'sbatch --parsable --account=teams --qos=g7_team -p cpu -N1 --cpus-per-task=2 --time=00:02:00 --wrap="true"' >"$out/02-overlimit.stdout" 2>"$out/02-overlimit.stderr"
rc=$?
set -e
if ((rc==0)); then echo 'FAIL: over-limit job was accepted' >&2; exit 1; fi
echo "PASS: 2-CPU request denied by team QoS; sbatch_exit=$rc" >"$out/02-overlimit.assertion"
# An unauthorized OS user cannot submit to mainrun.
set +e
su -s /bin/bash labteam -c 'sbatch --parsable --account=mainrun --qos=g7_main -p cpu -N1 --wrap="true"' >"$out/03-unauthorized.stdout" 2>"$out/03-unauthorized.stderr"
rc=$?
set -e
if ((rc==0)); then echo 'FAIL: unauthorized user was accepted into mainrun' >&2; exit 1; fi
echo "PASS: unauthorized account denied; sbatch_exit=$rc" >"$out/03-unauthorized.assertion"
# Finished jobs should have correct account attribution (sacct might lag slightly).
for i in $(seq 1 30); do
  sacct -n -P -j "$main_id,$team_id" -o JobID,Account,State >"$out/04-accounting.txt" 2>&1
  if grep -q '|mainrun|' "$out/04-accounting.txt" && grep -q '|teams|' "$out/04-accounting.txt"; then break; fi
  sleep 2
done
grep -q '|mainrun|' "$out/04-accounting.txt"
grep -q '|teams|' "$out/04-accounting.txt"
echo 'PASS: two Slurm accounts, policy rejection, unauthorized rejection, accounting attribution'
