#!/usr/bin/env bash
# Run INSIDE controller after services stabilize. Idempotent database bootstrap.
set -euo pipefail
for i in $(seq 1 60); do
  if sacctmgr -n -P show cluster >/dev/null 2>&1; then break; fi
  sleep 2
done
sacctmgr -i add cluster v5lab 2>/dev/null || true
sacctmgr -i add qos g7_main 2>/dev/null || true
sacctmgr -i add qos g7_team 2>/dev/null || true
sacctmgr -i modify qos where name=g7_main set Priority=1000 Preempt=g7_team
sacctmgr -i modify qos where name=g7_team set Priority=10 MaxTRESPerJob=cpu=1 MaxWall=00:05:00 MaxSubmitJobsPerUser=2 Flags=DenyOnLimit
sacctmgr -i add account mainrun Description=MainTraining Organization=ERAV5 2>/dev/null || true
sacctmgr -i add account teams Description=TeamExperiments Organization=ERAV5 2>/dev/null || true
sacctmgr -i modify account where name=mainrun set QOS=g7_main DefaultQOS=g7_main
sacctmgr -i modify account where name=teams set QOS=g7_team DefaultQOS=g7_team
sacctmgr -i add user labmain Account=mainrun DefaultAccount=mainrun 2>/dev/null || true
sacctmgr -i add user labteam Account=teams DefaultAccount=teams 2>/dev/null || true
# Ensure the expected associations now exist; never report success on empty query.
sacctmgr -n -P show associations format=account,user,qos | grep -q '^mainrun|labmain|'
sacctmgr -n -P show associations format=account,user,qos | grep -q '^teams|labteam|'
scontrol reconfigure
echo 'Created and verified two Slurm accounting associations and QoS definitions.'
sacctmgr -P show assoc format=account,user,qos
sacctmgr -P show qos format=name,priority,maxtresperjob,maxwall,maxsubmitjobsperuser
