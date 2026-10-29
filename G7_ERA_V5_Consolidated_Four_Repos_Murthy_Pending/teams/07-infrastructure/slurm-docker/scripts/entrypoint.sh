#!/usr/bin/env bash
set -euo pipefail
role="${1:-}"
if [[ "$role" == keygen ]]; then
  mkdir -p /mnt/keyshare
  if [[ ! -s /mnt/keyshare/munge.key ]]; then
    umask 077
    openssl rand 1024 > /mnt/keyshare/munge.key
  fi
  exit 0
fi
mkdir -p /run/munge /etc/munge /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /var/log/munge
chmod 0755 /run/munge /var/log/munge  # MUNGE clients (slurm/slurmdbd) must traverse socket directory
chown munge:munge /run/munge /var/log/munge
cp /mnt/keyshare/munge.key /etc/munge/munge.key
chown munge:munge /etc/munge/munge.key
chmod 0400 /etc/munge/munge.key
# Start MUNGE as the munge user and wait for the daemon socket before validating credentials.
gosu munge munged
for i in $(seq 1 30); do
  if [[ -S /run/munge/munge.socket.2 ]]; then
    break
  fi
  sleep 1
done
[[ -S /run/munge/munge.socket.2 ]] || { echo 'munged socket missing' >&2; exit 1; }
# Verify local authenticator, not just daemon pid.
gosu munge munge -n | gosu munge unmunge >/dev/null
# Local MUNGE credentials from a non-root Slurm identity MUST work too.
gosu slurm munge -n | gosu slurm unmunge >/dev/null
case "$role" in
  dbd)
    : "${SLURM_DB_PASSWORD:?Missing SLURM_DB_PASSWORD}"
    umask 077
    cat >/etc/slurm/slurmdbd.conf <<EOF
AuthType=auth/munge
DbdHost=dbd
DbdPort=6819
SlurmUser=slurm
StorageType=accounting_storage/mysql
StorageHost=db
StoragePort=3306
StorageLoc=slurm_acct_db
StorageUser=slurm
StoragePass=${SLURM_DB_PASSWORD}
LogFile=/dev/stdout
DebugLevel=info
EOF
    chmod 600 /etc/slurm/slurmdbd.conf
    chown slurm:slurm /etc/slurm/slurmdbd.conf
    exec slurmdbd -D -vvv
    ;;
  controller)
    chown -R slurm:slurm /var/spool/slurmctld /var/log/slurm
    exec slurmctld -D -vvv
    ;;
  worker)
    mkdir -p /var/spool/slurmd
    exec slurmd -D -vvv
    ;;
  *) echo "Unknown role: $role" >&2; exit 64;;
esac
