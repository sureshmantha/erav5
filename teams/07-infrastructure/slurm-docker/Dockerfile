# CPU-only test lab: this is NOT the GPU production node image.
FROM debian:bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    slurm-wlm slurmdbd munge libmunge2 mariadb-client openssl \
    python3 procps iproute2 curl ca-certificates tini gosu \
    && rm -rf /var/lib/apt/lists/*
RUN id slurm >/dev/null 2>&1 || (groupadd -r slurm && useradd -r -g slurm slurm)
RUN useradd -m -u 15001 -s /bin/bash labmain \
 && useradd -m -u 15002 -s /bin/bash labteam \
 && mkdir -p /var/spool/slurmd /var/spool/slurmctld /run/munge /var/log/slurm /etc/slurm \
 && chown -R slurm:slurm /var/spool/slurmctld /var/log/slurm \
 && chmod 1777 /var/spool/slurmd
COPY conf/slurm.conf /etc/slurm/slurm.conf
# Explicitly disable cgroups in the CPU-only Docker lab; Docker Desktop does not
# provide the systemd/cgroup environment needed for Slurm cgroup enforcement.
COPY conf/cgroup.conf /etc/slurm/cgroup.conf
COPY scripts/ /lab/scripts/
RUN chmod +x /lab/scripts/*.sh
ENTRYPOINT ["/usr/bin/tini", "--", "/lab/scripts/entrypoint.sh"]
