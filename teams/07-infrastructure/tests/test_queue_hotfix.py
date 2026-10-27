"""Configuration regression checks; NOT a substitute for Slurm Docker integration."""
from pathlib import Path
import unittest
R = Path(__file__).resolve().parents[1]
class HotfixStaticTests(unittest.TestCase):
    def test_munge_socket_directory_traversable(self):
        entry = (R / 'slurm-docker/scripts/entrypoint.sh').read_text()
        self.assertIn('chmod 0755 /run/munge',entry)
        self.assertNotIn('chmod 0700 /run/munge',entry)
        self.assertIn('gosu slurm munge -n',entry)
    def test_docker_cpu_topology_is_explicitly_virtual(self):
        conf = (R / 'slurm-docker/conf/slurm.conf').read_text()
        dockerfile = (R / 'slurm-docker/Dockerfile').read_text()
        self.assertIn('SlurmdParameters=config_overrides', conf)
        self.assertIn('JobAcctGatherType=jobacct_gather/none', conf)
        self.assertIn('ThreadsPerCore=1', conf)
        self.assertIn(
            'COPY conf/cgroup.conf /etc/slurm/cgroup.conf', dockerfile
        )
        cgroup = (R / 'slurm-docker/conf/cgroup.conf').read_text()
        self.assertIn('CgroupPlugin=cgroup/v2', cgroup)
        self.assertIn('IgnoreSystemd=yes', cgroup)
    def test_overlimit_requests_explicitly_rejected(self):
        qos = (R / 'slurm-docker/scripts/init_accounts.sh').read_text()
        self.assertIn('Flags=DenyOnLimit', qos)
    def test_collective_does_not_use_standalone_torchrun(self):
        collective = (R / 'tests/test_collective.py').read_text()
        self.assertIn('MASTER_ADDR="127.0.0.1"', collective)
        self.assertNotIn('torchrun','--standalone',collective)
if __name__ == '__main__': unittest.main()
