"""Static checks only; NOT proof that Slurm enforces policies at runtime."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
class QueueStatic(unittest.TestCase):
    def test_scheduler_policy_directives_present(self):
        conf=(ROOT/'slurm-docker/conf/slurm.conf').read_text()
        for directive in ('AccountingStorageEnforce=associations,limits,qos',
                          'AccountingStorageType=accounting_storage/slurmdbd',
                          'PreemptType=preempt/qos','RequeueExit=75',
                          'NodeName=compute[1-2]'):
            self.assertIn(directive,conf)
    def test_two_distinct_accounts_with_real_limits(self):
        script=(ROOT/'slurm-docker/scripts/init_accounts.sh').read_text()
        for term in ('add account mainrun','add account teams',
                     'MaxTRESPerJob=cpu=1','MaxWall=00:05:00',
                     'labmain Account=mainrun','labteam Account=teams'):
            self.assertIn(term,script)
    def test_denial_assertions_present(self):
        script=(ROOT/'slurm-docker/scripts/test_accounts.sh').read_text()
        self.assertIn('FAIL: over-limit job was accepted',script)
        self.assertIn('FAIL: unauthorized user was accepted',script)
        self.assertIn('sacct -n -P',script)
if __name__=='__main__': unittest.main()
