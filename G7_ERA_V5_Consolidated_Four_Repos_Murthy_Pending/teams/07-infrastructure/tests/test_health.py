from pathlib import Path
import json
import subprocess
import sys
import unittest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/"health"))
from xid_scan import analyze,scan
from gpu_health import inspect

class HealthTests(unittest.TestCase):
    def test_xid_classification(self):
        status,events,counts=analyze('NVRM: Xid (PCI:0000:4b:00): 79, fell off bus\nNVRM: Xid (0000:2a:00): 43, app died\n')
        self.assertEqual(status,'fail')
        self.assertEqual([e['code'] for e in events],[79,43])
        self.assertEqual([e['severity'] for e in events],['fail','warn'])
        self.assertEqual(counts,{'43':1,'79':1})
    def test_unsupported_xid_is_warning(self):
        status,events,_=analyze('NVRM: Xid (0000:2a:00): 999, unknown')
        self.assertEqual(status,'warn')
    def test_fixture_and_missing_logs(self):
        fixture=BASE/'health/fixtures/xids_sample.log'
        self.assertEqual(scan('file',str(fixture))['status'],'fail')
        self.assertEqual(scan('file',str(fixture)+'.missing')['status'],'skipped')
    def test_health_machine_readable_json(self):
        # No fixed expectation about GPU: this test is runnable on CPU and GPU.
        result=inspect(mode='prolog',expected_gpus=999,gpu_required=True,xid_source='file',xid_file=str(BASE/'health/fixtures/xids_sample.log'))
        self.assertEqual(result['schema'],'v5-health/1')
        self.assertEqual(result['verdict'],'fail')
        self.assertFalse(result['admit_training'])
        self.assertTrue(all(c['status'] in ('pass','warn','fail','skipped') for c in result['checks']))
        self.assertIsNone(result.get('secret'))
    def test_health_cli_exit_code_and_json(self):
        cp=subprocess.run([sys.executable,str(BASE/'health/gpu_health.py'),'--mode','prolog',
                           '--require-gpu','--expected-gpus','999','--xid-source','file',
                           '--xid-file',str(BASE/'health/fixtures/xids_sample.log')],
                          capture_output=True,text=True)
        self.assertEqual(cp.returncode,2)
        self.assertEqual(json.loads(cp.stdout)['verdict'],'fail')

if __name__=='__main__':unittest.main()
