"""Tests adapter shape ONLY, not actual DeepSpeed installation."""
import sys,json,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'integrations'))
from deepspeed_contract import validate,ConfigurationError,save_engine_checkpoint_all_ranks,restore_engine_checkpoint_all_ranks
class T(unittest.TestCase):
 def test_example_config(self):
  c=json.loads((R/'integrations/deepspeed_stage2.example.json').read_text());self.assertEqual(validate(c)['zero_stage'],2)
 def test_batch_mismatch_detected(self):
  c=json.loads((R/'integrations/deepspeed_stage2.example.json').read_text());c['train_batch_size']=9
  with self.assertRaises(ConfigurationError):validate(c)
 def test_fake_engine_contract_only(self):
  class Fake:
   def save_checkpoint(self,*a,**kw):self.kw=kw;return True
   def load_checkpoint(self,*a,**kw):return '/tmp/ckpt',{'config_sha':'abc'}
  f=Fake();self.assertTrue(save_engine_checkpoint_all_ranks(f,'/x','step1',{'run_id':'a','global_step':1,'data_cursor':1,'config_sha':'abc'}))
  self.assertFalse(f.kw['save_latest']);self.assertEqual(restore_engine_checkpoint_all_ranks(f,'/x','step1','abc')['config_sha'],'abc')
  with self.assertRaises(RuntimeError):restore_engine_checkpoint_all_ranks(f,'/x','step1','changed')
if __name__=='__main__':unittest.main()
