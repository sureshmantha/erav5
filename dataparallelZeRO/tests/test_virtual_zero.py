"""Run with: python -m unittest discover -s tests -v"""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import virtual_zero as z

class TestZeRO32(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.results = z.validate()

    def test_exactly_32_nonoverlapping_shards(self):
        bounds=z.partitions(z.NUM_PARAMS)
        self.assertEqual(len(bounds),32)
        self.assertEqual(bounds[0][0],0)
        self.assertEqual(bounds[-1][1],z.NUM_PARAMS)
        self.assertTrue(all(bounds[i][1]==bounds[i+1][0] for i in range(31)))

    def test_identical_model_updates_across_stages(self):
        ref=self.results[0]
        for variant in self.results[1:]:
            self.assertLess((ref['final']-variant['final']).abs().max().item(),1e-12)
            self.assertLess(max(abs(a-b) for a,b in zip(ref['losses'],variant['losses'])),1e-12)

    def test_real_forward_and_backward_training(self):
        self.assertLess(self.results[0]['losses'][-1],self.results[0]['losses'][0])
        self.assertEqual(len(self.results[0]['losses']),4)

    def test_resident_optimizer_and_gradient_ownership(self):
        for stage, expected_weights_full, expected_grad_full in [(0,True,True),(1,True,True),(2,True,False),(3,False,False)]:
            v=self.results[stage]
            for rank in v['rank_state_elements']:
                self.assertEqual(rank['weight']==z.NUM_PARAMS,expected_weights_full)
                self.assertEqual(rank['gradient']==z.NUM_PARAMS,expected_grad_full)
                target=2*z.NUM_PARAMS if stage==0 else 2*(z.partitions(z.NUM_PARAMS)[rank['rank']][1]-z.partitions(z.NUM_PARAMS)[rank['rank']][0])
                self.assertEqual(rank['moments'],target)

    def test_ideal_bytes_formulas(self):
        self.assertEqual([z.resident_bytes_per_parameter(x,32) for x in range(4)],
                         [16,4.375,2.4375,.5])
        self.assertEqual(z.stage3_extra_buffer_bytes(),z.LAYER_ELEMENTS[0]*2)

    def test_collective_traffic_assumptions(self):
        t=z.memory_table()
        self.assertEqual(t[0]['modeled_ring_bytes_per_rank'],t[1]['modeled_ring_bytes_per_rank'])
        self.assertEqual(t[1]['modeled_ring_bytes_per_rank'],t[2]['modeled_ring_bytes_per_rank'])
        self.assertEqual(t[3]['modeled_ring_bytes_per_rank']/t[0]['modeled_ring_bytes_per_rank'],1.5)

    def test_native_torch_adam_step(self):
        # Independently validate hand-coded Adam once against actual torch.optim.Adam.
        start=z.initial_weights()
        grad=torch.linspace(-.4,.6,z.NUM_PARAMS,dtype=torch.float64)
        native=torch.nn.Parameter(start.clone())
        optimizer=torch.optim.Adam([native],lr=z.LR,betas=(z.BETA1,z.BETA2),eps=z.EPS,weight_decay=0.)
        native.grad=grad.clone()
        optimizer.step()
        ours=start.clone()
        z.adam_inplace(ours,torch.zeros_like(ours),torch.zeros_like(ours),grad,1)
        self.assertLess((native.detach()-ours).abs().max().item(),2e-13)

if __name__=='__main__': unittest.main()
