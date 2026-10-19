"""Unit tests: unittest discover -v, no GPU or Internet required."""
import math,unittest
import torch
import experiment as e

class TestSession11(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)
    def test_manual_adam_agrees(self):
        r=e.hand_adam()
        self.assertEqual(len(r),5)
        for row in r:
            for key in ('m','v','mhat','vhat','update','weight'):
                self.assertAlmostEqual(row[key+'_hand'],row[key+'_torch'],places=10)
    def test_no_biased_gradient_leakage(self):
        s=e.CausalTokenModel(256)
        x=torch.randint(0,e.V,(2,e.CONTEXT))
        y=s(x)
        x[:,5:]=(x[:,5:]+11)%e.V
        y2=s(x)
        self.assertTrue(torch.allclose(y[:,:5],y2[:,:5],atol=1e-6,rtol=1e-6))
    def test_train_validation_not_identical(self):
        self.assertEqual(len(e.TRAIN),2048)
        self.assertEqual(len(e.VAL),512)
        tr={tuple(x) for x in e.TRAIN.tolist()};val={tuple(x) for x in e.VAL.tolist()}
        self.assertFalse(tr.intersection(val),'train and held-out corpora overlap')
    def test_schedule_boundaries(self):
        p=.003
        self.assertAlmostEqual(e.lr_at(20,p,'wsd'),p)
        self.assertAlmostEqual(e.lr_at(21,p,'wsd'),p)
        self.assertAlmostEqual(e.lr_at(240,p,'wsd'),p)
        self.assertAlmostEqual(e.lr_at(300,p,'wsd'),p*.1)
        self.assertLess(e.lr_at(200,p,'cosine'),p)
        self.assertGreater(e.lr_at(200,p,'cosine'),p*.1)
    def test_reproducible_initialization_and_batches(self):
        e.seed_all(23);a=e.CausalTokenModel(256)
        e.seed_all(23);b=e.CausalTokenModel(256)
        for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p,q))
        self.assertTrue(torch.equal(e.batch_for(9),e.batch_for(9)))
    def test_one_step_update_ratios(self):
        r=e.run_model(256,.001,'wsd',1,23,track=True,eval_every=1)
        self.assertEqual(len(r['layers']),6)
        for layer in r['layers']:
            self.assertTrue(math.isfinite(layer['update_to_weight']))
            self.assertGreater(layer['weight_norm_before'],0.)
            self.assertGreater(layer['update_norm'],0.)

if __name__=='__main__':unittest.main()
