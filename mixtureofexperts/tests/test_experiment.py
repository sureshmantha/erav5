"""Session 14 contract tests (CPU, no external network)."""
import copy, json, math, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import TinyLM, SparseMoE, convert, dataset, batches, training_loss, setup, count_active


def test_initial_vocabulary_sanity():
    setup(140926)
    d=TinyLM()
    docs,_=dataset(Path(__file__).resolve().parents[1]/'data/era_v5_clean_300.jsonl',48)
    probe=batches(docs['val'],48,8,3,seed=7042)
    ce=sum(float(training_loss(d,x,torch.device('cpu'))[0].detach()) for x in probe)/len(probe)
    assert abs(ce-math.log(257))<0.15,(ce,math.log(257))


def test_causal_mask_prevents_future_leakage():
    setup(13)
    m=TinyLM();m.eval()
    a=torch.randint(0,256,(1,16)); b=a.clone(); b[:,9:]=torch.randint(0,256,(1,7))
    with torch.inference_mode():
        ya=m(a);yb=m(b)
    assert torch.allclose(ya[:,:9,:],yb[:,:9,:],atol=1e-5,rtol=0), (ya[:,:9,:]-yb[:,:9,:]).abs().max()


def test_conversion_preserves_function():
    setup(7)
    d=TinyLM()
    m=convert(d,6,2)
    x=torch.randint(0,256,(2,20))
    with torch.no_grad():err=(d(x)-m(x)).abs().max().item()
    assert err < 3e-6,err
    assert sum(p.numel() for p in m.parameters())>sum(p.numel() for p in d.parameters())
    assert count_active(m)<sum(p.numel() for p in m.parameters())


def test_sparse_dispatch_does_not_evaluate_unselected_experts():
    setup(22)
    f=SparseMoE(12,24,6,2)
    x=torch.randn(1,1,12,requires_grad=True)
    y=f(x).sum()+0.01*f.last_aux
    y.backward()
    counts=f.last_counts
    assert sum(counts)==2 and len([q for q in counts if q])==2
    for i,c in enumerate(counts):
        grads=[v.grad for v in f.experts[i].parameters()]
        assert (all(g is None for g in grads) if c==0 else all(g is not None for g in grads))
    assert torch.isfinite(x.grad).all()


def test_doc_level_split_and_no_boundary_crossing():
    p=Path(__file__).resolve().parents[1]/'data/era_v5_clean_300.jsonl'
    docs,meta=dataset(p,48)
    assert meta['train_docs']==245 and meta['val_docs']==55
    assert all(len(d)>=49 for s in docs.values() for d in s)
    for split in docs:
        for x,y in batches(docs[split],48,4,5,split=='val'):
            assert torch.equal(x[:,1:],y[:,:-1])


def test_previous_experiment_observed_reduction():
    p=Path(__file__).resolve().parents[1]/'artifacts/verified/results.json'
    if not p.exists(): return
    x=json.loads(p.read_text()); h=x['history']
    assert h[0]['val_ce']>x['dense_last_val']
    assert abs(x['dense_last_val']-x['moe_initial_val'])<1e-5
    assert x['moe_final_val']<x['moe_initial_val']
    assert all(min(row)>0 for row in x['routing_on_heldout'])


def test_router_counts_and_auxiliary_loss():
    setup(5)
    m=convert(TinyLM(),6,2)
    ce,aux=training_loss(m,(torch.randint(0,256,(2,12)),torch.randint(0,256,(2,12))),torch.device('cpu'))
    assert torch.isfinite(ce) and torch.isfinite(aux)
    for b in m.blocks:
        assert sum(b.ffn.last_counts)==2*12*2
        assert 0 < b.ffn.last_aux.item() < 6

