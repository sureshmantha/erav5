"""Session 14: verifiable dense-FFN -> sparse MoE upcycling experiment.

PyTorch CPU/CUDA, byte next-token LM. No 32-GPU or 20M-parameter claims.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, os, platform, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VOCAB=257
SEED=140926


def setup(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)


def dataset(path, T, seed=SEED):
    """Split by document id, not tokens; never produce context across a document boundary."""
    docs={'train':[],'val':[]}; ids={'train':[],'val':[]}; sources=collections.Counter()
    raw=Path(path).read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    for line in raw.decode('utf-8').splitlines():
        item=json.loads(line)
        docid=item['id']; sources[item['source']]+=1
        text='Instruction: '+item['instruction']+'\n'
        if item.get('input'): text+='Input: '+item['input']+'\n'
        text+='Response: '+item['output']+'\n'
        bytearray_ids=np.frombuffer(text.encode('utf-8'),dtype=np.uint8).astype(np.int64)
        if len(bytearray_ids)<T+1: continue
        slot='val' if (int.from_bytes(hashlib.sha256(docid.encode('utf-8')).digest()[:8],'big')%5==0) else 'train'
        docs[slot].append(torch.from_numpy(bytearray_ids)); ids[slot].append(docid)
    assert not set(ids['train']) & set(ids['val'])
    assert docs['train'] and docs['val']
    return docs, {'sha256':digest,'train_docs':len(ids['train']),'val_docs':len(ids['val']),
                  'source_rows':dict(sources),'train_bytes':sum(map(len, docs['train'])),
                  'val_bytes':sum(map(len, docs['val'])),'split':'SHA256(id) % 5 == 0 for val'}


def batches(docs, T, B, n, seed):
    r=random.Random(seed)
    out=[]
    for _ in range(n):
        xs=[]; ys=[]
        for _ in range(B):
            d=docs[r.randrange(len(docs))]
            s=r.randrange(len(d)-T)
            xs.append(d[s:s+T]);ys.append(d[s+1:s+T+1])
        out.append((torch.stack(xs),torch.stack(ys)))
    return out

class FFN(nn.Module):
    def __init__(self,d,w):
        super().__init__(); self.up=nn.Linear(d,w); self.down=nn.Linear(w,d)
    def forward(self,x): return self.down(F.gelu(self.up(x)))

class SparseMoE(nn.Module):
    """Top-k token routing with renormalized selected-expert weights; genuine sparse dispatch.

    Every expert is *initially cloned* from the pretrained dense FFN, so weighted sum of
    identical experts preserves function. Clones later diverge under routed gradients.
    No shared expert; no expert capacity truncation; every selected token is processed.
    """
    def __init__(self,d,w,experts=6,top_k=2):
        super().__init__()
        assert 1<=top_k<=experts
        self.router=nn.Linear(d,experts,bias=False)
        self.experts=nn.ModuleList(FFN(d,w) for _ in range(experts))
        self.top_k=top_k; self.n_experts=experts
        self.last_counts=None; self.last_aux=None
        nn.init.normal_(self.router.weight,mean=0,std=0.03)
    @classmethod
    def from_dense(cls, dense, n, k):
        d=dense.up.in_features; w=dense.up.out_features
        obj=cls(d,w,n,k)
        for expert in obj.experts: expert.load_state_dict(dense.state_dict())
        return obj
    def forward(self,x):
        shape=x.shape; flat=x.reshape(-1,shape[-1]); count=flat.shape[0]
        scores=self.router(flat.float())  # router decisions in FP32
        top_logits, top_idx = scores.topk(self.top_k,dim=-1)
        weights=top_logits.softmax(-1).to(dtype=flat.dtype) # sum to one
        probs=scores.softmax(-1)
        shares=torch.bincount(top_idx.reshape(-1),minlength=self.n_experts)
        assignment_frac=shares.float()/(count*self.top_k)
        # Switch-style differentiable load regularizer; N*sum(f_i*p_i) == 1 at uniform balance
        aux=self.n_experts*(assignment_frac.detach()*probs.mean(0)).sum()
        # Actual expert computation only on routed tokens, not all experts and all tokens.
        result=torch.zeros_like(flat)
        for e,expert in enumerate(self.experts):
            rows,slots=(top_idx==e).nonzero(as_tuple=True)
            if rows.numel():
                updates=expert(flat.index_select(0,rows))*weights[rows,slots,None]
                result=result.index_add(0,rows,updates)
        self.last_counts=shares.detach().cpu().tolist()
        self.last_aux=aux
        return result.reshape(shape)

class Block(nn.Module):
    def __init__(self,d,w,heads):
        super().__init__()
        self.n1=nn.LayerNorm(d);self.attn=nn.MultiheadAttention(d,heads,batch_first=True)
        self.n2=nn.LayerNorm(d);self.ffn=FFN(d,w)
    def forward(self,x,mask):
        h=self.n1(x); a,_=self.attn(h,h,h,need_weights=False,attn_mask=mask,is_causal=True)
        x=x+a
        return x+self.ffn(self.n2(x))

class TinyLM(nn.Module):
    def __init__(self,d=64,w=128,layers=2,heads=4,ctx=48):
        super().__init__()
        self.cfg={'d':d,'ffn':w,'layers':layers,'heads':heads,'ctx':ctx,'vocab':VOCAB}
        self.token=nn.Embedding(VOCAB,d); self.pos=nn.Embedding(ctx,d)
        self.blocks=nn.ModuleList([Block(d,w,heads) for _ in range(layers)])
        self.final_norm=nn.LayerNorm(d);self.head=nn.Linear(d,VOCAB,bias=False)
        self.head.weight=self.token.weight
        nn.init.normal_(self.token.weight,mean=0.0,std=0.02)  # S9: near-uniform initial logits
    def forward(self,ids):
        b,t=ids.shape; assert t<=self.cfg['ctx']
        x=self.token(ids)+self.pos(torch.arange(t,device=ids.device))[None,:,:]
        mask=torch.ones((t,t),dtype=torch.bool,device=ids.device).triu(1)
        for block in self.blocks:x=block(x,mask)
        return self.head(self.final_norm(x))

def convert(dense,n=6,k=2):
    new=copy.deepcopy(dense)
    for block in new.blocks: block.ffn=SparseMoE.from_dense(block.ffn,n,k)
    return new

def training_loss(m,batch,device):
    x,y=(z.to(device) for z in batch)
    logits=m(x)
    ce=F.cross_entropy(logits.flatten(0,1),y.reshape(-1))
    aux=[b.ffn.last_aux for b in m.blocks if isinstance(b.ffn,SparseMoE)]
    return ce, (torch.stack(aux).mean() if aux else ce.new_zeros(()))

@torch.inference_mode()
def evaluate(m, eval_batches, device):
    m.eval(); vals=[]
    for batch in eval_batches:
        loss,_=training_loss(m,batch,device); vals.append(float(loss))
    m.train()
    return sum(vals)/len(vals)

def train(m, data, eval_batches, device, steps, batch, T, seed, lr, aux_coeff, phase, eval_every=20):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.999),weight_decay=0.01)
    train_batches=batches(data,T,batch,steps,seed)
    history=[]; t0=time.perf_counter()
    val0=evaluate(m,eval_batches,device)
    history.append({'phase':phase,'step':0,'val_ce':val0,'train_ce':None,'aux':None,'updates':0,'tokens_seen':0,'seconds':0.0})
    print(f'{phase:6s} step=  0 held_out_CE={val0:.6f}',flush=True)
    for step,(x,y) in enumerate(train_batches,1):
        opt.zero_grad(set_to_none=True)
        loss, aux=training_loss(m,(x,y),device)
        objective=loss+(aux_coeff*aux if phase=='moe' else 0)
        objective.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if step==1 or step%eval_every==0 or step==steps:
            val=evaluate(m,eval_batches,device)
            elapsed=time.perf_counter()-t0
            row={'phase':phase,'step':step,'val_ce':val,'train_ce':float(loss.detach()),'aux':float(aux.detach()),
                 'tokens_seen':step*batch*T,'seconds':elapsed,'tokens_per_sec':step*batch*T/elapsed}
            if phase=='moe': row['usage']=[blk.ffn.last_counts for blk in m.blocks]
            history.append(row)
            print(f'{phase:6s} step={step:3d} held_out_CE={val:.6f} train_CE={row["train_ce"]:.6f} throughput={row["tokens_per_sec"]:.0f}tok/s',flush=True)
    return history, opt

def count_active(model, token_per_expert=True):
    shared=sum(p.numel() for n,p in model.named_parameters() if '.experts.' not in n)
    for b in model.blocks:
        if isinstance(b.ffn,SparseMoE):
            each=sum(p.numel() for p in b.ffn.experts[0].parameters())
            shared+=b.ffn.top_k*each
    return shared

def run(args):
    setup(args.seed)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    docs, provenance=dataset(args.data,args.ctx)
    eval_batches=batches(docs['val'],args.ctx,args.batch,args.eval_batches,seed=7042)
    dense=TinyLM(d=args.d, w=args.ffn, layers=args.layers, heads=args.heads, ctx=args.ctx).to(device)
    cfg=dict(d=dense.cfg,device=str(device),steps_dense=args.dense_steps,steps_moe=args.moe_steps,
             batch=args.batch,context=args.ctx,lr=args.lr,aux_coeff=args.aux_coeff,
             experts=args.experts,top_k=args.top_k,seed=args.seed,py_version=platform.python_version(),
             torch_version=torch.__version__,dataset=provenance)
    print('DATA',json.dumps(provenance,indent=2),flush=True)
    print('PARAMS DENSE',sum(p.numel() for p in dense.parameters()),flush=True)
    dh,_=train(dense,docs['train'],eval_batches,device,args.dense_steps,args.batch,args.ctx,
               args.seed+7,args.lr,0,'dense',args.eval_every)
    checkpoint={'model':copy.deepcopy(dense.state_dict()),'cfg':dense.cfg}
    torch.save(checkpoint,root/'dense_last.pt')
    # Keep original frozen for exact comparisons; different optimizer state for a new architecture.
    moe=convert(dense,args.experts,args.top_k).to(device)
    probe_x=eval_batches[0][0][:2].to(device)
    with torch.inference_mode():
        dense.eval();moe.eval(); difference=float((dense(probe_x)-moe(probe_x)).abs().max())
    assert difference < 2e-5, f'Non-preserving conversion: {difference}'
    cfg['conversion_max_logit_abs_error']=difference
    cfg['param_dense']=sum(p.numel() for p in dense.parameters())
    cfg['param_moe_total']=sum(p.numel() for p in moe.parameters())
    cfg['param_moe_active_per_token']=count_active(moe)
    cfg['active_fraction']=cfg['param_moe_active_per_token']/cfg['param_moe_total']
    print('UPCYCLING', {k:cfg[k] for k in ('param_dense','param_moe_total','param_moe_active_per_token','conversion_max_logit_abs_error')},flush=True)
    mh,_=train(moe,docs['train'],eval_batches,device,args.moe_steps,args.batch,args.ctx,
               args.seed+18,args.lr,args.aux_coeff,'moe',args.eval_every)
    torch.save({'model':moe.state_dict(),'cfg':cfg},root/'moe_last.pt')
    usage=[]
    with torch.inference_mode():
        moe.eval()
        for bi,(x,_) in enumerate(eval_batches):
            _=moe(x.to(device))
            counts=[block.ffn.last_counts for block in moe.blocks]
            if not usage:usage=[[0]*args.experts for _ in counts]
            for lay,row in enumerate(counts):
                for e,c in enumerate(row): usage[lay][e]+=c
    final={'config':cfg,'history':dh+mh,'routing_on_heldout':usage,'dense_last_val':dh[-1]['val_ce'],
           'moe_initial_val':mh[0]['val_ce'],'moe_final_val':mh[-1]['val_ce'],
           'validation_improved_after_conversion':bool(mh[-1]['val_ce']<mh[0]['val_ce']),
           'train_ce_improved_in_moe':bool(mh[-1]['train_ce']<mh[1]['train_ce'])}
    (root/'results.json').write_text(json.dumps(final,indent=2,default=str))
    return final

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/era_v5_clean_300.jsonl'); p.add_argument('--out',default='artifacts')
    p.add_argument('--dense-steps',type=int,default=120);p.add_argument('--moe-steps',type=int,default=120)
    p.add_argument('--d',type=int,default=64);p.add_argument('--ffn',type=int,default=128)
    p.add_argument('--layers',type=int,default=2);p.add_argument('--heads',type=int,default=4)
    p.add_argument('--ctx',type=int,default=48);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=0.0007);p.add_argument('--aux-coeff',type=float,default=0.01)
    p.add_argument('--experts',type=int,default=6);p.add_argument('--top-k',type=int,default=2)
    p.add_argument('--eval-batches',type=int,default=6);p.add_argument('--eval-every',type=int,default=20)
    p.add_argument('--seed',type=int,default=SEED);p.add_argument('--cuda',action='store_true')
    a=p.parse_args();run(a)
