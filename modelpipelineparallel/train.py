"""ERA V5 Session 13: controlled standard-vs-reversible decoder experiments.
Full runs require an accelerator and TinyStories download; CPU smoke uses synthetic data.
"""
import argparse, json, math, os, random, time, hashlib
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    width:int=320
    layers:int=16
    heads:int=8
    vocab:int=256
    context:int=128
    batch:int=4
    tokens:int=50_000_000
    lr:float=3e-4
    seed:int=20260919
    mode:str='baseline'
    data:str='tinystories'
    output:str='runs'
    log_every:int=25
    save_every:int=500
    max_steps:int=0

def seed_everything(s):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

class Attention(nn.Module):
    def __init__(self,d,h):
        super().__init__(); assert d%h==0
        self.d=d;self.h=h;self.qkv=nn.Linear(d,3*d,bias=False);self.out=nn.Linear(d,d,bias=False)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(x).reshape(b,t,3,self.h,d//self.h).permute(2,0,3,1,4).unbind(0)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
        return self.out(a.transpose(1,2).contiguous().reshape(b,t,d))

class MLP(nn.Module):
    def __init__(self,d):
        super().__init__();self.up=nn.Linear(d,4*d,bias=False);self.down=nn.Linear(4*d,d,bias=False)
    def forward(self,x):return self.down(F.gelu(self.up(x)))

class StandardBlock(nn.Module):
    def __init__(self,d,h):
        super().__init__();self.n1=nn.LayerNorm(d);self.a=Attention(d,h);self.n2=nn.LayerNorm(d);self.m=MLP(d)
    def forward(self,x):
        x=x+self.a(self.n1(x));return x+self.m(self.n2(x))

class RevBlock(nn.Module):
    """Additive reversible coupling: y1=x1+F(x2); y2=x2+G(y1)."""
    def __init__(self,d,h):
        super().__init__();self.n1=nn.LayerNorm(d);self.a=Attention(d,h);self.n2=nn.LayerNorm(d);self.m=MLP(d)
    def f(self,x):return self.a(self.n1(x))
    def g(self,x):return self.m(self.n2(x))
    def forward(self,x1,x2):
        y1=x1+self.f(x2);y2=x2+self.g(y1);return y1,y2
    @torch.no_grad()
    def invert(self,y1,y2):
        x2=y2-self.g(y1);x1=y1-self.f(x2);return x1,x2

class ReversibleSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x1,x2,modules):
        with torch.no_grad():
            for b in modules:x1,x2=b(x1,x2)
        ctx.modules=modules;ctx.save_for_backward(x1.detach(),x2.detach())
        return x1,x2
    @staticmethod
    def backward(ctx,gy1,gy2):
        y1,y2=ctx.saved_tensors
        for b in reversed(ctx.modules):
            with torch.no_grad():x1,x2=b.invert(y1,y2)
            with torch.enable_grad():
                xx1=x1.detach().requires_grad_(True);xx2=x2.detach().requires_grad_(True)
                z1,z2=b(xx1,xx2)
                params=tuple(b.parameters())
                grads=torch.autograd.grad((z1,z2),(xx1,xx2)+params,(gy1,gy2),allow_unused=True)
            for p,g in zip(params,grads[2:]):
                if g is not None:
                    if p.grad is None:p.grad=g.detach()
                    else:p.grad.add_(g.detach())
            gy1,gy2=grads[:2];y1,y2=x1.detach(),x2.detach()
        return gy1,gy2,None

class LM(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        self.tok=nn.Embedding(c.vocab,c.width);self.pos=nn.Embedding(c.context,c.width)
        blocks=RevBlock if c.mode=='reversible' else StandardBlock
        self.blocks=nn.ModuleList([blocks(c.width,c.heads) for _ in range(c.layers)])
        self.final=nn.LayerNorm(c.width);self.head=nn.Linear(c.width,c.vocab,bias=False)
        self.head.weight=self.tok.weight
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
    def forward(self,ids):
        b,t=ids.shape;assert t<=self.c.context
        x=self.tok(ids)+self.pos(torch.arange(t,device=ids.device))[None]
        if self.c.mode=='reversible':
            x1,x2=ReversibleSequence.apply(x,x,tuple(self.blocks));x=(x1+x2)*.5
        else:
            for block in self.blocks:x=block(x)
        return self.head(self.final(x))
    def nparams(self):return sum(p.numel() for p in self.parameters())

class TinyStoriesBytes:
    """Real UTF8 byte tokenizer; 1 token = 1 byte. No pretrained tokenizer hidden state."""
    def __init__(self,seed):
        try:from datasets import load_dataset
        except ImportError as exc:raise RuntimeError('pip install datasets') from exc
        self.source='roneneldan/TinyStories';self.it=iter(load_dataset(self.source,split='train',streaming=True).shuffle(seed=seed,buffer_size=1000));self.buf=[]
    def take(self,n):
        while len(self.buf)<n:
            try:row=next(self.it)
            except StopIteration:raise RuntimeError('Dataset exhausted before requested tokens')
            self.buf.extend(row['text'].encode('utf-8')+b'\n')
        out=self.buf[:n];del self.buf[:n]
        return torch.tensor(out,dtype=torch.long)

class SyntheticBytes:
    def __init__(self,seed):self.g=torch.Generator().manual_seed(seed)
    def take(self,n):return torch.randint(0,256,(n,),generator=self.g)

def optimizer(c,m):return torch.optim.AdamW(m.parameters(),lr=c.lr,weight_decay=.01)

def train(c):
    assert c.tokens>0 and c.batch>0 and c.context>=2
    os.makedirs(c.output,exist_ok=True);seed_everything(c.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=LM(c).to(device);opt=optimizer(c,model)
    source=TinyStoriesBytes(c.seed) if c.data=='tinystories' else SyntheticBytes(c.seed)
    target_per_step=c.batch*c.context
    total_steps=math.ceil(c.tokens/target_per_step)
    if c.max_steps:total_steps=min(total_steps,c.max_steps)
    counted=0;first=None;log=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.perf_counter()
    for step in range(1,total_steps+1):
        # Each position predicts one next byte; sequence boundaries belong to the same continuous stream.
        ids=source.take(c.batch*(c.context+1)).reshape(c.batch,c.context+1).to(device)
        opt.zero_grad(set_to_none=True)
        logits=model(ids[:,:-1]);loss=F.cross_entropy(logits.reshape(-1,c.vocab),ids[:,1:].reshape(-1))
        loss.backward();opt.step();counted+=target_per_step
        if step==1:first=float(loss.item())
        if step==1 or step% c.log_every==0 or step==total_steps:
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            rec=dict(step=step,loss=float(loss.item()),tokens=counted,elapsed_s=elapsed,tokens_s=counted/elapsed,peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
            log.append(rec);print(json.dumps(rec),flush=True)
        if step%c.save_every==0 or step==total_steps:
            ckpt={'model':model.state_dict(),'optimizer':opt.state_dict(),'config':asdict(c),'step':step,'counted_tokens':counted}
            torch.save(ckpt,os.path.join(c.output,'latest.pt'))
    summary={'config':asdict(c),'device':str(device),'parameters':model.nparams(),'requested_tokens':c.tokens,'executed_tokens':counted,'first_loss':first,'last_loss':log[-1]['loss'],'throughput_tokens_s':log[-1]['tokens_s'],'peak_cuda_bytes':log[-1]['peak_cuda_bytes'],'full_target_complete':counted>=c.tokens,'records':log}
    with open(os.path.join(c.output,'summary.json'),'w') as f:json.dump(summary,f,indent=2)
    return summary

def find_batch(c,max_batch=256):
    if not torch.cuda.is_available():raise RuntimeError('Maximum GPU batch size requires CUDA. CPU results cannot substitute for VRAM.')
    dev='cuda';seed_everything(c.seed);tested=[];batch=1
    while batch<=max_batch:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            m=LM(c).to(dev);o=optimizer(c,m)
            x=torch.randint(0,c.vocab,(batch,c.context+1),device=dev)
            o.zero_grad(set_to_none=True);logits=m(x[:,:-1]);loss=F.cross_entropy(logits.reshape(-1,c.vocab),x[:,1:].reshape(-1));loss.backward();o.step();torch.cuda.synchronize()
            tested.append({'batch':batch,'ok':True,'peak_bytes':torch.cuda.max_memory_allocated()});batch*=2
        except torch.cuda.OutOfMemoryError:
            tested.append({'batch':batch,'ok':False});break
        finally:
            for name in ('m','o','x','logits','loss'):
                if name in locals():del locals()[name]
            torch.cuda.empty_cache()
    successes=[v['batch'] for v in tested if v['ok']]
    return max(successes),tested

def self_test():
    torch.set_num_threads(1);seed_everything(11)
    c=Config(width=32,layers=2,heads=4,vocab=256,context=12,batch=2)
    a=RevBlock(32,4).double();x=torch.randn(2,12,32,dtype=torch.double)
    y1,y2=a(x,x*0.7);z1,z2=a.invert(y1,y2)
    inverse_error=max((z1-x).abs().max().item(),(z2-x*.7).abs().max().item())
    assert inverse_error<1e-11,inverse_error
    direct=LM(Config(width=32,layers=2,heads=4,context=12,batch=2,mode='reversible')).double()
    ids=torch.randint(0,256,(2,12));loss=direct(ids).pow(2).mean();loss.backward()
    grads1={k:p.grad.clone() for k,p in direct.named_parameters()}
    # Explicit graph reference with identical weights, bypassing custom reversible backward.
    ref=LM(Config(width=32,layers=2,heads=4,context=12,batch=2,mode='reversible')).double();ref.load_state_dict(direct.state_dict())
    x=ref.tok(ids)+ref.pos(torch.arange(12))[None];x1=x2=x
    for block in ref.blocks:x1,x2=block(x1,x2)
    loss2=ref.head(ref.final((x1+x2)*.5)).pow(2).mean();loss2.backward()
    errs={k:(grads1[k]-p.grad).abs().max().item() for k,p in ref.named_parameters()}
    maxerr=max(errs.values());assert maxerr<1e-8,(maxerr,errs)
    return dict(reconstruction_max_abs_error=inverse_error,gradient_max_abs_error=maxerr,parameter_count_20m_baseline=LM(Config()).nparams(),parameter_count_20m_reversible=LM(Config(mode='reversible')).nparams())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--mode',choices=['baseline','reversible'],default='baseline');p.add_argument('--data',choices=['tinystories','synthetic'],default='tinystories');p.add_argument('--batch',type=int,default=4);p.add_argument('--tokens',type=int,default=50_000_000);p.add_argument('--context',type=int,default=128);p.add_argument('--output',default='runs');p.add_argument('--max-steps',type=int,default=0);p.add_argument('--find-batch',action='store_true');p.add_argument('--width',type=int,default=320);p.add_argument('--layers',type=int,default=16);p.add_argument('--heads',type=int,default=8)
    a=p.parse_args();c=Config(mode=a.mode,data=a.data,batch=a.batch,tokens=a.tokens,context=a.context,output=a.output,max_steps=a.max_steps,width=a.width,layers=a.layers,heads=a.heads)
    if a.test:print(json.dumps(self_test(),indent=2))
    elif a.find_batch:print(json.dumps(find_batch(c),indent=2))
    else:train(c)
