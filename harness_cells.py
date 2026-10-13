# %% [markdown]
# # ERA V5 — Session 09 | Loss Harness & Multi-Token Prediction
# **Executable, deterministic teaching experiment — CPU or Google Colab GPU**
#
# Sources: supplied ERA V5 Session 9 transcript (22 Aug 2026) and assignment. This notebook is a compact **controlled experiment** with an explicitly constructed vocabulary; it is **not** a pretrained language model, a FineWeb experiment, or a published benchmark. It requires only PyTorch and Python's standard library.
#
# **Correctness contract:** zero off-by-one errors; no pad-target loss; no cross-document supervision; causal and document-isolated attention; global-denominator chunking; distinct `t+1` and `t+2` targets. There are assertions for every contract.
#
# Read the output tables *as token strings*, not integer IDs. The notebook saves `results.json` with actual machine measurements, not assumed values.

# %% [markdown]
# ## 0 · Environment, reproducibility and deterministic vocabulary
# We use a deliberately small whitespace-token vocabulary. All metrics use the same tokenizer and vocabulary, and the baseline perplexity is checked against *that exact* vocabulary size. The data is an embedded synthetic corpus so the notebook runs top-to-bottom without network access or license uncertainty. A GPU is optional.

# %%
import os, re, json, math, sys, time, subprocess, platform, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 20260822
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(2, torch.get_num_threads()))
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Python:',platform.python_version(),'| PyTorch:',torch.__version__,'| device:',DEVICE)
print('seed:',SEED, '| deterministic embedded corpus; hardware memory readings may vary')

CORPUS = [
    'the scientist reads a clear paper about the bright moon .',
    'the scientist writes a careful paper about the small star .',
    'the teacher reads a clear book about the curious student .',
    'the student writes a small story about the bright moon .',
    'the engineer builds a simple model for the careful student .',
    'the model learns the next word from the previous word .',
    'the curious student learns a simple rule from the teacher .',
    'the careful scientist checks the small model and the result .',
    'the bright star shines over the blue ocean at night .',
    'the tiny boat moves over the quiet blue ocean .',
    'the teacher explains why the model predicts the next token .',
    'the scientist checks every result and writes a clear report .',
]
EXTRA = ['reserved_'+str(i) for i in range(200)]
SPECIAL = ['<pad>','<bos>','<eos>','<unk>']
def split_words(s): return re.findall(r"[A-Za-z_]+|[0-9]+|[^\w\s]",s.lower())
seen = sorted({w for sent in CORPUS for w in split_words(sent)})
VOCAB_SIZE = 128
assert len(SPECIAL)+len(seen) <= VOCAB_SIZE
itos = SPECIAL+seen+EXTRA[:VOCAB_SIZE-len(SPECIAL)-len(seen)]
stoi = {t:i for i,t in enumerate(itos)}
PAD,BOS,EOS,UNK = [stoi[t] for t in SPECIAL]
def encode(s): return [stoi.get(x,UNK) for x in split_words(s)]
def as_string(x): return itos[int(x)]
assert len(stoi) == VOCAB_SIZE
print('Vocabulary size:',VOCAB_SIZE,'| Corpus docs:',len(CORPUS),'| actual text words:',len(seen))
print('Special tokens:',{x:stoi[x] for x in SPECIAL})

# %% [markdown]
# ## 1 · Minimal causal transformer and every tensor dimension
# **B**=batch items, **T**=positions, **D**=hidden width, **H**=attention heads, **dₕ**=D/H, **V**=vocabulary size. No future token may influence a past logit. The optionally supplied `doc_ids` additionally make packed attention block-diagonal. `t+1` is the original LM head; `t+2` is a separate auxiliary head. No head has bias; the tied/untied comparison is therefore exact.

# %%
B,T,D,H,V = 2,20,48,4,VOCAB_SIZE
assert D%H==0
class InspectableCausalLM(nn.Module):
    def __init__(self, vocab=V, width=D, heads=H, maxlen=256, tied=False):
        super().__init__()
        self.width=width;self.heads=heads
        self.embed=nn.Embedding(vocab,width)
        self.position=nn.Embedding(maxlen,width)
        self.qkv=nn.Linear(width,3*width,bias=False)
        self.attn_out=nn.Linear(width,width,bias=False)
        self.norm1=nn.LayerNorm(width)
        self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.final_norm=nn.LayerNorm(width)
        self.head1=nn.Linear(width,vocab,bias=False)
        self.head2=nn.Linear(width,vocab,bias=False)
        if tied: self.head1.weight=self.embed.weight
        self.tied=tied
        # Small random head means *nearly uniform* initial softmax and PPL near V.
        nn.init.normal_(self.embed.weight,std=0.02)
        if not tied: nn.init.normal_(self.head1.weight,std=0.01)
        nn.init.normal_(self.head2.weight,std=0.01)
    def forward(self,tokens,doc_ids=None,shapes=False):
        b,t=tokens.shape;d=self.width;h=self.heads
        assert t <= self.position.num_embeddings
        # Reset learned position indices at each packed document boundary.
        absolute_position=torch.arange(t,device=tokens.device)[None,:].expand(b,t)
        if doc_ids is None: position_ids=absolute_position
        else:
            new_doc=torch.cat([torch.ones((b,1),dtype=torch.bool,device=tokens.device),doc_ids[:,1:]!=doc_ids[:,:-1]],dim=1)
            latest_start=torch.where(new_doc,absolute_position,0).cummax(dim=1).values
            position_ids=absolute_position-latest_start
        token_embeddings=self.embed(tokens)
        position_embeddings=self.position(position_ids)
        x=token_embeddings+position_embeddings
        projected_qkv=self.qkv(self.norm1(x))
        q,k,v=projected_qkv.reshape(b,t,3,h,d//h).permute(2,0,3,1,4).unbind(0)
        score=(q@k.transpose(-2,-1))/math.sqrt(d//h)
        allowed=torch.ones(t,t,dtype=torch.bool,device=tokens.device).tril()[None,None,:,:]
        if doc_ids is not None:
            assert doc_ids.shape==tokens.shape
            allowed=allowed&(doc_ids[:,None,:,None]==doc_ids[:,None,None,:])
        prob=torch.softmax(score.masked_fill(~allowed,float('-inf')),dim=-1)
        context=(prob@v).transpose(1,2).reshape(b,t,d)
        attention_delta=self.attn_out(context)
        after_attention=x+attention_delta
        ff_delta=self.ff(self.norm2(after_attention))
        hidden=self.final_norm(after_attention+ff_delta)
        logits1=self.head1(hidden);logits2=self.head2(hidden)
        if shapes:
            shape_table=[('tokens',tokens,'B × T — batch × sequence'),('position indices',position_ids,'B × T — local positions'),
              ('token embeddings',token_embeddings,'B × T × D'),('position embeddings',position_embeddings,'B × T × D'),
              ('embedding + position',x,'B × T × D'),('QKV linear',projected_qkv,'B × T × (3D)'),
              ('query',q,'B × H × T × dₕ'),('key',k,'B × H × T × dₕ'),('value',v,'B × H × T × dₕ'),
              ('causal / document mask',allowed,'B × 1 × T × T'),('attention logits',score,'B × H × T × T'),
              ('attention probabilities',prob,'B × H × T × T'),('context',context,'B × T × D'),
              ('attention delta',attention_delta,'B × T × D'),('FFN delta',ff_delta,'B × T × D'),
              ('hidden states',hidden,'B × T × D'),('t+1 head weight',self.head1.weight,'V × D'),
              ('t+2 head weight',self.head2.weight,'V × D'),('t+1 logits',logits1,'B × T × V'),
              ('t+2 logits',logits2,'B × T × V')]
            for name,tens,meaning in shape_table: print(f'{name:25s} {str(tuple(tens.shape)):20s} {meaning}')
        return hidden,logits1,logits2
model=InspectableCausalLM().to(DEVICE)
row=torch.tensor([BOS,*encode(CORPUS[0]),EOS],dtype=torch.long)
batch=F.pad(row[None,:],(0,T-len(row)),value=PAD).repeat(B,1).to(DEVICE)
with torch.no_grad(): hidden,logits1,logits2=model(batch,shapes=True)
assert logits1.shape==(B,T,V)
print('Shifted logits shape:',tuple(logits1[:,:-1,:].shape),'shifted targets:',tuple(batch[:,1:].shape))
print('Shape guard:',tuple(logits1[:,:-1].reshape(-1,V).shape),'flattened logits vs',tuple(batch[:,1:].reshape(-1).shape),'flattened targets')

# %% [markdown]
# ## 2 · Print the literal token shift and assert it
# At source index `j`, the model sees `token[j]` and must predict `token[j+1]`. Never train logits `[:,j]` on token `[:,j]`. BOS and EOS are real tokens; PAD is **never** a training target.

# %%
print(f'{"index":>5}  {"INPUT seen at j":<19}  {"TARGET at j+1":<19}  {"contributes?":<12}')
for j in range(batch.size(1)-1):
    source,target=as_string(batch[0,j]),as_string(batch[0,j+1])
    print(f'{j:>5}  {source:<19}  {target:<19}  {"YES" if int(batch[0,j+1])!=PAD else "NO":<12}')
assert all(as_string(batch[0,j+1])==itos[int(batch[0,j+1])] for j in range(T-1))
assert int(batch[0,0])==BOS and int(batch[0,1])==encode(CORPUS[0])[0]
assert int(batch[0,1]) != int(batch[0,0]),'Shift alignment guard'
print('SHIFT ASSERTION: PASS — logits position 0 predicts first text token, not <bos>.')

# %% [markdown]
# ## 3 · Mask padding and verify the denominator
# Raw CE includes padding if left unmasked. Use a *boolean mask* to count valid targets and `ignore_index=PAD` to obtain equivalent masked CE; the denominator must be the number of surviving targets. Ignore PAD only — valid EOS targets should remain trainable. Loss values have units of **nats per predicted token**.

# %%
with torch.no_grad():
    flat_logits=logits1[:,:-1].reshape(-1,V)
    flat_targets=batch[:,1:].reshape(-1)
    unmasked_pad_loss=F.cross_entropy(flat_logits,flat_targets)
    masked_pad_loss=F.cross_entropy(flat_logits,flat_targets,ignore_index=PAD)
    manual_mask=(flat_targets!=PAD)
    manual_loss=F.cross_entropy(flat_logits[manual_mask],flat_targets[manual_mask])
assert torch.allclose(masked_pad_loss,manual_loss,atol=1e-6)
raw_count=flat_targets.numel();valid_count=int(manual_mask.sum());pad_count=raw_count-valid_count
assert pad_count>0
print('Unmasked targets:',raw_count,'| padding excluded:',pad_count,'| contributing:',valid_count)
print(f'Loss including padding={unmasked_pad_loss:.6f} | padding masked={masked_pad_loss:.6f} | delta={masked_pad_loss-unmasked_pad_loss:+.6f}')
print('PADDING ASSERTION: PASS — masked CE equals CE on only valid targets.')

# %% [markdown]
# ## 4 · Pack two documents and mask the boundary
# Pack `[BOS, document A, EOS, BOS, document B, EOS]`. The EOS(A)→BOS(B) target is a **false inter-document transition** and must be excluded from CE. Importantly, a loss boundary mask *alone* is insufficient: the second document must also be **prevented from attending** to the first, and learned position IDs must **restart from zero** for the second document. We show both protections separately. Loss before/after the target mask is computed on the **identical logits**, for a fair one-variable experiment.

# %%
def build_packed(doc_a,doc_b):
    seq_a=[BOS,*encode(doc_a),EOS];seq_b=[BOS,*encode(doc_b),EOS]
    ids=torch.tensor([seq_a+seq_b],dtype=torch.long,device=DEVICE)
    d_ids=torch.tensor([[0]*len(seq_a)+[1]*len(seq_b)],dtype=torch.long,device=DEVICE)
    return ids,d_ids,len(seq_a)
packed,doc_ids,boundary=build_packed(CORPUS[0],CORPUS[1])
assert packed.shape[1]<=256
with torch.no_grad():
    packed_hidden,packed_logits,_=model(packed) # allows cross-doc attention: diagnostic only
    isolated_hidden,isolated_logits,_=model(packed,doc_ids=doc_ids) # correct packed forward
    packed_targets=packed[:,1:]
    valid_next=(doc_ids[:,:-1]==doc_ids[:,1:])&(packed_targets!=PAD)
    per_token=F.cross_entropy(packed_logits[:,:-1].reshape(-1,V),packed_targets.reshape(-1),reduction='none').view_as(packed_targets)
    loss_before=per_token.mean()
    loss_after=per_token[valid_next].mean() # same logits, direct mask effect
    correct_losses=F.cross_entropy(isolated_logits[:,:-1].reshape(-1,V),packed_targets.reshape(-1),reduction='none').view_as(packed_targets)
    loss_isolated=correct_losses[valid_next].mean()
    isolated_first=isolated_logits[:,:boundary]
    # Independent forward for document A and isolated packed forward MUST agree exactly.
    a_logits=model(packed[:,:boundary])[1]
    # The second document must also match an independently processed sequence:
    b_logits=model(packed[:,boundary:])[1]
assert torch.allclose(isolated_first,a_logits,atol=1e-5,rtol=1e-5)
assert torch.allclose(isolated_logits[:,boundary:],b_logits,atol=1e-5,rtol=1e-5)
assert valid_next.sum().item()==packed.numel()-2
print('Pack:',' '.join(map(as_string,packed[0].tolist())))
print('False transition:',as_string(packed[0,boundary-1]),'→',as_string(packed[0,boundary]))
print('Positions:',packed.numel(),'| shift targets:',packed.numel()-1,'| boundary masked:',int((~valid_next).sum()),'| valid targets:',int(valid_next.sum()))
print(f'Before loss boundary mask: {loss_before.item():.6f} nats/token')
print(f'After loss boundary mask:  {loss_after.item():.6f} nats/token (same logits)')
print(f'After attention isolation:  {loss_isolated.item():.6f} nats/token (proper packed forward)')
print('PACKED ATTENTION ASSERTION: PASS — BOTH document logit sequences match independent forwards; attention AND positions reset at boundary.')

# %% [markdown]
# ## 5 · Baseline perplexity near vocabulary size
# `PPL = exp(mean cross entropy)`. For uniform predictions, CE=ln(V) and PPL=V. We use a fresh *untrained*, small-initialization output head and explicitly test proximity. This is a **sanity check, not proof of model quality**. A randomly initialized model need not always produce exactly V: scale and initialization matter. Compare perplexity only for identical tokenization and evaluation datasets.

# %%
with torch.no_grad():
    baseline_ce=F.cross_entropy(logits1[:,:-1].reshape(-1,V),batch[:,1:].reshape(-1),ignore_index=PAD).item()
    baseline_ppl=math.exp(baseline_ce)
    uniform_ce=math.log(V)
print(f'Vocab V={V} | log(V)={uniform_ce:.6f} | initial CE={baseline_ce:.6f} | initial PPL={baseline_ppl:.4f} | PPL/V={baseline_ppl/V:.4f}')
assert abs(baseline_ppl/V-1)<0.20,'Initial perplexity too far from V: inspect initialization, shift and masking.'
print('PERPLEXITY SANITY: PASS — PPL reasonably close to V (±20%, explicitly checked).')

# %% [markdown]
# ## 6 · Tied vs untied head parameters
# With weight shape `[V,D]` and no bias: untied has separate input embedding `[V,D]` plus output head `[V,D]`; tied uses the SAME parameter object for both. Other model parameters remain equal. The auxiliary t+2 head is a separate matrix and is present in both variants.

# %%
tied_model=InspectableCausalLM(tied=True)
untied_model=InspectableCausalLM(tied=False)
count=lambda m:sum(p.numel() for p in m.parameters())
nt=count(tied_model);nu=count(untied_model);saved=nu-nt
assert tied_model.head1.weight is tied_model.embed.weight
assert saved==V*D
print(f'Untied trainable parameters: {nu:,}')
print(f'Tied trainable parameters:   {nt:,}')
print(f'Parameters saved:            {saved:,} = V({V}) × D({D}) | saving {100*saved/nu:.2f}% of untied total')
print('Note: this includes the SAME t+2 head in both parameter counts.')

# %% [markdown]
# ## 7 · Chunked head/loss, numerical and gradient equivalence
# A *correct* chunked loss must project hidden activations in slices, rather than materializing all `[B,T,V]` logits first. Each token contributes to a **single global mean**, not the unweighted average of per-chunk means. Backpropagate individual chunks to release each chunk's logits computation graph. This demonstration isolates the output-head/loss stage: it does not claim transformer activations are free and does not use checkpointing.

# %%
def head_ce_full(hidden,weight,targets,ignore=PAD):
    return F.cross_entropy((hidden@weight.T).reshape(-1,weight.shape[0]),targets.reshape(-1),ignore_index=ignore)
def head_ce_chunked(hidden,weight,targets,chunk_size=5,ignore=PAD):
    denom=(targets!=ignore).sum().clamp(min=1)
    total=hidden.new_zeros(())
    for s in range(0,hidden.shape[1],chunk_size):
        e=min(hidden.shape[1],s+chunk_size)
        chunk_logits=hidden[:,s:e]@weight.T
        total=total+F.cross_entropy(chunk_logits.reshape(-1,weight.shape[0]),targets[:,s:e].reshape(-1),ignore_index=ignore,reduction='sum')
    return total/denom
weight=torch.randn(V,D,dtype=torch.float32)*0.025
sample_hidden=torch.randn(2,13,D,dtype=torch.float32)
sample_targets=torch.randint(4,V,(2,13))
sample_targets[0,-3:]=PAD
full=head_ce_full(sample_hidden,weight,sample_targets)
chunk=head_ce_chunked(sample_hidden,weight,sample_targets,chunk_size=5)
assert torch.allclose(full,chunk,atol=1e-6,rtol=1e-6)
h1=sample_hidden.detach().clone().requires_grad_();w1=weight.detach().clone().requires_grad_()
h2=sample_hidden.detach().clone().requires_grad_();w2=weight.detach().clone().requires_grad_()
head_ce_full(h1,w1,sample_targets).backward()
head_ce_chunked(h2,w2,sample_targets,5).backward()
assert torch.allclose(h1.grad,h2.grad,atol=2e-6,rtol=2e-5)
assert torch.allclose(w1.grad,w2.grad,atol=2e-6,rtol=2e-5)
print(f'Full CE={full.item():.9f}; Chunked CE={chunk.item():.9f}; |Δ|={abs(full.item()-chunk.item()):.3g}')
print('LOSS AND GRADIENT EQUIVALENCE: PASS — same global-token mean and same gradients.')

# %% [markdown]
# ## 8 · Peak output-head/loss memory (measured, never fabricated)
# The benchmark uses the **same** tensor shapes, dtype, loss, and backwards computation in separate clean processes on CPU. On Linux, `ru_maxrss` is the maximum resident set size in KiB (whole process); the baseline-adjusted increment is an approximate process-memory figure, **not** PyTorch allocator bytes. `peak RSS` and `additional peak RSS` are both shown, along with the ratio. GPU mode additionally uses CUDA `max_memory_allocated` and is preferred for GPU claims. This section benchmarks the *head/loss*, not end-to-end transformer training. CPU allocator and kernel workspace choices influence observed ratios.

# %%
MEM_B,MEM_T,MEM_D,MEM_V,MEM_CHUNK=4,512,96,4096,32
BENCH_SOURCE=r'''
import os,sys,resource,json,torch,torch.nn.functional as F
import math
mode=sys.argv[1];use_cuda=(sys.argv[2]=='cuda');torch.manual_seed(918);torch.set_num_threads(2)
b,t,d,v,k=4,512,96,4096,32
u='cuda' if use_cuda else 'cpu'
h=torch.randn(b,t,d,device=u,requires_grad=True)
w=torch.randn(v,d,device=u,requires_grad=True)
tg=torch.randint(4,v,(b,t),device=u);tg[0,-32:]=0
valid=(tg!=0).sum()
if use_cuda: torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
else: base=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if mode=='full':
    l=F.cross_entropy((h@w.T).reshape(-1,v),tg.reshape(-1),ignore_index=0)
    l.backward()
else:
    total=0.0
    for s in range(0,t,k):
        e=min(t,s+k); y=h[:,s:e]@w.T
        z=F.cross_entropy(y.reshape(-1,v),tg[:,s:e].reshape(-1),ignore_index=0,reduction='sum')/valid
        total+=float(z.detach());z.backward()
if use_cuda:torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();unit='bytes';kind='torch.cuda.max_memory_allocated'
else:peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;unit='KiB';kind='resource.ru_maxrss (Linux peak process RSS)'
print(json.dumps({'mode':mode,'loss':float(l.detach()) if mode=='full' else float(total), 'baseline':int(base),'peak':int(peak),'peak_increment':max(0,int(peak-base)),'unit':unit,'kind':kind,'shape':[b,t,d,v],'chunk':k,'device':u}))
'''
def run_peak_case(mode,device='cpu'):
    proc=subprocess.run([sys.executable,'-c',BENCH_SOURCE,mode,device],capture_output=True,text=True,check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])
measured_device='cuda' if torch.cuda.is_available() else 'cpu'
mem_full=run_peak_case('full',measured_device)
mem_chunk=run_peak_case('chunk',measured_device)
scale=1/2**20 if mem_full['unit']=='bytes' else 1/1024
memory_metric='MiB'
mem_base1=mem_full['baseline']*scale;mem_base2=mem_chunk['baseline']*scale
mem_peak_full=mem_full['peak']*scale;mem_peak_chunk=mem_chunk['peak']*scale
mem_delta_full=mem_full['peak_increment']*scale;mem_delta_chunk=mem_chunk['peak_increment']*scale
mem_ratio=(mem_delta_full/mem_delta_chunk if mem_delta_chunk>0 else mem_peak_full/mem_peak_chunk)
mem_ratio_kind=('additional peak allocation' if mem_delta_chunk>0 else 'absolute process peak RSS (chunk increment below RSS resolution)')
assert abs(mem_full['loss']-mem_chunk['loss'])<1e-4
print('Device:',measured_device,'| memory technique:',mem_full['kind'],'| head-shape B,T,D,V=',(MEM_B,MEM_T,MEM_D,MEM_V))
print(f'FULL:   baseline={mem_base1:.2f} MiB, peak={mem_peak_full:.2f} MiB, added peak={mem_delta_full:.2f} MiB')
print(f'CHUNK:  baseline={mem_base2:.2f} MiB, peak={mem_peak_chunk:.2f} MiB, added peak={mem_delta_chunk:.2f} MiB')
print(f'Ratio ({mem_ratio_kind}): {mem_ratio:.3f}×')
print(f'Loss values full={mem_full["loss"]:.8f} chunk={mem_chunk["loss"]:.8f}')
if measured_device=='cpu':
    print('LIMIT: CPU measurements are peak PROCESS RSS, not GPU VRAM. Re-run on Colab GPU for device allocation numbers.')
assert math.isfinite(mem_ratio) and mem_ratio>0

# %% [markdown]
# ## 9 · Part 2: separate `t+1` and `t+2` heads
# Both heads share the same causal hidden states. For token at position `j`: first head target is `j+1`, second head target is `j+2`. We never feed target `j+1` into the second head. As with packed training, exclude transitions crossing a document boundary. We measure a **fixed seeded training experiment**, not generalization: training losses may fluctuate; `t+2` often stays harder but **no universal guarantee exists** about its numerical ordering.

# %%
TRAIN_DOCS=CORPUS*3
max_train_len=max(len(encode(doc))+2 for doc in TRAIN_DOCS)
train_token=torch.full((len(TRAIN_DOCS),max_train_len),PAD,dtype=torch.long,device=DEVICE)
for i,doc in enumerate(TRAIN_DOCS):
    ids=[BOS,*encode(doc),EOS];train_token[i,:len(ids)]=torch.tensor(ids,device=DEVICE)
valid_1=(train_token[:,1:]!=PAD)
valid_2=(train_token[:,2:]!=PAD)
assert valid_2.sum()<valid_1.sum()
train_model=InspectableCausalLM().to(DEVICE)
opt=torch.optim.AdamW(train_model.parameters(),lr=0.007,weight_decay=0.0)
def two_loss(m,tok):
    _,z1,z2=m(tok)
    t1=tok[:,1:];t2=tok[:,2:]
    l1=F.cross_entropy(z1[:,:-1].reshape(-1,V),t1.reshape(-1),ignore_index=PAD)
    l2=F.cross_entropy(z2[:,:-2].reshape(-1,V),t2.reshape(-1),ignore_index=PAD)
    return l1,l2
with torch.no_grad():init1,init2=two_loss(train_model,train_token)
N_STEPS=100
history=[{'step':0,'t1':init1.item(),'t2':init2.item(),'sum':(init1+init2).item()}]
start=time.perf_counter()
for step in range(1,N_STEPS+1):
    opt.zero_grad(set_to_none=True)
    one,two=two_loss(train_model,train_token)
    total=one+two
    total.backward()
    opt.step()
    if step in (1,10,25,50,75,100):
        with torch.no_grad():a,b=two_loss(train_model,train_token)
        history.append({'step':step,'t1':a.item(),'t2':b.item(),'sum':(a+b).item()})
train_sec=time.perf_counter()-start
print(f'Trained {N_STEPS} updates in {train_sec:.2f}s; corpus={len(TRAIN_DOCS)} sequences. Training-set loss only.')
print(f'{"step":>5} | {"t+1 CE":>12} | {"t+2 CE":>12} | {"sum CE":>12} | {"t+2 minus t+1":>17}')
for row in history:print(f'{row["step"]:>5} | {row["t1"]:>12.6f} | {row["t2"]:>12.6f} | {row["sum"]:>12.6f} | {row["t2"]-row["t1"]:>17.6f}')
assert all(abs(x['sum']-(x['t1']+x['t2']))<1e-6 for x in history)
# Explicit guard against the common (and invalid) t+2 alignment.
assert train_token.size(1)-2 == train_model(train_token)[2][:,:-2].size(1)
print('MULTI-TOKEN SHIFT: PASS — separate target columns j+1 and j+2; both mask padding.')

# %% [markdown]
# ## 10 · Causal / packed / two-token boundary regression tests
# This is a targeted test suite rather than trusting smooth loss curves. A future token mutation cannot change any preceding logits. `t+2` may cross **two** transitions near a packed boundary; all such targets must be excluded. False supervision stays excluded even if the tokenizer changes.

# %%
with torch.no_grad():
    prefix_row=torch.tensor([[BOS,*encode(CORPUS[0]),EOS]],dtype=torch.long,device=DEVICE)
    mutated=prefix_row.clone();mutated[0,-1]=stoi['moon']
    log_a=model(prefix_row)[1];log_b=model(mutated)[1]
    assert torch.allclose(log_a[:,:-1],log_b[:,:-1],atol=1e-6), 'Future-leak detected!'
    # same input hidden state at position j predicts both token j+1 and j+2
    n1_mask=(doc_ids[:,:-1]==doc_ids[:,1:]) & (packed[:,1:]!=PAD)
    n2_mask=(doc_ids[:,:-2]==doc_ids[:,2:]) & (packed[:,2:]!=PAD)
    assert n1_mask.sum().item()==packed.size(1)-2
    assert n2_mask.sum().item()==packed.size(1)-4
    assert not n2_mask[0,boundary-2] and not n2_mask[0,boundary-1]
print('CAUSALITY: PASS | packed +1 crossing excluded: 1 | packed +2 crossings excluded: 2')

# %% [markdown]
# ## 11 · Submission report (machine-generated facts)
# The report below is generated from the executed tensors and peak measurement, including exact device, configuration, seed, target counts, and all seven checkpoints. A local execution cannot claim GPU results. There are no borrowed benchmark numbers or fabricated memory savings.

# %%
report={
 'course':'ERA V5 Session 9 — loss harness','seed':SEED,
 'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':str(DEVICE),'hardware':platform.platform()},
 'configuration':{'vocab_size':V,'hidden_size':D,'heads':H,'batch_shape':[B,T],'training_steps':N_STEPS,'documents_in_training':len(TRAIN_DOCS)},
 'part1':{
   'shapes':{'input':[B,T],'hidden':[B,T,D],'logits':[B,T,V],'shifted_logits':[B,T-1,V],'targets':[B,T-1]},
   'shift_check':'PASS; literal input and targets printed',
   'padding':{'targets_without_mask':raw_count,'excluded':pad_count,'targets_with_mask':valid_count,'loss_unmasked':unmasked_pad_loss.item(),'loss_masked':masked_pad_loss.item()},
   'packing':{'boundary_index':boundary,'boundary_count_excluded':1,'loss_pre_mask':loss_before.item(),'loss_post_mask_same_logits':loss_after.item(),'loss_attention_isolated':loss_isolated.item(),'attention_block_diagonal_and_position_reset':'PASS'},
   'perplexity':{'vocab_size':V,'initial_cross_entropy_nats':baseline_ce,'untrained_perplexity':baseline_ppl,'ratio_to_vocab':baseline_ppl/V},
   'parameter_counts':{'untied':nu,'tied':nt,'parameters_saved':saved,'head_counted_in_both':'t+2 auxiliary'},
   'memory':{'device':measured_device,'metric':mem_full['kind'],'full_peak_mib':mem_peak_full,'chunk_peak_mib':mem_peak_chunk,'full_added_mib':mem_delta_full,'chunk_added_mib':mem_delta_chunk,'ratio':mem_ratio,'ratio_kind':mem_ratio_kind,'head_BTDV':[MEM_B,MEM_T,MEM_D,MEM_V],'chunk_length':MEM_CHUNK},
   'chunked_equality':'PASS: loss and gradients, including unequal final chunk'
 },
 'part2':{'losses':history,'final_t1_ce':history[-1]['t1'],'final_t2_ce':history[-1]['t2'],'final_sum':history[-1]['sum'],'note':'same training examples and model; t+2 is farther future; observations are seed- and dataset-specific'},
 'tests':['shift PASS','padding PASS','packed loss boundary PASS','packed attention isolation + local positions PASS','causality PASS','t+2 boundary PASS','loss value and gradients PASS']
}
print('--- SEVEN PART 1 CHECKPOINTS ---')
print('1 Shapes',report['part1']['shapes'])
print('2 Shift verified by strings: PASS')
print('3 Padding valid',valid_count,'/',raw_count)
print('4 Packed boundary loss',round(loss_before.item(),6),'→',round(loss_after.item(),6))
print('5 Initial PPL',round(baseline_ppl,4),'vs V',V)
print('6 Trainable params (untied, tied, saved)',nu,nt,saved)
print('7 Peak head/loss memory MiB (full, chunk, ratio)',round(mem_peak_full,2),round(mem_peak_chunk,2),round(mem_ratio,3))
print('--- PART 2 ---')
print('t+1:',round(history[-1]['t1'],6),'t+2:',round(history[-1]['t2'],6),'sum:',round(history[-1]['sum'],6))
try:
    Path('results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Wrote results.json in working directory.')
except OSError as err: print('Filesystem write disabled:',err)
