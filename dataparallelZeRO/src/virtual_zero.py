"""32 logical-rank ZeRO teaching simulator, intentionally NOT a distributed GPU runtime.

Uses real PyTorch forward/backward on CPU, simulated ownership/collective transfers,
real sharded Adam updates, and analytical BF16/FP32 memory & ring-traffic estimates.
"""
from __future__ import annotations
import dataclasses
import time
from typing import List
import torch
import torch.nn.functional as F

WORLD_SIZE = 32
INPUT_DIM = 16
HIDDEN_DIM = 32
CLASSES = 8
LAYER_ELEMENTS = (HIDDEN_DIM * INPUT_DIM + HIDDEN_DIM, CLASSES * HIDDEN_DIM + CLASSES)
NUM_PARAMS = sum(LAYER_ELEMENTS)
LR = 0.008
BETA1, BETA2, EPS = .9, .999, 1e-8

@dataclasses.dataclass
class Rank:
    id: int
    left: int
    right: int
    params: torch.Tensor
    first_moment: torch.Tensor
    second_moment: torch.Tensor
    grad: torch.Tensor | None = None
    owner_gradient: torch.Tensor | None = None


def partitions(count: int, world: int = WORLD_SIZE):
    return [(i * count // world, (i+1) * count // world) for i in range(world)]


def initial_weights(seed: int = 20260912):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(NUM_PARAMS, generator=gen, dtype=torch.float64) * .055).detach()


def rank_data(seed: int = 312, world: int = WORLD_SIZE, batch_per_rank: int = 2):
    """Fixed deterministic supervised toy data; no external corpus claims."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(world, batch_per_rank, INPUT_DIM, generator=g, dtype=torch.float64)
    teacher = torch.randn(INPUT_DIM, CLASSES, generator=g, dtype=torch.float64)
    y = (x @ teacher).argmax(-1)
    return [(x[i], y[i]) for i in range(world)]


def linear_pieces(ranks: list[Rank], absolute_start: int, length: int):
    """The correct sliced owner fragments, without constructing an entire flat model."""
    end = absolute_start + length
    pieces = []
    for r in ranks:
        lo = max(r.left, absolute_start)
        hi = min(r.right, end)
        if lo < hi:
            pieces.append(r.params[lo-r.left:hi-r.left])
    assert sum(p.numel() for p in pieces) == length
    return pieces


class JustInTimeShardedLinear(torch.autograd.Function):
    """Gather a layer in forward and rematerialize it in backward, then shard its gradient.

    This is **local torch.cat**, not a hardware or NCCL collective. Backward saves only
    input and owner views, not the gathered parameter tensor.
    """
    @staticmethod
    def forward(ctx, x, out_dim: int, in_dim: int, *pieces):
        full = torch.cat(pieces, dim=0)
        weight = full[:out_dim*in_dim].reshape(out_dim, in_dim)
        bias = full[out_dim*in_dim:]
        ctx.save_for_backward(x, *pieces)
        ctx.dims = (out_dim, in_dim)
        ctx.sizes = [p.numel() for p in pieces]
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, *pieces = ctx.saved_tensors
        out_dim, in_dim = ctx.dims
        # Second parameter materialization during backward: ZeRO-3-like behavior.
        full = torch.cat(pieces, dim=0)
        w = full[:out_dim*in_dim].reshape(out_dim, in_dim)
        grad_x = grad_output @ w
        grad_w = grad_output.transpose(-1,-2) @ x
        grad_b = grad_output.sum(dim=0)
        grad_flat = torch.cat((grad_w.reshape(-1), grad_b))
        return (grad_x, None, None, *grad_flat.split(ctx.sizes))


def full_forward(x, flat):
    first, second = LAYER_ELEMENTS
    w1 = flat[:HIDDEN_DIM*INPUT_DIM].reshape(HIDDEN_DIM, INPUT_DIM)
    b1 = flat[HIDDEN_DIM*INPUT_DIM:first]
    w2 = flat[first:first+CLASSES*HIDDEN_DIM].reshape(CLASSES, HIDDEN_DIM)
    b2 = flat[first+CLASSES*HIDDEN_DIM:]
    return F.linear(F.relu(F.linear(x,w1,b1)), w2, b2)


def stage3_forward(x, ranks):
    first, second = LAYER_ELEMENTS
    p1 = linear_pieces(ranks, 0, first)
    h = F.relu(JustInTimeShardedLinear.apply(x, HIDDEN_DIM, INPUT_DIM, *p1))
    p2 = linear_pieces(ranks, first, second)
    return JustInTimeShardedLinear.apply(h, CLASSES, HIDDEN_DIM, *p2)


def initialize(stage: int, start: torch.Tensor, world: int = WORLD_SIZE):
    assert stage in (0, 1, 2, 3)
    result=[]
    for i,(lo,hi) in enumerate(partitions(start.numel(),world)):
        # Keep model parameter full except stage 3.
        p = start[lo:hi].clone() if stage==3 else start.clone()
        # Optimizer states stay full only for baseline (stage 0).
        m = torch.zeros_like(start if stage==0 else start[lo:hi])
        v = torch.zeros_like(m)
        p.requires_grad_(stage==3)
        result.append(Rank(i,lo,hi,p,m,v))
    return result


def adam_inplace(p:torch.Tensor,m:torch.Tensor,v:torch.Tensor,g:torch.Tensor,step:int):
    """Adam, with no weight decay, identical update for every stage."""
    with torch.no_grad():
        m.mul_(BETA1).add_(g,alpha=1-BETA1)
        v.mul_(BETA2).addcmul_(g,g,value=1-BETA2)
        update = (m / (1-BETA1**step)) / ((v/(1-BETA2**step)).sqrt()+EPS)
        p.add_(update,alpha=-LR)


def reconstruct(stage:int,ranks:list[Rank]):
    return torch.cat([r.params.detach() for r in ranks]) if stage==3 else ranks[0].params.detach().clone()


def optimizer_step(stage:int,ranks:list[Rank],step:int):
    if stage==0:
        for r in ranks:
            assert r.grad is not None
            adam_inplace(r.params,r.first_moment,r.second_moment,r.grad,step)
    else:
        for r in ranks:
            assert r.grad is not None
            if stage==3:
                # Parameters, gradients, optimizer moments owned only by this rank.
                adam_inplace(r.params,r.first_moment,r.second_moment,r.grad,step)
            else:
                # Only shard owners update; afterward all-gather updated weights.
                segment = r.params[r.left:r.right]
                adam_inplace(segment,r.first_moment,r.second_moment,
                             r.owner_gradient if stage==1 else r.grad,step)
        if stage in (1,2):
            assembled = torch.cat([r.params[r.left:r.right] for r in ranks]).detach()
            for r in ranks:
                r.params.copy_(assembled)


def execute(stage:int,steps:int=4,world:int=WORLD_SIZE, verbose:bool=False):
    assert world == WORLD_SIZE, 'This assignment deliberately fixes 32 virtual ranks.'
    ranks = initialize(stage, initial_weights(), world)
    data=rank_data(world=world)
    losses=[]; step_ms=[]
    with torch.no_grad():
        _=full_forward(data[0][0], initial_weights())
    for step in range(1,steps+1):
        t0=time.perf_counter()
        if stage==3:
            for r in ranks:
                r.params.grad = None
            batch_losses=[]
            for x,y in data:
                score = stage3_forward(x,ranks)
                loss = F.cross_entropy(score,y)
                (loss / world).backward()
                batch_losses.append(loss.detach().item())
            # Each owner has its reduced/scattered gradient accumulated on its own shard.
            for r in ranks: r.grad = r.params.grad.detach().clone()
        else:
            local_full_grads=[]; batch_losses=[]
            for rank,(x,y) in zip(ranks,data):
                w = rank.params.clone().detach().requires_grad_(True)
                score = full_forward(x,w)
                loss = F.cross_entropy(score,y)
                loss.backward()
                local_full_grads.append(w.grad.detach().clone())
                batch_losses.append(loss.detach().item())
            if stage==0:
                average = torch.stack(local_full_grads).mean(0)
                for r in ranks: r.grad = average.clone() # all-reduce
            elif stage==1:
                # Each rank retains its *full local gradient*; reduce-scatter the
                # synchronized gradient only to the optimizer shard's owner.
                for r,g in zip(ranks,local_full_grads):
                    r.grad = g.clone()
                    r.owner_gradient = sum((q[r.left:r.right] for q in local_full_grads),
                                           torch.zeros(r.right-r.left,dtype=torch.float64))/world
            elif stage==2:
                # Reduce-scatter directly into owner buffers, without retaining full reduced grads.
                for r in ranks:
                    r.grad = sum((g[r.left:r.right] for g in local_full_grads),
                                 torch.zeros(r.right-r.left,dtype=torch.float64)) / world
        losses.append(sum(batch_losses)/world)
        optimizer_step(stage,ranks,step)
        step_ms.append((time.perf_counter()-t0)*1000)
        if verbose: print(f'  step {step}: pre-update global CE={losses[-1]:.10f}, elapsed_cpu_ms={step_ms[-1]:.2f}')
        # Exact agreement of full model copies in baseline / stages 1/2.
        if stage!=3:
            assert all(torch.equal(r.params,ranks[0].params) for r in ranks)
    return dict(stage=stage, world=world, losses=losses, final=reconstruct(stage,ranks),
                mean_cpu_step_ms=sum(step_ms)/len(step_ms), cpu_step_ms=step_ms,
                rank_state_elements=[dict(rank=r.id,weight=r.params.numel(),
                    gradient=r.grad.numel(),moments=r.first_moment.numel()+r.second_moment.numel()) for r in ranks])


def resident_bytes_per_parameter(stage, world=32):
    # Ideal accounting: 2B low precision parameters, 2B low precision gradients,
    # 4B fp32 master weights + 4B momentum + 4B variance = 12B optimizer state.
    return ({0:16,1:4+12/world,2:2+14/world,3:16/world})[stage]


def memory_table(params=NUM_PARAMS,world=WORLD_SIZE):
    # 'comm' uses ring volume per rank in units of S = params * 2B,
    # not wall-clock transfer time, and excludes latency / overlap / temporary buffers.
    factor=(world-1)/world
    out=[]
    for stage in range(4):
        p= params * resident_bytes_per_parameter(stage, world)
        # Illustrative, optimally staged ring collective: stage0 AR (2S),
        # stage1/2 RS + AG (2S), stage3 RS + forward AG + backward AG (3S).
        units = factor*(3 if stage==3 else 2)
        comm = params*2*units
        out.append(dict(stage=stage,resident_bytes_per_rank=p,
                    resident_MiB_per_rank=p/2**20,
                    resident_fraction_of_dp=p/(params*16),
                    modeled_ring_bytes_per_rank=comm,
                    modeled_ring_MiB_per_rank=comm/2**20,
                    communication_S_units=units,
                    # Dense forward/backward computation is not sharded across data ranks.
                    forward_backward_relative_flops_per_rank=1.0,
                    optimizer_math_elements_per_rank=params if stage==0 else params/world,
                    optimizer_math_fraction=1.0 if stage==0 else 1/world))
    return out


def stage3_extra_buffer_bytes(layer_sizes=LAYER_ELEMENTS):
    # At least one gathered BF16 layer parameter buffer (excluding fragmented grads & caching).
    return max(layer_sizes)*2


def projected_state_for_30b(world=32):
    return memory_table(params=30_000_000_000, world=world)


def validate():
    assert len(partitions(NUM_PARAMS)) == WORLD_SIZE
    ranges=partitions(NUM_PARAMS)
    assert ranges[0][0]==0 and ranges[-1][1]==NUM_PARAMS
    assert all(ranges[i][1]==ranges[i+1][0] for i in range(WORLD_SIZE-1))
    assert sum(end-start for start,end in ranges)==NUM_PARAMS
    assert max(end-start for start,end in ranges)-min(end-start for start,end in ranges)<=1
    x=rank_data()[0][0]
    shard=initialize(3,initial_weights())
    with torch.no_grad():
        a=full_forward(x,initial_weights())
        b=stage3_forward(x,shard)
    assert torch.allclose(a,b,atol=1e-14,rtol=1e-14)
    assert sum(r.params.numel() for r in shard)==NUM_PARAMS
    assert all(r.params.numel() < NUM_PARAMS for r in shard)
    for n in (8,32):
        assert resident_bytes_per_parameter(0,n)==16
        assert resident_bytes_per_parameter(3,n)==16/n
        assert resident_bytes_per_parameter(1,n)==4+12/n
        assert resident_bytes_per_parameter(2,n)==2+14/n
    results=[execute(stage=s,steps=4) for s in range(4)]
    ref=results[0]
    for r in results[1:]:
        max_diff=(ref['final']-r['final']).abs().max().item()
        max_loss_diff=max(abs(a-b) for a,b in zip(ref['losses'],r['losses']))
        assert max_diff<1e-12,(r['stage'],max_diff)
        assert max_loss_diff<1e-12,(r['stage'],max_loss_diff)
        r['max_parameter_error_vs_baseline']=max_diff
        r['max_loss_error_vs_baseline']=max_loss_diff
    ref['max_parameter_error_vs_baseline']=0.0
    ref['max_loss_error_vs_baseline']=0.0
    assert ref['losses'][-1] < ref['losses'][0], ref['losses']
    return results

if __name__ == '__main__':
    print('ENV: torch',torch.__version__,'CUDA',torch.cuda.is_available(),'CPU threads',torch.get_num_threads())
    print('Model params',NUM_PARAMS,'32 virtual ranks')
    results=validate()
    for r in results:
        print(r['stage'],['%.9f'%v for v in r['losses']],
              'err',r['max_parameter_error_vs_baseline'],'mean CPU ms',round(r['mean_cpu_step_ms'],2))
    for x in memory_table(): print(x)
