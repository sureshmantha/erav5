from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class BlockSelfAttention(nn.Module):
    def __init__(self,d_model:int,n_heads:int):
        super().__init__(); assert d_model%n_heads==0
        self.n_heads=n_heads; self.head_dim=d_model//n_heads
        self.qkv=nn.Linear(d_model,3*d_model,bias=False)
        self.out=nn.Linear(d_model,d_model,bias=False)
    def forward(self,x,allowed_mask):
        B,L,D=x.shape
        qkv=self.qkv(x).reshape(B,L,3,self.n_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        mask=allowed_mask[:,None,:,:]
        scores=scores.masked_fill(~mask,-1e4)
        weights=torch.softmax(scores,dim=-1)
        y=torch.matmul(weights,v).transpose(1,2).contiguous().reshape(B,L,D)
        return self.out(y)


class TinyCausalLM(nn.Module):
    def __init__(self,vocab_size:int,max_seq_len:int,d_model:int=32,n_heads:int=4,d_ff:int=64):
        super().__init__()
        self.token_emb=nn.Embedding(vocab_size,d_model)
        self.pos_emb=nn.Embedding(max_seq_len,d_model)
        self.ln1=nn.LayerNorm(d_model); self.attn=BlockSelfAttention(d_model,n_heads)
        self.ln2=nn.LayerNorm(d_model)
        self.ff=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),nn.Linear(d_ff,d_model))
        self.ln_f=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vocab_size,bias=False)
    def forward(self,input_ids,attention_mask,position_ids):
        x=self.token_emb(input_ids)+self.pos_emb(position_ids)
        x=x+self.attn(self.ln1(x),attention_mask)
        x=x+self.ff(self.ln2(x))
        return self.head(self.ln_f(x))


def initialize_model(vocab_size:int,max_seq_len:int,seed:int=20260801):
    torch.manual_seed(seed)
    model=TinyCausalLM(vocab_size,max_seq_len)
    return model


def compute_next_token_losses(model:nn.Module,input_ids:torch.Tensor,attention_mask:torch.Tensor,position_ids:torch.Tensor,loss_mask:torch.Tensor):
    logits=model(input_ids,attention_mask,position_ids)
    B,L,V=logits.shape
    targets=input_ids[:,1:].contiguous()
    pred=logits[:,:-1,:].contiguous()
    per=F.cross_entropy(pred.reshape(-1,V),targets.reshape(-1),reduction="none").reshape(B,L-1)
    target_mask=loss_mask[:,1:]
    denom=target_mask.sum().clamp_min(1.0)
    loss=(per*target_mask).sum()/denom
    sample_den=target_mask.sum(dim=1).clamp_min(1.0)
    sample_loss=(per*target_mask).sum(dim=1)/sample_den
    return loss,per,target_mask,sample_loss


def grad_norm(model:nn.Module)->float:
    total=0.0
    for p in model.parameters():
        if p.grad is not None:
            total+=float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)
