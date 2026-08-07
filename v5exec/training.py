from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .batches import build_batch
from .checkpoint import CheckpointManager
from .hashing import sha256_json
from .io import count_jsonl
from .ledgers import Ledgers
from .model import compute_next_token_losses, grad_norm


class SimulatedCrash(RuntimeError):
    def __init__(self,expected_next_batch_id:str):
        super().__init__(f"deliberate crash before {expected_next_batch_id}")
        self.expected_next_batch_id=expected_next_batch_id


@dataclass
class TrainingStats:
    train_seconds: float=0.0
    loader_seconds: float=0.0
    raw_tokens: int=0
    loss_tokens: int=0
    capacity_tokens: int=0
    optimizer_steps: int=0


def _token_rows(run_id,step,batch,per_token,target_mask,tokenizer):
    rows=[]
    per=per_token.detach().cpu(); mask=target_mask.detach().cpu()
    for b,sid in enumerate(batch.sample_ids):
        for i in range(per.shape[1]):
            if not bool(mask[b,i].item()): continue
            target_id=int(batch.input_ids[b,i+1].item())
            loss=float(per[b,i].item())
            rows.append({
                "run_id":run_id,"global_step":step,"batch_id":batch.batch_id,"sample_id":sid,
                "prediction_position":i,"target_position":i+1,"token_id":target_id,
                "token_display":tokenizer.token_display(target_id),"loss":round(loss,8),
                "perplexity":round(math.exp(min(loss,20.0)),8),
                "source_spans":batch.source_spans[b],
            })
    return rows


def train_range(*,run_id:str,model,optimizer,plan:dict,catalog:dict,tokenizer,ledgers:Ledgers,checkpoints:CheckpointManager,start_batch:int,end_batch:int,checkpoint_every:int,stats:TrainingStats,logger,crash_after_step:int|None=None,parent_checkpoint_hash:str|None=None,branch:str="main"):
    for batch_idx in range(start_batch,end_batch):
        t0=time.perf_counter(); entry=plan["batches"][batch_idx]; batch=build_batch(entry,catalog,tokenizer.PAD_ID); stats.loader_seconds+=time.perf_counter()-t0
        # Batch is consumed only after the optimizer update succeeds.
        t1=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        loss,per,target_mask,sample_loss=compute_next_token_losses(model,batch.input_ids,batch.attention_mask,batch.position_ids,batch.loss_mask)
        loss.backward(); gnorm=grad_norm(model); optimizer.step(); train_dt=time.perf_counter()-t1
        stats.train_seconds+=train_dt; stats.raw_tokens+=batch.nonpad_tokens; stats.loss_tokens+=int(target_mask.sum().item()); stats.capacity_tokens+=batch.capacity_tokens; stats.optimizer_steps+=1
        step=batch_idx+1
        consumption={
            "run_id":run_id,"branch":branch,"global_step":step,"ledger_offset":count_jsonl(ledgers.consumption),
            "rank":0,"world_size":1,"microbatch_index":0,"data_loader_version":"v5exec-1.0",
            "batch_index":batch_idx,"batch_id":batch.batch_id,"batch_hash":batch.batch_hash,"stage":batch.stage,
            "sample_ids":batch.sample_ids,"sample_hashes":batch.sample_hashes,"source_spans":batch.source_spans,
            "shard_ids":sorted({sp["shard_id"] for spans in batch.source_spans for sp in spans}),
            "packing_policies":[catalog[sid].policy for sid in batch.sample_ids],
            "lanes":batch.lanes,"lane_token_counts":batch.lane_token_counts,"nonpad_tokens":batch.nonpad_tokens,
            "capacity_tokens":batch.capacity_tokens,"loss_bearing_tokens":int(target_mask.sum().item()),
            "packing_utilization":round(batch.nonpad_tokens/batch.capacity_tokens,8),
            "useful_loss_utilization":round(int(target_mask.sum().item())/batch.capacity_tokens,8),
            "attention_mask_hash":sha256_json(batch.attention_mask.int().tolist()),
            "loss_mask_hash":sha256_json(batch.loss_mask.int().tolist()),"position_ids_hash":sha256_json(batch.position_ids.tolist()),
            "opus_decision_ids":batch.opus_decision_ids,"tokenizer_hash":tokenizer.tokenizer_hash,"plan_hash":plan["plan_sha256"],
        }
        ledgers.append_consumption(consumption)
        learning={
            "run_id":run_id,"branch":branch,"global_step":step,"batch_id":batch.batch_id,"batch_hash":batch.batch_hash,
            "mean_loss":round(float(loss.item()),8),"mean_perplexity":round(math.exp(min(float(loss.item()),20.0)),8),
            "sample_losses":[round(float(x),8) for x in sample_loss.detach().cpu().tolist()],
            "sample_ids":batch.sample_ids,"gradient_norm":round(float(gnorm),8),"stage":batch.stage,
            "loss_bearing_tokens":int(target_mask.sum().item()),"train_seconds":round(train_dt,8),
            "source_spans":batch.source_spans,"opus_decision_ids":batch.opus_decision_ids,
            "learning_signal":"high" if float(loss.item())>4.0 else "medium" if float(loss.item())>2.0 else "low",
        }
        ledgers.append_learning(learning)
        ledgers.append_token_learning_rows(_token_rows(run_id,step,batch,per,target_mask,tokenizer))

        if step%checkpoint_every==0 or step==end_batch:
            meta=checkpoints.save(model=model,optimizer=optimizer,step=step,next_batch_index=step,
                                  ledger_offsets=ledgers.offsets(),plan_hash=plan["plan_sha256"],tokenizer_hash=tokenizer.tokenizer_hash,
                                  parent_checkpoint_hash=parent_checkpoint_hash,branch=branch)
            logger.pass_("checkpoint_saved",f"step={step} hash={meta['checkpoint_sha256'][:12]}")
            parent_checkpoint_hash=meta["checkpoint_sha256"]
            if crash_after_step is not None and step==crash_after_step:
                expected=plan["batches"][step]["batch_id"] if step<len(plan["batches"]) else "<END>"
                logger.info(f"crash simulated after durable checkpoint step={step}; expected_next_batch={expected}")
                raise SimulatedCrash(expected)
    return parent_checkpoint_hash
