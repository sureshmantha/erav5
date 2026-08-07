from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch

from .hashing import sha256_json
from .packing import PackedSample


@dataclass
class Batch:
    batch_id: str
    stage: str
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    sample_ids: list[str]
    sample_hashes: list[str]
    source_spans: list[list[dict]]
    lanes: list[str]
    opus_decision_ids: list[str]
    batch_hash: str
    lane_token_counts: dict[str,int]
    nonpad_tokens: int
    loss_bearing_tokens: int
    capacity_tokens: int


def build_batch(entry: dict, catalog: dict[str,PackedSample], pad_id: int) -> Batch:
    samples=[catalog[x["sample_id"]] for x in entry["items"]]
    input_ids=torch.tensor([s.input_ids for s in samples],dtype=torch.long)
    loss_mask=torch.tensor([s.loss_mask for s in samples],dtype=torch.float32)
    attention_mask=torch.tensor([s.attention_mask for s in samples],dtype=torch.bool)
    position_ids=torch.tensor([s.position_ids for s in samples],dtype=torch.long)
    nonpad=int((input_ids!=pad_id).sum().item())
    loss_tokens=int(loss_mask.sum().item())
    capacity=input_ids.numel()
    lane_counts=Counter()
    for s in samples:
        lane_counts[s.lane]+=sum(1 for t in s.input_ids if t!=pad_id)
    payload={
        "batch_id":entry["batch_id"],"stage":entry["stage"],"sample_hashes":[s.sample_hash for s in samples],
        "input_ids":input_ids.tolist(),"loss_mask":loss_mask.int().tolist(),"attention_mask":attention_mask.int().tolist(),
        "position_ids":position_ids.tolist(),"source_spans":[s.source_spans for s in samples],
    }
    batch_hash=sha256_json(payload)
    return Batch(entry["batch_id"],entry["stage"],input_ids,loss_mask,attention_mask,position_ids,
                 [s.sample_id for s in samples],[s.sample_hash for s in samples],[s.source_spans for s in samples],
                 [s.lane for s in samples],[x["opus_decision_id"] for x in entry["items"]],batch_hash,
                 dict(lane_counts),nonpad,loss_tokens,capacity)
