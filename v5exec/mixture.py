from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .hashing import sha256_json, stable_u64
from .packing import PackedSample


@dataclass
class OpusDecision:
    decision_id: str
    stage: str
    lane: str
    sample_id: str
    score: float
    base_decision: str
    final_decision: str
    protected_floor_override: bool
    reason: str
    quality: float
    novelty: float
    difficulty_fit: float
    occurrence: int
    shard_ids: list[str]
    gradient_alignment: float
    gold_proxy_sample_id: str | None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class OpusLikeSelector:
    """Auditable OPUS-style selector used by the execution proof.

    It combines a measured cheap gradient-alignment proxy with sample quality,
    byte-token novelty, curriculum difficulty fit and replay pressure. It is not
    claimed to reproduce an unpublished production selector.
    """
    def __init__(self, accept_threshold: float = 0.70, defer_threshold: float = 0.56):
        self.accept_threshold=accept_threshold; self.defer_threshold=defer_threshold

    @staticmethod
    def _novelty(sample: PackedSample) -> float:
        toks=[t for t in sample.input_ids if t>=4]
        if not toks: return 0.0
        return min(1.0, len(set(toks))/max(24.0, len(toks)*0.35))

    @staticmethod
    def _target_difficulty(stage_name: str) -> int:
        return {"S0":0,"S1":1,"S2":2,"S3":3,"S4":3,"S5":4}.get(stage_name,2)

    def score(self, sample: PackedSample, stage_name: str, occurrence: int, alignment_record: dict | None = None) -> tuple[float,float,float,float,str|None]:
        novelty=self._novelty(sample)
        target=self._target_difficulty(stage_name)
        fit=max(0.0,1.0-abs(sample.difficulty-target)/4.0)
        replay_penalty=max(0.82,1.0-0.02*occurrence)
        raw_alignment=float(alignment_record.get("gradient_alignment",0.0)) if alignment_record else 0.0
        alignment_unit=(raw_alignment+1.0)/2.0
        gold_id=alignment_record.get("gold_proxy_sample_id") if alignment_record else None
        score=(0.35*sample.quality+0.20*novelty+0.15*fit+0.25*alignment_unit+0.05*replay_penalty)
        return round(score,6),round(novelty,6),round(fit,6),round(raw_alignment,8),gold_id

    def base_decision(self, score: float) -> str:
        if score>=self.accept_threshold: return "accepted"
        if score>=self.defer_threshold: return "deferred"
        return "rejected"


def _largest_remainder(weights: dict[str,float], slots: int) -> dict[str,int]:
    raw={k:v*slots for k,v in weights.items()}
    base={k:int(math.floor(x)) for k,x in raw.items()}
    remaining=slots-sum(base.values())
    order=sorted(weights,key=lambda k:(-(raw[k]-base[k]),k))
    for k in order[:remaining]: base[k]+=1
    return base


def _apply_floors(counts: dict[str,int], floors: dict[str,float], slots: int) -> dict[str,int]:
    counts=dict(counts)
    min_counts={k:(math.ceil(v*slots) if v>0 else 0) for k,v in floors.items()}
    for lane,m in min_counts.items():
        counts[lane]=max(counts.get(lane,0),m)
    while sum(counts.values())>slots:
        candidates=[k for k in counts if counts[k]>min_counts.get(k,0)]
        if not candidates: raise RuntimeError("protected floors exceed stage capacity")
        k=max(candidates,key=lambda x:(counts[x]-min_counts.get(x,0),x))
        counts[k]-=1
    while sum(counts.values())<slots:
        k=max(counts,key=lambda x:(counts[x]-min_counts.get(x,0),x))
        counts[k]+=1
    return counts


def compile_schedule(policy: dict, samples: list[PackedSample], *, slots_per_stage: int, batch_size: int, seed: int, proxy_alignment: dict | None = None) -> tuple[dict,list[dict],list[dict]]:
    if not proxy_alignment or not proxy_alignment.get("scores"):
        raise ValueError("OPUS gradient proxy scores are required; compute them with compute_gradient_proxy_scores()")
    pools=defaultdict(list)
    for s in samples: pools[s.lane].append(s)
    for lane in pools:
        pools[lane].sort(key=lambda s:(stable_u64(seed,lane,s.sample_id),s.sample_id))
    selector=OpusLikeSelector(policy["opus"]["accept_threshold"],policy["opus"]["defer_threshold"])
    usage=Counter(); decisions=[]; selected_occurrences=[]; stage_proofs=[]

    for sidx,stage in enumerate(policy["stages"]):
        name=stage["name"]; weights=stage["weights"]; floors=stage["floors"]
        base=_largest_remainder(weights,slots_per_stage)
        quotas=_apply_floors(base,floors,slots_per_stage)
        stage_selected=[]; stage_decisions=[]
        for lane in sorted(quotas):
            need=quotas[lane]
            if need<=0: continue
            if lane not in pools or not pools[lane]: raise RuntimeError(f"no training samples for lane {lane}")
            floor_count=math.ceil(floors.get(lane,0)*slots_per_stage) if floors.get(lane,0)>0 else 0
            chosen=0; attempts=0; max_attempts=max(need*8,16)
            while chosen<need and attempts<max_attempts:
                pool=pools[lane]
                sample=pool[(usage[(name,lane,"cursor")]+attempts)%len(pool)]
                occurrence=usage[sample.sample_id]
                alignment_record=(proxy_alignment or {}).get("scores",{}).get(sample.sample_id) if proxy_alignment else None
                score,novelty,fit,gradient_alignment,gold_proxy_sample_id=selector.score(sample,name,occurrence,alignment_record)
                base_decision=selector.base_decision(score)
                # When a lane's target is its protected floor, OPUS cannot reduce it.
                quota_is_floor=need<=floor_count
                override=base_decision!="accepted" and quota_is_floor
                final="floor_override" if override else base_decision
                reason=("protected floor would be crossed" if override else
                        "sample above accept threshold" if final=="accepted" else
                        "borderline sample deferred for a later stage" if final=="deferred" else
                        "sample below selector threshold")
                did=sha256_json({"stage":name,"lane":lane,"sample":sample.sample_id,"occ":occurrence,"attempt":attempts})[:20]
                d=OpusDecision("opus_"+did,name,lane,sample.sample_id,score,base_decision,final,override,reason,sample.quality,novelty,fit,occurrence,sorted({sp["shard_id"] for sp in sample.source_spans}),gradient_alignment,gold_proxy_sample_id)
                decisions.append(d.as_dict()); stage_decisions.append(d.as_dict())
                attempts+=1
                if final in {"accepted","floor_override"}:
                    stage_selected.append({"sample_id":sample.sample_id,"lane":lane,"opus_decision_id":d.decision_id,"opus_final_decision":final,"occurrence":occurrence})
                    usage[sample.sample_id]+=1; chosen+=1
            usage[(name,lane,"cursor")]+=attempts
            if chosen<need:
                raise RuntimeError(f"selector could not fill quota stage={name} lane={lane} chosen={chosen} need={need}")

        # Deterministic interleave so batches see mixed lanes rather than lane blocks.
        stage_selected.sort(key=lambda x:(stable_u64(seed,name,x["sample_id"],x["occurrence"]),x["lane"],x["sample_id"]))
        counts=Counter(x["lane"] for x in stage_selected)
        proof={
            "stage":name,"slots":slots_per_stage,"weights":weights,"floors":floors,
            "quota_counts":quotas,"actual_sample_counts":dict(counts),
            "actual_sample_shares":{k:counts[k]/slots_per_stage for k in weights},
            "floor_compliant":all(counts.get(k,0)/slots_per_stage+1e-12>=v for k,v in floors.items()),
        }
        stage_proofs.append(proof)
        selected_occurrences.extend({**x,"stage":name,"stage_index":sidx} for x in stage_selected)

    if len(selected_occurrences)%batch_size: raise RuntimeError("selected slots not divisible by batch size")
    batches=[]
    for i in range(0,len(selected_occurrences),batch_size):
        items=selected_occurrences[i:i+batch_size]
        stage=items[0]["stage"]
        if any(x["stage"]!=stage for x in items): raise RuntimeError("batch crosses curriculum stage")
        batch_id="batch_"+sha256_json({"seed":seed,"index":i//batch_size,"stage":stage,"items":items})[:20]
        batches.append({"batch_index":i//batch_size,"batch_id":batch_id,"stage":stage,"items":items})
    plan={
        "schema_version":1,"seed":seed,"slots_per_stage":slots_per_stage,"batch_size":batch_size,
        "stage_proofs":stage_proofs,"batches":batches,
    }
    plan["plan_sha256"]=sha256_json({k:v for k,v in plan.items() if k!="plan_sha256"})
    return plan,decisions,stage_proofs
