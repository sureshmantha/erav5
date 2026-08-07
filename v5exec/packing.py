from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil
from typing import Iterable

from .firewall import EvaluationFirewall
from .hashing import sha256_json
from .tokenizer import FrozenByteTokenizer

ASSISTANT_ROLES = {"assistant", "assistant_plan", "tool_call", "assistant_revision", "final"}


@dataclass
class PackedSample:
    sample_id: str
    lane: str
    policy: str
    input_ids: list[int]
    loss_mask: list[int]
    attention_mask: list[list[int]]
    position_ids: list[int]
    segment_ids: list[int]
    source_spans: list[dict]
    sample_hash: str
    quality: float
    difficulty: int
    indic_tier: str | None = None

    def recipe(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "lane": self.lane,
            "policy": self.policy,
            "source_spans": self.source_spans,
            "sample_hash": self.sample_hash,
            "quality": self.quality,
            "difficulty": self.difficulty,
            "indic_tier": self.indic_tier,
        }


def _build_attention(segment_ids: list[int], input_ids: list[int], pad_id: int) -> list[list[int]]:
    L = len(input_ids)
    out = [[0] * L for _ in range(L)]
    for i in range(L):
        if input_ids[i] == pad_id:
            out[i][i] = 1
            continue
        for j in range(i + 1):
            if input_ids[j] != pad_id and segment_ids[i] == segment_ids[j]:
                out[i][j] = 1
    return out


def _finalize_sample(*, lane: str, policy: str, tokens: list[int], loss: list[int], positions: list[int], segments: list[int], spans: list[dict], seq_len: int, tokenizer: FrozenByteTokenizer, quality: float, difficulty: int, indic_tier: str | None) -> PackedSample:
    if len(tokens) > seq_len:
        raise ValueError("sample exceeds sequence length")
    pad = seq_len - len(tokens)
    tokens = tokens + [tokenizer.PAD_ID] * pad
    loss = loss + [0] * pad
    positions = positions + [0] * pad
    segments = segments + [-1] * pad
    attn = _build_attention(segments, tokens, tokenizer.PAD_ID)
    payload = {
        "lane": lane, "policy": policy, "input_ids": tokens, "loss_mask": loss,
        "attention_mask": attn, "position_ids": positions, "segment_ids": segments,
        "source_spans": spans, "tokenizer_hash": tokenizer.tokenizer_hash,
    }
    sample_hash = sha256_json(payload)
    sample_id = f"sample_{sample_hash[:20]}"
    return PackedSample(sample_id, lane, policy, tokens, loss, attn, positions, segments, spans, sample_hash, quality, difficulty, indic_tier)


def _record_chunks(record: dict, capacity: int) -> list[tuple[list[int], int, int]]:
    ids = record["token_ids"]
    if not ids:
        return [([], 0, 0)]
    return [(ids[i:i+capacity], i, min(i+capacity, len(ids))) for i in range(0, len(ids), capacity)]


def best_fit_samples(records: list[dict], seq_len: int, tokenizer: FrozenByteTokenizer, lane: str, policy: str = "best_fit_block_isolated") -> list[PackedSample]:
    """Best-fit-decreasing packing. Documents remain isolated by block-causal attention."""
    capacity = seq_len
    items = []
    for r in records:
        EvaluationFirewall.assert_record_trainable(r)
        for chunk, start, end in _record_chunks(r, seq_len - 2):
            items.append((len(chunk) + 2, r, chunk, start, end))
    items.sort(key=lambda x: (-x[0], x[1]["record_id"], x[3]))
    bins: list[dict] = []
    for length, r, chunk, start, end in items:
        best_idx = None
        best_remaining = None
        for i, b in enumerate(bins):
            remaining = seq_len - b["used"]
            if length <= remaining:
                after = remaining - length
                if best_remaining is None or after < best_remaining:
                    best_idx, best_remaining = i, after
        if best_idx is None:
            bins.append({"used": 0, "items": []})
            best_idx = len(bins) - 1
        bins[best_idx]["items"].append((r, chunk, start, end))
        bins[best_idx]["used"] += length

    samples = []
    for bidx, b in enumerate(bins):
        tokens=[]; loss=[]; positions=[]; segments=[]; spans=[]
        qualities=[]; diffs=[]; tiers=[]
        for seg_id, (r, chunk, start, end) in enumerate(b["items"]):
            seg_tokens = [tokenizer.BOS_ID] + chunk + [tokenizer.EOS_ID]
            tokens.extend(seg_tokens)
            loss.extend([0] + [1] * len(chunk) + [1])
            positions.extend(list(range(len(seg_tokens))))
            segments.extend([seg_id] * len(seg_tokens))
            spans.append({
                "shard_id": r["shard_id"], "record_id": r["record_id"], "doc_id": r["doc_id"], "start_token": start,
                "end_token": end, "segment_id": seg_id, "content_hash": r["raw_document_hash"],
            })
            qualities.append(r["quality"]); diffs.append(r["difficulty"]); tiers.append(r.get("indic_tier"))
        tier = next((t for t in tiers if t), None)
        samples.append(_finalize_sample(lane=lane, policy=policy, tokens=tokens, loss=loss, positions=positions,
                                        segments=segments, spans=spans, seq_len=seq_len, tokenizer=tokenizer,
                                        quality=sum(qualities)/len(qualities), difficulty=max(diffs), indic_tier=tier))
    return samples


def greedy_code_samples(records: list[dict], seq_len: int, tokenizer: FrozenByteTokenizer, lane: str = "code") -> list[PackedSample]:
    """Fast greedy packing for code; file/chunk boundaries get separate attention segments."""
    items=[]
    for r in sorted(records, key=lambda x: x["record_id"]):
        EvaluationFirewall.assert_record_trainable(r)
        for chunk,start,end in _record_chunks(r, seq_len-2):
            items.append((r,chunk,start,end))
    bins=[]; current=[]; used=0
    for item in items:
        n=len(item[1])+2
        if current and used+n>seq_len:
            bins.append(current); current=[]; used=0
        current.append(item); used+=n
    if current: bins.append(current)
    samples=[]
    for group in bins:
        tokens=[];loss=[];positions=[];segments=[];spans=[];qs=[];ds=[]
        for seg_id,(r,chunk,start,end) in enumerate(group):
            st=[tokenizer.BOS_ID]+chunk+[tokenizer.EOS_ID]
            tokens+=st; loss += [0]+[1]*len(chunk)+[1]; positions+=list(range(len(st))); segments += [seg_id]*len(st)
            spans.append({"shard_id":r["shard_id"],"record_id":r["record_id"],"doc_id":r["doc_id"],"start_token":start,"end_token":end,"segment_id":seg_id,"content_hash":r["raw_document_hash"]})
            qs.append(r["quality"]); ds.append(r["difficulty"])
        samples.append(_finalize_sample(lane=lane, policy="greedy_code_block_isolated", tokens=tokens, loss=loss,
                                        positions=positions, segments=segments, spans=spans, seq_len=seq_len,
                                        tokenizer=tokenizer, quality=sum(qs)/len(qs), difficulty=max(ds), indic_tier=None))
    return samples


def agentic_samples(records: list[dict], seq_len: int, tokenizer: FrozenByteTokenizer, lane: str = "agentic") -> list[PackedSample]:
    """One trace per sample; loss only on assistant/tool-call generations."""
    out=[]
    for r in sorted(records, key=lambda x:x["record_id"]):
        EvaluationFirewall.assert_record_trainable(r)
        tokens=[tokenizer.BOS_ID]; loss=[0]; positions=[0]; segments=[0]; spans=[]; pos=1
        source_offset=0
        for seg in r["segments"]:
            room = seq_len - len(tokens) - 1
            if room <= 0: break
            seg_ids=seg["token_ids"][:room]
            is_loss=1 if seg["role"] in ASSISTANT_ROLES else 0
            start=len(tokens)
            tokens += seg_ids + [tokenizer.EOS_ID]
            loss += [is_loss]*len(seg_ids) + [is_loss]
            positions += list(range(pos, pos+len(seg_ids)+1)); pos += len(seg_ids)+1
            segments += [0]*(len(seg_ids)+1)
            spans.append({"shard_id":r["shard_id"],"record_id":r["record_id"],"doc_id":r["doc_id"],"role":seg["role"],"start_token":source_offset,"end_token":source_offset+len(seg_ids),"packed_start":start,"packed_end":len(tokens),"content_hash":seg["content_hash"]})
            source_offset += len(seg["token_ids"])
        out.append(_finalize_sample(lane=lane, policy="structure_preserving_agentic", tokens=tokens, loss=loss,
                                    positions=positions, segments=segments, spans=spans, seq_len=seq_len,
                                    tokenizer=tokenizer, quality=r["quality"], difficulty=r["difficulty"], indic_tier=None))
    return out


def long_context_samples(records: list[dict], seq_len: int, tokenizer: FrozenByteTokenizer, lane: str = "long") -> list[PackedSample]:
    out=[]
    for r in sorted(records, key=lambda x:x["record_id"]):
        EvaluationFirewall.assert_record_trainable(r)
        for chunk,start,end in _record_chunks(r, seq_len-2):
            st=[tokenizer.BOS_ID]+chunk+[tokenizer.EOS_ID]
            spans=[{"shard_id":r["shard_id"],"record_id":r["record_id"],"doc_id":r["doc_id"],"start_token":start,"end_token":end,"segment_id":0,"content_hash":r["raw_document_hash"]}]
            out.append(_finalize_sample(lane=lane, policy="long_context_single_document", tokens=st,
                                        loss=[0]+[1]*len(chunk)+[1], positions=list(range(len(st))),
                                        segments=[0]*len(st), spans=spans, seq_len=seq_len, tokenizer=tokenizer,
                                        quality=r["quality"], difficulty=r["difficulty"], indic_tier=r.get("indic_tier")))
    return out


def build_sample_catalog(records_by_lane: dict[str,list[dict]], seq_len: int, tokenizer: FrozenByteTokenizer) -> tuple[list[PackedSample], dict[str,PackedSample]]:
    samples=[]
    for lane, records in sorted(records_by_lane.items()):
        if lane == "agentic":
            lane_samples=agentic_samples(records,seq_len,tokenizer,lane)
        elif lane == "code":
            lane_samples=greedy_code_samples(records,seq_len,tokenizer,lane)
        elif lane == "long":
            lane_samples=long_context_samples(records,seq_len,tokenizer,lane)
        else:
            lane_samples=best_fit_samples(records,seq_len,tokenizer,lane)
        samples.extend(lane_samples)
    by_id={s.sample_id:s for s in samples}
    if len(by_id)!=len(samples):
        raise RuntimeError("sample hash collision/duplicate catalog entry")
    return samples,by_id


def validate_sample(sample: PackedSample, tokenizer: FrozenByteTokenizer) -> list[str]:
    errors=[]; L=len(sample.input_ids)
    if not (len(sample.loss_mask)==len(sample.position_ids)==len(sample.segment_ids)==len(sample.attention_mask)==L):
        errors.append("shape_mismatch")
    for i,row in enumerate(sample.attention_mask):
        if len(row)!=L: errors.append("attention_shape")
        for j,v in enumerate(row):
            if v and j>i: errors.append("future_attention")
            if v and sample.input_ids[i]!=tokenizer.PAD_ID and sample.segment_ids[i]!=sample.segment_ids[j]: errors.append("cross_segment_attention")
    for i,tid in enumerate(sample.input_ids):
        if tid==tokenizer.PAD_ID and sample.loss_mask[i]!=0: errors.append("pad_has_loss")
    return sorted(set(errors))
