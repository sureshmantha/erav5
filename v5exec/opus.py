from __future__ import annotations

import hashlib
from collections import defaultdict

import torch
import torch.nn.functional as F

from .model import compute_next_token_losses, initialize_model
from .packing import PackedSample
from .hashing import sha256_json


def _model_tensor_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _head_gradient_signature(model: torch.nn.Module, sample: PackedSample) -> torch.Tensor:
    input_ids = torch.tensor([sample.input_ids], dtype=torch.long)
    loss_mask = torch.tensor([sample.loss_mask], dtype=torch.float32)
    attention = torch.tensor([sample.attention_mask], dtype=torch.bool)
    positions = torch.tensor([sample.position_ids], dtype=torch.long)
    model.zero_grad(set_to_none=True)
    loss, _, _, _ = compute_next_token_losses(model, input_ids, attention, positions, loss_mask)
    grad = torch.autograd.grad(loss, model.head.weight, retain_graph=False, create_graph=False)[0]
    return grad.detach().reshape(-1).cpu()


def compute_gradient_proxy_scores(samples: list[PackedSample], vocab_size: int, seq_len: int, *, seed: int = 20260802) -> dict:
    """Compute a real, cheap gradient-alignment proxy for OPUS-style selection.

    The highest-quality sample in each capability lane is the lane's golden proxy.
    Candidate alignment is cosine similarity between its LM-head gradient and the
    golden-proxy gradient under a frozen tiny probe model. This is intentionally a
    small demonstrator of the selector contract, not a claim to reproduce the
    course's production/private implementation.
    """
    torch.manual_seed(seed)
    probe = initialize_model(vocab_size, seq_len, seed=seed)
    probe.eval()
    by_lane = defaultdict(list)
    for sample in samples:
        by_lane[sample.lane].append(sample)
    golden = {
        lane: sorted(pool, key=lambda s: (-s.quality, s.difficulty, s.sample_id))[0]
        for lane, pool in by_lane.items()
    }
    golden_grad = {lane: _head_gradient_signature(probe, sample) for lane, sample in golden.items()}
    scores = {}
    for sample in samples:
        grad = _head_gradient_signature(probe, sample)
        ref = golden_grad[sample.lane]
        cosine = float(F.cosine_similarity(grad.unsqueeze(0), ref.unsqueeze(0), dim=1, eps=1e-12).item())
        scores[sample.sample_id] = {
            "gradient_alignment": round(cosine, 8),
            "gold_proxy_sample_id": golden[sample.lane].sample_id,
            "lane": sample.lane,
        }
    return {
        "schema_version": 1,
        "method": "LM-head gradient cosine against highest-quality lane proxy",
        "probe_seed": seed,
        "probe_model_hash": _model_tensor_hash(probe),
        "golden_samples": {lane: sample.sample_id for lane, sample in golden.items()},
        "scores": scores,
        "scores_sha256": sha256_json(scores),
    }
