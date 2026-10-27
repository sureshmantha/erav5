#!/usr/bin/env python3
"""ERA V5 G7 CPU checkpoint reference. Trusted local files; NOT distributed DCP.

Boundary: optimizer.step() + scheduler.step() finished; gradients zeroed. CPU exact replay
for the explicitly deterministic fixture. Production DCP/S3 fencing lives in design doc.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import tempfile
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

FORMAT = "era-v5-g7-cpu/1"

class CheckpointError(RuntimeError):
    pass

def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()

def canonical_json(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

def durable_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())

def fsync_directory(path: Path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

def rng_capture() -> dict:
    npst = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": npst[0], "keys": npst[1].tolist(), "pos": int(npst[2]),
                  "has_gauss": int(npst[3]), "cached_gaussian": float(npst[4])},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }

def rng_restore(d: dict) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(d) != required:
        raise CheckpointError(f"RNG schema mismatch; expected {required}, got {set(d)}")
    random.setstate(d["python"])
    n = d["numpy"]
    np.random.set_state((n["algorithm"], np.array(n["keys"], dtype=np.uint32),
                         int(n["pos"]), int(n["has_gauss"]), float(n["cached_gaussian"])))
    torch.set_rng_state(d["torch_cpu"])
    if d["torch_cuda"]:
        if not torch.cuda.is_available() or len(d["torch_cuda"]) != torch.cuda.device_count():
            raise CheckpointError("CUDA RNG device topology does not match")
        torch.cuda.set_rng_state_all(d["torch_cuda"])

def strict_compatible(expected: dict, incoming: dict, label: str) -> None:
    if set(expected) != set(incoming):
        raise CheckpointError(f"{label} key mismatch missing={sorted(set(expected)-set(incoming))} extra={sorted(set(incoming)-set(expected))}")
    for key, tensor in expected.items():
        other = incoming[key]
        if isinstance(tensor, torch.Tensor) and (not isinstance(other, torch.Tensor) or tensor.shape != other.shape or tensor.dtype != other.dtype):
            raise CheckpointError(f"{label}.{key} tensor shape/dtype mismatch")

class CheckpointManager:
    """Local single-writer transactional snapshot. Caller supplies ordered CPU state.

    Fail closed by default. No network/S3 guarantees. API maintains immutable versions
    and an atomically replaced latest pointer. Not safe for multiple simultaneous writers.
    """
    def __init__(self, checkpoint_dir: str | Path):
        self.root = Path(checkpoint_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler.LRScheduler, step: int,
             data_cursor: int, config_sha: str, *, failpoint: str = "") -> Path:
        if not isinstance(step, int) or step < 0 or data_cursor < 0:
            raise CheckpointError("Invalid step/data cursor")
        name = f"step_{step:08d}_{uuid.uuid4().hex}"
        staging = self.root / (name + ".tmp")
        final = self.root / name
        staging.mkdir(mode=0o700)
        try:
            state = {
                "schema": FORMAT, "step": step, "data_cursor": int(data_cursor),
                "config_sha": config_sha, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": rng_capture(),
            }
            with (staging / "state.pt").open("xb") as fh:
                torch.save(state, fh)
                fh.flush(); os.fsync(fh.fileno())
            manifest = {"schema": FORMAT, "step": step, "data_cursor": data_cursor,
                        "config_sha": config_sha, "payload": "state.pt",
                        "sha256": sha256_file(staging / "state.pt"),
                        "bytes": (staging / "state.pt").stat().st_size,
                        "created_utc": datetime.now(timezone.utc).isoformat()}
            durable_write(staging / "MANIFEST.json", canonical_json(manifest))
            fsync_directory(staging)
            if failpoint == "before_seal":
                raise CheckpointError("INJECTED_FAILURE_BEFORE_SEAL")
            os.replace(staging, final)
            fsync_directory(self.root)
            # Do not move latest backward, even in a manual recovery branch.
            previous = self.pointer_or_none()
            if previous is not None:
                prior = self._read_manifest(self.root / previous)
                if int(prior["step"]) > step:
                    raise CheckpointError("Refusing to move latest to an earlier step")
            pointer_tmp = self.root / (".latest." + uuid.uuid4().hex)
            durable_write(pointer_tmp, canonical_json({"generation": name, "step": step,
                                                         "manifest_sha256": sha256_file(final / "MANIFEST.json")}))
            os.replace(pointer_tmp, self.root / "latest.json")
            fsync_directory(self.root)
            return final
        finally:
            # Retain interrupted temp folder for forensics; never use it for restore.
            pass

    def pointer_or_none(self) -> str | None:
        ptr = self.root / "latest.json"
        if not ptr.exists():
            return None
        data = json.loads(ptr.read_text())
        name = data["generation"]
        if Path(name).name != name or "/" in name or "\\" in name or name.endswith(".tmp"):
            raise CheckpointError("Unsafe generation pointer")
        if sha256_file(self.root / name / "MANIFEST.json") != data["manifest_sha256"]:
            raise CheckpointError("Latest manifest hash mismatch")
        return name

    def _read_manifest(self, gen: Path) -> dict:
        manifest = json.loads((gen / "MANIFEST.json").read_text())
        if manifest["schema"] != FORMAT or manifest["payload"] != "state.pt":
            raise CheckpointError("Unknown checkpoint manifest format")
        payload = gen / "state.pt"
        if sha256_file(payload) != manifest["sha256"] or payload.stat().st_size != manifest["bytes"]:
            raise CheckpointError("Checkpoint payload corrupt or truncated")
        return manifest

    def load(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler.LRScheduler,
             config_sha: str, generation: str | None = None) -> dict:
        name = generation or self.pointer_or_none()
        if name is None:
            raise CheckpointError("No committed checkpoint")
        if Path(name).name != name or name.endswith(".tmp"):
            raise CheckpointError("Invalid generation name")
        gen = self.root / name
        manifest = self._read_manifest(gen)
        if manifest["config_sha"] != config_sha:
            raise CheckpointError("Config fingerprint mismatch")
        state = torch.load(gen / "state.pt", map_location="cpu", weights_only=True)
        required = {"schema", "step", "data_cursor", "config_sha", "model", "optimizer", "scheduler", "rng"}
        if set(state) != required or state["schema"] != FORMAT:
            raise CheckpointError("Checkpoint state schema mismatch")
        if any(state[k] != manifest[k] for k in ("step", "data_cursor", "config_sha")):
            raise CheckpointError("Manifest/state identity mismatch")
        strict_compatible(model.state_dict(), state["model"], "model")
        if set(state["optimizer"]) != set(optimizer.state_dict()):
            raise CheckpointError("Optimizer state schema mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng_restore(state["rng"])
        return {"step": state["step"], "data_cursor": state["data_cursor"], "generation": name}

# Known deterministic CPU reference: every random source and batch cursor are persisted.
CFG_SHA = hashlib.sha256(b"g7-cpu-reference-v1:features=8:classes=4:dropout=0.25:adamw").hexdigest()

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Dropout(p=0.25), nn.Linear(16, 4))
    def forward(self, x):
        return self.net(x)

def make_objects():
    model = TinyNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    return model, optimizer, scheduler

def prepare_determinism(seed=104729):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def check_numpy_bridge() -> None:
    """Fail early with an actionable message instead of four opaque subprocess errors."""
    try:
        probe = torch.from_numpy(np.array([1.0], dtype=np.float32))
        _ = probe.numpy()
    except Exception as exc:
        raise CheckpointError(
            f"PyTorch/NumPy bridge unavailable (torch={torch.__version__}, numpy={np.__version__}): {exc}. "
            "Use scripts/setup_local_venv.sh or install numpy<2 in an isolated Python 3.11 environment."
        ) from exc


def step_once(model, optimizer, scheduler, cursor: int) -> dict:
    # Combine all three independent seeded RNG sources for the acceptance fixture.
    batch_id = random.randrange(0, 10**7)
    a = np.random.standard_normal((5, 8)).astype("float32")
    x = torch.from_numpy(a) + torch.randn(5, 8) * 0.02
    y = torch.tensor([(batch_id + cursor + j) % 4 for j in range(5)], dtype=torch.long)
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    if not bool(torch.isfinite(loss)):
        raise CheckpointError("Nonfinite loss: refusing to advance")
    loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    return {"step": cursor + 1, "batch_id": batch_id, "loss_hex": struct.pack("!d", float(loss.item())).hex(),
            "lr_hex": struct.pack("!d", float(scheduler.get_last_lr()[0])).hex()}

def train_cli(args):
    check_numpy_bridge()
    prepare_determinism()
    model, opt, sched = make_objects()
    manager = CheckpointManager(args.directory)
    start = 0
    if args.resume:
        previous = manager.load(model, opt, sched, CFG_SHA)
        start = previous["step"]
        if previous["data_cursor"] != start:
            raise CheckpointError("Data cursor does not align with checkpoint optimizer step")
    records = [step_once(model, opt, sched, n) for n in range(start, args.steps)]
    if args.save:
        manager.save(model, opt, sched, args.steps, args.steps, CFG_SHA)
    # Serialize model/optimizer final state in stable ordered tensor form for comparison.
    digest = hashlib.sha256()
    for label, record in (("model", model.state_dict()), ("optimizer", opt.state_dict())):
        def walk(obj):
            if isinstance(obj, torch.Tensor):
                digest.update(str(obj.dtype).encode()); digest.update(str(tuple(obj.shape)).encode())
                digest.update(obj.detach().cpu().contiguous().numpy().tobytes()); return
            if isinstance(obj, dict):
                for k in sorted(obj, key=str): digest.update(repr(k).encode()); walk(obj[k])
            elif isinstance(obj, (list, tuple)):
                for v in obj: walk(v)
            else: digest.update(repr(obj).encode())
        digest.update(label.encode()); walk(record)
    result = {"records": records, "step": args.steps, "state_digest": digest.hexdigest(), "rng_digest":
              hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
    print(json.dumps(result, sort_keys=True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd", required=True)
    t=sub.add_parser("train", help="Deterministic CPU acceptance fixture")
    t.add_argument("--directory", required=True); t.add_argument("--steps", type=int, required=True)
    t.add_argument("--resume", action="store_true"); t.add_argument("--save", action="store_true")
    ins=sub.add_parser("inspect", help="Validate latest committed checkpoint")
    ins.add_argument("--directory", required=True)
    args=p.parse_args()
    if args.cmd=="train": train_cli(args)
    else:
        cm=CheckpointManager(args.directory); name=cm.pointer_or_none()
        if not name: raise CheckpointError("No latest checkpoint")
        print(json.dumps({"generation":name,"manifest":cm._read_manifest(cm.root/name)},indent=2))
if __name__ == "__main__":
    try: main()
    except (CheckpointError, OSError, ValueError) as exc:
        print(f"CHECKPOINT_ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
