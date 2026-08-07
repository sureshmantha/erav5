from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .hashing import canonical_json_bytes


def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
    os.replace(tmp, p)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Any], *, immutable: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as f:
        for row in rows:
            f.write(canonical_json_bytes(row) + b"\n")
    os.replace(tmp, p)
    if immutable:
        p.chmod(0o444)


def append_jsonl(path: str | Path, row: Any) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("ab") as f:
        f.write(canonical_json_bytes(row) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    return count_jsonl(p)



def append_many_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("ab") as f:
        for row in rows:
            f.write(canonical_json_bytes(row) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    return count_jsonl(p)


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_jsonl(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("rb") as f:
        return sum(1 for line in f if line.strip())
