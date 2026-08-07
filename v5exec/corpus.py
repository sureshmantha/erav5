from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable

from .hashing import sha256_json


def normalize_text(text: str) -> str:
    # Brahmic-safe for this demo: preserve code points except canonical NFC and line endings.
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def load_documents(path: str | Path) -> list[dict]:
    docs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        if "text" in doc:
            doc["text"] = normalize_text(doc["text"])
        if "segments" in doc:
            for seg in doc["segments"]:
                seg["content"] = normalize_text(seg["content"])
        doc["raw_document_hash"] = sha256_json({k: v for k, v in doc.items() if k != "raw_document_hash"})
        docs.append(doc)
    return docs


def cleaning_pipeline_hash() -> str:
    return sha256_json({
        "name": "v5-demo-cleaning-v1",
        "normalization": "NFC + CRLF/CR to LF",
        "pii": "fixture corpus contains no PII",
        "dedup": "doc_id uniqueness + content hash uniqueness",
        "contamination": "explicit split tags + training firewall",
    })
