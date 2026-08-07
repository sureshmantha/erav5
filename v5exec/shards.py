from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .corpus import cleaning_pipeline_hash
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json
from .io import write_json, write_jsonl, read_jsonl
from .tokenizer import FrozenByteTokenizer


REQUIRED_MANIFEST_FIELDS = {
    "shard_id", "split", "lane", "kind", "tokenizer_hash", "content_sha256",
    "token_count", "record_count", "source_document_hashes", "cleaning_pipeline_hash",
    "license_set", "provenance_set", "gate_status", "manifest_sha256",
}


def _record_from_doc(doc: dict, tokenizer: FrozenByteTokenizer) -> dict:
    base = {
        "record_id": doc["doc_id"],
        "doc_id": doc["doc_id"],
        "split": doc["split"],
        "lane": doc["lane"],
        "kind": doc["kind"],
        "language": doc.get("language", "und"),
        "quality": float(doc.get("quality", 0.8)),
        "difficulty": int(doc.get("difficulty", 1)),
        "source": doc.get("source", "fixture"),
        "license": doc.get("license", "CC-BY-4.0"),
        "raw_document_hash": doc["raw_document_hash"],
        "indic_tier": doc.get("indic_tier"),
        "contamination_status": doc.get("contamination_status", "clean"),
    }
    if "segments" in doc:
        segments = []
        for idx, seg in enumerate(doc["segments"]):
            segments.append({
                "segment_index": idx,
                "role": seg["role"],
                "token_ids": tokenizer.encode(seg["content"]),
                "content_hash": sha256_json(seg),
            })
        base["segments"] = segments
        base["token_count"] = sum(len(s["token_ids"]) for s in segments)
    else:
        token_ids = tokenizer.encode(doc["text"])
        base["token_ids"] = token_ids
        base["token_count"] = len(token_ids)
    return base


def _immutable_shard_bytes(records: list[dict]) -> bytes:
    return b"".join(canonical_json_bytes(r) + b"\n" for r in records)


def build_shards(documents: list[dict], tokenizer: FrozenByteTokenizer, shard_dir: str | Path, manifest_dir: str | Path) -> list[dict]:
    shard_dir, manifest_dir = Path(shard_dir), Path(manifest_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for doc in documents:
        groups[(doc["split"], doc["lane"], doc["kind"])].append(doc)

    manifests = []
    for (split, lane, kind), docs in sorted(groups.items()):
        shard_id = f"{split}__{lane}__{kind}"
        records = [_record_from_doc(d, tokenizer) for d in sorted(docs, key=lambda x: x["doc_id"])]
        for record in records:
            record["shard_id"] = shard_id
        shard_path = shard_dir / f"{shard_id}.jsonl"
        expected_bytes = _immutable_shard_bytes(records)
        expected_hash = sha256_bytes(expected_bytes)
        write_jsonl(shard_path, records, immutable=False)
        actual_hash = sha256_file(shard_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"non-deterministic shard write: {shard_id}")
        # Prove a deterministic rebuild yields identical bytes before locking the file.
        rebuild_hash = sha256_bytes(_immutable_shard_bytes(records))
        if rebuild_hash != actual_hash:
            raise RuntimeError(f"immutable rebuild hash mismatch: {shard_id}")
        shard_path.chmod(0o444)

        manifest_core = {
            "schema_version": 1,
            "shard_id": shard_id,
            "shard_path": str(shard_path.name),
            "split": split,
            "lane": lane,
            "kind": kind,
            "tokenizer_hash": tokenizer.tokenizer_hash,
            "content_sha256": actual_hash,
            "rebuild_sha256": rebuild_hash,
            "token_count": sum(r["token_count"] for r in records),
            "record_count": len(records),
            "source_document_hashes": [r["raw_document_hash"] for r in records],
            "cleaning_pipeline_hash": cleaning_pipeline_hash(),
            "license_set": sorted({r["license"] for r in records}),
            "provenance_set": sorted({r["source"] for r in records}),
            "language_set": sorted({r["language"] for r in records}),
            "gate_status": "PASS" if all(r["contamination_status"] == "clean" for r in records) else "FAIL",
            "eval_overlap_status": "blocked" if split != "train" else "not_eval",
            "immutable": True,
        }
        manifest = dict(manifest_core)
        manifest["manifest_sha256"] = sha256_json(manifest_core)
        manifest_path=manifest_dir / f"{shard_id}.manifest.json"
        write_json(manifest_path, manifest)
        manifest_path.chmod(0o444)
        manifests.append(manifest)
    return manifests


def validate_manifest(manifest: dict, shard_dir: str | Path, tokenizer_hash: str) -> tuple[bool, list[str]]:
    errors = []
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        errors.append("missing:" + ",".join(sorted(missing)))
    if manifest.get("tokenizer_hash") != tokenizer_hash:
        errors.append("tokenizer_hash_mismatch")
    core = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if sha256_json(core) != manifest.get("manifest_sha256"):
        errors.append("manifest_hash_mismatch")
    shard_path = Path(shard_dir) / manifest.get("shard_path", "")
    if not shard_path.exists():
        errors.append("missing_shard")
    else:
        if sha256_file(shard_path) != manifest.get("content_sha256"):
            errors.append("content_hash_mismatch")
    if manifest.get("rebuild_sha256") != manifest.get("content_sha256"):
        errors.append("immutable_rebuild_mismatch")
    if manifest.get("gate_status") != "PASS":
        errors.append("gate_status_not_pass")
    return not errors, errors


def load_records_for_manifest(manifest: dict, shard_dir: str | Path) -> list[dict]:
    return read_jsonl(Path(shard_dir) / manifest["shard_path"])
