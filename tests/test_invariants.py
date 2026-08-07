from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import torch

from v5exec.batches import build_batch
from v5exec.checkpoint import CheckpointManager
from v5exec.corpus import load_documents
from v5exec.firewall import EvaluationFirewall, EvaluationFirewallError
from v5exec.io import read_json, read_jsonl
from v5exec.mixture import compile_schedule
from v5exec.model import initialize_model
from v5exec.opus import compute_gradient_proxy_scores
from v5exec.packing import build_sample_catalog, validate_sample
from v5exec.shards import build_shards, load_records_for_manifest, validate_manifest
from v5exec.tokenizer import FrozenByteTokenizer


ROOT = Path(os.environ.get("V5_REPO_ROOT", Path(__file__).resolve().parents[1]))


def build_fixture(temp_root: Path):
    tok = FrozenByteTokenizer()
    docs = load_documents(ROOT / "data/documents.jsonl")
    manifests = build_shards(docs, tok, temp_root / "shards", temp_root / "manifests")
    records = defaultdict(list)
    for m in manifests:
        if m["split"] == "train":
            EvaluationFirewall.assert_manifest_trainable(m)
            records[m["lane"]].extend(load_records_for_manifest(m, temp_root / "shards"))
    samples, catalog = build_sample_catalog(records, 64, tok)
    return tok, manifests, samples, catalog


class TestCoreInvariants(unittest.TestCase):
    def test_frozen_tokenizer_roundtrip_and_hash(self):
        tok1 = FrozenByteTokenizer(); tok2 = FrozenByteTokenizer()
        texts = ["hello", "भारत", "తెలుగు", "def f(x): return x+1"]
        self.assertTrue(tok1.verify_roundtrip(texts))
        self.assertEqual(tok1.tokenizer_hash, tok2.tokenizer_hash)
        self.assertEqual(tok1.VOCAB_SIZE, 260)

    def test_immutable_shards_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tok, manifests, _, _ = build_fixture(root)
            train = next(m for m in manifests if m["split"] == "train")
            ok, errors = validate_manifest(train, root / "shards", tok.tokenizer_hash)
            self.assertTrue(ok, errors)
            shard = root / "shards" / train["shard_path"]
            shard.chmod(0o644)
            with shard.open("ab") as f:
                f.write(b"{}\n")
            ok2, errors2 = validate_manifest(train, root / "shards", tok.tokenizer_hash)
            self.assertFalse(ok2)
            self.assertIn("content_hash_mismatch", errors2)

    def test_evaluation_firewall_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tok, manifests, _, _ = build_fixture(Path(td))
            for split in ("eval", "validation"):
                blocked_manifest = next(m for m in manifests if m["split"] == split)
                with self.assertRaises(EvaluationFirewallError):
                    EvaluationFirewall.assert_manifest_trainable(blocked_manifest)

    def test_packing_masks_positions_and_agentic_loss(self):
        with tempfile.TemporaryDirectory() as td:
            tok, _, samples, _ = build_fixture(Path(td))
            self.assertEqual({s.policy for s in samples}, {
                "best_fit_block_isolated", "greedy_code_block_isolated",
                "structure_preserving_agentic", "long_context_single_document"
            })
            for sample in samples:
                self.assertEqual(validate_sample(sample, tok), [])
            agent = next(s for s in samples if s.policy == "structure_preserving_agentic")
            self.assertGreater(sum(agent.loss_mask), 0)
            self.assertGreater(sum(1 for x in agent.loss_mask if x == 0), 0)
            # Causal: no query may attend to a future key.
            self.assertTrue(all(not agent.attention_mask[i][j] for i in range(64) for j in range(i + 1, 64)))
            # For an ordinary packed sample, positions reset at document/chunk boundaries.
            ordinary = next(s for s in samples if s.policy == "best_fit_block_isolated" and len(s.source_spans) > 1)
            starts = []
            last_seg = None
            for i, seg in enumerate(ordinary.segment_ids):
                if seg >= 0 and seg != last_seg:
                    starts.append(ordinary.position_ids[i])
                    last_seg = seg
            self.assertTrue(all(x == 0 for x in starts))

    def test_schedule_determinism_floors_and_opus_classes(self):
        with tempfile.TemporaryDirectory() as td:
            tok, _, samples, _ = build_fixture(Path(td))
            policy = read_json(ROOT / "configs/mixture_policy.json")
            proxy = compute_gradient_proxy_scores(samples, tok.VOCAB_SIZE, 64, seed=20260802)
            p1, d1, proofs1 = compile_schedule(policy, samples, slots_per_stage=32, batch_size=4, seed=1667, proxy_alignment=proxy)
            p2, d2, proofs2 = compile_schedule(policy, samples, slots_per_stage=32, batch_size=4, seed=1667, proxy_alignment=proxy)
            self.assertEqual(p1["plan_sha256"], p2["plan_sha256"])
            self.assertEqual([x["batch_id"] for x in p1["batches"]], [x["batch_id"] for x in p2["batches"]])
            self.assertTrue(all(x["floor_compliant"] for x in proofs1))
            statuses = {x["final_decision"] for x in d1}
            self.assertTrue({"accepted", "deferred", "rejected", "floor_override"}.issubset(statuses))


    def test_opus_gradient_proxy_is_real_bounded_and_hashed(self):
        with tempfile.TemporaryDirectory() as td:
            tok, _, samples, _ = build_fixture(Path(td))
            # A small cross-lane subset keeps the unit test fast while executing real autograd.
            subset = []
            seen = set()
            for sample in samples:
                if sample.lane not in seen or len(subset) < 2:
                    subset.append(sample); seen.add(sample.lane)
                if len(seen) >= 4 and len(subset) >= 6:
                    break
            proxy = compute_gradient_proxy_scores(subset, tok.VOCAB_SIZE, 64, seed=20260802)
            self.assertTrue(proxy["probe_model_hash"])
            self.assertTrue(proxy["scores_sha256"])
            self.assertEqual(set(proxy["scores"]), {x.sample_id for x in subset})
            self.assertTrue(all(-1.000001 <= row["gradient_alignment"] <= 1.000001 for row in proxy["scores"].values()))
            self.assertTrue(all(row["gold_proxy_sample_id"] for row in proxy["scores"].values()))

    def test_batch_hash_reconstruction(self):
        with tempfile.TemporaryDirectory() as td:
            tok, _, samples, catalog = build_fixture(Path(td))
            policy = read_json(ROOT / "configs/mixture_policy.json")
            proxy = compute_gradient_proxy_scores(samples, tok.VOCAB_SIZE, 64, seed=20260802)
            plan, _, _ = compile_schedule(policy, samples, slots_per_stage=32, batch_size=4, seed=1667, proxy_alignment=proxy)
            b1 = build_batch(plan["batches"][3], catalog, tok.PAD_ID)
            b2 = build_batch(plan["batches"][3], catalog, tok.PAD_ID)
            self.assertEqual(b1.batch_id, b2.batch_id)
            self.assertEqual(b1.batch_hash, b2.batch_hash)
            self.assertEqual(b1.source_spans, b2.source_spans)
            self.assertEqual(b1.input_ids.tolist(), b2.input_ids.tolist())

    def test_checkpoint_binds_cursor_and_ledger_offsets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tok = FrozenByteTokenizer()
            model = initialize_model(tok.VOCAB_SIZE, 64, seed=7)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            mgr = CheckpointManager(root / "checkpoints")
            offsets = {"consumption": 8, "learning": 8, "token_learning": 1234, "opus": 99}
            meta = mgr.save(model=model, optimizer=opt, step=8, next_batch_index=8,
                            ledger_offsets=offsets, plan_hash="plan123", tokenizer_hash=tok.tokenizer_hash)
            model2 = initialize_model(tok.VOCAB_SIZE, 64, seed=99)
            opt2 = torch.optim.AdamW(model2.parameters(), lr=9e-3)
            payload = mgr.load(meta, model2, opt2)
            self.assertEqual(payload["next_batch_index"], 8)
            self.assertEqual(payload["ledger_offsets"], offsets)
            self.assertEqual(payload["tokenizer_hash"], tok.tokenizer_hash)


class TestGeneratedEvidence(unittest.TestCase):
    def test_generated_demo_artifacts_when_present(self):
        artifact = ROOT / "submission_artifacts"
        if not (artifact / "evidence.json").exists():
            self.skipTest("run python run_demo.py to generate integration evidence")
        evidence = read_json(artifact / "evidence.json")
        self.assertEqual(evidence["overall_result"], "PASS")
        plan = read_json(artifact / "manifests/mixture_plan.json")
        consumed = read_jsonl(artifact / "ledgers/consumption.jsonl")
        self.assertEqual([x["batch_id"] for x in consumed], [x["batch_id"] for x in plan["batches"]])
        resume = read_json(artifact / "resume_report.json")
        replay = read_json(artifact / "replay_report.json")
        self.assertTrue(resume["matched"] and resume["catalog_hash_match"])
        self.assertTrue(replay["match"] and replay["catalog_hash_match"])
        self.assertEqual(read_json(artifact / "fork_report.json")["pass"], True)
        packing = read_json(artifact / "packing_report.json")
        self.assertTrue(packing["all_samples_valid"] and packing["policy_count"] == 4)
        proxy = read_json(artifact / "manifests/opus_proxy_manifest.json")
        opus = read_jsonl(artifact / "ledgers/opus_decisions.jsonl")
        self.assertTrue(proxy["probe_model_hash"] and proxy["scores_sha256"])
        self.assertTrue(all("gradient_alignment" in row and row["gold_proxy_sample_id"] for row in opus))


if __name__ == "__main__":
    unittest.main()
