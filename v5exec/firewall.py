from __future__ import annotations


class EvaluationFirewallError(RuntimeError):
    pass


class EvaluationFirewall:
    """Two-sided firewall: manifests and records must both be explicitly training data."""

    @staticmethod
    def assert_manifest_trainable(manifest: dict) -> None:
        if manifest.get("split") != "train":
            raise EvaluationFirewallError(f"blocked non-train shard: {manifest.get('shard_id')} split={manifest.get('split')}")
        if manifest.get("gate_status") != "PASS":
            raise EvaluationFirewallError(f"blocked failed shard: {manifest.get('shard_id')}")
        if manifest.get("eval_overlap_status") != "not_eval":
            raise EvaluationFirewallError(f"blocked eval-overlap shard: {manifest.get('shard_id')}")

    @staticmethod
    def assert_record_trainable(record: dict) -> None:
        if record.get("split") != "train":
            raise EvaluationFirewallError(f"blocked non-train record: {record.get('record_id')}")
        if record.get("contamination_status") != "clean":
            raise EvaluationFirewallError(f"blocked contaminated record: {record.get('record_id')}")
