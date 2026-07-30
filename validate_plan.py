#!/usr/bin/env python3
"""Validate the Bharat V5-40B mixture and curriculum arithmetic.

No third-party packages are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Dict, Iterable


LANES = (
    "general",
    "indic",
    "non_indic",
    "code",
    "math",
    "science",
    "agentic",
    "reasoning",
    "long_context",
)


@dataclass(frozen=True)
class Stage:
    tokens_trillion: float
    mix: Dict[str, float]


STAGES: Dict[str, Stage] = {
    "S0_foundation": Stage(
        0.60,
        dict(zip(LANES, (50, 18, 4, 10, 6, 6, 1, 1, 4))),
    ),
    "S1_broad_competence": Stage(
        2.10,
        dict(zip(LANES, (38, 20, 3, 15, 7, 8, 2, 2, 5))),
    ),
    "S2_capability_build": Stage(
        1.80,
        dict(zip(LANES, (22, 20, 2, 21, 10, 10, 6, 5, 4))),
    ),
    "S3_specialization": Stage(
        1.20,
        dict(zip(LANES, (12, 20, 2, 23, 10, 8, 12, 9, 4))),
    ),
    "S4_capability_anneal": Stage(
        0.24,
        dict(zip(LANES, (10, 20, 1, 20, 10, 10, 12, 12, 5))),
    ),
    "S5_final_cooldown": Stage(
        0.06,
        dict(zip(LANES, (12, 20, 1, 18, 10, 10, 10, 14, 5))),
    ),
}


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-9) -> None:
    if not isclose(actual, expected, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def weighted_shares(stages: Iterable[Stage]) -> Dict[str, float]:
    stages = tuple(stages)
    total_tokens = sum(stage.tokens_trillion for stage in stages)
    exposure = {lane: 0.0 for lane in LANES}
    for stage in stages:
        for lane, pct in stage.mix.items():
            exposure[lane] += stage.tokens_trillion * pct / 100.0
    return {lane: value / total_tokens * 100.0 for lane, value in exposure.items()}


def validate() -> None:
    total_tokens = sum(stage.tokens_trillion for stage in STAGES.values())
    assert_close(total_tokens, 6.0, "total pretraining tokens (T)")

    for name, stage in STAGES.items():
        missing = set(LANES) - set(stage.mix)
        extra = set(stage.mix) - set(LANES)
        if missing or extra:
            raise AssertionError(f"{name}: lane mismatch; missing={missing}, extra={extra}")
        assert_close(sum(stage.mix.values()), 100.0, f"{name} mix sum")

    reserve = STAGES["S4_capability_anneal"].tokens_trillion + STAGES["S5_final_cooldown"].tokens_trillion
    assert_close(reserve, 0.30, "late-stage reserve (T)")
    assert_close(reserve / total_tokens * 100.0, 5.0, "late-stage reserve (%)")

    # Protected stage-level floors.
    for name, stage in STAGES.items():
        expected_indic_floor = 18.0 if name == "S0_foundation" else 20.0
        if stage.mix["indic"] < expected_indic_floor:
            raise AssertionError(f"{name}: Indic floor violated")
        if stage.mix["general"] < 10.0:
            raise AssertionError(f"{name}: general floor violated")
        if stage.mix["code"] < 10.0:
            raise AssertionError(f"{name}: code floor violated")
        if stage.mix["math"] + stage.mix["science"] < 10.0:
            raise AssertionError(f"{name}: math+science floor violated")
        if stage.mix["non_indic"] < 1.0:
            raise AssertionError(f"{name}: non-Indic floor violated")

    shares = weighted_shares(STAGES.values())
    assert_close(sum(shares.values()), 100.0, "weighted share sum", tol=1e-7)

    indic_tiers = {
        "verified_native": 35.0,
        "unverified_native": 25.0,
        "translated": 20.0,
        "synthetic": 20.0,
    }
    assert_close(sum(indic_tiers.values()), 100.0, "Indic tier sum")

    print("Bharat V5-40B plan validation: PASS")
    print(f"Total tokens: {total_tokens:.2f}T")
    print(f"Anneal + cooldown reserve: {reserve:.2f}T ({reserve / total_tokens * 100:.2f}%)")
    print("\nRealized weighted shares:")
    for lane in LANES:
        print(f"  {lane:14s} {shares[lane]:6.2f}%")


if __name__ == "__main__":
    validate()
