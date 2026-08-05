#!/usr/bin/env python3
"""Seal the preregistered 124M attention c-proj capacity decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "examples/nanogpt/configs/selection_artifacts"
PLAN = ARTIFACTS / "124m_attention_vo_factorial_cproj_capacity_plan.json"
PARENT = ARTIFACTS / "124m_attention_qk_cproj_only_partial_control_result.json"
CANDIDATE_CONFIG = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qk_cproj_only_qk64_outputgain_"
    "cprojratio10_5tpp_lr24e4.json"
)
OUTPUT = ARTIFACTS / "124m_attention_cproj_ratio10_promotion_result.json"

TERMINAL_MINIMUM_IMPROVEMENT_CE = 0.02
EARLY_MAXIMUM_WORSENING_CE = 0.01
EARLY_MAXIMUM_WORSENED_EVALUATIONS = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _curve(result: dict[str, Any], steps: list[int]) -> list[float]:
    rows = result.get("run", {}).get("fixed_evaluations")
    if not isinstance(rows, list):
        raise ValueError("result lacks run.fixed_evaluations")
    by_step: dict[int, float] = {}
    for row in rows:
        value = row.get("validation_ce")
        if value is not None:
            by_step[int(row["step"])] = float(value)
    if set(by_step) != set(steps):
        raise ValueError(f"fixed evaluation steps differ: {sorted(by_step)} != {steps}")
    curve = [by_step[step] for step in steps]
    if not all(math.isfinite(value) for value in curve):
        raise ValueError("fixed evaluation curve contains a non-finite value")
    return curve


def _verify_clean_identity(
    result: dict[str, Any],
    *,
    config_sha256: str,
    dataset_manifest_sha256: str,
    fixed_eval_indices_sha256: str,
    label: str,
) -> None:
    run = result.get("run", {})
    expected = {
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "fixed_eval_indices_sha256": fixed_eval_indices_sha256,
    }
    for field, value in expected.items():
        if run.get(field) != value:
            raise ValueError(f"{label}.run.{field} does not match registered identity")
    if run.get("exit_code") != 0 or run.get("classification") != "clean":
        raise ValueError(f"{label} is not a clean terminal run")


def analyze(
    plan: dict[str, Any],
    parent_result: dict[str, Any],
    candidate_result: dict[str, Any],
) -> dict[str, Any]:
    """Apply the immutable terminal-improvement and early-regression rule."""
    identity = plan["identity"]
    steps = [int(step) for step in plan["fixed_curves"]["steps"]]
    _verify_clean_identity(
        parent_result,
        config_sha256=identity["active_parent_config_sha256"],
        dataset_manifest_sha256=identity["dataset_manifest_sha256"],
        fixed_eval_indices_sha256=identity["fixed_eval_indices_sha256"],
        label="parent",
    )
    candidate_config_sha256 = _sha256(CANDIDATE_CONFIG)
    _verify_clean_identity(
        candidate_result,
        config_sha256=candidate_config_sha256,
        dataset_manifest_sha256=identity["dataset_manifest_sha256"],
        fixed_eval_indices_sha256=identity["fixed_eval_indices_sha256"],
        label="candidate",
    )

    parent = _curve(parent_result, steps)
    candidate = _curve(candidate_result, steps)
    qk_only = [float(value) for value in plan["fixed_curves"]["qk_only_L00"]]
    improvement = [base - test for base, test in zip(parent, candidate)]
    delta_to_qk_only = [test - control for test, control in zip(candidate, qk_only)]
    recoverable_gap = [base - control for base, control in zip(parent, qk_only)]
    recovery_fraction = [
        gain / gap if gap > 0 else float("nan")
        for gain, gap in zip(improvement, recoverable_gap)
    ]

    early_worsened = [
        step
        for step, gain in zip(steps[:-1], improvement[:-1])
        if gain < -EARLY_MAXIMUM_WORSENING_CE
    ]
    terminal_gate = improvement[-1] >= TERMINAL_MINIMUM_IMPROVEMENT_CE
    early_gate = len(early_worsened) <= EARLY_MAXIMUM_WORSENED_EVALUATIONS
    promote = terminal_gate and early_gate

    return {
        "schema_version": "mai_124m_attention_cproj_ratio10_promotion_result_v1",
        "source_plan": str(PLAN.relative_to(ROOT)),
        "source_plan_sha256": _sha256(PLAN),
        "parent_result": str(PARENT.relative_to(ROOT)),
        "parent_result_sha256": _sha256(PARENT),
        "candidate_config": str(CANDIDATE_CONFIG.relative_to(ROOT)),
        "candidate_config_sha256": candidate_config_sha256,
        "identity": identity,
        "steps": steps,
        "curves": {
            "qk_cproj_ratio01_parent": parent,
            "qk_cproj_ratio10_candidate": candidate,
            "qk_only": qk_only,
        },
        "comparisons": {
            "improvement_over_ratio01_ce": improvement,
            "delta_to_qk_only_ce": delta_to_qk_only,
            "recovery_fraction_of_parent_qk_only_gap": recovery_fraction,
        },
        "registered_gate": {
            "terminal_minimum_improvement_ce": TERMINAL_MINIMUM_IMPROVEMENT_CE,
            "early_maximum_worsening_ce": EARLY_MAXIMUM_WORSENING_CE,
            "early_maximum_worsened_evaluations": (
                EARLY_MAXIMUM_WORSENED_EVALUATIONS
            ),
        },
        "decision_evidence": {
            "terminal_improvement_ce": improvement[-1],
            "early_worsened_steps": early_worsened,
            "terminal_gate": terminal_gate,
            "early_gate": early_gate,
        },
        "decision": {
            "promote_cproj_ratio10_to_full_attention": promote,
            "classification": (
                "PROMOTE_CPROJ_RATIO10_TO_FULL_ATTENTION"
                if promote
                else "REJECT_CPROJ_RATIO10_TRANSFER"
            ),
            "next_action": (
                "Resolve one full QK+V+c_proj ratio-0.10 config, then require a fresh exact-config MFU gate before launch."
                if promote
                else "Do not sweep c-proj ratios; reject fixed-chart capacity as sufficient and return to a directional c-proj representation test."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = analyze(
        json.loads(PLAN.read_text()),
        json.loads(PARENT.read_text()),
        json.loads(args.candidate_result.read_text()),
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
