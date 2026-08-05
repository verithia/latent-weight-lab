#!/usr/bin/env python3
"""Seal the preregistered 124M attention V/c-proj factorial decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_vo_factorial_cproj_capacity_plan.json"
)
OUTPUT = (
    ROOT
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_vo_factorial_result.json"
)


def _curve(result: dict[str, Any], steps: list[int]) -> list[float]:
    rows = result.get("run", {}).get("fixed_evaluations")
    if rows is None:
        rows = result.get("loss", {}).get("fixed_evaluations")
    if not isinstance(rows, list):
        raise ValueError("result lacks fixed evaluations")
    by_step: dict[int, float] = {}
    for row in rows:
        value = row.get("validation_ce", row.get("validation"))
        if value is not None:
            by_step[int(row["step"])] = float(value)
    if set(by_step) != set(steps):
        raise ValueError(f"fixed evaluation steps differ: {sorted(by_step)} != {steps}")
    curve = [by_step[step] for step in steps]
    if not all(math.isfinite(value) for value in curve):
        raise ValueError("fixed evaluation curve contains a non-finite value")
    return curve


def _verify_identity(plan: dict[str, Any], result: dict[str, Any]) -> None:
    run = result.get("run", {})
    identity = plan["identity"]
    expected = {
        "config_sha256": identity["active_parent_config_sha256"],
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "fixed_eval_indices_sha256": identity["fixed_eval_indices_sha256"],
    }
    for field, value in expected.items():
        if run.get(field) != value:
            raise ValueError(f"run.{field} does not match the preregistered identity")
    if run.get("exit_code") != 0 or run.get("classification") != "clean":
        raise ValueError("factorial arm is not a clean terminal run")


def analyze(plan: dict[str, Any], l01_result: dict[str, Any]) -> dict[str, Any]:
    """Calculate main effects, interaction, and the preregistered decision."""
    _verify_identity(plan, l01_result)
    fixed = plan["fixed_curves"]
    steps = [int(step) for step in fixed["steps"]]
    l00 = [float(value) for value in fixed["qk_only_L00"]]
    l10 = [float(value) for value in fixed["qk_plus_v_L10"]]
    l01 = _curve(l01_result, steps)
    l11 = [float(value) for value in fixed["qk_plus_v_plus_cproj_L11"]]

    v_shapley = [
        0.5 * ((v10 - v00) + (v11 - v01))
        for v00, v10, v01, v11 in zip(l00, l10, l01, l11)
    ]
    cproj_shapley = [
        0.5 * ((v01 - v00) + (v11 - v10))
        for v00, v10, v01, v11 in zip(l00, l10, l01, l11)
    ]
    interaction = [
        v11 - v10 - v01 + v00
        for v00, v10, v01, v11 in zip(l00, l10, l01, l11)
    ]
    residual = [
        v_effect + cproj_effect - (v11 - v00)
        for v00, v11, v_effect, cproj_effect in zip(
            l00, l11, v_shapley, cproj_shapley
        )
    ]
    if max(abs(value) for value in residual) > 1e-12:
        raise AssertionError("factorial Shapley identity failed")

    cproj_support = sum(value > 0 for value in cproj_shapley)
    interaction_support = sum(value > 0 for value in interaction)
    cproj_gate = cproj_shapley[-1] >= 0.05 and cproj_support >= 3
    interaction_gate = interaction[-1] >= 0.03 and interaction_support >= 3
    authorize = cproj_gate or interaction_gate

    return {
        "schema_version": "mai_124m_attention_vo_factorial_result_v1",
        "source_plan": str(PLAN.relative_to(ROOT)),
        "identity": plan["identity"],
        "steps": steps,
        "curves": {"L00": l00, "L10": l10, "L01": l01, "L11": l11},
        "effects": {
            "v_shapley_ce": v_shapley,
            "cproj_shapley_ce": cproj_shapley,
            "v_by_cproj_interaction_ce": interaction,
            "identity_residual": residual,
        },
        "decision_evidence": {
            "positive_cproj_shapley_evaluations": cproj_support,
            "positive_interaction_evaluations": interaction_support,
            "terminal_cproj_shapley_ce": cproj_shapley[-1],
            "terminal_interaction_ce": interaction[-1],
            "cproj_gate": cproj_gate,
            "interaction_gate": interaction_gate,
        },
        "decision": {
            "authorize_cproj_ratio10": authorize,
            "selected_branch": (
                "qk_cproj_only_cprojratio10" if authorize else "no_cproj_capacity_run"
            ),
            "reason": (
                "Preregistered c-proj capacity gate passed."
                if authorize
                else "Neither preregistered c-proj capacity condition passed."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l01-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    result = analyze(plan, json.loads(args.l01_result.read_text()))
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
