#!/usr/bin/env python3
"""Apply the preregistered terminal gate to the 124M/20TPP QK+V result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "examples/nanogpt/configs/selection_artifacts"
PLAN = ARTIFACTS / "124m_attention_qkv_only_20tpp_plan.json"
MFU_RESULT = ARTIFACTS / "124m_attention_qkv_only_20tpp_mfu_result.json"
CONFIG = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_qkv_only_qk64_outputgain_20tpp_lr24e4.json"
)
OUTPUT = ARTIFACTS / "124m_attention_qkv_only_20tpp_decision.json"

FIXED_STEPS = [2373, 4746, 7119, 9489]
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_INDICES_SHA256 = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def _fixed_curve(run: dict[str, Any]) -> list[float]:
    rows = run.get("fixed_evaluations")
    if not isinstance(rows, list):
        raise ValueError("candidate lacks run.fixed_evaluations")
    by_step: dict[int, float] = {}
    for row in rows:
        step = int(row["step"])
        if step in by_step:
            raise ValueError(f"duplicate fixed evaluation at step {step}")
        by_step[step] = float(row["validation_ce"])
    if sorted(by_step) != FIXED_STEPS:
        raise ValueError(
            f"fixed evaluation steps differ: {sorted(by_step)} != {FIXED_STEPS}"
        )
    curve = [by_step[step] for step in FIXED_STEPS]
    if not all(math.isfinite(value) for value in curve):
        raise ValueError("fixed evaluation curve contains a non-finite value")
    return curve


def analyze(
    plan: dict[str, Any],
    mfu_result: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if plan.get("schema_version") != "mai_124m_attention_qkv_only_20tpp_plan_v1":
        raise ValueError("plan schema is incompatible")
    if mfu_result.get("classification") != (
        "QKV_ONLY_124M_20TPP_EXACT_CONFIG_MFU_PASSED"
    ):
        raise ValueError("exact-config MFU result did not pass")

    config_sha256 = sha256(CONFIG)
    if mfu_result.get("config", {}).get("sha256") != config_sha256:
        raise ValueError("MFU result does not match the immutable config")
    if float(mfu_result["passing_preflight"]["mfu_fraction"]) < 0.20:
        raise ValueError("MFU result is below the registered threshold")

    run = candidate.get("run")
    if not isinstance(run, dict):
        raise ValueError("candidate lacks run identity")
    expected = {
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "fixed_eval_indices_sha256": FIXED_EVAL_INDICES_SHA256,
        "mfu_certificate_sha256": mfu_result["passing_preflight"][
            "certificate_sha256"
        ],
    }
    for field, value in expected.items():
        if run.get(field) != value:
            raise ValueError(f"candidate.run.{field} does not match registered identity")
    for field in ("provenance_sha256", "status_sha256", "log_sha256", "checkpoint_sha256"):
        _require_sha256(run.get(field), f"candidate.run.{field}")
    if run.get("exit_code") != 0 or run.get("classification") != "clean":
        raise ValueError("candidate is not a clean terminal run")

    curve = _fixed_curve(run)
    terminal = curve[-1]
    dense = float(plan["matched_results"]["dense_124m_20tpp"]["terminal_validation_ce"])
    full = float(plan["matched_results"]["full_attention_124m_20tpp"]["terminal_validation_ce"])
    maximum_dense_delta = float(
        plan["terminal_gate"]["maximum_validation_ce_delta_to_dense"]
    )
    minimum_full_improvement = float(
        plan["terminal_gate"]["minimum_validation_ce_improvement_over_full_attention"]
    )
    delta_to_dense = terminal - dense
    improvement_over_full = full - terminal
    dense_gate = delta_to_dense <= maximum_dense_delta
    full_gate = improvement_over_full >= minimum_full_improvement
    passed = dense_gate and full_gate

    return {
        "schema_version": "mai_124m_attention_qkv_only_20tpp_decision_v1",
        "source_plan": str(PLAN.relative_to(ROOT)),
        "source_plan_sha256": sha256(PLAN),
        "mfu_result": str(MFU_RESULT.relative_to(ROOT)),
        "mfu_result_sha256": sha256(MFU_RESULT),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": config_sha256,
        "identity": expected,
        "fixed_steps": FIXED_STEPS,
        "candidate_validation_ce": curve,
        "comparisons": {
            "terminal_candidate_validation_ce": terminal,
            "terminal_dense_validation_ce": dense,
            "terminal_full_attention_validation_ce": full,
            "terminal_delta_to_dense_ce": delta_to_dense,
            "terminal_improvement_over_full_attention_ce": improvement_over_full,
        },
        "registered_gate": {
            "maximum_validation_ce_delta_to_dense": maximum_dense_delta,
            "minimum_validation_ce_improvement_over_full_attention": minimum_full_improvement,
        },
        "decision_evidence": {
            "dense_gate": dense_gate,
            "full_attention_gate": full_gate,
        },
        "decision": {
            "confirmed": passed,
            "classification": (
                plan["terminal_gate"]["pass_classification"]
                if passed
                else plan["terminal_gate"]["fail_classification"]
            ),
            "next_action": (
                plan["terminal_gate"]["pass_action"]
                if passed
                else plan["terminal_gate"]["fail_action"]
            ),
        },
        "scope": {
            "partial_attention_only": True,
            "dense_c_proj": True,
            "materialized_inference_parameters_unchanged": True,
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-result", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = analyze(
        json.loads(PLAN.read_text()),
        json.loads(MFU_RESULT.read_text()),
        json.loads(args.candidate_result.read_text()),
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(args.output)
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
