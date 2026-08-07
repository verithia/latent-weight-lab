from examples.nanogpt.analyze_qk_cfc_20tpp_terminal_activation_drift import (
    aggregate_comparisons,
    classify,
    compare_rows,
)

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _rows(post_ratio: float) -> list[dict[str, float | int | str]]:
    rows = []
    for run, scale in (("parent", 1.0), ("candidate", post_ratio)):
        for layer in range(12):
            for regime in ("HEAD", "MID", "TAIL"):
                rows.append(
                    {
                        "run": run,
                        "layer": layer,
                        "regime": regime,
                        "pregelu_hard_rank": 100.0,
                        "postgelu_hard_rank": 80.0 * scale,
                        "postgelu_to_pregelu_hard_rank": 0.8 * scale,
                        "update_to_residual_rms": 0.2,
                        "residual_update_parallel_energy": 0.1,
                        "residual_update_cos_mean": 0.02,
                        "residual_update_cka": 0.03,
                    }
                )
    return rows


def test_classifies_post_gelu_conversion_deficit() -> None:
    compared = compare_rows(_rows(0.8), "parent", "candidate")
    aggregate = aggregate_comparisons(compared)
    decision = classify(
        aggregate,
        {"parent": 3.1, "candidate": 3.2},
        parent="parent",
        candidate="candidate",
        minimum_material_ratio=0.1,
        minimum_cosine_shift=0.02,
    )
    assert decision["classification"] == "POST_GELU_CONVERSION_DEFICIT"
    assert decision["probe_reproduces_terminal_order"] is True
    assert decision["training_or_structure_authorized"] is False


def test_classifies_unresolved_when_ratios_match() -> None:
    compared = compare_rows(_rows(1.0), "parent", "candidate")
    aggregate = aggregate_comparisons(compared)
    decision = classify(
        aggregate,
        {"parent": 3.1, "candidate": 3.11},
        parent="parent",
        candidate="candidate",
        minimum_material_ratio=0.1,
        minimum_cosine_shift=0.02,
    )
    assert decision["classification"] == "DISTRIBUTED_OR_UNRESOLVED_TERMINAL_DRIFT"


def test_plan_pins_entrypoint_and_forbids_training() -> None:
    plan = json.loads(
        (
            ROOT
            / "examples/nanogpt/configs/selection_artifacts/124m_qk_cfc_20tpp_terminal_activation_drift_plan.json"
        ).read_text()
    )
    source = ROOT / "examples/nanogpt/analyze_qk_cfc_20tpp_terminal_activation_drift.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == plan["identity"][
        "entrypoint_sha256"
    ]
    assert plan["authorization"]["parameter_updates"] == 0
    assert plan["authorization"]["automatic_training_arm"] is False
    assert plan["decision_rule"]["thresholds_changed_after_measurement"] is False
