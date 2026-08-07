from examples.nanogpt.analyze_qk_cfc_20tpp_phase_direction import (
    aggregate_phase_rows,
    classify,
    compare_phase_rows,
    factorial_effects,
)

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _activation_rows(direction_shift: float) -> list[dict[str, float | int | str]]:
    rows = []
    for step in (2373, 4746, 7119, 9489):
        for layer in range(12):
            for regime in ("HEAD", "MID", "TAIL"):
                shift = direction_shift if step >= 7119 and layer >= 8 else 0.0
                rows.append(
                    {
                        "step": step,
                        "layer": layer,
                        "regime": regime,
                        "pregelu_hard_rank": 100.0,
                        "postgelu_hard_rank": 80.0,
                        "postgelu_to_pregelu_hard_rank": 0.8,
                        "update_to_residual_rms": 0.2,
                        "residual_update_parallel_energy": 0.1,
                        "residual_update_cos_mean": 0.02 + shift,
                        "residual_update_cka": 0.03,
                    }
                )
    return rows


def test_phase_comparison_localizes_late_direction_shift() -> None:
    compared = compare_phase_rows(_activation_rows(0.03), 2373)
    aggregate = aggregate_phase_rows(compared)
    assert aggregate["4746"]["late"]["residual_update_cos_mean_delta_mean_abs"] == 0.0
    assert abs(
        aggregate["7119"]["late"]["residual_update_cos_mean_delta_mean_abs"]
        - 0.03
    ) < 1e-12


def test_factorial_effects_use_complete_two_by_two_contrast() -> None:
    result = factorial_effects(
        step=7119,
        reference_pair_current_context_ce=3.20,
        current_cfc_reference_cproj_ce=3.21,
        reference_cfc_current_cproj_ce=3.205,
        native_current_ce=3.225,
    )
    assert abs(float(result["cfc_cproj_interaction_ce"]) - 0.01) < 1e-12
    assert abs(float(result["cfc_effect_with_reference_cproj_ce"]) - 0.01) < 1e-12


def test_classifies_coadapted_residual_direction_drift() -> None:
    aggregate = aggregate_phase_rows(
        compare_phase_rows(_activation_rows(0.03), 2373)
    )
    factorial = [
        factorial_effects(
            step=step,
            reference_pair_current_context_ce=3.20,
            current_cfc_reference_cproj_ce=3.20,
            reference_cfc_current_cproj_ce=3.20,
            native_current_ce=3.206 if step == 7119 else 3.20,
        )
        for step in (4746, 7119, 9489)
    ]
    decision = classify(
        aggregate,
        factorial,
        failure_step=7119,
        minimum_material_ratio=0.1,
        minimum_cosine_shift=0.02,
        minimum_factorial_ce=0.005,
    )
    assert decision["classification"] == "COADAPTED_RESIDUAL_DIRECTION_DRIFT"
    assert decision["earliest_material_late_direction_shift_step"] == 7119
    assert decision["training_or_structure_authorized"] is False


def test_plan_pins_analyzer_and_forbids_training() -> None:
    plan = json.loads(
        (
            ROOT
            / "examples/nanogpt/configs/selection_artifacts/124m_qk_cfc_20tpp_phase_direction_plan.json"
        ).read_text()
    )
    analyzer = ROOT / "examples/nanogpt/analyze_qk_cfc_20tpp_phase_direction.py"
    assert hashlib.sha256(analyzer.read_bytes()).hexdigest() == plan["identity"][
        "entrypoint_sha256"
    ]
    assert plan["observed_before_registration"]["available_snapshot_steps"] == [0]
    assert plan["authorization"]["language_model_training"] is False
    assert plan["decision_rule"]["thresholds_changed_after_measurement"] is False
