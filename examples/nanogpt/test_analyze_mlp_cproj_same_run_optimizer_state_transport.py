from __future__ import annotations

import pytest
import torch

from examples.nanogpt.analyze_mlp_cproj_same_run_optimizer_state_transport import (
    COMPONENTS,
    DISCOVERY_STEPS,
    HELDOUT_PROBE_STEPS,
    LAYERS,
    PROBE_STEPS,
    REFERENCE_POST_STEPS,
    SNAPSHOT_STEPS,
    TERMINAL_STEP,
    phase_target_step,
    reconstruct_components,
    validate_plan,
)


def valid_plan() -> dict[str, object]:
    return {
        "schema_version": (
            "mai_124m_mlp_cproj_same_run_optimizer_state_transport_plan_v2"
        ),
        "analysis": {
            "parameter_updates": 0,
            "same_run_only": True,
            "layers": list(LAYERS),
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "probe_steps": list(PROBE_STEPS),
            "reference_post_steps": list(REFERENCE_POST_STEPS),
            "discovery_steps": list(DISCOVERY_STEPS),
            "terminal_step": TERMINAL_STEP,
            "polynomial_rank": 4,
            "polynomial_degree": 2,
            "activation_rows": 2048,
            "terminal_activations_from_same_run_checkpoint": True,
            "components": list(COMPONENTS),
            "heldout_probe_steps": list(HELDOUT_PROBE_STEPS),
            "output_additive_projection": True,
            "state_component_reconstruction": (
                "exact_post_step_optimizer_identity"
            ),
            "recomputed_polar_request": "diagnostic_only",
            "authorization_metric": (
                "heldout_future_functional_positive_line_recovery"
            ),
            "future_phase_target_by_probe": {
                str(step): phase_target_step(index)
                for index, step in enumerate(PROBE_STEPS)
            },
        },
        "decision_rule": {
            "thresholds": {
                "state_identity_max_relative_error": 1e-4,
                "causal_heldout_functional_line_recovery_minimum": 0.80,
            }
        },
        "authorization": {
            "run_zero_update_state_transport_analysis": True,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def test_phase_target_mapping_has_no_cross_run_off_by_one() -> None:
    expected = [99, 297, 594, 891, 1188, 1485, 1782, 2079, 2373, None]
    assert [phase_target_step(index) for index in range(len(PROBE_STEPS))] == expected
    with pytest.raises(IndexError):
        phase_target_step(len(PROBE_STEPS))


def test_plan_freezes_same_run_inputs_and_fail_closed_thresholds() -> None:
    plan = valid_plan()
    validate_plan(plan)
    plan["analysis"]["same_run_only"] = False
    with pytest.raises(ValueError, match="same_run_only"):
        validate_plan(plan)


def test_exact_state_components_recompose_without_polar_replay() -> None:
    before = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    realized = torch.tensor([[0.1, -0.2], [0.3, -0.4]])
    feedback_before = torch.tensor([[0.5, 0.25], [-0.5, -0.25]])
    residual_after = torch.tensor([[0.7, -0.1], [0.2, -0.3]])
    state = {
        "weight_before_step": before,
        "weight_after_step": before + realized,
        "compression_residual_before_step": feedback_before,
        "compression_residual_after_step": residual_after,
    }
    components, error = reconstruct_components(
        state, {"error_feedback_decay": 0.5}
    )
    assert error < 1e-6
    assert torch.allclose(components["realized"], realized)
    assert torch.equal(components["unrepresented"], residual_after)
    assert torch.allclose(
        components["requested"] + components["feedback"],
        components["corrected"],
    )
