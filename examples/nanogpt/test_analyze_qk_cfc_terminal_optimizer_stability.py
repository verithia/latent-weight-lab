from examples.nanogpt.analyze_qk_cfc_terminal_optimizer_stability import classify


def _metrics(stateless: float, momentum: float, corrected: float):
    return {
        "stateless_requested": {"late": {"cosine": stateless}},
        "momentum_requested": {"late": {"cosine": momentum}},
        "feedback_corrected": {"late": {"cosine": corrected}},
    }


RULE = {
    "minimum_stable_late_cosine": 0.1,
    "minimum_material_cosine_gain": 0.05,
}


def test_classifies_momentum_stability() -> None:
    assert classify(_metrics(0.02, 0.20, 0.18), RULE)["classification"] == (
        "MOMENTUM_STABILIZES_TASK_TANGENT"
    )


def test_classifies_feedback_dominance_and_destabilization() -> None:
    assert classify(_metrics(0.02, 0.04, 0.20), RULE)["classification"] == (
        "ERROR_FEEDBACK_DOMINATES_TARGET_STABILITY"
    )
    assert classify(_metrics(0.02, 0.20, 0.04), RULE)["classification"] == (
        "ERROR_FEEDBACK_DESTABILIZES_STABLE_MOMENTUM_TARGET"
    )


def test_classifies_unstable_optimizer_state() -> None:
    assert classify(_metrics(0.02, 0.05, 0.07), RULE)["classification"] == (
        "OPTIMIZER_STATE_DOES_NOT_STABILIZE_TASK_TANGENT"
    )
