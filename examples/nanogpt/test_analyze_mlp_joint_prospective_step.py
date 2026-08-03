from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_joint_prospective_step import (
    historical_double_decay_update,
    interaction_decision,
)


def test_historical_double_decay_applies_the_second_decay_to_endpoint() -> None:
    weight = torch.tensor([[2.0]])
    canonical = torch.tensor([[-0.5]])
    update = historical_double_decay_update(
        weight,
        canonical,
        learning_rate=0.1,
        weight_decay=0.2,
    )
    assert torch.equal(update, torch.tensor([[-0.53]]))


def rows(cfc: float, cproj: float, joint: float):
    result = []
    for window in ("validation_1", "validation_2"):
        result.extend(
            [
                {"window": window, "variant": "baseline", "ce": 5.0},
                {"window": window, "variant": "cfc_only", "ce": 5.0 + cfc},
                {
                    "window": window,
                    "variant": "cproj_only",
                    "ce": 5.0 + cproj,
                },
                {"window": window, "variant": "joint", "ce": 5.0 + joint},
            ]
        )
    return result


def decide(cfc: float, cproj: float, joint: float):
    return interaction_decision(
        rows(cfc, cproj, joint),
        additive_tolerance=0.10,
        destructive_threshold=0.25,
        cooperative_threshold=0.10,
    )


def test_interaction_decision_separates_additive_destructive_and_cooperative():
    assert decide(-0.02, -0.03, -0.05)["classification"] == (
        "CFC_CPROJ_UPDATES_ARE_FINITE_CE_ADDITIVE"
    )
    assert decide(-0.02, -0.03, -0.02)["classification"] == (
        "DESTRUCTIVE_CFC_CPROJ_UPDATE_INTERACTION"
    )
    assert decide(-0.02, -0.03, -0.07)["classification"] == (
        "COOPERATIVE_CFC_CPROJ_UPDATE_INTERACTION"
    )


def test_additive_but_harmful_updates_select_individual_direction_fix():
    decision = decide(0.02, 0.03, 0.05)
    assert decision["classification"] == (
        "CFC_CPROJ_UPDATES_ARE_FINITE_CE_ADDITIVE"
    )
    assert decision["joint_helpful_on_all_windows"] is False
    assert decision["next_action"] == (
        "FIX_INDIVIDUAL_PROSPECTIVE_DIRECTIONS_NOT_JOINT_MLP_CHART"
    )
