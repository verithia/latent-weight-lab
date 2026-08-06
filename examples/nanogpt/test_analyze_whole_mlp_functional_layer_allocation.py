from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_whole_mlp_functional_layer_allocation import (
    aggregate_results,
    evaluate_patch,
    mlp_state_fingerprint,
    mlp_tree_identity,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_repaired_attention_whole_mlp_functional_layer_allocation_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "layers": list(range(12)),
            "primary_window": {
                "seed": 20260907,
                "batch_size": 16,
                "block_size": 1024,
                "batches": 32,
            },
            "confirmation_window": {
                "seed": 20260908,
                "batch_size": 16,
                "block_size": 1024,
                "batches": 32,
            },
            "cumulative_sizes": [1, 2, 3, 4, 6, 8, 10, 12],
            "maximum_selected_donor_layers": 4,
        },
        "authorization": {"run_language_model_training": False},
    }


def decision_plan() -> dict:
    plan = valid_plan()
    plan["decision_rule"] = {
        "control_requirements": {
            "donor_primary_ce_gain_over_joint_minimum": 0.01,
            "donor_confirmation_ce_gain_over_joint_minimum": 0.01,
            "all12_primary_ce_gain_over_joint_minimum": 0.005,
            "all12_confirmation_ce_gain_over_joint_minimum": 0.005,
        },
        "candidate_requirements": {
            "maximum_donor_layers": 4,
            "primary_gain_fraction_of_all12_minimum": 0.75,
            "confirmation_gain_fraction_of_all12_minimum": 0.75,
            "primary_ce_gain_over_joint_minimum": 0.005,
            "confirmation_ce_gain_over_joint_minimum": 0.005,
            "confirmation_ce_above_all12_maximum": 0.005,
        },
    }
    return plan


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["primary_window"]["seed"] += 1
    with pytest.raises(ValueError):
        validate_plan(changed)


class TinyMLP(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))

    def forward(self, values):
        return values * self.weight


class TinyBlock(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.mlp = TinyMLP(value)


class TinyTransformer(torch.nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.h = torch.nn.ModuleList([TinyBlock(value) for value in values])


class TinyModel(torch.nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.transformer = TinyTransformer(values)

    def forward(self, inputs, targets):
        del targets
        values = inputs.float()
        for block in self.transformer.h:
            values = block.mlp(values)
        return None, values.sum()


def test_functional_patch_restores_modules_and_states() -> None:
    joint = TinyModel((2.0, 3.0))
    donor = TinyModel((5.0, 7.0))
    joint_identity = mlp_tree_identity(joint)
    donor_identity = mlp_tree_identity(donor)
    joint_fingerprint = mlp_state_fingerprint(joint)
    donor_fingerprint = mlp_state_fingerprint(donor)
    tokens = [torch.ones(1, 2, dtype=torch.long)]
    loss, restored = evaluate_patch(joint, donor, tokens, (0,), "cpu")
    assert loss == 15.0
    assert restored is True
    assert mlp_tree_identity(joint) == joint_identity
    assert mlp_tree_identity(donor) == donor_identity
    assert mlp_state_fingerprint(joint) == joint_fingerprint
    assert mlp_state_fingerprint(donor) == donor_fingerprint


def synthetic_rows(scale: float = 1.0) -> list[dict]:
    rows = []
    baselines = {"primary": 2.0, "confirmation": 2.1}
    for window, baseline in baselines.items():
        for variant, gain in (
            ("joint_control", 0.0),
            ("donor_control", 0.04 * scale),
            ("all12_functional_patch", 0.04 * scale),
        ):
            rows.append(
                {
                    "window": window,
                    "variant": variant,
                    "loss": baseline - gain,
                    "exact_module_restore_after_eval": True,
                }
            )
        for layer in range(12):
            gain = (0.012 if layer == 3 else 0.01 if layer == 0 else 0.0001)
            rows.append(
                {
                    "window": window,
                    "variant": f"single_layer_{layer}",
                    "loss": baseline - gain * scale,
                    "exact_module_restore_after_eval": True,
                }
            )
        for k in (1, 2, 3, 4, 6, 8, 10, 12):
            gain = min(0.04, 0.011 * k) * scale
            rows.append(
                {
                    "window": window,
                    "variant": f"cumulative_top_{k}",
                    "loss": baseline - gain,
                    "exact_module_restore_after_eval": True,
                }
            )
    return rows


def test_aggregate_selects_smallest_passing_layer_count() -> None:
    result = aggregate_results(synthetic_rows(), decision_plan())
    assert result["passed"] is True
    assert result["selected_k"] == 4
    assert result["selected_layers"][:2] == [3, 0]
    assert result["authorization"]["language_model_training_authorized"] is False


def test_aggregate_rejects_weak_controls() -> None:
    result = aggregate_results(synthetic_rows(scale=0.1), decision_plan())
    assert result["passed"] is False
    assert result["classification"] == "REJECT_WHOLE_MLP_FUNCTIONAL_LAYER_ALLOCATION"
