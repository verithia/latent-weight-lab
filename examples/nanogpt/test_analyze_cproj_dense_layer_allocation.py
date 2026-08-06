from __future__ import annotations

import copy

import pytest
import torch

from examples.nanogpt.analyze_cproj_dense_layer_allocation import (
    aggregate_results,
    evaluate_restore,
    validate_plan,
)


def valid_plan() -> dict:
    return {
        "schema_version": "mai_124m_repaired_attention_cproj_dense_layer_allocation_plan_v1",
        "analysis": {
            "parameter_updates": 0,
            "layers": list(range(12)),
            "primary_window": {
                "seed": 20260905,
                "batch_size": 16,
                "block_size": 1024,
                "batches": 32,
            },
            "confirmation_window": {
                "seed": 20260906,
                "batch_size": 16,
                "block_size": 1024,
                "batches": 32,
            },
            "cumulative_sizes": [1, 2, 3, 4, 6, 8, 10, 12],
            "maximum_selected_dense_layers": 4,
        },
        "authorization": {"run_language_model_training": False},
    }


def test_plan_validation_fails_closed() -> None:
    validate_plan(valid_plan())
    changed = copy.deepcopy(valid_plan())
    changed["analysis"]["confirmation_window"]["batches"] = 16
    with pytest.raises(ValueError):
        validate_plan(changed)


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_proj = torch.nn.Linear(2, 2, bias=False)


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMLP()


class TinyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = torch.nn.ModuleList([TinyBlock(), TinyBlock()])


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = TinyTransformer()

    def forward(self, inputs, targets):
        del inputs, targets
        total = sum(block.mlp.c_proj.weight.sum() for block in self.transformer.h)
        return None, total


def test_evaluate_restore_restores_bitwise() -> None:
    model = TinyModel()
    originals = [block.mlp.c_proj.weight.detach().clone() for block in model.transformer.h]
    donor = {0: torch.ones(2, 2), 1: 2.0 * torch.ones(2, 2)}
    tokens = [torch.zeros(1, 2, dtype=torch.long)]
    loss, restored = evaluate_restore(model, tokens, donor, (0, 1), "cpu")
    assert loss == 12.0
    assert restored is True
    for block, original in zip(model.transformer.h, originals, strict=True):
        assert torch.equal(block.mlp.c_proj.weight, original)


def synthetic_rows(scale: float = 1.0) -> list[dict]:
    rows = []
    baselines = {"primary": 2.0, "confirmation": 2.1}
    for window, baseline in baselines.items():
        rows.append(
            {
                "window": window,
                "variant": "joint_control",
                "loss": baseline,
                "exact_restore_after_eval": True,
            }
        )
        rows.append(
            {
                "window": window,
                "variant": "all12_restore",
                "loss": baseline - 0.04 * scale,
                "exact_restore_after_eval": True,
            }
        )
        for layer in range(12):
            gain = (0.012 if layer == 3 else 0.01 if layer == 0 else 0.0001)
            rows.append(
                {
                    "window": window,
                    "variant": f"single_layer_{layer}",
                    "loss": baseline - gain * scale,
                    "exact_restore_after_eval": True,
                }
            )
        for k in (1, 2, 3, 4, 6, 8, 10, 12):
            gain = min(0.04, 0.011 * k) * scale
            rows.append(
                {
                    "window": window,
                    "variant": f"cumulative_top_{k}",
                    "loss": baseline - gain,
                    "exact_restore_after_eval": True,
                }
            )
    return rows


def decision_plan() -> dict:
    plan = valid_plan()
    plan["decision_rule"] = {
        "restore_control_requirements": {
            "all12_primary_ce_gain_minimum": 0.01,
            "all12_confirmation_ce_gain_minimum": 0.01,
        },
        "candidate_requirements": {
            "maximum_dense_layers": 4,
            "primary_gain_fraction_of_all12_minimum": 0.75,
            "confirmation_gain_fraction_of_all12_minimum": 0.75,
            "primary_ce_gain_minimum": 0.01,
            "confirmation_ce_gain_minimum": 0.01,
            "confirmation_ce_above_all12_maximum": 0.01,
        },
    }
    return plan


def test_aggregate_selects_smallest_passing_layer_count() -> None:
    result = aggregate_results(synthetic_rows(), decision_plan())
    assert result["passed"] is True
    assert result["selected_k"] == 3
    assert result["selected_layers"][:2] == [3, 0]
    assert result["authorization"]["language_model_training_authorized"] is False


def test_aggregate_rejects_weak_restore_control() -> None:
    result = aggregate_results(synthetic_rows(scale=0.1), decision_plan())
    assert result["passed"] is False
    assert result["classification"] == "REJECT_CPROJ_DENSE_LAYER_EXCEPTION_ALLOCATION"
