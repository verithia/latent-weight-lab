from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cproj_layer0_general_output_capacity import (
    fit_general_output_actions,
    select_general_action,
)


def test_general_output_fit_reconstructs_representable_action() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(6, 10, generator=generator)
    action = torch.randn(6, 6, generator=generator) * 1e-3
    target = weight + action @ weight
    candidates, diagnostics = fit_general_output_actions(
        weight,
        target,
        relative_ridge=1e-8,
        minimum_singular_value=0.95,
    )
    assert torch.allclose(candidates["full"], target, atol=2e-5, rtol=2e-5)
    assert diagnostics["full"]["endpoint_recovery"] > 0.999
    assert diagnostics["full"]["minimum_singular_value_i_plus_action"] >= 0.95


def test_skew_and_symmetric_components_sum_to_full_delta() -> None:
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(5, 9, generator=generator)
    action = torch.randn(5, 5, generator=generator) * 1e-3
    target = weight + action @ weight
    candidates, _ = fit_general_output_actions(
        weight,
        target,
        relative_ridge=1e-8,
        minimum_singular_value=0.95,
    )
    skew_delta = candidates["skew"] - weight
    symmetric_delta = candidates["symmetric"] - weight
    full_delta = candidates["full"] - weight
    assert torch.allclose(skew_delta + symmetric_delta, full_delta, atol=2e-5, rtol=2e-5)


def _rows(candidate_gains: dict[str, tuple[float, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"variant": "output32_all", "val_ce_seed_1": 5.0, "val_ce_seed_2": 5.1},
        {"variant": "directed16_layer0", "val_ce_seed_1": 4.999, "val_ce_seed_2": 5.099},
        {"variant": "dense_layer0", "val_ce_seed_1": 4.998, "val_ce_seed_2": 5.098},
    ]
    for name, gains in candidate_gains.items():
        rows.append(
            {
                "variant": f"general_{name}_layer0",
                "val_ce_seed_1": 5.0 - gains[0],
                "val_ce_seed_2": 5.1 - gains[1],
            }
        )
    return rows


def _diagnostics() -> dict[str, dict[str, float]]:
    return {
        name: {
            "ridge": 1e-6,
            "trust_scale": 1.0,
            "minimum_singular_value_i_plus_action": 0.99,
            "action_fro": 0.1,
            "delta_energy": 1.0,
            "target_residual_energy": 2.0,
            "endpoint_recovery": 0.8,
        }
        for name in ("skew", "symmetric", "full")
    }


def test_selects_smallest_passing_component() -> None:
    result = select_general_action(
        _rows(
            {
                "skew": (0.0016, 0.0016),
                "symmetric": (0.0018, 0.0018),
                "full": (0.0019, 0.0019),
            }
        ),
        seeds=[1, 2],
        minimum_gain=0.0012,
        minimum_dense_recovery=0.75,
        diagnostics=_diagnostics(),
    )
    assert result["selected_variant"] == "general_skew_layer0"
    assert result["authorization"]["production_implementation"] is True
    assert result["authorization"]["language_model_training"] is False


def test_rejects_when_no_component_beats_directed_and_dense_fraction() -> None:
    result = select_general_action(
        _rows(
            {
                "skew": (0.0008, 0.0008),
                "symmetric": (0.0010, 0.0010),
                "full": (0.0011, 0.0011),
            }
        ),
        seeds=[1, 2],
        minimum_gain=0.0012,
        minimum_dense_recovery=0.75,
        diagnostics=_diagnostics(),
    )
    assert result["decision"] == "REJECT_LAYER0_GENERAL_OUTPUT_ACTION"
    assert result["authorization"]["exact_config_mfu_preflight"] is False
