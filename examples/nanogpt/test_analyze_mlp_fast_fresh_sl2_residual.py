from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_fast_fresh_sl2_residual import (
    CANDIDATES,
    aggregate_results,
    apply_pair_stage,
    coordinates_to_generators,
    fit_pair_coordinates,
    fit_tracefree_flow,
    tracefree_matched_permutations,
)


def test_tracefree_generators_and_maps_have_unit_determinant() -> None:
    coordinates = torch.tensor(
        [[0.2, -0.1, 0.3], [-0.4, 0.2, -0.1]],
        dtype=torch.float64,
    )
    generators = coordinates_to_generators(coordinates)
    torch.testing.assert_close(
        generators.diagonal(dim1=1, dim2=2).sum(dim=1),
        torch.zeros(2, dtype=torch.float64),
    )
    maps = torch.matrix_exp(generators)
    torch.testing.assert_close(
        torch.linalg.det(maps),
        torch.ones(2, dtype=torch.float64),
    )


def test_exact_pair_normal_equation_recovers_tangent_coordinates() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(11, 4, generator=generator)
    pairs = torch.tensor([[0, 1], [2, 3]])
    expected = torch.tensor(
        [[0.03, -0.02, 0.01], [-0.04, 0.01, 0.02]],
        dtype=torch.float64,
    )
    tangent = coordinates_to_generators(expected)
    pair_values = torch.stack(
        (source[:, pairs[:, 0]], source[:, pairs[:, 1]]), dim=-1
    )
    target_pairs = torch.einsum(
        "rpk,pkj->rpj", pair_values.double(), tangent
    )
    residual = torch.zeros_like(source)
    residual[:, pairs[:, 0]] = target_pairs[:, :, 0].float()
    residual[:, pairs[:, 1]] = target_pairs[:, :, 1].float()
    observed = fit_pair_coordinates(
        source, residual, pairs, "full"
    )
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-7)


def test_apply_pair_stage_is_finite_and_unit_determinant() -> None:
    source = torch.randn(8, 6)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    coordinates = torch.randn(3, 3, dtype=torch.float64) * 0.05
    result, diagnostics = apply_pair_stage(
        source, pairs, coordinates
    )
    assert torch.isfinite(result).all()
    assert diagnostics["maximum_determinant_error"] < 1e-5
    assert diagnostics["maximum_condition_number"] >= 1.0


def test_tracefree_flow_improves_small_finite_target() -> None:
    generator = torch.Generator().manual_seed(11)
    source = torch.randn(12, 8, generator=generator)
    permutations = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    pairs = permutations[0].reshape(-1, 2)
    coordinates = torch.randn(
        4, 3, generator=generator, dtype=torch.float64
    ) * 1e-3
    target, _ = apply_pair_stage(source, pairs, coordinates)
    requested = target - source
    fitted, diagnostics = fit_tracefree_flow(
        source,
        requested,
        permutations,
        stages=1,
        component="full",
    )
    assert diagnostics["requested_update_recovery"] > 0.999
    assert float((requested - fitted).square().sum()) < float(
        requested.square().sum()
    )


def test_tracefree_matching_returns_valid_unique_perfect_matchings(
    tmp_path,
) -> None:
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(7, 8, generator=generator)
    residual = torch.randn(7, 8, generator=generator)
    permutations, diagnostics = tracefree_matched_permutations(
        source,
        residual,
        stages=2,
        neighbors=4,
        seed=19,
        native_cache=tmp_path,
    )
    assert permutations.shape == (2, 8)
    edges: set[tuple[int, int]] = set()
    for permutation in permutations:
        torch.testing.assert_close(
            torch.sort(permutation).values, torch.arange(8)
        )
        for left, right in permutation.reshape(-1, 2).tolist():
            edge = tuple(sorted((left, right)))
            assert edge not in edges
            edges.add(edge)
    assert diagnostics["native_output_validated"]


def _synthetic_rows(
    *,
    equal_ratio: float,
    attribution_ratio: float,
    future_equal_ratio: float,
    future_attribution_ratio: float,
) -> tuple[list[dict], list[dict]]:
    rows = []
    finite = []
    values = {candidate: 1.0 for candidate in CANDIDATES}
    future = dict(values)
    validation = dict(values)
    for stages, control in ((8, "fresh_hidden88"), (16, "fresh_hidden112")):
        full = f"fresh_hidden64_plus_sl2_{stages}"
        skew = f"{full}_skew_only"
        values[control] = 1.0
        validation[control] = 1.0
        future[control] = 1.0
        values[skew] = equal_ratio / attribution_ratio
        validation[skew] = equal_ratio / attribution_ratio
        future[skew] = (
            future_equal_ratio / future_attribution_ratio
        )
        values[full] = equal_ratio
        validation[full] = equal_ratio
        future[full] = future_equal_ratio
    for window in ("fit", "holdout"):
        for candidate in CANDIDATES:
            rows.append(
                {
                    "candidate": candidate,
                    "window": window,
                    "current_weight_recovery": values[candidate],
                    "current_weight_energy": 1.0,
                    "current_residual_fixed_scale_recovery": values[
                        candidate
                    ],
                    "current_residual_energy": 1.0,
                    "future_residual_positive_line_recovery": future[
                        candidate
                    ],
                    "future_residual_energy": 1.0,
                    "current_output_positive_line_recovery": values[
                        candidate
                    ],
                    "current_output_fixed_scale_recovery": values[
                        candidate
                    ],
                    "current_output_energy": 1.0,
                    "train_gradient_predicted_ce_decrease": values[
                        candidate
                    ],
                    "validation_gradient_predicted_ce_decrease": (
                        validation[candidate]
                    ),
                }
            )
    for phase in (0, 60, 120, 180):
        for window in ("fit", "holdout"):
            for candidate in CANDIDATES:
                loss = (
                    1.9
                    if candidate
                    in {
                        "fresh_hidden64_plus_sl2_8",
                        "fresh_hidden64_plus_sl2_16",
                    }
                    else 2.0
                )
                finite.append(
                    {
                        "base_update": phase,
                        "window": window,
                        "candidate": candidate,
                        "loss": loss,
                    }
                )
    return rows, finite


def test_registered_gate_passes_only_with_equal_and_attribution_gain() -> None:
    rows, finite = _synthetic_rows(
        equal_ratio=1.06,
        attribution_ratio=1.06,
        future_equal_ratio=1.11,
        future_attribution_ratio=1.11,
    )
    result = aggregate_results(rows, finite)
    assert result["decision"] == (
        "SELECT_TRACEFREE_PAIR_CHART_FOR_IMPLEMENTATION_PREFLIGHT"
    )
    assert result["passing_depths"] == [8, 16]
    assert result["selected_branch"] == "sl2_8"


def test_registered_gate_rejects_topology_only_gain() -> None:
    rows, finite = _synthetic_rows(
        equal_ratio=1.06,
        attribution_ratio=1.01,
        future_equal_ratio=1.11,
        future_attribution_ratio=1.01,
    )
    result = aggregate_results(rows, finite)
    assert result["decision"] == "REJECT_SPARSE_TRACEFREE_PAIR_BLOCKS"
    assert result["passing_depths"] == []
