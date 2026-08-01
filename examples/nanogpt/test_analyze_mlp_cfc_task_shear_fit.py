import torch

from examples.nanogpt.analyze_mlp_cfc_task_shear_fit import (
    aggregate,
    apply_pair_stage,
    fit_pair_coordinates,
    fit_pair_flow,
    fit_pair_recipe,
)


def test_pair_coordinates_recover_skew_shear_tangent() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(5, 4, generator=generator)
    pairs = torch.tensor([[0, 1], [2, 3]])
    expected = torch.tensor([[0.03, -0.02], [-0.01, 0.04]])
    left, right = pairs.unbind(dim=1)
    u = source[:, left]
    v = source[:, right]
    residual = torch.zeros_like(source)
    residual[:, left] = expected[:, 0] * v + expected[:, 1] * v
    residual[:, right] = expected[:, 0] * u - expected[:, 1] * u
    actual = fit_pair_coordinates(
        source, residual, pairs, family="skew_shear"
    )
    torch.testing.assert_close(actual.float(), expected, atol=2e-6, rtol=2e-6)


def test_pair_stage_has_unit_determinant() -> None:
    source = torch.randn(6, 4, generator=torch.Generator().manual_seed(11))
    pairs = torch.tensor([[0, 1], [2, 3]])
    coordinates = torch.tensor([[0.1, 0.03], [-0.05, 0.07]])
    _result, diagnostics = apply_pair_stage(source, pairs, coordinates)
    assert diagnostics["maximum_determinant_error"] < 1e-6
    assert diagnostics["maximum_condition_number"] < 1.3


def test_pair_flow_reports_exact_coordinate_count() -> None:
    source = torch.randn(7, 8, generator=torch.Generator().manual_seed(13))
    requested = torch.randn(7, 8, generator=torch.Generator().manual_seed(17)) * 1e-3
    permutations = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 2, 1, 3, 4, 6, 5, 7],
        ]
    )
    _update, diagnostics = fit_pair_flow(
        source,
        requested,
        permutations,
        stages=2,
        family="skew_shear",
    )
    assert diagnostics["coordinates"] == 16
    assert diagnostics["requested_update_recovery"] > 0.0

    recipe_update, recipe_diagnostics, recipe = fit_pair_recipe(
        source,
        requested,
        permutations,
        stages=2,
        family="skew_shear",
    )
    torch.testing.assert_close(recipe_update, _update)
    assert recipe_diagnostics == diagnostics
    assert len(recipe) == 2


def test_aggregate_requires_all_layer_nonnegative_delta() -> None:
    rows = []
    for layer, control, candidate in ((0, 0.2, 0.23), (1, 0.2, 0.19)):
        for name, recovery in (
            ("fresh64", 0.15),
            ("fresh88", control),
            ("fresh64_shear24", candidate),
            ("fresh64_skew_shear12", candidate),
            ("fresh48_skew_shear20", candidate),
        ):
            rows.append(
                {
                    "layer": layer,
                    "candidate": name,
                    "target_energy": 1.0,
                    "residual_energy": 1.0 - recovery,
                    "fixed_scale_recovery": recovery,
                }
            )
    result = aggregate(
        rows,
        [],
        minimum_layer_delta=0.0,
        minimum_aggregate_ratio=1.0,
        maximum_determinant_error=1e-6,
        maximum_condition_number=1.1,
    )
    assert result["decision"] == "REJECT_EQUAL_COORDINATE_TASK_SHEAR_FIT"
    assert result["selected_candidate"] is None
