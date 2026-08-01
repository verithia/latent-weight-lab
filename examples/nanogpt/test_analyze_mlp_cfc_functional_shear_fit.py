import torch

from examples.nanogpt.analyze_mlp_cfc_functional_shear_fit import (
    CONTROL,
    FUNCTIONAL_BOTH,
    FUNCTIONAL_FIT,
    FUNCTIONAL_TOPOLOGY,
    WEIGHT_SHEAR,
    aggregate,
    fit_functional_shear_flow,
    functional_shear_scores,
    gelu_derivative,
)


def test_gelu_derivative_matches_autograd() -> None:
    values = torch.linspace(-3.0, 3.0, 17, requires_grad=True)
    torch.nn.functional.gelu(values).sum().backward()
    torch.testing.assert_close(gelu_derivative(values.detach()), values.grad)


def test_functional_scores_identify_injected_shear_pair() -> None:
    source = torch.tensor([[1.0, 0.0, 0.2, 0.0], [0.0, 1.0, 0.0, 0.2]])
    inputs = torch.tensor([[1.0, 0.5], [-0.5, 1.0], [0.7, -1.2]])
    pre = torch.zeros(3, 4)
    cproj = torch.eye(4)
    requested = torch.zeros_like(source)
    requested[:, 0] = 0.1 * source[:, 1]
    requested[:, 1] = 0.1 * source[:, 0]
    scores, _diagnostics = functional_shear_scores(
        source, requested, inputs, pre, cproj
    )
    maximum = torch.nonzero(scores == scores.max())[0].tolist()
    assert set(maximum) == {0, 1}


def test_functional_scores_match_explicit_output_projection() -> None:
    generator = torch.Generator().manual_seed(19)
    source = torch.randn(3, 4, generator=generator)
    requested = torch.randn(3, 4, generator=generator) * 0.01
    inputs = torch.randn(5, 3, generator=generator)
    pre = torch.randn(5, 4, generator=generator)
    cproj = torch.randn(2, 4, generator=generator)
    scores, _diagnostics = functional_shear_scores(
        source, requested, inputs, pre, cproj
    )
    slopes = gelu_derivative(pre)
    target_output = (slopes * (inputs @ requested)) @ cproj.T
    for left in range(4):
        for right in range(left + 1, 4):
            direction = torch.zeros_like(source)
            direction[:, left] = source[:, right]
            direction[:, right] = source[:, left]
            output = (slopes * (inputs @ direction)) @ cproj.T
            expected = (
                (target_output * output).sum().square()
                / output.square().sum().clamp_min(1e-30)
            )
            torch.testing.assert_close(scores[left, right], expected)


def test_functional_flow_recovers_single_shear_stage() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    inputs = torch.tensor([[1.0, 0.5], [-0.5, 1.0], [0.7, -1.2]])
    pre = torch.zeros(3, 2)
    cproj = torch.eye(2)
    coordinates = torch.tensor([[0.02, 0.0]], dtype=torch.float64)
    pairs = torch.tensor([[0, 1]])
    from examples.nanogpt.analyze_mlp_cfc_task_shear_fit import apply_pair_stage

    target, _diagnostics = apply_pair_stage(source, pairs, coordinates)
    requested = target - source
    fitted, diagnostics = fit_functional_shear_flow(
        source,
        requested,
        inputs,
        pre,
        cproj,
        torch.tensor([[0, 1]]),
        stages=1,
    )
    assert diagnostics["functional_requested_recovery"] > 0.999
    torch.testing.assert_close(fitted, requested, atol=1e-4, rtol=5e-3)


def _metric_rows(functional_value: float) -> list[dict]:
    rows = []
    for window in ("fit", "holdout"):
        for candidate, value in (
            (CONTROL, 0.2),
            (WEIGHT_SHEAR, 0.2),
            (FUNCTIONAL_TOPOLOGY, 0.2),
            (FUNCTIONAL_FIT, 0.2),
            (FUNCTIONAL_BOTH, functional_value),
        ):
            rows.append(
                {
                    "window": window,
                    "layer": 0,
                    "candidate": candidate,
                    "weight_target_energy": 1.0,
                    "weight_fixed_scale_recovery": 0.2,
                    "post_gelu_target_energy": 1.0,
                    "post_gelu_fixed_scale_recovery": value,
                    "mlp_output_target_energy": 1.0,
                    "mlp_output_fixed_scale_recovery": value,
                    "predicted_ce_decrease": value,
                }
            )
    return rows


def test_aggregate_promotes_functional_gain() -> None:
    result = aggregate(
        _metric_rows(0.24),
        [],
        minimum_functional_ratio=1.1,
        minimum_ce_descent_ratio=1.0,
        minimum_weight_ratio=0.9,
        maximum_determinant_error=1e-6,
        maximum_condition_number=1.1,
    )
    assert result["decision"] == "PROMOTE_FUNCTIONAL_SHEAR_TO_HELDOUT_CE"


def test_aggregate_rejects_small_functional_gain() -> None:
    result = aggregate(
        _metric_rows(0.21),
        [],
        minimum_functional_ratio=1.1,
        minimum_ce_descent_ratio=1.0,
        minimum_weight_ratio=0.9,
        maximum_determinant_error=1e-6,
        maximum_condition_number=1.1,
    )
    assert result["decision"] == "REJECT_ACTIVATION_WEIGHTED_SHEAR"
