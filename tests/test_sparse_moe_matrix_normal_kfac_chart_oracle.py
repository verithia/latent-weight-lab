import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_matrix_normal_kfac_chart_oracle import (
    MatrixNormalKFACChart,
    build_full_kfac_roots,
    coordinate_count,
    factor_operator_cosine,
    normalized_covariance_root,
    trainable_state,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import LayerState


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_sparse_moe_matrix_normal_kfac_chart_oracle_plan.json"


def tiny_state(seed: int = 7) -> LayerState:
    generator = torch.Generator().manual_seed(seed)
    return LayerState(
        torch.randn(2, 4, generator=generator) * 0.1,
        torch.randn(2, 6, 4, generator=generator) * 0.1,
        torch.randn(2, 4, 6, generator=generator) * 0.1,
    )


def identity_roots() -> dict[str, torch.Tensor]:
    return {
        "fc_left": torch.eye(6).expand(2, -1, -1),
        "fc_right": torch.eye(4).expand(2, -1, -1),
        "proj_left": torch.eye(4).expand(2, -1, -1),
        "proj_right": torch.eye(6).expand(2, -1, -1),
    }


def make_tiny(*, shaped: bool) -> MatrixNormalKFACChart:
    return MatrixNormalKFACChart(
        base=tiny_state(), roots=identity_roots() if shaped else None,
        latent_width=8, fht_layers=3, fc_scale=0.02,
        proj_scale=0.02 / (24.0 ** 0.5), seed=19, layer=0, device="cpu",
        shaped=shaped,
    )


def test_exact_registered_coordinate_accounting_exceeds_200x() -> None:
    plan = json.loads(PLAN.read_text())
    count = coordinate_count(
        tensor_layers=12, experts=8, hidden_width=1536, latent_width=5120
    )
    assert count["per_expert"] == 11776
    assert count["compact"] == 1130496
    assert count["dense"] == 226492416
    assert count["compression"] == plan["candidate"]["paired_parameter_compression_ratio"]
    assert count["compression"] > 200.0


def test_covariance_root_is_symmetric_and_reconstructs_registered_metric() -> None:
    generator = torch.Generator().manual_seed(23)
    rows = torch.randn(17, 4, generator=generator)
    weights = torch.rand(17, generator=generator)
    root, stats = normalized_covariance_root(rows, weights, ridge_ratio=1e-6)
    weighted = rows.T @ (weights[:, None] * rows) / weights.sum()
    expected = 4.0 * weighted / weighted.trace() + 1e-6 * torch.eye(4)
    torch.testing.assert_close(root, root.T, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(root @ root.T, expected, rtol=2e-4, atol=2e-4)
    assert stats["rows"] == 17
    assert stats["minimum_eigenvalue"] >= 0.0


def test_full_kfac_builder_returns_all_two_sided_expert_roots() -> None:
    generator = torch.Generator().manual_seed(29)
    state = tiny_state()
    inputs = torch.randn(24, 4, generator=generator)
    errors = torch.randn(24, 4, generator=generator)
    roots, rows = build_full_kfac_roots(
        state, inputs, errors, ridge_ratio=1e-6,
        minimum_assignments=1, device="cpu",
    )
    assert roots["fc_left"].shape == (2, 6, 6)
    assert roots["fc_right"].shape == (2, 4, 4)
    assert roots["proj_left"].shape == (2, 4, 4)
    assert roots["proj_right"].shape == (2, 6, 6)
    assert min(row["assignments"] for row in rows) == 24
    assert factor_operator_cosine(roots, roots)["mean"] > 1.0 - 1e-12


def test_identity_shaping_is_exact_control_and_stepzero_is_dense_base() -> None:
    candidate, control = make_tiny(shaped=True), make_tiny(shaped=False)
    generator = torch.Generator().manual_seed(31)
    inputs = torch.randn(2, 3, 4, generator=generator)
    directions = torch.randn(2, 3, 4, generator=generator)
    candidate_zero = candidate.function_and_jvp(inputs, directions, conditional=True)
    control_zero = control.function_and_jvp(inputs, directions, conditional=False)
    for left, right in zip(candidate_zero, control_zero):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    with torch.no_grad():
        candidate.fc_latent.normal_(generator=generator, std=0.01)
        candidate.proj_latent.normal_(generator=generator, std=0.01)
        candidate.hidden_bias.normal_(generator=generator, std=0.01)
        control.fc_latent.copy_(candidate.fc_latent)
        control.proj_latent.copy_(candidate.proj_latent)
        control.hidden_bias.copy_(candidate.hidden_bias)
    candidate_live = candidate.function_and_jvp(inputs, directions, conditional=True)
    control_live = control.function_and_jvp(inputs, directions, conditional=False)
    for left, right in zip(candidate_live, control_live):
        torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-6)


def test_analytic_jvp_matches_finite_difference_and_all_coordinates_get_gradient() -> None:
    module = make_tiny(shaped=True)
    generator = torch.Generator().manual_seed(37)
    with torch.no_grad():
        module.fc_latent.normal_(generator=generator, std=0.01)
        module.proj_latent.normal_(generator=generator, std=0.01)
        module.hidden_bias.normal_(generator=generator, std=0.01)
    inputs = torch.randn(2, 3, 4, generator=generator)
    directions = torch.randn(2, 3, 4, generator=generator)
    output, analytic = module.function_and_jvp(inputs, directions, conditional=True)
    epsilon = 1e-3
    plus = module.function_and_jvp(
        inputs + epsilon * directions, torch.zeros_like(inputs), conditional=True
    )[0]
    minus = module.function_and_jvp(
        inputs - epsilon * directions, torch.zeros_like(inputs), conditional=True
    )[0]
    torch.testing.assert_close(
        analytic, (plus - minus) / (2 * epsilon), rtol=8e-3, atol=8e-4
    )
    (output.square().mean() + 0.1 * analytic.square().mean()).backward()
    for parameter in module.trainable_parameters(conditional=True):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert set(trainable_state(module)) == {"fc_latent", "proj_latent", "hidden_bias"}
