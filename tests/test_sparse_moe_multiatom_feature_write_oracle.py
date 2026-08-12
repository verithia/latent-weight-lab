import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_multiatom_feature_write_oracle import (
    coordinate_count,
    make_module,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_sparse_moe_multiatom_feature_write_oracle_plan.json"


def load_plan() -> dict:
    return json.loads(PLAN.read_text())


def test_exact_full_mlp_accounting_exceeds_200x() -> None:
    plan = load_plan()
    count = coordinate_count(
        tensor_layers=12, experts=8, hidden_width=1536, padded_width=2048
    )
    assert count["feature"] == 589824
    assert count["bias"] == 147456
    assert count["sparse_write"] == 294912
    assert count["output_flow"] == 46080
    assert count["compact"] == 1078272
    assert count["dense"] == 226492416
    assert count["compression"] == plan["candidate"]["paired_parameter_compression_ratio"]
    assert count["compression"] > 200.0


def test_two_supports_are_collision_free_and_balanced() -> None:
    plan = load_plan()
    module = make_module(plan, 0, "cpu", independent=True)
    support = module.write_support
    assert support.shape == (1536, 2)
    assert torch.all(support[:, 0] != support[:, 1])
    for stream in range(2):
        assert torch.equal(
            torch.bincount(support[:, stream], minlength=768),
            torch.full((768,), 2, dtype=torch.long),
        )


def test_candidate_control_have_equal_coordinates_and_step_zero_function() -> None:
    plan = load_plan()
    candidate = make_module(plan, 0, "cpu", independent=True)
    control = make_module(plan, 0, "cpu", independent=False)
    expected = plan["candidate"]["total_coordinates_per_layer"]
    assert candidate.compact_parameter_count(paired=True) == expected == 89856
    assert control.compact_parameter_count(paired=False) == expected
    for left, right in zip(
        candidate.trainable_parameters(paired=True),
        control.trainable_parameters(paired=False),
    ):
        torch.testing.assert_close(left, right)
    assert not torch.equal(
        candidate.atom_signs[1], candidate.atom_signs[0]
    )
    assert torch.equal(control.atom_signs[1], control.atom_signs[0])
    inputs = torch.randn(8, 2, 768)
    directions = torch.randn_like(inputs)
    candidate_output = candidate.function_details(
        inputs, directions, paired=True
    )
    control_output = control.function_details(
        inputs, directions, paired=False
    )
    for left, right in zip(candidate_output, control_output):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_analytic_jvp_and_generated_write_columns_match_function() -> None:
    plan = load_plan()
    module = make_module(plan, 0, "cpu", independent=True)
    generator = torch.Generator().manual_seed(53)
    with torch.no_grad():
        module.feature_spectra.normal_(generator=generator, std=0.03)
        module.hidden_bias.normal_(generator=generator, std=0.01)
        module.write_coefficients.normal_(generator=generator, std=0.02)
    inputs = torch.randn(8, 2, 768, generator=generator)
    directions = torch.randn_like(inputs)
    output, analytic, hidden = module.function_details(
        inputs, directions, paired=True
    )
    writes = module.generated_write_columns()
    torch.testing.assert_close(output, torch.bmm(hidden, writes), rtol=2e-5, atol=2e-5)
    epsilon = 1e-3
    plus, _ = module.function_and_jvp(
        inputs + epsilon * directions,
        torch.zeros_like(directions),
        conditional=True,
    )
    minus, _ = module.function_and_jvp(
        inputs - epsilon * directions,
        torch.zeros_like(directions),
        conditional=True,
    )
    torch.testing.assert_close(
        analytic, (plus - minus) / (2.0 * epsilon), rtol=7e-3, atol=6e-4
    )


def test_every_counted_group_receives_finite_gradient() -> None:
    plan = load_plan()
    module = make_module(plan, 0, "cpu", independent=True)
    with torch.no_grad():
        module.write_coefficients.normal_(std=0.02)
    inputs = torch.randn(8, 2, 768)
    directions = torch.randn_like(inputs)
    output, output_jvp = module.function_and_jvp(
        inputs, directions, conditional=True
    )
    loss = output.square().mean() + 0.1 * output_jvp.square().mean()
    loss.backward()
    for parameter in module.trainable_parameters(paired=True):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
