import json
from pathlib import Path

import torch

from examples.nanogpt.analyze_sparse_moe_state_conditioned_butterfly_transport_oracle import (
    StateConditionedButterflyAtom,
    _mixed_radix_flow_with_jvp,
    angle_count_per_side_expert,
    coordinate_count,
    make_module,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_sparse_moe_state_conditioned_butterfly_transport_oracle_plan.json"


def load_plan() -> dict:
    return json.loads(PLAN.read_text())


def test_coordinate_accounting_is_exact_and_above_200x() -> None:
    plan = load_plan()
    candidate = plan["candidate"]
    assert angle_count_per_side_expert(768) == 3840
    count = coordinate_count(experts=8, input_width=768, hidden_width=1536)
    assert count == candidate["total_coordinates_per_layer"] == 92160
    assert 18874368 / count == candidate["paired_parameter_compression_ratio"] == 204.8


def test_mixed_radix_flow_preserves_norm_and_jvp() -> None:
    generator = torch.Generator().manual_seed(31)
    value = torch.randn(2, 3, 768, generator=generator)
    tangent = torch.randn(2, 3, 768, generator=generator)
    binary = torch.randn(2, 3, 8, 384, generator=generator) * 0.1
    binary_jvp = torch.randn(2, 3, 8, 384, generator=generator) * 0.03
    cross = torch.randn(2, 3, 3, 256, generator=generator) * 0.1
    cross_jvp = torch.randn(2, 3, 3, 256, generator=generator) * 0.03
    output, output_jvp = _mixed_radix_flow_with_jvp(
        value, tangent, binary, binary_jvp, cross, cross_jvp
    )
    torch.testing.assert_close(output.norm(dim=-1), value.norm(dim=-1), rtol=2e-5, atol=2e-5)
    epsilon = 1e-3
    plus, _ = _mixed_radix_flow_with_jvp(
        value + epsilon * tangent,
        torch.zeros_like(tangent),
        binary + epsilon * binary_jvp,
        torch.zeros_like(binary_jvp),
        cross + epsilon * cross_jvp,
        torch.zeros_like(cross_jvp),
    )
    minus, _ = _mixed_radix_flow_with_jvp(
        value - epsilon * tangent,
        torch.zeros_like(tangent),
        binary - epsilon * binary_jvp,
        torch.zeros_like(binary_jvp),
        cross - epsilon * cross_jvp,
        torch.zeros_like(cross_jvp),
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(output_jvp, finite_difference, rtol=3e-3, atol=3e-3)


def test_candidate_control_have_identical_count_and_distinct_conditioning() -> None:
    plan = load_plan()
    candidate = make_module(plan, 0, "cpu", conditional=True)
    control = make_module(plan, 0, "cpu", conditional=False)
    expected = plan["candidate"]["total_coordinates_per_layer"]
    assert candidate.compact_parameter_count(conditional=True) == expected
    assert control.compact_parameter_count(conditional=False) == expected
    assert candidate.beta == 1.0
    assert control.beta == 0.0
    assert set(candidate.state_dict()) == set(control.state_dict())


def test_module_analytic_jvp_matches_finite_difference() -> None:
    plan = load_plan()
    generator = torch.Generator().manual_seed(37)
    for conditional in (False, True):
        module = StateConditionedButterflyAtom(
            experts=1,
            input_width=768,
            hidden_width=1536,
            tensor_layers=12,
            seed=20261189,
            layer=0,
            device="cpu",
            conditional=conditional,
            beta=1.0,
            feature_shift_scale=0.25,
            raw_angle_initial_tanh=0.125,
        )
        inputs = torch.randn(1, 2, 768, generator=generator)
        directions = torch.randn(1, 2, 768, generator=generator)
        _, analytic = module.function_and_jvp(
            inputs, directions, conditional=conditional
        )
        epsilon = 1e-3
        plus, _ = module.function_and_jvp(
            inputs + epsilon * directions,
            torch.zeros_like(directions),
            conditional=conditional,
        )
        minus, _ = module.function_and_jvp(
            inputs - epsilon * directions,
            torch.zeros_like(directions),
            conditional=conditional,
        )
        finite_difference = (plus - minus) / (2.0 * epsilon)
        torch.testing.assert_close(analytic, finite_difference, rtol=7e-3, atol=5e-4)


def test_all_counted_parameter_groups_receive_finite_gradients() -> None:
    plan = load_plan()
    module = make_module(plan, 0, "cpu", conditional=True)
    inputs = torch.randn(8, 2, 768)
    directions = torch.randn_like(inputs)
    output, output_jvp = module.function_and_jvp(
        inputs, directions, conditional=True
    )
    loss = output.square().mean() + 0.1 * output_jvp.square().mean()
    loss.backward()
    for parameter in module.trainable_parameters(conditional=True):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

