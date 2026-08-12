import json
from pathlib import Path

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_sparse_moe_paired_sparse_write_chart_oracle import (
    coordinate_count,
    identity_permutation,
    make_module,
    permute_atoms,
    support_indices,
)
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_sparse_moe_paired_sparse_write_chart_oracle_plan.json"


def load_plan() -> dict:
    return json.loads(PLAN.read_text())


def test_exact_full_mlp_accounting_exceeds_200x() -> None:
    plan = load_plan()
    count = coordinate_count(
        tensor_layers=12, experts=8, hidden_width=1536, padded_width=2048
    )
    assert count["cfc"] == 540672
    assert count["sparse_write"] == 442368
    assert count["output_flow"] == 46080
    assert count["compact"] == 1029120
    assert count["dense"] == 226492416
    assert count["compression"] == plan["candidate"]["paired_parameter_compression_ratio"]
    assert count["compression"] > 200.0


def test_support_is_collision_free_and_exactly_balanced() -> None:
    support = support_indices()
    assert support.shape == (1536, 3)
    assert all(len(set(row)) == 3 for row in support.tolist())
    for stream in range(3):
        assert torch.equal(
            torch.bincount(support[:, stream], minlength=768),
            torch.full((768,), 2, dtype=torch.long),
        )


def test_candidate_control_have_equal_state_and_count() -> None:
    plan = load_plan()
    candidate = make_module(plan, 0, "cpu", paired=True)
    control = make_module(plan, 0, "cpu", paired=False)
    expected = plan["candidate"]["total_coordinates_all_layers"] // 12
    assert candidate.compact_parameter_count(paired=True) == expected == 85760
    assert control.compact_parameter_count(paired=False) == expected
    assert set(candidate.state_dict()) == set(control.state_dict())
    for name in candidate.state_dict():
        torch.testing.assert_close(candidate.state_dict()[name], control.state_dict()[name])


def test_dense_complete_neuron_permutation_is_function_invariant() -> None:
    generator = torch.Generator().manual_seed(43)
    inputs = torch.randn(2, 5, 7, generator=generator)
    directions = torch.randn_like(inputs)
    c_fc = torch.randn(2, 11, 7, generator=generator)
    c_proj_t = torch.randn(2, 11, 7, generator=generator)
    permutation = torch.stack((torch.randperm(11, generator=generator), torch.randperm(11, generator=generator)))
    permuted_c_fc = permute_atoms(c_fc, permutation)
    permuted_c_proj_t = permute_atoms(c_proj_t, permutation)
    original = dense_function_and_jvp(inputs, directions, c_fc, c_proj_t)
    permuted = dense_function_and_jvp(
        inputs, directions, permuted_c_fc, permuted_c_proj_t
    )
    torch.testing.assert_close(original[0], permuted[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(original[1], permuted[1], rtol=1e-5, atol=1e-5)


def test_analytic_jvp_and_generated_write_columns_match_function() -> None:
    plan = load_plan()
    module = make_module(plan, 0, "cpu", paired=True)
    generator = torch.Generator().manual_seed(47)
    with torch.no_grad():
        module.spectral_1.normal_(generator=generator, std=0.03)
        module.spectral_2.normal_(generator=generator, std=0.03)
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
        inputs + epsilon * directions, torch.zeros_like(directions),
        conditional=True,
    )
    minus, _ = module.function_and_jvp(
        inputs - epsilon * directions, torch.zeros_like(directions),
        conditional=True,
    )
    torch.testing.assert_close(
        analytic, (plus - minus) / (2.0 * epsilon), rtol=6e-3, atol=5e-4
    )


def test_every_counted_group_receives_finite_gradient() -> None:
    plan = load_plan()
    module = make_module(plan, 0, "cpu", paired=True)
    inputs = torch.randn(8, 2, 768)
    directions = torch.randn_like(inputs)
    output, output_jvp, hidden = module.function_details(
        inputs, directions, paired=True
    )
    writes = module.generated_write_columns()
    loss = output.square().mean() + 0.1 * output_jvp.square().mean()
    loss = loss + 0.25 * hidden.square().mean() + 0.25 * writes.square().mean()
    loss.backward()
    for parameter in module.trainable_parameters(paired=True):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
