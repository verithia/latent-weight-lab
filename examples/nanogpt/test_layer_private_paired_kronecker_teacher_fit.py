from __future__ import annotations

import torch

from examples.nanogpt.analyze_layer_private_paired_kronecker_teacher_fit import (
    FC_SHAPE,
    PairedKroneckerMLP,
    classify,
    materialize,
    randomized_kronecker_svd,
)


def test_materialize_matches_torch_kron() -> None:
    generator = torch.Generator().manual_seed(7)
    group = torch.randn(3, 4, 5, generator=generator)
    channel = torch.randn(3, 2, 3, generator=generator)
    expected = sum((torch.kron(group[r], channel[r]) for r in range(3)), torch.zeros(8, 15))
    torch.testing.assert_close(materialize(group, channel), expected)


def test_randomized_svd_recovers_synthetic_kronecker_rank() -> None:
    generator = torch.Generator().manual_seed(11)
    group = torch.randn(2, 3, 5, generator=generator)
    channel = torch.randn(2, 4, 2, generator=generator)
    target = materialize(group, channel)
    from examples.nanogpt.analyze_layer_private_paired_kronecker_teacher_fit import MatrixShape
    shape = MatrixShape(3, 4, 5, 2)
    fitted_group, fitted_channel = randomized_kronecker_svd(target, shape, 2, seed=13, niter=8)
    torch.testing.assert_close(materialize(fitted_group, fitted_channel), target, rtol=2e-4, atol=2e-4)


def test_registered_parameter_count_and_forward() -> None:
    per_term = FC_SHAPE.outer_out * FC_SHAPE.outer_in + FC_SHAPE.inner_out * FC_SHAPE.inner_in
    assert per_term == 3328
    assert 12 * 2 * 16 * per_term == 1_277_952
    group_fc = torch.randn(1, 2, 1)
    channel_fc = torch.randn(1, 2, 2)
    group_proj = torch.randn(1, 1, 2)
    channel_proj = torch.randn(1, 2, 2)
    module = PairedKroneckerMLP(fc_group=group_fc, fc_channel=channel_fc, proj_group=group_proj, proj_channel=channel_proj)
    assert module(torch.randn(5, 2)).shape == (5, 2)


def test_classification_requires_all_gates() -> None:
    summary = {"output": {"mean_explained_target_energy": 0.95, "minimum_explained_target_energy": 0.8}, "input_jvp": {"mean_explained_target_energy": 0.85, "minimum_explained_target_energy": 0.6}}
    gates = {"minimum_mean_output_recovery": 0.9, "minimum_worst_output_recovery": 0.75, "minimum_mean_input_jvp_recovery": 0.8, "minimum_worst_input_jvp_recovery": 0.5, "maximum_fixed_validation_cross_entropy_gap": 0.05}
    assert classify(summary, 0.04, True, gates)
    assert not classify(summary, 0.06, True, gates)
    assert not classify(summary, 0.0, False, gates)
