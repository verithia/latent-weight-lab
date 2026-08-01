import torch

from examples.nanogpt.analyze_mlp_cfc_conjugated_general import (
    basis_rows,
    fit_conjugated_chart,
    make_bases,
)


def test_basis_rows_is_exactly_invertible() -> None:
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(16, 7, generator=generator)
    permutations, inverses, signs = make_bases(16, 1, 43)
    transformed = basis_rows(
        values,
        permutations[0],
        inverses[0],
        signs[0],
        block_size=8,
        inverse=False,
    )
    recovered = basis_rows(
        transformed,
        permutations[0],
        inverses[0],
        signs[0],
        block_size=8,
        inverse=True,
    )
    assert torch.allclose(values, recovered, atol=2e-6, rtol=2e-6)


def test_general_stage_recovers_representable_update() -> None:
    generator = torch.Generator().manual_seed(47)
    weight = torch.randn(16, 7, generator=generator)
    permutations, inverses, signs = make_bases(16, 1, 53)
    source = basis_rows(
        weight,
        permutations[0],
        inverses[0],
        signs[0],
        block_size=8,
        inverse=False,
    )
    matrices = torch.randn(8, 2, 2, generator=generator) * 1e-3
    target_basis = (matrices @ source.reshape(8, 2, 7)).reshape_as(source)
    target = basis_rows(
        target_basis,
        permutations[0],
        inverses[0],
        signs[0],
        block_size=8,
        inverse=True,
    )
    fitted, diagnostics = fit_conjugated_chart(
        weight,
        target,
        stages=1,
        seed=53,
        block_size=8,
        family="general",
        damping=1e-8,
    )
    assert diagnostics["coordinates"] == 32
    assert torch.allclose(fitted, target, atol=2e-6, rtol=2e-4)


def test_orthogonal_stage_recovers_representable_update() -> None:
    generator = torch.Generator().manual_seed(59)
    weight = torch.randn(16, 7, generator=generator)
    permutations, inverses, signs = make_bases(16, 1, 61)
    source = basis_rows(
        weight,
        permutations[0],
        inverses[0],
        signs[0],
        block_size=8,
        inverse=False,
    )
    angles = torch.randn(8, generator=generator) * 1e-3
    pairs = source.reshape(8, 2, 7)
    direction = torch.stack((-pairs[:, 1], pairs[:, 0]), dim=1)
    target_basis = (angles[:, None, None] * direction).reshape_as(source)
    target = basis_rows(
        target_basis,
        permutations[0],
        inverses[0],
        signs[0],
        block_size=8,
        inverse=True,
    )
    fitted, diagnostics = fit_conjugated_chart(
        weight,
        target,
        stages=1,
        seed=61,
        block_size=8,
        family="orthogonal",
        damping=1e-8,
    )
    assert diagnostics["coordinates"] == 8
    assert torch.allclose(fitted, target, atol=2e-6, rtol=2e-4)
