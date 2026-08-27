from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_nonfht_basis import (
    AdditiveSinusoidalCoordinateField,
    DiagonalToeplitzDiagonal,
    LiveTensorNetwork,
    LearnedSparseExpander,
    MATRIX_COLUMN_MODES,
    MATRIX_ROW_MODES,
    SinusoidalCoordinateField,
    cg_project,
    coordinate_vjp,
)


def finite_difference(module, direction: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-3
    originals = [tensor.detach().clone() for tensor in module.coordinate_tensors]
    offset = 0
    with torch.no_grad():
        for tensor in module.coordinate_tensors:
            count = tensor.numel()
            tensor.add_(direction[offset : offset + count].reshape_as(tensor), alpha=epsilon)
            offset += count
        plus = module.weight().clone()
        for tensor, original in zip(module.coordinate_tensors, originals, strict=True):
            tensor.copy_(original)
        offset = 0
        for tensor in module.coordinate_tensors:
            count = tensor.numel()
            tensor.add_(direction[offset : offset + count].reshape_as(tensor), alpha=-epsilon)
            offset += count
        minus = module.weight().clone()
        for tensor, original in zip(module.coordinate_tensors, originals, strict=True):
            tensor.copy_(original)
    return (plus - minus) / (2.0 * epsilon)


def check_jvp_and_adjoint(module) -> None:
    torch.manual_seed(4)
    with torch.no_grad():
        for coordinate in module.coordinate_tensors:
            coordinate.normal_(std=0.02)
    direction = torch.randn(module.trainable_scalar_count)
    analytic = module.jvp(direction)
    numerical = finite_difference(module, direction)
    torch.testing.assert_close(analytic, numerical, rtol=5e-3, atol=8e-4)
    target = torch.randn_like(analytic)
    torch.testing.assert_close(
        torch.sum(analytic * target),
        torch.dot(direction, coordinate_vjp(module, target)),
        rtol=3e-5,
        atol=3e-5,
    )


def test_dtd_state_and_jvp() -> None:
    module = DiagonalToeplitzDiagonal(5, 7, branches=3, seed=11)
    assert module.trainable_scalar_count == 3 * (5 + 7 + 11)
    check_jvp_and_adjoint(module)


def test_expander_state_and_jvp() -> None:
    module = LearnedSparseExpander(
        5,
        7,
        padded_features=8,
        depth=3,
        group_size=2,
        seed=13,
    )
    assert module.trainable_scalar_count == 3 * (8 // 2 // 2) * 4 + 7
    check_jvp_and_adjoint(module)


def test_known_tangent_is_recovered() -> None:
    module = DiagonalToeplitzDiagonal(4, 6, branches=2, seed=17)
    direction = torch.randn(module.trainable_scalar_count)
    target = module.jvp(direction)
    result = cg_project(
        module,
        target,
        maximum_iterations=128,
        relative_tolerance=1e-8,
        damping_ratio=1e-9,
    )
    assert result["cg_projection_capture"] > 0.999


def test_live_tensor_network_jvp_and_adjoint() -> None:
    for topology, bond in (("open", 3), ("ring", 2)):
        module = LiveTensorNetwork(
            4,
            4,
            row_modes=(2, 2),
            column_modes=(2, 2),
            topology=topology,
            bond=bond,
            seed=19,
        )
        check_jvp_and_adjoint(module)
    transposed = LiveTensorNetwork(
        6,
        4,
        row_modes=(2, 3),
        column_modes=(2, 2),
        topology="open",
        bond=3,
        seed=29,
    )
    assert transposed.weight().shape == (4, 6)
    check_jvp_and_adjoint(transposed)


def test_live_tensor_network_known_tangent_is_recovered() -> None:
    module = LiveTensorNetwork(
        4,
        4,
        row_modes=(2, 2),
        column_modes=(2, 2),
        topology="ring",
        bond=2,
        seed=23,
    )
    direction = torch.randn(module.trainable_scalar_count)
    target = module.jvp(direction)
    result = cg_project(
        module,
        target,
        maximum_iterations=256,
        relative_tolerance=1e-8,
        damping_ratio=1e-9,
    )
    assert result["cg_projection_capture"] > 0.999


def test_sinusoidal_coordinate_field_jvp_and_adjoint() -> None:
    module = SinusoidalCoordinateField(5, 7, rank=3, seed=31)
    assert module.trainable_scalar_count == (5 + 7) * 3
    check_jvp_and_adjoint(module)


def test_sinusoidal_coordinate_known_tangent_is_recovered() -> None:
    module = SinusoidalCoordinateField(4, 6, rank=2, seed=37)
    direction = torch.randn(module.trainable_scalar_count)
    target = module.jvp(direction)
    result = cg_project(
        module,
        target,
        maximum_iterations=256,
        relative_tolerance=1e-8,
        damping_ratio=1e-9,
    )
    assert result["cg_projection_capture"] > 0.999


def test_additive_sinusoidal_coordinate_field_jvp_and_adjoint() -> None:
    module = AdditiveSinusoidalCoordinateField(5, 7, rank=3, seed=41)
    assert module.trainable_scalar_count == (5 + 7) * 3
    check_jvp_and_adjoint(module)


def test_additive_sinusoidal_coordinate_known_tangent_is_recovered() -> None:
    module = AdditiveSinusoidalCoordinateField(4, 6, rank=2, seed=43)
    direction = torch.randn(module.trainable_scalar_count)
    target = module.jvp(direction)
    result = cg_project(
        module,
        target,
        maximum_iterations=256,
        relative_tolerance=1e-8,
        damping_ratio=1e-9,
    )
    assert result["cg_projection_capture"] > 0.999


def test_registered_full_size_budgets() -> None:
    dtd = DiagonalToeplitzDiagonal(768, 3072, branches=3, seed=1)
    expander_fc = LearnedSparseExpander(
        768,
        3072,
        padded_features=4096,
        depth=10,
        group_size=4,
        seed=2,
    )
    expander_proj = LearnedSparseExpander(
        3072,
        768,
        padded_features=4096,
        depth=10,
        group_size=4,
        seed=3,
    )
    dense = 3072 * 768
    assert dtd.trainable_scalar_count == 23037
    assert expander_fc.trainable_scalar_count == 23552
    assert expander_proj.trainable_scalar_count == 21248
    assert max(
        dtd.trainable_scalar_count,
        expander_fc.trainable_scalar_count,
        expander_proj.trainable_scalar_count,
    ) <= dense // 100
    tt = LiveTensorNetwork(
        768,
        3072,
        row_modes=MATRIX_ROW_MODES,
        column_modes=MATRIX_COLUMN_MODES,
        topology="open",
        bond=24,
        seed=5,
    )
    tensor_ring = LiveTensorNetwork(
        3072,
        768,
        row_modes=MATRIX_ROW_MODES,
        column_modes=MATRIX_COLUMN_MODES,
        topology="ring",
        bond=17,
        seed=7,
    )
    assert tt.trainable_scalar_count == 23521
    assert tensor_ring.trainable_scalar_count == 22253
    assert tt.trainable_scalar_count <= dense // 100
    assert tensor_ring.trainable_scalar_count <= dense // 100
    sinusoidal = SinusoidalCoordinateField(
        768, 3072, rank=6, seed=11
    )
    additive_sinusoidal = AdditiveSinusoidalCoordinateField(
        768, 3072, rank=6, seed=13
    )
    assert sinusoidal.trainable_scalar_count == 23040
    assert additive_sinusoidal.trainable_scalar_count == 23040
    assert sinusoidal.trainable_scalar_count <= dense // 100
    assert additive_sinusoidal.trainable_scalar_count <= dense // 100
