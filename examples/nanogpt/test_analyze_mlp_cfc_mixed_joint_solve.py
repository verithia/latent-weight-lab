from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_cfc_mixed_joint_solve import (
    classify,
    mixed_jvp,
    mixed_vjp,
    solve_mixed_tangent,
)


def test_mixed_jvp_vjp_are_adjoint() -> None:
    torch.manual_seed(17)
    source = torch.randn(5, 8)
    skew_pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    shear_pairs = torch.tensor([[1, 6], [3, 7]])
    coordinates = torch.randn(5)
    cotangent = torch.randn_like(source)
    lhs = (mixed_jvp(source, skew_pairs, shear_pairs, coordinates) * cotangent).sum()
    rhs = (coordinates * mixed_vjp(source, skew_pairs, shear_pairs, cotangent)).sum()
    torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-5)


def test_mixed_solver_recovers_representable_tangent() -> None:
    torch.manual_seed(19)
    source = torch.randn(7, 8)
    skew_pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    shear_pairs = torch.tensor([[1, 6], [3, 7]])
    coordinates = torch.randn(5) * 0.01
    target = mixed_jvp(source, skew_pairs, shear_pairs, coordinates)
    solved, diagnostics = solve_mixed_tangent(
        source,
        target,
        skew_pairs,
        shear_pairs,
        iterations=32,
        damping=1e-8,
    )
    assert float((solved - target).norm() / target.norm()) < 1e-4
    assert diagnostics["coordinates"] == 5


def _rows(mixed_single: float, mixed_hybrid: float) -> list[dict[str, object]]:
    offsets = {
        "baseline": 0.0,
        "production_cfc": -0.001,
        "production_cproj": -0.002,
        "production_joint": -0.003,
        "dense_norm_cfc": -0.002,
        "hybrid_norm_cfc": -0.004,
        "mixed_joint_cfc": mixed_single,
        "hybrid_mixed_joint_cfc": mixed_hybrid,
        "cproj64_only": -0.0021,
        "hybrid_cproj64": -0.0031,
    }
    return [
        {
            "window": f"window_{window}",
            "batch_index": batch,
            "point_id": point,
            "ce": 6.0 + offset,
        }
        for window in (1, 2)
        for batch in range(32)
        for point, offset in offsets.items()
    ]


def test_classify_accepts_reliable_material_gain() -> None:
    result = classify(
        _rows(-0.0016, -0.0036),
        confidence_z=2.576,
        minimum_fraction=0.25,
        mean_fraction=0.4,
    )
    assert result["classification"] == "MIXED_JOINT_CFC_SOLVE_PASSES"


def test_classify_rejects_missing_hybrid_gain() -> None:
    result = classify(
        _rows(-0.0016, -0.003),
        confidence_z=2.576,
        minimum_fraction=0.25,
        mean_fraction=0.4,
    )
    assert result["classification"] == "MIXED_JOINT_CFC_SOLVE_REJECTED"
