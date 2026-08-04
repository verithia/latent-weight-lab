from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_attention_fixed_right_tangent import (
    coordinate_dot,
    fixed_right_adjoint,
    fixed_right_tangent,
    fixed_frames,
    project_fixed_right_tangent,
)


class AttentionFixedRightTangentTest(unittest.TestCase):
    def test_adjoint_identity(self) -> None:
        torch.manual_seed(3)
        weight = torch.randn(9, 7, dtype=torch.float64)
        input_right = torch.linalg.qr(torch.randn(7, 2, dtype=torch.float64)).Q
        output_right = torch.linalg.qr(torch.randn(9, 2, dtype=torch.float64)).Q
        coordinates = (
            torch.randn(7, 2, dtype=torch.float64),
            torch.randn(9, 2, dtype=torch.float64),
        )
        target = torch.randn_like(weight)
        mapped = fixed_right_tangent(
            weight=weight,
            input_right=input_right,
            output_right=output_right,
            coordinates=coordinates,
        )
        adjoint = fixed_right_adjoint(
            weight=weight,
            input_right=input_right,
            output_right=output_right,
            direction=target,
        )
        torch.testing.assert_close(
            (mapped * target).sum(),
            coordinate_dot(coordinates, adjoint),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_projection_recovers_in_span_target(self) -> None:
        torch.manual_seed(7)
        weight = torch.randn(10, 8, dtype=torch.float64)
        input_right = torch.linalg.qr(torch.randn(8, 2, dtype=torch.float64)).Q
        coordinates = (torch.randn(8, 2, dtype=torch.float64), None)
        target = fixed_right_tangent(
            weight=weight,
            input_right=input_right,
            output_right=None,
            coordinates=coordinates,
        )
        projected, diagnostics = project_fixed_right_tangent(
            weight=weight,
            input_right=input_right,
            output_right=None,
            target=target,
            maximum_iterations=200,
            tolerance=1e-10,
            ridge=0.0,
        )
        torch.testing.assert_close(projected, target, rtol=1e-8, atol=1e-8)
        self.assertTrue(diagnostics["cg_converged"])
        self.assertLess(diagnostics["projection_residual_dot_fraction"], 1e-8)

    def test_weight_derived_frames_are_orthonormal(self) -> None:
        torch.manual_seed(11)
        weight = torch.randn(12, 9, dtype=torch.float64)
        for basis in ("weight_svd", "weight_random_range"):
            input_right, output_right = fixed_frames(
                weight=weight,
                rank=3,
                base_seed=31,
                layer=2,
                target="qk_shared",
                basis=basis,
                power_iterations=1,
            )
            assert input_right is not None and output_right is not None
            torch.testing.assert_close(
                input_right.T @ input_right,
                torch.eye(3, dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )
            torch.testing.assert_close(
                output_right.T @ output_right,
                torch.eye(3, dtype=torch.float64),
                rtol=1e-10,
                atol=1e-10,
            )


if __name__ == "__main__":
    unittest.main()
