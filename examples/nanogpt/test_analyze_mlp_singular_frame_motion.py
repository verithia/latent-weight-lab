from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_mlp_singular_frame_motion import (
    singular_frame_decomposition,
)


class AnalyzeMLPSingularFrameMotionTest(unittest.TestCase):
    def assert_fraction(self, metrics: dict[str, float | int | str], name: str, value: float) -> None:
        self.assertAlmostEqual(float(metrics[name]), value, places=10)

    def test_pure_singular_value_motion(self) -> None:
        base = torch.diag(torch.tensor([4.0, 3.0, 2.0], dtype=torch.float64))
        delta = torch.diag(torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64))
        metrics = singular_frame_decomposition(base, delta)
        self.assert_fraction(metrics, "singular_value_motion_energy_fraction", 1.0)
        self.assert_fraction(metrics, "in_frame_mixing_energy_fraction", 0.0)
        self.assert_fraction(metrics, "subspace_rotation_residual_energy_fraction", 0.0)

    def test_pure_in_frame_mixing(self) -> None:
        base = torch.diag(torch.tensor([4.0, 3.0, 2.0], dtype=torch.float64))
        delta = torch.tensor(
            [[0.0, 0.2, 0.0], [-0.1, 0.0, 0.4], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        metrics = singular_frame_decomposition(base, delta)
        self.assert_fraction(metrics, "singular_value_motion_energy_fraction", 0.0)
        self.assert_fraction(metrics, "in_frame_mixing_energy_fraction", 1.0)
        self.assert_fraction(metrics, "subspace_rotation_residual_energy_fraction", 0.0)

    def test_tall_matrix_left_subspace_rotation(self) -> None:
        base = torch.tensor(
            [[4.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
        delta = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [0.3, -0.4]],
            dtype=torch.float64,
        )
        metrics = singular_frame_decomposition(base, delta)
        self.assert_fraction(metrics, "subspace_rotation_residual_energy_fraction", 1.0)
        self.assertEqual(
            metrics["residual_interpretation"],
            "left_expansion_output_subspace_rotation",
        )

    def test_wide_matrix_right_rowspace_rotation(self) -> None:
        base = torch.tensor(
            [[4.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float64,
        )
        delta = torch.tensor(
            [[0.0, 0.0, 0.3], [0.0, 0.0, -0.4]],
            dtype=torch.float64,
        )
        metrics = singular_frame_decomposition(base, delta)
        self.assert_fraction(metrics, "subspace_rotation_residual_energy_fraction", 1.0)
        self.assertEqual(
            metrics["residual_interpretation"],
            "right_expansion_input_rowspace_rotation",
        )

    def test_mixed_motion_has_exact_energy_and_reconstruction_closure(self) -> None:
        generator = torch.Generator().manual_seed(20260729)
        base = torch.randn(7, 4, generator=generator, dtype=torch.float64)
        delta = torch.randn(7, 4, generator=generator, dtype=torch.float64)
        metrics = singular_frame_decomposition(base, delta)
        fractions = sum(
            float(metrics[name])
            for name in (
                "singular_value_motion_energy_fraction",
                "in_frame_mixing_energy_fraction",
                "subspace_rotation_residual_energy_fraction",
            )
        )
        self.assertAlmostEqual(fractions, 1.0, places=10)
        self.assertLess(float(metrics["component_energy_relative_error"]), 1e-12)
        self.assertLess(float(metrics["reconstruction_relative_error"]), 1e-12)
        self.assertLess(float(metrics["maximum_component_absolute_cosine"]), 1e-12)


if __name__ == "__main__":
    unittest.main()
