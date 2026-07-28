from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_parameter_trajectory import summarize_parameter


class AnalyzeParameterTrajectoryTest(unittest.TestCase):
    def test_straight_path_is_one_dimensional_and_affine(self) -> None:
        direction = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1
        tensors = [direction * scale for scale in (0.0, 0.25, 0.5, 0.75, 1.0)]
        summary, coordinates, polynomial = summarize_parameter(
            name="transformer.h.2.mlp.c_fc.weight",
            steps=[0, 1, 2, 3, 4],
            tensors=tensors,
            device="cpu",
        )
        self.assertGreater(summary["pc1_energy"], 0.999999)
        self.assertEqual(summary["dimension_99pct"], 1)
        self.assertAlmostEqual(summary["path_length_over_chord"], 1.0, places=6)
        self.assertLess(summary["max_relative_chord_residual"], 1e-6)
        self.assertGreater(summary["mean_consecutive_increment_cosine"], 0.999999)
        self.assertEqual(len(coordinates), 5)
        self.assertGreater(polynomial[0]["r2"], 0.999999)

    def test_curved_path_reports_nonzero_chord_residual_and_turning(self) -> None:
        first = torch.zeros((2, 2))
        tensors = [
            first,
            torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
            torch.tensor([[2.0, 1.0], [0.0, 0.0]]),
        ]
        summary, _, _ = summarize_parameter(
            name="transformer.h.0.mlp.c_proj.weight",
            steps=[0, 1, 2, 3],
            tensors=tensors,
            device="cpu",
        )
        self.assertGreater(summary["max_relative_chord_residual"], 0.1)
        self.assertGreater(summary["median_turn_degrees"], 1.0)
        self.assertGreater(summary["path_length_over_chord"], 1.0)


if __name__ == "__main__":
    unittest.main()
