from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_attention_polar_spectral_oracle import (
    polar_spectral_metrics,
)


class AttentionPolarSpectralOracleTest(unittest.TestCase):
    def test_diagonal_direction_is_entirely_spectral(self) -> None:
        weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        direction = torch.diag(torch.tensor([1.0, -2.0, 3.0, -4.0]))
        metrics = polar_spectral_metrics(weight, direction, [1, 2, 4])
        self.assertAlmostEqual(metrics["spectral_recovery"], 1.0, places=10)
        self.assertAlmostEqual(
            metrics["top_singular_rank4_recovery"], 1.0, places=10
        )
        self.assertAlmostEqual(
            metrics["oracle_spectral_rank1_recovery"], 16.0 / 30.0, places=10
        )

    def test_off_diagonal_direction_is_orthogonal_orbit_motion(self) -> None:
        weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        direction = torch.zeros_like(weight)
        direction[0, 1] = 2.0
        direction[1, 0] = -1.0
        metrics = polar_spectral_metrics(weight, direction, [2])
        self.assertAlmostEqual(metrics["spectral_recovery"], 0.0, places=10)

    def test_rectangular_column_complement_is_not_spectral(self) -> None:
        weight = torch.tensor(
            [[4.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
        direction = torch.zeros_like(weight)
        direction[2, 0] = 1.0
        metrics = polar_spectral_metrics(weight, direction, [1, 2])
        self.assertAlmostEqual(metrics["spectral_recovery"], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
