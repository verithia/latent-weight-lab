from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_mlp_tangent_drift import (
    energy_capture,
    tangent_pair_metrics,
    temporal_basis,
)


class AnalyzeMLPTangentDriftTest(unittest.TestCase):
    @staticmethod
    def _rows(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        coefficients = torch.tensor(
            [[-2.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 1.0], [2.0, -1.0]]
        )
        return coefficients[:, :1] * first + coefficients[:, 1:] * second

    def test_temporal_basis_recovers_known_plane(self) -> None:
        rows = self._rows(torch.eye(6)[0:1], torch.eye(6)[1:2])
        centered, eigenvalues, basis = temporal_basis(rows, maximum_rank=2)
        self.assertEqual(tuple(basis.shape), (6, 2))
        self.assertGreater(float(eigenvalues[1]), 0.0)
        self.assertAlmostEqual(energy_capture(centered, basis), 1.0, places=6)
        self.assertAlmostEqual(float((basis.T @ basis - torch.eye(2)).abs().max()), 0.0, places=6)

    def test_identical_tangents_have_unit_overlap_and_capture(self) -> None:
        rows = self._rows(torch.eye(6)[0:1], torch.eye(6)[1:2])
        centered, _, basis = temporal_basis(rows, maximum_rank=2)
        metrics = tangent_pair_metrics(
            left_centered=centered,
            left_basis=basis,
            right_rows=rows + 7.0,
            right_centered=centered,
            right_basis=basis,
            rank=2,
        )
        self.assertAlmostEqual(metrics["mean_squared_canonical_cosine"], 1.0, places=6)
        self.assertAlmostEqual(metrics["right_prior_centered_capture"], 1.0, places=6)
        self.assertAlmostEqual(metrics["right_prior_increment_capture"], 1.0, places=6)
        self.assertLess(metrics["maximum_principal_angle_degrees"], 0.05)

    def test_orthogonal_tangents_have_zero_overlap_and_capture(self) -> None:
        left_rows = self._rows(torch.eye(6)[0:1], torch.eye(6)[1:2])
        right_rows = self._rows(torch.eye(6)[2:3], torch.eye(6)[3:4])
        left_centered, _, left_basis = temporal_basis(left_rows, maximum_rank=2)
        right_centered, _, right_basis = temporal_basis(right_rows, maximum_rank=2)
        metrics = tangent_pair_metrics(
            left_centered=left_centered,
            left_basis=left_basis,
            right_rows=right_rows,
            right_centered=right_centered,
            right_basis=right_basis,
            rank=2,
        )
        self.assertAlmostEqual(metrics["mean_squared_canonical_cosine"], 0.0, places=6)
        self.assertAlmostEqual(metrics["right_prior_centered_capture"], 0.0, places=6)
        self.assertAlmostEqual(metrics["right_prior_increment_capture"], 0.0, places=6)
        self.assertAlmostEqual(metrics["maximum_principal_angle_degrees"], 90.0, places=5)


if __name__ == "__main__":
    unittest.main()
