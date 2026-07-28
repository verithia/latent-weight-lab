from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_mlp_trajectory_structure import (
    paired_metrics,
    pearson,
    subspace_overlap,
)


class AnalyzeMLPTrajectoryStructureTest(unittest.TestCase):
    def test_correlations_and_subspace_overlap(self) -> None:
        vector = torch.arange(1, 9, dtype=torch.float32)
        self.assertAlmostEqual(pearson(vector, vector * 3), 1.0, places=6)
        basis = torch.eye(8)[:, :3]
        self.assertAlmostEqual(subspace_overlap(basis, basis), 1.0, places=6)
        self.assertAlmostEqual(subspace_overlap(basis, torch.eye(8)[:, 3:6]), 0.0, places=6)

    def test_transposed_pair_has_identical_matrix_and_singular_subspaces(self) -> None:
        generator = torch.Generator().manual_seed(7)
        c_fc = torch.randn((12, 4), generator=generator)
        c_proj = c_fc.T.contiguous()
        metrics = paired_metrics(c_fc, c_proj, ranks=[2, 4])
        self.assertAlmostEqual(metrics["frobenius_cosine_cfc_cproj_transpose"], 1.0, places=6)
        self.assertAlmostEqual(metrics["expansion_channel_delta_norm_pearson"], 1.0, places=6)
        self.assertAlmostEqual(metrics["residual_channel_delta_norm_pearson"], 1.0, places=6)
        self.assertAlmostEqual(metrics["residual_subspace_overlap_rank4"], 1.0, places=5)
        self.assertAlmostEqual(metrics["expansion_subspace_overlap_rank4"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
