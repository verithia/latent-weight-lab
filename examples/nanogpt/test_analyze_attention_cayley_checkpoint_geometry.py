from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_attention_cayley_checkpoint_geometry import (
    cayley_generator,
    frame_drift_metrics,
    low_rank_skew_difference_fro,
    low_rank_skew_singular_values,
    seeded_right_frame,
)


class AttentionCayleyCheckpointGeometryTest(unittest.TestCase):
    def test_identical_right_frame_has_zero_functional_drift(self) -> None:
        initial = seeded_right_frame(16, 3, 19)
        left = torch.randn(16, 3)
        metrics = frame_drift_metrics(
            initial_right=initial,
            final_right=initial.clone(),
            final_left=left,
        )
        self.assertAlmostEqual(metrics["right_subspace_mean_squared_cosine"], 1.0, places=10)
        self.assertAlmostEqual(metrics["right_subspace_projector_distance"], 0.0, places=7)
        self.assertAlmostEqual(metrics["fixed_right_generator_relative_error"], 0.0, places=10)

    def test_in_span_gauge_change_preserves_subspace_but_changes_frame(self) -> None:
        initial = seeded_right_frame(16, 3, 23)
        transform = torch.tensor(
            [[1.0, 0.2, 0.0], [0.0, 1.0, -0.3], [0.1, 0.0, 1.0]]
        )
        final = initial @ transform
        metrics = frame_drift_metrics(
            initial_right=initial,
            final_right=final,
            final_left=torch.randn(16, 3),
        )
        self.assertAlmostEqual(metrics["right_subspace_mean_squared_cosine"], 1.0, places=10)
        self.assertGreater(metrics["right_normalized_relative_drift"], 0.0)

    def test_orthogonal_motion_is_detected(self) -> None:
        initial = torch.eye(8)[:, :2]
        final = torch.eye(8)[:, 2:4]
        metrics = frame_drift_metrics(
            initial_right=initial,
            final_right=final,
            final_left=torch.randn(8, 2),
        )
        self.assertAlmostEqual(metrics["right_subspace_mean_squared_cosine"], 0.0, places=10)
        self.assertAlmostEqual(metrics["right_subspace_projector_distance"], 1.0, places=10)
        self.assertAlmostEqual(metrics["right_subspace_max_angle_degrees"], 90.0, places=6)

    def test_small_skew_calculation_matches_dense_matrix(self) -> None:
        torch.manual_seed(5)
        left = torch.randn(12, 3)
        first = torch.randn(12, 3)
        second = torch.randn(12, 3)
        dense_first = cayley_generator(left, first)
        dense_second = cayley_generator(left, second)
        singular_values = low_rank_skew_singular_values(left, first)
        self.assertAlmostEqual(
            float(singular_values.norm()), float(dense_first.norm()), places=9
        )
        self.assertAlmostEqual(
            low_rank_skew_difference_fro(
                left=left, first_right=first, second_right=second
            ),
            float((dense_first - dense_second).norm()),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
