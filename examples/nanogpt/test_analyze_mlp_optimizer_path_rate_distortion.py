from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_mlp_optimizer_path_rate_distortion import (
    adamw_replay_path,
    cumulative_path,
    normalized_like,
)


class OptimizerPathRateDistortionTest(unittest.TestCase):
    def test_normalized_like_matches_each_reference_norm(self) -> None:
        direction = torch.tensor([[[3.0, 4.0]], [[1.0, 0.0]]])
        reference = torch.tensor([[[6.0, 8.0]], [[0.0, 2.0]]])
        result = normalized_like(direction, reference)
        torch.testing.assert_close(
            result.flatten(1).norm(dim=1), reference.flatten(1).norm(dim=1)
        )

    def test_cumulative_path_uses_intervals_and_learning_rates(self) -> None:
        directions = torch.tensor([[[1.0]], [[2.0]], [[9.0]]])
        result = cumulative_path(
            directions, steps=[0, 2, 5], learning_rates=[0.5, 0.25, 1.0]
        )
        torch.testing.assert_close(result.flatten(), torch.tensor([0.0, 1.0, 2.5]))

    def test_adamw_constant_gradient_has_constant_descent(self) -> None:
        gradient_descent = -torch.ones(3, 1, 1)
        result = adamw_replay_path(
            gradient_descent,
            steps=[0, 1, 3],
            learning_rates=[0.1, 0.1, 0.1],
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
        )
        torch.testing.assert_close(
            result.flatten(), torch.tensor([0.0, -0.1, -0.3]), rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
