from __future__ import annotations

import unittest

import torch

from examples.nanogpt.analyze_mlp_state_basis_rate_distortion import (
    common_support_frontier,
    independent_support_frontier,
    minimum_prefix,
)


class StateBasisRateDistortionTest(unittest.TestCase):
    def test_minimum_prefix(self) -> None:
        cumulative = torch.tensor([0.4, 0.7, 0.9, 1.0])
        self.assertEqual(minimum_prefix(cumulative, 0.5), 2)
        self.assertEqual(minimum_prefix(cumulative, 0.9), 3)
        self.assertEqual(minimum_prefix(cumulative, 1.0), 4)

    def test_common_support_uses_variance_weights(self) -> None:
        basis = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        result = common_support_frontier(
            basis, torch.tensor([0.75, 0.25]), [0.5, 0.9]
        )
        self.assertEqual(result[0.5], 1)
        self.assertEqual(result[0.9], 2)

    def test_independent_support_counts_each_basis(self) -> None:
        basis = torch.tensor([[[3.0, 0.0]], [[1.0, 1.0]]])
        result = independent_support_frontier(basis, [0.6])
        self.assertEqual(result[0.6], [1, 2])


if __name__ == "__main__":
    unittest.main()
