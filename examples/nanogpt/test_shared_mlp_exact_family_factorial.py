from __future__ import annotations

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_shared_mlp_exact_family_factorial import (
    WeightMLP,
    classify_factorial,
)


def test_weight_mlp_matches_explicit_function() -> None:
    c_fc = torch.randn(8, 4)
    c_proj = torch.randn(4, 8)
    values = torch.randn(5, 4)
    expected = F.linear(F.gelu(F.linear(values, c_fc)), c_proj)
    torch.testing.assert_close(WeightMLP(c_fc, c_proj)(values), expected)


def test_factorial_classification_is_exhaustive() -> None:
    assert classify_factorial(False, True, False) == "C_FC_RESTRICTION"
    assert classify_factorial(True, False, False) == "C_PROJ_RESTRICTION"
    assert classify_factorial(False, False, False) == "BILATERAL_RESTRICTION"
    assert classify_factorial(True, True, False) == "NONLINEAR_INTERACTION_RESTRICTION"
    assert classify_factorial(True, True, True) == "NO_LOCALIZED_RESTRICTION"
