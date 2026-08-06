from __future__ import annotations

import torch

from examples.nanogpt.audit_late_cproj_optimizer_state_cross_run import (
    mismatch_metrics,
)


def test_mismatch_metrics_distinguish_exact_and_diverged_paths() -> None:
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    exact = mismatch_metrics(reference.clone(), reference)
    assert exact["bitwise_equal"]
    assert exact["relative_frobenius_difference"] == 0.0
    changed = reference.clone()
    changed[1, 0] += 0.25
    mismatch = mismatch_metrics(changed, reference)
    assert not mismatch["bitwise_equal"]
    assert mismatch["maximum_absolute_difference"] == 0.25
    assert mismatch["mismatched_elements"] == 1
    assert mismatch["relative_frobenius_difference"] > 0.0
