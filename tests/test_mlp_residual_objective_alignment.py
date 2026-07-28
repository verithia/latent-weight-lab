from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_residual_objective_alignment import (
    residual_moment_losses,
    residual_moments,
)


def test_residual_moments_match_expected_scale_and_direction() -> None:
    residual = torch.tensor([[3.0, 4.0], [4.0, 3.0]])
    update = 2.0 * residual

    moments = residual_moments(residual, update)

    torch.testing.assert_close(
        moments["log_rms_ratio"], torch.tensor(2.0).log()
    )
    torch.testing.assert_close(moments["cosine"], torch.tensor(1.0))
    torch.testing.assert_close(
        moments["parallel_energy"], torch.tensor(1.0)
    )


def test_residual_moment_losses_are_zero_at_target_and_differentiable() -> None:
    residual = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    update = torch.tensor(
        [[0.5, -0.25], [0.1, 0.3]], requires_grad=True
    )
    observed = residual_moments(residual, update)
    target = {
        key: value.detach().clone() for key, value in observed.items()
    }
    matched = residual_moment_losses(observed, target)
    assert all(float(value.detach()) == 0.0 for value in matched.values())

    shifted_target = {
        "log_rms_ratio": target["log_rms_ratio"] + 0.2,
        "cosine": target["cosine"] - 0.1,
        "parallel_energy": target["parallel_energy"] + 0.05,
    }
    losses = residual_moment_losses(observed, shifted_target)
    losses["joint"].backward()

    assert float(losses["joint"].detach()) > 0.0
    assert update.grad is not None
    assert torch.isfinite(update.grad).all()
