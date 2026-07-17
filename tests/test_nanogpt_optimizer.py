import pytest
import torch

from examples.nanogpt.train import apply_scheduled_lr


def test_scheduled_lr_respects_adamw_group_scale_and_default_scale_one():
    optimizer = torch.optim.AdamW([
        {"params": [torch.nn.Parameter(torch.ones(()))], "lr_scale": 0.2},
        {"params": [torch.nn.Parameter(torch.ones(()))]},
    ], lr=1.0)
    apply_scheduled_lr(optimizer, 0.0024)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.00048)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.0024)
