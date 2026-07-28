import pytest
import torch

from examples.nanogpt.train import (
    apply_scheduled_lr,
    freeze_mlp_hidden_chart_gradients,
    schedule_mlp_cproj_chart_gradients,
)


def test_scheduled_lr_respects_adamw_group_scale_and_default_scale_one():
    optimizer = torch.optim.AdamW([
        {"params": [torch.nn.Parameter(torch.ones(()))], "lr_scale": 0.2},
        {"params": [torch.nn.Parameter(torch.ones(()))]},
    ], lr=1.0)
    apply_scheduled_lr(optimizer, 0.0024)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.00048)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.0024)


class _ChartedMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_block_rotation = torch.nn.Linear(2, 2, bias=False)
        self.hidden_log_gain = torch.nn.Parameter(torch.zeros(2))
        self.output_block_rotation = torch.nn.Linear(2, 2, bias=False)
        self.residual_output_log_gain = torch.nn.Parameter(torch.zeros(2))
        self.c_proj = torch.nn.Module()
        self.c_proj.generator = torch.nn.Module()
        self.c_proj.generator.latent = torch.nn.Parameter(torch.zeros(2, 2))
        self.other = torch.nn.Parameter(torch.zeros(2))


def test_hidden_chart_gradient_freeze_preserves_other_chart_gradients():
    model = _ChartedMLP()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    assert (
        freeze_mlp_hidden_chart_gradients(
            model,
            iter_num=59,
            stop_iter=60,
        )
        == 0
    )
    assert all(parameter.grad is not None for parameter in model.parameters())

    frozen = freeze_mlp_hidden_chart_gradients(
        model,
        iter_num=60,
        stop_iter=60,
    )
    assert frozen == 2
    assert model.hidden_block_rotation.weight.grad is None
    assert model.hidden_log_gain.grad is None
    assert model.residual_output_log_gain.grad is not None


def test_delayed_chart_then_fixed_base_gradient_schedule() -> None:
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.mlp = _ChartedMLP()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    held, frozen = schedule_mlp_cproj_chart_gradients(
        model,
        iter_num=59,
        start_iter=60,
        freeze_base_at_start=True,
    )
    assert held == 4
    assert frozen == 0
    assert model.block.mlp.hidden_block_rotation.weight.grad is None
    assert model.block.mlp.hidden_log_gain.grad is None
    assert model.block.mlp.output_block_rotation.weight.grad is None
    assert model.block.mlp.residual_output_log_gain.grad is None
    assert model.block.mlp.c_proj.generator.latent.grad is not None
    assert model.block.mlp.other.grad is not None

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    held, frozen = schedule_mlp_cproj_chart_gradients(
        model,
        iter_num=60,
        start_iter=60,
        freeze_base_at_start=True,
    )
    assert held == 0
    assert frozen == 1
    assert model.block.mlp.hidden_block_rotation.weight.grad is not None
    assert model.block.mlp.residual_output_log_gain.grad is not None
    assert model.block.mlp.c_proj.generator.latent.grad is None
    assert model.block.mlp.other.grad is not None
