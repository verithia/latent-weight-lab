from argparse import Namespace

import pytest
import torch

from examples.nanogpt.train import (
    apply_scheduled_lr,
    conditioned_output_gate_config_kwargs,
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


def test_conditioned_output_gate_training_config_is_not_dropped() -> None:
    kwargs = conditioned_output_gate_config_kwargs(
        Namespace(
            block_fht_mlp_residual_conditioned_output_gate=True,
            block_fht_mlp_residual_conditioned_output_gate_scale=0.5,
            block_fht_mlp_residual_conditioned_output_gate_layers=[0, 11],
            block_fht_mlp_residual_conditioned_output_gate_bias=False,
            block_fht_mlp_residual_conditioned_output_gate_fixed_basis=True,
            block_fht_mlp_residual_conditioned_output_gate_untied_bases=True,
            block_fht_mlp_residual_conditioned_output_gate_basis_block_size=8,
            block_fht_mlp_residual_conditioned_output_gate_basis_seed=123,
            block_fht_mlp_residual_conditioned_output_gate_update_basis_seed=456,
            block_fht_mlp_residual_conditioned_output_gate_output_basis_seed=789,
            block_fht_mlp_conditioned_output_gate_source="postgelu",
            block_fht_mlp_conditioned_output_gate_projection_seed=101112,
            block_fht_mlp_conditioned_output_gate_rms_epsilon=1e-5,
            block_fht_mlp_postgelu_hidden_self_gate=True,
            block_fht_mlp_postgelu_hidden_self_gate_scale=0.75,
            block_fht_mlp_postgelu_hidden_self_gate_layers=[0],
            block_fht_mlp_postgelu_hidden_self_gate_heads=2,
            block_fht_mlp_postgelu_hidden_self_gate_head_seed_stride=99991,
            block_fht_mlp_postgelu_hidden_self_gate_basis_block_size=4,
            block_fht_mlp_postgelu_hidden_self_gate_condition_basis_seed=13,
            block_fht_mlp_postgelu_hidden_self_gate_update_basis_seed=17,
            block_fht_mlp_postgelu_hidden_self_gate_output_basis_seed=19,
            block_fht_mlp_postgelu_hidden_self_gate_rms_epsilon=2e-5,
        )
    )
    assert kwargs == {
        "block_fht_mlp_residual_conditioned_output_gate": True,
        "block_fht_mlp_residual_conditioned_output_gate_scale": 0.5,
        "block_fht_mlp_residual_conditioned_output_gate_layers": (0, 11),
        "block_fht_mlp_residual_conditioned_output_gate_bias": False,
        "block_fht_mlp_residual_conditioned_output_gate_fixed_basis": True,
        "block_fht_mlp_residual_conditioned_output_gate_untied_bases": True,
        "block_fht_mlp_residual_conditioned_output_gate_basis_block_size": 8,
        "block_fht_mlp_residual_conditioned_output_gate_basis_seed": 123,
        "block_fht_mlp_residual_conditioned_output_gate_update_basis_seed": 456,
        "block_fht_mlp_residual_conditioned_output_gate_output_basis_seed": 789,
        "block_fht_mlp_conditioned_output_gate_source": "postgelu",
        "block_fht_mlp_conditioned_output_gate_projection_seed": 101112,
        "block_fht_mlp_conditioned_output_gate_rms_epsilon": 1e-5,
        "block_fht_mlp_postgelu_hidden_self_gate": True,
        "block_fht_mlp_postgelu_hidden_self_gate_scale": 0.75,
        "block_fht_mlp_postgelu_hidden_self_gate_layers": (0,),
        "block_fht_mlp_postgelu_hidden_self_gate_heads": 2,
        "block_fht_mlp_postgelu_hidden_self_gate_head_seed_stride": 99991,
        "block_fht_mlp_postgelu_hidden_self_gate_basis_block_size": 4,
        "block_fht_mlp_postgelu_hidden_self_gate_condition_basis_seed": 13,
        "block_fht_mlp_postgelu_hidden_self_gate_update_basis_seed": 17,
        "block_fht_mlp_postgelu_hidden_self_gate_output_basis_seed": 19,
        "block_fht_mlp_postgelu_hidden_self_gate_rms_epsilon": 2e-5,
    }


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
