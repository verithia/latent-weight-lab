from __future__ import annotations

import torch

from examples.nanogpt.model import GPTConfig, MLP, freeze_non_block_fht


def chart_config(enabled: bool = True) -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        bias=False,
        block_fht_mlp_activation_chart=enabled,
        block_fht_mlp_activation_chart_channel_scale=4.0,
        block_fht_mlp_activation_chart_common_scale=2.0,
        block_fht_mlp_activation_chart_gauge_scale=3.0,
    )


def test_activation_chart_coordinates_are_centered_and_paired() -> None:
    mlp = MLP(chart_config(), layer_id=0)
    assert mlp.activation_chart_channel_log_gain is not None
    assert mlp.activation_chart_common_log_gain is not None
    assert mlp.activation_chart_gauge_log_gain is not None
    with torch.no_grad():
        mlp.activation_chart_channel_log_gain.copy_(
            torch.linspace(-1.0, 1.0, 32)
        )
        mlp.activation_chart_common_log_gain.fill_(0.25)
        mlp.activation_chart_gauge_log_gain.fill_(0.10)
    scales = mlp.activation_chart_log_scales()
    assert scales is not None
    pre, post = scales
    assert torch.allclose(pre.mean(), torch.tensor(0.20), atol=1e-6)
    assert torch.allclose(post.mean(), torch.tensor(0.80), atol=1e-6)
    assert torch.allclose(post - pre, torch.full_like(pre, 0.60))
    assert torch.allclose(
        pre - pre.mean(),
        4.0
        * (
            mlp.activation_chart_channel_log_gain
            - mlp.activation_chart_channel_log_gain.mean()
        ),
    )


def test_zero_activation_chart_is_exact_forward_identity() -> None:
    torch.manual_seed(19)
    plain = MLP(chart_config(enabled=False), layer_id=0)
    charted = MLP(chart_config(enabled=True), layer_id=0)
    charted.load_state_dict(plain.state_dict(), strict=False)
    inputs = torch.randn(3, 5, 8)
    assert torch.equal(plain(inputs), charted(inputs))


def test_freeze_keeps_only_activation_chart_trainable() -> None:
    mlp = MLP(chart_config(enabled=True), layer_id=0)
    freeze_non_block_fht(mlp, train_embeddings=False)
    trainable = {
        name for name, parameter in mlp.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        "activation_chart_channel_log_gain",
        "activation_chart_common_log_gain",
        "activation_chart_gauge_log_gain",
    }


def test_activation_chart_coordinates_receive_gradients() -> None:
    torch.manual_seed(23)
    mlp = MLP(chart_config(enabled=True), layer_id=0)
    mlp(torch.randn(2, 4, 8)).square().mean().backward()
    for parameter in (
        mlp.activation_chart_channel_log_gain,
        mlp.activation_chart_common_log_gain,
        mlp.activation_chart_gauge_log_gain,
    ):
        assert parameter is not None
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
