import torch
import torch.nn.functional as F

from latent_weight_lab.block_fht import (
    BlockFHT,
    BlockFHTLinear,
    block_fht_slice_torch,
    flush_block_fht_weight_cache,
    prepare_block_fht_weight_cache,
    signs_for,
    sign_word_for,
)
from examples.nanogpt.model import freeze_non_block_fht


def test_sign_word_uses_32_positions():
    bits = sign_word_for(seed=999, block=3, layer=2, word=4)
    signs = [1 if ((bits >> bit) & 1) else -1 for bit in range(32)]
    assert len(signs) == 32
    assert set(signs) <= {-1, 1}


def test_vectorized_signs_match_scalar_hash_bits():
    size = 65
    got = signs_for(torch.empty(1), block=3, layer=2, seed=999, block_size=size)
    expected = torch.tensor(
        [1.0 if ((sign_word_for(999, 3, 2, pos >> 5) >> (pos & 31)) & 1) else -1.0 for pos in range(size)]
    )
    assert torch.equal(got, expected)


def test_slice_matches_full_forward_cpu():
    bfht = BlockFHT(7, size=31, layers=2, seed=123)
    sliced = bfht.slice(3, 17)
    full = bfht()
    assert torch.allclose(sliced, full[3:17])


def test_backward_cpu():
    bfht = BlockFHT(7, size=31, layers=2, seed=123)
    loss = bfht.slice(3, 17).square().sum()
    loss.backward()
    assert bfht.latent.grad is not None
    assert bfht.latent.grad.shape == bfht.latent.shape


def test_external_latent_parameter():
    latent = torch.nn.Parameter(torch.randn(8))
    bfht = BlockFHT(latent, size=40, layers=1, seed=5)
    assert bfht.latent is latent


def test_supported_cuda_block_range_metadata():
    small = BlockFHT(32, size=64, layers=1, seed=1)
    large = BlockFHT((1 << 23) - 17, size=1 << 23, layers=1, seed=1)
    assert small.block_size == 32
    assert large.block_size == 1 << 23


def test_reference_function():
    latent = torch.randn(8, requires_grad=True)
    out = block_fht_slice_torch(latent, size=40, layers=2, seed=7, start=5, stop=23)
    assert out.shape == (18,)
    out.square().sum().backward()
    assert latent.grad is not None


def test_block_fht_linear_matches_materialized_weight():
    layer = BlockFHTLinear(5, 3, bias=True, latent_dim=8, layers=2, seed=11)
    x = torch.randn(4, 5)
    out = layer(x)
    expected = F.linear(x, layer.weight, layer.bias)
    assert torch.allclose(out, expected)


def test_block_fht_linear_cached_grad_matches_dynamic():
    torch.manual_seed(123)
    dynamic = BlockFHTLinear(5, 3, bias=True, latent_dim=8, layers=2, seed=11)
    cached = BlockFHTLinear(5, 3, bias=True, latent_dim=8, layers=2, seed=11)
    cached.load_state_dict(dynamic.state_dict())
    x = torch.randn(4, 5)

    dynamic_loss = dynamic(x).square().mean()
    dynamic_loss.backward()

    prepare_block_fht_weight_cache(cached)
    cached_loss = cached(x).square().mean()
    cached_loss.backward()
    flush_block_fht_weight_cache(cached)

    assert torch.allclose(cached_loss, dynamic_loss)
    assert torch.allclose(cached.generator.latent.grad, dynamic.generator.latent.grad, atol=1e-6)
    assert torch.allclose(cached.bias.grad, dynamic.bias.grad, atol=1e-6)


def test_forward_fused_matches_materialized_with_both_gains():
    layer = BlockFHTLinear(5, 3, bias=True, latent_dim=8, layers=2, seed=11, output_gain=True, input_gain=True)
    x = torch.randn(4, 5)
    assert torch.allclose(layer.forward_fused(x), F.linear(x, layer.weight, layer.bias))


def test_freeze_restores_input_gain_trainability():
    layer = BlockFHTLinear(5, 3, latent_dim=8, layers=2, output_gain=True, input_gain=True)
    freeze_non_block_fht(torch.nn.Sequential(layer), train_embeddings=False)
    assert layer.input_gain.requires_grad and layer.output_gain.requires_grad


def test_spectral_zero_correction_matches_same_seed_block_fht():
    base = BlockFHTLinear(8, 12, bias=True, latent_dim=8, layers=2, seed=9)
    structured = BlockFHTLinear(8, 12, bias=True, latent_dim=8, layers=2, seed=9, spectral_rank=2, spectral_out_groups=3, spectral_in_groups=2)
    structured.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(3, 8)
    assert torch.allclose(structured.weight, base.weight)
    assert torch.allclose(structured(x), base(x))


def test_spectral_core_and_group_gains_receive_gradients_and_disable_cache():
    layer = BlockFHTLinear(8, 12, latent_dim=8, layers=2, seed=9, spectral_rank=2, spectral_out_groups=3, spectral_in_groups=2)
    with torch.no_grad():
        layer.spectral_core[0, 0] = 0.1
    layer.materialize_weight_cache()
    assert layer._cached_weight is None
    loss = layer(torch.randn(4, 8)).square().mean()
    loss.backward()
    assert torch.isfinite(layer.spectral_core.grad).all()
    assert torch.isfinite(layer.spectral_log_out_gain.grad).all()
    assert torch.isfinite(layer.spectral_log_in_gain.grad).all()
    freeze_non_block_fht(torch.nn.Sequential(layer), train_embeddings=False)
    assert layer.spectral_core.requires_grad and layer.spectral_log_out_gain.requires_grad and layer.spectral_log_in_gain.requires_grad
