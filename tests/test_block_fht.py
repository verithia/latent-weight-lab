import torch
import torch.nn.functional as F

from latent_weight_lab.block_fht import (
    BlockFHT,
    BlockFHTLinear,
    block_fht_slice_torch,
    flush_block_fht_weight_cache,
    prepare_block_fht_weight_cache,
    sign_word_for,
)


def test_sign_word_uses_32_positions():
    bits = sign_word_for(seed=999, block=3, layer=2, word=4)
    signs = [1 if ((bits >> bit) & 1) else -1 for bit in range(32)]
    assert len(signs) == 32
    assert set(signs) <= {-1, 1}


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
