import pytest
import torch

from latent_weight_lab.block_fht import BlockFHT, BlockFHTLinear, block_fht_slice_torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _check_cuda_matches_reference(latent_size: int, size: int, start: int, stop: int):
    torch.manual_seed(0)
    latent = torch.randn(latent_size, dtype=torch.float32)
    ref_latent = latent.clone().requires_grad_(True)
    ref = block_fht_slice_torch(ref_latent, size=size, layers=1, seed=123, start=start, stop=stop)
    ref.square().sum().backward()

    bfht = BlockFHT(latent.cuda(), size=size, layers=1, seed=123)
    out = bfht.slice(start, stop)
    out.square().sum().backward()
    torch.cuda.synchronize()

    assert torch.allclose(out.detach().cpu(), ref.detach(), atol=2e-6, rtol=2e-6)
    assert torch.allclose(bfht.latent.grad.detach().cpu(), ref_latent.grad, atol=2e-5, rtol=2e-5)


def test_cuda_shared_memory_backend_matches_reference():
    _check_cuda_matches_reference(latent_size=4096, size=8192, start=17, stop=4099)


def test_cuda_global_memory_backend_matches_reference():
    _check_cuda_matches_reference(latent_size=20000, size=65536, start=111, stop=4096)


def test_cuda_scales_to_2_23_block_size():
    bfht = BlockFHT((1 << 23) - 17, size=1 << 23, layers=1, seed=9).cuda()
    out = bfht.slice(12345, 12345 + 1024)
    out.square().mean().backward()
    torch.cuda.synchronize()
    assert out.shape == (1024,)
    assert bfht.latent.grad is not None


def test_cuda_fused_linear_forward_matches_materialized_weight():
    torch.manual_seed(123)
    layer = BlockFHTLinear(7, 5, bias=True, latent_dim=32, layers=2, seed=99).cuda()
    x = torch.randn(11, 7, device="cuda")
    fused = layer.forward_fused(x)
    materialized = layer(x)
    torch.cuda.synchronize()
    assert torch.allclose(fused, materialized, atol=2e-5, rtol=2e-5)


def test_cuda_fused_linear_forward_supports_batched_input():
    torch.manual_seed(321)
    layer = BlockFHTLinear(8, 6, bias=False, latent_dim=32, layers=1, seed=7).cuda()
    x = torch.randn(3, 4, 8, device="cuda")
    fused = layer.forward_fused(x)
    materialized = layer(x)
    torch.cuda.synchronize()
    assert fused.shape == (3, 4, 6)
    assert torch.allclose(fused, materialized, atol=2e-5, rtol=2e-5)


def test_cuda_fused_linear_forward_supports_weight_scale():
    torch.manual_seed(456)
    layer = BlockFHTLinear(8, 6, bias=False, latent_dim=32, layers=2, seed=8).cuda()
    x = torch.randn(5, 8, device="cuda")
    scale = 0.125
    fused = layer.forward_fused(x, weight_scale=scale)
    materialized = torch.nn.functional.linear(x, layer.weight * scale)
    torch.cuda.synchronize()
    assert torch.allclose(fused, materialized, atol=2e-5, rtol=2e-5)


def test_cuda_fused_linear_forward_supports_float16():
    torch.manual_seed(789)
    layer = BlockFHTLinear(8, 6, bias=False, latent_dim=32, layers=2, seed=9).cuda().half()
    x = torch.randn(5, 8, device="cuda", dtype=torch.float16)
    scale = 0.5
    fused = layer.forward_fused(x, weight_scale=scale)
    materialized = torch.nn.functional.linear(x.float(), layer.weight.float() * scale).half()
    torch.cuda.synchronize()
    assert fused.dtype == torch.float16
    assert torch.allclose(fused, materialized, atol=2e-3, rtol=2e-3)


def test_cuda_fused_linear_forward_supports_bfloat16():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 unsupported on this CUDA device")
    torch.manual_seed(987)
    layer = BlockFHTLinear(8, 6, bias=False, latent_dim=32, layers=2, seed=10).cuda().bfloat16()
    x = torch.randn(5, 8, device="cuda", dtype=torch.bfloat16)
    scale = 0.5
    fused = layer.forward_fused(x, weight_scale=scale)
    materialized = torch.nn.functional.linear(x.float(), layer.weight.float() * scale).bfloat16()
    torch.cuda.synchronize()
    assert fused.dtype == torch.bfloat16
    assert torch.allclose(fused, materialized, atol=2e-2, rtol=2e-2)
