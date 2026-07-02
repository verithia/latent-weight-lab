import pytest
import torch

from latent_weight_lab.block_fht import BlockFHT, block_fht_slice_torch


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
