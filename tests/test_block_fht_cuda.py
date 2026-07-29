import pytest
import torch

from latent_weight_lab.block_fht import (
    BlockFHT,
    BlockFHTLinear,
    _fixed_basis_transform_torch,
    block_fht_slice_torch,
    fixed_basis_transform,
    postgelu_multihead_mix,
)


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


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("shared_input", [False, True])
@pytest.mark.parametrize("basis_block_size", [8, 256])
def test_cuda_fixed_basis_transform_matches_reference_and_gradient(
    inverse: bool,
    shared_input: bool,
    basis_block_size: int,
) -> None:
    torch.manual_seed(1234)
    bases = 2
    tokens = 7
    width = 512 if basis_block_size == 256 else 32
    permutations = torch.stack(
        [torch.randperm(width) for _ in range(bases)]
    )
    signs = torch.randint(0, 2, (bases, width)).float().mul_(2).sub_(1)
    shape = (tokens, width) if shared_input else (bases, tokens, width)
    values = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    reference_values = values.detach().clone().requires_grad_(True)
    reference = _fixed_basis_transform_torch(
        reference_values,
        permutations,
        signs,
        basis_block_size,
        inverse=inverse,
        shared_input=shared_input,
    )
    actual_values = values.detach().cuda().requires_grad_(True)
    actual = fixed_basis_transform(
        actual_values,
        permutations.cuda(),
        signs.cuda(),
        basis_block_size,
        inverse=inverse,
        shared_input=shared_input,
    )
    gradient = torch.randn_like(reference)
    reference.backward(gradient)
    actual.backward(gradient.cuda())
    torch.cuda.synchronize()

    assert torch.allclose(
        actual.detach().cpu(), reference.detach(), atol=2e-6, rtol=2e-6
    )
    assert torch.allclose(
        actual_values.grad.detach().cpu(),
        reference_values.grad,
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_postgelu_multihead_mix_matches_eager(dtype: torch.dtype) -> None:
    torch.manual_seed(123)
    heads, tokens, width, block_size = 2, 7, 32, 8
    permutations = [
        torch.stack([torch.randperm(width) for _ in range(heads)])
        for _ in range(3)
    ]
    signs = [
        torch.randint(0, 2, (heads, width)).float().mul_(2).sub_(1)
        for _ in range(3)
    ]
    activated = torch.randn(tokens, width, dtype=dtype, requires_grad=True)
    condition = torch.randn(tokens, width, dtype=dtype, requires_grad=True)
    slope = torch.randn(heads, width, dtype=torch.float32, requires_grad=True)
    reference_activated = activated.detach().clone().requires_grad_(True)
    reference_condition = condition.detach().clone().requires_grad_(True)
    reference_slope = slope.detach().clone().requires_grad_(True)
    condition_spectral = fixed_basis_transform(
        reference_condition.cuda(),
        permutations[0].cuda(),
        signs[0].cuda(),
        block_size,
        shared_input=True,
    )
    update_spectral = fixed_basis_transform(
        reference_activated.cuda(),
        permutations[1].cuda(),
        signs[1].cuda(),
        block_size,
        shared_input=True,
    )
    correction = (
        update_spectral
        * condition_spectral
        * reference_slope.cuda().to(dtype=dtype)[:, None, :]
    )
    transformed = fixed_basis_transform(
        correction,
        permutations[2].cuda(),
        signs[2].cuda(),
        block_size,
        inverse=True,
        shared_input=False,
    )
    reference = reference_activated.cuda() + transformed.sum(dim=0)
    actual_activated = activated.detach().cuda().requires_grad_(True)
    actual_condition = condition.detach().cuda().requires_grad_(True)
    actual_slope = slope.detach().cuda().requires_grad_(True)
    actual = postgelu_multihead_mix(
        actual_activated,
        actual_condition,
        actual_slope,
        permutations[0].cuda(),
        signs[0].cuda(),
        permutations[1].cuda(),
        signs[1].cuda(),
        permutations[2].cuda(),
        signs[2].cuda(),
        block_size,
        1.0,
    )
    gradient = torch.randn_like(reference)
    reference.backward(gradient)
    actual.backward(gradient)
    atol = 3e-2 if dtype == torch.bfloat16 else 2e-5
    rtol = 3e-2 if dtype == torch.bfloat16 else 2e-5
    assert torch.allclose(actual, reference, atol=atol, rtol=rtol)
    assert torch.allclose(
        actual_activated.grad,
        reference_activated.grad.cuda(),
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        actual_condition.grad,
        reference_condition.grad.cuda(),
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        actual_slope.grad,
        reference_slope.grad.cuda(),
        atol=atol,
        rtol=rtol,
    )
