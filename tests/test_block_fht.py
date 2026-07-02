import torch

from latent_weight_lab.block_fht import BlockFHT, block_fht_slice_torch, sign_word_for


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


def test_reference_function():
    latent = torch.randn(8, requires_grad=True)
    out = block_fht_slice_torch(latent, size=40, layers=2, seed=7, start=5, stop=23)
    assert out.shape == (18,)
    out.square().sum().backward()
    assert latent.grad is not None
