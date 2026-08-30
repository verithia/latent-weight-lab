import subprocess
import sys
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_global_ternary_private_rank5_functional import (
    GlobalTernaryPrivateRankBank,
    decode_ternary,
    deployment_accounting,
    encode_ternary,
    pack_ternary_2bit,
    terminal_artifact,
    unpack_ternary_2bit,
)


def test_direct_entrypoint_resolves_repository_package() -> None:
    script = Path(__file__).with_name(
        "analyze_mlp_global_ternary_private_rank5_functional.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_ternary_pack_roundtrip_and_decode() -> None:
    codes = torch.tensor(
        [[-1, 0, 1, -1], [1, 1, 0, -1]], dtype=torch.int8
    )
    packed = pack_ternary_2bit(codes)
    assert packed.numel() == 2
    restored = unpack_ternary_2bit(packed, codes.numel()).reshape_as(codes)
    assert torch.equal(restored, codes)
    scales = torch.tensor([0.5, 2.0], dtype=torch.float16)
    decoded = decode_ternary(
        packed, scales, tuple(codes.shape), device="cpu"
    )
    assert torch.equal(decoded, codes.float() * scales.float()[:, None])


def test_encode_ternary_is_finite_and_bounded() -> None:
    value = torch.tensor(
        [[-2.0, -0.1, 0.2, 1.0], [0.0, 0.0, 0.0, 0.0]]
    )
    codes, scales = encode_ternary(value)
    assert int(codes.min()) >= -1
    assert int(codes.max()) <= 1
    assert torch.isfinite(scales).all()


def test_exact_h48_accounting() -> None:
    accounting = deployment_accounting(1408, 12, 768, 5)
    assert accounting["ternary_atom_values"] == 2_162_688
    assert accounting["ternary_atom_bytes"] == 540_672
    assert accounting["fp16_scale_bytes"] == 5_632
    assert accounting["fp16_private_factor_values"] == 261_120
    assert accounting["fp16_private_factor_bytes"] == 522_240
    assert accounting["fp16_layer_gain_bytes"] == 33_792
    assert accounting["total_checkpoint_payload_bytes"] == 1_102_336
    assert accounting["continuous_coordinate_values"] == 278_016
    assert accounting["cached_all_layer_fp16_endpoint_bytes"] == 51_904_512


def test_zero_transport_starts_at_global_bank() -> None:
    generator = torch.Generator().manual_seed(7)
    base_u = torch.randn(8, 4, generator=generator)
    base_v = torch.randn(8, 4, generator=generator)
    source = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    bank = GlobalTernaryPrivateRankBank(
        base_u, base_v, source, [0, 1], 2, seed=11
    )
    for layer in (0, 1):
        local_u, local_v, _gain = bank.factors(layer)
        assert torch.equal(local_u, base_u)
        assert torch.equal(local_v, base_v)


def test_terminal_artifact_payload_matches_accounting() -> None:
    generator = torch.Generator().manual_seed(13)
    raw_u = torch.randn(8, 4, generator=generator)
    raw_v = torch.randn(8, 4, generator=generator)
    u_codes, u_scales = encode_ternary(raw_u)
    v_codes, v_scales = encode_ternary(raw_v)
    base_u = u_codes.float() * u_scales.float()[:, None]
    base_v = v_codes.float() * v_scales.float()[:, None]
    source = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    bank = GlobalTernaryPrivateRankBank(
        base_u, base_v, source, [0, 1], 2, seed=17
    )
    accounting = deployment_accounting(8, 2, 4, 2)
    artifact = terminal_artifact(
        bank, u_codes, v_codes, u_scales, v_scales, accounting
    )
    assert artifact["accounted_payload_bytes"] == accounting[
        "total_checkpoint_payload_bytes"
    ]
    assert artifact["u_shape"] == [8, 4]
    assert artifact["a_u"].dtype == torch.float16


def test_private_transport_has_nonzero_initial_gradient() -> None:
    generator = torch.Generator().manual_seed(19)
    base_u = torch.randn(8, 4, generator=generator)
    base_v = torch.randn(8, 4, generator=generator)
    source = torch.zeros(8, dtype=torch.long)
    bank = GlobalTernaryPrivateRankBank(
        base_u, base_v, source, [0], 2, seed=23
    )
    inputs = torch.randn(6, 4, generator=generator)
    output, _ = bank.forward_function(0, inputs, None)
    output.square().mean().backward()
    assert bank.b_u.grad is not None and float(bank.b_u.grad.norm()) > 0
    assert bank.b_v.grad is not None and float(bank.b_v.grad.norm()) > 0
    assert torch.isfinite(bank.b_u.grad).all()
    assert torch.isfinite(bank.b_v.grad).all()
