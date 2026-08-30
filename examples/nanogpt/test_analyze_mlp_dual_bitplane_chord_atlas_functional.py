import json
import subprocess
import sys
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_dual_bitplane_chord_atlas_functional import (
    DualBitplaneChordBank,
    acquire_cross_layer_planes,
    decode_binary,
    deployment_accounting,
    encode_binary,
    pack_binary,
    terminal_artifact,
    unpack_binary,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_dual_bitplane_chord_atlas_functional_plan.json"


def test_direct_entrypoint_resolves_repository_package() -> None:
    script = Path(__file__).with_name(
        "analyze_mlp_dual_bitplane_chord_atlas_functional.py"
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


def test_binary_pack_roundtrip_and_decode() -> None:
    codes = torch.tensor(
        [[[-1, 1, -1, 1], [1, 1, -1, -1]], [[1, -1, 1, -1], [-1, 1, 1, -1]]],
        dtype=torch.int8,
    )
    packed = pack_binary(codes)
    assert packed.numel() == 2
    restored = unpack_binary(packed, codes.numel()).reshape_as(codes)
    assert torch.equal(restored, codes)
    scales = torch.tensor([[0.5, 2.0], [3.0, 4.0]], dtype=torch.float16)
    decoded = decode_binary(packed, scales, tuple(codes.shape), device="cpu")
    torch.testing.assert_close(decoded, codes.float() * scales.float()[:, :, None])


def test_exact_h50_accounting_matches_plan() -> None:
    plan = json.loads(PLAN.read_text())
    accounting = deployment_accounting()
    expected = plan["exact_deployment_accounting"]
    assert accounting["total_checkpoint_payload_bytes"] == expected["total_checkpoint_bytes"]
    assert accounting["binary_endpoint_bytes"] == 884_736
    assert accounting["fp16_endpoint_scale_bytes"] == 18_432
    assert accounting["fp16_chord_coordinate_bytes"] == 221_184
    assert accounting["checkpoint_byte_fraction"] < 0.01


def test_cross_layer_acquisition_and_source_identity() -> None:
    generator = torch.Generator().manual_seed(7)
    layers = [0, 1]
    inputs = {layer: torch.randn(8, 4, generator=generator) for layer in layers}
    teacher_u = {layer: torch.randn(6, 4, generator=generator) for layer in layers}
    teacher_v = {layer: torch.randn(6, 4, generator=generator) for layer in layers}
    acquired = acquire_cross_layer_planes(
        inputs,
        teacher_u,
        teacher_v,
        layers,
        atoms_per_layer_per_plane=2,
        device="cpu",
    )
    assert acquired["raw_u"].shape == (2, 4, 4)
    assert acquired["initial_u"].shape == (2, 2, 4)
    assert torch.equal(acquired["source_layers"][0], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(acquired["source_layers"][1], torch.tensor([1, 1, 0, 0]))
    assert torch.equal(acquired["initial_u"].sum(dim=(1, 2)), torch.tensor([4.0, 4.0]))


def test_terminal_payload_and_coordinate_gradient() -> None:
    generator = torch.Generator().manual_seed(11)
    raw_u = torch.randn(2, 8, 4, generator=generator)
    raw_v = torch.randn(2, 8, 4, generator=generator)
    u_codes, u_scales = encode_binary(raw_u)
    v_codes, v_scales = encode_binary(raw_v)
    base_u = u_codes.float() * u_scales.float()[:, :, None]
    base_v = v_codes.float() * v_scales.float()[:, :, None]
    initial_u = torch.randn(2, 2, 8, generator=generator)
    initial_v = torch.randn(2, 2, 8, generator=generator)
    bank = DualBitplaneChordBank(base_u, base_v, initial_u, initial_v, [0, 1])
    inputs = torch.randn(6, 4, generator=generator)
    output, _ = bank.forward_function(0, inputs, None)
    output.square().mean().backward()
    assert bank.coordinate_u.grad is not None and float(bank.coordinate_u.grad.norm()) > 0
    assert bank.coordinate_v.grad is not None and float(bank.coordinate_v.grad.norm()) > 0
    accounting = deployment_accounting(layers=2, width=4, atoms=8, planes=2)
    artifact = terminal_artifact(
        bank, u_codes, v_codes, u_scales, v_scales, accounting
    )
    assert artifact["accounted_payload_bytes"] == accounting["total_checkpoint_payload_bytes"]
    assert artifact["coordinate_u"].dtype == torch.float16
