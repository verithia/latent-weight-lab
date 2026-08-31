import json
import subprocess
import sys
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_w0_conditioned_block_atlas import (
    BlockAtlasDecoder,
    basis_diagnostics,
    blockify_bundle,
    compact_payload,
    deployment_accounting,
    procedural_blind_blocks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_w0_conditioned_block_atlas_plan.json"


def tiny_decoder(*, linear: bool = False, unpaired: bool = False) -> BlockAtlasDecoder:
    return BlockAtlasDecoder(
        block_width=4, shared_width=12, latent_width=3,
        deployment_layers=12, positions=2, measured_layers=[0],
        components=3, seed=11, linear=linear, unpaired=unpaired,
    )


def test_direct_entrypoint_help() -> None:
    script = Path(__file__).with_name("analyze_mlp_w0_conditioned_block_atlas.py")
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], cwd="/tmp",
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_accounting_matches_frozen_plan() -> None:
    plan = json.loads(PLAN.read_text())
    accounting = deployment_accounting()
    assert accounting["total_fp16_values"] == 368_832
    assert accounting["total_checkpoint_payload_bytes"] == plan["persistent_state"]["total_bytes"]
    assert accounting["checkpoint_byte_fraction"] < 0.01
    assert accounting["persistent_w0_bytes"] == 0
    assert accounting["persistent_row_or_block_code_bytes"] == 0


def test_blockify_preserves_paired_coordinates() -> None:
    bundle = {
        "detector_w0": torch.arange(48.0).reshape(6, 8),
        "write_w0": torch.arange(48.0).reshape(6, 8) + 100,
        "detector_pcs": torch.arange(96.0).reshape(2, 6, 8),
        "write_pcs": torch.arange(96.0).reshape(2, 6, 8) + 200,
    }
    out = blockify_bundle(bundle, block_width=4)
    assert out["detector_w0_blocks"].shape == (12, 4)
    assert out["write_pc_blocks"].shape == (2, 12, 4)
    assert out["block_positions"].tolist() == [0, 1] * 6


def test_zero_state_and_all_paired_gradients() -> None:
    decoder = tiny_decoder()
    detector = torch.randn(10, 4)
    write = torch.randn(10, 4)
    positions = torch.arange(10) % 2
    zero_u, zero_v = decoder.predict(0, detector, write, positions, zero_codes=True)
    assert torch.count_nonzero(zero_u) == 0
    assert torch.count_nonzero(zero_v) == 0
    u, v = decoder.predict(0, detector, write, positions)
    (u.square().mean() + v.square().mean()).backward()
    for name in (
        "a", "c", "d_u", "d_v", "layer_embeddings",
        "position_embeddings", "codes_u",
    ):
        parameter = getattr(decoder, name)
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_linear_control_is_key_independent() -> None:
    decoder = tiny_decoder(linear=True)
    detector = torch.randn(10, 4)
    write = torch.randn(10, 4)
    positions = torch.arange(10) % 2
    first = decoder.predict(0, detector, write, positions)
    second = decoder.predict(0, 9 * detector, -4 * write, 1 - positions)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_blind_blocks_preserve_device() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bundle = {
        "detector_w0_blocks": torch.randn(10, 4, device=device),
        "write_w0_blocks": torch.randn(10, 4, device=device),
    }
    copied = procedural_blind_blocks([bundle], seed=13, device=device)[0]
    assert copied["detector_w0_blocks"].device.type == torch.device(device).type
    assert torch.isfinite(copied["write_w0_blocks"]).all()


def test_compact_payload_excludes_codes_and_w0() -> None:
    decoder = BlockAtlasDecoder(
        block_width=32, shared_width=2048, latent_width=16,
        deployment_layers=12, positions=24, measured_layers=[0, 6, 11],
        components=16, seed=17,
    )
    payload = compact_payload(decoder, deployment_accounting())
    assert payload["accounted_payload_bytes"] == 737_664
    assert "codes_u" not in payload["tensors"]
    assert all("w0" not in key for key in payload["tensors"])


def test_local_bases_are_full_rank() -> None:
    diagnostics = basis_diagnostics(tiny_decoder())
    assert diagnostics["d_u"]["numerical_rank"] == 4
    assert diagnostics["d_v"]["numerical_rank"] == 4
