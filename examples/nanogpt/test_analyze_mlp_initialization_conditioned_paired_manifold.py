import json
import subprocess
import sys
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_initialization_conditioned_paired_manifold import (
    PairedNeuronDecoder,
    compact_payload,
    deployment_accounting,
    paired_displacement_pcs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_initialization_conditioned_paired_neuron_manifold_plan.json"


def test_direct_entrypoint_help() -> None:
    script = Path(__file__).with_name(
        "analyze_mlp_initialization_conditioned_paired_manifold.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd="/tmp", capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_accounting_matches_frozen_plan() -> None:
    plan = json.loads(PLAN.read_text())
    accounting = deployment_accounting()
    expected = plan["persistent_state"]
    assert accounting["total_fp16_values"] == expected["total_fp16_values"]
    assert accounting["total_checkpoint_payload_bytes"] == expected["total_bytes"]
    assert accounting["checkpoint_byte_fraction"] < 0.01
    assert accounting["persistent_w0_bytes"] == 0
    assert accounting["persistent_empirical_basis_bytes"] == 0
    assert accounting["static_key_matrix_flops"] > 0
    assert accounting["live_latent_refresh_matrix_flops"] > 0
    assert accounting["dense_mlp_matrix_flops_per_token_after_materialization"] > 0


def test_joint_displacement_pcs_are_paired_and_normalized() -> None:
    generator = torch.Generator().manual_seed(11)
    states, hidden, width = 7, 9, 5
    fc = torch.randn(states, hidden, width, generator=generator)
    proj = torch.randn(states, width, hidden, generator=generator)
    bundle = paired_displacement_pcs(fc, proj, components=3, device="cpu")
    assert bundle["detector_pcs"].shape == (3, hidden, width)
    assert bundle["write_pcs"].shape == (3, hidden, width)
    norms = (
        bundle["detector_pcs"].square().flatten(1).sum(1)
        + bundle["write_pcs"].square().flatten(1).sum(1)
    )
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=2e-5, rtol=2e-5)
    assert 0 < bundle["retained_energy_fraction"] <= 1


def tiny_decoder(*, linear: bool = False, unpaired: bool = False) -> PairedNeuronDecoder:
    return PairedNeuronDecoder(
        width=8, shared_width=6, latent_width=3,
        deployment_layers=12, measured_layers=[0], components=3,
        seed=13, linear=linear, unpaired=unpaired,
    )


def test_zero_latent_is_bit_exact_and_paired_gradients_exist() -> None:
    decoder = tiny_decoder()
    generator = torch.Generator().manual_seed(17)
    detector = torch.randn(10, 8, generator=generator)
    write = torch.randn(10, 8, generator=generator)
    zero_u, zero_v = decoder.predict(0, detector, write, zero_codes=True)
    assert torch.count_nonzero(zero_u) == 0
    assert torch.count_nonzero(zero_v) == 0
    u, v = decoder.predict(0, detector, write)
    (u.square().mean() + v.square().mean()).backward()
    for name in ("p_u", "p_v", "q_u", "q_v", "c", "layer_embeddings", "codes_u"):
        parameter = getattr(decoder, name)
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_linear_control_is_w0_independent() -> None:
    decoder = tiny_decoder(linear=True)
    generator = torch.Generator().manual_seed(19)
    first = torch.randn(10, 8, generator=generator)
    second = torch.randn(10, 8, generator=generator)
    reference = decoder.predict(0, first, second)
    changed = decoder.predict(0, 7 * first, -3 * second)
    torch.testing.assert_close(reference[0], changed[0])
    torch.testing.assert_close(reference[1], changed[1])


def test_compact_payload_excludes_codes_and_w0() -> None:
    decoder = PairedNeuronDecoder(
        width=768, shared_width=176, latent_width=16,
        deployment_layers=12, measured_layers=[0, 6, 11], components=16,
        seed=23,
    )
    accounting = deployment_accounting()
    payload = compact_payload(decoder, accounting)
    assert payload["accounted_payload_bytes"] == 1_091_584
    assert "codes_u" not in payload["tensors"]
    assert "detector_w0" not in payload["tensors"]
    assert "write_w0" not in payload["tensors"]
