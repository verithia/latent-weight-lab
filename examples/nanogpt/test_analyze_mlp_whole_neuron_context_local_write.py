import json
import subprocess
import sys
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_w0_conditioned_block_atlas import (
    basis_diagnostics,
    procedural_blind_blocks,
)
from examples.nanogpt.analyze_mlp_whole_neuron_context_local_write import (
    WholeNeuronContextLocalWriteDecoder,
    compact_payload,
    deployment_accounting,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_whole_neuron_context_local_write_plan.json"
)


def tiny_decoder(
    *,
    context_mode: str = "exact",
    learned_local_decoder: bool = False,
) -> WholeNeuronContextLocalWriteDecoder:
    return WholeNeuronContextLocalWriteDecoder(
        width=8,
        block_width=4,
        shared_width=12,
        context_width=3,
        latent_width=3,
        deployment_layers=12,
        measured_layers=[0],
        components=3,
        seed=11,
        context_mode=context_mode,
        learned_local_decoder=learned_local_decoder,
    )


def test_direct_entrypoint_help() -> None:
    script = Path(__file__).with_name(
        "analyze_mlp_whole_neuron_context_local_write.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_accounting_matches_frozen_plan() -> None:
    plan = json.loads(PLAN.read_text())
    accounting = deployment_accounting()
    assert accounting["total_fp16_values"] == 446_656
    assert accounting["total_checkpoint_payload_bytes"] == 893_312
    assert (
        accounting["total_checkpoint_payload_bytes"]
        == plan["persistent_state"]["total_bytes"]
    )
    assert accounting["checkpoint_byte_fraction"] < 0.01
    assert accounting["persistent_w0_bytes"] == 0
    assert accounting["procedural_local_decoder_values"] == 0


def test_complete_neuron_context_changes_local_block_write() -> None:
    exact = tiny_decoder()
    none = tiny_decoder(context_mode="none")
    none.load_state_dict(exact.state_dict())
    detector = torch.randn(10, 4)
    write = torch.randn(10, 4)
    positions = torch.arange(10) % 2
    exact_u, exact_v = exact.predict(0, detector, write, positions)
    none_u, none_v = none.predict(0, detector, write, positions)
    assert not torch.equal(exact_u, none_u)
    assert not torch.equal(exact_v, none_v)


def test_shuffled_context_is_deterministic_and_distinct() -> None:
    exact = tiny_decoder()
    shuffled = tiny_decoder(context_mode="shuffled")
    shuffled.load_state_dict(exact.state_dict())
    detector = torch.randn(10, 4)
    write = torch.randn(10, 4)
    positions = torch.arange(10) % 2
    exact_u, _ = exact.predict(0, detector, write, positions)
    first, _ = shuffled.predict(0, detector, write, positions)
    second, _ = shuffled.predict(0, detector, write, positions)
    torch.testing.assert_close(first, second)
    assert not torch.equal(exact_u, first)


def test_zero_state_and_all_charged_gradients() -> None:
    decoder = tiny_decoder()
    detector = torch.randn(10, 4)
    write = torch.randn(10, 4)
    positions = torch.arange(10) % 2
    zero_u, zero_v = decoder.predict(
        0, detector, write, positions, zero_codes=True
    )
    assert torch.count_nonzero(zero_u) == 0
    assert torch.count_nonzero(zero_v) == 0
    u, v = decoder.predict(0, detector, write, positions)
    (u.square().mean() + v.square().mean()).backward()
    for name in (
        "p",
        "a",
        "b",
        "c",
        "layer_embeddings",
        "position_embeddings",
        "codes_u",
    ):
        parameter = getattr(decoder, name)
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert not decoder.d_u.requires_grad
    assert not decoder.d_v.requires_grad


def test_learned_decoder_control_is_charged_and_same_initialization() -> None:
    frozen = tiny_decoder()
    learned = tiny_decoder(learned_local_decoder=True)
    torch.testing.assert_close(frozen.d_u, learned.d_u)
    torch.testing.assert_close(frozen.d_v, learned.d_v)
    assert learned.d_u.requires_grad
    assert learned.d_v.requires_grad


def test_blind_complete_neuron_rows_preserve_shape_and_device() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bundle = {
        "detector_w0_blocks": torch.randn(10, 4, device=device),
        "write_w0_blocks": torch.randn(10, 4, device=device),
    }
    copied = procedural_blind_blocks([bundle], seed=13, device=device)[0]
    decoder = tiny_decoder().to(device)
    whole = decoder._whole_rows(
        copied["detector_w0_blocks"], copied["write_w0_blocks"]
    )
    assert whole.shape == (5, 16)
    assert whole.device.type == torch.device(device).type
    assert torch.isfinite(whole).all()


def test_compact_payload_excludes_codes_w0_and_procedural_decoder() -> None:
    decoder = WholeNeuronContextLocalWriteDecoder(
        width=768,
        block_width=32,
        shared_width=1024,
        context_width=128,
        latent_width=16,
        deployment_layers=12,
        measured_layers=[0, 6, 11],
        components=16,
        seed=17,
    )
    payload = compact_payload(decoder, deployment_accounting())
    assert payload["accounted_payload_bytes"] == 893_312
    assert "codes_u" not in payload["tensors"]
    assert "d_u" not in payload["tensors"]
    assert "d_v" not in payload["tensors"]
    assert all("w0" not in key for key in payload["tensors"])


def test_procedural_local_bases_are_full_rank() -> None:
    diagnostics = basis_diagnostics(tiny_decoder())
    assert diagnostics["d_u"]["numerical_rank"] == 4
    assert diagnostics["d_v"]["numerical_rank"] == 4
