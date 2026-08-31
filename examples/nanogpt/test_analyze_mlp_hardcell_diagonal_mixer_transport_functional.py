import json
import subprocess
import sys
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_hardcell_diagonal_mixer_transport_functional import (
    HardCellDiagonalMixerTransport,
    deployment_accounting,
    terminal_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_hardcell_diagonal_mixer_transport_functional_plan.json"


def test_direct_entrypoint_resolves_repository_package() -> None:
    script = Path(__file__).with_name(
        "analyze_mlp_hardcell_diagonal_mixer_transport_functional.py"
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


def test_exact_h67_accounting_matches_plan() -> None:
    plan = json.loads(PLAN.read_text())
    accounting = deployment_accounting()
    expected = plan["exact_deployment_accounting"]
    assert accounting["fp16_cell_tangent_values"] == 30_720
    assert accounting["total_latent_values"] == expected["total_latent_values"]
    assert accounting["total_checkpoint_payload_bytes"] == expected[
        "total_checkpoint_bytes"
    ]
    assert accounting["latent_value_fraction"] < 0.01


def tiny_student() -> HardCellDiagonalMixerTransport:
    generator = torch.Generator().manual_seed(7)
    detector = {0: torch.randn(48, 12, generator=generator)}
    write = {0: torch.randn(48, 12, generator=generator)}
    anchors = torch.randn(1, 12, generator=generator)
    return HardCellDiagonalMixerTransport(
        detector,
        write,
        [0],
        anchors,
        banks=2,
        codes_per_bank=3,
        router_rank=2,
        tangent_rank=3,
        transport_diagonals_per_side=3,
        mixer_groups=3,
        mixer_group_width=4,
        router_q_seed=11,
        router_p_seed=13,
        tangent_v_seed=17,
        code_gain_initial=1.0,
        router_bias_initial=0.0,
        temperature=1.0,
        layer_diagonal_initial=1.0,
        transport_diagonal_initial=0.0,
        offset_initial=0.0,
        cell_tangent_initial=0.0,
    )


def test_h67_analytic_hardcell_jvp_matches_autograd() -> None:
    student = tiny_student()
    student.eval()
    with torch.no_grad():
        student.codebook.normal_()
        student.tangent_u.normal_()
        student.cell_tangent.normal_(std=0.2)
        student.input_transport_diagonals.normal_(std=0.1)
        student.output_transport_diagonals.normal_(std=0.1)
    generator = torch.Generator().manual_seed(31)
    inputs = torch.randn(3, 12, generator=generator, requires_grad=True)
    directions = torch.randn(3, 12, generator=generator)
    output, analytic = student.forward_function(0, inputs, directions)
    assert analytic is not None
    automatic_output, automatic = torch.autograd.functional.jvp(
        lambda value: student.forward_function(0, value, None)[0],
        inputs,
        directions,
    )
    torch.testing.assert_close(output, automatic_output)
    torch.testing.assert_close(analytic, automatic, atol=4e-5, rtol=4e-5)


def test_cell_tangent_controls_hard_value_identity_and_gradients() -> None:
    student = tiny_student()
    with torch.no_grad():
        student.codebook.normal_()
        student.tangent_u.normal_()
        student.cell_tangent.normal_(std=0.2)
    inputs = torch.randn(5, 12, generator=torch.Generator().manual_seed(37))
    directions = torch.randn(5, 12, generator=torch.Generator().manual_seed(41))
    student.eval()
    _, full_action = student.forward_function(0, inputs, directions)
    _, static_action = student.forward_function(
        0, inputs, directions, mode="static_tangent_only"
    )
    assert full_action is not None and static_action is not None
    assert not torch.allclose(full_action, static_action)
    _, value_action = student.forward_function(
        0, inputs, directions, mode="hard_value_only"
    )
    _, parent_action = student.forward_function(
        0, inputs, directions, mode="step_zero_parent"
    )
    torch.testing.assert_close(value_action, parent_action, atol=0.0, rtol=0.0)
    student.train()
    output, action = student.forward_function(0, inputs, directions)
    assert action is not None
    (output.square().mean() + action.square().mean()).backward()
    for parameter in (
        student.cell_tangent,
        student.router_q,
        student.input_transport_diagonals,
        student.output_transport_diagonals,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.norm()) > 0


def test_h67_terminal_artifact_exact_payload() -> None:
    student = tiny_student()
    accounting = deployment_accounting(
        layers=1,
        width=12,
        banks=2,
        codes_per_bank=3,
        router_rank=2,
        tangent_rank=3,
        transport_diagonals_per_side=3,
    )
    artifact = terminal_artifact(student, accounting)
    assert artifact["accounted_payload_bytes"] == accounting[
        "total_checkpoint_payload_bytes"
    ]
