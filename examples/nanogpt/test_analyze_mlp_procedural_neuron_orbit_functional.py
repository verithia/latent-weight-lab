import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_procedural_neuron_orbit_functional import (
    ProceduralCompleteNeuronOrbit,
    deployment_accounting,
    terminal_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_procedural_neuron_orbit_functional_plan.json"


def test_direct_entrypoint_resolves_repository_package() -> None:
    script = Path(__file__).with_name(
        "analyze_mlp_procedural_neuron_orbit_functional.py"
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


def test_exact_h68_accounting_matches_plan() -> None:
    plan = json.loads(PLAN.read_text())
    accounting = deployment_accounting()
    expected = plan["exact_deployment_accounting"]
    assert accounting["total_latent_values"] == expected["total_latent_values"]
    assert accounting["total_checkpoint_payload_bytes"] == expected[
        "total_checkpoint_bytes"
    ]
    assert accounting["latent_value_fraction"] < 0.01
    assert accounting["persistent_procedural_parent_bytes"] == 0


def tiny_student() -> ProceduralCompleteNeuronOrbit:
    generator = torch.Generator().manual_seed(7)
    detector = {0: torch.randn(48, 12, generator=generator)}
    write = {0: torch.randn(48, 12, generator=generator)}
    return ProceduralCompleteNeuronOrbit(
        detector,
        write,
        [0],
        transport_diagonals_per_side=3,
        mixer_groups=3,
        mixer_group_width=4,
        pre_gain_initial=1.0,
        pre_bias_initial=0.0,
        post_gain_initial=1.0,
        transport_diagonal_initial=0.0,
        output_offset_initial=0.0,
    )


def test_h68_step_zero_parent_is_exact() -> None:
    student = tiny_student()
    inputs = torch.randn(7, 12, generator=torch.Generator().manual_seed(11))
    output, _ = student.forward_function(
        0, inputs, None, mode="step_zero_parent"
    )
    expected = F.gelu(inputs @ student.base_detector[0].T) @ student.base_write[0]
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)


def test_h68_analytic_jvp_matches_autograd() -> None:
    student = tiny_student()
    with torch.no_grad():
        student.pre_gain.normal_(mean=1.0, std=0.1)
        student.pre_bias.normal_(std=0.1)
        student.post_gain.normal_(mean=1.0, std=0.1)
        student.input_transport_diagonals.normal_(std=0.1)
        student.output_transport_diagonals.normal_(std=0.1)
        student.output_offset.normal_(std=0.1)
    generator = torch.Generator().manual_seed(13)
    inputs = torch.randn(5, 12, generator=generator, requires_grad=True)
    directions = torch.randn(5, 12, generator=generator)
    output, analytic = student.forward_function(0, inputs, directions)
    assert analytic is not None
    automatic_output, automatic = torch.autograd.functional.jvp(
        lambda value: student.forward_function(0, value, None)[0],
        inputs,
        directions,
    )
    torch.testing.assert_close(output, automatic_output)
    torch.testing.assert_close(analytic, automatic, atol=4e-5, rtol=4e-5)


def test_h68_all_compact_coordinates_receive_gradients_and_serialize() -> None:
    student = tiny_student()
    inputs = torch.randn(5, 12, generator=torch.Generator().manual_seed(17))
    directions = torch.randn(5, 12, generator=torch.Generator().manual_seed(19))
    output, action = student.forward_function(0, inputs, directions)
    assert action is not None
    (output.square().mean() + action.square().mean()).backward()
    for parameter in student.parameters():
        assert parameter.grad is not None
        assert float(parameter.grad.norm()) > 0
    accounting = deployment_accounting(
        layers=1,
        width=12,
        hidden_width=48,
        transport_diagonals_per_side=3,
    )
    artifact = terminal_artifact(student, accounting)
    assert artifact["accounted_payload_bytes"] == accounting[
        "total_checkpoint_payload_bytes"
    ]
    assert "base_detector" not in artifact
    assert "base_write" not in artifact
