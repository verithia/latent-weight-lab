import json
import subprocess
import sys
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_layer_private_diagonal_mixer_transport_functional import (
    LayerPrivateDiagonalMixerTransport,
    apply_diagonal_mixer_transport,
    deployment_accounting,
    mixed_radix_involution,
    terminal_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_layer_private_diagonal_mixer_transport_functional_plan.json"


def test_direct_entrypoint_resolves_repository_package() -> None:
    script = Path(__file__).with_name(
        "analyze_mlp_layer_private_diagonal_mixer_transport_functional.py"
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


def test_exact_h66_accounting_matches_plan() -> None:
    plan = json.loads(PLAN.read_text())
    accounting = deployment_accounting()
    expected = plan["exact_deployment_accounting"]
    assert accounting["total_latent_values"] == expected["total_latent_values"]
    assert accounting["total_checkpoint_payload_bytes"] == expected[
        "total_checkpoint_bytes"
    ]
    assert accounting["latent_value_fraction"] < 0.01
    assert accounting["extra_mlp_matmul_fraction"] < 0.07
    assert accounting["extra_transport_operation_upper_bound_fraction"] < 0.01


def tiny_student() -> LayerPrivateDiagonalMixerTransport:
    generator = torch.Generator().manual_seed(7)
    detector = {0: torch.randn(48, 12, generator=generator)}
    write = {0: torch.randn(48, 12, generator=generator)}
    anchors = torch.randn(1, 12, generator=generator)
    return LayerPrivateDiagonalMixerTransport(
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
    )


def test_mixed_radix_mixer_is_an_involutory_isometry() -> None:
    values = torch.randn(9, 12, generator=torch.Generator().manual_seed(29))
    transformed = mixed_radix_involution(values, groups=3, group_width=4)
    recovered = mixed_radix_involution(transformed, groups=3, group_width=4)
    torch.testing.assert_close(recovered, values, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        transformed.square().sum(dim=-1),
        values.square().sum(dim=-1),
        atol=2e-5,
        rtol=2e-5,
    )


def test_zero_diagonal_transport_is_exact_identity() -> None:
    values = torch.randn(7, 12, generator=torch.Generator().manual_seed(31))
    diagonals = torch.zeros(3, 12)
    transformed = apply_diagonal_mixer_transport(
        values,
        diagonals,
        groups=3,
        group_width=4,
        mode="full",
    )
    torch.testing.assert_close(transformed, values, atol=2e-5, rtol=2e-5)


def test_h66_analytic_jvp_matches_autograd() -> None:
    student = tiny_student()
    student.eval()
    with torch.no_grad():
        student.codebook.normal_()
        student.tangent_u.normal_()
        student.input_transport_diagonals.normal_(std=0.1)
        student.output_transport_diagonals.normal_(std=0.1)
    generator = torch.Generator().manual_seed(37)
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
    torch.testing.assert_close(analytic, automatic, atol=3e-5, rtol=3e-5)


def test_hard_value_identity_gradients_and_artifact() -> None:
    student = tiny_student()
    with torch.no_grad():
        student.codebook.normal_()
        student.tangent_u.normal_()
    inputs = torch.randn(5, 12, generator=torch.Generator().manual_seed(41))
    directions = torch.randn(5, 12, generator=torch.Generator().manual_seed(43))
    student.eval()
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
        student.router_q,
        student.router_p,
        student.input_transport_diagonals,
        student.output_transport_diagonals,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.norm()) > 0
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
