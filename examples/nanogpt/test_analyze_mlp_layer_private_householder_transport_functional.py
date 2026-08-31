import json
import subprocess
import sys
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_layer_private_householder_transport_functional import (
    LayerPrivateHouseholderTransport,
    apply_householder_product,
    deployment_accounting,
    terminal_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_layer_private_householder_transport_functional_plan.json"


def test_direct_entrypoint_resolves_repository_package() -> None:
    script = Path(__file__).with_name(
        "analyze_mlp_layer_private_householder_transport_functional.py"
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


def test_exact_h65_accounting_matches_plan() -> None:
    plan = json.loads(PLAN.read_text())
    accounting = deployment_accounting()
    expected = plan["exact_deployment_accounting"]
    assert accounting["total_latent_values"] == expected["total_latent_values"]
    assert accounting["total_checkpoint_payload_bytes"] == expected[
        "total_checkpoint_bytes"
    ]
    assert accounting["latent_value_fraction"] < 0.01
    assert accounting["extra_mlp_matmul_fraction"] < 0.07


def tiny_student() -> LayerPrivateHouseholderTransport:
    generator = torch.Generator().manual_seed(7)
    detector = {0: torch.randn(8, 4, generator=generator)}
    write = {0: torch.randn(8, 4, generator=generator)}
    anchors = torch.randn(1, 4, generator=generator)
    return LayerPrivateHouseholderTransport(
        detector,
        write,
        [0],
        anchors,
        banks=2,
        codes_per_bank=3,
        router_rank=2,
        tangent_rank=2,
        reflectors_per_side=2,
        router_q_seed=11,
        router_p_seed=13,
        tangent_v_seed=17,
        input_reflector_seed=19,
        output_reflector_seed=23,
        code_gain_initial=1.0,
        router_bias_initial=0.0,
        temperature=1.0,
        layer_diagonal_initial=1.0,
        reflector_epsilon=1e-12,
        offset_initial=0.0,
    )


def test_householder_product_preserves_norm() -> None:
    generator = torch.Generator().manual_seed(29)
    values = torch.randn(9, 4, generator=generator)
    vectors = torch.randn(3, 4, generator=generator)
    result = apply_householder_product(
        values, vectors, count=3, epsilon=1e-12
    )
    torch.testing.assert_close(
        result.square().sum(dim=-1),
        values.square().sum(dim=-1),
        atol=2e-5,
        rtol=2e-5,
    )


def test_householder_jvp_matches_autograd() -> None:
    student = tiny_student()
    student.eval()
    with torch.no_grad():
        student.codebook.normal_()
        student.tangent_u.normal_()
    generator = torch.Generator().manual_seed(31)
    inputs = torch.randn(3, 4, generator=generator, requires_grad=True)
    direction = torch.randn(3, 4, generator=generator)
    output, analytic = student.forward_function(0, inputs, direction)
    assert analytic is not None
    automatic_output, automatic = torch.autograd.functional.jvp(
        lambda value: student.forward_function(0, value, None)[0],
        inputs,
        direction,
    )
    torch.testing.assert_close(output, automatic_output)
    torch.testing.assert_close(analytic, automatic, atol=2e-5, rtol=2e-5)


def test_hard_value_identity_gradients_and_artifact() -> None:
    student = tiny_student()
    with torch.no_grad():
        student.codebook.normal_()
        student.tangent_u.normal_()
    inputs = torch.randn(5, 4, generator=torch.Generator().manual_seed(37))
    directions = torch.randn(5, 4, generator=torch.Generator().manual_seed(41))
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
        student.input_reflectors,
        student.output_reflectors,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.norm()) > 0
    accounting = deployment_accounting(
        layers=1,
        width=4,
        banks=2,
        codes_per_bank=3,
        router_rank=2,
        tangent_rank=2,
        reflectors_per_side=2,
    )
    artifact = terminal_artifact(student, accounting)
    assert artifact["accounted_payload_bytes"] == accounting[
        "total_checkpoint_payload_bytes"
    ]
