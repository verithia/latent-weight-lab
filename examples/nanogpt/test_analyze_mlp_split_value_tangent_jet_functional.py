import json
import subprocess
import sys
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_split_value_tangent_jet_functional import (
    SplitValueTangentJet,
    deployment_accounting,
    terminal_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_split_value_tangent_jet_functional_plan.json"


def test_direct_entrypoint_resolves_repository_package() -> None:
    script = Path(__file__).with_name(
        "analyze_mlp_split_value_tangent_jet_functional.py"
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


def test_exact_h60_accounting_matches_plan() -> None:
    plan = json.loads(PLAN.read_text())
    accounting = deployment_accounting()
    expected = plan["exact_deployment_accounting"]
    assert accounting["total_latent_values"] == expected["total_latent_values"]
    assert accounting["total_checkpoint_payload_bytes"] == expected[
        "total_checkpoint_bytes"
    ]
    assert accounting["latent_value_fraction"] < 0.01
    assert accounting["extra_mlp_matmul_fraction"] < 0.10


def tiny_student() -> SplitValueTangentJet:
    generator = torch.Generator().manual_seed(7)
    detector = {0: torch.randn(8, 4, generator=generator)}
    write = {0: torch.randn(8, 4, generator=generator)}
    anchors = torch.randn(1, 4, generator=generator)
    return SplitValueTangentJet(
        detector,
        write,
        [0],
        anchors,
        value_rank=2,
        tangent_rank=2,
        router_seed=11,
        value_v_seed=13,
        tangent_v_seed=17,
        static_initial=1.0,
        amplitude_initial=0.1,
        bias_initial=0.0,
        tangent_initial=1.0,
        offset_initial=0.0,
    )


def test_analytic_jvp_matches_autograd() -> None:
    student = tiny_student()
    with torch.no_grad():
        student.value_u.normal_()
        student.tangent_u.normal_()
        student.output_offset.normal_()
    generator = torch.Generator().manual_seed(19)
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


def test_affine_jet_is_anchored() -> None:
    student = tiny_student()
    with torch.no_grad():
        student.value_u.zero_()
        student.tangent_u.normal_()
        student.output_offset.normal_()
    anchor = student.anchors[0].unsqueeze(0)
    output, _ = student.forward_function(0, anchor, None, mode="affine_jet_only")
    parent, _ = student.forward_function(0, anchor, None, mode="step_zero_parent")
    torch.testing.assert_close(output - parent, student.output_offset[0].unsqueeze(0))


def test_compact_artifact_and_gradient_flow() -> None:
    student = tiny_student()
    inputs = torch.randn(5, 4, generator=torch.Generator().manual_seed(23))
    output, _ = student.forward_function(0, inputs, None)
    output.square().mean().backward()
    assert student.value_u.grad is not None
    assert float(student.value_u.grad.norm()) > 0
    assert student.tangent_u.grad is not None
    assert float(student.tangent_u.grad.norm()) > 0
    assert student.output_offset.grad is not None
    accounting = deployment_accounting(
        layers=1, width=4, value_rank=2, tangent_rank=2
    )
    artifact = terminal_artifact(student, accounting)
    assert artifact["accounted_payload_bytes"] == accounting[
        "total_checkpoint_payload_bytes"
    ]
    assert artifact["anchors"].dtype == torch.float16
