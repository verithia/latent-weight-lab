import json
import subprocess
import sys
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_parallel_residual_mlp_functional import (
    ParallelResidualMLP,
    deployment_accounting,
    terminal_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_parallel_residual_mlp_functional_plan.json"


def test_direct_entrypoint_resolves_repository_package() -> None:
    script = Path(__file__).with_name(
        "analyze_mlp_parallel_residual_mlp_functional.py"
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


def test_exact_h59_accounting_matches_plan() -> None:
    plan = json.loads(PLAN.read_text())
    accounting = deployment_accounting()
    expected = plan["exact_deployment_accounting"]
    assert accounting["total_latent_values"] == expected["total_latent_values"]
    assert accounting["total_checkpoint_payload_bytes"] == expected["total_checkpoint_bytes"]
    assert accounting["latent_value_fraction"] < 0.01
    assert accounting["extra_mlp_matmul_fraction"] < 0.10


def tiny_student() -> ParallelResidualMLP:
    generator = torch.Generator().manual_seed(7)
    detector = {0: torch.randn(8, 4, generator=generator)}
    write = {0: torch.randn(8, 4, generator=generator)}
    return ParallelResidualMLP(
        detector,
        write,
        [0],
        residual_width=3,
        detector_seed=11,
        gain_initial=1.0,
        bias_initial=0.0,
    )


def test_analytic_jvp_matches_autograd() -> None:
    student = tiny_student()
    with torch.no_grad():
        student.write.normal_(generator=torch.Generator().manual_seed(13))
    generator = torch.Generator().manual_seed(17)
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


def test_compact_artifact_and_gradient_flow() -> None:
    student = tiny_student()
    inputs = torch.randn(5, 4, generator=torch.Generator().manual_seed(23))
    output, _ = student.forward_function(0, inputs, None)
    output.square().mean().backward()
    assert student.write.grad is not None
    assert float(student.write.grad.norm()) > 0
    assert student.detector.grad is not None
    accounting = deployment_accounting(layers=1, width=4, residual_width=3)
    artifact = terminal_artifact(student, accounting)
    assert artifact["accounted_payload_bytes"] == accounting[
        "total_checkpoint_payload_bytes"
    ]
    assert artifact["detector"].dtype == torch.float16
