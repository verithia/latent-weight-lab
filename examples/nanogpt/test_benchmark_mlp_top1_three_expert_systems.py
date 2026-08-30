import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from examples.nanogpt.benchmark_mlp_top1_three_expert_systems import (
    deployment_accounting,
    dispatch_with_routes,
)


def test_direct_entrypoint_resolves_repository_package() -> None:
    script = Path(__file__).with_name(
        "benchmark_mlp_top1_three_expert_systems.py"
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


def test_exact_h49_accounting() -> None:
    accounting = deployment_accounting()
    assert accounting["binary_endpoint_values"] == 4_718_592
    assert accounting["binary_endpoint_bytes"] == 589_824
    assert accounting["fp16_endpoint_scale_bytes"] == 12_288
    assert accounting["fp16_private_factor_values"] == 184_320
    assert accounting["fp16_private_factor_bytes"] == 368_640
    assert accounting["fp16_layer_gain_bytes"] == 73_728
    assert accounting["fp16_router_values"] == 27_684
    assert accounting["fp16_router_bytes"] == 55_368
    assert accounting["total_checkpoint_bytes"] == 1_099_848
    assert accounting["continuous_coordinate_values"] == 248_868


def test_top1_dispatch_matches_rowwise_expert_reference() -> None:
    generator = torch.Generator().manual_seed(49)
    inputs = torch.randn(7, 4, generator=generator)
    expert_u = torch.randn(3, 5, 4, generator=generator)
    expert_v = torch.randn(3, 5, 4, generator=generator)
    routes = torch.tensor([0, 1, 2, 1, 0, 2, 2])
    actual = dispatch_with_routes(inputs, routes, expert_u, expert_v)
    expected = torch.empty_like(inputs)
    for row, expert in enumerate(routes.tolist()):
        expected[row] = F.gelu(inputs[row] @ expert_u[expert].T) @ expert_v[
            expert
        ]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_top1_dispatch_handles_empty_expert() -> None:
    generator = torch.Generator().manual_seed(51)
    inputs = torch.randn(6, 4, generator=generator)
    expert_u = torch.randn(3, 5, 4, generator=generator)
    expert_v = torch.randn(3, 5, 4, generator=generator)
    routes = torch.tensor([0, 0, 2, 2, 0, 2])
    output = dispatch_with_routes(inputs, routes, expert_u, expert_v)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
