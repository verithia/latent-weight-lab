from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_w0_quantile_field import (
    field_state_accounting,
    load_w0_snapshot,
    synthetic_self_check,
    w0_quantile_assignment,
)
from examples.nanogpt.parameter_trajectory import SCHEMA_VERSION


def test_exact_h10c_accounting() -> None:
    record = field_state_accounting(
        rows=3072,
        columns=768,
        category_count=1024,
        deployment_matrix_count=24,
        maximum_fraction=0.01,
    )
    assert record["gain_scalars_per_matrix"] == 3_840
    assert record["value_scalars_per_matrix"] == 1_024
    assert record["private_bytes"] == 233_472
    assert record["residual_coordinates_per_matrix"] == 18_728
    assert record["residual_bytes"] == 898_944
    assert record["total_checkpoint_bytes"] == 1_132_416
    assert 0.00999 < float(record["total_checkpoint_byte_fraction"]) <= 0.01


def test_quantile_assignment_is_deterministic_and_populated() -> None:
    generator = torch.Generator(device="cpu").manual_seed(7)
    w0 = torch.randn(512, 256, generator=generator)
    first = w0_quantile_assignment(w0, category_count=64)
    second = w0_quantile_assignment(w0.clone(), category_count=64)
    assert torch.equal(first, second)
    assert first.min() == 0
    assert first.max() == 63
    assert torch.unique(first).numel() == 64


def test_field_family_reconstructs_own_member() -> None:
    assert synthetic_self_check("cpu") > 0.999


def test_single_step_w0_loader_does_not_require_a_path(tmp_path) -> None:
    path = tmp_path / "step_00000000.pt"
    name = "transformer.h.0.mlp.c_fc.weight"
    tensor = torch.randn(8, 4)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "step": 0,
            "run_identity": {"name": "test"},
            "run_identity_sha256": "abc",
            "model_config": {"n_layer": 1},
            "storage_dtype": "float32",
            "execution_provenance": None,
            "parameters": {name: tensor},
        },
        path,
    )
    values, metadata = load_w0_snapshot(path, layers={0}, targets={"mlp.c_fc"})
    assert torch.equal(values[name], tensor)
    assert metadata["step"] == 0
