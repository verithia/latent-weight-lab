import json
from pathlib import Path

import torch

from examples.nanogpt.benchmark_mlp_dual_bitplane_chord_atlas_systems import (
    PLAN_SCHEMA_VERSION,
    deployment_accounting,
    materialize_endpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_dual_bitplane_chord_atlas_systems_plan.json"


def test_h50_accounting_matches_frozen_plan() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    observed = deployment_accounting()
    expected = plan["exact_checkpoint_accounting"]
    for key, value in observed.items():
        assert value == expected[key]
    assert observed["total_checkpoint_bytes"] == 1_124_352
    assert observed["checkpoint_byte_fraction"] < 0.01


def test_materialization_exact_two_plane_formula() -> None:
    signs = torch.tensor(
        [
            [[1.0, -1.0], [-1.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ]
    )
    scales = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    coordinates = torch.tensor([[11.0, 13.0], [17.0, 19.0]])
    observed = materialize_endpoint(signs, scales, coordinates)
    expected = (
        coordinates[0, :, None] * scales[0, :, None] * signs[0]
        + coordinates[1, :, None] * scales[1, :, None] * signs[1]
    )
    torch.testing.assert_close(observed, expected)


def test_materialization_rejects_wrong_plane_count() -> None:
    signs = torch.ones(3, 2, 2)
    scales = torch.ones(3, 2)
    coordinates = torch.ones(3, 2)
    try:
        materialize_endpoint(signs, scales, coordinates)
    except ValueError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("wrong plane count was accepted")
