from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_optimizer_probe_span import (
    analyze_rows,
    load_probe_inventory,
)
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


PARAMETER = "transformer.h.6.mlp.c_fc.weight"


def write_probe(path: Path, step: int, offset: float) -> None:
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) + offset
    state = {
        "weight_before_step": weight,
        "gradient_after_clip": weight + 1,
        "momentum_buffer_before_step": weight + 2,
        "combined_momentum_update": weight + 3,
        "polar_update": weight + 4,
        "applied_direction_per_lr": -(weight + 5),
    }
    torch.save(
        {
            "schema_version": OPTIMIZER_PROBE_SCHEMA_VERSION,
            "step": step,
            "run_identity_sha256": "a" * 64,
            "execution_provenance": {"git_commit": "b" * 40},
            "parameters": {PARAMETER: state},
        },
        path,
    )


def test_inventory_and_orientations() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        paths = []
        for step in (0, 2, 5):
            path = root / f"step_{step:06d}.pt"
            write_probe(path, step, float(step))
            paths.append(path)
        steps, values, metadata = load_probe_inventory(
            paths, layers={6}, targets={"mlp.c_fc"}
        )
        assert steps == [0, 2, 5]
        assert metadata["run_identity_sha256"] == "a" * 64
        assert set(values) == {PARAMETER}
        torch.testing.assert_close(
            values[PARAMETER]["raw_gradient_descent"][0],
            -(torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1),
        )


def test_chronological_probe_analysis() -> None:
    generator = torch.Generator().manual_seed(7)
    rows = torch.randn((12, 32), generator=generator)
    spectra, transfer, drift = analyze_rows(
        rows,
        parameter=PARAMETER,
        field="exact_applied_direction",
        steps=list(range(12)),
        discovery_stop=5,
        validation_stop=9,
        ranks=[1, 2, 4],
    )
    assert len(spectra) == 2
    assert {row["split"] for row in transfer} == {
        "discovery",
        "validation",
        "test",
    }
    assert len(drift) == 2
