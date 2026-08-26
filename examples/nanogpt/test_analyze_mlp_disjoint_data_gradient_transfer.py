from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_disjoint_data_gradient_transfer import (
    affine_basis,
    load_raw_gradient_run,
    row_cosine,
    summarize_rows,
)
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION


PARAMETER = "transformer.h.6.mlp.c_fc.weight"


def write_probe(path: Path, step: int, full: bool) -> None:
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    state = {
        "weight_before_step": weight,
        "gradient_after_clip": weight + step + 1,
    }
    if full:
        state.update(
            momentum_buffer_before_step=weight + 2,
            combined_momentum_update=weight + 3,
            polar_update=weight + 4,
            applied_direction_per_lr=weight + 5,
        )
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


def test_loader_accepts_raw_only_and_full_probes() -> None:
    for full in (False, True):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for step in (0, 2, 5):
                write_probe(root / f"step_{step:06d}.pt", step, full)
            steps, inventory, metadata = load_raw_gradient_run(
                root, layer=6, targets={"mlp.c_fc"}
            )
            assert steps == [0, 2, 5]
            assert set(inventory) == {PARAMETER}
            assert metadata["run_identity_sha256"] == "a" * 64
            torch.testing.assert_close(
                inventory[PARAMETER]["gradient"][0],
                -(torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1),
            )


def test_affine_basis_and_cosine() -> None:
    generator = torch.Generator().manual_seed(7)
    rows = torch.randn((8, 32), generator=generator)
    bases = affine_basis(rows, 4)
    assert set(bases) == {
        "discovery_centered",
        "discovery_mean_plus_centered",
    }
    for basis in bases.values():
        torch.testing.assert_close(
            basis.T @ basis,
            torch.eye(4),
            atol=1e-5,
            rtol=1e-5,
        )
    torch.testing.assert_close(row_cosine(rows, rows), torch.ones(8, dtype=torch.float64))


def test_summary_groups_metrics() -> None:
    rows = [
        {"parameter": "p", "split": "test", "metric": 0.2},
        {"parameter": "p", "split": "test", "metric": 0.4},
    ]
    summary = summarize_rows(rows, ("parameter", "split"), "metric")
    assert len(summary) == 1
    assert abs(summary[0]["mean"] - 0.3) < 1e-12
    assert summary[0]["minimum"] == 0.2
