from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from examples.nanogpt.analyze_mlp_disjoint_data_state_transfer import (
    load_weight_run,
    nested_bases,
    row_cosine,
    row_norm_ratio,
    summarize_metric,
)
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION


PARAMETER = "transformer.h.6.mlp.c_fc.weight"


def test_weight_loader_ignores_extra_probe_fields() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        for step in (0, 2, 5):
            weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) + step
            torch.save(
                {
                    "schema_version": OPTIMIZER_PROBE_SCHEMA_VERSION,
                    "step": step,
                    "run_identity_sha256": "a" * 64,
                    "execution_provenance": {"git_commit": "b" * 40},
                    "parameters": {
                        PARAMETER: {
                            "weight_before_step": weight,
                            "gradient_after_clip": weight + 1,
                        }
                    },
                },
                root / f"step_{step:06d}.pt",
            )
        steps, inventory, metadata = load_weight_run(
            root, layer=6, targets={"mlp.c_fc"}
        )
        assert steps == [0, 2, 5]
        assert set(inventory) == {PARAMETER}
        assert metadata["run_identity_sha256"] == "a" * 64
        torch.testing.assert_close(
            inventory[PARAMETER][-1],
            torch.arange(12, dtype=torch.float32).reshape(3, 4) + 5,
        )


def test_nested_bases_are_orthonormal_and_metrics_are_exact() -> None:
    generator = torch.Generator().manual_seed(11)
    rows = torch.randn((12, 64), generator=generator)
    bases = nested_bases(rows, 6)
    assert set(bases) == {"centered", "mean_plus_centered"}
    for basis in bases.values():
        torch.testing.assert_close(
            basis.T @ basis,
            torch.eye(6),
            atol=1e-5,
            rtol=1e-5,
        )
    torch.testing.assert_close(row_cosine(rows, rows), torch.ones(12, dtype=torch.float64))
    torch.testing.assert_close(
        row_norm_ratio(rows, rows * 2), torch.full((12,), 2.0, dtype=torch.float64)
    )


def test_summary_groups_displacement_cosine() -> None:
    rows = [
        {"parameter": "p", "split": "test", "displacement_cosine": 0.2},
        {"parameter": "p", "split": "test", "displacement_cosine": 0.4},
    ]
    summary = summarize_metric(
        rows, ("parameter", "split"), "displacement_cosine"
    )
    assert len(summary) == 1
    assert abs(summary[0]["mean"] - 0.3) < 1e-12
    assert summary[0]["minimum"] == 0.2
    assert summary[0]["maximum"] == 0.4
