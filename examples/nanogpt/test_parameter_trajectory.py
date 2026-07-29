from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from examples.nanogpt.parameter_trajectory import (
    SCHEMA_VERSION,
    collect_parameters,
    snapshot_path,
    validate_arguments,
    write_parameter_snapshot,
)


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = torch.nn.Linear(3, 5, bias=False)
        self.c_proj = torch.nn.Linear(5, 3, bias=False)


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = MLP()


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList([Block(), Block()])
        self.unrelated = torch.nn.Linear(3, 3, bias=False)


def config_as_dataclass() -> object:
    from dataclasses import make_dataclass

    config_type = make_dataclass("Config", [("n_layer", int)])
    return config_type(n_layer=2)


class ParameterTrajectoryTest(unittest.TestCase):
    def test_collects_only_requested_layer_parameters(self) -> None:
        selected = collect_parameters(
            TinyModel(),
            targets=["mlp.c_fc", "mlp.c_proj"],
            dtype="float32",
        )
        self.assertEqual(
            set(selected),
            {
                "transformer.h.0.mlp.c_fc.weight",
                "transformer.h.0.mlp.c_proj.weight",
                "transformer.h.1.mlp.c_fc.weight",
                "transformer.h.1.mlp.c_proj.weight",
            },
        )
        self.assertTrue(all(value.dtype == torch.float32 for value in selected.values()))

    def test_collects_only_requested_transformer_layers(self) -> None:
        selected = collect_parameters(
            TinyModel(),
            targets=["mlp.c_proj"],
            dtype="float32",
            layers=[1],
        )
        self.assertEqual(
            set(selected),
            {"transformer.h.1.mlp.c_proj.weight"},
        )

    def test_collects_all_unique_named_parameters(self) -> None:
        model = TinyModel()
        selected = collect_parameters(
            model,
            targets=[],
            dtype="float32",
            all_parameters=True,
        )
        self.assertEqual(
            set(selected),
            {name for name, _parameter in model.named_parameters()},
        )
        with self.assertRaisesRegex(ValueError, "cannot filter"):
            collect_parameters(
                model,
                targets=[],
                dtype="float32",
                layers=[0],
                all_parameters=True,
            )

    def test_snapshot_is_atomic_identity_bound_and_idempotent(self) -> None:
        model = TinyModel()
        identity = {"config_sha256": "a" * 64, "data_manifest": {"sha256": "b" * 64}}
        with tempfile.TemporaryDirectory() as raw:
            out_dir = Path(raw)
            first = write_parameter_snapshot(
                model=model,
                out_dir=out_dir,
                step=6,
                targets=["mlp.c_fc", "mlp.c_proj"],
                dtype="float32",
                layers=[0],
                all_parameters=False,
                model_config=config_as_dataclass(),
                run_identity=identity,
                execution_provenance={"git_commit": "c" * 40},
            )
            second = write_parameter_snapshot(
                model=model,
                out_dir=out_dir,
                step=6,
                targets=["mlp.c_fc", "mlp.c_proj"],
                dtype="float32",
                layers=[0],
                all_parameters=False,
                model_config=config_as_dataclass(),
                run_identity=identity,
                execution_provenance={"git_commit": "c" * 40},
            )
            self.assertEqual(first, second)
            self.assertEqual(first, snapshot_path(out_dir, 6))
            self.assertFalse(list(first.parent.glob("*.part")))
            payload = torch.load(first, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
            self.assertEqual(payload["step"], 6)
            self.assertEqual(payload["layers"], [0])
            self.assertFalse(payload["all_parameters"])
            self.assertEqual(len(payload["parameters"]), 2)
            self.assertEqual(payload["execution_provenance"]["git_commit"], "c" * 40)

    def test_validation_rejects_bad_interval_and_empty_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_arguments(
                argparse.Namespace(
                    trajectory_snapshot_interval=-1,
                    trajectory_snapshot_targets=["mlp.c_fc"],
                    trajectory_snapshot_layers=None,
                    trajectory_snapshot_all_parameters=False,
                )
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_arguments(
                argparse.Namespace(
                    trajectory_snapshot_interval=1,
                    trajectory_snapshot_targets=[],
                    trajectory_snapshot_layers=None,
                    trajectory_snapshot_all_parameters=False,
                )
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_arguments(
                argparse.Namespace(
                    trajectory_snapshot_interval=1,
                    trajectory_snapshot_targets=["mlp.c_proj"],
                    trajectory_snapshot_layers=[0, 0],
                    trajectory_snapshot_all_parameters=False,
                )
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            validate_arguments(
                argparse.Namespace(
                    trajectory_snapshot_interval=60,
                    trajectory_snapshot_targets=[],
                    trajectory_snapshot_layers=[0],
                    trajectory_snapshot_all_parameters=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
