from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    collect_parameters,
    optimizer_probe_path,
    snapshot_path,
    validate_arguments,
    write_optimizer_probe,
    write_parameter_snapshot,
)
from examples.nanogpt.muon import Muon


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

    def test_optimizer_probe_captures_exact_muon_pre_step_state(self) -> None:
        model = TinyModel()
        selected = [
            model.transformer.h[0].mlp.c_proj.weight,
            model.transformer.h[1].mlp.c_proj.weight,
        ]
        optimizer = Muon(
            selected,
            lr=0.02,
            momentum=0.9,
            weight_decay=0.1,
            ns_steps=3,
        )
        for index, parameter in enumerate(selected):
            parameter.grad = torch.full_like(parameter, 0.2 + index)
            optimizer.state[parameter]["momentum_buffer"] = torch.full_like(
                parameter, 0.05 + index
            )
        identity = {
            "config_sha256": "a" * 64,
            "data_manifest": {"sha256": "b" * 64},
        }
        with tempfile.TemporaryDirectory() as raw:
            out_dir = Path(raw)
            path = write_optimizer_probe(
                model=model,
                optimizer=optimizer,
                out_dir=out_dir,
                step=60,
                targets=["mlp.c_proj"],
                dtype="float32",
                layers=[0, 1],
                model_config=config_as_dataclass(),
                run_identity=identity,
                execution_provenance={"git_commit": "c" * 40},
            )
            self.assertEqual(path, optimizer_probe_path(out_dir, 60))
            payload = torch.load(
                path, map_location="cpu", weights_only=False
            )
            self.assertEqual(
                payload["schema_version"],
                OPTIMIZER_PROBE_SCHEMA_VERSION,
            )
            self.assertEqual(len(payload["parameters"]), 2)
            first = payload["parameters"][
                "transformer.h.0.mlp.c_proj.weight"
            ]
            expected = 0.2 + 0.9 * (0.9 * 0.05 + 0.2)
            torch.testing.assert_close(
                first["combined_momentum_update"],
                torch.full_like(
                    first["combined_momentum_update"], expected
                ),
            )
            self.assertFalse(list(path.parent.glob("*.part")))

    def test_optimizer_probe_can_store_weight_and_gradient_only(self) -> None:
        model = TinyModel()
        parameter = model.transformer.h[0].mlp.c_proj.weight
        optimizer = Muon(
            [parameter],
            lr=0.02,
            momentum=0.9,
            weight_decay=0.1,
            ns_steps=3,
        )
        parameter.grad = torch.full_like(parameter, 0.25)
        with tempfile.TemporaryDirectory() as raw:
            path = write_optimizer_probe(
                model=model,
                optimizer=optimizer,
                out_dir=Path(raw),
                step=0,
                targets=["mlp.c_proj"],
                dtype="float16",
                layers=[0],
                model_config=config_as_dataclass(),
                run_identity={"config_sha256": "a" * 64},
                execution_provenance=None,
                fields=["weight_before_step", "gradient_after_clip"],
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(
                payload["fields"],
                ["weight_before_step", "gradient_after_clip"],
            )
            stored = payload["parameters"][
                "transformer.h.0.mlp.c_proj.weight"
            ]
            self.assertEqual(
                set(stored), {"weight_before_step", "gradient_after_clip"}
            )
            self.assertEqual(stored["weight_before_step"].dtype, torch.float16)

    def test_validation_rejects_bad_optimizer_probe(self) -> None:
        base = dict(
            trajectory_snapshot_interval=0,
            trajectory_snapshot_targets=["mlp.c_proj"],
            trajectory_snapshot_layers=None,
            trajectory_snapshot_all_parameters=False,
            optimizer_probe_targets=["mlp.c_proj"],
            optimizer_probe_layers=[0],
            optimizer_probe_dtype="float32",
            optimizer_probe_fields=None,
        )
        with self.assertRaisesRegex(ValueError, "sorted unique"):
            validate_arguments(
                argparse.Namespace(
                    **base,
                    optimizer_probe_steps=[60, 0],
                )
            )
        with self.assertRaisesRegex(ValueError, "probe-layers"):
            validate_arguments(
                argparse.Namespace(
                    **{
                        **base,
                        "optimizer_probe_layers": None,
                    },
                    optimizer_probe_steps=[0, 60],
                )
            )


if __name__ == "__main__":
    unittest.main()
