"""Atomic, model-only parameter snapshots for training-trajectory analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch


SCHEMA_VERSION = "nanogpt_parameter_trajectory_v1"
OPTIMIZER_PROBE_SCHEMA_VERSION = "nanogpt_optimizer_probe_v1"
OPTIMIZER_PROBE_FIELDS = (
    "weight_before_step",
    "gradient_after_clip",
    "momentum_buffer_before_step",
    "combined_momentum_update",
    "polar_update",
    "applied_direction_per_lr",
)
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parameter_matches(
    name: str,
    targets: Iterable[str],
    layers: Iterable[int] | None = None,
    *,
    all_parameters: bool = False,
) -> bool:
    if all_parameters:
        return True
    if not name.startswith("transformer.h."):
        return False
    if layers is not None:
        components = name.split(".")
        if len(components) < 3 or not components[2].isdigit():
            return False
        if int(components[2]) not in set(layers):
            return False
    return any(name.endswith(f".{target}.weight") for target in targets)


def collect_parameters(
    model: torch.nn.Module,
    *,
    targets: list[str],
    dtype: str,
    layers: list[int] | None = None,
    all_parameters: bool = False,
) -> dict[str, torch.Tensor]:
    if (
        not all_parameters
        and (
            not targets
            or any(not isinstance(target, str) or not target for target in targets)
        )
    ):
        raise ValueError("trajectory snapshot targets must be non-empty strings")
    if all_parameters and layers is not None:
        raise ValueError(
            "all-parameter trajectory snapshots cannot filter transformer layers"
        )
    if dtype not in DTYPES:
        raise ValueError(f"unsupported trajectory snapshot dtype: {dtype}")
    selected = {
        name: parameter.detach().to(device="cpu", dtype=DTYPES[dtype]).contiguous()
        for name, parameter in model.named_parameters()
        if parameter_matches(
            name,
            targets,
            layers,
            all_parameters=all_parameters,
        )
    }
    if not selected:
        raise ValueError(f"no parameters matched trajectory targets: {targets}")
    return selected


def snapshot_path(out_dir: Path, step: int) -> Path:
    if step < 0:
        raise ValueError("trajectory snapshot step must be non-negative")
    return out_dir / "parameter_trajectory" / f"step_{step:06d}.pt"


def optimizer_probe_path(out_dir: Path, step: int) -> Path:
    if step < 0:
        raise ValueError("optimizer probe step must be non-negative")
    return out_dir / "optimizer_probe" / f"step_{step:06d}.pt"


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".part",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_parameter_snapshot(
    *,
    model: torch.nn.Module,
    out_dir: Path,
    step: int,
    targets: list[str],
    dtype: str,
    layers: list[int] | None,
    all_parameters: bool = False,
    model_config: Any,
    run_identity: dict[str, Any],
    execution_provenance: dict[str, Any] | None,
) -> Path:
    destination = snapshot_path(out_dir, step)
    run_identity_sha256 = canonical_digest(run_identity)
    if destination.exists():
        observed = torch.load(destination, map_location="cpu", weights_only=False)
        if (
            not isinstance(observed, dict)
            or observed.get("schema_version") != SCHEMA_VERSION
            or observed.get("step") != step
            or observed.get("run_identity_sha256") != run_identity_sha256
        ):
            raise ValueError(f"existing trajectory snapshot identity mismatch: {destination}")
        return destination

    parameters = collect_parameters(
        model,
        targets=targets,
        dtype=dtype,
        layers=layers,
        all_parameters=all_parameters,
    )
    inventory = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "bytes": value.numel() * value.element_size(),
        }
        for name, value in sorted(parameters.items())
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "step": step,
        "targets": list(targets),
        "layers": None if layers is None else list(layers),
        "all_parameters": bool(all_parameters),
        "storage_dtype": dtype,
        "model_config": asdict(model_config),
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256,
        "execution_provenance": execution_provenance,
        "tensor_inventory": inventory,
        "parameters": parameters,
        "created_at_unix": time.time(),
    }
    _atomic_torch_save(destination, payload)
    return destination


def write_optimizer_probe(
    *,
    model: torch.nn.Module,
    optimizer: Any,
    out_dir: Path,
    step: int,
    targets: list[str],
    dtype: str,
    layers: list[int],
    model_config: Any,
    run_identity: dict[str, Any],
    execution_provenance: dict[str, Any] | None,
    fields: list[str] | None = None,
) -> Path:
    """Atomically capture pre-step gradients and exact Muon state directions."""
    from examples.nanogpt.muon import Muon, zeropower_via_newtonschulz5

    destination = optimizer_probe_path(out_dir, step)
    run_identity_sha256 = canonical_digest(run_identity)
    if destination.exists():
        observed = torch.load(
            destination, map_location="cpu", weights_only=False
        )
        if (
            not isinstance(observed, dict)
            or observed.get("schema_version")
            != OPTIMIZER_PROBE_SCHEMA_VERSION
            or observed.get("step") != step
            or observed.get("run_identity_sha256") != run_identity_sha256
        ):
            raise ValueError(
                f"existing optimizer probe identity mismatch: {destination}"
            )
        return destination
    if dtype not in DTYPES:
        raise ValueError(f"unsupported optimizer probe dtype: {dtype}")
    if not targets or not layers:
        raise ValueError("optimizer probe targets and layers must be non-empty")
    selected_fields = tuple(OPTIMIZER_PROBE_FIELDS if fields is None else fields)
    if (
        not selected_fields
        or len(set(selected_fields)) != len(selected_fields)
        or any(field not in OPTIMIZER_PROBE_FIELDS for field in selected_fields)
    ):
        raise ValueError(
            "optimizer probe fields must be unique supported field names"
        )

    suboptimizers = getattr(optimizer, "optimizers", [optimizer])
    owners: dict[int, tuple[Muon, dict[str, Any]]] = {}
    for candidate in suboptimizers:
        if not isinstance(candidate, Muon):
            continue
        for group in candidate.param_groups:
            for parameter in group["params"]:
                owners[id(parameter)] = (candidate, group)

    tensors: dict[str, dict[str, torch.Tensor]] = {}
    hyperparameters: dict[str, dict[str, float | int]] = {}
    selected_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter_matches(name, targets, layers)
    }
    if not selected_parameters:
        raise ValueError("no parameters matched the optimizer probe")
    storage_dtype = DTYPES[dtype]
    for name, parameter in sorted(selected_parameters.items()):
        owner = owners.get(id(parameter))
        if owner is None:
            raise ValueError(f"optimizer probe parameter is not Muon-owned: {name}")
        muon, group = owner
        if parameter.grad is None:
            raise ValueError(f"optimizer probe parameter has no gradient: {name}")
        gradient = parameter.grad.detach().float()
        momentum = float(group["momentum"])
        ns_steps = int(group["ns_steps"])
        weight_decay = float(group["weight_decay"])
        need_buffer = any(
            field
            in {
                "momentum_buffer_before_step",
                "combined_momentum_update",
                "polar_update",
                "applied_direction_per_lr",
            }
            for field in selected_fields
        )
        buffer: torch.Tensor | None = None
        combined: torch.Tensor | None = None
        polar: torch.Tensor | None = None
        scale: float | None = None
        applied_direction: torch.Tensor | None = None
        if need_buffer:
            state = muon.state.get(parameter, {})
            stored_buffer = state.get("momentum_buffer")
            if stored_buffer is None:
                buffer = torch.zeros_like(gradient)
            else:
                buffer = stored_buffer.detach().float()
        if any(
            field
            in {
                "combined_momentum_update",
                "polar_update",
                "applied_direction_per_lr",
            }
            for field in selected_fields
        ):
            assert buffer is not None
            new_buffer = momentum * buffer + gradient
            combined = gradient + momentum * new_buffer
        if any(
            field in {"polar_update", "applied_direction_per_lr"}
            for field in selected_fields
        ):
            assert combined is not None
            polar = zeropower_via_newtonschulz5(
                combined, steps=ns_steps
            ).float()
            scale = max(
                1.0,
                polar.shape[0] / max(1, polar.numel() / polar.shape[0]),
            ) ** 0.5
        if "applied_direction_per_lr" in selected_fields:
            assert polar is not None and scale is not None
            applied_direction = (
                -weight_decay * parameter.detach().float()
                - scale * polar
            )

        def cpu(value: torch.Tensor) -> torch.Tensor:
            return value.to(
                device="cpu", dtype=storage_dtype
            ).contiguous()

        available = {
            "weight_before_step": parameter.detach(),
            "gradient_after_clip": gradient,
            "momentum_buffer_before_step": buffer,
            "combined_momentum_update": combined,
            "polar_update": polar,
            "applied_direction_per_lr": applied_direction,
        }
        tensors[name] = {
            field: cpu(available[field])
            for field in selected_fields
            if available[field] is not None
        }
        if set(tensors[name]) != set(selected_fields):
            raise RuntimeError("optimizer probe did not construct every requested field")
        hyperparameters[name] = {
            "lr": float(group["lr"]),
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "polar_scale": scale,
        }

    inventory = {
        name: {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel(),
                "bytes": value.numel() * value.element_size(),
            }
            for key, value in values.items()
        }
        for name, values in tensors.items()
    }
    payload = {
        "schema_version": OPTIMIZER_PROBE_SCHEMA_VERSION,
        "step": step,
        "targets": list(targets),
        "layers": list(layers),
        "storage_dtype": dtype,
        "fields": list(selected_fields),
        "model_config": asdict(model_config),
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256,
        "execution_provenance": execution_provenance,
        "tensor_inventory": inventory,
        "hyperparameters": hyperparameters,
        "parameters": tensors,
        "created_at_unix": time.time(),
    }
    _atomic_torch_save(destination, payload)
    return destination


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trajectory-snapshot-interval", type=int, default=0)
    parser.add_argument(
        "--trajectory-snapshot-targets",
        nargs="+",
        default=["mlp.c_fc", "mlp.c_proj"],
    )
    parser.add_argument(
        "--trajectory-snapshot-dtype",
        choices=sorted(DTYPES),
        default="float32",
    )
    parser.add_argument(
        "--trajectory-snapshot-layers",
        nargs="+",
        type=int,
        default=None,
        help="optional zero-based transformer layers to retain",
    )
    parser.add_argument(
        "--trajectory-snapshot-all-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "store every unique named model parameter; intended for sparse "
            "model-state trajectory checkpoints, not per-update traces"
        ),
    )
    parser.add_argument(
        "--optimizer-probe-steps",
        nargs="+",
        type=int,
        default=None,
        help="outer optimizer steps at which to capture pre-step Muon state",
    )
    parser.add_argument(
        "--optimizer-probe-targets",
        nargs="+",
        default=["mlp.c_proj"],
    )
    parser.add_argument(
        "--optimizer-probe-layers",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--optimizer-probe-dtype",
        choices=sorted(DTYPES),
        default="float32",
    )
    parser.add_argument(
        "--optimizer-probe-fields",
        nargs="+",
        choices=OPTIMIZER_PROBE_FIELDS,
        default=None,
        help=(
            "optional stored-field subset; defaults to the complete exact "
            "Muon pre-step probe"
        ),
    )


def validate_arguments(args: argparse.Namespace) -> None:
    if args.trajectory_snapshot_interval < 0:
        raise ValueError("--trajectory-snapshot-interval must be non-negative")
    if args.trajectory_snapshot_interval > 0:
        targets = args.trajectory_snapshot_targets
        all_parameters = bool(
            getattr(args, "trajectory_snapshot_all_parameters", False)
        )
        if not all_parameters and (not isinstance(targets, list) or not targets):
            raise ValueError("--trajectory-snapshot-targets must be non-empty")
        if not all_parameters and any(
            not isinstance(target, str) or not target for target in targets
        ):
            raise ValueError("--trajectory-snapshot-targets must contain non-empty strings")
        layers = getattr(args, "trajectory_snapshot_layers", None)
        if all_parameters and layers is not None:
            raise ValueError(
                "--trajectory-snapshot-all-parameters cannot be combined with "
                "--trajectory-snapshot-layers"
            )
        if layers is not None:
            if (
                not isinstance(layers, list)
                or not layers
                or any(not isinstance(layer, int) or layer < 0 for layer in layers)
            ):
                raise ValueError(
                    "--trajectory-snapshot-layers must contain non-negative integers"
                )
            if len(set(layers)) != len(layers):
                raise ValueError("--trajectory-snapshot-layers must be unique")
    probe_steps = getattr(args, "optimizer_probe_steps", None)
    if probe_steps is not None:
        if (
            not isinstance(probe_steps, list)
            or not probe_steps
            or any(not isinstance(step, int) or step < 0 for step in probe_steps)
            or probe_steps != sorted(set(probe_steps))
        ):
            raise ValueError(
                "--optimizer-probe-steps must contain sorted unique "
                "non-negative integers"
            )
        probe_targets = getattr(args, "optimizer_probe_targets", None)
        if (
            not isinstance(probe_targets, list)
            or not probe_targets
            or any(
                not isinstance(target, str) or not target
                for target in probe_targets
            )
        ):
            raise ValueError(
                "--optimizer-probe-targets must contain non-empty strings"
            )
        probe_layers = getattr(args, "optimizer_probe_layers", None)
        if (
            not isinstance(probe_layers, list)
            or not probe_layers
            or any(
                not isinstance(layer, int) or layer < 0
                for layer in probe_layers
            )
            or len(set(probe_layers)) != len(probe_layers)
        ):
            raise ValueError(
                "--optimizer-probe-layers must contain unique non-negative "
                "integers"
            )
        probe_fields = getattr(args, "optimizer_probe_fields", None)
        if probe_fields is not None and (
            not isinstance(probe_fields, list)
            or not probe_fields
            or len(set(probe_fields)) != len(probe_fields)
            or any(field not in OPTIMIZER_PROBE_FIELDS for field in probe_fields)
        ):
            raise ValueError(
                "--optimizer-probe-fields must contain unique supported fields"
            )
