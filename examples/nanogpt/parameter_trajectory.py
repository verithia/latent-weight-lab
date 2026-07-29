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
) -> bool:
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
) -> dict[str, torch.Tensor]:
    if not targets or any(not isinstance(target, str) or not target for target in targets):
        raise ValueError("trajectory snapshot targets must be non-empty strings")
    if dtype not in DTYPES:
        raise ValueError(f"unsupported trajectory snapshot dtype: {dtype}")
    selected = {
        name: parameter.detach().to(device="cpu", dtype=DTYPES[dtype]).contiguous()
        for name, parameter in model.named_parameters()
        if parameter_matches(name, targets, layers)
    }
    if not selected:
        raise ValueError(f"no parameters matched trajectory targets: {targets}")
    return selected


def snapshot_path(out_dir: Path, step: int) -> Path:
    if step < 0:
        raise ValueError("trajectory snapshot step must be non-negative")
    return out_dir / "parameter_trajectory" / f"step_{step:06d}.pt"


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


def validate_arguments(args: argparse.Namespace) -> None:
    if args.trajectory_snapshot_interval < 0:
        raise ValueError("--trajectory-snapshot-interval must be non-negative")
    if args.trajectory_snapshot_interval > 0:
        targets = args.trajectory_snapshot_targets
        if not isinstance(targets, list) or not targets:
            raise ValueError("--trajectory-snapshot-targets must be non-empty")
        if any(not isinstance(target, str) or not target for target in targets):
            raise ValueError("--trajectory-snapshot-targets must contain non-empty strings")
        layers = getattr(args, "trajectory_snapshot_layers", None)
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
