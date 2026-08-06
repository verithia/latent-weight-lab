"""Atomic, model-only parameter snapshots for training-trajectory analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


SCHEMA_VERSION = "nanogpt_parameter_trajectory_v1"
FULL_STATE_SCHEMA_VERSION = "nanogpt_parameter_trajectory_v2"
SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {SCHEMA_VERSION, FULL_STATE_SCHEMA_VERSION}
)
OPTIMIZER_PROBE_SCHEMA_VERSION = "nanogpt_optimizer_probe_v2"
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
    candidates = dict(model.named_parameters())
    # Structured replacement weights are deliberately registered as
    # trainable persistent buffers so the target module does not duplicate
    # them as ordinary parameters.  Targeted trajectory acquisition must see
    # those tensors in the same namespace as ordinary weights.  Full-state
    # acquisition keeps its historical split (parameters here, buffers in
    # ``collect_persistent_buffers``) to avoid duplicating trainable buffers.
    if not all_parameters:
        candidates.update(
            {
                name: value
                for name, value in model.named_buffers()
                if value.requires_grad
            }
        )
    selected = {
        name: parameter.detach().to(device="cpu", dtype=DTYPES[dtype]).contiguous()
        for name, parameter in candidates.items()
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


def persistent_buffer_names(model: torch.nn.Module) -> set[str]:
    """Return unique named buffers that participate in ``state_dict``."""

    return set(dict(model.named_buffers())).intersection(model.state_dict())


def collect_persistent_buffers(
    model: torch.nn.Module,
    *,
    dtype: str,
) -> dict[str, torch.Tensor]:
    """Copy forward-critical persistent buffers without duplicating parameters."""

    if dtype not in DTYPES:
        raise ValueError(f"unsupported trajectory snapshot dtype: {dtype}")
    names = persistent_buffer_names(model)
    selected: dict[str, torch.Tensor] = {}
    for name, value in model.named_buffers():
        if name not in names:
            continue
        detached = value.detach()
        if detached.is_floating_point():
            detached = detached.to(device="cpu", dtype=DTYPES[dtype])
        else:
            detached = detached.to(device="cpu")
        selected[name] = detached.contiguous()
    return selected


def snapshot_path(out_dir: Path, step: int) -> Path:
    if step < 0:
        raise ValueError("trajectory snapshot step must be non-negative")
    return out_dir / "parameter_trajectory" / f"step_{step:06d}.pt"


def optimizer_probe_path(out_dir: Path, step: int) -> Path:
    if step < 0:
        raise ValueError("optimizer probe step must be non-negative")
    return out_dir / "optimizer_probe" / f"step_{step:06d}.pt"


@dataclass
class PendingOptimizerProbe:
    """CPU-resident pre-step state awaiting the real optimizer result.

    Preparing a probe may copy tensors from the accelerator, but it must not
    run an alternate optimizer calculation on the live accelerator.  The
    actual applied direction is reconstructed from the post-step weight.
    """

    destination: Path
    existing: bool
    payload: dict[str, Any]
    parameter_references: dict[str, torch.Tensor]
    optimizer_owners: dict[str, Any]


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
    all_buffers: bool = False,
    model_config: Any,
    run_identity: dict[str, Any],
    execution_provenance: dict[str, Any] | None,
) -> Path:
    if all_buffers and not all_parameters:
        raise ValueError(
            "all-buffer trajectory snapshots require all parameters"
        )
    destination = snapshot_path(out_dir, step)
    run_identity_sha256 = canonical_digest(run_identity)
    expected_schema_version = (
        FULL_STATE_SCHEMA_VERSION if all_buffers else SCHEMA_VERSION
    )
    if destination.exists():
        observed = torch.load(destination, map_location="cpu", weights_only=False)
        if (
            not isinstance(observed, dict)
            or observed.get("schema_version") != expected_schema_version
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
    buffers = (
        collect_persistent_buffers(model, dtype=dtype) if all_buffers else {}
    )
    buffer_inventory = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "bytes": value.numel() * value.element_size(),
        }
        for name, value in sorted(buffers.items())
    }
    payload = {
        "schema_version": expected_schema_version,
        "step": step,
        "targets": list(targets),
        "layers": None if layers is None else list(layers),
        "all_parameters": bool(all_parameters),
        "all_buffers": bool(all_buffers),
        "storage_dtype": dtype,
        "model_config": asdict(model_config),
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256,
        "execution_provenance": execution_provenance,
        "tensor_inventory": inventory,
        "parameters": parameters,
        "buffer_inventory": buffer_inventory,
        "buffers": buffers,
        "created_at_unix": time.time(),
    }
    _atomic_torch_save(destination, payload)
    return destination


def prepare_optimizer_probe(
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
) -> PendingOptimizerProbe:
    """Copy raw pre-step Muon state to CPU without extra accelerator math."""
    from examples.nanogpt.muon import Muon
    from examples.nanogpt.muon_matched_givens import (
        MuonDirectedProduct,
        MuonMatchedGivens,
    )

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
        return PendingOptimizerProbe(
            destination=destination,
            existing=True,
            payload={},
            parameter_references={},
            optimizer_owners={},
        )
    if dtype not in DTYPES:
        raise ValueError(f"unsupported optimizer probe dtype: {dtype}")
    if not targets or not layers:
        raise ValueError("optimizer probe targets and layers must be non-empty")

    suboptimizers = getattr(optimizer, "optimizers", [optimizer])
    supported_optimizers = (Muon, MuonDirectedProduct, MuonMatchedGivens)
    owners: dict[int, tuple[Any, dict[str, Any]]] = {}
    for candidate in suboptimizers:
        if not isinstance(candidate, supported_optimizers):
            continue
        for group in candidate.param_groups:
            for parameter in group["params"]:
                owners[id(parameter)] = (candidate, group)

    tensors: dict[str, dict[str, torch.Tensor]] = {}
    hyperparameters: dict[str, dict[str, Any]] = {}
    named_trainable_tensors = dict(model.named_parameters())
    named_trainable_tensors.update(
        {
            name: value
            for name, value in model.named_buffers()
            if value.requires_grad
        }
    )
    selected_parameters = {
        name: parameter
        for name, parameter in named_trainable_tensors.items()
        if parameter_matches(name, targets, layers)
    }
    if not selected_parameters:
        raise ValueError("no parameters matched the optimizer probe")
    storage_dtype = DTYPES[dtype]
    for name, parameter in sorted(selected_parameters.items()):
        owner = owners.get(id(parameter))
        if owner is None:
            raise ValueError(f"optimizer probe parameter is not Muon-owned: {name}")
        owned_optimizer, group = owner
        if parameter.grad is None:
            raise ValueError(f"optimizer probe parameter has no gradient: {name}")
        gradient = parameter.grad.detach()
        state = owned_optimizer.state.get(parameter, {})
        buffer = state.get("momentum_buffer")
        if buffer is None:
            buffer = torch.zeros_like(gradient, device="cpu")
        else:
            buffer = buffer.detach().to(
                device="cpu", dtype=storage_dtype, copy=True
            ).contiguous()
        momentum = float(group["momentum"])
        ns_steps = int(group["ns_steps"])
        weight_decay = float(group["weight_decay"])

        def cpu(value: torch.Tensor) -> torch.Tensor:
            return value.to(
                device="cpu", dtype=storage_dtype, copy=True
            ).contiguous()

        tensors[name] = {
            "weight_before_step": cpu(parameter.detach()),
            "gradient_after_clip": cpu(gradient),
            "momentum_buffer_before_step": buffer,
        }
        if isinstance(
            owned_optimizer, (MuonDirectedProduct, MuonMatchedGivens)
        ):
            compression_residual = state.get("compression_residual")
            if compression_residual is None:
                compression_residual = torch.zeros_like(
                    gradient, device="cpu"
                )
            else:
                compression_residual = compression_residual.detach().to(
                    device="cpu", dtype=storage_dtype, copy=True
                ).contiguous()
            tensors[name]["compression_residual_before_step"] = (
                compression_residual
            )
        rows = parameter.shape[0]
        columns = max(1, parameter.numel() / rows)
        scale = max(1.0, rows / columns) ** 0.5
        hyperparameters[name] = {
            "optimizer_kind": type(owned_optimizer).__name__,
            "lr": float(group["lr"]),
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "polar_scale": scale,
        }
        if isinstance(
            owned_optimizer, (MuonDirectedProduct, MuonMatchedGivens)
        ):
            hyperparameters[name].update(
                {
                    "error_feedback": bool(
                        group.get("error_feedback", False)
                    ),
                    "error_feedback_decay": float(
                        group.get("error_feedback_decay", 1.0)
                    ),
                    "error_feedback_max_nominal_steps": group.get(
                        "error_feedback_max_nominal_steps"
                    ),
                }
            )

    payload = {
        "schema_version": OPTIMIZER_PROBE_SCHEMA_VERSION,
        "capture_protocol": "pre_step_cpu_state_post_step_realized_direction_v2",
        "step": step,
        "targets": list(targets),
        "layers": list(layers),
        "storage_dtype": dtype,
        "model_config": asdict(model_config),
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256,
        "execution_provenance": execution_provenance,
        "hyperparameters": hyperparameters,
        "parameters": tensors,
        "prepared_at_unix": time.time(),
    }
    return PendingOptimizerProbe(
        destination=destination,
        existing=False,
        payload=payload,
        parameter_references=selected_parameters,
        optimizer_owners={
            name: owners[id(parameter)][0]
            for name, parameter in selected_parameters.items()
        },
    )


def write_optimizer_probe(pending: PendingOptimizerProbe) -> Path:
    """Seal a prepared probe after the real optimizer step.

    All derived matrix arithmetic runs on CPU after the live optimizer has
    already mutated the model.  ``applied_direction_per_lr`` is the realized
    weight-space action, including floating-point effects, rather than a
    second pre-step simulation of Muon on the training accelerator.
    """
    if pending.existing:
        return pending.destination
    payload = pending.payload
    storage_dtype = DTYPES[str(payload["storage_dtype"])]
    tensors = payload["parameters"]
    for name, parameter in pending.parameter_references.items():
        state = tensors[name]
        hyperparameters = payload["hyperparameters"][name]
        before = state["weight_before_step"].float()
        after = parameter.detach().to(
            device="cpu", dtype=storage_dtype, copy=True
        ).contiguous()
        gradient = state["gradient_after_clip"].float()
        buffer = state["momentum_buffer_before_step"].float()
        momentum = float(hyperparameters["momentum"])
        learning_rate = float(hyperparameters["lr"])
        weight_decay = float(hyperparameters["weight_decay"])
        polar_scale = float(hyperparameters["polar_scale"])
        if learning_rate == 0.0:
            raise ValueError("optimizer probe requires a non-zero learning rate")
        new_buffer = momentum * buffer + gradient
        combined = gradient + momentum * new_buffer
        realized = (after.float() - before) / learning_rate
        inferred_polar = -(
            realized + weight_decay * before
        ) / polar_scale

        def stored(value: torch.Tensor) -> torch.Tensor:
            return value.to(dtype=storage_dtype).contiguous()

        state.update(
            {
                "weight_after_step": after,
                "combined_momentum_update": stored(combined),
                "applied_direction_per_lr": stored(realized),
            }
        )
        optimizer_kind = str(hyperparameters.get("optimizer_kind", "Muon"))
        if optimizer_kind == "Muon":
            state["polar_update"] = stored(inferred_polar)
        else:
            owner = pending.optimizer_owners[name]
            optimizer_state = owner.state.get(parameter, {})
            momentum_after = optimizer_state.get("momentum_buffer")
            if momentum_after is None:
                raise ValueError(
                    f"structured optimizer probe lost momentum state: {name}"
                )
            state["momentum_buffer_after_step"] = stored(
                momentum_after.detach().float()
            )
            if "compression_residual_before_step" in state:
                compression_after = optimizer_state.get(
                    "compression_residual"
                )
                if compression_after is None:
                    compression_after = torch.zeros_like(
                        parameter, dtype=torch.float32
                    )
                state["compression_residual_after_step"] = stored(
                    compression_after.detach().float()
                )

    payload["tensor_inventory"] = {
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
    payload["created_at_unix"] = time.time()
    _atomic_torch_save(pending.destination, payload)
    return pending.destination


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
        "--trajectory-snapshot-all-buffers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "with --trajectory-snapshot-all-parameters, also store every "
            "unique persistent named model buffer needed for exact forward "
            "reconstruction"
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


def validate_arguments(args: argparse.Namespace) -> None:
    if args.trajectory_snapshot_interval < 0:
        raise ValueError("--trajectory-snapshot-interval must be non-negative")
    if args.trajectory_snapshot_interval > 0:
        targets = args.trajectory_snapshot_targets
        all_parameters = bool(
            getattr(args, "trajectory_snapshot_all_parameters", False)
        )
        all_buffers = bool(
            getattr(args, "trajectory_snapshot_all_buffers", False)
        )
        if all_buffers and not all_parameters:
            raise ValueError(
                "--trajectory-snapshot-all-buffers requires "
                "--trajectory-snapshot-all-parameters"
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
