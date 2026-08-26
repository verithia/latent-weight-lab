#!/usr/bin/env python3
"""Analyze high-cadence MLP gradient, Muon, and applied-update spans.

The analyzer consumes exact optimizer probes from one registered run.  It
separates raw clipped gradients, prior momentum, combined Muon input, polar
directions, and actually applied directions; reports full-horizon spectra,
chronological basis transfer, phase mean drift, cross-operator capture, and
fixed-BlockFHT held-out controls.  No language-model parameters are updated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import (
    blockfht_capture,
    chronological_splits,
    energy_capture,
    file_sha256,
    parse_float_list,
    phase_mean_rows,
    spectrum_record,
)
from examples.nanogpt.analyze_mlp_chart_fit import TARGET_SEED_OFFSETS
from examples.nanogpt.analyze_mlp_tangent_drift import temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    parse_int_list,
    write_csv,
)
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


FIELD_ORIENTATION = {
    "raw_gradient_descent": ("gradient_after_clip", -1.0),
    "momentum_buffer_descent": ("momentum_buffer_before_step", -1.0),
    "combined_momentum_descent": ("combined_momentum_update", -1.0),
    "muon_polar_descent": ("polar_update", -1.0),
    "exact_applied_direction": ("applied_direction_per_lr", 1.0),
}


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def select_parameter(name: str, layers: set[int], targets: set[str]) -> bool:
    match = PARAMETER_PATTERN.match(name)
    return bool(
        match
        and int(match.group("layer")) in layers
        and match.group("target") in targets
    )


def load_probe_inventory(
    paths: list[Path],
    *,
    layers: set[int],
    targets: set[str],
) -> tuple[list[int], dict[str, dict[str, list[torch.Tensor]]], dict[str, Any]]:
    if len(paths) < 3:
        raise ValueError("at least three optimizer probes are required")
    steps: list[int] = []
    values: dict[str, dict[str, list[torch.Tensor]]] = {}
    identity: str | None = None
    provenance: dict[str, Any] | None = None
    inventories: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError(f"unexpected optimizer probe schema: {path}")
        step = int(payload.get("step", -1))
        if steps and step <= steps[-1]:
            raise ValueError("optimizer probe steps must be strictly increasing")
        observed_identity = payload.get("run_identity_sha256")
        if identity is None:
            identity = observed_identity
            provenance = payload.get("execution_provenance")
        elif observed_identity != identity:
            raise ValueError("optimizer probes do not share one run identity")
        selected = {
            name: state
            for name, state in payload.get("parameters", {}).items()
            if select_parameter(name, layers, targets)
        }
        if not selected:
            raise ValueError(f"probe has no selected MLP parameters: {path}")
        if values and set(selected) != set(values):
            raise ValueError("optimizer probe parameter inventory changed")
        for name, state in selected.items():
            if set(state) != {
                "weight_before_step",
                "gradient_after_clip",
                "momentum_buffer_before_step",
                "combined_momentum_update",
                "polar_update",
                "applied_direction_per_lr",
            }:
                raise ValueError(f"incomplete tensor inventory for {name}")
            destination = values.setdefault(
                name, {field: [] for field in FIELD_ORIENTATION}
            )
            reference_shape = state["weight_before_step"].shape
            for field, (source, sign) in FIELD_ORIENTATION.items():
                tensor = state[source]
                if tensor.shape != reference_shape:
                    raise ValueError(f"optimizer probe shape mismatch: {name}")
                destination[field].append(tensor.contiguous() * sign)
        steps.append(step)
        inventories.append(
            {"path": str(path), "bytes": path.stat().st_size}
        )
        del payload
    return steps, values, {
        "run_identity_sha256": identity,
        "execution_provenance": provenance,
        "files": inventories,
    }


def basis_columns(
    discovery: torch.Tensor,
    ranks: list[int],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    _centered, eigenvalues, basis = temporal_basis(
        discovery, maximum_rank=max(ranks)
    )
    return eigenvalues, basis, basis.shape[1]


def analyze_rows(
    rows: torch.Tensor,
    *,
    parameter: str,
    field: str,
    steps: list[int],
    discovery_stop: int,
    validation_stop: int,
    ranks: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    match = PARAMETER_PATTERN.match(parameter)
    assert match is not None
    common = {
        "parameter": parameter,
        "layer": int(match.group("layer")),
        "target": match.group("target"),
        "field": field,
    }
    splits = chronological_splits(
        steps,
        discovery_stop=discovery_stop,
        validation_stop=validation_stop,
    )
    total = rows.double().square().sum().clamp_min(1e-30)
    mean = rows.mean(dim=0)
    mean_fraction = float(rows.shape[0] * mean.double().square().sum() / total)
    spectra = [
        {
            **common,
            **spectrum_record(rows, centered=centered),
            "mean_direction_energy_fraction": mean_fraction,
        }
        for centered in (False, True)
    ]
    discovery = rows[splits["discovery"]]
    eigenvalues, basis, available = basis_columns(discovery, ranks)
    transfer: list[dict[str, Any]] = []
    for rank in ranks:
        if rank > available:
            continue
        centered_basis = basis[:, :rank]
        mean_column = discovery.mean(dim=0).unsqueeze(1)
        if rank == 1:
            affine_basis = mean_column / mean_column.norm().clamp_min(1e-30)
        else:
            affine_basis = torch.linalg.qr(
                torch.cat((mean_column, basis[:, : rank - 1]), dim=1),
                mode="reduced",
            ).Q
        for basis_kind, selected in (
            ("discovery_centered", centered_basis),
            ("discovery_mean_plus_centered", affine_basis),
        ):
            for split_name, indices in splits.items():
                transfer.append(
                    {
                        **common,
                        "basis_kind": basis_kind,
                        "rank": rank,
                        "split": split_name,
                        "sample_count": len(indices),
                        "energy_capture": energy_capture(rows[indices], selected),
                        "discovery_centered_eigen_energy": float(
                            eigenvalues[:rank].sum()
                            / eigenvalues.sum().clamp_min(1e-30)
                        ),
                    }
                )
    drift = [{**common, **record} for record in phase_mean_rows(rows, splits)]
    return spectra, transfer, drift


def cross_operator_rows(
    fields: dict[str, torch.Tensor],
    *,
    parameter: str,
    steps: list[int],
    discovery_stop: int,
    validation_stop: int,
    ranks: list[int],
) -> list[dict[str, Any]]:
    splits = chronological_splits(
        steps,
        discovery_stop=discovery_stop,
        validation_stop=validation_stop,
    )
    target_field = "exact_applied_direction"
    result: list[dict[str, Any]] = []
    for source_field in (
        "raw_gradient_descent",
        "combined_momentum_descent",
        "muon_polar_descent",
        "exact_applied_direction",
    ):
        source = fields[source_field]
        _values, basis, available = basis_columns(
            source[splits["discovery"]], ranks
        )
        for rank in ranks:
            if rank > available:
                continue
            selected = basis[:, :rank]
            for split_name, indices in splits.items():
                target = fields[target_field][indices]
                centered = target - target.mean(dim=0, keepdim=True)
                result.append(
                    {
                        "parameter": parameter,
                        "source_field": source_field,
                        "target_field": target_field,
                        "rank": rank,
                        "split": split_name,
                        "raw_target_energy_capture": energy_capture(target, selected),
                        "centered_target_energy_capture": energy_capture(
                            centered, selected
                        ),
                    }
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="6")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--ranks", default="1,2,4,8,16,32,48")
    parser.add_argument("--latent-ratios", default="0.001,0.0025,0.005,0.01")
    parser.add_argument("--latent-rows", type=int, default=0)
    parser.add_argument("--block-fht-layers", type=int, default=2)
    parser.add_argument("--block-fht-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = set(parse_int_list(args.layers))
    targets = {item for item in args.targets.split(",") if item}
    ranks = parse_int_list(args.ranks)
    ratios = parse_float_list(args.latent_ratios)
    if not layers or not targets or not ranks or ranks != sorted(set(ranks)):
        raise ValueError("layers, targets, and ordered ranks are required")
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, values, input_metadata = load_probe_inventory(
        paths, layers=layers, targets=targets
    )
    if not (
        steps[0] < args.discovery_stop < args.validation_stop <= steps[-1]
    ):
        raise ValueError("invalid chronological split")

    spectra: list[dict[str, Any]] = []
    transfer: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for parameter, stored_fields in sorted(values.items()):
        match = PARAMETER_PATTERN.match(parameter)
        assert match is not None
        device_fields: dict[str, torch.Tensor] = {}
        for field, tensors in stored_fields.items():
            rows = torch.stack(tensors).to(
                device=args.device, dtype=torch.float32
            ).flatten(1)
            device_fields[field] = rows
            field_result = analyze_rows(
                rows,
                parameter=parameter,
                field=field,
                steps=steps,
                discovery_stop=args.discovery_stop,
                validation_stop=args.validation_stop,
                ranks=ranks,
            )
            spectra.extend(field_result[0])
            transfer.extend(field_result[1])
            drift.extend(field_result[2])

            test_indices = [
                index
                for index, step in enumerate(steps)
                if step >= args.validation_stop
            ]
            raw_test = rows[test_indices]
            centered_test = raw_test - raw_test.mean(dim=0, keepdim=True)
            seed = (
                args.block_fht_seed
                + int(match.group("layer")) * 4
                + TARGET_SEED_OFFSETS[match.group("target")]
            )
            for ratio in ratios:
                for centered, selected in (
                    (False, raw_test),
                    (True, centered_test),
                ):
                    block_rows.append(
                        {
                            "parameter": parameter,
                            "layer": int(match.group("layer")),
                            "target": match.group("target"),
                            "field": field,
                            "split": "test",
                            "centered": centered,
                            "seed": seed,
                            **blockfht_capture(
                                selected,
                                ratio=ratio,
                                latent_rows=args.latent_rows,
                                layers=args.block_fht_layers,
                                seed=seed,
                            ),
                        }
                    )
        cross.extend(
            cross_operator_rows(
                device_fields,
                parameter=parameter,
                steps=steps,
                discovery_stop=args.discovery_stop,
                validation_stop=args.validation_stop,
                ranks=ranks,
            )
        )
        del device_fields
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    output_files = {
        "span_spectrum": args.output / "optimizer_probe_span_spectrum.csv",
        "basis_transfer": args.output / "optimizer_probe_basis_transfer.csv",
        "phase_mean_drift": args.output / "optimizer_probe_phase_mean_drift.csv",
        "cross_operator": args.output / "optimizer_probe_cross_operator_capture.csv",
        "fixed_blockfht": args.output / "optimizer_probe_fixed_blockfht_capture.csv",
    }
    for path, rows in (
        (output_files["span_spectrum"], spectra),
        (output_files["basis_transfer"], transfer),
        (output_files["phase_mean_drift"], drift),
        (output_files["cross_operator"], cross),
        (output_files["fixed_blockfht"], block_rows),
    ):
        write_csv(path, rows)
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_optimizer_probe_span_v1",
        "steps": steps,
        "sample_count": len(steps),
        "parameters": sorted(values),
        "fields": list(FIELD_ORIENTATION),
        "split": {
            "discovery_step_lt": args.discovery_stop,
            "validation_step_lt": args.validation_stop,
            "test_step_gte": args.validation_stop,
        },
        "ranks": ranks,
        "latent_ratios": ratios,
        "input": input_metadata,
        "method": {
            "orientations": FIELD_ORIENTATION,
            "spectrum": "exact temporal Gram spectrum, raw and mean-centered",
            "basis_transfer": "discovery-only centered PCs frozen before validation/test",
            "cross_operator": "discovery source-field PCs projected onto exact applied directions",
            "fixed_blockfht": "production-style exact Euclidean projector on untouched test probes",
        },
        "analysis_execution": {
            "git_commit": git_commit(script.parents[2]),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "outputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in output_files.items()
        },
        "limitations": [
            "One stochastic optimizer trajectory is not the global solution manifold.",
            "Euclidean capture is a necessary control, not a functional-loss guarantee.",
            "A 100-probe horizon bounds centered empirical rank by 99.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "sample_count": len(steps),
                "parameters": len(values),
                "spectra": len(spectra),
                "basis_rows": len(transfer),
                "cross_rows": len(cross),
                "blockfht_rows": len(block_rows),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
