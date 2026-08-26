#!/usr/bin/env python3
"""Audit whether common-initialization MLP weight paths share an affine basis.

This zero-update analysis compares two dense-Muon runs that have identical
step-zero weights and optimizer schedules but disjoint training-data RNG
streams. It distinguishes low temporal dimension within one realized path
from transfer of that path's dense ambient embedding to another path.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import (
    chronological_splits,
    energy_capture,
    file_sha256,
    spectrum_record,
)
from examples.nanogpt.analyze_mlp_optimizer_probe_span import select_parameter
from examples.nanogpt.analyze_mlp_raw_gradient_rolling_prediction import phase_for_step
from examples.nanogpt.analyze_mlp_tangent_drift import temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_weight_run(
    probe_dir: Path,
    *,
    layer: int,
    targets: set[str],
) -> tuple[list[int], dict[str, list[torch.Tensor]], dict[str, Any]]:
    paths = sorted(probe_dir.glob("step_*.pt"))
    if len(paths) < 3:
        raise ValueError(f"at least three probes are required: {probe_dir}")
    steps: list[int] = []
    inventory: dict[str, list[torch.Tensor]] = {}
    identity: str | None = None
    provenance: dict[str, Any] | None = None
    files: list[dict[str, Any]] = []
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
            if select_parameter(name, {layer}, targets)
        }
        if not selected:
            raise ValueError(f"probe has no selected MLP parameters: {path}")
        if inventory and set(selected) != set(inventory):
            raise ValueError("optimizer probe parameter inventory changed")
        for name, state in selected.items():
            if "weight_before_step" not in state:
                raise ValueError(f"weight_before_step is missing for {name}")
            inventory.setdefault(name, []).append(
                state["weight_before_step"].contiguous()
            )
        steps.append(step)
        files.append({"path": str(path), "bytes": path.stat().st_size})
        del payload
    return steps, inventory, {
        "run_identity_sha256": identity,
        "execution_provenance": provenance,
        "files": files,
        "bytes": sum(row["bytes"] for row in files),
    }


def nested_bases(rows: torch.Tensor, maximum_rank: int) -> dict[str, torch.Tensor]:
    if rows.ndim != 2 or not 1 <= maximum_rank <= rows.shape[0]:
        raise ValueError("basis rank must fit a two-dimensional sample matrix")
    _uncentered, _values, centered = temporal_basis(
        rows, maximum_rank=maximum_rank
    )
    mean = rows.mean(dim=0).unsqueeze(1)
    mean = mean / mean.norm().clamp_min(1e-30)
    if maximum_rank == 1:
        affine = mean
    else:
        affine = torch.linalg.qr(
            torch.cat((mean, centered[:, : maximum_rank - 1]), dim=1),
            mode="reduced",
        ).Q
    return {
        "centered": centered[:, :maximum_rank],
        "mean_plus_centered": affine[:, :maximum_rank],
    }


def row_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    numerator = (first.double() * second.double()).sum(dim=1)
    denominator = (
        first.double().square().sum(dim=1).sqrt()
        * second.double().square().sum(dim=1).sqrt()
    ).clamp_min(1e-30)
    return numerator / denominator


def row_norm_ratio(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (
        second.double().square().sum(dim=1).sqrt()
        / first.double().square().sum(dim=1).sqrt().clamp_min(1e-30)
    )


def summarize_metric(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    metric: str,
) -> list[dict[str, Any]]:
    groups = sorted({tuple(row[key] for key in keys) for row in rows})
    result: list[dict[str, Any]] = []
    for group in groups:
        members = [row for row in rows if tuple(row[key] for key in keys) == group]
        values = torch.tensor(
            [float(row[metric]) for row in members], dtype=torch.float64
        )
        item = {key: value for key, value in zip(keys, group, strict=True)}
        item.update(
            sample_count=len(members),
            mean=float(values.mean()),
            minimum=float(values.min()),
            maximum=float(values.max()),
        )
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a-probe-dir", required=True, type=Path)
    parser.add_argument("--run-b-probe-dir", required=True, type=Path)
    parser.add_argument("--run-a-name", default="stream_a")
    parser.add_argument("--run-b-name", default="stream_b")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--ranks", default="1,3,6,12,16,24,48")
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--full-rank-gate", type=int, default=16)
    parser.add_argument("--full-capture-gate", type=float, default=0.90)
    parser.add_argument("--causal-rank-gate", type=int, default=48)
    parser.add_argument("--causal-capture-gate", type=float, default=0.80)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {value for value in args.targets.split(",") if value}
    ranks = [int(value) for value in args.ranks.split(",")]
    if not targets or ranks != sorted(set(ranks)) or ranks[0] < 1:
        raise ValueError("targets and ranks must be nonempty ordered unique values")
    if args.full_rank_gate not in ranks or args.causal_rank_gate not in ranks:
        raise ValueError("binding ranks must appear in --ranks")

    steps_a, run_a, metadata_a = load_weight_run(
        args.run_a_probe_dir, layer=args.layer, targets=targets
    )
    steps_b, run_b, metadata_b = load_weight_run(
        args.run_b_probe_dir, layer=args.layer, targets=targets
    )
    if steps_a != steps_b:
        raise ValueError("the two runs do not have identical probe steps")
    if set(run_a) != set(run_b):
        raise ValueError("the two runs do not have identical parameter inventories")
    steps = steps_a
    splits = chronological_splits(
        steps,
        discovery_stop=args.discovery_stop,
        validation_stop=args.validation_stop,
    )
    all_indices = list(range(len(steps)))
    names = (args.run_a_name, args.run_b_name)
    spectrum_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []

    for parameter in sorted(run_a):
        initial_equal = torch.equal(run_a[parameter][0], run_b[parameter][0])
        maximum_difference = float(
            (run_a[parameter][0].float() - run_b[parameter][0].float()).abs().max()
        )
        identity_rows.append(
            {
                "parameter": parameter,
                "step": steps[0],
                "bitwise_equal": initial_equal,
                "maximum_absolute_difference": maximum_difference,
            }
        )
        if not initial_equal:
            raise ValueError(f"step-zero weight mismatch for {parameter}")

        stacked = {
            names[0]: torch.stack(run_a[parameter]).to(args.device, torch.float32),
            names[1]: torch.stack(run_b[parameter]).to(args.device, torch.float32),
        }
        displacements = {
            name: (values - values[0:1]).flatten(1)
            for name, values in stacked.items()
        }

        for run_name, rows in displacements.items():
            for centered in (False, True):
                spectrum_rows.append(
                    {
                        "parameter": parameter,
                        "run": run_name,
                        "path": "individual",
                        **spectrum_record(rows, centered=centered),
                    }
                )
        joint = torch.cat((displacements[names[0]], displacements[names[1]]), dim=0)
        for centered in (False, True):
            spectrum_rows.append(
                {
                    "parameter": parameter,
                    "run": "joint",
                    "path": "two_stream_union",
                    **spectrum_record(joint, centered=centered),
                }
            )
        del joint

        cosines = row_cosine(displacements[names[0]], displacements[names[1]])
        norm_ratios = row_norm_ratio(displacements[names[0]], displacements[names[1]])
        relative_errors = (
            (displacements[names[0]] - displacements[names[1]])
            .double()
            .square()
            .sum(dim=1)
            .sqrt()
            / displacements[names[0]]
            .double()
            .square()
            .sum(dim=1)
            .sqrt()
            .clamp_min(1e-30)
        )
        for index, step in enumerate(steps):
            matched_rows.append(
                {
                    "parameter": parameter,
                    "probe_index": index,
                    "step": step,
                    "split": phase_for_step(
                        step, args.discovery_stop, args.validation_stop
                    ),
                    "displacement_cosine": float(cosines[index]),
                    "stream_b_over_stream_a_norm": float(norm_ratios[index]),
                    "difference_over_stream_a_norm": float(relative_errors[index]),
                }
            )

        for source_name, target_name in ((names[0], names[1]), (names[1], names[0])):
            source = displacements[source_name]
            target = displacements[target_name]
            fits = {
                "full_source": (source, {"all": all_indices}),
                "discovery_source": (
                    source[splits["discovery"]],
                    {**splits, "all": all_indices},
                ),
            }
            for fit_name, (fit_rows, eval_splits) in fits.items():
                bases = nested_bases(fit_rows, max(ranks))
                for basis_kind, maximum_basis in bases.items():
                    for rank in ranks:
                        basis = maximum_basis[:, :rank]
                        for eval_split, indices in eval_splits.items():
                            for eval_run, eval_rows in (
                                (source_name, source),
                                (target_name, target),
                            ):
                                basis_rows.append(
                                    {
                                        "parameter": parameter,
                                        "source_run": source_name,
                                        "target_run": target_name,
                                        "fit_window": fit_name,
                                        "basis_kind": basis_kind,
                                        "rank": rank,
                                        "eval_run": eval_run,
                                        "eval_relation": (
                                            "self" if eval_run == source_name else "cross"
                                        ),
                                        "eval_split": eval_split,
                                        "sample_count": len(indices),
                                        "displacement_energy_capture": energy_capture(
                                            eval_rows[indices], basis
                                        ),
                                    }
                                )

    full_binding = [
        row
        for row in basis_rows
        if row["fit_window"] == "full_source"
        and row["basis_kind"] == "mean_plus_centered"
        and row["rank"] == args.full_rank_gate
        and row["eval_relation"] == "cross"
        and row["eval_split"] == "all"
    ]
    causal_binding = [
        row
        for row in basis_rows
        if row["fit_window"] == "discovery_source"
        and row["basis_kind"] == "mean_plus_centered"
        and row["rank"] == args.causal_rank_gate
        and row["eval_relation"] == "cross"
        and row["eval_split"] == "test"
    ]
    if len(full_binding) != 2 * len(run_a) or len(causal_binding) != 2 * len(run_a):
        raise RuntimeError("binding gate inventory is incomplete")
    full_minimum = min(float(row["displacement_energy_capture"]) for row in full_binding)
    causal_minimum = min(
        float(row["displacement_energy_capture"]) for row in causal_binding
    )
    gate = {
        "step_zero_bitwise_equal": all(row["bitwise_equal"] for row in identity_rows),
        "full_source_rank": args.full_rank_gate,
        "full_source_cross_capture_minimum": full_minimum,
        "full_source_cross_capture_threshold": args.full_capture_gate,
        "causal_discovery_rank": args.causal_rank_gate,
        "causal_cross_test_capture_minimum": causal_minimum,
        "causal_cross_test_capture_threshold": args.causal_capture_gate,
        "shared_weight_path_basis_authorized": (
            full_minimum >= args.full_capture_gate
            and causal_minimum >= args.causal_capture_gate
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "identity": args.output / "step_zero_identity.csv",
        "spectra": args.output / "weight_path_spectra.csv",
        "basis": args.output / "weight_path_basis_transfer.csv",
        "matched": args.output / "matched_step_displacement.csv",
        "matched_summary": args.output / "matched_step_displacement_summary.csv",
        "gate": args.output / "gate.json",
    }
    write_csv(paths["identity"], identity_rows)
    write_csv(paths["spectra"], spectrum_rows)
    write_csv(paths["basis"], basis_rows)
    write_csv(paths["matched"], matched_rows)
    write_csv(
        paths["matched_summary"],
        summarize_metric(matched_rows, ("parameter", "split"), "displacement_cosine"),
    )
    paths["gate"].write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_hashes = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }
    metadata = {
        "schema_version": "nanogpt_mlp_disjoint_data_state_transfer_v1",
        "method": "common-gauge cross-data affine MLP displacement transfer",
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
        "command": [str(Path(__file__).resolve()), *list(__import__("sys").argv[1:])],
        "runs": {names[0]: metadata_a, names[1]: metadata_b},
        "steps": steps,
        "split": {
            "discovery_step_lt": args.discovery_stop,
            "validation_step_lt": args.validation_stop,
            "test_step_gte": args.validation_stop,
        },
        "ranks": ranks,
        "binding_gate": gate,
        "outputs": output_hashes,
        "runtime_seconds": time.time() - started,
        "limitations": [
            "Two data streams do not prove transfer across model seeds or tasks.",
            "The full-source fit is a noncausal necessary oracle, not a deployable decoder.",
            "Euclidean displacement capture is not a task-metric or loss-space guarantee.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": gate, "metadata": str(metadata_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
