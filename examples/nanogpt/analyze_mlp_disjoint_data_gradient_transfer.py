#!/usr/bin/env python3
"""Compare dense MLP gradient geometry across two common-gauge data streams.

The two registered runs must have the same model initialization and exact
probe steps.  This audit measures whether a compact basis learned on one data
stream transfers to the other, including ambient temporal bases, leading
singular-factor fields, grouped-Hadamard support, and causal factor transport.
It never updates language-model parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_gradient_factor_field import (
    coordinate_energy,
    fit_union_basis,
    frame_capture,
    grouped_fht_frame,
    support_capture,
)
from examples.nanogpt.analyze_mlp_highcadence_basis import (
    chronological_splits,
    energy_capture,
    file_sha256,
    phase_mean_rows,
    spectrum_record,
)
from examples.nanogpt.analyze_mlp_optimizer_probe_span import select_parameter
from examples.nanogpt.analyze_mlp_raw_gradient_factor_transport import (
    canonical_overlap,
    exact_singular_factors,
)
from examples.nanogpt.analyze_mlp_raw_gradient_rolling_prediction import (
    phase_for_step,
)
from examples.nanogpt.analyze_mlp_tangent_drift import temporal_basis
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION


REQUIRED_FIELDS = {"weight_before_step", "gradient_after_clip"}


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_raw_gradient_run(
    probe_dir: Path,
    *,
    layer: int,
    targets: set[str],
) -> tuple[list[int], dict[str, dict[str, list[torch.Tensor]]], dict[str, Any]]:
    paths = sorted(probe_dir.glob("step_*.pt"))
    if len(paths) < 3:
        raise ValueError(f"at least three probes are required: {probe_dir}")
    steps: list[int] = []
    inventory: dict[str, dict[str, list[torch.Tensor]]] = {}
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
            if not REQUIRED_FIELDS <= set(state):
                raise ValueError(f"raw gradient fields are incomplete for {name}")
            weight = state["weight_before_step"].contiguous()
            gradient = state["gradient_after_clip"].contiguous()
            if weight.shape != gradient.shape:
                raise ValueError(f"weight/gradient shape mismatch for {name}")
            destination = inventory.setdefault(name, {"weight": [], "gradient": []})
            destination["weight"].append(weight)
            destination["gradient"].append(-gradient)
        steps.append(step)
        files.append({"path": str(path), "bytes": path.stat().st_size})
        del payload
    return steps, inventory, {
        "run_identity_sha256": identity,
        "execution_provenance": provenance,
        "files": files,
        "bytes": sum(row["bytes"] for row in files),
    }


def affine_basis(rows: torch.Tensor, rank: int) -> dict[str, torch.Tensor]:
    if not 1 <= rank <= rows.shape[0]:
        raise ValueError("rank must fit the discovery sample count")
    _centered, _values, centered = temporal_basis(rows, maximum_rank=rank)
    mean = rows.mean(dim=0).unsqueeze(1)
    mean = mean / mean.norm().clamp_min(1e-30)
    if rank == 1:
        affine = mean
    else:
        affine = torch.linalg.qr(
            torch.cat((mean, centered[:, : rank - 1]), dim=1),
            mode="reduced",
        ).Q
    return {
        "discovery_centered": centered[:, :rank],
        "discovery_mean_plus_centered": affine[:, :rank],
    }


def row_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    numerator = (first.double() * second.double()).sum(dim=1)
    denominator = (
        first.double().square().sum(dim=1).sqrt()
        * second.double().square().sum(dim=1).sqrt()
    ).clamp_min(1e-30)
    return numerator / denominator


def summarize_rows(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    metric: str,
) -> list[dict[str, Any]]:
    groups = sorted({tuple(row[key] for key in keys) for row in rows})
    summary: list[dict[str, Any]] = []
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
            p10=float(torch.quantile(values, 0.10)),
        )
        summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a-probe-dir", required=True, type=Path)
    parser.add_argument("--run-b-probe-dir", required=True, type=Path)
    parser.add_argument("--run-a-name", default="stream_a")
    parser.add_argument("--run-b-name", default="stream_b")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--ranks", default="1,3,6,12,24,48")
    parser.add_argument("--factor-rank", type=int, default=6)
    parser.add_argument("--history-probes", type=int, default=10)
    parser.add_argument("--support-fractions", default="0.001,0.01")
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    targets = {value for value in args.targets.split(",") if value}
    ranks = [int(value) for value in args.ranks.split(",")]
    support_fractions = [float(value) for value in args.support_fractions.split(",")]
    if (
        not targets
        or ranks != sorted(set(ranks))
        or not all(0 < value <= 0.01 for value in support_fractions)
        or args.history_probes < 2
    ):
        raise ValueError("invalid targets, ranks, support fractions, or history")

    steps_a, run_a, metadata_a = load_raw_gradient_run(
        args.run_a_probe_dir, layer=args.layer, targets=targets
    )
    steps_b, run_b, metadata_b = load_raw_gradient_run(
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
    if max(ranks) > len(splits["discovery"]):
        raise ValueError("requested temporal rank exceeds discovery samples")

    initialization_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    factor_overlap_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []

    names = (args.run_a_name, args.run_b_name)
    for parameter in sorted(run_a):
        initial_equal = torch.equal(
            run_a[parameter]["weight"][0], run_b[parameter]["weight"][0]
        )
        initialization_rows.append(
            {
                "parameter": parameter,
                "step": steps[0],
                "bitwise_equal": initial_equal,
                "maximum_absolute_difference": float(
                    (
                        run_a[parameter]["weight"][0].float()
                        - run_b[parameter]["weight"][0].float()
                    ).abs().max()
                ),
            }
        )
        if not initial_equal:
            raise ValueError(f"step-zero weight mismatch for {parameter}")

        matrices = {
            names[0]: torch.stack(run_a[parameter]["gradient"]).to(
                args.device, dtype=torch.float32
            ),
            names[1]: torch.stack(run_b[parameter]["gradient"]).to(
                args.device, dtype=torch.float32
            ),
        }
        flattened = {name: value.flatten(1) for name, value in matrices.items()}
        for run_name, rows in flattened.items():
            total = rows.double().square().sum().clamp_min(1e-30)
            mean = rows.mean(dim=0)
            mean_fraction = float(rows.shape[0] * mean.double().square().sum() / total)
            for centered in (False, True):
                spectrum_rows.append(
                    {
                        "parameter": parameter,
                        "run": run_name,
                        **spectrum_record(rows, centered=centered),
                        "mean_direction_energy_fraction": mean_fraction,
                    }
                )
            drift_rows.extend(
                {
                    "parameter": parameter,
                    "run": run_name,
                    **row,
                }
                for row in phase_mean_rows(rows, splits)
            )

        for source_name, target_name in (
            (names[0], names[1]),
            (names[1], names[0]),
        ):
            source_discovery = flattened[source_name][splits["discovery"]]
            for rank in ranks:
                bases = affine_basis(source_discovery, rank)
                for basis_kind, basis in bases.items():
                    for split_name, indices in splits.items():
                        basis_rows.append(
                            {
                                "parameter": parameter,
                                "source_run": source_name,
                                "target_run": target_name,
                                "basis_kind": basis_kind,
                                "rank": rank,
                                "eval_split": split_name,
                                "sample_count": len(indices),
                                "energy_capture": energy_capture(
                                    flattened[target_name][indices], basis
                                ),
                            }
                        )

        cosines = row_cosine(flattened[names[0]], flattened[names[1]])
        for index, step in enumerate(steps):
            matched_rows.append(
                {
                    "parameter": parameter,
                    "probe_index": index,
                    "step": step,
                    "split": phase_for_step(
                        step, args.discovery_stop, args.validation_stop
                    ),
                    "gradient_cosine": float(cosines[index]),
                }
            )

        factor_fields: dict[str, dict[str, list[torch.Tensor]]] = {}
        for run_name, directions in matrices.items():
            left, singular, right = exact_singular_factors(
                directions, args.factor_rank
            )
            factor_fields[run_name] = {
                "left_output": left,
                "right_input": right,
                "singular": singular,
            }

        for index, step in enumerate(steps):
            for side in ("left_output", "right_input"):
                mean, minimum, maximum = canonical_overlap(
                    factor_fields[names[0]][side][index],
                    factor_fields[names[1]][side][index],
                )
                factor_overlap_rows.append(
                    {
                        "parameter": parameter,
                        "side": side,
                        "probe_index": index,
                        "step": step,
                        "split": phase_for_step(
                            step, args.discovery_stop, args.validation_stop
                        ),
                        "mean_squared_canonical_overlap": mean,
                        "minimum_squared_canonical_overlap": minimum,
                        "maximum_squared_canonical_overlap": maximum,
                    }
                )

        for source_name, target_name in (
            (names[0], names[1]),
            (names[1], names[0]),
        ):
            for side in ("left_output", "right_input"):
                native_source = factor_fields[source_name][side]
                native_target = factor_fields[target_name][side]
                for domain, source_frames, target_frames in (
                    ("native", native_source, native_target),
                    (
                        "grouped_fht",
                        [grouped_fht_frame(frame) for frame in native_source],
                        [grouped_fht_frame(frame) for frame in native_target],
                    ),
                ):
                    discovery_energy = sum(
                        coordinate_energy(source_frames[index])
                        for index in splits["discovery"]
                    )
                    dimension = source_frames[0].shape[0]
                    for fraction in support_fractions:
                        count = max(1, math.floor(fraction * dimension))
                        support = torch.topk(
                            discovery_energy, count, sorted=False
                        ).indices
                        for split_name, indices in splits.items():
                            captures = [
                                support_capture(target_frames[index], support)
                                for index in indices
                            ]
                            values = torch.tensor(captures, dtype=torch.float64)
                            support_rows.append(
                                {
                                    "parameter": parameter,
                                    "source_run": source_name,
                                    "target_run": target_name,
                                    "side": side,
                                    "domain": domain,
                                    "support_fraction": fraction,
                                    "active_coordinates": count,
                                    "ambient_coordinates": dimension,
                                    "eval_split": split_name,
                                    "sample_count": len(indices),
                                    "mean_capture": float(values.mean()),
                                    "minimum_capture": float(values.min()),
                                    "enrichment": float(values.mean() / fraction),
                                }
                            )

        new_fields = factor_fields[names[1]]
        for side in ("left_output", "right_input"):
            frames = new_fields[side]
            singular = new_fields["singular"]
            for index in range(args.history_probes, len(steps)):
                for rank in ranks:
                    basis = fit_union_basis(
                        frames,
                        singular,
                        range(index - args.history_probes, index),
                        min(rank, args.history_probes * args.factor_rank),
                    )
                    causal_rows.append(
                        {
                            "parameter": parameter,
                            "run": names[1],
                            "side": side,
                            "probe_index": index,
                            "step": steps[index],
                            "split": phase_for_step(
                                steps[index],
                                args.discovery_stop,
                                args.validation_stop,
                            ),
                            "union_rank": rank,
                            "current_frame_capture": frame_capture(
                                frames[index], basis
                            ),
                        }
                    )

        del matrices, flattened, factor_fields
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    matched_summary = summarize_rows(
        matched_rows, ("parameter", "split"), "gradient_cosine"
    )
    factor_summary = summarize_rows(
        factor_overlap_rows,
        ("parameter", "side", "split"),
        "mean_squared_canonical_overlap",
    )
    causal_summary = summarize_rows(
        causal_rows,
        ("parameter", "run", "side", "split", "union_rank"),
        "current_frame_capture",
    )

    rank48_gate = [
        row
        for row in basis_rows
        if row["basis_kind"] == "discovery_mean_plus_centered"
        and row["rank"] == 48
        and row["eval_split"] == "test"
    ]
    support_gate = [
        row
        for row in support_rows
        if row["domain"] == "grouped_fht"
        and abs(float(row["support_fraction"]) - 0.01) < 1e-12
        and row["eval_split"] == "test"
    ]
    gate = {
        "step_zero_bitwise_equal": all(
            bool(row["bitwise_equal"]) for row in initialization_rows
        ),
        "rank48_bidirectional_test_capture_minimum": min(
            float(row["energy_capture"]) for row in rank48_gate
        ),
        "rank48_threshold": 0.40,
        "grouped_fht_one_percent_test_capture_minimum": min(
            float(row["mean_capture"]) for row in support_gate
        ),
        "grouped_fht_one_percent_test_enrichment_minimum": min(
            float(row["enrichment"]) for row in support_gate
        ),
        "structured_capture_threshold": 0.20,
        "structured_enrichment_threshold": 5.0,
    }
    gate["shared_compact_basis_authorized"] = bool(
        gate["step_zero_bitwise_equal"]
        and gate["rank48_bidirectional_test_capture_minimum"] >= 0.40
        and gate["grouped_fht_one_percent_test_capture_minimum"] >= 0.20
        and gate["grouped_fht_one_percent_test_enrichment_minimum"] >= 5.0
    )

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "initialization": args.output / "step_zero_identity.csv",
        "spectrum": args.output / "gradient_spectrum.csv",
        "drift": args.output / "phase_mean_drift.csv",
        "basis": args.output / "cross_stream_basis_transfer.csv",
        "matched": args.output / "matched_step_gradient_cosine.csv",
        "matched_summary": args.output / "matched_step_gradient_cosine_summary.csv",
        "factor_overlap": args.output / "matched_step_factor_overlap.csv",
        "factor_summary": args.output / "matched_step_factor_overlap_summary.csv",
        "support": args.output / "cross_stream_structured_support.csv",
        "causal": args.output / "new_stream_causal_factor_capture.csv",
        "causal_summary": args.output / "new_stream_causal_factor_summary.csv",
        "gate": args.output / "gate.json",
    }
    table_rows = {
        "initialization": initialization_rows,
        "spectrum": spectrum_rows,
        "drift": drift_rows,
        "basis": basis_rows,
        "matched": matched_rows,
        "matched_summary": matched_summary,
        "factor_overlap": factor_overlap_rows,
        "factor_summary": factor_summary,
        "support": support_rows,
        "causal": causal_rows,
        "causal_summary": causal_summary,
    }
    for name, rows in table_rows.items():
        write_csv(outputs[name], rows)
    outputs["gate"].write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_disjoint_data_gradient_transfer_v1",
        "source_commit": git_commit(script.parents[2]),
        "entrypoint": str(script),
        "entrypoint_sha256": file_sha256(script),
        "command": sys.argv,
        "runs": {
            names[0]: metadata_a,
            names[1]: metadata_b,
        },
        "steps": steps,
        "split": {
            "discovery_step_lt": args.discovery_stop,
            "validation_step_lt": args.validation_stop,
            "test_step_gte": args.validation_stop,
        },
        "ranks": ranks,
        "factor_rank": args.factor_rank,
        "history_probes": args.history_probes,
        "support_fractions": support_fractions,
        "binding_gate": gate,
        "runtime_seconds": time.time() - started,
        "outputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in outputs.items()
        },
        "limitations": [
            "Two common-initialization data streams do not prove transfer across model seeds or tasks.",
            "Euclidean and factor-frame capture are necessary structural controls, not loss-space sufficiency.",
            "The empirical centered temporal rank remains bounded by the 100 probes.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "parameters": len(run_a),
                "sample_count": len(steps),
                "gate": gate,
                "metadata": str(metadata_path),
                "metadata_sha256": file_sha256(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
