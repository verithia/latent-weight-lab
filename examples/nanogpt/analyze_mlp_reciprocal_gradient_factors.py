#!/usr/bin/env python3
"""Audit reciprocal singular frames of paired dense MLP gradients.

For c_fc and c_proj at one layer, compare the two hidden-space frames and
the two residual-space frames at the same step and under strict preceding-
history prediction.  This is a zero-update structural diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_disjoint_data_gradient_transfer import (
    git_commit,
    load_raw_gradient_run,
    summarize_rows,
)
from examples.nanogpt.analyze_mlp_gradient_factor_field import (
    fit_union_basis,
    frame_capture,
)
from examples.nanogpt.analyze_mlp_highcadence_basis import (
    chronological_splits,
    file_sha256,
)
from examples.nanogpt.analyze_mlp_raw_gradient_factor_transport import (
    canonical_overlap,
    exact_singular_factors,
)
from examples.nanogpt.analyze_mlp_raw_gradient_rolling_prediction import (
    phase_for_step,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv


def paired_parameter_names(inventory: dict[str, Any]) -> tuple[str, str]:
    fc = [name for name in inventory if name.endswith("mlp.c_fc.weight")]
    proj = [name for name in inventory if name.endswith("mlp.c_proj.weight")]
    if len(fc) != 1 or len(proj) != 1:
        raise ValueError("exactly one c_fc and one c_proj parameter are required")
    if inventory[fc[0]]["gradient"][0].shape != tuple(
        reversed(inventory[proj[0]]["gradient"][0].shape)
    ):
        raise ValueError("paired MLP gradient shapes must be transposes")
    return fc[0], proj[0]


def factor_pair(
    inventory: dict[str, dict[str, list[torch.Tensor]]],
    *,
    factor_rank: int,
    device: str,
) -> dict[str, dict[str, Any]]:
    fc_name, proj_name = paired_parameter_names(inventory)
    fc = torch.stack(inventory[fc_name]["gradient"]).to(device, torch.float32)
    proj = torch.stack(inventory[proj_name]["gradient"]).to(device, torch.float32)
    fc_left, fc_singular, fc_right = exact_singular_factors(fc, factor_rank)
    proj_left, proj_singular, proj_right = exact_singular_factors(proj, factor_rank)
    return {
        "hidden": {
            "first_name": "c_fc_left_error",
            "first": fc_left,
            "first_singular": fc_singular,
            "second_name": "c_proj_right_activation",
            "second": proj_right,
            "second_singular": proj_singular,
        },
        "residual": {
            "first_name": "c_fc_right_input",
            "first": fc_right,
            "first_singular": fc_singular,
            "second_name": "c_proj_left_output_error",
            "second": proj_left,
            "second_singular": proj_singular,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a-probe-dir", required=True, type=Path)
    parser.add_argument("--run-b-probe-dir", required=True, type=Path)
    parser.add_argument("--run-a-name", default="stream_a")
    parser.add_argument("--run-b-name", default="stream_b")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--factor-rank", type=int, default=6)
    parser.add_argument("--history-probes", type=int, default=10)
    parser.add_argument("--union-ranks", default="6,12,24,48")
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    union_ranks = [int(value) for value in args.union_ranks.split(",")]
    if (
        args.factor_rank < 1
        or args.history_probes < 2
        or union_ranks != sorted(set(union_ranks))
        or max(union_ranks) > args.factor_rank * args.history_probes
    ):
        raise ValueError("invalid factor rank, history, or union ranks")

    runs: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    step_reference: list[int] | None = None
    for run_name, probe_dir in (
        (args.run_a_name, args.run_a_probe_dir),
        (args.run_b_name, args.run_b_probe_dir),
    ):
        steps, inventory, run_metadata = load_raw_gradient_run(
            probe_dir,
            layer=args.layer,
            targets={"mlp.c_fc", "mlp.c_proj"},
        )
        if step_reference is None:
            step_reference = steps
        elif steps != step_reference:
            raise ValueError("runs must have identical probe steps")
        runs[run_name] = factor_pair(
            inventory, factor_rank=args.factor_rank, device=args.device
        )
        metadata[run_name] = run_metadata
    assert step_reference is not None
    steps = step_reference
    splits = chronological_splits(
        steps,
        discovery_stop=args.discovery_stop,
        validation_stop=args.validation_stop,
    )

    overlap_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for run_name, spaces in runs.items():
        for space, fields in spaces.items():
            dimension = fields["first"][0].shape[0]
            chance = args.factor_rank / dimension
            for index, step in enumerate(steps):
                mean, minimum, maximum = canonical_overlap(
                    fields["first"][index], fields["second"][index]
                )
                overlap_rows.append(
                    {
                        "run": run_name,
                        "space": space,
                        "dimension": dimension,
                        "probe_index": index,
                        "step": step,
                        "split": phase_for_step(
                            step, args.discovery_stop, args.validation_stop
                        ),
                        "mean_squared_canonical_overlap": mean,
                        "minimum_squared_canonical_overlap": minimum,
                        "maximum_squared_canonical_overlap": maximum,
                        "chance_overlap": chance,
                        "chance_enrichment": mean / chance,
                    }
                )

            sides = (
                (
                    fields["first_name"],
                    fields["first"],
                    fields["first_singular"],
                ),
                (
                    fields["second_name"],
                    fields["second"],
                    fields["second_singular"],
                ),
            )
            for target_name, target_frames, _target_singular in sides:
                for source_name, source_frames, source_singular in sides:
                    relation = "self" if source_name == target_name else "reciprocal"
                    for index in range(args.history_probes, len(steps)):
                        history = range(index - args.history_probes, index)
                        for rank in union_ranks:
                            basis = fit_union_basis(
                                source_frames, source_singular, history, rank
                            )
                            history_rows.append(
                                {
                                    "run": run_name,
                                    "space": space,
                                    "source": source_name,
                                    "target": target_name,
                                    "relation": relation,
                                    "probe_index": index,
                                    "step": steps[index],
                                    "split": phase_for_step(
                                        steps[index],
                                        args.discovery_stop,
                                        args.validation_stop,
                                    ),
                                    "union_rank": rank,
                                    "current_frame_capture": frame_capture(
                                        target_frames[index], basis
                                    ),
                                }
                            )

    overlap_summary = summarize_rows(
        overlap_rows,
        ("run", "space", "split"),
        "mean_squared_canonical_overlap",
    )
    history_summary = summarize_rows(
        history_rows,
        (
            "run",
            "space",
            "source",
            "target",
            "relation",
            "split",
            "union_rank",
        ),
        "current_frame_capture",
    )
    overlap_gate_rows = [row for row in overlap_summary if row["split"] == "test"]
    history_gate_rows = [
        row
        for row in history_summary
        if row["relation"] == "reciprocal"
        and row["split"] == "test"
        and row["union_rank"] == 48
    ]
    gate = {
        "same_step_test_overlap_minimum": min(
            float(row["mean"]) for row in overlap_gate_rows
        ),
        "same_step_overlap_threshold": 0.40,
        "reciprocal_history_rank48_test_capture_minimum": min(
            float(row["mean"]) for row in history_gate_rows
        ),
        "reciprocal_history_capture_threshold": 0.40,
    }
    gate["online_reciprocal_factor_oracle_authorized"] = bool(
        gate["same_step_test_overlap_minimum"]
        >= gate["same_step_overlap_threshold"]
        and gate["reciprocal_history_rank48_test_capture_minimum"]
        >= gate["reciprocal_history_capture_threshold"]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "overlap": args.output / "same_step_reciprocal_overlap.csv",
        "overlap_summary": args.output / "same_step_reciprocal_overlap_summary.csv",
        "history": args.output / "causal_reciprocal_capture.csv",
        "history_summary": args.output / "causal_reciprocal_capture_summary.csv",
        "gate": args.output / "gate.json",
    }
    for path, rows in (
        (outputs["overlap"], overlap_rows),
        (outputs["overlap_summary"], overlap_summary),
        (outputs["history"], history_rows),
        (outputs["history_summary"], history_summary),
    ):
        write_csv(path, rows)
    outputs["gate"].write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    result = {
        "schema_version": "nanogpt_mlp_reciprocal_gradient_factors_v1",
        "source_commit": git_commit(script.parents[2]),
        "entrypoint": str(script),
        "entrypoint_sha256": file_sha256(script),
        "command": sys.argv,
        "runs": metadata,
        "steps": steps,
        "split": {
            "discovery_step_lt": args.discovery_stop,
            "validation_step_lt": args.validation_stop,
            "test_step_gte": args.validation_stop,
        },
        "factor_rank": args.factor_rank,
        "history_probes": args.history_probes,
        "union_ranks": union_ranks,
        "binding_gate": gate,
        "runtime_seconds": time.time() - started,
        "outputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in outputs.items()
        },
        "limitations": [
            "Reciprocal factor alignment does not itself encode persistent weights.",
            "Two data streams and one layer do not establish task-wide universality.",
            "A pass authorizes only a zero-update representation oracle, never training.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
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
