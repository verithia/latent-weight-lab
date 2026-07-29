#!/usr/bin/env python3
"""Fit sparse Givens flows whose connectivity comes from activation geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    collect_activations,
    file_sha256,
    git_commit,
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_global_givens_transport_fit import (
    fit_global_givens_transport,
    parse_cells,
)
from examples.nanogpt.analyze_mlp_orthogonal_transport_oracle import (
    orthogonal_transport_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
    write_csv,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)


RANDOM_BASELINE = {
    8: 0.005192742770692878,
    16: 0.01036702612531659,
    32: 0.020558347308714874,
}


def tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(memoryview(value.detach().cpu().contiguous().numpy()))
    return digest.hexdigest()


def activation_matched_permutations(
    values: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Greedily edge-color the strongest activation-correlation graph.

    Every stage is a perfect matching, and no channel pair is reused.  Only
    integer connectivity is returned; no dense eigenbasis or learned factor
    is stored.
    """
    if values.ndim != 2 or values.shape[1] % 2:
        raise ValueError("activations must be [samples, positive even width]")
    width = int(values.shape[1])
    if stages <= 0 or neighbors < stages or neighbors >= width:
        raise ValueError("require 0 < stages <= neighbors < width")
    matrix = values.float()
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    matrix = matrix / matrix.square().sum(dim=0).sqrt().clamp_min(1e-12)
    correlation = (matrix.T @ matrix).abs()
    correlation.fill_diagonal_(-1.0)
    scores, indices = torch.topk(correlation, k=neighbors, dim=1)
    scores = scores.cpu()
    indices = indices.cpu()
    del correlation, matrix

    edge_scores: dict[tuple[int, int], float] = {}
    for left in range(width):
        for raw_score, raw_right in zip(
            scores[left].tolist(),
            indices[left].tolist(),
            strict=True,
        ):
            right = int(raw_right)
            edge = (left, right) if left < right else (right, left)
            edge_scores[edge] = max(
                float(raw_score), edge_scores.get(edge, -1.0)
            )
    ordered_edges = sorted(
        (
            (score, left, right)
            for (left, right), score in edge_scores.items()
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_edges: set[tuple[int, int]] = set()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutations: list[torch.Tensor] = []
    diagnostics: list[dict[str, float | int]] = []
    for stage in range(stages):
        occupied = [False] * width
        pairs: list[tuple[int, int, float, bool]] = []
        for score, left, right in ordered_edges:
            edge = (left, right)
            if (
                edge in used_edges
                or occupied[left]
                or occupied[right]
            ):
                continue
            occupied[left] = occupied[right] = True
            used_edges.add(edge)
            pairs.append((left, right, score, True))
            if len(pairs) == width // 2:
                break
        remaining = [index for index, value in enumerate(occupied) if not value]
        if remaining:
            order = torch.randperm(
                len(remaining), generator=generator
            ).tolist()
            remaining = [remaining[index] for index in order]
        while remaining:
            left = remaining.pop()
            partner_index = next(
                (
                    index
                    for index, right in enumerate(remaining)
                    if (
                        (min(left, right), max(left, right))
                        not in used_edges
                    )
                ),
                None,
            )
            if partner_index is None:
                # The strong-edge greedy pass can leave two vertices whose
                # edge appeared in an earlier stage.  Repair one current-stage
                # pair with a two-edge swap.  The complement of at most
                # ``stage`` used partners per vertex is extremely dense, so
                # this deterministic 2-opt repair is sufficient without
                # weakening the no-reused-pair invariant.
                repaired = False
                for right_index, right in enumerate(remaining):
                    for pair_index, (
                        prior_left,
                        prior_right,
                        _prior_score,
                        _prior_candidate,
                    ) in enumerate(pairs):
                        prior_edge = (
                            min(prior_left, prior_right),
                            max(prior_left, prior_right),
                        )
                        for first, second in (
                            (prior_left, prior_right),
                            (prior_right, prior_left),
                        ):
                            first_edge = (
                                min(left, first),
                                max(left, first),
                            )
                            second_edge = (
                                min(right, second),
                                max(right, second),
                            )
                            if (
                                first_edge in used_edges
                                or second_edge in used_edges
                                or first_edge == second_edge
                            ):
                                continue
                            used_edges.remove(prior_edge)
                            used_edges.add(first_edge)
                            used_edges.add(second_edge)
                            pairs[pair_index] = (
                                left,
                                first,
                                edge_scores.get(first_edge, 0.0),
                                first_edge in edge_scores,
                            )
                            pairs.append(
                                (
                                    right,
                                    second,
                                    edge_scores.get(second_edge, 0.0),
                                    second_edge in edge_scores,
                                )
                            )
                            remaining.pop(right_index)
                            repaired = True
                            break
                        if repaired:
                            break
                    if repaired:
                        break
                if not repaired:
                    raise RuntimeError(
                        "could not complete a unique activation matching"
                    )
                continue
            right = remaining.pop(partner_index)
            edge = (min(left, right), max(left, right))
            used_edges.add(edge)
            pairs.append(
                (
                    left,
                    right,
                    float(scores[left][indices[left] == right][0])
                    if bool((indices[left] == right).any())
                    else 0.0,
                    False,
                )
            )
        if len(pairs) != width // 2:
            raise RuntimeError("activation matching is incomplete")
        permutation = torch.tensor(
            [index for left, right, _score, _candidate in pairs for index in (left, right)],
            dtype=torch.long,
        )
        if not torch.equal(torch.sort(permutation).values, torch.arange(width)):
            raise RuntimeError("activation matching is not a permutation")
        permutations.append(permutation)
        diagnostics.append(
            {
                "stage": stage,
                "pairs": len(pairs),
                "candidate_edge_fraction": (
                    sum(candidate for *_rest, candidate in pairs)
                    / len(pairs)
                ),
                "mean_abs_correlation": (
                    sum(score for _left, _right, score, _candidate in pairs)
                    / len(pairs)
                ),
            }
        )
    return torch.stack(permutations), diagnostics


def aggregate(
    rows: list[dict[str, Any]],
    stage_counts: list[int],
) -> tuple[list[dict[str, Any]], str]:
    output: list[dict[str, Any]] = []
    for window in ("fit", "holdout"):
        for stages in stage_counts:
            selected = [
                row
                for row in rows
                if row["connectivity_window"] == window
                and int(row["stages"]) == stages
            ]
            energy = torch.tensor(
                [float(row["chord_fro"]) ** 2 for row in selected],
                dtype=torch.float64,
            )
            recovery = torch.tensor(
                [float(row["endpoint_recovery"]) for row in selected],
                dtype=torch.float64,
            )
            weighted = float((energy * recovery).sum() / energy.sum())
            baseline = RANDOM_BASELINE[stages]
            output.append(
                {
                    "connectivity_window": window,
                    "stages": stages,
                    "cells": len(selected),
                    "coordinates_per_layer": int(selected[0]["coordinates"]),
                    "coordinate_fraction": float(
                        selected[0]["coordinate_fraction"]
                    ),
                    "energy_weighted_endpoint_recovery": weighted,
                    "recovery_over_random_connectivity": weighted / baseline,
                    "minimum_endpoint_recovery": min(
                        float(row["endpoint_recovery"]) for row in selected
                    ),
                    "minimum_recovery_over_random_connectivity": min(
                        float(row["endpoint_recovery"]) / baseline
                        for row in selected
                    ),
                }
            )
    selected32 = [row for row in output if int(row["stages"]) == 32]
    if (
        len(selected32) == 2
        and min(
            float(row["recovery_over_random_connectivity"])
            for row in selected32
        )
        >= 2.0
        and min(
            float(row["minimum_recovery_over_random_connectivity"])
            for row in selected32
        )
        >= 1.5
    ):
        decision = "PROMOTE_ACTIVATION_MATCHED_GIVENS_TO_124M_TRAINING"
    else:
        decision = "REJECT_SPARSE_ACTIVATION_MATCHED_GIVENS"
    return output, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cells", default="0:0,0:180,6:60,11:120")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--stages", default="8,16,32")
    parser.add_argument("--neighbors", type=int, default=128)
    parser.add_argument("--sample-cap", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--fit-seed", type=int, default=20260729)
    parser.add_argument("--holdout-seed", type=int, default=20260730)
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--matching-seed", type=int, default=314159)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    cells = parse_cells(args.cells)
    boundaries = parse_int_list(args.phase_boundaries)
    stage_counts = parse_int_list(args.stages)
    if any(stage not in RANDOM_BASELINE for stage in stage_counts):
        raise ValueError("stage counts lack registered random baselines")
    end_by_start = dict(zip(boundaries[:-1], boundaries[1:], strict=True))
    layers = sorted({layer for layer, _phase in cells})
    required_steps = sorted(
        {step for _layer, phase in cells for step in (phase, end_by_start[phase])}
    )
    paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in required_steps]
    if any(not path.is_file() for path in paths):
        raise ValueError("required phase snapshots are absent")
    steps, values, snapshot_metadata = load_snapshots(
        paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    step_index = {step: index for index, step in enumerate(steps)}

    initial_path = args.snapshot_dir / "step_000000.pt"
    initial_payload = load_snapshot(initial_path)
    model = model_from_snapshot(initial_payload, args.device)
    batches_needed = (
        args.sample_cap + args.batch_size * args.block_size - 1
    ) // (args.batch_size * args.block_size)
    activation_windows: dict[str, dict[tuple[int, str], torch.Tensor]] = {}
    try:
        for name, seed in (
            ("fit", args.fit_seed),
            ("holdout", args.holdout_seed),
        ):
            batches = fixed_validation_batches(
                args.data_dir,
                args.batch_size,
                args.block_size,
                batches_needed,
                seed,
            )
            activation_windows[name] = collect_activations(
                model,
                batches,
                layers,
                args.sample_cap,
                args.device,
            )
    finally:
        del model, initial_payload
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    max_stages = max(stage_counts)
    connectivity: dict[tuple[str, int], torch.Tensor] = {}
    connectivity_diagnostics: dict[str, Any] = {}
    for window, collected in activation_windows.items():
        for layer in layers:
            permutations, diagnostics = activation_matched_permutations(
                collected[(layer, "post_gelu")].to(args.device),
                stages=max_stages,
                neighbors=args.neighbors,
                seed=args.matching_seed + 1009 * layer,
            )
            connectivity[(window, layer)] = permutations.cpu()
            connectivity_diagnostics[f"{window}_layer{layer}"] = {
                "sha256": tensor_sha256(permutations),
                "stages": diagnostics,
            }
    del activation_windows
    if "cuda" in args.device:
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for layer, phase_start in cells:
        name = f"transformer.h.{layer}.mlp.c_proj.weight"
        phase_end = end_by_start[phase_start]
        source = values[name][step_index[phase_start]].to(args.device)
        target = values[name][step_index[phase_end]].to(args.device)
        oracle = orthogonal_transport_metrics(source, target)
        for window in ("fit", "holdout"):
            for stages in stage_counts:
                fit = fit_global_givens_transport(
                    source,
                    target,
                    stages=stages,
                    seed=args.matching_seed,
                    steps=args.fit_steps,
                    learning_rate=args.learning_rate,
                    permutations=connectivity[(window, layer)][:stages],
                )
                row = {
                    "parameter": name,
                    "layer": layer,
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "connectivity_window": window,
                    "connectivity_sha256": tensor_sha256(
                        connectivity[(window, layer)][:stages]
                    ),
                    "right_oracle_recovery": oracle["right_endpoint_recovery"],
                    **fit,
                }
                row["fraction_of_right_oracle"] = (
                    float(row["endpoint_recovery"])
                    / float(row["right_oracle_recovery"])
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        del source, target

    aggregates, decision = aggregate(rows, stage_counts)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "activation_matched_givens_fit.csv"
    aggregate_path = args.output / "activation_matched_givens_fit_aggregate.csv"
    connectivity_path = args.output / "activation_matched_connectivity.pt"
    write_csv(detail_path, rows)
    write_csv(aggregate_path, aggregates)
    torch.save(
        {
            "schema_version": "activation_matched_givens_connectivity_v1",
            "source_step": 0,
            "fit_seed": args.fit_seed,
            "holdout_seed": args.holdout_seed,
            "neighbors": args.neighbors,
            "connectivity": {
                f"{window}_layer{layer}": value
                for (window, layer), value in connectivity.items()
            },
            "diagnostics": connectivity_diagnostics,
        },
        connectivity_path,
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_activation_matched_givens_fit_v1",
        "decision": decision,
        "decision_rule": (
            "promote only if both disjoint connectivity windows at 32 stages "
            "recover >=2x the registered random-connectivity aggregate and "
            "every pilot cell recovers >=1.5x its random aggregate"
        ),
        "random_connectivity_baseline": RANDOM_BASELINE,
        "snapshot_metadata": snapshot_metadata,
        "cells": [{"layer": layer, "phase_start": phase} for layer, phase in cells],
        "stage_counts": stage_counts,
        "neighbors": args.neighbors,
        "sample_cap": args.sample_cap,
        "fit_steps": args.fit_steps,
        "learning_rate": args.learning_rate,
        "connectivity_diagnostics": connectivity_diagnostics,
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "outputs": {
            "detail_sha256": file_sha256(detail_path),
            "aggregate_sha256": file_sha256(aggregate_path),
            "connectivity_sha256": file_sha256(connectivity_path),
        },
        "limitations": [
            "Angles are endpoint-fit oracles; only connectivity is causal and phase-start-derived.",
            "Connectivity is selected from step-0 post-GELU correlations and fixed across all phases.",
            "The pilot uses four preregistered cells before any language-model training.",
        ],
    }
    metadata_path = args.output / "activation_matched_givens_fit_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "aggregates": aggregates,
                "decision": decision,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
