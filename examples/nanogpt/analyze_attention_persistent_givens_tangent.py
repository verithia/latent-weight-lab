#!/usr/bin/env python3
"""Test whether step-0-selected sparse attention edges persist later.

Connectivity is selected once from the exact step-0 dense-Muon direction and
then frozen.  All decision metrics use only later directions and chords, so
the gate measures temporal persistence rather than in-sample edge fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_fht_block_skew_tangent import (
    TARGETS,
    file_sha256,
    project,
    select_target,
    weighted_summary,
    write_csv,
)
from examples.nanogpt.analyze_parameter_trajectory import load_snapshots
from examples.nanogpt.muon_matched_givens import (
    muon_matched_permutations,
    random_unique_matchings,
)
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION


Coordinates = tuple[torch.Tensor, ...]


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class FixedGivensSide:
    """Identity-tangent JVP and adjoint for frozen pair connectivity."""

    def __init__(
        self,
        *,
        weight: torch.Tensor,
        side: str,
        permutations: torch.Tensor,
    ) -> None:
        if side not in {"input", "output"}:
            raise ValueError(f"unsupported side {side}")
        self.weight = weight
        self.side = side
        self.values = weight if side == "input" else weight.T
        if (
            permutations.ndim != 2
            or permutations.shape[1] != self.values.shape[1]
        ):
            raise ValueError("permutation shape mismatch")
        self.permutations = permutations.to(weight.device)
        self.inverse_permutations = torch.argsort(
            self.permutations, dim=1
        )

    @property
    def coordinate_count(self) -> int:
        return int(self.permutations.numel() // 2)

    def zeros(self) -> torch.Tensor:
        return torch.zeros(
            self.permutations.shape[0],
            self.permutations.shape[1] // 2,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    def jvp(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape != self.zeros().shape:
            raise ValueError("coordinate shape mismatch")
        delta_values = torch.zeros_like(self.values)
        for stage in range(self.permutations.shape[0]):
            permutation = self.permutations[stage]
            inverse = self.inverse_permutations[stage]
            pairs = self.values.index_select(-1, permutation).reshape(
                self.values.shape[0], -1, 2
            )
            angle = coordinates[stage]
            delta_pairs = torch.stack(
                (-pairs[..., 1] * angle, pairs[..., 0] * angle), dim=-1
            )
            delta_values = delta_values + delta_pairs.reshape_as(
                self.values
            ).index_select(-1, inverse)
        if self.side == "input":
            return -delta_values
        return delta_values.T

    def adjoint(self, direction: torch.Tensor) -> torch.Tensor:
        direction_values = -direction if self.side == "input" else direction.T
        gradients: list[torch.Tensor] = []
        for stage in range(self.permutations.shape[0]):
            permutation = self.permutations[stage]
            source = self.values.index_select(-1, permutation).reshape(
                self.values.shape[0], -1, 2
            )
            cotangent = direction_values.index_select(
                -1, permutation
            ).reshape_as(source)
            gradients.append(
                (-source[..., 1] * cotangent[..., 0]
                 + source[..., 0] * cotangent[..., 1]).sum(dim=0)
            )
        return torch.stack(gradients)


class PersistentGivensTangent:
    def __init__(
        self,
        *,
        weight: torch.Tensor,
        sides: tuple[str, ...],
        permutations: dict[str, torch.Tensor],
        stages: int,
    ) -> None:
        self.weight = weight
        self.charts = tuple(
            FixedGivensSide(
                weight=weight,
                side=side,
                permutations=permutations[side][:stages],
            )
            for side in sides
        )

    @property
    def coordinate_count(self) -> int:
        return sum(chart.coordinate_count for chart in self.charts)

    def zeros(self) -> Coordinates:
        return tuple(chart.zeros() for chart in self.charts)

    def jvp(self, coordinates: Coordinates) -> torch.Tensor:
        return sum(
            (
                chart.jvp(value)
                for chart, value in zip(self.charts, coordinates, strict=True)
            ),
            torch.zeros_like(self.weight),
        )

    def adjoint(self, direction: torch.Tensor) -> Coordinates:
        return tuple(chart.adjoint(direction) for chart in self.charts)


def parameter_name(layer: int, target: str) -> str:
    suffix = "attn.c_proj" if target == "cproj" else "attn.c_attn"
    return f"transformer.h.{layer}.{suffix}.weight"


def select_connectivity(
    *,
    probe: dict[str, Any],
    layers: list[int],
    stages: int,
    neighbors: int,
    matching_seed: int,
    random_seed: int,
) -> tuple[
    dict[str, dict[tuple[int, str, str], torch.Tensor]],
    list[dict[str, Any]],
]:
    n_embd = int(probe["model_config"]["n_embd"])
    output: dict[str, dict[tuple[int, str, str], torch.Tensor]] = {
        "task_selected": {},
        "random": {},
    }
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for target_index, (target, metadata) in enumerate(TARGETS.items()):
            record = probe["parameters"][parameter_name(layer, target)]
            weight = select_target(
                record["weight_before_step"], target, n_embd
            ).float()
            direction = select_target(
                record["applied_direction_per_lr"], target, n_embd
            ).float()
            for side_index, side in enumerate(metadata["sides"]):
                source = weight if side == "input" else weight.T
                requested = direction if side == "input" else direction.T
                seed = (
                    matching_seed
                    + layer * 4096
                    + target_index * 256
                    + side_index * 128
                )
                selected, diagnostics = muon_matched_permutations(
                    source,
                    requested,
                    stages=stages,
                    neighbors=neighbors,
                    seed=seed,
                )
                random = random_unique_matchings(
                    width=source.shape[1],
                    stages=stages,
                    seed=random_seed + seed,
                )
                key = (layer, target, side)
                output["task_selected"][key] = selected
                output["random"][key] = random
                rows.append(
                    {
                        "layer": layer,
                        "target": target,
                        "side": side,
                        "stages": stages,
                        "neighbors": neighbors,
                        "task_selected_sha256": tensor_sha256(selected),
                        "random_sha256": tensor_sha256(random),
                        "mean_candidate_edge_fraction": sum(
                            float(row["candidate_edge_fraction"])
                            for row in diagnostics
                        ) / len(diagnostics),
                        "mean_abs_coordinate_gradient": sum(
                            float(row["mean_abs_coordinate_gradient"])
                            for row in diagnostics
                        ) / len(diagnostics),
                    }
                )
    return output, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema_version")
        != "mai_124m_attention_persistent_givens_gate_plan_v1"
    ):
        raise ValueError("unexpected plan schema")
    oracle = plan["oracle"]
    layers = [int(value) for value in oracle["layers"]]
    boundaries = [int(value) for value in oracle["phase_boundaries"]]
    selection_step = int(oracle["selection_step"])
    evaluation_steps = [int(value) for value in oracle["evaluation_steps"]]
    stage_counts = [int(value) for value in oracle["stage_counts"]]
    maximum_stages = max(stage_counts)
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries
    ]
    probe_steps = [selection_step, *evaluation_steps]
    probe_paths = [
        args.probe_dir / f"step_{step:06d}.pt" for step in probe_steps
    ]
    missing = [
        str(path)
        for path in (*snapshot_paths, *probe_paths)
        if not path.is_file()
    ]
    if missing:
        raise ValueError("missing inputs: " + ", ".join(missing))
    steps, values, snapshot_metadata = load_snapshots(
        snapshot_paths,
        layers=set(layers),
        targets={"attn.c_attn", "attn.c_proj"},
    )
    probes = {
        step: torch.load(path, map_location="cpu", weights_only=False)
        for step, path in zip(probe_steps, probe_paths, strict=True)
    }
    if any(
        probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION
        for probe in probes.values()
    ):
        raise ValueError("unexpected optimizer probe schema")
    identities = {probe["run_identity_sha256"] for probe in probes.values()}
    if identities != {snapshot_metadata["run_identity_sha256"]}:
        raise ValueError("snapshot and optimizer probe identities differ")
    connectivity, connectivity_rows = select_connectivity(
        probe=probes[selection_step],
        layers=layers,
        stages=maximum_stages,
        neighbors=int(oracle["neighbors"]),
        matching_seed=int(oracle["matching_seed"]),
        random_seed=int(oracle["random_seed"]),
    )
    step_index = {step: index for index, step in enumerate(steps)}
    end_by_start = dict(zip(boundaries[:-1], boundaries[1:], strict=True))
    rows: list[dict[str, Any]] = []
    for stages_count in stage_counts:
        for connectivity_name, selected_connectivity in connectivity.items():
            for phase_start in evaluation_steps:
                phase_end = end_by_start[phase_start]
                probe = probes[phase_start]
                n_embd = int(probe["model_config"]["n_embd"])
                for layer in layers:
                    for target, metadata in TARGETS.items():
                        name = parameter_name(layer, target)
                        record = probe["parameters"][name]
                        weight = select_target(
                            record["weight_before_step"], target, n_embd
                        ).to(args.device, dtype=torch.float32)
                        dense = select_target(
                            record["applied_direction_per_lr"], target, n_embd
                        ).to(args.device, dtype=torch.float32)
                        chord = select_target(
                            values[name][step_index[phase_end]]
                            - values[name][step_index[phase_start]],
                            target,
                            n_embd,
                        ).to(args.device, dtype=torch.float32)
                        permutations = {
                            side: selected_connectivity[(layer, target, side)]
                            for side in metadata["sides"]
                        }
                        chart = PersistentGivensTangent(
                            weight=weight,
                            sides=metadata["sides"],
                            permutations=permutations,
                            stages=stages_count,
                        )
                        for kind, requested in (
                            ("dense_muon_direction", dense),
                            ("phase_chord", chord),
                        ):
                            _, diagnostics = project(
                                chart,
                                requested,
                                maximum_iterations=int(oracle["cg_iterations"]),
                                tolerance=float(oracle["cg_tolerance"]),
                                ridge=float(oracle["ridge"]),
                            )
                            coordinate_fraction = (
                                chart.coordinate_count / weight.numel()
                            )
                            row = {
                                "stages": stages_count,
                                "connectivity": connectivity_name,
                                "selection_step": selection_step,
                                "phase_start": phase_start,
                                "phase_end": phase_end,
                                "layer": layer,
                                "target": target,
                                "kind": kind,
                                "coordinate_count": chart.coordinate_count,
                                "ambient_count": weight.numel(),
                                "coordinate_fraction": coordinate_fraction,
                                "normalized_enrichment": (
                                    diagnostics["energy_recovery"]
                                    / coordinate_fraction
                                ),
                                **diagnostics,
                            }
                            rows.append(row)
                            print(json.dumps(row, sort_keys=True), flush=True)
                        del chart, weight, dense, chord
                        if args.device.startswith("cuda"):
                            torch.cuda.empty_cache()
    thresholds = plan["decision_rule"]["thresholds"]
    summaries: dict[str, Any] = {}
    promoted: list[int] = []
    for stages_count in stage_counts:
        stage_summary: dict[str, Any] = {}
        for connectivity_name in connectivity:
            selected = [
                row
                for row in rows
                if int(row["stages"]) == stages_count
                and row["connectivity"] == connectivity_name
            ]
            stage_summary[connectivity_name] = {
                "dense_muon_direction": weighted_summary(
                    selected, "dense_muon_direction"
                ),
                "phase_chord": weighted_summary(selected, "phase_chord"),
                "dense_muon_direction_by_target": {
                    target: weighted_summary(
                        [row for row in selected if row["target"] == target],
                        "dense_muon_direction",
                    )
                    for target in TARGETS
                },
            }
        task = stage_summary["task_selected"]
        random = stage_summary["random"]
        dense = task["dense_muon_direction"]
        chord = task["phase_chord"]
        dense_over_random = dense["energy_recovery"] / max(
            random["dense_muon_direction"]["energy_recovery"], 1e-30
        )
        chord_over_random = chord["energy_recovery"] / max(
            random["phase_chord"]["energy_recovery"], 1e-30
        )
        passed = (
            dense["energy_recovery"]
            >= float(thresholds["aggregate_dense_recovery_minimum"])
            and dense["normalized_enrichment"]
            >= float(thresholds["dense_enrichment_minimum"])
            and dense_over_random
            >= float(thresholds["dense_over_random_minimum"])
            and chord["energy_recovery"]
            >= float(thresholds["aggregate_chord_recovery_minimum"])
            and chord_over_random
            >= float(thresholds["chord_over_random_minimum"])
            and all(
                summary["energy_recovery"]
                >= float(thresholds["per_target_dense_recovery_minimum"])
                for summary in task["dense_muon_direction_by_target"].values()
            )
            and max(
                dense["maximum_orthogonality_error"],
                chord["maximum_orthogonality_error"],
            )
            <= float(thresholds["maximum_projection_error"])
            and max(
                dense["maximum_relative_normal_residual"],
                chord["maximum_relative_normal_residual"],
            )
            <= float(thresholds["maximum_normal_residual"])
        )
        summaries[str(stages_count)] = {
            **stage_summary,
            "task_dense_over_random": dense_over_random,
            "task_chord_over_random": chord_over_random,
            "registered_gate_passed": passed,
        }
        if passed:
            promoted.append(stages_count)
    selected_stages = (
        max(
            promoted,
            key=lambda value: (
                summaries[str(value)]["task_selected"]
                ["dense_muon_direction"]["energy_recovery"]
                - summaries[str(value)]["task_selected"]
                ["dense_muon_direction"]["coordinate_fraction"],
                -value,
            ),
        )
        if promoted
        else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cells_path = args.output / "attention_persistent_givens_cells.csv"
    connectivity_path = args.output / "attention_persistent_givens_connectivity.pt"
    write_csv(cells_path, rows)
    torch.save(
        {"connectivity": connectivity, "summary": connectivity_rows},
        connectivity_path,
    )
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "mai_124m_attention_persistent_givens_tangent_v1",
        "source_commit": git_commit(repo_root),
        "source_sha256": file_sha256(Path(__file__)),
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "run_identity_sha256": snapshot_metadata["run_identity_sha256"],
        "selection": {
            "step": selection_step,
            "rows": connectivity_rows,
            "artifact": {
                "path": str(connectivity_path),
                "sha256": file_sha256(connectivity_path),
            },
        },
        "summaries": summaries,
        "decision": {
            "classification": (
                "PROMOTE_PERSISTENT_GIVENS"
                if selected_stages is not None
                else "REJECT_PERSISTENT_GIVENS"
            ),
            "selected_stages": selected_stages,
            "thresholds": thresholds,
        },
        "cells_csv": {
            "path": str(cells_path),
            "sha256": file_sha256(cells_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    result_path = args.output / "attention_persistent_givens_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
