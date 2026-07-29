#!/usr/bin/env python3
"""Test whether deeper Muon-matched c_proj updates are task-functional.

This diagnostic uses one coherent dense-Muon replay.  At each registered
phase start it constructs nested 32- and 64-stage hidden-side Givens updates
from the exact applied dense-Muon direction.  It then compares the updates in
three metrics:

1. ordinary weight-space Frobenius recovery;
2. post-GELU activation-weighted residual-output recovery;
3. recorded-training-gradient, disjoint-validation-gradient, and finite-step
   fixed-batch CE effects.

No endpoint information is used to select connectivity or angles.  Future
phase chords are scoring-only.  The diagnostic performs no optimizer update,
does not learn a basis, and does not write a training checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_muon_matched_givens import (
    diagonal_metric_causal_givens_update,
)
from examples.nanogpt.analyze_mlp_optimizer_state_direction import (
    reconstruct_directions,
)
from examples.nanogpt.analyze_mlp_task_gradient_direction import (
    collect_cproj_gradients,
    direction_metrics,
)
from examples.nanogpt.analyze_parameter_trajectory import (
    load_snapshots,
    parse_int_list,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.muon_matched_givens import (
    muon_matched_permutations,
)
from examples.nanogpt.parameter_trajectory import (
    OPTIMIZER_PROBE_SCHEMA_VERSION,
)


CANDIDATES = ("dense_exact", "stage32", "stage64", "incremental64")
WINDOWS = ("fit", "holdout")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def fixed_scale_recovery(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> float:
    target = target.double()
    prediction = prediction.double()
    denominator = target.square().sum().clamp_min(1e-30)
    return float(
        1.0 - (target - prediction).square().sum() / denominator
    )


def output_space_metrics(
    hidden: torch.Tensor,
    target_weight_update: torch.Tensor,
    predicted_weight_update: torch.Tensor,
) -> dict[str, float]:
    """Measure weight updates after multiplication by post-GELU rows."""
    if hidden.ndim != 2:
        raise ValueError("hidden activations must be [samples, width]")
    if (
        target_weight_update.ndim != 2
        or predicted_weight_update.shape != target_weight_update.shape
        or hidden.shape[1] != target_weight_update.shape[1]
    ):
        raise ValueError("activation and weight-update shapes disagree")
    hidden = hidden.float()
    target_output = hidden @ target_weight_update.float().T
    predicted_output = hidden @ predicted_weight_update.float().T
    line = direction_metrics(target_output, predicted_output)
    return {
        "fixed_scale_recovery": fixed_scale_recovery(
            target_output, predicted_output
        ),
        "positive_step_line_recovery": line[
            "positive_step_line_recovery"
        ],
        "cosine": line["cosine"],
        "target_output_energy": float(
            target_output.double().square().sum()
        ),
        "prediction_output_energy": float(
            predicted_output.double().square().sum()
        ),
    }


def task_descent_metrics(
    gradient: torch.Tensor,
    update: torch.Tensor,
) -> dict[str, float]:
    """Return first-order CE decrease from applying ``update``."""
    if gradient.shape != update.shape:
        raise ValueError("gradient and update shapes disagree")
    gradient = gradient.double()
    update = update.double()
    decrease = -(gradient * update).sum()
    return {
        "predicted_ce_decrease": float(decrease),
        "update_fro": float(update.norm()),
        "predicted_ce_decrease_per_fro": float(
            decrease / update.norm().clamp_min(1e-30)
        ),
    }


class PostGELUCollector:
    def __init__(self, model: torch.nn.Module, layers: list[int]) -> None:
        self.layers = set(layers)
        self.values: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles = []
        for layer, block in enumerate(model.transformer.h):
            if layer not in self.layers:
                continue
            self.handles.append(
                block.mlp.gelu.register_forward_hook(self._hook(layer))
            )

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            self.values[layer].append(
                output.detach()
                .float()
                .reshape(-1, output.shape[-1])
                .cpu()
            )

        return hook

    def tensors(self) -> dict[int, torch.Tensor]:
        if set(self.values) != self.layers:
            raise RuntimeError("post-GELU activation collection is incomplete")
        return {
            layer: torch.cat(self.values[layer], dim=0)
            for layer in sorted(self.layers)
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def evaluate_and_collect(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    layers: list[int],
    device: str,
) -> tuple[float, dict[int, torch.Tensor]]:
    collector = PostGELUCollector(model, layers)
    losses: list[float] = []
    try:
        with torch.no_grad():
            for tokens in batches:
                tokens = tokens.to(device)
                inputs = tokens[:, :-1].contiguous()
                targets = tokens[:, 1:].contiguous()
                _logits, loss = model(inputs, targets)
                if loss is None:
                    raise RuntimeError("model did not return a loss")
                losses.append(float(loss))
        return sum(losses) / len(losses), collector.tensors()
    finally:
        collector.close()


def evaluate_with_updates(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    updates: dict[int, torch.Tensor],
    device: str,
) -> float:
    """Evaluate one simultaneous selected-layer perturbation and restore."""
    parameters = {
        layer: model.transformer.h[layer].mlp.c_proj.weight
        for layer in updates
    }
    with torch.no_grad():
        for layer, parameter in parameters.items():
            parameter.add_(
                updates[layer].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
    losses: list[float] = []
    try:
        with torch.no_grad():
            for tokens in batches:
                tokens = tokens.to(device)
                inputs = tokens[:, :-1].contiguous()
                targets = tokens[:, 1:].contiguous()
                _logits, loss = model(inputs, targets)
                if loss is None:
                    raise RuntimeError("model did not return a loss")
                losses.append(float(loss))
        return sum(losses) / len(losses)
    finally:
        with torch.no_grad():
            for layer, parameter in parameters.items():
                parameter.sub_(
                    updates[layer].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )


def weighted(rows: list[dict[str, Any]], value: str, energy: str) -> float:
    weights = torch.tensor(
        [float(row[energy]) for row in rows],
        dtype=torch.float64,
    )
    values = torch.tensor(
        [float(row[value]) for row in rows],
        dtype=torch.float64,
    )
    return float((weights * values).sum() / weights.sum().clamp_min(1e-30))


def aggregate_results(
    rows: list[dict[str, Any]],
    finite_rows: list[dict[str, Any]],
    *,
    minimum_output_recovery_ratio: float,
    minimum_task_descent_ratio: float,
    minimum_finite_step_wins: int,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        fit = [row for row in selected if row["window"] == "fit"]
        groups[candidate] = {
            "cells_times_windows": len(selected),
            "current_output_fixed_scale_recovery": weighted(
                selected,
                "current_output_fixed_scale_recovery",
                "current_target_output_energy",
            ),
            "current_output_positive_line_recovery": weighted(
                selected,
                "current_output_positive_line_recovery",
                "current_target_output_energy",
            ),
            "future_output_positive_line_recovery": weighted(
                selected,
                "future_output_positive_line_recovery",
                "future_target_output_energy",
            ),
            "recorded_train_gradient_predicted_ce_decrease": sum(
                float(row["train_gradient_predicted_ce_decrease"])
                for row in fit
            ),
            "recorded_train_gradient_update_fro": sum(
                float(row["update_fro"]) for row in fit
            ),
            "validation_gradient_predicted_ce_decrease": {
                window: sum(
                    float(row["validation_gradient_predicted_ce_decrease"])
                    for row in selected
                    if row["window"] == window
                )
                for window in WINDOWS
            },
        }
    stage32 = groups["stage32"]
    stage64 = groups["stage64"]
    output_ratio = (
        float(stage64["current_output_fixed_scale_recovery"])
        / max(
            float(stage32["current_output_fixed_scale_recovery"]),
            1e-30,
        )
    )
    task_ratio = (
        float(stage64["recorded_train_gradient_predicted_ce_decrease"])
        / max(
            float(stage32["recorded_train_gradient_predicted_ce_decrease"]),
            1e-30,
        )
    )
    comparisons: list[dict[str, Any]] = []
    wins = 0
    for phase_start in sorted(
        {int(row["phase_start"]) for row in finite_rows}
    ):
        for window in WINDOWS:
            selected = {
                str(row["candidate"]): float(row["loss"])
                for row in finite_rows
                if int(row["phase_start"]) == phase_start
                and row["window"] == window
            }
            delta = selected["stage32"] - selected["stage64"]
            won = delta > 0.0
            wins += int(won)
            comparisons.append(
                {
                    "phase_start": phase_start,
                    "window": window,
                    "stage32_minus_stage64_loss": delta,
                    "stage64_wins": won,
                }
            )
    if (
        output_ratio >= minimum_output_recovery_ratio
        and task_ratio >= minimum_task_descent_ratio
        and wins >= minimum_finite_step_wins
    ):
        decision = "FUNCTIONAL_GAIN_PRESENT_TEST_TEMPORAL_REFRESH"
    elif (
        output_ratio < 1.20
        or task_ratio < 1.05
        or wins <= 2
    ):
        decision = "REJECT_FROBENIUS_STAGE_DEPTH_FUNCTIONAL_MISMATCH"
    else:
        decision = "MIXED_FUNCTIONAL_SIGNAL_NO_TRAINING_PROMOTION"
    return {
        "candidate_metrics": groups,
        "stage64_over_stage32": {
            "current_output_fixed_scale_recovery_ratio": output_ratio,
            "recorded_train_gradient_ce_decrease_ratio": task_ratio,
            "finite_step_ce_wins": wins,
            "finite_step_ce_comparisons": len(comparisons),
            "comparisons": comparisons,
        },
        "decision": decision,
        "decision_rule": {
            "functional_gain_present": {
                "minimum_output_recovery_ratio": (
                    minimum_output_recovery_ratio
                ),
                "minimum_task_descent_ratio": minimum_task_descent_ratio,
                "minimum_finite_step_wins": minimum_finite_step_wins,
            },
            "functional_mismatch": (
                "output-recovery ratio below 1.20, task-descent ratio below "
                "1.05, or no more than two finite-step wins"
            ),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--stages", default="32,64")
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--fit-seed", type=int, default=20260802)
    parser.add_argument("--holdout-seed", type=int, default=20260803)
    parser.add_argument("--matching-seed", type=int, default=161803)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--minimum-output-recovery-ratio", type=float, default=1.25
    )
    parser.add_argument(
        "--minimum-task-descent-ratio", type=float, default=1.15
    )
    parser.add_argument("--minimum-finite-step-wins", type=int, default=6)
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    boundaries = parse_int_list(args.phase_boundaries)
    stages = parse_int_list(args.stages)
    if stages != [32, 64]:
        raise ValueError("the preregistered nested stage counts are 32,64")
    if len(boundaries) != 5:
        raise ValueError("the preregistered phase boundary count is five")
    phase_pairs = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries
    ]
    probe_paths = [
        args.probe_dir / f"step_{start:06d}.pt"
        for start, _end in phase_pairs
    ]
    missing = [
        str(path)
        for path in (*snapshot_paths, *probe_paths, args.plan)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"required registered inputs are absent: {missing}")

    steps, values, snapshot_metadata = load_snapshots(
        snapshot_paths,
        layers=set(layers),
        targets={"mlp.c_proj"},
    )
    if steps != boundaries:
        raise ValueError("snapshot steps do not match phase boundaries")
    step_index = {step: index for index, step in enumerate(steps)}
    batches_by_window = {
        "fit": fixed_validation_batches(
            args.data_dir,
            args.batch_size,
            args.block_size + 1,
            args.batches,
            args.fit_seed,
        ),
        "holdout": fixed_validation_batches(
            args.data_dir,
            args.batch_size,
            args.block_size + 1,
            args.batches,
            args.holdout_seed,
        ),
    }

    rows: list[dict[str, Any]] = []
    finite_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    run_identity: str | None = None
    for phase_start, phase_end in phase_pairs:
        payload = load_snapshot(
            args.snapshot_dir / f"step_{phase_start:06d}.pt"
        )
        model = model_from_snapshot(payload, args.device)
        probe_path = args.probe_dir / f"step_{phase_start:06d}.pt"
        probe = torch.load(
            probe_path, map_location="cpu", weights_only=False
        )
        if probe.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
            raise ValueError(f"unexpected optimizer probe: {probe_path}")
        observed_identity = str(probe["run_identity_sha256"])
        if run_identity is None:
            run_identity = observed_identity
        elif run_identity != observed_identity:
            raise ValueError("optimizer probes have inconsistent identities")

        candidates_by_layer: dict[str, dict[int, torch.Tensor]] = {
            candidate: {} for candidate in CANDIDATES
        }
        future_by_layer: dict[int, torch.Tensor] = {}
        train_gradients: dict[int, torch.Tensor] = {}
        for layer in layers:
            parameter = f"transformer.h.{layer}.mlp.c_proj.weight"
            source = values[parameter][step_index[phase_start]].to(
                args.device
            )
            target = values[parameter][step_index[phase_end]].to(
                args.device
            )
            state = {
                name: tensor.to(args.device)
                for name, tensor in probe["parameters"][parameter].items()
            }
            torch.testing.assert_close(
                state["weight_before_step"],
                source,
                rtol=0.0,
                atol=0.0,
            )
            hyperparameters = probe["hyperparameters"][parameter]
            directions = reconstruct_directions(state, hyperparameters)
            requested = (
                float(hyperparameters["lr"])
                * directions["exact_applied_direction"]
            )
            permutations, diagnostics = muon_matched_permutations(
                source,
                directions["exact_applied_direction"],
                stages=64,
                neighbors=args.neighbors,
                seed=args.matching_seed + 1009 * layer + phase_start,
            )
            for diagnostic in diagnostics:
                matching_rows.append(
                    {
                        "layer": layer,
                        "phase_start": phase_start,
                        **diagnostic,
                    }
                )
            projected: dict[int, torch.Tensor] = {}
            for stage in stages:
                update, _fit = diagonal_metric_causal_givens_update(
                    source,
                    requested,
                    stages=stage,
                    seed=args.matching_seed,
                    permutations=permutations[:stage],
                )
                projected[stage] = update
            candidates_by_layer["dense_exact"][layer] = requested
            candidates_by_layer["stage32"][layer] = projected[32]
            candidates_by_layer["stage64"][layer] = projected[64]
            candidates_by_layer["incremental64"][layer] = (
                projected[64] - projected[32]
            )
            future_by_layer[layer] = target - source
            train_gradients[layer] = state["gradient_after_clip"]

        baseline_loss: dict[str, float] = {}
        activations: dict[str, dict[int, torch.Tensor]] = {}
        validation_gradients: dict[str, dict[int, torch.Tensor]] = {}
        for window in WINDOWS:
            baseline_loss[window], activations[window] = (
                evaluate_and_collect(
                    model,
                    batches_by_window[window],
                    layers,
                    args.device,
                )
            )
            validation_gradients[window], _gradient_loss = (
                collect_cproj_gradients(
                    model,
                    batches_by_window[window],
                    layers,
                    args.device,
                )
            )
            finite_rows.append(
                {
                    "phase_start": phase_start,
                    "phase_end": phase_end,
                    "window": window,
                    "candidate": "baseline",
                    "loss": baseline_loss[window],
                    "loss_change_from_baseline": 0.0,
                }
            )
            for candidate in CANDIDATES:
                loss = evaluate_with_updates(
                    model,
                    batches_by_window[window],
                    candidates_by_layer[candidate],
                    args.device,
                )
                finite_rows.append(
                    {
                        "phase_start": phase_start,
                        "phase_end": phase_end,
                        "window": window,
                        "candidate": candidate,
                        "loss": loss,
                        "loss_change_from_baseline": (
                            loss - baseline_loss[window]
                        ),
                    }
                )

        for window in WINDOWS:
            for layer in layers:
                hidden = activations[window][layer].to(args.device)
                dense = candidates_by_layer["dense_exact"][layer]
                stage32 = candidates_by_layer["stage32"][layer]
                residual_after32 = dense - stage32
                future = future_by_layer[layer]
                for candidate in CANDIDATES:
                    update = candidates_by_layer[candidate][layer]
                    current = output_space_metrics(hidden, dense, update)
                    future_metrics = output_space_metrics(
                        hidden, future, update
                    )
                    train = task_descent_metrics(
                        train_gradients[layer], update
                    )
                    validation = task_descent_metrics(
                        validation_gradients[window][layer], update.cpu()
                    )
                    marginal = (
                        output_space_metrics(
                            hidden, residual_after32, update
                        )
                        if candidate == "incremental64"
                        else None
                    )
                    row = {
                        "layer": layer,
                        "phase_start": phase_start,
                        "phase_end": phase_end,
                        "window": window,
                        "candidate": candidate,
                        "current_output_fixed_scale_recovery": current[
                            "fixed_scale_recovery"
                        ],
                        "current_output_positive_line_recovery": current[
                            "positive_step_line_recovery"
                        ],
                        "current_output_cosine": current["cosine"],
                        "current_target_output_energy": current[
                            "target_output_energy"
                        ],
                        "future_output_positive_line_recovery": (
                            future_metrics["positive_step_line_recovery"]
                        ),
                        "future_output_cosine": future_metrics["cosine"],
                        "future_target_output_energy": future_metrics[
                            "target_output_energy"
                        ],
                        "train_gradient_predicted_ce_decrease": train[
                            "predicted_ce_decrease"
                        ],
                        "validation_gradient_predicted_ce_decrease": (
                            validation["predicted_ce_decrease"]
                        ),
                        "update_fro": train["update_fro"],
                        "train_gradient_ce_decrease_per_fro": train[
                            "predicted_ce_decrease_per_fro"
                        ],
                        "marginal_after32_output_fixed_scale_recovery": (
                            marginal["fixed_scale_recovery"]
                            if marginal is not None
                            else ""
                        ),
                        "marginal_after32_output_cosine": (
                            marginal["cosine"]
                            if marginal is not None
                            else ""
                        ),
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

        del model, payload, probe
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    aggregate = aggregate_results(
        rows,
        finite_rows,
        minimum_output_recovery_ratio=(
            args.minimum_output_recovery_ratio
        ),
        minimum_task_descent_ratio=args.minimum_task_descent_ratio,
        minimum_finite_step_wins=args.minimum_finite_step_wins,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "muon_matched_functional_metric.csv"
    finite_path = args.output / "muon_matched_functional_finite_ce.csv"
    matching_path = args.output / "muon_matched_functional_matchings.csv"
    aggregate_path = (
        args.output / "muon_matched_functional_metric_aggregate.json"
    )
    write_csv(detail_path, rows)
    write_csv(finite_path, finite_rows)
    write_csv(matching_path, matching_rows)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_muon_matched_functional_metric_v1",
        "decision": aggregate["decision"],
        "causal_protocol": (
            "nested connectivity and angles use only the exact current "
            "coherent Muon direction; future phase chords are scoring-only"
        ),
        "parameter_updates": 0,
        "learned_dense_basis": False,
        "lora_adapter": False,
        "layers": layers,
        "phase_boundaries": boundaries,
        "stage_counts": stages,
        "neighbors": args.neighbors,
        "validation_windows": {
            "fit_seed": args.fit_seed,
            "holdout_seed": args.holdout_seed,
            "batches": args.batches,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
        },
        "input_run_identity_sha256": run_identity,
        "snapshot_metadata": snapshot_metadata,
        "plan": {
            "path": str(args.plan),
            "sha256": file_sha256(args.plan),
        },
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "inputs": {
            "snapshots": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in snapshot_paths
            ],
            "optimizer_probes": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in probe_paths
            ],
        },
        "outputs": {
            "detail_sha256": file_sha256(detail_path),
            "finite_ce_sha256": file_sha256(finite_path),
            "matchings_sha256": file_sha256(matching_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": [
            "The finite-step CE intervention updates five representative c_proj layers, not all model parameters.",
            "The recorded clipped training gradient is exact for its training batch; validation gradients use two disjoint fixed windows.",
            "This diagnoses one dense-Muon optimizer trajectory, not the global low-loss manifold.",
        ],
    }
    metadata_path = (
        args.output / "muon_matched_functional_metric_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": aggregate["decision"],
                "aggregate": aggregate,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
