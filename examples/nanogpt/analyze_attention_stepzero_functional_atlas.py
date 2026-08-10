#!/usr/bin/env python3
"""Step-zero task-oriented Kronecker atlas for attention V and c_proj.

This module contains the representation-critical part of the preregistered
zero-update oracle.  It intentionally does not launch training.  A layerwise
atlas is selected from uncentered empirical second moments of the exact
step-zero model's linear inputs and backpropagated CE errors.  Later dense
states and directions may be projected into the frozen atlas, but they never
participate in selecting its basis.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from examples.nanogpt.analyze_attention_affine_delta_path_oracle import (
    batch_digest,
    minimum_layer_recovery,
    solve_span_coefficients,
    target_tensor,
    trajectory_inventory,
    weighted,
    write_rows,
)
from examples.nanogpt.analyze_attention_paper_activation_oracle import (
    AttentionFunctionalMetric,
    all_finite,
    file_sha256,
    terminal_attention_metrics,
)
from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_paper_activation_oracle import (
    cgls,
    explained_energy,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.train import require_block_fht_native_extension
from latent_weight_lab.block_fht import block_fht_grad_latent, block_fht_slice


TARGETS = ("v", "cproj")
REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_attention_stepzero_functional_atlas_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_stepzero_functional_atlas_result_v1"


def empirical_second_moment(rows: torch.Tensor) -> torch.Tensor:
    """Return the uncentered KFAC/Fisher factor ``E[row row^T]``."""

    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("second-moment rows must be nonempty and rank two")
    values = rows.float()
    return values.T @ values / values.shape[0]


def top_kronecker_pairs(
    output_eigenvalues: torch.Tensor,
    input_eigenvalues: torch.Tensor,
    coordinate_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select tensor-product eigenvectors by KFAC eigenvalue product."""

    if output_eigenvalues.ndim != 1 or input_eigenvalues.ndim != 1:
        raise ValueError("Kronecker eigenvalues must be vectors")
    total = output_eigenvalues.numel() * input_eigenvalues.numel()
    if not 0 < int(coordinate_count) <= total:
        raise ValueError("coordinate_count is outside the Kronecker basis")
    products = (
        output_eigenvalues.clamp_min(0).reshape(-1, 1)
        * input_eigenvalues.clamp_min(0).reshape(1, -1)
    )
    values, indices = torch.topk(
        products.reshape(-1), int(coordinate_count), sorted=True
    )
    input_width = input_eigenvalues.numel()
    return indices // input_width, indices % input_width, values


@dataclass(frozen=True)
class KroneckerAtlas:
    """An orthonormal subset of a matrix Kronecker eigenbasis."""

    output_basis: torch.Tensor
    input_basis: torch.Tensor
    output_indices: torch.Tensor
    input_indices: torch.Tensor
    scores: torch.Tensor

    @classmethod
    def from_second_moments(
        cls,
        input_second_moment: torch.Tensor,
        output_second_moment: torch.Tensor,
        coordinate_count: int,
    ) -> "KroneckerAtlas":
        if (
            input_second_moment.ndim != 2
            or input_second_moment.shape[0] != input_second_moment.shape[1]
            or output_second_moment.ndim != 2
            or output_second_moment.shape[0] != output_second_moment.shape[1]
        ):
            raise ValueError("KFAC factors must be square matrices")
        input_values, input_vectors = torch.linalg.eigh(
            input_second_moment.float()
        )
        output_values, output_vectors = torch.linalg.eigh(
            output_second_moment.float()
        )
        output_indices, input_indices, scores = top_kronecker_pairs(
            output_values,
            input_values,
            int(coordinate_count),
        )
        return cls(
            output_basis=output_vectors.contiguous(),
            input_basis=input_vectors.contiguous(),
            output_indices=output_indices.contiguous(),
            input_indices=input_indices.contiguous(),
            scores=scores.contiguous(),
        )

    @property
    def coordinate_count(self) -> int:
        return int(self.output_indices.numel())

    @property
    def shape(self) -> tuple[int, int]:
        return self.output_basis.shape[0], self.input_basis.shape[0]

    def apply(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Map compact coordinates to a full matrix without a learned basis."""

        if coordinates.ndim != 1 or coordinates.numel() != self.coordinate_count:
            raise ValueError("coordinate vector does not match the atlas")
        core = coordinates.new_zeros(
            self.output_basis.shape[1], self.input_basis.shape[1]
        )
        core.index_put_(
            (
                self.output_indices.to(device=coordinates.device),
                self.input_indices.to(device=coordinates.device),
            ),
            coordinates,
            accumulate=True,
        )
        output = self.output_basis.to(
            device=coordinates.device, dtype=coordinates.dtype
        )
        inputs = self.input_basis.to(
            device=coordinates.device, dtype=coordinates.dtype
        )
        return output @ core @ inputs.T

    def adjoint(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply the exact Frobenius adjoint of :meth:`apply`."""

        if weight.ndim != 2 or tuple(weight.shape) != self.shape:
            raise ValueError("weight matrix does not match the atlas")
        output = self.output_basis.to(device=weight.device, dtype=weight.dtype)
        inputs = self.input_basis.to(device=weight.device, dtype=weight.dtype)
        core = output.T @ weight @ inputs
        return core[
            self.output_indices.to(device=weight.device),
            self.input_indices.to(device=weight.device),
        ]

    def fixed_storage_bytes(self) -> int:
        tensors = (
            self.output_basis,
            self.input_basis,
            self.output_indices,
            self.input_indices,
            self.scores,
        )
        return sum(value.numel() * value.element_size() for value in tensors)


def kronecker_subspace_overlap(
    left: KroneckerAtlas,
    right: KroneckerAtlas,
    *,
    chunk_size: int = 256,
) -> float:
    """Return normalized ``||Q_left^T Q_right||_F^2`` without dense Q."""

    if left.shape != right.shape or left.coordinate_count != right.coordinate_count:
        raise ValueError("atlas overlap requires equal shapes and coordinate counts")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    device = left.output_basis.device
    output_cross = left.output_basis.T @ right.output_basis.to(device)
    input_cross = left.input_basis.T @ right.input_basis.to(device)
    right_output = right.output_indices.to(device)
    right_input = right.input_indices.to(device)
    total = output_cross.new_zeros(())
    for start in range(0, left.coordinate_count, int(chunk_size)):
        stop = min(start + int(chunk_size), left.coordinate_count)
        left_output = left.output_indices[start:stop].to(device)
        left_input = left.input_indices[start:stop].to(device)
        output_inner = output_cross[left_output[:, None], right_output[None, :]]
        input_inner = input_cross[left_input[:, None], right_input[None, :]]
        total = total + (output_inner * input_inner).square().sum()
    return float(total / left.coordinate_count)


class AttentionKFACCollector:
    """Capture step-zero inputs and output errors for V and attention c_proj."""

    def __init__(
        self,
        model: torch.nn.Module,
        layers: Iterable[int],
        sample_cap: int,
    ) -> None:
        self.layers = set(int(layer) for layer in layers)
        self.sample_cap = int(sample_cap)
        if self.sample_cap <= 0:
            raise ValueError("sample_cap must be positive")
        self.inputs: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(list)
        self.errors: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(list)
        self.counts: dict[tuple[int, str], int] = defaultdict(int)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer, block in enumerate(model.transformer.h):
            if layer not in self.layers:
                continue
            self.handles.append(
                block.attn.c_attn.register_forward_hook(
                    self._hook(layer, "v", final_value_slice=True)
                )
            )
            self.handles.append(
                block.attn.c_proj.register_forward_hook(
                    self._hook(layer, "cproj", final_value_slice=False)
                )
            )

    def _hook(self, layer: int, target: str, *, final_value_slice: bool):
        def hook(module, inputs, output):
            if not torch.is_tensor(output) or not output.requires_grad:
                raise RuntimeError("KFAC acquisition requires a CE backward pass")
            key = (layer, target)
            source = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            take = min(self.sample_cap - self.counts[key], source.shape[0])
            if take <= 0:
                return
            self.inputs[key].append(source[:take].cpu())
            self.counts[key] += int(take)

            def save_error(gradient: torch.Tensor) -> None:
                error = gradient
                if final_value_slice:
                    n_embd = int(module.out_features) // 3
                    error = error[..., 2 * n_embd :]
                error = error.detach().float().reshape(-1, error.shape[-1])
                self.errors[key].append(error[:take].cpu())

            output.register_hook(save_error)

        return hook

    def complete(self) -> bool:
        return all(
            self.counts[(layer, target)] >= self.sample_cap
            and sum(value.shape[0] for value in self.errors[(layer, target)])
            >= self.sample_cap
            for layer in self.layers
            for target in TARGETS
        )

    def rows(self, layer: int, target: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(layer), target)
        if target not in TARGETS or not self.inputs[key] or not self.errors[key]:
            raise ValueError(f"missing KFAC rows for {key}")
        return (
            torch.cat(self.inputs[key], dim=0)[: self.sample_cap],
            torch.cat(self.errors[key], dim=0)[: self.sample_cap],
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_stepzero_second_moments(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    layers: Iterable[int],
    sample_cap: int,
    device: str,
) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
    """Run exact step-zero CE backward passes and return KFAC factors."""

    selected_layers = [int(layer) for layer in layers]
    collector = AttentionKFACCollector(model, selected_layers, sample_cap)
    model.eval()
    try:
        for batch in batches:
            if batch.ndim != 2 or batch.shape[1] < 2:
                raise ValueError("KFAC batches must contain at least two tokens")
            tokens = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            model.zero_grad(set_to_none=True)
            _logits, loss = model(tokens, targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("step-zero calibration loss is not finite")
            loss.backward()
            if collector.complete():
                break
        if not collector.complete():
            raise RuntimeError("step-zero KFAC sample cap was not reached")
        result = {}
        for layer in selected_layers:
            for target in TARGETS:
                inputs, errors = collector.rows(layer, target)
                result[(layer, target)] = (
                    empirical_second_moment(inputs),
                    empirical_second_moment(errors),
                )
        return result
    finally:
        model.zero_grad(set_to_none=True)
        collector.close()


def random_kronecker_atlas(
    shape: tuple[int, int],
    coordinate_count: int,
    *,
    seed: int,
    device: str,
) -> KroneckerAtlas:
    """Create an identical-budget isotropic tensor-product control."""

    output_width, input_width = shape
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    output_basis = torch.linalg.qr(
        torch.randn(output_width, output_width, generator=generator, device=device),
        mode="reduced",
    ).Q
    input_basis = torch.linalg.qr(
        torch.randn(input_width, input_width, generator=generator, device=device),
        mode="reduced",
    ).Q
    indices = torch.randperm(
        output_width * input_width, generator=generator, device=device
    )[: int(coordinate_count)]
    return KroneckerAtlas(
        output_basis=output_basis,
        input_basis=input_basis,
        output_indices=indices // input_width,
        input_indices=indices % input_width,
        scores=torch.ones(int(coordinate_count), device=device),
    )


@dataclass(frozen=True)
class LinearChart:
    name: str
    coordinate_count: int
    apply: Callable[[torch.Tensor], torch.Tensor]
    adjoint: Callable[[torch.Tensor], torch.Tensor]
    fixed_storage_bytes: int


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def validate_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected functional-atlas executable plan schema")
    protocol = plan["protocol"]
    frozen = {
        "parameter_updates": 0,
        "coordinate_fraction": 0.01,
        "calibration_centering": False,
        "calibration_batch_size": 2,
        "calibration_block_size": 256,
        "calibration_batches": 4,
        "calibration_rows_per_layer": 2048,
        "calibration_metric_seeds": [20260811, 20260812],
        "fit_metric_seed": 20260809,
        "eval_metric_seed": 20260810,
        "metric_batch_size": 2,
        "metric_block_size": 256,
        "metric_batches": 2,
        "trajectory_discovery_max_step": 1140,
        "trajectory_heldout_min_step": 1200,
        "heldout_probe_steps": [1782, 2372],
        "cgls_iterations": 32,
        "span_relative_cutoff": 1e-08,
        "block_fht_layers": 2,
        "block_fht_seed": 1000,
        "random_kronecker_seed": 20260813,
    }
    for field, expected in frozen.items():
        if protocol.get(field) != expected:
            raise ValueError(f"frozen functional-atlas protocol changed: {field}")
    thresholds = plan["decision_rule"]["thresholds"]
    if thresholds != {
        "aggregate_recovery_minimum": 0.8,
        "minimum_every_layer_recovery": 0.6,
        "minimum_late_layer_8_to_11_recovery": 0.6,
        "minimum_absolute_gain_over_blockfht": 0.1,
        "minimum_calibration_split_subspace_overlap": 0.75,
    }:
        raise ValueError("functional-atlas thresholds changed")
    if any(bool(value) for value in plan["authorization"].values()):
        raise ValueError("zero-update plan must not pre-authorize a successor")
    identity = plan["identity"]
    paths = {
        Path(__file__): identity["entrypoint_sha256"],
        REPO_ROOT / identity["design"]: identity["design_sha256"],
        REPO_ROOT / identity["dense_config"]: identity["dense_config_sha256"],
        args.terminal_checkpoint: identity["terminal_checkpoint_sha256"],
        args.data_dir / "manifest.json": identity["dataset_manifest_sha256"],
    }
    for path, expected in paths.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"pinned functional-atlas identity mismatch: {path}")
    inventory, digest = trajectory_inventory(args.trajectory_dir)
    if (
        len(inventory) != int(identity["trajectory_file_count"])
        or sum(int(item["size"]) for item in inventory)
        != int(identity["trajectory_total_bytes"])
        or digest != identity["trajectory_inventory_sha256"]
    ):
        raise ValueError("trajectory inventory mismatch")
    for name, expected in identity["optimizer_probe_sha256"].items():
        path = args.probe_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"optimizer probe mismatch: {path}")
    if Path(identity["trajectory_directory"]) != args.trajectory_dir:
        raise ValueError("trajectory directory differs from plan")
    if Path(identity["optimizer_probe_directory"]) != args.probe_dir:
        raise ValueError("probe directory differs from plan")
    if Path(identity["output_directory_must_be_absent"]) != args.output_dir:
        raise ValueError("output directory differs from plan")


def fit_chart_coordinate(
    chart: LinearChart,
    metric: AttentionFunctionalMetric,
    weight: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    target = metric.apply(weight)
    template = weight.new_zeros(chart.coordinate_count)

    def apply(coordinate: torch.Tensor) -> torch.Tensor:
        return metric.apply(chart.apply(coordinate))

    def adjoint(output: torch.Tensor) -> torch.Tensor:
        return chart.adjoint(metric.adjoint(output))

    return cgls(apply, adjoint, target, template, int(iterations))


def chart_metrics(
    chart: LinearChart,
    fit_metric: AttentionFunctionalMetric,
    eval_metric: AttentionFunctionalMetric,
    weight: torch.Tensor,
    coordinate: torch.Tensor,
    fit_prediction: torch.Tensor,
) -> dict[str, float]:
    prediction_weight = chart.apply(coordinate)
    fit_target = fit_metric.apply(weight)
    eval_target = eval_metric.apply(weight)
    fit_recovery, fit_energy = explained_energy(fit_target, fit_prediction)
    eval_recovery, eval_energy = explained_energy(
        eval_target, eval_metric.apply(prediction_weight)
    )
    euclidean_recovery, euclidean_energy = explained_energy(
        weight, prediction_weight
    )
    return {
        "fit_recovery": fit_recovery,
        "fit_energy": fit_energy,
        "eval_recovery": eval_recovery,
        "eval_energy": eval_energy,
        "euclidean_recovery": euclidean_recovery,
        "euclidean_energy": euclidean_energy,
    }


def analyze_one_chart(
    *,
    chart: LinearChart,
    target: str,
    layer: int,
    steps: list[int],
    snapshots: dict[int, dict[str, torch.Tensor]],
    probes: dict[int, dict[str, Any]],
    parameter_name: str,
    n_embd: int,
    discovery_max: int,
    heldout_min: int,
    heldout_probes: set[int],
    fit_metric: AttentionFunctionalMetric,
    eval_metric: AttentionFunctionalMetric,
    cgls_iterations: int,
    span_relative_cutoff: float,
    device: str,
) -> list[dict[str, Any]]:
    initial = target_tensor(
        snapshots[steps[0]][parameter_name], target, n_embd
    ).to(device)
    state_weights = {steps[0]: torch.zeros_like(initial)}
    coordinates = {steps[0]: initial.new_zeros(chart.coordinate_count)}
    rows: list[dict[str, Any]] = []
    for step in steps[1:]:
        current = target_tensor(
            snapshots[step][parameter_name], target, n_embd
        ).to(device)
        displacement = current - initial
        coordinate, prediction, iterations = fit_chart_coordinate(
            chart, fit_metric, displacement, cgls_iterations
        )
        state_weights[step] = displacement
        coordinates[step] = coordinate
        rows.append(
            {
                "arm": chart.name,
                "kind": "state",
                "target": target,
                "layer": layer,
                "step_start": 0,
                "step_end": step,
                "split": "discovery" if step <= discovery_max else "heldout",
                "coordinate_count": chart.coordinate_count,
                "iterations": iterations,
                **chart_metrics(
                    chart, fit_metric, eval_metric, displacement, coordinate, prediction
                ),
            }
        )
    for start, end in zip(steps[:-1], steps[1:], strict=True):
        if start <= discovery_max < end:
            continue
        chord = state_weights[end] - state_weights[start]
        coordinate, prediction, iterations = fit_chart_coordinate(
            chart, fit_metric, chord, cgls_iterations
        )
        rows.append(
            {
                "arm": chart.name,
                "kind": "chord",
                "target": target,
                "layer": layer,
                "step_start": start,
                "step_end": end,
                "split": "discovery" if end <= discovery_max else "heldout",
                "coordinate_count": chart.coordinate_count,
                "iterations": iterations,
                **chart_metrics(
                    chart, fit_metric, eval_metric, chord, coordinate, prediction
                ),
            }
        )
    discovery_steps = [step for step in steps if 0 < step <= discovery_max]
    coordinate_basis = torch.stack(
        [coordinates[step] for step in discovery_steps], dim=1
    )
    output_basis = torch.stack(
        [
            fit_metric.apply(chart.apply(coordinates[step])).reshape(-1)
            for step in discovery_steps
        ],
        dim=1,
    )
    for step in [value for value in steps if value >= heldout_min]:
        fit_target = fit_metric.apply(state_weights[step]).reshape(-1)
        coefficients, rank = solve_span_coefficients(
            output_basis, fit_target, float(span_relative_cutoff)
        )
        coordinate = coordinate_basis @ coefficients
        eval_target = eval_metric.apply(state_weights[step])
        recovery, energy = explained_energy(
            eval_target, eval_metric.apply(chart.apply(coordinate))
        )
        rows.append(
            {
                "arm": chart.name,
                "kind": "discovery_span",
                "target": target,
                "layer": layer,
                "step_start": 0,
                "step_end": step,
                "split": "heldout",
                "coordinate_count": chart.coordinate_count,
                "span_rank": rank,
                "eval_recovery": recovery,
                "eval_energy": energy,
            }
        )
    for step, payload in probes.items():
        direction = target_tensor(
            payload["parameters"][parameter_name]["applied_direction_per_lr"],
            target,
            n_embd,
        ).to(device)
        coordinate, prediction, iterations = fit_chart_coordinate(
            chart, fit_metric, direction, cgls_iterations
        )
        rows.append(
            {
                "arm": chart.name,
                "kind": "muon_direction",
                "target": target,
                "layer": layer,
                "step_start": step,
                "step_end": step,
                "split": "heldout" if step in heldout_probes else "discovery",
                "coordinate_count": chart.coordinate_count,
                "iterations": iterations,
                **chart_metrics(
                    chart, fit_metric, eval_metric, direction, coordinate, prediction
                ),
            }
        )
    return rows


def summarize_arm(rows: list[dict[str, Any]], arm: str, target: str) -> dict[str, Any]:
    kinds = {
        "state": "state",
        "chord": "local_chord",
        "discovery_span": "discovery_span",
        "muon_direction": "exact_muon",
    }
    output: dict[str, Any] = {}
    for kind, label in kinds.items():
        selected = [
            row
            for row in rows
            if row["arm"] == arm
            and row["target"] == target
            and row["kind"] == kind
            and row["split"] == "heldout"
        ]
        late = [row for row in selected if int(row["layer"]) >= 8]
        output[label] = {
            "aggregate_eval_recovery": weighted(
                selected, "eval_recovery", "eval_energy"
            ),
            "minimum_layer_eval_recovery": minimum_layer_recovery(
                selected, "eval_recovery", "eval_energy"
            ),
            "minimum_late_layer_eval_recovery": minimum_layer_recovery(
                late, "eval_recovery", "eval_energy"
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--terminal-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan, args)
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    require_block_fht_native_extension(True)
    started = time.time()
    protocol = plan["protocol"]
    layers = [int(value) for value in protocol["layers"]]
    steps = [int(value) for value in protocol["trajectory_steps"]]
    probe_steps = [int(value) for value in protocol["probe_steps"]]
    heldout_probes = {int(value) for value in protocol["heldout_probe_steps"]}
    inventory, inventory_sha = trajectory_inventory(args.trajectory_dir)

    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    run_identity = None
    stepzero_payload = None
    for step in steps:
        payload = load_snapshot(args.trajectory_dir / f"step_{step:06d}.pt")
        if run_identity is None:
            run_identity = payload["run_identity_sha256"]
            stepzero_payload = payload
        elif payload["run_identity_sha256"] != run_identity:
            raise ValueError("trajectory snapshots do not share one run identity")
        snapshots[step] = payload["parameters"]
    if run_identity != plan["identity"]["trajectory_run_identity_sha256"]:
        raise ValueError("trajectory run identity mismatch")
    probes = {}
    for step in probe_steps:
        payload = torch.load(
            args.probe_dir / f"step_{step:06d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if payload["run_identity_sha256"] != run_identity:
            raise ValueError("optimizer probe run identity mismatch")
        probes[step] = payload
    assert stepzero_payload is not None

    calibration_moments = []
    calibration_batch_hashes = []
    for seed in protocol["calibration_metric_seeds"]:
        batches = fixed_validation_batches(
            args.data_dir,
            int(protocol["calibration_batch_size"]),
            int(protocol["calibration_block_size"]) + 1,
            int(protocol["calibration_batches"]),
            int(seed),
        )
        calibration_batch_hashes.append(batch_digest(batches))
        model = model_from_snapshot(stepzero_payload, args.device)
        calibration_moments.append(
            collect_stepzero_second_moments(
                model,
                batches,
                layers,
                int(protocol["calibration_rows_per_layer"]),
                args.device,
            )
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if calibration_batch_hashes[0] == calibration_batch_hashes[1]:
        raise ValueError("calibration splits are identical")

    fit_batches = fixed_validation_batches(
        args.data_dir,
        int(protocol["metric_batch_size"]),
        int(protocol["metric_block_size"]),
        int(protocol["metric_batches"]),
        int(protocol["fit_metric_seed"]),
    )
    eval_batches = fixed_validation_batches(
        args.data_dir,
        int(protocol["metric_batch_size"]),
        int(protocol["metric_block_size"]),
        int(protocol["metric_batches"]),
        int(protocol["eval_metric_seed"]),
    )
    fit_batch_sha = batch_digest(fit_batches)
    eval_batch_sha = batch_digest(eval_batches)
    if fit_batch_sha == eval_batch_sha:
        raise ValueError("fit and evaluation functional batches are identical")
    fit_inputs = terminal_attention_metrics(
        args.terminal_checkpoint, fit_batches, layers, args.device
    )
    eval_inputs = terminal_attention_metrics(
        args.terminal_checkpoint, eval_batches, layers, args.device
    )
    config = json.loads((REPO_ROOT / plan["identity"]["dense_config"]).read_text())
    n_embd = int(config["n_embd"])
    latent_std = float(config.get("block_fht_latent_init_std", 0.02))
    rows: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    storage: list[dict[str, Any]] = []

    for layer in layers:
        print(f"analyzing layer {layer}", flush=True)
        for target, spec in protocol["targets"].items():
            parameter_name = f"transformer.h.{layer}.{spec['parameter']}"
            initial = target_tensor(
                snapshots[steps[0]][parameter_name], target, n_embd
            ).to(args.device)
            coordinate_count = max(
                1, round(initial.numel() * float(protocol["coordinate_fraction"]))
            )
            primary = KroneckerAtlas.from_second_moments(
                calibration_moments[0][(layer, target)][0].to(args.device),
                calibration_moments[0][(layer, target)][1].to(args.device),
                coordinate_count,
            )
            confirmation = KroneckerAtlas.from_second_moments(
                calibration_moments[1][(layer, target)][0].to(args.device),
                calibration_moments[1][(layer, target)][1].to(args.device),
                coordinate_count,
            )
            overlap = kronecker_subspace_overlap(primary, confirmation)
            overlaps.append({"target": target, "layer": layer, "overlap": overlap})
            random_atlas = random_kronecker_atlas(
                primary.shape,
                coordinate_count,
                seed=(
                    int(protocol["random_kronecker_seed"])
                    + int(spec["seed_stride"]) * layer
                    + int(spec["seed_offset"])
                ),
                device=args.device,
            )
            block_seed = (
                int(protocol["block_fht_seed"])
                + int(spec["seed_stride"]) * layer
                + int(spec["seed_offset"])
            )
            block_template = initial.new_zeros(coordinate_count)
            weight_scale = float(spec["target_std"]) / latent_std

            def apply_block(coordinate: torch.Tensor) -> torch.Tensor:
                return (
                    block_fht_slice(
                        coordinate,
                        initial.numel(),
                        int(protocol["block_fht_layers"]),
                        block_seed,
                        0,
                        initial.numel(),
                    )
                    * weight_scale
                ).view_as(initial)

            def adjoint_block(weight: torch.Tensor) -> torch.Tensor:
                return block_fht_grad_latent(
                    block_template,
                    (weight.reshape(-1) * weight_scale).contiguous(),
                    initial.numel(),
                    int(protocol["block_fht_layers"]),
                    block_seed,
                    0,
                    initial.numel(),
                )

            charts = (
                LinearChart(
                    "stepzero_kfac",
                    coordinate_count,
                    primary.apply,
                    primary.adjoint,
                    primary.fixed_storage_bytes(),
                ),
                LinearChart(
                    "random_kronecker",
                    coordinate_count,
                    random_atlas.apply,
                    random_atlas.adjoint,
                    random_atlas.fixed_storage_bytes(),
                ),
                LinearChart(
                    "blockfht",
                    coordinate_count,
                    apply_block,
                    adjoint_block,
                    0,
                ),
            )
            fit_metric = AttentionFunctionalMetric(
                target=target, **fit_inputs[layer]
            )
            eval_metric = AttentionFunctionalMetric(
                target=target, **eval_inputs[layer]
            )
            for chart in charts:
                storage.append(
                    {
                        "arm": chart.name,
                        "target": target,
                        "layer": layer,
                        "coordinate_count": coordinate_count,
                        "fixed_storage_bytes": chart.fixed_storage_bytes,
                    }
                )
                rows.extend(
                    analyze_one_chart(
                        chart=chart,
                        target=target,
                        layer=layer,
                        steps=steps,
                        snapshots=snapshots,
                        probes=probes,
                        parameter_name=parameter_name,
                        n_embd=n_embd,
                        discovery_max=int(protocol["trajectory_discovery_max_step"]),
                        heldout_min=int(protocol["trajectory_heldout_min_step"]),
                        heldout_probes=heldout_probes,
                        fit_metric=fit_metric,
                        eval_metric=eval_metric,
                        cgls_iterations=int(protocol["cgls_iterations"]),
                        span_relative_cutoff=float(protocol["span_relative_cutoff"]),
                        device=args.device,
                    )
                )

    summaries: dict[str, Any] = {}
    thresholds = plan["decision_rule"]["thresholds"]
    for target in protocol["targets"]:
        target_summaries = {
            arm: summarize_arm(rows, arm, target)
            for arm in ("stepzero_kfac", "random_kronecker", "blockfht")
        }
        overlap_values = [
            float(row["overlap"]) for row in overlaps if row["target"] == target
        ]
        target_summaries["calibration_overlap"] = {
            "mean": sum(overlap_values) / len(overlap_values),
            "minimum": min(overlap_values),
        }
        checks: dict[str, bool] = {
            "calibration_overlap": min(overlap_values)
            >= float(thresholds["minimum_calibration_split_subspace_overlap"])
        }
        for metric in (
            "state",
            "local_chord",
            "discovery_span",
            "exact_muon",
        ):
            primary_metric = target_summaries["stepzero_kfac"][metric]
            block_metric = target_summaries["blockfht"][metric]
            checks[f"{metric}_aggregate"] = float(
                primary_metric["aggregate_eval_recovery"]
            ) >= float(thresholds["aggregate_recovery_minimum"])
            checks[f"{metric}_every_layer"] = float(
                primary_metric["minimum_layer_eval_recovery"]
            ) >= float(thresholds["minimum_every_layer_recovery"])
            checks[f"{metric}_late_layers"] = float(
                primary_metric["minimum_late_layer_eval_recovery"]
            ) >= float(thresholds["minimum_late_layer_8_to_11_recovery"])
            gain = float(primary_metric["aggregate_eval_recovery"]) - float(
                block_metric["aggregate_eval_recovery"]
            )
            primary_metric["absolute_gain_over_blockfht"] = gain
            checks[f"{metric}_gain_over_blockfht"] = gain >= float(
                thresholds["minimum_absolute_gain_over_blockfht"]
            )
        target_summaries["checks"] = checks
        target_summaries["passed"] = all(checks.values())
        summaries[target] = target_summaries

    args.output_dir.mkdir(parents=True)
    cells_path = args.output_dir / "attention_stepzero_functional_atlas_cells.csv"
    overlap_path = args.output_dir / "attention_stepzero_functional_atlas_overlap.csv"
    storage_path = args.output_dir / "attention_stepzero_functional_atlas_storage.csv"
    write_rows(cells_path, rows)
    write_rows(overlap_path, overlaps)
    write_rows(storage_path, storage)
    passed = [target for target, summary in summaries.items() if summary["passed"]]
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "ATTENTION_STEPZERO_FUNCTIONAL_ATLAS_PASS_ALL"
            if len(passed) == len(protocol["targets"])
            else "ATTENTION_STEPZERO_FUNCTIONAL_ATLAS_REJECT"
        ),
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(),
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "trajectory_inventory_sha256": inventory_sha,
            "trajectory_run_identity_sha256": run_identity,
            "terminal_checkpoint_sha256": file_sha256(args.terminal_checkpoint),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "calibration_batch_sha256": calibration_batch_hashes,
            "fit_metric_batch_sha256": fit_batch_sha,
            "eval_metric_batch_sha256": eval_batch_sha,
        },
        "protocol": protocol,
        "summaries": summaries,
        "decision": {
            "passed_targets": passed,
            "structured_approximation_gate_authorized": len(passed)
            == len(protocol["targets"]),
            "model_implementation_authorized": False,
            "mfu_preflight_authorized": False,
            "language_model_training_authorized": False,
            "larger_rung_authorized": False,
        },
        "artifacts": {
            "cells": {"path": str(cells_path), "sha256": file_sha256(cells_path)},
            "overlap": {
                "path": str(overlap_path),
                "sha256": file_sha256(overlap_path),
            },
            "storage": {
                "path": str(storage_path),
                "sha256": file_sha256(storage_path),
            },
        },
        "all_reported_values_finite": all_finite(summaries),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
