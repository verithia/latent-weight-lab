#!/usr/bin/env python3
"""Test whether causal activation frames align with future dense c_proj motion.

For each registered phase start, this diagnostic reconstructs the dense GPT
from an all-parameter trajectory snapshot and collects pre- and post-GELU
activations on two disjoint deterministic validation windows.  It then asks
whether their principal subspaces capture the *future* phase endpoint update
of ``mlp.c_proj``.  Every capture is normalized by the ``k / 3072`` expected
energy of a task-independent random k-dimensional subspace.

This is a structure-selection diagnostic.  It does not train a candidate,
learn a basis, or use a future endpoint to construct the activation frame.
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

from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.parameter_trajectory import (
    FULL_STATE_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    persistent_buffer_names,
)


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


def parse_int_list(value: str) -> list[int]:
    output = [int(part) for part in value.split(",") if part]
    if not output or output != sorted(set(output)):
        raise argparse.ArgumentTypeError(
            "expected a non-empty sorted unique integer list"
        )
    return output


def randomized_principal_basis(
    values: torch.Tensor,
    rank: int,
    *,
    center: bool,
    seed: int,
    oversample: int = 16,
    power_iterations: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return a deterministic randomized right-principal basis and spectrum."""
    if values.ndim != 2:
        raise ValueError("principal-basis input must be rank two")
    matrix = values.float()
    if center:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    maximum = min(matrix.shape)
    if rank <= 0 or rank > maximum:
        raise ValueError(f"rank must be in [1, {maximum}]")
    sketch_rank = min(maximum, rank + max(0, int(oversample)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    omega = torch.randn(
        matrix.shape[1],
        sketch_rank,
        generator=generator,
        dtype=torch.float32,
    ).to(matrix.device)
    q = torch.linalg.qr(matrix @ omega, mode="reduced").Q
    for _ in range(int(power_iterations)):
        z = torch.linalg.qr(matrix.T @ q, mode="reduced").Q
        q = torch.linalg.qr(matrix @ z, mode="reduced").Q
    _left, singular, right_h = torch.linalg.svd(
        q.T @ matrix,
        full_matrices=False,
    )
    total_energy = float(matrix.square().sum())
    return (
        right_h[:rank].T.contiguous(),
        singular[:rank].contiguous(),
        total_energy,
    )


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape != right.shape
    ):
        raise ValueError("subspaces must have the same [width, rank] shape")
    return float((left.T @ right).square().sum() / left.shape[1])


def update_energy_capture(
    update: torch.Tensor,
    basis: torch.Tensor,
) -> float:
    if update.ndim != 2 or basis.ndim != 2:
        raise ValueError("update and basis must be rank two")
    if update.shape[1] != basis.shape[0]:
        raise ValueError("update width and basis width disagree")
    denominator = update.float().square().sum().clamp_min(1e-30)
    return float((update.float() @ basis.float()).square().sum() / denominator)


def mean_direction(values: torch.Tensor) -> torch.Tensor:
    mean = values.float().mean(dim=0)
    return mean / mean.norm().clamp_min(1e-30)


class ActivationCollector:
    def __init__(
        self,
        model: GPT,
        layers: list[int],
        sample_cap: int,
    ) -> None:
        self.layers = set(layers)
        self.sample_cap = int(sample_cap)
        self.values: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(
            list
        )
        self.counts: dict[tuple[int, str], int] = defaultdict(int)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer, block in enumerate(model.transformer.h):
            if layer not in self.layers:
                continue
            self.handles.append(
                block.mlp.c_fc.register_forward_hook(
                    self._hook(layer, "pre_gelu")
                )
            )
            self.handles.append(
                block.mlp.gelu.register_forward_hook(
                    self._hook(layer, "post_gelu")
                )
            )

    def _hook(self, layer: int, point: str):
        def hook(_module, _inputs, output):
            key = (layer, point)
            remaining = self.sample_cap - self.counts[key]
            if remaining <= 0:
                return
            rows = output.detach().float().reshape(-1, output.shape[-1])
            rows = rows[:remaining].cpu()
            self.values[key].append(rows)
            self.counts[key] += int(rows.shape[0])

        return hook

    def complete(self) -> bool:
        return all(
            self.counts[(layer, point)] >= self.sample_cap
            for layer in self.layers
            for point in ("pre_gelu", "post_gelu")
        )

    def tensor(self, layer: int, point: str) -> torch.Tensor:
        return torch.cat(self.values[(layer, point)], dim=0)[
            : self.sample_cap
        ]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_snapshot(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS
        or payload.get("all_parameters") is not True
    ):
        raise ValueError(f"not an all-parameter trajectory snapshot: {path}")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(f"snapshot has no parameters: {path}")
    return payload


def model_from_snapshot(payload: dict[str, Any], device: str) -> GPT:
    config = GPTConfig(**payload["model_config"])
    with torch.device(device):
        model = GPT(config)
    destination = dict(model.named_parameters())
    source = payload["parameters"]
    if set(destination) != set(source):
        missing = sorted(set(destination) - set(source))
        unexpected = sorted(set(source) - set(destination))
        raise ValueError(
            "snapshot parameter inventory mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in destination.items():
            parameter.copy_(source[name].to(device=device, dtype=parameter.dtype))
        if payload.get("schema_version") == FULL_STATE_SCHEMA_VERSION:
            if payload.get("all_buffers") is not True:
                raise ValueError("full-state snapshot does not declare all buffers")
            source_buffers = payload.get("buffers")
            if not isinstance(source_buffers, dict):
                raise ValueError("full-state snapshot has no buffer mapping")
            destination_buffers = dict(model.named_buffers())
            expected_buffers = persistent_buffer_names(model)
            if set(source_buffers) != expected_buffers:
                missing = sorted(expected_buffers - set(source_buffers))
                unexpected = sorted(set(source_buffers) - expected_buffers)
                raise ValueError(
                    "snapshot persistent-buffer inventory mismatch: "
                    f"missing={missing} unexpected={unexpected}"
                )
            for name in sorted(expected_buffers):
                buffer = destination_buffers[name]
                buffer.copy_(
                    source_buffers[name].to(device=device, dtype=buffer.dtype)
                )
    model.eval()
    return model


def collect_activations(
    model: GPT,
    batches: list[torch.Tensor],
    layers: list[int],
    sample_cap: int,
    device: str,
) -> dict[tuple[int, str], torch.Tensor]:
    collector = ActivationCollector(model, layers, sample_cap)
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch.to(device), None)
                if collector.complete():
                    break
        if not collector.complete():
            raise RuntimeError("activation sample cap was not reached")
        return {
            (layer, point): collector.tensor(layer, point)
            for layer in layers
            for point in ("pre_gelu", "post_gelu")
        }
    finally:
        collector.close()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["point"]), int(row["rank"]))].append(row)
    output: dict[str, Any] = {}
    for (point, rank), selected in sorted(groups.items()):
        key = f"{point}_rank{rank}"
        metrics = (
            "fit_update_energy_enrichment",
            "holdout_update_energy_enrichment",
            "minimum_update_energy_enrichment",
            "fit_holdout_subspace_enrichment",
            "fit_update_top_subspace_enrichment",
            "holdout_update_top_subspace_enrichment",
            "fit_activation_energy_fraction",
            "holdout_activation_energy_fraction",
        )
        output[key] = {
            "cells": len(selected),
            **{
                f"mean_{metric}": sum(float(row[metric]) for row in selected)
                / len(selected)
                for metric in metrics
            },
            "minimum_cell_update_energy_enrichment": min(
                float(row["minimum_update_energy_enrichment"])
                for row in selected
            ),
            "cells_minimum_update_energy_enrichment_ge_1p5": sum(
                float(row["minimum_update_energy_enrichment"]) >= 1.5
                for row in selected
            ),
        }
    post = output.get("post_gelu_rank64")
    pre = output.get("pre_gelu_rank64")
    if post is None or pre is None:
        raise ValueError("rank 64 must be included for the preregistered decision")
    post_minimum = float(post["mean_minimum_update_energy_enrichment"])
    post_stability = float(post["mean_fit_holdout_subspace_enrichment"])
    post_cells = int(post["cells_minimum_update_energy_enrichment_ge_1p5"])
    pre_minimum = float(pre["mean_minimum_update_energy_enrichment"])
    if (
        post_minimum >= 2.0
        and post_stability >= 2.0
        and post_cells >= 15
        and post_minimum >= 1.2 * pre_minimum
    ):
        decision = "PROMOTE_POSTGELU_ACTIVATION_FRAME"
    elif post_minimum >= 1.5 and post_cells >= 10:
        decision = "PARTIAL_POSTGELU_SIGNAL_REQUIRES_STABILITY_FIX"
    else:
        decision = "REJECT_ACTIVATION_PCA_AS_TRANSPORT_FRAME"
    return {
        "groups": output,
        "decision": decision,
        "decision_rule": {
            "promote": (
                "post-GELU rank64 mean minimum-window update-energy "
                "enrichment >=2; mean cross-window subspace enrichment >=2; "
                "at least 15/20 cells have minimum-window enrichment >=1.5; "
                "post-GELU mean >=1.2x pre-GELU mean"
            ),
            "partial": (
                "post-GELU rank64 mean minimum-window update-energy "
                "enrichment >=1.5 and at least 10/20 cells >=1.5"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--phase-boundaries", default="0,60,120,180,238")
    parser.add_argument("--ranks", default="16,64,128")
    parser.add_argument("--sample-cap", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--fit-seed", type=int, default=20260729)
    parser.add_argument("--holdout-seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = parse_int_list(args.layers)
    boundaries = parse_int_list(args.phase_boundaries)
    ranks = parse_int_list(args.ranks)
    if len(boundaries) < 2 or max(ranks) > args.sample_cap:
        raise ValueError("invalid phase boundaries, ranks, or sample cap")
    batches_needed = (
        args.sample_cap + args.batch_size * args.block_size - 1
    ) // (args.batch_size * args.block_size)
    fit_batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        batches_needed,
        args.fit_seed,
    )
    holdout_batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        batches_needed,
        args.holdout_seed,
    )
    snapshot_paths = [
        args.snapshot_dir / f"step_{step:06d}.pt" for step in boundaries
    ]
    missing = [str(path) for path in snapshot_paths if not path.is_file()]
    if missing:
        raise ValueError(f"required snapshots are absent: {missing}")

    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for phase_index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        start_path = snapshot_paths[phase_index]
        end_path = snapshot_paths[phase_index + 1]
        start_payload = load_snapshot(start_path)
        end_payload = load_snapshot(end_path)
        if int(start_payload["step"]) != start or int(end_payload["step"]) != end:
            raise ValueError("snapshot step does not match its registered path")
        if (
            start_payload["run_identity_sha256"]
            != end_payload["run_identity_sha256"]
        ):
            raise ValueError("phase snapshots have different run identities")
        if not inventory:
            inventory = [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in snapshot_paths
            ]
        model = model_from_snapshot(start_payload, args.device)
        try:
            fit = collect_activations(
                model, fit_batches, layers, args.sample_cap, args.device
            )
            holdout = collect_activations(
                model, holdout_batches, layers, args.sample_cap, args.device
            )
        finally:
            del model
            if "cuda" in args.device:
                torch.cuda.empty_cache()

        for layer in layers:
            parameter_name = f"transformer.h.{layer}.mlp.c_proj.weight"
            update = (
                end_payload["parameters"][parameter_name].float()
                - start_payload["parameters"][parameter_name].float()
            ).to(args.device)
            maximum_rank = max(ranks)
            update_basis, update_singular, update_total = (
                randomized_principal_basis(
                    update,
                    maximum_rank,
                    center=False,
                    seed=1000003 + 101 * phase_index + layer,
                )
            )
            for point_index, point in enumerate(("pre_gelu", "post_gelu")):
                fit_values = fit[(layer, point)].to(args.device)
                holdout_values = holdout[(layer, point)].to(args.device)
                fit_basis, fit_singular, fit_total = (
                    randomized_principal_basis(
                        fit_values,
                        maximum_rank,
                        center=True,
                        seed=2000003 + 1009 * phase_index + 31 * layer + point_index,
                    )
                )
                holdout_basis, holdout_singular, holdout_total = (
                    randomized_principal_basis(
                        holdout_values,
                        maximum_rank,
                        center=True,
                        seed=3000017 + 1009 * phase_index + 31 * layer + point_index,
                    )
                )
                fit_mean = mean_direction(fit_values)
                holdout_mean = mean_direction(holdout_values)
                mean_random = 1.0 / update.shape[1]
                common = {
                    "layer": layer,
                    "phase_start": start,
                    "phase_end": end,
                    "point": point,
                    "samples": args.sample_cap,
                    "width": update.shape[1],
                    "fit_mean_update_energy_enrichment": (
                        update_energy_capture(update, fit_mean[:, None])
                        / mean_random
                    ),
                    "holdout_mean_update_energy_enrichment": (
                        update_energy_capture(update, holdout_mean[:, None])
                        / mean_random
                    ),
                    "fit_holdout_mean_abs_cosine": float(
                        torch.dot(fit_mean, holdout_mean).abs()
                    ),
                    "update_total_energy": update_total,
                }
                for rank in ranks:
                    random_fraction = rank / update.shape[1]
                    fit_selected = fit_basis[:, :rank]
                    holdout_selected = holdout_basis[:, :rank]
                    update_selected = update_basis[:, :rank]
                    fit_capture = update_energy_capture(update, fit_selected)
                    holdout_capture = update_energy_capture(
                        update, holdout_selected
                    )
                    row = {
                        **common,
                        "rank": rank,
                        "random_subspace_fraction": random_fraction,
                        "fit_update_energy_capture": fit_capture,
                        "holdout_update_energy_capture": holdout_capture,
                        "fit_update_energy_enrichment": (
                            fit_capture / random_fraction
                        ),
                        "holdout_update_energy_enrichment": (
                            holdout_capture / random_fraction
                        ),
                        "minimum_update_energy_enrichment": (
                            min(fit_capture, holdout_capture)
                            / random_fraction
                        ),
                        "fit_holdout_subspace_overlap": subspace_overlap(
                            fit_selected, holdout_selected
                        ),
                        "fit_holdout_subspace_enrichment": (
                            subspace_overlap(
                                fit_selected, holdout_selected
                            )
                            / random_fraction
                        ),
                        "fit_update_top_subspace_overlap": subspace_overlap(
                            fit_selected, update_selected
                        ),
                        "holdout_update_top_subspace_overlap": subspace_overlap(
                            holdout_selected, update_selected
                        ),
                        "fit_update_top_subspace_enrichment": (
                            subspace_overlap(
                                fit_selected, update_selected
                            )
                            / random_fraction
                        ),
                        "holdout_update_top_subspace_enrichment": (
                            subspace_overlap(
                                holdout_selected, update_selected
                            )
                            / random_fraction
                        ),
                        "fit_activation_energy_fraction": float(
                            fit_singular[:rank].square().sum()
                            / max(fit_total, 1e-30)
                        ),
                        "holdout_activation_energy_fraction": float(
                            holdout_singular[:rank].square().sum()
                            / max(holdout_total, 1e-30)
                        ),
                        "update_top_energy_fraction": float(
                            update_singular[:rank].square().sum()
                            / max(update_total, 1e-30)
                        ),
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
            del update
        del start_payload, end_payload, fit, holdout
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    aggregate = aggregate_rows(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "activation_update_alignment.csv"
    aggregate_path = args.output / "activation_update_alignment_aggregate.json"
    write_csv(detail_path, rows)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_mlp_activation_update_alignment_v1",
        "causal_protocol": (
            "phase-start activations predict the future dense-Muon c_proj "
            "endpoint update; fit and holdout token windows are disjoint"
        ),
        "layers": layers,
        "phase_boundaries": boundaries,
        "ranks": ranks,
        "sample_cap": args.sample_cap,
        "fit_seed": args.fit_seed,
        "holdout_seed": args.holdout_seed,
        "activation_centering": "subtract per-window channel mean before PCA",
        "random_reference": "k / hidden_width",
        "snapshot_files": inventory,
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
            "detail": {
                "path": str(detail_path),
                "sha256": file_sha256(detail_path),
            },
            "aggregate": {
                "path": str(aggregate_path),
                "sha256": file_sha256(aggregate_path),
            },
        },
        "limitations": [
            "Activation PCA is an observable diagnostic basis, not a deployed mapping parameterization.",
            "Only five preregistered representative layers and four phase endpoints are analyzed.",
            "A positive endpoint-energy result would still require a causal compact transport implementation and an MFU-qualified training run.",
        ],
    }
    metadata_path = args.output / "activation_update_alignment_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "aggregate": aggregate,
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
