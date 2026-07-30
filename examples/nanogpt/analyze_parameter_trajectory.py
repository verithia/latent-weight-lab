#!/usr/bin/env python3
"""Analyze dense layer-weight trajectories without claiming a solution-manifold dimension.

The primary calculation mirrors the Mapping Networks paper's layerwise snapshot
PCA, but makes the missing protocol explicit.  It also reports chord residuals,
path curvature, temporal polynomial fits, and matrix-axis norm evolution.  A
cross-layer t-SNE view is optional and is treated only as a visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.parameter_trajectory import SCHEMA_VERSION


PARAMETER_PATTERN = re.compile(
    r"^transformer\.h\.(?P<layer>\d+)\."
    r"(?P<target>(?:mlp\.(?:c_fc|c_proj)|attn\.(?:c_attn|c_proj)))"
    r"\.weight$"
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    result = [int(item) for item in value.split(",") if item]
    if any(item < 0 for item in result):
        raise ValueError("steps/layers must be non-negative")
    return result


def energy_dimension(eigenvalues: torch.Tensor, fraction: float) -> int:
    if eigenvalues.numel() == 0 or float(eigenvalues.sum()) <= 0.0:
        return 0
    cumulative = eigenvalues.cumsum(0) / eigenvalues.sum()
    return int(torch.searchsorted(cumulative, torch.tensor(fraction, device=cumulative.device)).item() + 1)


def pca_from_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if rows.ndim != 2:
        raise ValueError("PCA input must be a matrix")
    gram = rows @ rows.T
    gram = ((gram + gram.T) * 0.5).double()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    scores = eigenvectors * eigenvalues.sqrt().unsqueeze(0)
    return eigenvalues, scores


def polynomial_r2(time: torch.Tensor, values: torch.Tensor, degree: int) -> float:
    columns = [time.pow(power) for power in range(degree + 1)]
    design = torch.stack(columns, dim=1).double()
    target = values.double().unsqueeze(1)
    coefficients = torch.linalg.lstsq(design, target).solution
    prediction = (design @ coefficients).squeeze(1)
    centered = target.squeeze(1) - target.mean()
    denominator = centered.square().sum()
    if float(denominator) <= 0.0:
        return 1.0
    return float(1.0 - (target.squeeze(1) - prediction).square().sum() / denominator)


def axis_norm_metrics(matrix: torch.Tensor) -> dict[str, float]:
    axis0 = matrix.norm(dim=1)
    axis1 = matrix.norm(dim=0)

    def metrics(values: torch.Tensor, prefix: str) -> dict[str, float]:
        mean = values.mean().clamp_min(1e-30)
        minimum = values.min().clamp_min(1e-30)
        return {
            f"{prefix}_norm_mean": float(mean),
            f"{prefix}_norm_cv": float(values.std(unbiased=False) / mean),
            f"{prefix}_norm_max_min": float(values.max() / minimum),
        }

    return {**metrics(axis0, "axis0"), **metrics(axis1, "axis1")}


def summarize_parameter(
    *,
    name: str,
    steps: list[int],
    tensors: list[torch.Tensor],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(steps) < 3 or len(steps) != len(tensors):
        raise ValueError("trajectory analysis requires at least three aligned snapshots")
    matrix_shape = tuple(tensors[0].shape)
    if len(matrix_shape) != 2 or any(tuple(tensor.shape) != matrix_shape for tensor in tensors):
        raise ValueError(f"trajectory tensors must share one matrix shape: {name}")
    match = PARAMETER_PATTERN.match(name)
    if match is None:
        raise ValueError(f"unsupported parameter name: {name}")

    positions = torch.stack(tensors).to(device=device, dtype=torch.float32)
    centered = positions - positions.mean(dim=0, keepdim=True)
    centered_flat = centered.flatten(1)
    eigenvalues, scores = pca_from_rows(centered_flat)
    total_energy = eigenvalues.sum().clamp_min(1e-30)
    probabilities = eigenvalues / total_energy

    displacements = positions[1:] - positions[0]
    displacement_eigenvalues, _ = pca_from_rows(displacements.flatten(1))
    displacement_total = displacement_eigenvalues.sum().clamp_min(1e-30)

    chord = (positions[-1] - positions[0]).flatten()
    chord_norm = chord.norm().clamp_min(1e-30)
    all_displacements = (positions - positions[0]).flatten(1)
    chord_progress = (all_displacements @ chord) / chord_norm.square()
    chord_residuals = all_displacements - chord_progress.unsqueeze(1) * chord.unsqueeze(0)
    displacement_norms = all_displacements.norm(dim=1)
    relative_residuals = chord_residuals.norm(dim=1) / displacement_norms.clamp_min(1e-30)
    relative_residuals = relative_residuals[1:]

    increments = (positions[1:] - positions[:-1]).flatten(1)
    increment_norms = increments.norm(dim=1)
    path_length = increment_norms.sum()
    if increments.shape[0] > 1:
        consecutive_cosines = torch.nn.functional.cosine_similarity(
            increments[:-1],
            increments[1:],
            dim=1,
            eps=1e-30,
        )
        turn_degrees = torch.rad2deg(torch.acos(consecutive_cosines.clamp(-1.0, 1.0)))
    else:
        consecutive_cosines = torch.ones(1, device=positions.device)
        turn_degrees = torch.zeros(1, device=positions.device)

    normalized_time = torch.tensor(steps, device=positions.device, dtype=torch.float64)
    normalized_time = (normalized_time - normalized_time[0]) / max(steps[-1] - steps[0], 1)
    pc_rows: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        row: dict[str, Any] = {
            "parameter": name,
            "layer": int(match.group("layer")),
            "target": match.group("target"),
            "step": step,
        }
        for component in range(min(10, scores.shape[1])):
            row[f"pc{component + 1}"] = float(scores[index, component])
        pc_rows.append(row)

    polynomial_rows = [
        {
            "parameter": name,
            "layer": int(match.group("layer")),
            "target": match.group("target"),
            "component": component + 1,
            "polynomial_degree": component + 1,
            "r2": polynomial_r2(normalized_time, scores[:, component], component + 1),
            "interpretation": "descriptive temporal fit; not a manifold-dimension estimate",
        }
        for component in range(min(5, scores.shape[1]))
    ]

    positive = eigenvalues[eigenvalues > eigenvalues.max().clamp_min(1e-30) * 1e-8]
    participation = total_energy.square() / eigenvalues.square().sum().clamp_min(1e-30)
    summary = {
        "parameter": name,
        "layer": int(match.group("layer")),
        "target": match.group("target"),
        "snapshots": len(steps),
        "first_step": steps[0],
        "last_step": steps[-1],
        "matrix_axis0": matrix_shape[0],
        "matrix_axis1": matrix_shape[1],
        "sampled_path_rank_upper_bound": len(steps) - 1,
        "numerical_centered_rank": int(positive.numel()),
        "pc1_energy": float(probabilities[0]),
        "pc1_pc2_energy": float(probabilities[:2].sum()),
        "dimension_90pct": energy_dimension(eigenvalues, 0.90),
        "dimension_95pct": energy_dimension(eigenvalues, 0.95),
        "dimension_99pct": energy_dimension(eigenvalues, 0.99),
        "participation_dimension": float(participation),
        "init_relative_displacement_pc1_energy": float(displacement_eigenvalues[0] / displacement_total),
        "path_length_over_chord": float(path_length / chord_norm),
        "median_relative_chord_residual": float(relative_residuals.median()),
        "max_relative_chord_residual": float(relative_residuals.max()),
        "mean_consecutive_increment_cosine": float(consecutive_cosines.mean()),
        "minimum_consecutive_increment_cosine": float(consecutive_cosines.min()),
        "median_turn_degrees": float(turn_degrees.median()),
        "max_turn_degrees": float(turn_degrees.max()),
        "monotone_chord_progress_fraction": float(
            (chord_progress[1:] >= chord_progress[:-1] - 1e-7).float().mean()
        ),
        "terminal_displacement_fro": float(chord_norm),
    }
    del positions, centered, centered_flat, displacements, all_displacements, increments
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary, pc_rows, polynomial_rows


def spectral_metrics(matrix: torch.Tensor) -> dict[str, float | int]:
    singular = torch.linalg.svdvals(matrix.float())
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-30)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return {
        "fro": float(energy.sum().sqrt()),
        "spectral_norm": float(singular[0]),
        "stable_rank": float(energy.sum() / energy[0].clamp_min(1e-30)),
        "entropy_effective_rank": float(entropy.exp()),
        "top1_energy": float(probability[0]),
        "singular_dimension_90pct": energy_dimension(energy, 0.90),
    }


def load_snapshots(
    paths: list[Path],
    *,
    layers: set[int] | None,
    targets: set[str] | None,
) -> tuple[list[int], dict[str, list[torch.Tensor]], dict[str, Any]]:
    steps: list[int] = []
    values: dict[str, list[torch.Tensor]] = {}
    metadata: dict[str, Any] | None = None
    expected_names: set[str] | None = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory snapshot: {path}")
        if metadata is None:
            metadata = {
                "schema_version": payload["schema_version"],
                "run_identity": payload["run_identity"],
                "run_identity_sha256": payload["run_identity_sha256"],
                "model_config": payload["model_config"],
                "storage_dtype": payload["storage_dtype"],
                "execution_provenance": payload.get("execution_provenance"),
            }
        elif (
            payload.get("run_identity_sha256") != metadata["run_identity_sha256"]
            or payload.get("model_config") != metadata["model_config"]
        ):
            raise ValueError("snapshots do not belong to one training trajectory")
        selected: dict[str, torch.Tensor] = {}
        for name, tensor in payload["parameters"].items():
            match = PARAMETER_PATTERN.match(name)
            if match is None:
                continue
            layer = int(match.group("layer"))
            target = match.group("target")
            if layers is not None and layer not in layers:
                continue
            if targets is not None and target not in targets:
                continue
            selected[name] = tensor.detach().float().contiguous()
        if expected_names is None:
            expected_names = set(selected)
            if not expected_names:
                raise ValueError("snapshot filter selected no parameters")
            values = {name: [] for name in sorted(expected_names)}
        elif set(selected) != expected_names:
            raise ValueError("snapshot parameter inventory changed within the trajectory")
        steps.append(int(payload["step"]))
        for name, tensor in selected.items():
            values[name].append(tensor)

    if metadata is None or len(steps) < 3:
        raise ValueError("at least three snapshots are required")
    order = sorted(range(len(steps)), key=steps.__getitem__)
    sorted_steps = [steps[index] for index in order]
    if len(set(sorted_steps)) != len(sorted_steps):
        raise ValueError("trajectory snapshot steps must be unique")
    values = {
        name: [tensors[index] for index in order]
        for name, tensors in values.items()
    }
    return sorted_steps, values, metadata


def cross_layer_embedding(
    *,
    target: str,
    steps: list[int],
    values: dict[str, list[torch.Tensor]],
    device: str,
    tsne_seed: int,
) -> list[dict[str, Any]]:
    rows: list[torch.Tensor] = []
    labels: list[tuple[int, int]] = []
    for name, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(name)
        assert match is not None
        if match.group("target") != target:
            continue
        stacked = torch.stack(tensors).to(device=device, dtype=torch.float32)
        displacement = stacked - stacked[0]
        scale = displacement[-1].norm().clamp_min(1e-30)
        normalized = (displacement / scale).flatten(1)
        rows.extend(normalized[index] for index in range(len(steps)))
        labels.extend((int(match.group("layer")), step) for step in steps)
    if not rows:
        return []
    matrix = torch.stack(rows)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    eigenvalues, scores = pca_from_rows(centered)
    usable = min(32, int((eigenvalues > eigenvalues.max().clamp_min(1e-30) * 1e-8).sum()))
    features = scores[:, : max(usable, 2)].cpu().numpy()
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError("--tsne requires scikit-learn") from exc
    perplexity = min(30.0, max(5.0, (len(labels) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=tsne_seed,
    ).fit_transform(features)
    result: list[dict[str, Any]] = []
    for index, (layer, step) in enumerate(labels):
        result.append(
            {
                "target": target,
                "layer": layer,
                "step": step,
                "pca1": float(scores[index, 0]),
                "pca2": float(scores[index, 1]),
                "pca3": float(scores[index, 2]) if scores.shape[1] > 2 else 0.0,
                "tsne1": float(embedding[index, 0]),
                "tsne2": float(embedding[index, 1]),
                "tsne_input": f"top_{max(usable, 2)}_exact_gram_pca_scores",
                "tsne_perplexity": perplexity,
                "interpretation": "visualization only; t-SNE does not estimate dimension or near-affinity",
            }
        )
    del matrix, centered
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="")
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--spectral-steps", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--tsne-seed", type=int, default=20260728)
    args = parser.parse_args()

    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    layers_list = parse_int_list(args.layers)
    targets = {item for item in args.targets.split(",") if item}
    steps, values, metadata = load_snapshots(
        paths,
        layers=set(layers_list) if layers_list else None,
        targets=targets if targets else None,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []
    polynomial_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    requested_spectral_steps = set(parse_int_list(args.spectral_steps))
    step_to_index = {step: index for index, step in enumerate(steps)}
    missing_spectral_steps = sorted(requested_spectral_steps - step_to_index.keys())
    if missing_spectral_steps:
        raise ValueError(f"spectral steps are absent from snapshots: {missing_spectral_steps}")

    for name, tensors in sorted(values.items()):
        summary, coordinates, polynomial = summarize_parameter(
            name=name,
            steps=steps,
            tensors=tensors,
            device=args.device,
        )
        summaries.append(summary)
        pca_rows.extend(coordinates)
        polynomial_rows.extend(polynomial)
        match = PARAMETER_PATTERN.match(name)
        assert match is not None
        for step, tensor in zip(steps, tensors, strict=True):
            structure_rows.append(
                {
                    "parameter": name,
                    "layer": int(match.group("layer")),
                    "target": match.group("target"),
                    "step": step,
                    "weight_fro": float(tensor.norm()),
                    **axis_norm_metrics(tensor),
                }
            )
        for step in sorted(requested_spectral_steps):
            matrix = tensors[step_to_index[step]].to(args.device)
            spectral_rows.append(
                {
                    "parameter": name,
                    "layer": int(match.group("layer")),
                    "target": match.group("target"),
                    "step": step,
                    **spectral_metrics(matrix),
                }
            )
            del matrix
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    embedding_rows: list[dict[str, Any]] = []
    if args.tsne:
        for target in sorted(targets):
            embedding_rows.extend(
                cross_layer_embedding(
                    target=target,
                    steps=steps,
                    values=values,
                    device=args.device,
                    tsne_seed=args.tsne_seed,
                )
            )

    write_csv(args.output / "layerwise_trajectory_summary.csv", summaries)
    write_csv(args.output / "layerwise_pca_coordinates.csv", pca_rows)
    write_csv(args.output / "temporal_polynomial_fits.csv", polynomial_rows)
    write_csv(args.output / "matrix_axis_norms.csv", structure_rows)
    write_csv(args.output / "selected_step_spectra.csv", spectral_rows)
    write_csv(args.output / "cross_layer_pca_tsne.csv", embedding_rows)
    analysis_metadata = {
        **metadata,
        "snapshot_dir": str(args.snapshot_dir),
        "snapshot_paths": [str(path) for path in paths],
        "steps": steps,
        "parameters": sorted(values),
        "method": {
            "layerwise_pca": "exact mean-centered checkpoint positions via temporal Gram eigendecomposition",
            "displacement_pca": "uncentered init-relative displacements",
            "near_affinity": "distance to init-terminal chord plus consecutive-increment turning",
            "temporal_polynomials": "PC_k score regressed on powers 0..k of normalized training step",
            "cross_layer_tsne": (
                "t-SNE of top-32 exact Gram-PCA scores from terminal-norm-normalized init-relative "
                "displacements; visualization only"
                if args.tsne
                else "not requested"
            ),
        },
        "limitations": [
            "A single optimizer trajectory is not the manifold of good solutions.",
            "The sampled trajectory rank is bounded by snapshot_count - 1.",
            "High PCA concentration can reflect schedule, weight decay, or a directed optimizer path.",
            "t-SNE does not establish intrinsic dimension, smoothness, or affine structure.",
            "Token Geometry's 1D ray was measured per token under Ember and is not assumed for transformer matrices.",
        ],
    }
    (args.output / "analysis_metadata.json").write_text(
        json.dumps(analysis_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "snapshots": len(steps),
                "parameters": len(values),
                "summary": str(args.output / "layerwise_trajectory_summary.csv"),
                "tsne": args.tsne,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
