#!/usr/bin/env python3
"""Resolve the spatial structure hidden by temporal MLP trajectory PCA.

Temporal PCA can show that one optimizer path uses few directions without
showing whether those directions are low matrix-rank, shared between layers,
or coupled between c_fc and c_proj.  This companion analysis measures those
properties and produces the PCA/t-SNE views used only for visualization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_parameter_trajectory import (
    PARAMETER_PATTERN,
    axis_norm_metrics,
    load_snapshots,
    parse_int_list,
    spectral_metrics,
    write_csv,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().flatten()
    right = right.double().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator) <= 0.0:
        return 0.0
    return float((left @ right) / denominator)


def rank_values(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values.flatten(), stable=True)
    result = torch.empty_like(order, dtype=torch.float64)
    result[order] = torch.arange(order.numel(), device=order.device, dtype=torch.float64)
    return result


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return mean squared canonical cosine, one for identical subspaces."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("subspace bases must have the same matrix shape")
    rank = left.shape[1]
    if rank == 0:
        return 0.0
    return float((left.T @ right).square().sum() / rank)


def paired_metrics(
    c_fc_delta: torch.Tensor,
    c_proj_delta: torch.Tensor,
    *,
    ranks: list[int],
) -> dict[str, float]:
    if c_fc_delta.ndim != 2 or c_proj_delta.ndim != 2:
        raise ValueError("paired MLP deltas must be matrices")
    if c_fc_delta.shape != c_proj_delta.T.shape:
        raise ValueError("c_fc and transposed c_proj deltas must have matching shapes")
    result = {
        "frobenius_cosine_cfc_cproj_transpose": float(
            torch.nn.functional.cosine_similarity(
                c_fc_delta.flatten(),
                c_proj_delta.T.flatten(),
                dim=0,
                eps=1e-30,
            )
        ),
        "expansion_channel_delta_norm_pearson": pearson(
            c_fc_delta.norm(dim=1),
            c_proj_delta.norm(dim=0),
        ),
        "expansion_channel_delta_norm_spearman": pearson(
            rank_values(c_fc_delta.norm(dim=1)),
            rank_values(c_proj_delta.norm(dim=0)),
        ),
        "residual_channel_delta_norm_pearson": pearson(
            c_fc_delta.norm(dim=0),
            c_proj_delta.norm(dim=1),
        ),
        "residual_channel_delta_norm_spearman": pearson(
            rank_values(c_fc_delta.norm(dim=0)),
            rank_values(c_proj_delta.norm(dim=1)),
        ),
    }
    cfc_u, _, cfc_vh = torch.linalg.svd(c_fc_delta.float(), full_matrices=False)
    cproj_u, _, cproj_vh = torch.linalg.svd(c_proj_delta.float(), full_matrices=False)
    maximum = min(cfc_u.shape[1], cfc_vh.shape[0], cproj_u.shape[1], cproj_vh.shape[0])
    for requested_rank in ranks:
        rank = min(requested_rank, maximum)
        result[f"residual_subspace_overlap_rank{requested_rank}"] = subspace_overlap(
            cfc_vh[:rank].T,
            cproj_u[:, :rank],
        )
        result[f"expansion_subspace_overlap_rank{requested_rank}"] = subspace_overlap(
            cfc_u[:, :rank],
            cproj_vh[:rank].T,
        )
    return result


def temporal_pc_spatial_spectra(
    *,
    name: str,
    tensors: list[torch.Tensor],
    components: int,
    device: str,
) -> list[dict[str, Any]]:
    match = PARAMETER_PATTERN.match(name)
    if match is None:
        raise ValueError(f"unsupported parameter name: {name}")
    positions = torch.stack(tensors).to(device=device, dtype=torch.float32)
    centered = positions - positions.mean(dim=0, keepdim=True)
    flat = centered.flatten(1)
    gram = flat @ flat.T
    gram = (gram + gram.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    rows: list[dict[str, Any]] = []
    for component in range(min(components, flat.shape[0] - 1)):
        eigenvalue = eigenvalues[component]
        if float(eigenvalue) <= 0.0:
            continue
        direction = (
            eigenvectors[:, component].to(flat.dtype) @ flat
        ) / eigenvalue.sqrt().to(flat.dtype)
        matrix = direction.reshape_as(positions[0])
        rows.append(
            {
                "parameter": name,
                "layer": int(match.group("layer")),
                "target": match.group("target"),
                "component": component + 1,
                "temporal_energy_fraction": float(eigenvalue / eigenvalues.sum().clamp_min(1e-30)),
                **spectral_metrics(matrix),
                **axis_norm_metrics(matrix),
            }
        )
    del positions, centered, flat, gram
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def cross_layer_alignment_rows(
    *,
    values: dict[str, list[torch.Tensor]],
    device: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in ("mlp.c_fc", "mlp.c_proj"):
        selected: list[tuple[int, torch.Tensor]] = []
        for name, tensors in sorted(values.items()):
            match = PARAMETER_PATTERN.match(name)
            assert match is not None
            if match.group("target") != target:
                continue
            delta = (tensors[-1] - tensors[0]).flatten()
            selected.append((int(match.group("layer")), delta))
        matrix = torch.stack([delta for _, delta in selected]).to(device=device, dtype=torch.float32)
        matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-30)
        gram = matrix @ matrix.T
        for left_index, (left_layer, _) in enumerate(selected):
            for right_index in range(left_index + 1, len(selected)):
                right_layer = selected[right_index][0]
                result.append(
                    {
                        "target": target,
                        "left_layer": left_layer,
                        "right_layer": right_layer,
                        "adjacent_layers": right_layer == left_layer + 1,
                        "terminal_delta_cosine": float(gram[left_index, right_index]),
                        "terminal_delta_absolute_cosine": float(gram[left_index, right_index].abs()),
                    }
                )
        del matrix, gram
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return result


def paired_layer_rows(
    *,
    values: dict[str, list[torch.Tensor]],
    device: str,
    ranks: list[int],
) -> list[dict[str, Any]]:
    by_layer: dict[int, dict[str, list[torch.Tensor]]] = {}
    for name, tensors in values.items():
        match = PARAMETER_PATTERN.match(name)
        assert match is not None
        by_layer.setdefault(int(match.group("layer")), {})[match.group("target")] = tensors
    result: list[dict[str, Any]] = []
    for layer, targets in sorted(by_layer.items()):
        c_fc = (targets["mlp.c_fc"][-1] - targets["mlp.c_fc"][0]).to(
            device=device,
            dtype=torch.float32,
        )
        c_proj = (targets["mlp.c_proj"][-1] - targets["mlp.c_proj"][0]).to(
            device=device,
            dtype=torch.float32,
        )
        result.append({"layer": layer, **paired_metrics(c_fc, c_proj, ranks=ranks)})
        del c_fc, c_proj
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return result


def write_plots(trajectory_analysis: Path, output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca_rows = list(csv.DictReader((trajectory_analysis / "layerwise_pca_coordinates.csv").open()))
    tsne_rows = list(csv.DictReader((trajectory_analysis / "cross_layer_pca_tsne.csv").open()))
    written: list[str] = []
    for target in ("mlp.c_fc", "mlp.c_proj"):
        figure, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
        for layer, axis in enumerate(axes.flat):
            rows = sorted(
                (
                    row
                    for row in pca_rows
                    if row["target"] == target and int(row["layer"]) == layer
                ),
                key=lambda row: int(row["step"]),
            )
            steps = [int(row["step"]) for row in rows]
            axis.plot([float(row["pc1"]) for row in rows], [float(row["pc2"]) for row in rows], alpha=0.5)
            points = axis.scatter(
                [float(row["pc1"]) for row in rows],
                [float(row["pc2"]) for row in rows],
                c=steps,
                cmap="viridis",
                s=15,
            )
            axis.set_title(f"layer {layer}")
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
        figure.colorbar(points, ax=axes, label="optimizer step", shrink=0.7)
        destination = output / f"{target.replace('.', '_')}_layerwise_pca_paths.png"
        figure.suptitle(f"{target}: layerwise temporal PCA paths")
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        written.append(destination.name)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for target, axis in zip(("mlp.c_fc", "mlp.c_proj"), axes, strict=True):
        for layer in range(12):
            rows = sorted(
                (
                    row
                    for row in tsne_rows
                    if row["target"] == target and int(row["layer"]) == layer
                ),
                key=lambda row: int(row["step"]),
            )
            axis.plot(
                [float(row["tsne1"]) for row in rows],
                [float(row["tsne2"]) for row in rows],
                marker=".",
                markersize=2,
                linewidth=0.7,
                label=str(layer),
            )
        axis.set_title(f"{target}: cross-layer t-SNE")
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
    axes[1].legend(title="layer", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2)
    destination = output / "cross_layer_pca_tsne_paths.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    written.append(destination.name)
    return written


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--trajectory-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spectral-steps", default="60,120,180,238")
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--subspace-ranks", default="16,64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.components <= 0:
        raise ValueError("--components must be positive")
    started = time.time()
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    steps, values, snapshot_metadata = load_snapshots(paths, layers=None, targets=None)
    step_to_index = {step: index for index, step in enumerate(steps)}
    spectral_steps = parse_int_list(args.spectral_steps)
    missing = sorted(set(spectral_steps) - step_to_index.keys())
    if missing:
        raise ValueError(f"spectral steps are absent from snapshots: {missing}")
    ranks = parse_int_list(args.subspace_ranks)
    if not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError("--subspace-ranks must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    displacement_rows: list[dict[str, Any]] = []
    pc_direction_rows: list[dict[str, Any]] = []
    for name, tensors in sorted(values.items()):
        match = PARAMETER_PATTERN.match(name)
        assert match is not None
        initial = tensors[0]
        for step in spectral_steps:
            delta = (tensors[step_to_index[step]] - initial).to(
                device=args.device,
                dtype=torch.float32,
            )
            displacement_rows.append(
                {
                    "parameter": name,
                    "layer": int(match.group("layer")),
                    "target": match.group("target"),
                    "step": step,
                    **spectral_metrics(delta),
                    **axis_norm_metrics(delta),
                }
            )
            del delta
        pc_direction_rows.extend(
            temporal_pc_spatial_spectra(
                name=name,
                tensors=tensors,
                components=args.components,
                device=args.device,
            )
        )

    cross_rows = cross_layer_alignment_rows(values=values, device=args.device)
    pair_rows = paired_layer_rows(values=values, device=args.device, ranks=ranks)
    write_csv(args.output / "selected_step_displacement_spectra.csv", displacement_rows)
    write_csv(args.output / "temporal_pc_spatial_spectra.csv", pc_direction_rows)
    write_csv(args.output / "cross_layer_terminal_delta_alignment.csv", cross_rows)
    write_csv(args.output / "paired_cfc_cproj_terminal_delta_geometry.csv", pair_rows)
    plot_files = write_plots(args.trajectory_analysis, args.output) if args.plots else []

    script = Path(__file__).resolve()
    repo = script.parents[2]
    input_analysis_files = sorted(args.trajectory_analysis.glob("*"))
    metadata = {
        "schema_version": "nanogpt_mlp_trajectory_structure_v1",
        "snapshot_metadata": snapshot_metadata,
        "steps": steps,
        "spectral_steps": spectral_steps,
        "subspace_ranks": ranks,
        "analysis_execution": {
            "git_commit": git_commit(repo),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
        },
        "snapshot_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in paths
        ],
        "trajectory_analysis_inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in input_analysis_files
            if path.is_file()
        ],
        "plot_files": plot_files,
        "interpretation_limits": [
            "Temporal PC concentration is path dimension, not matrix rank or global solution dimension.",
            "Cross-layer cosines assume raw parameter coordinates have aligned channel identities.",
            "Subspace overlap is descriptive and does not establish causal functional coupling.",
            "t-SNE is visualization only and cannot establish smoothness, dimension, or near-affinity.",
        ],
    }
    (args.output / "structure_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "parameters": len(values),
                "snapshots": len(steps),
                "displacement_rows": len(displacement_rows),
                "pc_direction_rows": len(pc_direction_rows),
                "cross_layer_rows": len(cross_rows),
                "paired_layer_rows": len(pair_rows),
                "plots": plot_files,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
