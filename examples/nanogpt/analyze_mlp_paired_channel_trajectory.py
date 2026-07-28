"""Measure signed c_fc/c_proj hidden-channel motion along a trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().reshape(-1)
    right = right.double().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float((left @ right) / denominator)


def temporal_pca_energy(trajectory: torch.Tensor) -> dict[str, float | int]:
    centered = trajectory.double() - trajectory.double().mean(dim=0)
    eigenvalues = torch.linalg.eigvalsh(centered @ centered.transpose(0, 1))
    eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
    fractions = eigenvalues / eigenvalues.sum().clamp_min(
        torch.finfo(eigenvalues.dtype).tiny
    )
    cumulative = fractions.cumsum(0)

    def dimension(threshold: float) -> int:
        return int(
            torch.searchsorted(
                cumulative,
                torch.tensor(threshold, dtype=cumulative.dtype),
            )
        ) + 1

    return {
        "pc1_energy": float(fractions[0]),
        "pc1_pc2_energy": float(fractions[:2].sum()),
        "dimension_90": dimension(0.90),
        "dimension_95": dimension(0.95),
        "dimension_99": dimension(0.99),
    }


def layer_keys(layer: int) -> tuple[str, str]:
    prefix = f"transformer.h.{layer}.mlp"
    return f"{prefix}.c_fc.weight", f"{prefix}.c_proj.weight"


def normalized_radial_coordinates(
    initial_fc: torch.Tensor,
    initial_proj: torch.Tensor,
    current_fc: torch.Tensor,
    current_proj: torch.Tensor,
) -> dict[str, float]:
    delta_fc = (current_fc - initial_fc).double()
    delta_proj = (current_proj - initial_proj).double()
    unit_fc = initial_fc.double()
    unit_fc = unit_fc / unit_fc.norm(dim=1, keepdim=True).clamp_min(1e-30)
    unit_proj = initial_proj.double()
    unit_proj = unit_proj / unit_proj.norm(dim=0, keepdim=True).clamp_min(
        1e-30
    )
    radial_fc = (delta_fc * unit_fc).sum(dim=1)
    radial_proj = (delta_proj * unit_proj).sum(dim=0)
    common = (radial_fc + radial_proj) / math.sqrt(2.0)
    gauge = (radial_fc - radial_proj) / math.sqrt(2.0)
    total_energy = (
        delta_fc.square().sum() + delta_proj.square().sum()
    ).clamp_min(1e-30)
    return {
        "signed_radial_correlation": pearson(radial_fc, radial_proj),
        "independent_radial_capture": float(
            (radial_fc.square().sum() + radial_proj.square().sum())
            / total_energy
        ),
        "common_radial_capture": float(
            common.square().sum() / total_energy
        ),
        "gauge_radial_capture": float(gauge.square().sum() / total_energy),
    }


def finite_mean(values: list[float]) -> float:
    selected = [value for value in values if math.isfinite(value)]
    return float(sum(selected) / len(selected)) if selected else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream-metadata", type=Path)
    args = parser.parse_args()

    snapshots = sorted(args.trajectory.glob("step_*.pt"))
    if len(snapshots) < 3:
        raise ValueError("trajectory needs at least three snapshots")
    first_payload = torch.load(
        snapshots[0], map_location="cpu", weights_only=False
    )
    initial = first_payload["parameters"]
    layers = sorted(
        {
            int(name.split(".")[2])
            for name in initial
            if name.endswith("mlp.c_fc.weight")
        }
    )
    if not layers:
        raise ValueError("trajectory contains no MLP c_fc weights")

    rows: list[dict[str, float | int]] = []
    paired_log_trajectories: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in layers
    }
    previous_log_changes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for snapshot in snapshots:
        payload = torch.load(snapshot, map_location="cpu", weights_only=False)
        parameters = payload["parameters"]
        step = int(payload["step"])
        for layer in layers:
            fc_key, proj_key = layer_keys(layer)
            initial_fc = initial[fc_key]
            initial_proj = initial[proj_key]
            current_fc = parameters[fc_key]
            current_proj = parameters[proj_key]
            log_fc = (
                current_fc.norm(dim=1).clamp_min(1e-30).log()
                - initial_fc.norm(dim=1).clamp_min(1e-30).log()
            )
            log_proj = (
                current_proj.norm(dim=0).clamp_min(1e-30).log()
                - initial_proj.norm(dim=0).clamp_min(1e-30).log()
            )
            paired_log_trajectories[layer].append(
                torch.cat((log_fc, log_proj))
            )
            common_log = (log_fc.double() + log_proj.double()) / math.sqrt(
                2.0
            )
            gauge_log = (log_fc.double() - log_proj.double()) / math.sqrt(
                2.0
            )
            total_log_energy = (
                log_fc.double().square().sum()
                + log_proj.double().square().sum()
            ).clamp_min(1e-30)
            previous = previous_log_changes.get(layer)
            increment_correlation = float("nan")
            if previous is not None:
                increment_correlation = pearson(
                    log_fc - previous[0], log_proj - previous[1]
                )
            previous_log_changes[layer] = (log_fc, log_proj)
            radial = normalized_radial_coordinates(
                initial_fc, initial_proj, current_fc, current_proj
            )
            rows.append(
                {
                    "step": step,
                    "layer": layer,
                    "signed_log_norm_correlation": pearson(
                        log_fc, log_proj
                    ),
                    "signed_log_norm_increment_correlation": (
                        increment_correlation
                    ),
                    "common_log_energy_fraction": float(
                        common_log.square().sum() / total_log_energy
                    ),
                    "gauge_log_energy_fraction": float(
                        gauge_log.square().sum() / total_log_energy
                    ),
                    **radial,
                }
            )
        del payload

    pca_rows = {
        str(layer): temporal_pca_energy(torch.stack(trajectory))
        for layer, trajectory in paired_log_trajectories.items()
    }
    nonzero_rows = [row for row in rows if int(row["step"]) > 0]
    terminal_step = max(int(row["step"]) for row in rows)
    terminal_rows = [
        row for row in rows if int(row["step"]) == terminal_step
    ]
    metric_names = (
        "signed_log_norm_correlation",
        "signed_log_norm_increment_correlation",
        "common_log_energy_fraction",
        "gauge_log_energy_fraction",
        "signed_radial_correlation",
        "independent_radial_capture",
        "common_radial_capture",
        "gauge_radial_capture",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "paired_hidden_channel_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": "paired_hidden_channel_trajectory_v1",
        "trajectory": str(args.trajectory),
        "snapshot_count": len(snapshots),
        "steps": [int(path.stem.split("_")[-1]) for path in snapshots],
        "layers": layers,
        "upstream_metadata": (
            {
                "path": str(args.upstream_metadata),
                "sha256": sha256(args.upstream_metadata),
            }
            if args.upstream_metadata
            else None
        ),
        "source_sha256": sha256(Path(__file__)),
        "trajectory_mean": {
            metric: finite_mean(
                [float(row[metric]) for row in nonzero_rows]
            )
            for metric in metric_names
        },
        "terminal_mean": {
            metric: finite_mean(
                [float(row[metric]) for row in terminal_rows]
            )
            for metric in metric_names
        },
        "paired_log_norm_temporal_pca": pca_rows,
        "paired_log_norm_temporal_pca_mean": {
            metric: finite_mean(
                [float(layer_metrics[metric]) for layer_metrics in pca_rows.values()]
            )
            for metric in (
                "pc1_energy",
                "pc1_pc2_energy",
                "dimension_90",
                "dimension_95",
                "dimension_99",
            )
        },
        "csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
    }
    metadata_path = args.output / "paired_hidden_channel_summary.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
