#!/usr/bin/env python3
"""Gauge-invariant audit of the high-cadence MLP gradient factor field."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_optimizer_probe_span import load_probe_inventory
from examples.nanogpt.analyze_mlp_product_fht_tangent_anchor import git_commit
from examples.nanogpt.analyze_mlp_raw_gradient_factor_transport import (
    exact_singular_factors,
)
from examples.nanogpt.analyze_mlp_raw_gradient_rolling_prediction import phase_for_step
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from latent_weight_lab.block_fht import normalized_fht_last_dim


def energy_dimensions(values: torch.Tensor) -> dict[str, float | int]:
    values = values.double().clamp_min(0)
    total = values.sum().clamp_min(1e-30)
    fractions = torch.cumsum(values, dim=0) / total
    result: dict[str, float | int] = {
        "participation_dimension": float(total.square() / values.square().sum().clamp_min(1e-30)),
    }
    for threshold in (0.90, 0.95, 0.99):
        index = int(torch.searchsorted(fractions, threshold).item())
        result[f"dimension_{int(threshold * 100)}"] = min(index + 1, values.numel())
    return result


def projector_kernel(frames: list[torch.Tensor]) -> torch.Tensor:
    count = len(frames)
    rank = frames[0].shape[1]
    kernel = torch.empty((count, count), dtype=torch.float64, device=frames[0].device)
    for row in range(count):
        kernel[row, row] = 1.0
        for column in range(row):
            value = (frames[row].transpose(0, 1) @ frames[column]).double().square().sum() / rank
            kernel[row, column] = value
            kernel[column, row] = value
    return kernel


def centered_kernel_spectrum(kernel: torch.Tensor) -> torch.Tensor:
    centered = kernel - kernel.mean(dim=0, keepdim=True)
    centered = centered - centered.mean(dim=1, keepdim=True)
    return torch.linalg.eigvalsh(centered).flip(0).clamp_min(0)


def union_spectrum(
    frames: list[torch.Tensor],
    singular_values: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    columns = []
    for index, frame in enumerate(frames):
        if singular_values is None:
            columns.append(frame)
        else:
            weight = singular_values[index][: frame.shape[1]].double()
            weight = weight / weight.square().sum().sqrt().clamp_min(1e-30)
            columns.append(frame * weight.to(frame).unsqueeze(0))
    values = torch.linalg.svdvals(torch.cat(columns, dim=1))
    return values.double().square()


def fit_union_basis(
    frames: list[torch.Tensor],
    singular_values: list[torch.Tensor],
    indices: range,
    rank: int,
) -> torch.Tensor:
    weighted = []
    for index in indices:
        values = singular_values[index][: frames[index].shape[1]]
        values = values / values.square().sum().sqrt().clamp_min(1e-30)
        weighted.append(frames[index] * values.unsqueeze(0))
    return torch.linalg.svd(torch.cat(weighted, dim=1), full_matrices=False).U[:, :rank]


def frame_capture(frame: torch.Tensor, basis: torch.Tensor) -> float:
    return float((basis.transpose(0, 1) @ frame).double().square().sum() / frame.shape[1])


def largest_power_of_two_divisor(value: int) -> int:
    if value <= 0:
        raise ValueError("dimension must be positive")
    return value & -value


def grouped_fht_frame(frame: torch.Tensor) -> torch.Tensor:
    dimension, rank = frame.shape
    group = largest_power_of_two_divisor(dimension)
    values = frame.transpose(0, 1).reshape(rank, dimension // group, group)
    return normalized_fht_last_dim(values).reshape(rank, dimension).transpose(0, 1)


def coordinate_energy(frame: torch.Tensor) -> torch.Tensor:
    return frame.double().square().sum(dim=1)


def support_capture(frame: torch.Tensor, support: torch.Tensor) -> float:
    energy = coordinate_energy(frame)
    return float(energy[support].sum() / energy.sum().clamp_min(1e-30))


def summarize_causal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["parameter"], row["side"], row["split"], row["union_rank"]) for row in rows})
    result = []
    for key in keys:
        members = [
            row
            for row in rows
            if (row["parameter"], row["side"], row["split"], row["union_rank"]) == key
        ]
        values = torch.tensor([row["current_frame_capture"] for row in members], dtype=torch.float64)
        result.append(
            {
                "parameter": key[0],
                "side": key[1],
                "split": key[2],
                "union_rank": key[3],
                "sample_count": len(members),
                "mean_capture": float(values.mean()),
                "minimum_capture": float(values.min()),
                "p10_capture": float(torch.quantile(values, 0.10)),
            }
        )
    return result


def summarize_support(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "parameter",
        "side",
        "domain",
        "support_fraction",
        "support_source",
        "split",
    )
    keys = sorted({tuple(row[field] for field in fields) for row in rows})
    result = []
    for key in keys:
        members = [row for row in rows if tuple(row[field] for field in fields) == key]
        values = torch.tensor([row["frame_capture"] for row in members], dtype=torch.float64)
        item = {field: value for field, value in zip(fields, key, strict=True)}
        item.update(
            sample_count=len(members),
            mean_capture=float(values.mean()),
            minimum_capture=float(values.min()),
            p10_capture=float(torch.quantile(values, 0.10)),
        )
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--targets", default="mlp.c_fc,mlp.c_proj")
    parser.add_argument("--factor-rank", type=int, default=6)
    parser.add_argument("--union-ranks", default="1,3,6,12,24,48")
    parser.add_argument("--history-probes", type=int, default=10)
    parser.add_argument("--support-fractions", default="0.001,0.01")
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    union_ranks = [int(value) for value in args.union_ranks.split(",")]
    support_fractions = [float(value) for value in args.support_fractions.split(",")]
    targets = {value for value in args.targets.split(",") if value}
    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, inventory, input_metadata = load_probe_inventory(
        paths, layers={args.layer}, targets=targets
    )
    discovery_indices = [index for index, step in enumerate(steps) if step < args.discovery_stop]

    spectrum_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for parameter, fields in sorted(inventory.items()):
        gradients = torch.stack(fields["raw_gradient_descent"]).to(args.device, torch.float32)
        left, singular, right = exact_singular_factors(gradients, args.factor_rank)
        matrix_size = gradients.shape[1] * gradients.shape[2]
        for side, frames in (("left_output", left), ("right_input", right)):
            dimension = frames[0].shape[0]
            kernel_values = centered_kernel_spectrum(projector_kernel(frames))
            for weighting, values in (
                ("unweighted", union_spectrum(frames)),
                ("singular_value_weighted", union_spectrum(frames, singular)),
            ):
                spectrum_rows.append(
                    {
                        "parameter": parameter,
                        "side": side,
                        "spectrum_kind": f"union_{weighting}",
                        "ambient_dimension": dimension,
                        "factor_rank": args.factor_rank,
                        "dictionary_fraction_at_dimension_90": 0.0,
                        **energy_dimensions(values),
                    }
                )
                spectrum_rows[-1]["dictionary_fraction_at_dimension_90"] = (
                    int(spectrum_rows[-1]["dimension_90"]) * dimension / matrix_size
                )
            spectrum_rows.append(
                {
                    "parameter": parameter,
                    "side": side,
                    "spectrum_kind": "centered_projector_kernel",
                    "ambient_dimension": dimension,
                    "factor_rank": args.factor_rank,
                    "dictionary_fraction_at_dimension_90": 0.0,
                    **energy_dimensions(kernel_values),
                }
            )

            for index in range(args.history_probes, len(steps)):
                for union_rank in union_ranks:
                    basis = fit_union_basis(
                        frames,
                        singular,
                        range(index - args.history_probes, index),
                        min(union_rank, args.history_probes * args.factor_rank),
                    )
                    causal_rows.append(
                        {
                            "parameter": parameter,
                            "side": side,
                            "probe_index": index,
                            "step": steps[index],
                            "split": phase_for_step(
                                steps[index], args.discovery_stop, args.validation_stop
                            ),
                            "union_rank": union_rank,
                            "dictionary_scalar_fraction": union_rank * dimension / matrix_size,
                            "current_frame_capture": frame_capture(frames[index], basis),
                        }
                    )

            transformed = {
                "native": frames,
                "grouped_fht": [grouped_fht_frame(frame) for frame in frames],
            }
            for domain, domain_frames in transformed.items():
                discovery_energy = sum(coordinate_energy(domain_frames[index]) for index in discovery_indices)
                for fraction in support_fractions:
                    count = max(1, math.floor(fraction * dimension))
                    discovery_support = torch.topk(discovery_energy, count, sorted=False).indices
                    for index, step in enumerate(steps):
                        split = phase_for_step(step, args.discovery_stop, args.validation_stop)
                        support_rows.append(
                            {
                                "parameter": parameter,
                                "side": side,
                                "domain": domain,
                                "support_fraction": fraction,
                                "support_source": "discovery",
                                "probe_index": index,
                                "step": step,
                                "split": split,
                                "frame_capture": support_capture(domain_frames[index], discovery_support),
                            }
                        )
                        if index >= args.history_probes:
                            history_energy = sum(
                                coordinate_energy(domain_frames[past])
                                for past in range(index - args.history_probes, index)
                            )
                            history_support = torch.topk(history_energy, count, sorted=False).indices
                            support_rows.append(
                                {
                                    "parameter": parameter,
                                    "side": side,
                                    "domain": domain,
                                    "support_fraction": fraction,
                                    "support_source": "preceding_10",
                                    "probe_index": index,
                                    "step": step,
                                    "split": split,
                                    "frame_capture": support_capture(domain_frames[index], history_support),
                                }
                            )
        del gradients, left, singular, right
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    causal_summary = summarize_causal(causal_rows)
    support_summary = summarize_support(support_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    spectra_path = args.output / "factor_spectra.csv"
    causal_path = args.output / "causal_union_capture.csv"
    causal_summary_path = args.output / "causal_union_summary.csv"
    support_path = args.output / "structured_support_capture.csv"
    support_summary_path = args.output / "structured_support_summary.csv"
    write_csv(spectra_path, spectrum_rows)
    write_csv(causal_path, causal_rows)
    write_csv(causal_summary_path, causal_summary)
    write_csv(support_path, support_rows)
    write_csv(support_summary_path, support_summary)
    metadata = {
        "schema_version": "nanogpt_mlp_gradient_factor_field_v1",
        "method": "gauge-invariant rank-six raw-gradient factor-field audit",
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "input": input_metadata,
        "steps": steps,
        "factor_rank": args.factor_rank,
        "history_probes": args.history_probes,
        "union_ranks": union_ranks,
        "support_fractions": support_fractions,
        "limitations": [
            "The audit describes dense-path raw gradients and does not update a language model.",
            "Structured support is gauge invariant within each singular frame but is not a functional metric.",
        ],
        "runtime_seconds": time.time() - started,
        "factor_spectra_sha256": file_sha256(spectra_path),
        "causal_union_capture_sha256": file_sha256(causal_path),
        "causal_union_summary_sha256": file_sha256(causal_summary_path),
        "structured_support_capture_sha256": file_sha256(support_path),
        "structured_support_summary_sha256": file_sha256(support_summary_path),
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"metadata_sha256={file_sha256(metadata_path)}")


if __name__ == "__main__":
    main()
