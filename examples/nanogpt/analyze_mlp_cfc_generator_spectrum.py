#!/usr/bin/env python3
"""Measure spectra and window stability of exact ``c_fc`` orbit generators.

The successful bilateral-orbit oracle implies a tangent
``Omega_out W + W Omega_in``.  This zero-update diagnostic recovers the
minimum-norm skew generators, measures their nonzero singular spectra, and
compares them across independent task-gradient windows.  It decides whether
a direct low-rank skew factorization is plausible before any new chart is
implemented or trained.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    _optimizer_and_group_for_parameter,
    build_candidates,
    exact_muon_update,
    file_sha256,
    fixed_batches,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_residual_structure import write_csv
from examples.nanogpt.analyze_mlp_cfc_trust_radius import (
    collect_gradient_window,
)


SCHEMA_VERSION = "nanogpt_mlp_cfc_generator_spectrum_v1"


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def orbit_generator_coordinates(
    weight: torch.Tensor,
    residual: torch.Tensor,
    *,
    geometry: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return a compact exact representation of the bilateral generators.

    For ``W = U S V^T`` and residual ``E``, the minimum-norm output-side
    generator is represented by ``(A, K)`` as

    ``Omega_out = U A U^T + K U^T - U K^T``, with ``U^T K = 0``.

    The input-side generator is ``Omega_in = V R V^T``.  Within each pair of
    distinct singular directions, ``A`` and ``R`` are the unique coupled
    solution.  This representation retains the exact nonzero spectrum
    without materializing a 3072-square matrix.
    """
    weight_f = weight.float()
    residual_f = residual.float()
    if weight_f.ndim != 2 or residual_f.shape != weight_f.shape:
        raise ValueError("weight and residual must be same-shaped matrices")
    if geometry is None:
        u, singular, vh = torch.linalg.svd(weight_f, full_matrices=False)
    else:
        u, singular, vh = geometry
        if (
            u.shape != (weight_f.shape[0], weight_f.shape[1])
            or singular.shape != (weight_f.shape[1],)
            or vh.shape != (weight_f.shape[1], weight_f.shape[1])
        ):
            raise ValueError("SVD geometry has incompatible shapes")
    v = vh.T
    coordinates = u.T @ residual_f @ v
    diagonal = torch.diagonal(coordinates)
    radial = u @ torch.diag(diagonal) @ vh
    perpendicular = residual_f - u @ coordinates @ vh

    squared_gap = (
        singular[None, :].square() - singular[:, None].square()
    )
    indices = torch.arange(singular.numel(), device=singular.device)
    off_diagonal = indices[:, None] != indices[None, :]
    if torch.any(squared_gap[off_diagonal] == 0):
        raise ValueError("exactly repeated singular values make the joint generator non-identifiable")
    left_numerator = (
        singular[None, :] * coordinates
        + singular[:, None] * coordinates.T
    )
    right_numerator = (
        singular[:, None] * coordinates
        + singular[None, :] * coordinates.T
    )
    left = torch.where(
        off_diagonal,
        left_numerator / squared_gap,
        torch.zeros_like(coordinates),
    )
    right = torch.where(
        off_diagonal,
        -right_numerator / squared_gap,
        torch.zeros_like(coordinates),
    )
    perpendicular_coordinates = (
        perpendicular @ v / singular.clamp_min(1e-30)[None, :]
    )
    left_update = (
        perpendicular
        + u @ (left * singular[None, :]) @ vh
    )
    right_update = u @ (singular[:, None] * right) @ vh
    bilateral = residual_f - radial
    reconstruction_error = (
        left_update + right_update - bilateral
    ).norm() / bilateral.norm().clamp_min(1e-30)
    return {
        "left_core": left,
        "left_perpendicular": perpendicular_coordinates,
        "right_core": right,
        "bilateral_update": bilateral,
        "radial_update": radial,
        "reconstruction_error": reconstruction_error,
        "minimum_relative_squared_singular_gap": (
            squared_gap[off_diagonal].abs().min()
            / singular.square().max().clamp_min(1e-30)
        ),
    }


def compressed_left_generator(
    left_core: torch.Tensor,
    left_perpendicular: torch.Tensor,
) -> torch.Tensor:
    """Materialize the exact nonzero output generator in a <=2n basis."""
    if (
        left_core.ndim != 2
        or left_core.shape[0] != left_core.shape[1]
        or left_perpendicular.ndim != 2
        or left_perpendicular.shape[1] != left_core.shape[0]
    ):
        raise ValueError("invalid compact left-generator shapes")
    _q, upper = torch.linalg.qr(left_perpendicular.float(), mode="reduced")
    width = left_core.shape[0]
    result = torch.zeros(
        2 * width,
        2 * width,
        device=left_core.device,
        dtype=torch.float32,
    )
    result[:width, :width] = left_core.float()
    result[:width, width:] = -upper.T
    result[width:, :width] = upper
    return result


def generator_inner(
    first_core: torch.Tensor,
    first_perpendicular: torch.Tensor,
    second_core: torch.Tensor,
    second_perpendicular: torch.Tensor,
) -> torch.Tensor:
    """Frobenius inner product without materializing output generators."""
    return (
        (first_core.float() * second_core.float()).sum()
        + 2.0
        * (
            first_perpendicular.float()
            * second_perpendicular.float()
        ).sum()
    )


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    numerator = (first.float() * second.float()).sum()
    denominator = first.float().norm() * second.float().norm()
    return float(numerator / denominator.clamp_min(1e-30))


def compact_generator_cosine(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    side: str,
) -> float:
    if side == "left":
        dot = generator_inner(
            reference["left_core"],
            reference["left_perpendicular"],
            candidate["left_core"],
            candidate["left_perpendicular"],
        )
        first_norm = generator_inner(
            reference["left_core"],
            reference["left_perpendicular"],
            reference["left_core"],
            reference["left_perpendicular"],
        ).sqrt()
        second_norm = generator_inner(
            candidate["left_core"],
            candidate["left_perpendicular"],
            candidate["left_core"],
            candidate["left_perpendicular"],
        ).sqrt()
    elif side == "right":
        first = reference["right_core"].float()
        second = candidate["right_core"].float()
        dot = (first * second).sum()
        first_norm = first.norm()
        second_norm = second.norm()
    else:
        raise ValueError("side must be left or right")
    return float(dot / (first_norm * second_norm).clamp_min(1e-30))


def spectrum_metrics(singular_values: torch.Tensor) -> dict[str, float | int]:
    values = singular_values.detach().double().sort(descending=True).values
    if values.ndim != 1 or not values.numel() or not torch.isfinite(values).all():
        raise ValueError("singular spectrum must be finite and nonempty")
    energy = values.square()
    total = energy.sum().clamp_min(1e-30)
    probabilities = energy / total
    cumulative = probabilities.cumsum(dim=0)

    def energy_rank(threshold: float) -> int:
        target = torch.tensor(
            threshold, device=cumulative.device, dtype=cumulative.dtype
        )
        return int(torch.searchsorted(cumulative, target).item() + 1)

    positive = probabilities > 0
    entropy = -(
        probabilities[positive] * probabilities[positive].log()
    ).sum()
    pair_count = values.numel() // 2
    pair_error = 0.0
    if pair_count:
        paired = values[: 2 * pair_count].reshape(pair_count, 2)
        pair_error = float(
            (paired[:, 0] - paired[:, 1]).abs().max()
            / values[0].clamp_min(1e-30)
        )
    return {
        "dimension": int(values.numel()),
        "top_singular_value": float(values[0]),
        "frobenius_norm": float(total.sqrt()),
        "stable_rank": float(total / values[0].square().clamp_min(1e-30)),
        "entropy_effective_rank": float(entropy.exp()),
        "top_pair_energy": float(probabilities[:2].sum()),
        "rank50": energy_rank(0.50),
        "rank80": energy_rank(0.80),
        "rank90": energy_rank(0.90),
        "rank95": energy_rank(0.95),
        "rank99": energy_rank(0.99),
        "paired_singular_max_relative_error": pair_error,
    }


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def aggregate(
    spectra: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    *,
    coordinate_budget: int,
    output_width: int,
    input_width: int,
    minimum_median_cosine: float,
    maximum_reconstruction_error: float,
) -> dict[str, Any]:
    by_side: dict[str, dict[str, float | int]] = {}
    for side in ("left", "right"):
        selected = [row for row in spectra if row["side"] == side]
        by_side[side] = {
            "median_stable_rank": median(
                [float(row["stable_rank"]) for row in selected]
            ),
            "median_entropy_effective_rank": median(
                [float(row["entropy_effective_rank"]) for row in selected]
            ),
            "minimum_rank90": min(int(row["rank90"]) for row in selected),
            "median_rank90": median(
                [float(row["rank90"]) for row in selected]
            ),
            "maximum_rank90": max(int(row["rank90"]) for row in selected),
            "maximum_pair_error": max(
                float(row["paired_singular_max_relative_error"])
                for row in selected
            ),
        }
    stability_summary: dict[str, dict[str, float]] = {}
    for side, key in (
        ("left", "left_generator_cosine"),
        ("right", "right_generator_cosine"),
        ("bilateral_update", "bilateral_update_cosine"),
    ):
        values = [float(row[key]) for row in stability]
        stability_summary[side] = {
            "minimum": min(values),
            "median": median(values),
            "mean": sum(values) / len(values),
        }
    left_rank_cap = 2 * math.floor(
        coordinate_budget / (2 * output_width)
    )
    right_rank_cap = 2 * math.floor(
        coordinate_budget / (2 * input_width)
    )
    low_rank_supported = all(
        (
            int(by_side["left"]["maximum_rank90"]) <= left_rank_cap,
            int(by_side["right"]["maximum_rank90"]) <= right_rank_cap,
        )
    )
    stable = all(
        stability_summary[side]["median"] >= minimum_median_cosine
        for side in ("left", "right")
    )
    reconstruction_stable = max(
        float(row["reconstruction_error"]) for row in spectra
    ) <= maximum_reconstruction_error
    if not reconstruction_stable:
        decision = "GENERATOR_RECONSTRUCTION_INVALID"
    elif low_rank_supported and stable:
        decision = "DIRECT_LOW_RANK_SKEW_FACTORIZATION_SUPPORTED"
    elif stable:
        decision = "DENSE_STABLE_GENERATOR_REQUIRES_CONJUGATED_BASIS"
    else:
        decision = "DENSE_WINDOW_DEPENDENT_GENERATOR_REQUIRES_ADAPTIVE_CONJUGATED_BASIS"
    return {
        "decision": decision,
        "parameter_updates": 0,
        "spectra": by_side,
        "stability": stability_summary,
        "direct_low_rank_all_budget_caps": {
            "coordinate_budget": coordinate_budget,
            "left_output_generator_rank_cap": left_rank_cap,
            "right_input_generator_rank_cap": right_rank_cap,
            "note": "caps give the entire budget to one A B^T - B A^T generator; any bilateral split is stricter",
        },
        "gates": {
            "generator_reconstruction_stable": reconstruction_stable,
            "direct_low_rank_supported": low_rank_supported,
            "cross_window_generator_stable": stable,
        },
        "thresholds": {
            "minimum_median_reference_cosine": minimum_median_cosine,
            "maximum_reconstruction_error": maximum_reconstruction_error,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-cache", type=Path)
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    identity = plan["identity"]
    for path, expected in (
        (args.checkpoint, identity["checkpoint_sha256"]),
        (args.config, identity["config_sha256"]),
        (args.data_dir / "manifest.json", identity["dataset_manifest_sha256"]),
    ):
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"identity mismatch for {path}: {observed}")
    protocol = plan["fixed_protocol"]
    rule = plan["decision_rule"]
    layers = [int(value) for value in protocol["layers"]]
    seeds = [int(value) for value in protocol["fit_train_seeds"]]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    geometries: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight.detach().float()
        geometries[layer] = torch.linalg.svd(weight, full_matrices=False)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    print(json.dumps({"phase_complete": "load_and_weight_svd"}), flush=True)

    reference: dict[int, dict[str, torch.Tensor]] = {}
    spectrum_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    fit_losses: dict[str, float] = {}
    for window_index, seed in enumerate(seeds):
        batches = fixed_batches(
            args.data_dir,
            "train",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["fit_batches"]),
            seed=seed,
        )
        fit_loss, gradients = collect_gradient_window(
            model,
            batches,
            layers,
            device=args.device,
            dtype=torch.bfloat16,
        )
        fit_losses[str(seed)] = fit_loss
        for layer in layers:
            weight = model.transformer.h[layer].mlp.c_fc.weight
            owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
            buffer = owner.state[weight].get("momentum_buffer")
            if buffer is None:
                raise RuntimeError(f"missing c_fc momentum at layer {layer}")
            dense_update, descent, _diagnostics = exact_muon_update(
                weight.detach(),
                gradients[layer].to(weight.device),
                buffer,
                learning_rate=float(group["lr"]),
                momentum=float(group["momentum"]),
                weight_decay=float(group["weight_decay"]),
                ns_steps=int(group["ns_steps"]),
            )
            polar_descent = (
                descent
                + float(group["weight_decay"]) * weight.detach().float()
            )
            matched, _selections = build_candidates(
                weight.detach(),
                dense_update,
                polar_descent,
                parent_stages=int(protocol["parent_stages"]),
                residual_stages=int(protocol["residual_stages"]),
                neighbors=int(protocol["matching_neighbors"]),
                seed=int(protocol["matching_seed"]) + layer * 1009,
                learning_rate=float(group["lr"]),
                weight_decay=float(group["weight_decay"]),
                native_cache=args.native_cache,
            )
            residual = dense_update.float() - matched["fresh_expansion88"].float()
            compact = orbit_generator_coordinates(
                weight.detach(), residual, geometry=geometries[layer]
            )
            if window_index == 0:
                left_singular = torch.linalg.svdvals(
                    compressed_left_generator(
                        compact["left_core"],
                        compact["left_perpendicular"],
                    )
                )
                right_singular = torch.linalg.svdvals(compact["right_core"])
                reconstruction_error = float(compact["reconstruction_error"])
                minimum_gap = float(
                    compact["minimum_relative_squared_singular_gap"]
                )
                for side, values in (
                    ("left", left_singular),
                    ("right", right_singular),
                ):
                    spectrum_rows.append(
                        {
                            "layer": layer,
                            "side": side,
                            "fit_seed": seed,
                            "reconstruction_error": reconstruction_error,
                            "minimum_relative_squared_singular_gap": minimum_gap,
                            **spectrum_metrics(values),
                        }
                    )
                reference[layer] = {
                    key: value.detach().cpu()
                    for key, value in compact.items()
                    if key not in {
                        "reconstruction_error",
                        "minimum_relative_squared_singular_gap",
                    }
                }
            else:
                ref = {
                    key: value.to(weight.device)
                    for key, value in reference[layer].items()
                }
                stability_rows.append(
                    {
                        "reference_seed": seeds[0],
                        "candidate_seed": seed,
                        "layer": layer,
                        "left_generator_cosine": compact_generator_cosine(
                            ref, compact, "left"
                        ),
                        "right_generator_cosine": compact_generator_cosine(
                            ref, compact, "right"
                        ),
                        "bilateral_update_cosine": cosine(
                            ref["bilateral_update"],
                            compact["bilateral_update"],
                        ),
                        "candidate_reconstruction_error": float(
                            compact["reconstruction_error"]
                        ),
                    }
                )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "phase_complete": "fit_window",
                    "window": window_index + 1,
                    "windows_total": len(seeds),
                    "seed": seed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = aggregate(
        spectrum_rows,
        stability_rows,
        coordinate_budget=int(protocol["equal_coordinates_per_layer"]),
        output_width=int(protocol["output_width"]),
        input_width=int(protocol["input_width"]),
        minimum_median_cosine=float(rule["minimum_median_reference_cosine"]),
        maximum_reconstruction_error=float(rule["maximum_reconstruction_error"]),
    )
    result["fit_losses_bfloat16"] = fit_losses
    args.output.mkdir(parents=True, exist_ok=True)
    spectra_path = args.output / "cfc_generator_spectra.csv"
    stability_path = args.output / "cfc_generator_stability.csv"
    aggregate_path = args.output / "cfc_generator_spectrum_aggregate.json"
    write_csv(spectra_path, spectrum_rows)
    write_csv(stability_path, stability_rows)
    aggregate_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": result["decision"],
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "plan_sha256": file_sha256(args.plan),
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "protocol": protocol,
        "outputs": {
            "spectra_sha256": file_sha256(spectra_path),
            "stability_sha256": file_sha256(stability_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_generator_spectrum_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "aggregate": str(aggregate_path),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
