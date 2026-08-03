#!/usr/bin/env python3
"""Screen task-selected directed sparse output mixers for ``mlp.c_fc``.

Unlike Givens or symmetric-shear charts, a directed mixer fits each target
expansion channel from independently selected source channels.  This removes
reciprocal coefficient tying and degree-regular perfect matchings while
retaining a sparse, task-conditioned, non-learned basis.  Every candidate is
normalized to the deployed c_fc BF16 endpoint radius before held-out scoring.
No checkpoint state is changed or written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    file_sha256,
    fixed_batches,
    git_commit,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_dense_oracle_gap import (
    ExactVariantApplier,
    aggregate_direction_metrics,
    evaluate_candidates,
    family_fro,
    merge_updates,
    scale_family,
)
from examples.nanogpt.analyze_mlp_fixed_radius_capacity import (
    extract_reconstructed_capacity_updates,
    normalize_family_to_radius,
    quantized_update,
)
from examples.nanogpt.analyze_mlp_joint_step_response_surface import (
    paired_comparison,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nanogpt_mlp_cfc_directed_sparse_v1"


@torch.no_grad()
def fit_directed_sparse_mixer(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    incoming: int,
    ridge_ratio: float,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit ``target[:, j]`` from selected columns of ``source`` per ``j``."""
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must be same-shaped matrices")
    rows, width = source.shape
    if not 0 < incoming <= rows or incoming > width:
        raise ValueError("incoming count must fit both source rank and width")
    if not 0.0 < ridge_ratio < 1.0 or chunk_size <= 0:
        raise ValueError("invalid directed sparse solver settings")
    source_f = source.float()
    target_f = target.float()
    # Sparsify the exact minimum-norm full output action, rather than using
    # one-coordinate marginal scores.  This accounts for correlations among
    # the current weight columns before the selected support is refit.
    row_gram = source_f @ source_f.T
    row_scale = row_gram.diagonal().mean()
    row_gram.add_(
        torch.eye(rows, device=source.device, dtype=torch.float32)
        * (float(ridge_ratio) * row_scale)
    )
    minimum_norm_action = source_f.T @ torch.linalg.solve(row_gram, target_f)
    indices = torch.topk(
        minimum_norm_action.abs(), k=int(incoming), dim=0
    ).indices
    del row_gram, minimum_norm_action
    prediction = torch.empty_like(target_f)
    coefficients = torch.empty(
        incoming, width, device=source.device, dtype=torch.float32
    )
    for start in range(0, width, int(chunk_size)):
        stop = min(start + int(chunk_size), width)
        selected = indices[:, start:stop]
        # Advanced indexing gives [rows, incoming, columns]; solve one ridge
        # system per target column without materializing a full dense mixer.
        dictionary = source_f[:, selected].permute(2, 0, 1).contiguous()
        targets = target_f[:, start:stop].T.contiguous().unsqueeze(-1)
        gram = dictionary.transpose(1, 2) @ dictionary
        rhs = dictionary.transpose(1, 2) @ targets
        diagonal_mean = gram.diagonal(dim1=1, dim2=2).mean(dim=1)
        eye = torch.eye(incoming, device=source.device, dtype=torch.float32)
        gram.add_(eye.unsqueeze(0) * (float(ridge_ratio) * diagonal_mean)[:, None, None])
        solved = torch.linalg.solve(gram, rhs).squeeze(-1)
        coefficients[:, start:stop] = solved.T
        prediction[:, start:stop] = (
            dictionary @ solved.unsqueeze(-1)
        ).squeeze(-1).T
    residual = target_f - prediction
    target_energy = target_f.square().sum().clamp_min(1e-30)
    degrees = torch.bincount(indices.reshape(-1), minlength=width).float()
    self_edges = (
        indices
        == torch.arange(width, device=indices.device).unsqueeze(0)
    ).sum()
    return prediction, {
        "rows": int(rows),
        "width": int(width),
        "incoming_per_target": int(incoming),
        "coordinates": int(incoming * width),
        "ridge_ratio": float(ridge_ratio),
        "chunk_size": int(chunk_size),
        "target_recovery": float(1.0 - residual.square().sum() / target_energy),
        "target_cosine": float(
            (prediction * target_f).sum()
            / (prediction.norm() * target_f.norm()).clamp_min(1e-30)
        ),
        "coefficient_rms": float(coefficients.square().mean().sqrt()),
        "coefficient_max_abs": float(coefficients.abs().max()),
        "source_degree_mean": float(degrees.mean()),
        "source_degree_std": float(degrees.std(unbiased=False)),
        "source_degree_max": int(degrees.max()),
        "self_edge_fraction": float(self_edges / indices.numel()),
    }


def candidate_order(levels: list[int]) -> list[str]:
    names = [
        "baseline",
        "production_cfc",
        "production_cproj",
        "production_joint",
        "dense_norm_cfc",
        "hybrid_norm_cfc",
    ]
    for level in levels:
        names.extend((f"directed{level}_cfc", f"hybrid_directed{level}_cfc"))
    return names


def classify(
    rows: list[dict[str, Any]],
    levels: list[int],
    *,
    confidence_z: float,
    minimum_fraction: float,
    mean_fraction: float,
) -> dict[str, Any]:
    names = candidate_order(levels)
    means = {
        point: sum(float(row["ce"]) for row in rows if row["point_id"] == point)
        / sum(1 for row in rows if row["point_id"] == point)
        for point in names
    }
    comparisons = {
        "dense_single": paired_comparison(
            rows, "dense_norm_cfc", "production_cfc", confidence_z
        ),
        "dense_hybrid": paired_comparison(
            rows, "hybrid_norm_cfc", "production_joint", confidence_z
        ),
    }
    oracle_valid = all(
        comparisons[name]["candidate_reliably_better"]
        for name in ("dense_single", "dense_hybrid")
    )
    results = []
    for level in levels:
        single_id = f"directed{level}_cfc"
        hybrid_id = f"hybrid_directed{level}_cfc"
        comparisons[f"directed{level}_single"] = paired_comparison(
            rows, single_id, "production_cfc", confidence_z
        )
        comparisons[f"directed{level}_hybrid"] = paired_comparison(
            rows, hybrid_id, "production_joint", confidence_z
        )
        single_gap = means["production_cfc"] - means["dense_norm_cfc"]
        hybrid_gap = means["production_joint"] - means["hybrid_norm_cfc"]
        fractions = {
            "single": (
                (means["production_cfc"] - means[single_id]) / single_gap
                if single_gap > 0.0
                else math.nan
            ),
            "hybrid": (
                (means["production_joint"] - means[hybrid_id]) / hybrid_gap
                if hybrid_gap > 0.0
                else math.nan
            ),
        }
        finite = all(math.isfinite(value) for value in fractions.values())
        reliable = all(
            comparisons[f"directed{level}_{kind}"]["candidate_reliably_better"]
            for kind in ("single", "hybrid")
        )
        fraction_pass = finite and (
            min(fractions.values()) >= float(minimum_fraction)
            and sum(fractions.values()) / 2.0 >= float(mean_fraction)
        )
        results.append(
            {
                "incoming_per_target": int(level),
                "reliable_singleton_and_hybrid": reliable,
                "oracle_gap_fraction_recovered": {
                    key: value if math.isfinite(value) else None
                    for key, value in fractions.items()
                },
                "fraction_pass": fraction_pass,
                "passes": oracle_valid and reliable and fraction_pass,
            }
        )
    selected = next((row for row in results if row["passes"]), None)
    if not oracle_valid:
        classification = "HELDOUT_DENSE_CFC_ORACLE_NOT_STABLE"
        next_action = "DO_NOT_TRAIN_RESELECT_DISCRIMINATING_WINDOWS"
    elif selected is not None:
        classification = "DIRECTED_SPARSE_CFC_PASSES"
        next_action = "IMPLEMENT_SMALLEST_PASSING_LEVEL_AND_PREFLIGHT_ONLY"
    elif any(row["reliable_singleton_and_hybrid"] for row in results):
        classification = "DIRECTED_SPARSE_CFC_GAIN_TOO_SMALL"
        next_action = "DO_NOT_TRAIN_REASSESS_NONLINEAR_REACHABILITY"
    else:
        classification = "DIRECTED_SPARSE_CFC_REJECTED"
        next_action = "DO_NOT_TRAIN_REASSESS_NONLINEAR_REACHABILITY"
    return {
        "classification": classification,
        "selected_incoming_per_target": (
            None if selected is None else selected["incoming_per_target"]
        ),
        "next_action": next_action,
        "candidate_means": means,
        "comparisons": comparisons,
        "levels": results,
        "dense_cfc_oracle_valid": oracle_valid,
    }


def validate_plan(path: Path, checkpoint: Path, config: Path, data_dir: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
    }
    for key, value in actual.items():
        if value != plan["identity"][key]:
            raise ValueError(f"registered identity mismatch: {key}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = validate_plan(args.plan, args.checkpoint, args.config, args.data_dir)
    protocol = plan["protocol"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(config["dtype"])]
    train_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(config["batch_size"]),
        block_size=int(config["block_size"]) + 1,
        batches=int(protocol["gradient_accumulation_steps"]),
        seed=int(protocol["train_seed"]),
    )
    bases, dense_historical, reconstructed = extract_reconstructed_capacity_updates(
        args.checkpoint,
        config,
        train_batches,
        [int(protocol["production_residual_stages"])],
        device=args.device,
        dtype=dtype,
    )
    prod_cfc = reconstructed["production"]["c_fc"]
    prod_cproj = reconstructed["production"]["c_proj"]
    control_level = str(protocol["production_residual_stages"])
    control_cfc = {
        layer: quantized_update(bases["c_fc"][layer], update)
        for layer, update in reconstructed["raw"][control_level]["c_fc"].items()
    }
    reconstruction_error = max(
        float((prod_cfc[layer] - control_cfc[layer]).abs().max())
        for layer in prod_cfc
    )
    if reconstruction_error > float(protocol["control_max_abs_tolerance"]):
        raise RuntimeError(f"production c_fc reconstruction failed: {reconstruction_error}")
    levels = [int(value) for value in protocol["incoming_per_target"]]
    raw_candidates: dict[str, dict[int, torch.Tensor]] = {
        str(level): {} for level in levels
    }
    fit_rows: dict[str, dict[int, Any]] = {str(level): {} for level in levels}
    for layer in sorted(bases["c_fc"]):
        source = bases["c_fc"][layer].to(args.device).float().T.contiguous()
        target = dense_historical["c_fc"][layer].to(args.device).float().T.contiguous()
        for level in levels:
            predicted, row = fit_directed_sparse_mixer(
                source,
                target,
                incoming=level,
                ridge_ratio=float(protocol["ridge_ratio"]),
                chunk_size=int(protocol["solver_chunk_size"]),
            )
            raw_candidates[str(level)][layer] = predicted.T.contiguous().cpu()
            fit_rows[str(level)][layer] = row
        print(json.dumps({"directed_layer_complete": layer}), flush=True)
    cfc_radius = family_fro(prod_cfc)
    normalized = {}
    normalization = {}
    for level in levels:
        candidate, row = normalize_family_to_radius(
            bases["c_fc"], raw_candidates[str(level)], cfc_radius
        )
        if row["relative_radius_error"] > float(protocol["maximum_relative_radius_error"]):
            raise RuntimeError(f"directed{level} radius normalization failed")
        normalized[str(level)] = candidate
        normalization[str(level)] = row
    norm_dense_cfc = scale_family(
        dense_historical["c_fc"], cfc_radius / family_fro(dense_historical["c_fc"])
    )
    candidates: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        "baseline": {},
        "production_cfc": {"c_fc": prod_cfc},
        "production_cproj": {"c_proj": prod_cproj},
        "production_joint": merge_updates(prod_cfc, prod_cproj),
        "dense_norm_cfc": {"c_fc": norm_dense_cfc},
        "hybrid_norm_cfc": merge_updates(norm_dense_cfc, prod_cproj),
    }
    for level in levels:
        directed = normalized[str(level)]
        candidates[f"directed{level}_cfc"] = {"c_fc": directed}
        candidates[f"hybrid_directed{level}_cfc"] = merge_updates(directed, prod_cproj)
    if list(candidates) != candidate_order(levels):
        raise RuntimeError("candidate order differs from registration")
    model, _optimizer, checkpoint_payload = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    applier = ExactVariantApplier(model)
    windows = {
        f"window_{index + 1}": fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["evaluation_batch_size"]),
            block_size=int(protocol["evaluation_block_size"]) + 1,
            batches=int(protocol["validation_batches_per_window"]),
            seed=int(seed),
        )
        for index, seed in enumerate(protocol["validation_seeds"])
    }
    ce_rows = evaluate_candidates(
        model, applier, windows, candidates, device=args.device, dtype=dtype
    )
    rule = plan["decision_rule"]
    decision = classify(
        ce_rows,
        levels,
        confidence_z=float(rule["confidence_z"]),
        minimum_fraction=float(rule["minimum_oracle_gap_fraction"]),
        mean_fraction=float(rule["mean_oracle_gap_fraction"]),
    )
    direction_recovery = {
        str(level): aggregate_direction_metrics(
            norm_dense_cfc, normalized[str(level)]
        )
        for level in levels
    }
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        "ce": args.output / "heldout_ce.json",
        "fits": args.output / "directed_sparse_fits.json",
        "replay": args.output / "prospective_step_metadata.json",
    }
    paths["ce"].write_text(json.dumps(ce_rows, indent=2, sort_keys=True) + "\n")
    paths["fits"].write_text(json.dumps(fit_rows, indent=2, sort_keys=True) + "\n")
    replay = {
        "checkpoint_next_iter": int(checkpoint_payload["next_iter"]),
        "production_cfc_reconstruction_max_abs_error": reconstruction_error,
        "normalization": normalization,
        "direction_recovery_against_norm_dense": direction_recovery,
        "gradient_replay": reconstructed["metadata"],
    }
    paths["replay"].write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "parameter_updates_to_checkpoint": 0,
        "disposable_optimizer_steps": 2,
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
            "plan_sha256": file_sha256(args.plan),
        },
        "replay": replay,
        "outputs": {f"{name}_sha256": file_sha256(path) for name, path in paths.items()},
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
