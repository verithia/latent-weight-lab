#!/usr/bin/env python3
"""Screen equal-coordinate FHT-conjugated general output charts for c_fc."""

from __future__ import annotations

import argparse
import json
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
    _weight_decay_after_rotation,
    build_candidates,
    exact_muon_update,
    file_sha256,
    fixed_batches,
    load_model_and_optimizer,
)
from examples.nanogpt.analyze_mlp_cfc_residual_structure import (
    residual_metrics,
    validate_identity,
    write_csv,
)
from examples.nanogpt.analyze_mlp_cfc_trust_radius import (
    collect_gradient_window,
    repeated_losses,
    summarize,
)
from examples.nanogpt.model import normalized_fht_last_dim


SCHEMA_VERSION = "nanogpt_mlp_cfc_conjugated_general_v1"
CONTROL = "fresh88"
DENSE = "dense_exact"
ORTHOGONAL = "conjugated_orthogonal88"
GENERAL = ("conjugated_general22_seed0", "conjugated_general22_seed1")
CANDIDATES = (CONTROL, DENSE, ORTHOGONAL, *GENERAL)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def make_bases(
    features: int,
    stages: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features <= 0 or features % 2 or stages <= 0:
        raise ValueError("features must be positive/even and stages positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutations = torch.stack(
        [torch.randperm(features, generator=generator) for _ in range(stages)]
    )
    signs = torch.randint(
        0,
        2,
        (stages, features),
        generator=generator,
        dtype=torch.int64,
    ).float().mul_(2.0).sub_(1.0)
    return permutations, torch.argsort(permutations, dim=1), signs


def basis_rows(
    values: torch.Tensor,
    permutation: torch.Tensor,
    inverse_permutation: torch.Tensor,
    signs: torch.Tensor,
    *,
    block_size: int,
    inverse: bool,
) -> torch.Tensor:
    """Apply the fixed orthogonal basis along matrix rows."""
    if values.ndim != 2:
        raise ValueError("values must be a matrix")
    features = values.shape[0]
    if (
        block_size <= 0
        or block_size & (block_size - 1)
        or features % block_size
    ):
        raise ValueError("block_size must be a power of two dividing rows")
    transposed = values.T.contiguous()
    signs = signs.to(device=values.device, dtype=values.dtype)
    if inverse:
        transposed = transposed * signs
        grouped = transposed.reshape(
            values.shape[1], features // block_size, block_size
        )
        transposed = normalized_fht_last_dim(grouped).reshape_as(transposed)
        transposed = transposed.index_select(
            -1, inverse_permutation.to(values.device)
        )
    else:
        transposed = transposed.index_select(
            -1, permutation.to(values.device)
        )
        grouped = transposed.reshape(
            values.shape[1], features // block_size, block_size
        )
        transposed = normalized_fht_last_dim(grouped).reshape_as(transposed)
        transposed = transposed * signs
    return transposed.T.contiguous()


@torch.no_grad()
def fit_conjugated_chart(
    weight: torch.Tensor,
    target_rotation_update: torch.Tensor,
    *,
    stages: int,
    seed: int,
    block_size: int,
    family: str,
    damping: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Causally fit fixed FHT-conjugated 2x2 output blocks."""
    if family not in {"orthogonal", "general"}:
        raise ValueError("family must be orthogonal or general")
    source = weight.float().clone()
    target = target_rotation_update.float()
    accumulated = torch.zeros_like(source)
    permutations, inverses, signs = make_bases(source.shape[0], stages, seed)
    coordinate_rms: list[float] = []
    coordinate_max: list[float] = []
    recovery: list[float] = []
    for stage in range(int(stages)):
        residual = target - accumulated
        source_basis = basis_rows(
            source,
            permutations[stage],
            inverses[stage],
            signs[stage],
            block_size=block_size,
            inverse=False,
        )
        residual_basis = basis_rows(
            residual,
            permutations[stage],
            inverses[stage],
            signs[stage],
            block_size=block_size,
            inverse=False,
        )
        source_pairs = source_basis.reshape(source.shape[0] // 2, 2, -1)
        residual_pairs = residual_basis.reshape(source.shape[0] // 2, 2, -1)
        if family == "orthogonal":
            direction = torch.stack(
                (-source_pairs[:, 1], source_pairs[:, 0]), dim=1
            )
            denominator = direction.square().sum(dim=(1, 2)).clamp_min(1e-30)
            coordinates = (
                residual_pairs * direction
            ).sum(dim=(1, 2)) / denominator
            contribution_pairs = coordinates[:, None, None] * direction
            coordinate_rms.append(float(coordinates.square().mean().sqrt()))
            coordinate_max.append(float(coordinates.abs().max()))
        else:
            gram = source_pairs @ source_pairs.transpose(-1, -2)
            ridge = (
                gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                * (0.5 * float(damping))
            )
            identity = torch.eye(2, device=source.device)[None]
            gram = gram + ridge[:, None, None] * identity
            rhs = residual_pairs @ source_pairs.transpose(-1, -2)
            coordinates = torch.linalg.solve(
                gram, rhs.transpose(-1, -2)
            ).transpose(-1, -2)
            contribution_pairs = coordinates @ source_pairs
            coordinate_rms.append(float(coordinates.square().mean().sqrt()))
            coordinate_max.append(float(coordinates.abs().max()))
        contribution_basis = contribution_pairs.reshape_as(source_basis)
        contribution = basis_rows(
            contribution_basis,
            permutations[stage],
            inverses[stage],
            signs[stage],
            block_size=block_size,
            inverse=True,
        )
        requested_energy = residual.square().sum().clamp_min(1e-30)
        recovery.append(
            float(
                1.0
                - (residual - contribution).square().sum()
                / requested_energy
            )
        )
        accumulated.add_(contribution)
        source.add_(contribution)
    return accumulated, {
        "family": family,
        "stages": int(stages),
        "seed": int(seed),
        "block_size": int(block_size),
        "coordinates": int(
            stages
            * (source.shape[0] // 2)
            * (1 if family == "orthogonal" else 4)
        ),
        "coordinate_rms_mean": sum(coordinate_rms) / len(coordinate_rms),
        "coordinate_max_abs": max(coordinate_max),
        "mean_stage_requested_recovery": sum(recovery) / len(recovery),
        "final_rotation_recovery": float(
            1.0
            - (target - accumulated).square().sum()
            / target.square().sum().clamp_min(1e-30)
        ),
    }


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def aggregate(
    loss_rows: list[dict[str, Any]],
    *,
    windows: list[str],
    maximum_replicate_range: float,
    minimum_recovery: float,
    median_recovery: float,
) -> dict[str, Any]:
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for window in windows:
        summaries[window] = {}
        for candidate in ("baseline", *CANDIDATES):
            summaries[window][candidate] = summarize(
                [
                    float(row["loss"])
                    for row in loss_rows
                    if row["window"] == window
                    and row["candidate"] == candidate
                ]
            )
    stable = all(
        values["range"] <= maximum_replicate_range
        for window in summaries.values()
        for values in window.values()
    )
    dense_positive = all(
        values[DENSE]["maximum"] < values[CONTROL]["minimum"]
        for values in summaries.values()
    )
    results: dict[str, dict[str, Any]] = {}
    for candidate in (ORTHOGONAL, *GENERAL):
        recoveries = {}
        for window, values in summaries.items():
            gap = values[CONTROL]["mean"] - values[DENSE]["mean"]
            recoveries[window] = (
                values[CONTROL]["mean"] - values[candidate]["mean"]
            ) / max(gap, 1e-30)
        minimum = min(recoveries.values())
        med = median(list(recoveries.values()))
        results[candidate] = {
            "recovery_by_window": recoveries,
            "minimum_recovery": minimum,
            "median_recovery": med,
            "beats_fresh_every_window": all(
                values[candidate]["maximum"] < values[CONTROL]["minimum"]
                for values in summaries.values()
            ),
            "beats_orthogonal_every_window": all(
                values[candidate]["maximum"] < values[ORTHOGONAL]["minimum"]
                for values in summaries.values()
            ) if candidate in GENERAL else False,
            "sufficient": all(
                (
                    stable,
                    dense_positive,
                    minimum >= minimum_recovery,
                    med >= median_recovery,
                )
            ),
        }
    orthogonal_sufficient = results[ORTHOGONAL]["sufficient"]
    general_reproducible = all(
        results[candidate]["sufficient"]
        and results[candidate]["beats_orthogonal_every_window"]
        for candidate in GENERAL
    )
    if not dense_positive:
        decision = "DENSE_RESIDUAL_NOT_POSITIVE_CONTROL"
    elif not stable:
        decision = "NUMERICAL_REPLICATE_GATE_FAILED"
    elif orthogonal_sufficient:
        decision = "DENSE_CONNECTIVITY_SUFFICIENT_ORTHOGONAL"
    elif general_reproducible:
        decision = "PROMOTE_FHT_CONJUGATED_GENERAL2X2"
    elif any(results[candidate]["sufficient"] for candidate in GENERAL):
        decision = "GENERAL_BASIS_SEED_UNSTABLE"
    else:
        decision = "REJECT_EQUAL_COORDINATE_CONJUGATED_CHARTS"
    return {
        "decision": decision,
        "parameter_updates": 0,
        "summaries": summaries,
        "candidate_results": results,
        "gates": {
            "numerically_stable": stable,
            "dense_beats_fresh_every_window": dense_positive,
            "orthogonal_sufficient": orthogonal_sufficient,
            "general_reproducible": general_reproducible,
        },
        "thresholds": {
            "maximum_replicate_range": maximum_replicate_range,
            "minimum_recovery": minimum_recovery,
            "median_recovery": median_recovery,
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
    plan = validate_identity(args.checkpoint, args.config, args.data_dir, args.plan)
    protocol = plan["fixed_protocol"]
    rule = plan["decision_rule"]
    layers = [int(value) for value in protocol["layers"]]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fit_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["batch_size"]),
        block_size=int(protocol["block_size"]),
        batches=int(protocol["fit_batches"]),
        seed=int(protocol["fit_train_seed"]),
    )
    windows = [
        f"validation_{index + 1}"
        for index in range(len(protocol["validation_seeds"]))
    ]
    validation_batches = {
        window: fixed_batches(
            args.data_dir,
            "val",
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["validation_batches_per_window"]),
            seed=int(seed),
        )
        for window, seed in zip(
            windows, protocol["validation_seeds"], strict=True
        )
    }
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    fit_loss, gradients = collect_gradient_window(
        model, fit_batches, layers, device=args.device, dtype=torch.bfloat16
    )
    updates: dict[str, dict[int, torch.Tensor]] = {
        candidate: {} for candidate in CANDIDATES
    }
    metric_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
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
            descent + float(group["weight_decay"]) * weight.detach().float()
        )
        matched, _selections = build_candidates(
            weight.detach(),
            dense_update,
            polar_descent,
            parent_stages=int(protocol["control_parent_stages"]),
            residual_stages=int(protocol["control_residual_stages"]),
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        fresh = matched["fresh_expansion88"].float()
        updates[CONTROL][layer] = fresh.cpu()
        updates[DENSE][layer] = dense_update.float().cpu()
        target_rotation = dense_update.float() + (
            float(group["lr"])
            * float(group["weight_decay"])
            * weight.detach().float()
        )
        fitted = {
            ORTHOGONAL: fit_conjugated_chart(
                weight.detach(),
                target_rotation,
                stages=int(protocol["orthogonal_stages"]),
                seed=int(protocol["orthogonal_seed"]) + layer * 1009,
                block_size=int(protocol["basis_block_size"]),
                family="orthogonal",
                damping=float(protocol["general_damping"]),
            ),
            GENERAL[0]: fit_conjugated_chart(
                weight.detach(),
                target_rotation,
                stages=int(protocol["general_stages"]),
                seed=int(protocol["general_seeds"][0]) + layer * 1009,
                block_size=int(protocol["basis_block_size"]),
                family="general",
                damping=float(protocol["general_damping"]),
            ),
            GENERAL[1]: fit_conjugated_chart(
                weight.detach(),
                target_rotation,
                stages=int(protocol["general_stages"]),
                seed=int(protocol["general_seeds"][1]) + layer * 1009,
                block_size=int(protocol["basis_block_size"]),
                family="general",
                damping=float(protocol["general_damping"]),
            ),
        }
        for candidate, (rotation, diagnostics) in fitted.items():
            update = _weight_decay_after_rotation(
                weight.detach(),
                rotation,
                learning_rate=float(group["lr"]),
                weight_decay=float(group["weight_decay"]),
            )
            updates[candidate][layer] = update.cpu()
            metric_rows.append(
                {
                    "layer": layer,
                    "candidate": candidate,
                    **residual_metrics(dense_update.float(), update),
                }
            )
            fit_rows.append({"layer": layer, "candidate": candidate, **diagnostics})
        print(
            json.dumps(
                {"layer_complete": layer, "layers_total": len(layers)},
                sort_keys=True,
            ),
            flush=True,
        )

    repeats = int(protocol["evaluation_repeats"])
    loss_rows: list[dict[str, Any]] = []
    for window, batches in validation_batches.items():
        for candidate, candidate_updates in (
            ("baseline", None),
            *((name, updates[name]) for name in CANDIDATES),
        ):
            losses = repeated_losses(
                model,
                batches,
                candidate_updates,
                repeats=repeats,
                device=args.device,
                dtype=torch.float32,
            )
            for repeat, loss in enumerate(losses):
                loss_rows.append(
                    {
                        "window": window,
                        "candidate": candidate,
                        "repeat": repeat,
                        "loss": loss,
                    }
                )
        print(
            json.dumps(
                {"phase_complete": "validation_window", "window": window},
                sort_keys=True,
            ),
            flush=True,
        )
    result = aggregate(
        loss_rows,
        windows=windows,
        maximum_replicate_range=float(rule["maximum_replicate_range"]),
        minimum_recovery=float(rule["minimum_recovery"]),
        median_recovery=float(rule["median_recovery"]),
    )
    result["fit_gradient_loss_bfloat16"] = fit_loss
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "losses": args.output / "cfc_conjugated_general_losses.csv",
        "metrics": args.output / "cfc_conjugated_general_metrics.csv",
        "fits": args.output / "cfc_conjugated_general_fits.csv",
        "aggregate": args.output / "cfc_conjugated_general_aggregate.json",
    }
    write_csv(paths["losses"], loss_rows)
    write_csv(paths["metrics"], metric_rows)
    write_csv(paths["fits"], fit_rows)
    paths["aggregate"].write_text(
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
            f"{name}_sha256": file_sha256(path)
            for name, path in paths.items()
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_conjugated_general_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "aggregate": str(paths["aggregate"]),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
