#!/usr/bin/env python3
"""Select and hold out a scalar trust radius for the exact-current c_fc chart.

The hidden-88 checkpoint and all non-c_fc weights remain fixed.  The existing
fresh 64+24-stage expansion-side matcher is reconstructed from the registered
fit window.  Only a scalar multiplier on its finite update is selected on that
window.  The multiplier is then frozen and compared with zero step, the dense
exact direction, and equal-coordinate random connectivity on new fixed
validation windows.  This diagnostic performs zero parameter updates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import (
    _optimizer_and_group_for_parameter,
    build_candidates,
    evaluate_loss,
    evaluate_with_updates,
    exact_muon_update,
    file_sha256,
    fixed_batches,
    load_model_and_optimizer,
)
from examples.nanogpt.model import GPT
from examples.nanogpt.muon_matched_givens import MuonMatchedGivensLinear


SCHEMA_VERSION = "nanogpt_mlp_cfc_trust_radius_v1"
CONTROL_CANDIDATES = (
    "dense_exact",
    "fresh_expansion88",
    "random_expansion88",
)


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _autocast(device: str, dtype: torch.dtype):
    if not device.startswith("cuda") or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def collect_gradient_window(
    model: GPT,
    batches: list[torch.Tensor],
    layers: list[int],
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[float, dict[int, torch.Tensor]]:
    """Collect only the averaged c_fc gradient; no activation tensors."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, MuonMatchedGivensLinear):
            module.weight.requires_grad_(False)
    selected = {
        layer: model.transformer.h[layer].mlp.c_fc.weight
        for layer in layers
    }
    for parameter in selected.values():
        parameter.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    losses: list[float] = []
    model.prepare_block_fht_cache(dtype=dtype)
    try:
        for tokens in batches:
            tokens = tokens.to(device)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            with _autocast(device, dtype):
                _logits, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("model did not return task loss")
            losses.append(float(loss.detach()))
            (loss / len(batches)).backward()
    finally:
        model.flush_block_fht_cache()
    gradients: dict[int, torch.Tensor] = {}
    for layer, parameter in selected.items():
        if parameter.grad is None:
            raise RuntimeError(f"missing c_fc gradient for layer {layer}")
        gradients[layer] = parameter.grad.detach().float().cpu()
    return sum(losses) / len(losses), gradients


def scaled_updates(
    updates: dict[int, torch.Tensor], scale: float
) -> dict[int, torch.Tensor]:
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("trust scale must be positive and finite")
    return {
        layer: update.float() * float(scale)
        for layer, update in updates.items()
    }


def repeated_losses(
    model: GPT,
    batches: list[torch.Tensor],
    updates: dict[int, torch.Tensor] | None,
    *,
    repeats: int,
    device: str,
    dtype: torch.dtype,
) -> list[float]:
    if repeats < 2:
        raise ValueError("numerical-stability evaluation needs >=2 repeats")
    values = []
    for _ in range(int(repeats)):
        if updates is None:
            value = evaluate_loss(
                model, batches, device=device, dtype=dtype
            )
        else:
            value = evaluate_with_updates(
                model,
                batches,
                updates,
                device=device,
                dtype=dtype,
            )
        values.append(float(value))
    return values


def summarize(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("loss replicates must be nonempty and finite")
    return {
        "mean": sum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def choose_trust_scale(
    rows: list[dict[str, Any]],
    *,
    minimum_fit_improvement: float,
    tie_tolerance: float,
) -> dict[str, Any]:
    baseline = [
        float(row["loss"])
        for row in rows
        if row["phase"] == "fit" and row["candidate"] == "baseline"
    ]
    baseline_summary = summarize(baseline)
    by_scale: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["phase"] == "fit"
            and row["candidate"] == "fresh_expansion88"
        ):
            by_scale[float(row["scale"])].append(float(row["loss"]))
    if not by_scale:
        raise ValueError("fit line search contains no fresh88 scales")
    summaries = {scale: summarize(values) for scale, values in by_scale.items()}
    best_loss = min(value["mean"] for value in summaries.values())
    eligible = [
        scale
        for scale, value in summaries.items()
        if value["mean"] <= best_loss + float(tie_tolerance)
    ]
    selected = min(eligible)
    improvement = baseline_summary["mean"] - summaries[selected]["mean"]
    return {
        "selected_scale": selected,
        "baseline": baseline_summary,
        "scales": {
            str(scale): value
            for scale, value in sorted(summaries.items())
        },
        "fit_improvement": improvement,
        "minimum_fit_improvement": float(minimum_fit_improvement),
        "fit_gate_passed": improvement >= float(minimum_fit_improvement),
        "tie_tolerance": float(tie_tolerance),
        "tie_rule": "smallest positive scale within tolerance of minimum fit CE",
    }


def aggregate_validation(
    rows: list[dict[str, Any]],
    *,
    validation_windows: list[str],
    controls: list[str],
    numerical_range_tolerance: float,
    minimum_test_margin: float,
) -> dict[str, Any]:
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for window in validation_windows:
        summaries[window] = {}
        for candidate in ("baseline", *controls):
            values = [
                float(row["loss"])
                for row in rows
                if row["phase"] == "validation"
                and row["window"] == window
                and row["candidate"] == candidate
            ]
            summaries[window][candidate] = summarize(values)
    stability = all(
        value["range"] <= float(numerical_range_tolerance)
        for window in summaries.values()
        for value in window.values()
    )
    comparisons: dict[str, dict[str, Any]] = {}
    for window, values in summaries.items():
        fresh = values["fresh_expansion88"]
        comparisons[window] = {}
        for control in ("baseline", "dense_exact", "random_expansion88"):
            other = values[control]
            mean_margin = other["mean"] - fresh["mean"]
            robust_margin = other["minimum"] - fresh["maximum"]
            comparisons[window][control] = {
                "mean_ce_margin": mean_margin,
                "worst_replicate_ce_margin": robust_margin,
                "passed": robust_margin >= float(minimum_test_margin),
            }
    control_gate = all(
        comparison["passed"]
        for window in comparisons.values()
        for comparison in window.values()
    )
    return {
        "summaries": summaries,
        "comparisons": comparisons,
        "gates": {
            "numerically_stable": stability,
            "fresh88_beats_baseline_dense_and_random_on_every_window": control_gate,
        },
        "numerical_range_tolerance": float(numerical_range_tolerance),
        "minimum_test_margin": float(minimum_test_margin),
    }


def validate_identity(
    checkpoint: Path,
    config_path: Path,
    data_dir: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    identity = plan["identity"]
    actual = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_sha256": file_sha256(
            data_dir / "manifest.json"
        ),
    }
    for key, value in actual.items():
        if value != identity[key]:
            raise ValueError(f"registered identity mismatch: {key}")
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    fixed_digest = checkpoint_payload["run_identity"]["evaluation"][
        "fixed_eval_indices_sha256"
    ]
    if fixed_digest != identity["fixed_eval_indices_sha256"]:
        raise ValueError("registered fixed-evaluation identity mismatch")
    return plan, checkpoint_payload


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
    plan, identity_checkpoint = validate_identity(
        args.checkpoint, args.config, args.data_dir, args.plan
    )
    protocol = plan["fixed_protocol"]
    layers = [int(layer) for layer in protocol["layers"]]
    trust_scales = [float(value) for value in protocol["trust_scales"]]
    validation_windows = [
        f"validation_{index + 1}"
        for index in range(len(protocol["validation_seeds"]))
    ]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    gradient_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(protocol["gradient_dtype"])]
    evaluation_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[str(protocol["evaluation_dtype"])]
    fit_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["batch_size"]),
        block_size=int(protocol["block_size"]),
        batches=int(protocol["fit_batches"]),
        seed=int(protocol["fit_train_seed"]),
    )
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
            validation_windows, protocol["validation_seeds"], strict=True
        )
    }
    model, optimizer, checkpoint = load_model_and_optimizer(
        args.checkpoint, config, args.device
    )
    fit_loss, fit_gradients = collect_gradient_window(
        model,
        fit_batches,
        layers,
        device=args.device,
        dtype=gradient_dtype,
    )
    candidates_by_name: dict[str, dict[int, torch.Tensor]] = {
        candidate: {} for candidate in CONTROL_CANDIDATES
    }
    optimizer_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for layer in layers:
        weight = model.transformer.h[layer].mlp.c_fc.weight
        owner, group = _optimizer_and_group_for_parameter(optimizer, weight)
        buffer = owner.state[weight].get("momentum_buffer")
        if buffer is None:
            raise RuntimeError(f"missing c_fc momentum for layer {layer}")
        dense, descent, diagnostics = exact_muon_update(
            weight.detach(),
            fit_gradients[layer].to(weight.device),
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
        candidates, selections = build_candidates(
            weight.detach(),
            dense,
            polar_descent,
            parent_stages=int(protocol["parent_stages"]),
            residual_stages=int(protocol["residual_stages"]),
            neighbors=int(protocol["matching_neighbors"]),
            seed=int(protocol["matching_seed"]) + layer * 1009,
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            native_cache=args.native_cache,
        )
        for candidate in CONTROL_CANDIDATES:
            candidates_by_name[candidate][layer] = (
                candidates[candidate].detach().cpu()
            )
        optimizer_rows.append({"layer": layer, **diagnostics})
        selection_rows.extend(
            {"layer": layer, **selection} for selection in selections
        )

    repeats = int(protocol["evaluation_repeats"])
    rows: list[dict[str, Any]] = []
    baseline_fit = repeated_losses(
        model,
        fit_batches,
        None,
        repeats=repeats,
        device=args.device,
        dtype=evaluation_dtype,
    )
    for repeat, loss in enumerate(baseline_fit):
        rows.append(
            {
                "phase": "fit",
                "window": "fit",
                "candidate": "baseline",
                "scale": 0.0,
                "repeat": repeat,
                "loss": loss,
                "loss_change_from_baseline": 0.0,
            }
        )
    baseline_fit_mean = sum(baseline_fit) / len(baseline_fit)
    for scale in trust_scales:
        losses = repeated_losses(
            model,
            fit_batches,
            scaled_updates(candidates_by_name["fresh_expansion88"], scale),
            repeats=repeats,
            device=args.device,
            dtype=evaluation_dtype,
        )
        for repeat, loss in enumerate(losses):
            rows.append(
                {
                    "phase": "fit",
                    "window": "fit",
                    "candidate": "fresh_expansion88",
                    "scale": scale,
                    "repeat": repeat,
                    "loss": loss,
                    "loss_change_from_baseline": loss - baseline_fit_mean,
                }
            )
    selection = choose_trust_scale(
        rows,
        minimum_fit_improvement=float(
            plan["decision_rule"]["minimum_fit_ce_improvement"]
        ),
        tie_tolerance=float(
            plan["decision_rule"]["fit_tie_tolerance"]
        ),
    )
    selected_scale = float(selection["selected_scale"])

    for window, batches in validation_batches.items():
        baseline = repeated_losses(
            model,
            batches,
            None,
            repeats=repeats,
            device=args.device,
            dtype=evaluation_dtype,
        )
        baseline_mean = sum(baseline) / len(baseline)
        for repeat, loss in enumerate(baseline):
            rows.append(
                {
                    "phase": "validation",
                    "window": window,
                    "candidate": "baseline",
                    "scale": 0.0,
                    "repeat": repeat,
                    "loss": loss,
                    "loss_change_from_baseline": 0.0,
                }
            )
        for candidate in CONTROL_CANDIDATES:
            losses = repeated_losses(
                model,
                batches,
                scaled_updates(candidates_by_name[candidate], selected_scale),
                repeats=repeats,
                device=args.device,
                dtype=evaluation_dtype,
            )
            for repeat, loss in enumerate(losses):
                rows.append(
                    {
                        "phase": "validation",
                        "window": window,
                        "candidate": candidate,
                        "scale": selected_scale,
                        "repeat": repeat,
                        "loss": loss,
                        "loss_change_from_baseline": loss - baseline_mean,
                    }
                )

    validation = aggregate_validation(
        rows,
        validation_windows=validation_windows,
        controls=list(CONTROL_CANDIDATES),
        numerical_range_tolerance=float(
            plan["decision_rule"]["maximum_replicate_range"]
        ),
        minimum_test_margin=float(
            plan["decision_rule"]["minimum_validation_ce_margin"]
        ),
    )
    all_finite = all(math.isfinite(float(row["loss"])) for row in rows)
    passed = all(
        (
            selection["fit_gate_passed"],
            validation["gates"]["numerically_stable"],
            validation["gates"][
                "fresh88_beats_baseline_dense_and_random_on_every_window"
            ],
            all_finite,
        )
    )
    decision = (
        "SELECT_TRUST_SCALED_FRESH88_CFC_FOR_PRODUCTION_MFU_GATE"
        if passed
        else "REJECT_TRUST_SCALED_FRESH88_CFC"
    )
    aggregate = {
        "decision": decision,
        "parameter_updates": 0,
        "fit_gradient_loss_bfloat16": fit_loss,
        "selection": selection,
        "validation": validation,
        "gates": {
            "fit_improvement": selection["fit_gate_passed"],
            "numerically_stable": validation["gates"][
                "numerically_stable"
            ],
            "fresh88_beats_all_controls_every_window": validation[
                "gates"
            ]["fresh88_beats_baseline_dense_and_random_on_every_window"],
            "all_losses_finite": all_finite,
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    losses_path = args.output / "cfc_trust_radius_losses.csv"
    optimizer_path = args.output / "cfc_trust_radius_optimizer.csv"
    selections_path = args.output / "cfc_trust_radius_selections.json"
    aggregate_path = args.output / "cfc_trust_radius_aggregate.json"
    write_csv(losses_path, rows)
    write_csv(optimizer_path, optimizer_rows)
    selections_path.write_text(
        json.dumps(selection_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "parameter_updates": 0,
        "checkpoint_next_iter": int(checkpoint["next_iter"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(
            args.data_dir / "manifest.json"
        ),
        "fixed_eval_indices_sha256": identity_checkpoint["run_identity"][
            "evaluation"
        ]["fixed_eval_indices_sha256"],
        "plan_sha256": file_sha256(args.plan),
        "analysis_execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "device": args.device,
        },
        "protocol": protocol,
        "structural_invariants": {
            "parent_stages": int(protocol["parent_stages"]),
            "residual_stages": int(protocol["residual_stages"]),
            "learned_dense_basis": False,
            "dense_residual_adapter": False,
            "lora_adapter": False,
            "connectivity_or_coordinate_change": False,
        },
        "outputs": {
            "losses_sha256": file_sha256(losses_path),
            "optimizer_sha256": file_sha256(optimizer_path),
            "selections_sha256": file_sha256(selections_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        },
        "limitations": plan["limitations"],
    }
    metadata_path = args.output / "cfc_trust_radius_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "selected_scale": selected_scale,
                "aggregate": str(aggregate_path),
                "metadata": str(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
