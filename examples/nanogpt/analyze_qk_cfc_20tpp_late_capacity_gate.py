#!/usr/bin/env python3
"""Test a late-layer-only c_fc chart-capacity repair across one 20TPP run.

This is a zero-training-update gate.  At four same-run phases it builds a
stateless Muon task direction from one fixed training window, fits the current
directed-product chart, fits a wider chart only for layers 8--11, and fits an
equal-coordinate task-independent sparse-connectivity control.  It evaluates
direction recovery on both the fit window and a disjoint confirmation window,
then applies each fit-window update at the same scheduled-LR family radius and
measures paired validation CE.  Historical Muon momentum and compression
residuals are absent from the full-model snapshots, so no result is called an
exact optimizer replay.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_cfc_directed_product_terminal import (
    DirectedProductCfcApplier,
    cfc_modules,
    collect_cfc_gradients,
    scaled_to_dense_ratio,
)
from examples.nanogpt.analyze_mlp_cfc_directed_product_terminal_capacity import (
    fit_schedule,
)
from examples.nanogpt.analyze_mlp_cfc_exact_current_matcher import fixed_batches
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_mlp_dense_oracle_gap import (
    aggregate_direction_metrics,
)
from examples.nanogpt.analyze_qk_cfc_20tpp_phase_direction import evaluate_ce
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.muon import zeropower_via_newtonschulz5


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "mai_124m_qk_cfc_20tpp_late_capacity_gate_v1"
CANDIDATES = ("current_132", "late_wide_176", "late_random_176")


def cosine_lr(
    step: int,
    *,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    if step < warmup_iters:
        return learning_rate * (step + 1) / (warmup_iters + 1)
    if step > lr_decay_iters:
        return min_lr
    ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coefficient * (learning_rate - min_lr)


@torch.no_grad()
def stateless_muon_updates(
    modules,
    *,
    learning_rate: float,
    weight_decay: float,
    ns_steps: int,
) -> dict[int, torch.Tensor]:
    updates: dict[int, torch.Tensor] = {}
    for layer, module in enumerate(modules):
        weight = module.weight
        gradient = weight.grad
        if gradient is None:
            raise RuntimeError(f"missing c_fc gradient for layer {layer}")
        polar = zeropower_via_newtonschulz5(
            gradient.float(), steps=ns_steps
        ).float()
        scale = max(
            1.0,
            polar.shape[0] / max(1, polar.numel() / polar.shape[0]),
        ) ** 0.5
        updates[layer] = (
            learning_rate
            * (-scale * polar - weight_decay * weight.float())
        ).detach().cpu()
    return updates


def deterministic_random_sources(
    *, width: int, total: int, members: int, seed: int
) -> torch.Tensor:
    """Return task-independent, no-self, per-target unique sources.

    Each target receives a randomized affine traversal of the output-channel
    cycle.  The stride is coprime to ``width``, so the first ``total``
    non-self members are unique.  Shape is ``[members, total, width]``.
    """
    if width <= 2 or total <= 0 or total >= width or members <= 0:
        raise ValueError("invalid random sparse geometry")
    coprime = torch.tensor(
        [value for value in range(1, width) if math.gcd(value, width) == 1],
        dtype=torch.int64,
    )
    targets = torch.arange(width, dtype=torch.int64)
    rows: list[torch.Tensor] = []
    for member in range(members):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1000003 * member)
        stride_indices = torch.randint(
            0,
            len(coprime),
            (width,),
            generator=generator,
            dtype=torch.int64,
        )
        strides = coprime[stride_indices]
        positions = torch.arange(1, total + 1, dtype=torch.int64)
        values = (
            targets[:, None]
            + strides[:, None] * positions[None]
        ) % width
        rows.append(values.T.contiguous())
    return torch.stack(rows)


@torch.no_grad()
def fit_random_schedule(
    modules,
    dense: dict[int, torch.Tensor],
    *,
    layers: list[int],
    schedule: list[int],
    ridge_ratio: float,
    chunk_size: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    source = torch.stack(
        [modules[layer].weight.float().T for layer in layers], dim=0
    ).contiguous()
    target = torch.stack(
        [dense[layer].T.to(source.device) for layer in layers], dim=0
    ).contiguous()
    batch, rows, width = source.shape
    selected = deterministic_random_sources(
        width=width,
        total=sum(schedule),
        members=batch,
        seed=seed,
    ).to(source.device)
    transformed = source.clone()
    prediction = torch.zeros_like(target)
    offset = 0
    for incoming in schedule:
        remaining = target - prediction
        indices = selected[:, offset : offset + incoming]
        offset += incoming
        stage_update = torch.empty_like(remaining)
        eye = torch.eye(
            incoming, device=source.device, dtype=torch.float32
        )[None, None]
        for start in range(0, width, int(chunk_size)):
            stop = min(start + int(chunk_size), width)
            columns = stop - start
            chosen = indices[:, :, start:stop]
            dictionary = torch.gather(
                transformed.unsqueeze(3).expand(-1, -1, -1, columns),
                2,
                chosen[:, None].expand(-1, rows, -1, -1),
            ).permute(0, 3, 1, 2).contiguous()
            targets = (
                remaining[:, :, start:stop]
                .permute(0, 2, 1)
                .contiguous()
                .unsqueeze(-1)
            )
            gram = dictionary.transpose(-1, -2) @ dictionary
            rhs = dictionary.transpose(-1, -2) @ targets
            diagonal_mean = gram.diagonal(
                dim1=-2, dim2=-1
            ).mean(dim=-1)
            gram.add_(
                eye
                * (float(ridge_ratio) * diagonal_mean)[..., None, None]
            )
            coefficients = torch.linalg.solve(gram, rhs)
            stage_update[:, :, start:stop] = (
                (dictionary @ coefficients)
                .squeeze(-1)
                .permute(0, 2, 1)
            )
        transformed.add_(stage_update)
        prediction.add_(stage_update)
    return {
        layer: prediction[index].T.contiguous().cpu()
        for index, layer in enumerate(layers)
    }


def combine_late(
    current: dict[int, torch.Tensor],
    late: dict[int, torch.Tensor],
    late_layers: list[int],
) -> dict[int, torch.Tensor]:
    output = {layer: value.clone() for layer, value in current.items()}
    for layer in late_layers:
        output[layer] = late[layer].clone()
    return output


def subset(
    values: dict[int, torch.Tensor], layers: list[int]
) -> dict[int, torch.Tensor]:
    return {layer: values[layer] for layer in layers}


def paired_ce(
    candidate: list[float], current: list[float], confidence_z: float
) -> dict[str, Any]:
    if len(candidate) != len(current) or not candidate:
        raise ValueError("paired CE rows must be complete")
    differences = [left - right for left, right in zip(candidate, current, strict=True)]
    mean = sum(differences) / len(differences)
    standard_error = (
        statistics.stdev(differences) / math.sqrt(len(differences))
        if len(differences) > 1
        else 0.0
    )
    return {
        "candidate_minus_current_mean_ce": mean,
        "standard_error": standard_error,
        "upper_confidence_bound": mean + confidence_z * standard_error,
        "differences": differences,
    }


def classify(
    phase_rows: dict[str, Any],
    functional: dict[str, Any],
    rule: dict[str, Any],
) -> dict[str, Any]:
    confirmation = "holdout"
    failure_steps = [str(value) for value in rule["failure_steps"]]
    preservation_step = str(rule["preservation_step"])
    wide = "late_wide_176"
    current = "current_132"
    random = "late_random_176"

    def recovery(step: str, window: str, candidate: str) -> float:
        return float(
            phase_rows[step][window][candidate]["late_positive_line_recovery"]
        )

    failure_improvements = {
        step: {
            window: recovery(step, window, wide)
            - recovery(step, window, current)
            for window in ("fit", confirmation)
        }
        for step in failure_steps
    }
    random_margins = {
        step: recovery(step, confirmation, wide)
        - recovery(step, confirmation, random)
        for step in failure_steps
    }
    preservation_margin = (
        recovery(preservation_step, confirmation, wide)
        - recovery(preservation_step, confirmation, current)
    )
    functional_rows = {
        step: functional[step][wide]["vs_current"] for step in failure_steps
    }
    passes = (
        min(
            value
            for step in failure_improvements.values()
            for value in step.values()
        )
        >= float(rule["minimum_failure_recovery_improvement"])
        and min(random_margins.values())
        >= float(rule["minimum_holdout_margin_over_random"])
        and preservation_margin
        >= -float(rule["maximum_preservation_recovery_regression"])
        and max(
            float(value["upper_confidence_bound"])
            for value in functional_rows.values()
        )
        <= float(rule["maximum_functional_upper_bound_regression_ce"])
        and min(
            float(value["candidate_minus_current_mean_ce"])
            for value in functional_rows.values()
        )
        <= -float(rule["minimum_one_phase_mean_ce_improvement"])
    )
    return {
        "classification": (
            "PROMOTE_LATE_LAYER_CAPACITY_TO_IMPLEMENTATION_AND_MFU_GATE"
            if passes
            else "REJECT_LATE_LAYER_CAPACITY_REPAIR"
        ),
        "passes": passes,
        "failure_recovery_improvements": failure_improvements,
        "holdout_margins_over_random": random_margins,
        "preservation_holdout_margin": preservation_margin,
        "failure_functional_comparisons": functional_rows,
        "thresholds": rule,
        "parameter_updates_to_source": 0,
        "training_authorized": False,
        "mfu_preflight_authorized": passes,
    }


def validate_inputs(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "config_sha256": file_sha256(args.config),
        "phase_result_sha256": file_sha256(args.phase_result),
        "verification_sha256": file_sha256(args.verification),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
    }
    if observed != plan["identity"]:
        raise ValueError(f"late-capacity identity mismatch: {observed}")
    verification = json.loads(args.verification.read_text())
    if verification.get("passed") is not True:
        raise ValueError("phase acquisition verification is not accepted")
    if (
        plan["snapshot_sha256_by_step"]
        != verification["inventory"]["snapshot_sha256_by_step"]
    ):
        raise ValueError("registered snapshot inventory changed")
    phase = json.loads(args.phase_result.read_text())
    if phase.get("decision", {}).get("classification") != "PHASE_PRE_GELU_CAPACITY_DRIFT":
        raise ValueError("phase diagnosis does not authorize this audit")
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--phase-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != "mai_124m_qk_cfc_20tpp_late_capacity_gate_plan_v1":
        raise ValueError("unexpected late-capacity plan schema")
    verification = validate_inputs(args, plan)
    protocol = plan["protocol"]
    config = json.loads(args.config.read_text())
    dtype = getattr(torch, str(config["dtype"]))
    steps = [int(value) for value in protocol["steps"]]
    expected_hashes = verification["inventory"]["snapshot_sha256_by_step"]
    paths = [args.snapshot_dir / f"step_{step:06d}.pt" for step in steps]
    for step, path in zip(steps, paths, strict=True):
        if not path.is_file() or file_sha256(path) != expected_hashes[str(step)]:
            raise ValueError(f"snapshot identity mismatch at step {step}")

    fit_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["gradient_batch_size"]),
        block_size=int(config["block_size"]) + 1,
        batches=int(protocol["gradient_batches"]),
        seed=int(protocol["fit_seed"]),
    )
    holdout_batches = fixed_batches(
        args.data_dir,
        "train",
        batch_size=int(protocol["gradient_batch_size"]),
        block_size=int(config["block_size"]) + 1,
        batches=int(protocol["gradient_batches"]),
        seed=int(protocol["holdout_seed"]),
    )
    validation_batches = fixed_validation_batches(
        args.data_dir,
        int(protocol["validation_batch_size"]),
        int(config["block_size"]) + 1,
        int(protocol["validation_batches"]),
        int(protocol["validation_seed"]),
    )
    current_schedule = [int(value) for value in protocol["current_schedule"]]
    wide_schedule = [int(value) for value in protocol["wide_schedule"]]
    late_layers = [int(value) for value in protocol["late_layers"]]
    all_layers = list(range(12))
    started = time.time()
    phase_rows: dict[str, Any] = {}
    functional: dict[str, Any] = {}
    for step, path in zip(steps, paths, strict=True):
        print(f"phase {step}: loading and collecting gradients", flush=True)
        payload = load_snapshot(path)
        model = model_from_snapshot(payload, args.device)
        modules = cfc_modules(model)
        lr = cosine_lr(
            step,
            learning_rate=float(config["learning_rate"]),
            min_lr=float(config["min_lr"]),
            warmup_iters=int(config["warmup_iters"]),
            lr_decay_iters=int(config["lr_decay_iters"]),
        )
        fit_ce = collect_cfc_gradients(
            model, modules, fit_batches, device=args.device, dtype=dtype
        )
        dense_fit = stateless_muon_updates(
            modules,
            learning_rate=lr,
            weight_decay=float(config["weight_decay"]),
            ns_steps=int(config["muon_ns_steps"]),
        )
        current_raw, _ = fit_schedule(
            modules,
            dense_fit,
            schedule=current_schedule,
            ridge_ratio=float(protocol["ridge_ratio"]),
            chunk_size=int(protocol["chunk_size"]),
        )
        late_dense_fit = subset(dense_fit, late_layers)
        late_modules = [modules[layer] for layer in late_layers]
        wide_values, _ = fit_schedule(
            late_modules,
            {index: late_dense_fit[layer] for index, layer in enumerate(late_layers)},
            schedule=wide_schedule,
            ridge_ratio=float(protocol["ridge_ratio"]),
            chunk_size=int(protocol["chunk_size"]),
        )
        wide_late = {
            layer: wide_values[index]
            for index, layer in enumerate(late_layers)
        }
        random_late = fit_random_schedule(
            modules,
            dense_fit,
            layers=late_layers,
            schedule=wide_schedule,
            ridge_ratio=float(protocol["ridge_ratio"]),
            chunk_size=int(protocol["chunk_size"]),
            seed=int(protocol["random_seed"]) + step,
        )
        raw_candidates = {
            "current_132": current_raw,
            "late_wide_176": combine_late(current_raw, wide_late, late_layers),
            "late_random_176": combine_late(current_raw, random_late, late_layers),
        }
        updates = {
            name: scaled_to_dense_ratio(raw, dense_fit, float(protocol["radius_ratio"]))
            for name, raw in raw_candidates.items()
        }

        holdout_ce = collect_cfc_gradients(
            model, modules, holdout_batches, device=args.device, dtype=dtype
        )
        dense_holdout = stateless_muon_updates(
            modules,
            learning_rate=lr,
            weight_decay=float(config["weight_decay"]),
            ns_steps=int(config["muon_ns_steps"]),
        )
        phase_rows[str(step)] = {
            "scheduled_lr": lr,
            "gradient_mean_ce": {"fit": fit_ce, "holdout": holdout_ce},
            "fit": {},
            "holdout": {},
        }
        for window, target in (("fit", dense_fit), ("holdout", dense_holdout)):
            for name, update in updates.items():
                phase_rows[str(step)][window][name] = {
                    "all": aggregate_direction_metrics(target, update),
                    "late": aggregate_direction_metrics(
                        subset(target, late_layers), subset(update, late_layers)
                    ),
                    "late_positive_line_recovery": aggregate_direction_metrics(
                        subset(target, late_layers), subset(update, late_layers)
                    )["positive_line_recovery"],
                }

        applier = DirectedProductCfcApplier(modules)
        baseline_mean, baseline_losses = evaluate_ce(
            model,
            validation_batches,
            device=args.device,
            dtype=dtype,
        )
        functional[str(step)] = {
            "baseline_no_update": {
                "mean_ce": baseline_mean,
                "batch_ce": baseline_losses,
            }
        }
        for name in CANDIDATES:
            with applier.apply({"c_fc": updates[name]}):
                mean_ce, losses = evaluate_ce(
                    model,
                    validation_batches,
                    device=args.device,
                    dtype=dtype,
                )
            functional[str(step)][name] = {
                "mean_ce": mean_ce,
                "batch_ce": losses,
            }
        current_losses = functional[str(step)]["current_132"]["batch_ce"]
        for name in ("late_wide_176", "late_random_176"):
            functional[str(step)][name]["vs_current"] = paired_ce(
                functional[str(step)][name]["batch_ce"],
                current_losses,
                float(protocol["confidence_z"]),
            )
        del model, payload
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    decision = classify(phase_rows, functional, plan["decision_rule"])
    args.output.mkdir(parents=True)
    detail_path = args.output / "phase_capacity_rows.json"
    functional_path = args.output / "functional_ce.json"
    detail_path.write_text(json.dumps(phase_rows, indent=2, sort_keys=True) + "\n")
    functional_path.write_text(json.dumps(functional, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "phase_rows": phase_rows,
        "functional": functional,
        "identity": {
            **plan["identity"],
            "plan_sha256": file_sha256(args.plan),
            "phase_rows_sha256": file_sha256(detail_path),
            "functional_sha256": file_sha256(functional_path),
            "snapshot_run_identity_sha256": str(
                load_snapshot(paths[0])["run_identity_sha256"]
            ),
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "limitations": [
            "Full-model snapshots do not contain Muon momentum or compression residuals; task directions are stateless Muon controls.",
            "One-step fixed-token CE is a local action test, not a language-model training result.",
            "Only the measured late-band capacity hypothesis is tested; modulation, smoothness loss, and unrelated structures are not searched."
        ],
        "authorization": {
            "implementation_and_mfu_preflight": bool(decision["passes"]),
            "language_model_training": False,
            "larger_rung": False,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
