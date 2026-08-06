#!/usr/bin/env python3
"""Exact fixed-window CE test of scheduled late-c_proj endpoint oracles."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import model_from_snapshot
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import file_sha256
from examples.nanogpt.analyze_mlp_cproj_coadapted_orbit_geometry import buffer_name, git_commit, load_snapshot
from examples.nanogpt.analyze_mlp_cproj_multiscale_path import (
    cumulative_lr_coordinate,
    polynomial_predict,
)
from examples.nanogpt.analyze_mlp_cproj_predictive_manifold import fit_through_origin_basis
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.train import (
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)
from examples.nanogpt.verify_full_state_functional_replay import evaluate_validation_ce


PLAN_SCHEMA = "mai_124m_mlp_cproj_polynomial_oracle_ce_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_polynomial_oracle_ce_result_v1"
LAYERS = tuple(range(8, 12))
DISCOVERY_STEPS = (0, 99, 198, 297, 396, 495, 594, 693, 792, 891, 990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782)
RANKS = (1, 2, 4, 8, 16)
TERMINAL_STEP = 2373


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected polynomial-oracle CE plan schema")
    analysis = plan.get("analysis", {})
    expected = {
        "parameter_updates": 0,
        "layers": list(LAYERS),
        "discovery_steps": list(DISCOVERY_STEPS),
        "terminal_step": TERMINAL_STEP,
        "ranks": list(RANKS),
        "polynomial_degree": 2,
        "polynomial_coordinate": "cumulative_learning_rate",
        "restore_terminal_radius": True,
        "validation_batches": 400,
        "learned_basis_role": "diagnostic_oracle_only",
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            raise ValueError(f"polynomial-oracle analysis field changed: {key}")
    thresholds = {
        "exact_replay_absolute_tolerance_ce": 0.005,
        "candidate_maximum_validation_ce_gap": 0.005,
    }
    if plan.get("decision_rule", {}).get("thresholds") != thresholds:
        raise ValueError("polynomial-oracle thresholds changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_terminal_oracle_ce") is not True:
        raise ValueError("terminal oracle CE is not authorized")
    for key in (
        "use_learned_basis_in_candidate",
        "use_training_time_as_candidate_latent",
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def restore_radius(weight: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    return weight * (target_norm / weight.norm().clamp_min(1e-30))


def classify(rows: list[dict[str, Any]], replay_valid: bool) -> tuple[str, int | None]:
    if not replay_valid:
        return "INVALID_EXACT_REPLAY", None
    passing = [int(row["rank"]) for row in rows if row["passes_ce_gate"]]
    if passing:
        return "SMOOTH_SCHEDULED_ENVELOPE_FUNCTIONALLY_SUFFICIENT", min(passing)
    return "TASK_ADAPTIVE_RESIDUAL_FUNCTIONALLY_NECESSARY", None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--multiscale-result", type=Path, required=True)
    parser.add_argument("--trajectory-verification", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    identity = plan["identity"]
    pinned = {
        Path(__file__): identity["analyzer_sha256"],
        args.multiscale_result: identity["multiscale_result_sha256"],
        args.trajectory_verification: identity["trajectory_verification_sha256"],
        args.config: identity["config_sha256"],
    }
    for path, expected in pinned.items():
        if file_sha256(path) != expected:
            raise ValueError(f"pinned artifact SHA-256 mismatch: {path}")
    for relative, expected in identity["supporting_source_sha256"].items():
        if file_sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"supporting-source SHA-256 mismatch: {relative}")

    multiscale = json.loads(args.multiscale_result.read_text())
    if multiscale.get("classification") != "SCHEDULED_ENDPOINT_CURVE_WITHOUT_LOCAL_SECANT_TRANSPORT":
        raise ValueError("multiscale source classification changed")
    if multiscale.get("authorization", {}).get("terminal_polynomial_oracle_fixed_ce") is not True:
        raise ValueError("multiscale result did not authorize oracle CE")
    verification = json.loads(args.trajectory_verification.read_text())
    if verification.get("classification") != "ACCEPTED_COADAPTED_LATE_CPROJ_FULL_STATE_TRAJECTORY":
        raise ValueError("trajectory verification is not accepted")
    config = json.loads(args.config.read_text())
    manifest = Path(config["data_dir"]) / "manifest.json"
    if file_sha256(manifest) != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest changed")
    require_block_fht_native_extension(bool(config["block_fht_native_extension_required"]))
    fixed = make_fixed_eval_indices(
        Path(config["data_dir"]), int(config["eval_batch_size"]),
        int(config["block_size"]), int(config["eval_iters"]), int(config["eval_seed"]),
    )
    if fixed_eval_indices_digest(fixed) != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation indices changed")
    hashes = verification["inventory"]["snapshot_sha256_by_step"]
    run_identity = verification["identity"]["run_identity_sha256"]
    if run_identity != identity["run_identity_sha256"]:
        raise ValueError("run identity changed")
    steps = list(DISCOVERY_STEPS) + [TERMINAL_STEP]
    snapshots = {
        step: load_snapshot(
            args.snapshot_dir / f"step_{step:06d}.pt", hashes[str(step)], run_identity
        )
        for step in steps
    }
    terminal_snapshot = snapshots[TERMINAL_STEP]
    model = model_from_snapshot(terminal_snapshot, args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    eval_args = SimpleNamespace(**config)
    eval_args.device = args.device
    eval_args._ptdtype = dtype
    ctx = contextlib.nullcontext() if "cuda" not in args.device else torch.amp.autocast(device_type="cuda", dtype=dtype)
    accepted_terminal_ce = float(next(
        row["replayed_validation_ce"]
        for row in verification["functional_replay"]["rows"]
        if int(row["step"]) == TERMINAL_STEP
    ))
    terminal_progress = torch.tensor(
        [cumulative_lr_coordinate(TERMINAL_STEP, config)], device=args.device
    )
    discovery_progress = torch.tensor(
        [cumulative_lr_coordinate(step, config) for step in DISCOVERY_STEPS],
        device=args.device,
    )
    predicted_by_rank: dict[int, dict[int, torch.Tensor]] = {rank: {} for rank in RANKS}
    exact_terminal: dict[int, torch.Tensor] = {}
    for layer in LAYERS:
        weights = {
            step: snapshots[step]["buffers"][buffer_name(layer, "weight")].float().to(args.device)
            for step in steps
        }
        initial_norm = weights[0].norm().clamp_min(1e-30)
        normalized = {
            step: weight * (initial_norm / weight.norm().clamp_min(1e-30))
            for step, weight in weights.items()
        }
        discovery = torch.stack([
            (normalized[step] - normalized[0]).reshape(-1)
            for step in DISCOVERY_STEPS
        ])
        maximum_basis = fit_through_origin_basis(discovery[1:], max(RANKS))
        terminal_norm = weights[TERMINAL_STEP].norm()
        exact_terminal[layer] = weights[TERMINAL_STEP].clone()
        for rank in RANKS:
            basis = maximum_basis[:rank]
            coordinates = discovery @ basis.T
            predicted_coordinate = polynomial_predict(
                coordinates, discovery_progress, terminal_progress, 2
            )
            predicted_normalized = normalized[0] + (
                predicted_coordinate @ basis
            ).reshape_as(normalized[0])
            predicted_by_rank[rank][layer] = restore_radius(
                predicted_normalized, terminal_norm
            )

    started = time.time()
    exact_replay_ce = evaluate_validation_ce(
        model, data_dir=Path(config["data_dir"]), args=eval_args,
        indices=fixed["val"], ctx=ctx,
    )
    replay_error = abs(exact_replay_ce - accepted_terminal_ce)
    threshold = float(plan["decision_rule"]["thresholds"]["candidate_maximum_validation_ce_gap"])
    rows: list[dict[str, Any]] = []
    for rank in RANKS:
        with torch.no_grad():
            for layer in LAYERS:
                model.transformer.h[layer].mlp.c_proj.weight.copy_(predicted_by_rank[rank][layer])
        validation_ce = evaluate_validation_ce(
            model, data_dir=Path(config["data_dir"]), args=eval_args,
            indices=fixed["val"], ctx=ctx,
        )
        gap = validation_ce - exact_replay_ce
        rows.append({
            "rank": rank,
            "validation_ce": validation_ce,
            "exact_replay_ce": exact_replay_ce,
            "validation_ce_gap": gap,
            "passes_ce_gate": gap <= threshold,
        })
    with torch.no_grad():
        for layer in LAYERS:
            model.transformer.h[layer].mlp.c_proj.weight.copy_(exact_terminal[layer])
    replay_valid = replay_error <= float(
        plan["decision_rule"]["thresholds"]["exact_replay_absolute_tolerance_ce"]
    )
    classification, minimum_rank = classify(rows, replay_valid)
    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "polynomial_oracle_ce_rows.csv"
    result_path = args.output_dir / "polynomial_oracle_ce_result.json"
    write_csv(rows_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": classification,
        "execution": {"parameter_updates": 0, "runtime_seconds": time.time() - started, "device": args.device, "git_commit": git_commit(REPO_ROOT)},
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "analyzer_sha256": file_sha256(Path(__file__)),
            "multiscale_result_sha256": file_sha256(args.multiscale_result),
            "trajectory_verification_sha256": file_sha256(args.trajectory_verification),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(manifest),
            "fixed_eval_indices_sha256": fixed_eval_indices_digest(fixed),
            "run_identity_sha256": run_identity,
        },
        "accepted_terminal_ce": accepted_terminal_ce,
        "exact_replay_ce": exact_replay_ce,
        "exact_replay_absolute_error_ce": replay_error,
        "minimum_passing_rank": minimum_rank,
        "rows": rows,
        "decision_rule": plan["decision_rule"],
        "interpretation_contract": plan["interpretation_contract"],
        "authorization": plan["authorization"],
        "artifacts": {"rows": str(rows_path)},
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
