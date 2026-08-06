#!/usr/bin/env python3
"""Measure the causal bandwidth of the accepted late-c_proj manifold drift."""

from __future__ import annotations

import argparse
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

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import file_sha256
from examples.nanogpt.analyze_mlp_cproj_coadapted_orbit_geometry import buffer_name, git_commit, load_snapshot
from examples.nanogpt.analyze_mlp_cproj_predictive_manifold import (
    fit_through_origin_basis,
    functional_recovery,
    pooled_recovery,
    project_rows,
    terminal_post_gelu_activations,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.train import (
    cosine_lr,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)


PLAN_SCHEMA = "mai_124m_mlp_cproj_multiscale_path_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_multiscale_path_result_v1"
LAYERS = tuple(range(8, 12))
DISCOVERY_STEPS = (0, 99, 198, 297, 396, 495, 594, 693, 792, 891, 990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782)
HOLDOUT_STEPS = (1881, 1980, 2079, 2178, 2277, 2373)
RANKS = (1, 2, 4, 8, 16)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected multiscale-path plan schema")
    analysis = plan.get("analysis", {})
    expected = {
        "parameter_updates": 0,
        "layers": list(LAYERS),
        "discovery_steps": list(DISCOVERY_STEPS),
        "holdout_steps": list(HOLDOUT_STEPS),
        "ranks": list(RANKS),
        "primary_rank": 8,
        "rolling_horizon_indices": [1, 2, 3, 4, 5, 6],
        "polynomial_degree": 2,
        "polynomial_coordinate": "cumulative_learning_rate",
        "learned_basis_role": "diagnostic_oracle_only",
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            raise ValueError(f"multiscale analysis field changed: {key}")
    thresholds = {
        "secant_weight_recovery": 0.80,
        "secant_functional_recovery": 0.80,
        "online_max_updates": 198,
        "phase_max_updates": 594,
        "polynomial_endpoint_weight_recovery": 0.80,
        "polynomial_endpoint_functional_recovery": 0.80,
    }
    if plan.get("decision_rule", {}).get("thresholds") != thresholds:
        raise ValueError("multiscale thresholds changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_multiscale_analysis") is not True:
        raise ValueError("multiscale analysis is not authorized")
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


def cumulative_lr_coordinate(step: int, config: dict[str, Any]) -> float:
    schedule = SimpleNamespace(**config)
    total = sum(cosine_lr(index, schedule) for index in range(int(config["max_iters"])))
    prefix = sum(cosine_lr(index, schedule) for index in range(step))
    return prefix / max(total, 1e-30)


def polynomial_predict(
    discovery_coordinates: torch.Tensor,
    discovery_progress: torch.Tensor,
    holdout_progress: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    if degree != 2:
        raise ValueError("only the frozen quadratic control is supported")
    design = torch.stack((discovery_progress, discovery_progress.square()), dim=1)
    holdout_design = torch.stack((holdout_progress, holdout_progress.square()), dim=1)
    coefficients = torch.linalg.lstsq(design.float(), discovery_coordinates.float()).solution
    return holdout_design.float() @ coefficients


def classify(onset_updates: int | None, polynomial_pass: bool) -> str:
    if onset_updates is not None and onset_updates <= 198:
        return (
            "ONLINE_LOW_BANDWIDTH_SMOOTH_CURVE"
            if polynomial_pass
            else "ONLINE_LOW_BANDWIDTH_NONSTATIONARY_CURVE"
        )
    if onset_updates is not None and onset_updates <= 594:
        return (
            "PHASE_SCALE_SMOOTH_CURVE"
            if polynomial_pass
            else "PHASE_SCALE_ENVELOPE_ONLY"
        )
    if polynomial_pass:
        return "SCHEDULED_ENDPOINT_CURVE_WITHOUT_LOCAL_SECANT_TRANSPORT"
    return "ENDPOINT_ENVELOPE_WITHOUT_CAUSAL_SECANT_TRANSPORT"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--predictive-result", type=Path, required=True)
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
        args.predictive_result: identity["predictive_result_sha256"],
        args.trajectory_verification: identity["trajectory_verification_sha256"],
        args.config: identity["config_sha256"],
    }
    for path, expected in pinned.items():
        if file_sha256(path) != expected:
            raise ValueError(f"pinned artifact SHA-256 mismatch: {path}")
    supporting = identity["supporting_source_sha256"]
    for relative, expected in supporting.items():
        if file_sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"supporting-source SHA-256 mismatch: {relative}")

    predictive = json.loads(args.predictive_result.read_text())
    if predictive.get("classification") != "SMOOTH_ENDPOINT_WITH_NONTRANSPORTABLE_TANGENTS":
        raise ValueError("predictive source result has unexpected classification")
    if predictive.get("authorization", {}).get("multiscale_causal_path_analysis") is not True:
        raise ValueError("predictive result did not authorize multiscale analysis")
    verification = json.loads(args.trajectory_verification.read_text())
    if verification.get("classification") != "ACCEPTED_COADAPTED_LATE_CPROJ_FULL_STATE_TRAJECTORY":
        raise ValueError("trajectory verification is not accepted")
    config = json.loads(args.config.read_text())
    manifest = Path(config["data_dir"]) / "manifest.json"
    if file_sha256(manifest) != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    require_block_fht_native_extension(bool(config["block_fht_native_extension_required"]))
    fixed = make_fixed_eval_indices(
        Path(config["data_dir"]),
        int(config["eval_batch_size"]),
        int(config["block_size"]),
        int(config["eval_iters"]),
        int(config["eval_seed"]),
    )
    if fixed_eval_indices_digest(fixed) != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation indices changed")

    hashes = verification["inventory"]["snapshot_sha256_by_step"]
    run_identity = verification["identity"]["run_identity_sha256"]
    if run_identity != identity["run_identity_sha256"]:
        raise ValueError("run identity changed")
    all_steps = list(DISCOVERY_STEPS) + list(HOLDOUT_STEPS)
    snapshots = {
        step: load_snapshot(
            args.snapshot_dir / f"step_{step:06d}.pt", hashes[str(step)], run_identity
        )
        for step in all_steps
    }
    started = time.time()
    activations = terminal_post_gelu_activations(
        snapshots[HOLDOUT_STEPS[-1]], config, fixed,
        int(plan["analysis"]["functional_metric"]["activation_rows"]), args.device,
    )
    discovery_progress = torch.tensor(
        [cumulative_lr_coordinate(step, config) for step in DISCOVERY_STEPS],
        device=args.device,
    )
    holdout_progress = torch.tensor(
        [cumulative_lr_coordinate(step, config) for step in HOLDOUT_STEPS],
        device=args.device,
    )
    late_sequence = (DISCOVERY_STEPS[-1],) + HOLDOUT_STEPS
    rows: list[dict[str, Any]] = []
    chord_accumulator: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    polynomial_accumulator: dict[int, list[tuple[float, float]]] = {}

    for layer in LAYERS:
        weights = {
            step: snapshots[step]["buffers"][buffer_name(layer, "weight")].float().to(args.device)
            for step in all_steps
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
        holdout = torch.stack([
            (normalized[step] - normalized[0]).reshape(-1)
            for step in HOLDOUT_STEPS
        ])
        maximum_basis = fit_through_origin_basis(discovery[1:], max(RANKS))
        shape = tuple(int(value) for value in weights[0].shape)

        for rank in RANKS:
            basis = maximum_basis[:rank]
            discovery_coordinates = discovery @ basis.T
            predicted_coordinates = polynomial_predict(
                discovery_coordinates, discovery_progress, holdout_progress,
                int(plan["analysis"]["polynomial_degree"]),
            )
            predicted = predicted_coordinates @ basis
            polynomial_weight = pooled_recovery(holdout, predicted)
            polynomial_function = functional_recovery(
                holdout, predicted, activations[layer], shape
            )
            polynomial_accumulator.setdefault(rank, []).append(
                (polynomial_weight, polynomial_function)
            )
            rows.append({
                "row_type": "polynomial_endpoint_extrapolation",
                "layer": layer,
                "rank": rank,
                "polynomial_degree": 2,
                "endpoint_weight_recovery": polynomial_weight,
                "endpoint_functional_recovery": polynomial_function,
            })

            for horizon_index in plan["analysis"]["rolling_horizon_indices"]:
                pairs = list(zip(
                    late_sequence[:-horizon_index],
                    late_sequence[horizon_index:],
                    strict=True,
                ))
                targets = torch.stack([
                    (normalized[end] - normalized[start]).reshape(-1)
                    for start, end in pairs
                ])
                estimates = project_rows(targets, basis)
                weight_recovery = pooled_recovery(targets, estimates)
                functional = functional_recovery(
                    targets, estimates, activations[layer], shape
                )
                mean_updates = sum(end - start for start, end in pairs) / len(pairs)
                chord_accumulator.setdefault((rank, horizon_index), []).append(
                    (weight_recovery, functional, mean_updates)
                )
                rows.append({
                    "row_type": "rolling_holdout_secant",
                    "layer": layer,
                    "rank": rank,
                    "horizon_index": horizon_index,
                    "pairs": len(pairs),
                    "mean_updates": mean_updates,
                    "weight_recovery": weight_recovery,
                    "functional_recovery": functional,
                })
        del weights, normalized, discovery, holdout, maximum_basis
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    chord_summary: dict[str, Any] = {}
    passing_horizons: list[int] = []
    thresholds = plan["decision_rule"]["thresholds"]
    for (rank, horizon_index), values in sorted(chord_accumulator.items()):
        mean_weight = sum(value[0] for value in values) / len(values)
        mean_function = sum(value[1] for value in values) / len(values)
        mean_updates = round(sum(value[2] for value in values) / len(values))
        chord_summary[f"rank{rank}_horizon{horizon_index}"] = {
            "mean_updates": mean_updates,
            "mean_weight_recovery": mean_weight,
            "mean_functional_recovery": mean_function,
        }
        if (
            rank == 8
            and mean_weight >= thresholds["secant_weight_recovery"]
            and mean_function >= thresholds["secant_functional_recovery"]
        ):
            passing_horizons.append(mean_updates)
    onset = min(passing_horizons) if passing_horizons else None
    polynomial_summary = {
        str(rank): {
            "mean_weight_recovery": sum(value[0] for value in values) / len(values),
            "mean_functional_recovery": sum(value[1] for value in values) / len(values),
        }
        for rank, values in polynomial_accumulator.items()
    }
    primary_polynomial = polynomial_summary["8"]
    polynomial_pass = (
        primary_polynomial["mean_weight_recovery"]
        >= thresholds["polynomial_endpoint_weight_recovery"]
        and primary_polynomial["mean_functional_recovery"]
        >= thresholds["polynomial_endpoint_functional_recovery"]
    )
    classification = classify(onset, polynomial_pass)
    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "multiscale_path_rows.csv"
    result_path = args.output_dir / "multiscale_path_result.json"
    write_csv(rows_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": classification,
        "execution": {
            "parameter_updates": 0,
            "runtime_seconds": time.time() - started,
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "analyzer_sha256": file_sha256(Path(__file__)),
            "predictive_result_sha256": file_sha256(args.predictive_result),
            "trajectory_verification_sha256": file_sha256(args.trajectory_verification),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(manifest),
            "fixed_eval_indices_sha256": fixed_eval_indices_digest(fixed),
            "run_identity_sha256": run_identity,
        },
        "rank8_secant_onset_updates": onset,
        "chord_summary": chord_summary,
        "polynomial_endpoint_summary": polynomial_summary,
        "polynomial_rank8_pass": polynomial_pass,
        "decision_rule": plan["decision_rule"],
        "interpretation_contract": plan["interpretation_contract"],
        "authorization": plan["authorization"],
        "artifacts": {"rows": str(rows_path)},
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
