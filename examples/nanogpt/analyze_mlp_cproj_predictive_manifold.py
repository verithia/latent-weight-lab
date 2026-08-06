#!/usr/bin/env python3
"""Chronological holdout test of the accepted co-adapted c_proj path.

This is a zero-update diagnostic.  It removes the analytically known scalar
weight-decay radius, fits affine endpoint and instantaneous-tangent oracles on
the discovery prefix only, and evaluates later snapshots in raw weight space
and under a fixed terminal post-GELU functional metric.  Learned PCA bases are
diagnostic upper bounds only; they are never candidate parameters.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    ActivationCollector,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
)
from examples.nanogpt.analyze_mlp_cproj_coadapted_orbit_geometry import (
    buffer_name,
    git_commit,
    load_snapshot,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.muon_matched_givens import apply_givens_flow
from examples.nanogpt.train import (
    TokenBatchSource,
    cosine_lr,
    fixed_eval_indices_digest,
    get_batch,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)


PLAN_SCHEMA = "mai_124m_mlp_cproj_predictive_manifold_v2_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_predictive_manifold_v2_result_v1"
LATE_LAYERS = tuple(range(8, 12))
PRIMARY_RANK = 8
SUPPORTING_SOURCES = (
    "examples/nanogpt/analyze_mlp_activation_update_alignment.py",
    "examples/nanogpt/analyze_mlp_cproj_coadapted_orbit_geometry.py",
    "examples/nanogpt/model.py",
    "examples/nanogpt/muon_matched_givens.py",
    "examples/nanogpt/train.py",
)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected predictive-manifold plan schema")
    analysis = plan.get("analysis", {})
    if analysis.get("parameter_updates") != 0:
        raise ValueError("predictive-manifold analysis must perform zero updates")
    if analysis.get("layers") != list(LATE_LAYERS):
        raise ValueError("late-layer set changed")
    if analysis.get("discovery_steps") != [
        0, 99, 198, 297, 396, 495, 594, 693, 792, 891,
        990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782,
    ]:
        raise ValueError("discovery split changed")
    if analysis.get("holdout_steps") != [1881, 1980, 2079, 2178, 2277, 2373]:
        raise ValueError("holdout split changed")
    if analysis.get("ranks") != [1, 2, 4, 8, 16]:
        raise ValueError("rank controls changed")
    if analysis.get("learned_basis_role") != "diagnostic_oracle_only":
        raise ValueError("learned basis cannot become a candidate")
    expected = {
        "normalization_schedule_max_relative_error": 0.03,
        "last_step_replay_max_relative_error": 3e-5,
        "rank8_holdout_endpoint_weight_recovery": 0.80,
        "rank8_holdout_endpoint_functional_recovery": 0.80,
        "rank8_holdout_tangent_functional_recovery": 0.25,
        "rank8_endpoint_functional_minus_weight_recovery": 0.10,
    }
    if plan.get("decision_rule", {}).get("thresholds") != expected:
        raise ValueError("predictive-manifold thresholds changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_predictive_analysis") is not True:
        raise ValueError("zero-update predictive analysis is not authorized")
    for key in (
        "use_learned_basis_in_candidate",
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def inverse_givens_flow(
    values: torch.Tensor,
    angles: torch.Tensor,
    permutations: torch.Tensor,
    inverse_permutations: torch.Tensor,
) -> torch.Tensor:
    """Invert a staged right-side Givens flow exactly in reverse order."""
    result = values
    for stage in range(int(angles.shape[0]) - 1, -1, -1):
        result = apply_givens_flow(
            result,
            -angles[stage : stage + 1],
            permutations[stage : stage + 1],
            inverse_permutations[stage : stage + 1],
        )
    return result


def fit_through_origin_basis(rows: torch.Tensor, rank: int) -> torch.Tensor:
    """Fit a right PCA basis via the compact sample Gram matrix."""
    if rows.ndim != 2 or not 0 < rank <= min(rows.shape):
        raise ValueError("invalid rows/rank for through-origin PCA")
    gram = rows.float() @ rows.float().T
    eigenvalues, eigenvectors = torch.linalg.eigh((gram + gram.T) * 0.5)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    values = eigenvalues.index_select(0, order).clamp_min(1e-30)
    vectors = eigenvectors.index_select(1, order)
    basis = (vectors.T @ rows.float()) / values.sqrt().unsqueeze(1)
    return torch.linalg.qr(basis.T, mode="reduced").Q.T.contiguous()


def project_rows(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (rows.float() @ basis.float().T) @ basis.float()


def pooled_recovery(targets: torch.Tensor, estimates: torch.Tensor) -> float:
    denominator = targets.float().square().sum().clamp_min(1e-30)
    residual = (targets.float() - estimates.float()).square().sum()
    return float(1.0 - residual / denominator)


def functional_recovery(
    targets: torch.Tensor,
    estimates: torch.Tensor,
    activations: torch.Tensor,
    shape: tuple[int, int],
) -> float:
    target_energy = torch.zeros((), device=activations.device)
    residual_energy = torch.zeros((), device=activations.device)
    for target, estimate in zip(targets, estimates, strict=True):
        target_matrix = target.reshape(shape)
        residual_matrix = (target - estimate).reshape(shape)
        target_energy += (activations @ target_matrix.T).square().sum()
        residual_energy += (activations @ residual_matrix.T).square().sum()
    return float(1.0 - residual_energy / target_energy.clamp_min(1e-30))


def classify(gates: dict[str, bool]) -> str:
    if not gates["normalization_schedule_valid"] or not gates["last_step_replay_valid"]:
        return "INVALID_MECHANICAL_RECONSTRUCTION"
    if gates["endpoint_weight_predictive"] and gates["tangent_function_predictive"]:
        return "PREDICTIVE_ENDPOINT_AND_CAUSAL_TANGENT_ORACLE"
    if gates["endpoint_weight_predictive"]:
        return "SMOOTH_ENDPOINT_WITH_NONTRANSPORTABLE_TANGENTS"
    if gates["endpoint_function_predictive"]:
        return "FUNCTION_SPACE_ENDPOINT_WITHOUT_WEIGHT_CHART"
    if gates["tangent_function_predictive"]:
        return "LOCAL_FUNCTIONAL_TANGENT_WITHOUT_ENDPOINT_CHART"
    return "NO_PREDICTIVE_FIXED_AFFINE_CHART"


def cumulative_decay_scale(step: int, config: dict[str, Any]) -> float:
    schedule = SimpleNamespace(**config)
    log_scale = 0.0
    for iteration in range(step):
        factor = 1.0 - cosine_lr(iteration, schedule) * float(config["weight_decay"])
        if factor <= 0.0:
            raise ValueError("nonpositive scheduled decay factor")
        log_scale += math.log(factor)
    return math.exp(log_scale)


def reconstruct_last_step(
    snapshot: dict[str, Any],
    layer: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    post = snapshot["buffers"][buffer_name(layer, "weight")].float().to(device)
    decay = 1.0 - learning_rate * weight_decay
    if decay <= 0.0:
        raise ValueError("nonpositive last-step decay")
    rotated = post / decay
    residual_angles = snapshot["buffers"][
        buffer_name(layer, "residual_last_angles")
    ].float().to(device)
    residual_permutations = snapshot["buffers"][
        buffer_name(layer, "residual_selected_permutations")
    ].long().to(device)
    residual_inverse = snapshot["buffers"][
        buffer_name(layer, "residual_selected_inverse_permutations")
    ].long().to(device)
    after_parent = inverse_givens_flow(
        rotated, residual_angles, residual_permutations, residual_inverse
    )
    parent_angles = snapshot["buffers"][buffer_name(layer, "last_angles")].float().to(device)
    parent_permutations = snapshot["buffers"][
        buffer_name(layer, "selected_permutations")
    ].long().to(device)
    parent_inverse = snapshot["buffers"][
        buffer_name(layer, "selected_inverse_permutations")
    ].long().to(device)
    before = inverse_givens_flow(
        after_parent, parent_angles, parent_permutations, parent_inverse
    )
    replay = apply_givens_flow(
        before, parent_angles, parent_permutations, parent_inverse
    )
    replay = apply_givens_flow(
        replay, residual_angles, residual_permutations, residual_inverse
    ) * decay
    relative_error = float(
        (replay - post).norm() / post.norm().clamp_min(1e-30)
    )
    return before, post, relative_error


def terminal_post_gelu_activations(
    snapshot: dict[str, Any],
    config: dict[str, Any],
    fixed_indices: dict[str, torch.Tensor],
    sample_cap: int,
    device: str,
) -> dict[int, torch.Tensor]:
    model = model_from_snapshot(snapshot, device)
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]
    ctx = (
        contextlib.nullcontext()
        if "cuda" not in device
        else torch.amp.autocast(device_type="cuda", dtype=dtype)
    )
    collector = ActivationCollector(model, list(LATE_LAYERS), sample_cap)
    source = TokenBatchSource(Path(config["data_dir"]))
    try:
        model.prepare_block_fht_cache(dtype=dtype)
        x, _y = get_batch(
            Path(config["data_dir"]),
            "train",
            int(config["eval_batch_size"]),
            int(config["block_size"]),
            device,
            indices=fixed_indices["train"][0],
            source=source,
        )
        with torch.no_grad(), ctx:
            model(x, None)
        if not collector.complete():
            raise RuntimeError("fixed training batch did not fill activation sample cap")
        return {
            layer: collector.tensor(layer, "post_gelu").float().to(device)
            for layer in LATE_LAYERS
        }
    finally:
        collector.close()
        model.flush_block_fht_cache()
        del model
        if "cuda" in device:
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trajectory-verification", type=Path, required=True)
    parser.add_argument("--orbit-v2-result", type=Path, required=True)
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
        args.trajectory_verification: identity["trajectory_verification_sha256"],
        args.orbit_v2_result: identity["orbit_v2_result_sha256"],
        args.config: identity["config_sha256"],
    }
    for path, expected in pinned.items():
        if file_sha256(path) != expected:
            raise ValueError(f"pinned artifact SHA-256 mismatch: {path}")
    if set(identity["supporting_source_sha256"]) != set(SUPPORTING_SOURCES):
        raise ValueError("supporting-source inventory changed")
    for relative in SUPPORTING_SOURCES:
        path = REPO_ROOT / relative
        if file_sha256(path) != identity["supporting_source_sha256"][relative]:
            raise ValueError(f"supporting-source SHA-256 mismatch: {relative}")
    verification = json.loads(args.trajectory_verification.read_text())
    if verification.get("classification") != "ACCEPTED_COADAPTED_LATE_CPROJ_FULL_STATE_TRAJECTORY":
        raise ValueError("trajectory verification is not accepted")
    orbit_v2 = json.loads(args.orbit_v2_result.read_text())
    if orbit_v2.get("classification") != "ADAPTIVE_RIGHT_ORBIT_WITH_MOVING_SUPPORT_NOT_LAYER_LOCALIZED":
        raise ValueError("source orbit result has an unexpected classification")
    if orbit_v2.get("authorization", {}).get("functional_metric_path_analysis") is not True:
        raise ValueError("source orbit result did not authorize this analysis")
    if verification["identity"]["dataset_manifest_sha256"] != identity["dataset_manifest_sha256"]:
        raise ValueError("trajectory dataset identity changed")
    config = json.loads(args.config.read_text())
    if verification["identity"]["run_identity_sha256"] != identity["run_identity_sha256"]:
        raise ValueError("accepted run identity changed")
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
        raise ValueError("fixed evaluation indices SHA-256 mismatch")

    discovery_steps = plan["analysis"]["discovery_steps"]
    holdout_steps = plan["analysis"]["holdout_steps"]
    all_steps = discovery_steps + holdout_steps
    hashes = verification["inventory"]["snapshot_sha256_by_step"]
    run_identity = verification["identity"]["run_identity_sha256"]
    snapshots: dict[int, dict[str, Any]] = {}
    for step in all_steps:
        snapshots[step] = load_snapshot(
            args.snapshot_dir / f"step_{step:06d}.pt",
            hashes[str(step)],
            run_identity,
        )

    started = time.time()
    terminal = snapshots[holdout_steps[-1]]
    activations = terminal_post_gelu_activations(
        terminal,
        config,
        fixed,
        int(plan["analysis"]["functional_metric"]["activation_rows"]),
        args.device,
    )
    rows: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    maximum_schedule_error = 0.0
    maximum_replay_error = 0.0
    primary_layer_metrics: list[dict[str, float]] = []
    schedule = SimpleNamespace(**config)

    for layer in LATE_LAYERS:
        weights = {
            step: snapshots[step]["buffers"][buffer_name(layer, "weight")]
            .float()
            .to(args.device)
            for step in all_steps
        }
        initial_norm = weights[0].norm().clamp_min(1e-30)
        normalized: dict[int, torch.Tensor] = {}
        for step in all_steps:
            observed_ratio = float(weights[step].norm() / initial_norm)
            expected_ratio = cumulative_decay_scale(step, config)
            schedule_error = abs(observed_ratio - expected_ratio) / max(expected_ratio, 1e-30)
            maximum_schedule_error = max(maximum_schedule_error, schedule_error)
            normalized[step] = weights[step] * (initial_norm / weights[step].norm().clamp_min(1e-30))
            rows.append({
                "row_type": "radius_schedule",
                "layer": layer,
                "step": step,
                "observed_norm_ratio": observed_ratio,
                "expected_decay_ratio": expected_ratio,
                "relative_error": schedule_error,
            })

        endpoint_discovery = torch.stack([
            (normalized[step] - normalized[0]).reshape(-1)
            for step in discovery_steps[1:]
        ])
        endpoint_holdout = torch.stack([
            (normalized[step] - normalized[0]).reshape(-1)
            for step in holdout_steps
        ])
        tangent_discovery: list[torch.Tensor] = []
        tangent_holdout: list[torch.Tensor] = []
        for step in all_steps[1:]:
            lr = cosine_lr(step - 1, schedule)
            before, post, replay_error = reconstruct_last_step(
                snapshots[step], layer, lr, float(config["weight_decay"]), args.device
            )
            maximum_replay_error = max(maximum_replay_error, replay_error)
            before = before * (initial_norm / before.norm().clamp_min(1e-30))
            post = post * (initial_norm / post.norm().clamp_min(1e-30))
            tangent = (post - before).reshape(-1)
            if step in discovery_steps:
                tangent_discovery.append(tangent)
            else:
                tangent_holdout.append(tangent)
            rows.append({
                "row_type": "last_step_replay",
                "layer": layer,
                "step": step,
                "learning_rate": lr,
                "relative_error": replay_error,
            })
        tangent_discovery_tensor = torch.stack(tangent_discovery)
        tangent_holdout_tensor = torch.stack(tangent_holdout)

        maximum_rank = max(plan["analysis"]["ranks"])
        endpoint_basis = fit_through_origin_basis(endpoint_discovery, maximum_rank)
        tangent_basis = fit_through_origin_basis(tangent_discovery_tensor, maximum_rank)
        shape = tuple(int(value) for value in weights[0].shape)
        layer_primary: dict[str, float] = {}
        for rank in plan["analysis"]["ranks"]:
            endpoint_selected = endpoint_basis[:rank]
            tangent_selected = tangent_basis[:rank]
            endpoint_estimate = project_rows(endpoint_holdout, endpoint_selected)
            tangent_estimate = project_rows(tangent_holdout_tensor, tangent_selected)
            tangent_from_endpoint = project_rows(tangent_holdout_tensor, endpoint_selected)
            endpoint_weight = pooled_recovery(endpoint_holdout, endpoint_estimate)
            tangent_weight = pooled_recovery(tangent_holdout_tensor, tangent_estimate)
            endpoint_function = functional_recovery(
                endpoint_holdout, endpoint_estimate, activations[layer], shape
            )
            tangent_function = functional_recovery(
                tangent_holdout_tensor, tangent_estimate, activations[layer], shape
            )
            tangent_endpoint_function = functional_recovery(
                tangent_holdout_tensor, tangent_from_endpoint, activations[layer], shape
            )
            rows.append({
                "row_type": "chronological_holdout",
                "layer": layer,
                "rank": rank,
                "endpoint_weight_recovery": endpoint_weight,
                "endpoint_functional_recovery": endpoint_function,
                "endpoint_functional_minus_weight_recovery": endpoint_function - endpoint_weight,
                "tangent_weight_recovery": tangent_weight,
                "tangent_functional_recovery": tangent_function,
                "tangent_functional_recovery_in_endpoint_basis": tangent_endpoint_function,
            })
            if rank == PRIMARY_RANK:
                layer_primary = {
                    "endpoint_weight_recovery": endpoint_weight,
                    "endpoint_functional_recovery": endpoint_function,
                    "endpoint_functional_minus_weight_recovery": endpoint_function - endpoint_weight,
                    "tangent_weight_recovery": tangent_weight,
                    "tangent_functional_recovery": tangent_function,
                    "tangent_functional_recovery_in_endpoint_basis": tangent_endpoint_function,
                }
        primary_layer_metrics.append(layer_primary)
        aggregates[str(layer)] = layer_primary
        del weights, normalized, endpoint_discovery, endpoint_holdout
        del tangent_discovery_tensor, tangent_holdout_tensor, endpoint_basis, tangent_basis
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    primary = {
        key: sum(row[key] for row in primary_layer_metrics) / len(primary_layer_metrics)
        for key in primary_layer_metrics[0]
    }
    thresholds = plan["decision_rule"]["thresholds"]
    gates = {
        "normalization_schedule_valid": maximum_schedule_error <= thresholds["normalization_schedule_max_relative_error"],
        "last_step_replay_valid": maximum_replay_error <= thresholds["last_step_replay_max_relative_error"],
        "endpoint_weight_predictive": primary["endpoint_weight_recovery"] >= thresholds["rank8_holdout_endpoint_weight_recovery"],
        "endpoint_function_predictive": primary["endpoint_functional_recovery"] >= thresholds["rank8_holdout_endpoint_functional_recovery"],
        "tangent_function_predictive": primary["tangent_functional_recovery"] >= thresholds["rank8_holdout_tangent_functional_recovery"],
        "functional_metric_materially_changes_fit": primary["endpoint_functional_minus_weight_recovery"] >= thresholds["rank8_endpoint_functional_minus_weight_recovery"],
    }
    classification = classify(gates)
    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "predictive_manifold_rows.csv"
    result_path = args.output_dir / "predictive_manifold_result.json"
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
            "trajectory_verification_sha256": file_sha256(args.trajectory_verification),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(manifest),
            "fixed_eval_indices_sha256": fixed_eval_indices_digest(fixed),
            "run_identity_sha256": run_identity,
        },
        "mechanical_validation": {
            "maximum_normalization_schedule_relative_error": maximum_schedule_error,
            "maximum_last_step_replay_relative_error": maximum_replay_error,
        },
        "rank8_mean_holdout": primary,
        "rank8_by_layer": aggregates,
        "gates": gates,
        "decision_rule": plan["decision_rule"],
        "interpretation_contract": plan["interpretation_contract"],
        "authorization": plan["authorization"],
        "artifacts": {"rows": str(rows_path)},
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
