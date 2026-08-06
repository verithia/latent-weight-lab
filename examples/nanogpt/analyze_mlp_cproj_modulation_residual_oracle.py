#!/usr/bin/env python3
"""Test static modulation families against the function-critical c_proj residual.

The diagnostic reconstructs the preregistered rank-4 scheduled endpoint for
late c_proj layers, computes the exact terminal residual, and gives several
static modulation families an oracle functional least-squares fit on one fixed
training activation window.  It then evaluates the resulting full model on the
unchanged fixed validation window.  No language-model parameter is optimized.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import model_from_snapshot
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import file_sha256
from examples.nanogpt.analyze_mlp_cproj_coadapted_orbit_geometry import buffer_name, git_commit, load_snapshot
from examples.nanogpt.analyze_mlp_cproj_multiscale_path import cumulative_lr_coordinate, polynomial_predict
from examples.nanogpt.analyze_mlp_cproj_polynomial_oracle_ce import restore_radius
from examples.nanogpt.analyze_mlp_cproj_predictive_manifold import (
    fit_through_origin_basis,
    terminal_post_gelu_activations,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.train import (
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)
from examples.nanogpt.verify_full_state_functional_replay import evaluate_validation_ce


PLAN_SCHEMA = "mai_124m_mlp_cproj_modulation_residual_oracle_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_modulation_residual_oracle_result_v1"
LAYERS = tuple(range(8, 12))
DISCOVERY_STEPS = (
    0, 99, 198, 297, 396, 495, 594, 693, 792, 891,
    990, 1089, 1188, 1287, 1386, 1485, 1584, 1683, 1782,
)
TERMINAL_STEP = 2373
POLYNOMIAL_RANK = 4
CANDIDATES = (
    "scheduled_rank4",
    "paper_literal_dc",
    "output_additive",
    "hidden_additive",
    "two_way_additive",
    "output_gain",
    "hidden_gain",
    "two_way_gain",
    "exact_residual",
)


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected modulation-residual plan schema")
    analysis = plan.get("analysis", {})
    expected = {
        "parameter_updates": 0,
        "layers": list(LAYERS),
        "discovery_steps": list(DISCOVERY_STEPS),
        "terminal_step": TERMINAL_STEP,
        "polynomial_rank": POLYNOMIAL_RANK,
        "polynomial_degree": 2,
        "activation_rows": 2048,
        "fit_split": "fixed_train",
        "score_split": "fixed_validation_400_batches",
        "restore_terminal_radius": True,
        "families": list(CANDIDATES),
        "cg_iterations": 32,
        "alternating_iterations": 4,
        "ridge_ratio": 1e-6,
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            raise ValueError(f"modulation-residual analysis field changed: {key}")
    thresholds = plan.get("decision_rule", {}).get("thresholds", {})
    if thresholds != {
        "exact_replay_absolute_tolerance_ce": 0.005,
        "scheduled_rank4_replay_absolute_tolerance_ce": 0.005,
        "candidate_maximum_validation_ce_gap": 0.005,
    }:
        raise ValueError("modulation-residual thresholds changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_modulation_oracle") is not True:
        raise ValueError("modulation oracle is not authorized")
    for key in (
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
        "use_learned_basis_in_candidate",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def cg_solve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    ridge: float,
    iterations: int,
) -> torch.Tensor:
    """Deterministic conjugate-gradient solve for a regularized normal system."""
    x = torch.zeros_like(rhs)
    residual = rhs - (matvec(x) + ridge * x)
    direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    initial_sq = residual_sq.clamp_min(1e-30)
    for _ in range(iterations):
        action = matvec(direction) + ridge * direction
        denominator = torch.dot(direction, action)
        if not torch.isfinite(denominator) or denominator.abs() <= 1e-30:
            break
        alpha = residual_sq / denominator
        x = x + alpha * direction
        next_residual = residual - alpha * action
        next_sq = torch.dot(next_residual, next_residual)
        if not torch.isfinite(next_sq):
            raise FloatingPointError("nonfinite conjugate-gradient residual")
        if next_sq <= initial_sq * 1e-12:
            residual = next_residual
            break
        direction = next_residual + (next_sq / residual_sq.clamp_min(1e-30)) * direction
        residual = next_residual
        residual_sq = next_sq
    if not torch.isfinite(x).all():
        raise FloatingPointError("nonfinite conjugate-gradient solution")
    return x


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1e-20)


def fit_paper_literal_dc(
    hidden: torch.Tensor, target_output: torch.Tensor, weight: torch.Tensor, **_: Any
) -> torch.Tensor:
    signal = hidden.sum(dim=1)
    coefficient = _safe_ratio(
        (target_output * signal[:, None]).sum(),
        signal.square().sum() * target_output.shape[1],
    )
    return torch.full_like(weight, coefficient)


def fit_output_additive(
    hidden: torch.Tensor, target_output: torch.Tensor, weight: torch.Tensor, **_: Any
) -> torch.Tensor:
    signal = hidden.sum(dim=1)
    coefficients = _safe_ratio(
        (target_output * signal[:, None]).sum(dim=0),
        signal.square().sum(),
    )
    return coefficients[:, None].expand_as(weight).clone()


def fit_hidden_additive(
    hidden: torch.Tensor,
    target_output: torch.Tensor,
    weight: torch.Tensor,
    *,
    ridge_ratio: float,
    cg_iterations: int,
    **_: Any,
) -> torch.Tensor:
    target = target_output.mean(dim=1)
    rhs = hidden.T @ target
    scale = float(hidden.square().sum(dim=0).mean().clamp_min(1e-20))
    coefficients = cg_solve(
        lambda value: hidden.T @ (hidden @ value),
        rhs,
        ridge=ridge_ratio * scale,
        iterations=cg_iterations,
    )
    return coefficients[None, :].expand_as(weight).clone()


def fit_output_gain(
    hidden: torch.Tensor, target_output: torch.Tensor, weight: torch.Tensor, **_: Any
) -> torch.Tensor:
    base_output = hidden @ weight.T
    coefficients = _safe_ratio(
        (target_output * base_output).sum(dim=0),
        base_output.square().sum(dim=0),
    )
    return coefficients[:, None] * weight


def fit_hidden_gain(
    hidden: torch.Tensor,
    target_output: torch.Tensor,
    weight: torch.Tensor,
    *,
    ridge_ratio: float,
    cg_iterations: int,
    **_: Any,
) -> torch.Tensor:
    def forward(value: torch.Tensor) -> torch.Tensor:
        return (hidden * value[None, :]) @ weight.T

    def adjoint(value: torch.Tensor) -> torch.Tensor:
        return ((value @ weight) * hidden).sum(dim=0)

    rhs = adjoint(target_output)
    scale = float(
        (hidden.square().sum(dim=0) * weight.square().sum(dim=0))
        .mean()
        .clamp_min(1e-20)
    )
    coefficients = cg_solve(
        lambda value: adjoint(forward(value)),
        rhs,
        ridge=ridge_ratio * scale,
        iterations=cg_iterations,
    )
    return weight * coefficients[None, :]


def fit_two_way_additive(
    hidden: torch.Tensor,
    target_output: torch.Tensor,
    weight: torch.Tensor,
    *,
    ridge_ratio: float,
    cg_iterations: int,
    alternating_iterations: int,
    **_: Any,
) -> torch.Tensor:
    hidden_coefficients = torch.zeros(weight.shape[1], device=weight.device)
    output_coefficients = torch.zeros(weight.shape[0], device=weight.device)
    signal = hidden.sum(dim=1)
    scale = float(hidden.square().sum(dim=0).mean().clamp_min(1e-20))
    for _ in range(alternating_iterations):
        hidden_output = hidden @ hidden_coefficients
        output_coefficients = _safe_ratio(
            ((target_output - hidden_output[:, None]) * signal[:, None]).sum(dim=0),
            signal.square().sum(),
        )
        target = (target_output - signal[:, None] * output_coefficients).mean(dim=1)
        hidden_coefficients = cg_solve(
            lambda value: hidden.T @ (hidden @ value),
            hidden.T @ target,
            ridge=ridge_ratio * scale,
            iterations=cg_iterations,
        )
    return (
        output_coefficients[:, None] + hidden_coefficients[None, :]
    ).expand_as(weight).clone()


def fit_two_way_gain(
    hidden: torch.Tensor,
    target_output: torch.Tensor,
    weight: torch.Tensor,
    *,
    ridge_ratio: float,
    cg_iterations: int,
    alternating_iterations: int,
    **_: Any,
) -> torch.Tensor:
    hidden_coefficients = torch.zeros(weight.shape[1], device=weight.device)
    output_coefficients = torch.zeros(weight.shape[0], device=weight.device)
    base_output = hidden @ weight.T

    def hidden_forward(value: torch.Tensor) -> torch.Tensor:
        return (hidden * value[None, :]) @ weight.T

    def hidden_adjoint(value: torch.Tensor) -> torch.Tensor:
        return ((value @ weight) * hidden).sum(dim=0)

    scale = float(
        (hidden.square().sum(dim=0) * weight.square().sum(dim=0))
        .mean()
        .clamp_min(1e-20)
    )
    for _ in range(alternating_iterations):
        hidden_output = hidden_forward(hidden_coefficients)
        output_coefficients = _safe_ratio(
            ((target_output - hidden_output) * base_output).sum(dim=0),
            base_output.square().sum(dim=0),
        )
        target = target_output - base_output * output_coefficients[None, :]
        hidden_coefficients = cg_solve(
            lambda value: hidden_adjoint(hidden_forward(value)),
            hidden_adjoint(target),
            ridge=ridge_ratio * scale,
            iterations=cg_iterations,
        )
    return weight * (
        output_coefficients[:, None] + hidden_coefficients[None, :]
    )


FITTERS: dict[str, Callable[..., torch.Tensor]] = {
    "paper_literal_dc": fit_paper_literal_dc,
    "output_additive": fit_output_additive,
    "hidden_additive": fit_hidden_additive,
    "two_way_additive": fit_two_way_additive,
    "output_gain": fit_output_gain,
    "hidden_gain": fit_hidden_gain,
    "two_way_gain": fit_two_way_gain,
}


def recovery(target: torch.Tensor, prediction: torch.Tensor) -> float:
    denominator = target.double().square().sum().clamp_min(1e-30)
    return float(1.0 - (target.double() - prediction.double()).square().sum() / denominator)


def classify(rows: list[dict[str, Any]], replay_valid: bool, rank4_valid: bool, threshold: float) -> str:
    if not replay_valid or not rank4_valid:
        return "INVALID_ORACLE_REPLAY"
    by_name = {row["candidate"]: row for row in rows}
    if by_name["paper_literal_dc"]["validation_ce_gap"] <= threshold:
        return "PAPER_IDENTITY_MODULATION_SUFFICIENT"
    structured = [name for name in FITTERS if name != "paper_literal_dc"]
    if any(by_name[name]["validation_ce_gap"] <= threshold for name in structured):
        return "STATIC_STRUCTURED_MODULATION_SUFFICIENT"
    return "STATIC_MODULATION_INSUFFICIENT_REQUIRES_ADAPTIVE_STATE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--polynomial-result", type=Path, required=True)
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
        args.polynomial_result: identity["polynomial_result_sha256"],
        args.trajectory_verification: identity["trajectory_verification_sha256"],
        args.config: identity["config_sha256"],
    }
    for path, expected in pinned.items():
        if file_sha256(path) != expected:
            raise ValueError(f"pinned artifact SHA-256 mismatch: {path}")
    for relative, expected in identity["supporting_source_sha256"].items():
        if file_sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"supporting-source SHA-256 mismatch: {relative}")

    polynomial_result = json.loads(args.polynomial_result.read_text())
    if polynomial_result.get("classification") != "TASK_ADAPTIVE_RESIDUAL_FUNCTIONALLY_NECESSARY":
        raise ValueError("polynomial oracle does not authorize residual decomposition")
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
        step: load_snapshot(args.snapshot_dir / f"step_{step:06d}.pt", hashes[str(step)], run_identity)
        for step in steps
    }
    analysis = plan["analysis"]
    activations = terminal_post_gelu_activations(
        snapshots[TERMINAL_STEP], config, fixed, int(analysis["activation_rows"]), args.device
    )
    terminal_progress = torch.tensor(
        [cumulative_lr_coordinate(TERMINAL_STEP, config)], device=args.device
    )
    discovery_progress = torch.tensor(
        [cumulative_lr_coordinate(step, config) for step in DISCOVERY_STEPS], device=args.device
    )
    predicted: dict[int, torch.Tensor] = {}
    exact: dict[int, torch.Tensor] = {}
    fitted: dict[str, dict[int, torch.Tensor]] = {name: {} for name in FITTERS}
    layer_metrics: list[dict[str, Any]] = []
    for layer in LAYERS:
        weights = {
            step: snapshots[step]["buffers"][buffer_name(layer, "weight")].float().to(args.device)
            for step in steps
        }
        initial_norm = weights[0].norm().clamp_min(1e-30)
        normalized = {step: value * (initial_norm / value.norm().clamp_min(1e-30)) for step, value in weights.items()}
        discovery = torch.stack([(normalized[step] - normalized[0]).reshape(-1) for step in DISCOVERY_STEPS])
        basis = fit_through_origin_basis(discovery[1:], POLYNOMIAL_RANK)
        coordinates = discovery @ basis.T
        predicted_coordinate = polynomial_predict(coordinates, discovery_progress, terminal_progress, 2)
        predicted_normalized = normalized[0] + (predicted_coordinate @ basis).reshape_as(normalized[0])
        exact[layer] = weights[TERMINAL_STEP].clone()
        predicted[layer] = restore_radius(predicted_normalized, exact[layer].norm())
        residual = exact[layer] - predicted[layer]
        hidden = activations[layer].float()
        target_output = hidden @ residual.T
        for name, fitter in FITTERS.items():
            delta = fitter(
                hidden, target_output, predicted[layer],
                ridge_ratio=float(analysis["ridge_ratio"]),
                cg_iterations=int(analysis["cg_iterations"]),
                alternating_iterations=int(analysis["alternating_iterations"]),
            )
            candidate = restore_radius(predicted[layer] + delta, exact[layer].norm())
            fitted[name][layer] = candidate
            actual_delta = candidate - predicted[layer]
            layer_metrics.append({
                "candidate": name,
                "layer": layer,
                "weight_recovery": recovery(residual, actual_delta),
                "fit_functional_recovery": recovery(target_output, hidden @ actual_delta.T),
                "candidate_weight_norm": float(candidate.norm()),
                "exact_weight_norm": float(exact[layer].norm()),
            })

    model = model_from_snapshot(snapshots[TERMINAL_STEP], args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    eval_args = SimpleNamespace(**config)
    eval_args.device = args.device
    eval_args._ptdtype = dtype
    ctx = contextlib.nullcontext() if "cuda" not in args.device else torch.amp.autocast(device_type="cuda", dtype=dtype)
    started = time.time()

    def install(values: dict[int, torch.Tensor]) -> None:
        with torch.no_grad():
            for layer in LAYERS:
                model.transformer.h[layer].mlp.c_proj.weight.copy_(values[layer])

    exact_replay_ce = evaluate_validation_ce(model, data_dir=Path(config["data_dir"]), args=eval_args, indices=fixed["val"], ctx=ctx)
    rows: list[dict[str, Any]] = []
    candidate_weights: dict[str, dict[int, torch.Tensor]] = {
        "scheduled_rank4": predicted,
        **fitted,
        "exact_residual": exact,
    }
    for name in CANDIDATES:
        install(candidate_weights[name])
        validation_ce = evaluate_validation_ce(
            model, data_dir=Path(config["data_dir"]), args=eval_args, indices=fixed["val"], ctx=ctx
        )
        rows.append({
            "candidate": name,
            "validation_ce": validation_ce,
            "validation_ce_gap": validation_ce - exact_replay_ce,
        })
    install(exact)
    threshold = float(plan["decision_rule"]["thresholds"]["candidate_maximum_validation_ce_gap"])
    replay_tolerance = float(plan["decision_rule"]["thresholds"]["exact_replay_absolute_tolerance_ce"])
    rank4_tolerance = float(plan["decision_rule"]["thresholds"]["scheduled_rank4_replay_absolute_tolerance_ce"])
    expected_exact = float(polynomial_result["execution"]["exact_replay_ce"])
    expected_rank4 = float(polynomial_result["rank_results"]["4"]["validation_ce"])
    replay_valid = abs(exact_replay_ce - expected_exact) <= replay_tolerance
    rank4_row = next(row for row in rows if row["candidate"] == "scheduled_rank4")
    rank4_valid = abs(float(rank4_row["validation_ce"]) - expected_rank4) <= rank4_tolerance
    classification = classify(rows, replay_valid, rank4_valid, threshold)
    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "modulation_residual_oracle_rows.csv"
    layer_path = args.output_dir / "modulation_residual_oracle_layer_metrics.csv"
    result_path = args.output_dir / "modulation_residual_oracle_result.json"
    write_csv(rows_path, rows)
    write_csv(layer_path, layer_metrics)
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
            "polynomial_result_sha256": file_sha256(args.polynomial_result),
            "trajectory_verification_sha256": file_sha256(args.trajectory_verification),
            "config_sha256": file_sha256(args.config),
            "dataset_manifest_sha256": file_sha256(manifest),
            "fixed_eval_indices_sha256": fixed_eval_indices_digest(fixed),
            "run_identity_sha256": run_identity,
        },
        "replay": {
            "exact_validation_ce": exact_replay_ce,
            "expected_exact_validation_ce": expected_exact,
            "exact_replay_valid": replay_valid,
            "rank4_validation_ce": rank4_row["validation_ce"],
            "expected_rank4_validation_ce": expected_rank4,
            "rank4_replay_valid": rank4_valid,
        },
        "rows": rows,
        "aggregate_fit_metrics": {
            name: {
                "mean_weight_recovery": sum(row["weight_recovery"] for row in layer_metrics if row["candidate"] == name) / len(LAYERS),
                "mean_fit_functional_recovery": sum(row["fit_functional_recovery"] for row in layer_metrics if row["candidate"] == name) / len(LAYERS),
            }
            for name in FITTERS
        },
        "decision_rule": plan["decision_rule"],
        "interpretation_contract": plan["interpretation_contract"],
        "authorization": plan["authorization"],
        "artifacts": {"rows": str(rows_path), "layer_metrics": str(layer_path)},
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
