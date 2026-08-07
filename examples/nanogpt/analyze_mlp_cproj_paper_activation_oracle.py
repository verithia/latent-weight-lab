#!/usr/bin/env python3
"""Upper-bound the paper's elementwise mapping activation on c_proj motion.

This is a zero-update representability oracle.  It uses a fixed BlockFHT map
and a signed scale-matched tanh activation initialized exactly at the captured
step-zero weight.  Per-state latent coordinates and future tangent
coefficients are granted oracle least-squares fits.  Passing is therefore
necessary, not sufficient, for a causal compact mapper.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_mlp_cproj_same_run_optimizer_state_transport import (
    load_snapshot,
    parameter_name,
    terminal_post_gelu_activations,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.train import (
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)
from latent_weight_lab.block_fht import block_fht_grad_latent, block_fht_slice


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_mlp_cproj_paper_activation_oracle_plan_v1"
TIGHT_PLAN_SCHEMA = "mai_124m_mlp_cproj_paper_activation_oracle_plan_v2_tight"
RESULT_SCHEMA = "mai_124m_mlp_cproj_paper_activation_oracle_result_v1"
LAYERS = (8, 9, 10, 11)
PROBE_STEPS = (0, 593, 1187, 1484, 1781, 2078)
HELDOUT_PROBE_STEPS = (1781, 2078)
FUTURE_STEP_BY_PROBE = {
    0: 99,
    593: 891,
    1187: 1485,
    1484: 1782,
    1781: 2079,
    2078: 2373,
}


def activation_scale(
    initial_weight: torch.Tensor, multiplier: float = 2.0
) -> torch.Tensor:
    """Frozen signed-tanh range from the step-zero maximum magnitude."""
    if not math.isfinite(multiplier) or multiplier <= 1.0:
        raise ValueError("activation scale multiplier must be finite and > 1")
    return float(multiplier) * initial_weight.abs().amax().clamp_min(1e-12)


def activation_bias(initial_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    ratio = (initial_weight / scale).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return scale * torch.atanh(ratio)


def activated_weight_and_derivative(
    preactivation: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = scale * torch.tanh(preactivation / scale)
    derivative = 1.0 - (weight / scale).square()
    return weight, derivative


def explained_energy(target: torch.Tensor, prediction: torch.Tensor) -> tuple[float, float]:
    target = target.float().reshape(-1)
    prediction = prediction.float().reshape(-1)
    energy = target.double().square().sum().clamp_min(1e-30)
    residual = (target - prediction).double()
    recovery = 1.0 - residual.square().sum() / energy
    return float(recovery), float(energy)


def cgls(
    apply: Callable[[torch.Tensor], torch.Tensor],
    adjoint: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    template: torch.Tensor,
    iterations: int,
    relative_tolerance: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    coordinate = torch.zeros_like(template, dtype=torch.float32)
    residual = target.float().clone()
    gradient = adjoint(residual).float()
    direction = gradient.clone()
    gamma = gradient.double().square().sum()
    initial_gamma = gamma.clone()
    completed = 0
    for iteration in range(1, iterations + 1):
        projected = apply(direction).float()
        denominator = projected.double().square().sum()
        if denominator <= 0:
            break
        step = gamma / denominator
        coordinate.add_(direction, alpha=float(step))
        residual.sub_(projected, alpha=float(step))
        new_gradient = adjoint(residual).float()
        new_gamma = new_gradient.double().square().sum()
        completed = iteration
        if new_gamma <= initial_gamma * float(relative_tolerance) ** 2:
            break
        beta = new_gamma / gamma.clamp_min(1e-30)
        direction.mul_(float(beta)).add_(new_gradient)
        gradient = new_gradient
        gamma = new_gamma
    return coordinate, target.float() - residual, completed


def classify(
    range_valid: bool,
    heldout_image_recovery: float,
    heldout_tangent_recovery: float,
    threshold: float,
) -> str:
    if not range_valid:
        return "PAPER_ACTIVATION_RANGE_INVALID"
    if heldout_image_recovery < threshold:
        return "PAPER_ACTIVATION_IMAGE_INSUFFICIENT"
    if heldout_tangent_recovery < threshold:
        return "PAPER_ACTIVATION_TANGENT_INSUFFICIENT"
    return "PAPER_ACTIVATION_ORACLE_PASS"


def validate_plan(plan: dict[str, Any]) -> None:
    schema = plan.get("schema_version")
    if schema not in {PLAN_SCHEMA, TIGHT_PLAN_SCHEMA}:
        raise ValueError("paper activation oracle plan mismatch")
    analysis = plan.get("analysis", {})
    expected = {
        "parameter_updates": 0,
        "same_run_only": True,
        "layers": list(LAYERS),
        "probe_steps": list(PROBE_STEPS),
        "heldout_probe_steps": list(HELDOUT_PROBE_STEPS),
        "future_step_by_probe": {str(k): v for k, v in FUTURE_STEP_BY_PROBE.items()},
        "activation": "signed_scaled_tanh",
        "activation_scale": (
            "2*max_abs_step0_per_layer"
            if schema == PLAN_SCHEMA
            else "sqrt(10/9)*max_abs_step0_per_layer"
        ),
        "activation_bias": "s*atanh(W_step0/s)",
        "fixed_operator": "production_seeded_BlockFHT",
        "latent_ratio": 0.01,
        "block_fht_layers": 2,
        "activation_rows": 2048,
        "coordinate_fit": "oracle_cgls_in_inverse_activation_preactivation",
        "future_tangent_fit": "oracle_cgls_in_terminal_post_gelu_metric",
        "primary_control": "identity_BlockFHT_tangent",
        "generic_quadratic_control": "sealed_124m_quadratic_screen",
        "cgls_iterations": 32,
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            raise ValueError(f"paper activation analysis field changed: {key}")
    if schema == TIGHT_PLAN_SCHEMA:
        if analysis.get("activation_scale_multiplier") != math.sqrt(10.0 / 9.0):
            raise ValueError("tight activation multiplier changed")
        if analysis.get("minimum_step0_activation_derivative") != 0.1:
            raise ValueError("tight activation derivative floor changed")
        if analysis.get("step0_jacobian_condition_ceiling") != 10.0:
            raise ValueError("tight activation condition ceiling changed")
    if plan.get("decision_rule", {}).get("thresholds") != {
        "heldout_current_functional_image_recovery_minimum": 0.80,
        "heldout_future_functional_tangent_recovery_minimum": 0.80,
    }:
        raise ValueError("paper activation thresholds changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_oracle") is not True:
        raise ValueError("zero-update activation oracle is not authorized")
    for key in (
        "implement_production_candidate",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acquisition-result", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
        args.acquisition_result: identity["acquisition_result_sha256"],
        args.config: identity["config_sha256"],
        args.checkpoint: identity["checkpoint_sha256"],
        REPO_ROOT / identity["exact_state_result"]: identity["exact_state_result_sha256"],
        REPO_ROOT / identity["quadratic_result"]: identity["quadratic_result_sha256"],
    }
    if "parent_activation_result" in identity:
        pinned[REPO_ROOT / identity["parent_activation_result"]] = identity[
            "parent_activation_result_sha256"
        ]
    for path, expected in pinned.items():
        if file_sha256(path) != expected:
            raise ValueError(f"pinned artifact SHA-256 mismatch: {path}")
    for relative, expected in identity["supporting_source_sha256"].items():
        if file_sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"supporting source changed: {relative}")
    acquisition = json.loads(args.acquisition_result.read_text())
    if acquisition.get("classification") != (
        "ACCEPTED_SAME_RUN_CPROJ_PARAMETER_OPTIMIZER_TRAJECTORY"
    ):
        raise ValueError("paired-state acquisition is not accepted")
    config = json.loads(args.config.read_text())
    manifest = Path(config["data_dir"]) / "manifest.json"
    if file_sha256(manifest) != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    require_block_fht_native_extension(
        bool(config["block_fht_native_extension_required"])
    )
    fixed = make_fixed_eval_indices(
        Path(config["data_dir"]),
        int(config["eval_batch_size"]),
        int(config["block_size"]),
        int(config["eval_iters"]),
        int(config["eval_seed"]),
    )
    if fixed_eval_indices_digest(fixed) != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation indices changed")
    activations = terminal_post_gelu_activations(
        args.checkpoint,
        config,
        fixed,
        int(plan["analysis"]["activation_rows"]),
        args.device,
    )

    run_identity = acquisition["identity"]["run_identity_sha256"]
    snapshot_hashes = acquisition["inventory"]["snapshot_sha256_by_step"]
    required_snapshot_steps = sorted({0, *FUTURE_STEP_BY_PROBE.values()})
    snapshots = {
        step: load_snapshot(
            args.snapshot_dir / f"step_{step:06d}.pt",
            snapshot_hashes[str(step)],
            run_identity,
        )
        for step in required_snapshot_steps
    }
    probe_hashes = acquisition["inventory"]["probe_sha256_by_step"]
    probes: dict[int, dict[str, Any]] = {}
    for step in PROBE_STEPS:
        path = args.probe_dir / f"step_{step:06d}.pt"
        if file_sha256(path) != probe_hashes[str(step)]:
            raise ValueError(f"optimizer probe SHA-256 mismatch at step {step}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["run_identity_sha256"] != run_identity:
            raise ValueError("optimizer probe run identity changed")
        probes[step] = payload

    cgls_iterations = int(plan["analysis"]["cgls_iterations"])
    scale_multiplier = float(
        plan["analysis"].get("activation_scale_multiplier", 2.0)
    )
    latent_ratio = float(plan["analysis"]["latent_ratio"])
    fht_layers = int(plan["analysis"]["block_fht_layers"])
    base_seed = int(config["block_fht_seed"])
    latent_init_std = float(config["block_fht_latent_init_std"])
    target_std = 0.02 / math.sqrt(2 * int(config["n_layer"]))
    weight_scale = target_std / latent_init_std
    rows: list[dict[str, Any]] = []
    started = time.time()
    range_valid = True

    for layer in LAYERS:
        name = parameter_name(layer)
        initial = snapshots[0]["parameters"][name].float().to(args.device)
        scale = activation_scale(initial, scale_multiplier)
        bias = activation_bias(initial, scale)
        size = initial.numel()
        latent_dim = max(1, round(size * latent_ratio))
        seed = base_seed + layer * 4 + 3
        template = torch.zeros(latent_dim, device=args.device, dtype=torch.float32)

        def apply_a(coordinate: torch.Tensor) -> torch.Tensor:
            return (
                block_fht_slice(
                    coordinate,
                    size,
                    fht_layers,
                    seed,
                    0,
                    size,
                )
                * weight_scale
            ).view_as(initial)

        def adjoint_a(weight: torch.Tensor) -> torch.Tensor:
            return block_fht_grad_latent(
                template,
                (weight.reshape(-1) * weight_scale).contiguous(),
                size,
                fht_layers,
                seed,
                0,
                size,
            )

        hidden = activations[layer].float().to(args.device)
        for probe_step in PROBE_STEPS:
            state = probes[probe_step]["parameters"][name]
            current = state["weight_before_step"].float().to(args.device)
            maximum_ratio = float((current.abs().amax() / scale).detach())
            cell_range_valid = maximum_ratio < 1.0
            range_valid = range_valid and cell_range_valid
            clipped = (current / scale).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            target_preactivation_delta = scale * torch.atanh(clipped) - bias
            latent, _, fit_iterations = cgls(
                apply_a,
                adjoint_a,
                target_preactivation_delta,
                template,
                cgls_iterations,
            )
            mapped, derivative = activated_weight_and_derivative(
                bias + apply_a(latent), scale
            )
            raw_image, raw_image_energy = explained_energy(
                current - initial, mapped - initial
            )
            functional_image, functional_image_energy = explained_energy(
                F.linear(hidden, current - initial),
                F.linear(hidden, mapped - initial),
            )
            future_step = FUTURE_STEP_BY_PROBE[probe_step]
            future = snapshots[future_step]["parameters"][name].float().to(args.device)
            future_weight_target = future - current
            future_output_target = F.linear(hidden, future_weight_target)

            def solve_tangent(diagonal: torch.Tensor) -> tuple[float, float, int]:
                def apply_tangent(coordinate: torch.Tensor) -> torch.Tensor:
                    return F.linear(hidden, diagonal * apply_a(coordinate))

                def adjoint_tangent(output: torch.Tensor) -> torch.Tensor:
                    grad_weight = output.transpose(0, 1).matmul(hidden)
                    return adjoint_a(diagonal * grad_weight)

                _, prediction, iterations = cgls(
                    apply_tangent,
                    adjoint_tangent,
                    future_output_target,
                    template,
                    cgls_iterations,
                )
                recovery, energy = explained_energy(future_output_target, prediction)
                return recovery, energy, iterations

            activated_recovery, target_energy, tangent_iterations = solve_tangent(
                derivative
            )
            identity_recovery, _, identity_iterations = solve_tangent(
                torch.ones_like(derivative)
            )
            rows.append(
                {
                    "probe_step": probe_step,
                    "future_step": future_step,
                    "layer": layer,
                    "heldout": probe_step in HELDOUT_PROBE_STEPS,
                    "latent_dim": latent_dim,
                    "latent_ratio": latent_dim / size,
                    "seed": seed,
                    "activation_scale": float(scale),
                    "maximum_current_to_scale_ratio": maximum_ratio,
                    "range_valid": cell_range_valid,
                    "coordinate_fit_iterations": fit_iterations,
                    "raw_current_image_recovery": raw_image,
                    "raw_current_image_target_energy": raw_image_energy,
                    "functional_current_image_recovery": functional_image,
                    "functional_current_image_target_energy": functional_image_energy,
                    "activated_future_functional_tangent_recovery": activated_recovery,
                    "identity_future_functional_tangent_recovery": identity_recovery,
                    "future_functional_target_energy": target_energy,
                    "activated_tangent_iterations": tangent_iterations,
                    "identity_tangent_iterations": identity_iterations,
                    "mean_activation_derivative": float(derivative.mean()),
                    "minimum_activation_derivative": float(derivative.amin()),
                    "maximum_activation_derivative": float(derivative.amax()),
                }
            )

    def weighted(field: str, energy: str, subset: list[dict[str, Any]]) -> float:
        total = sum(float(row[energy]) for row in subset)
        return sum(float(row[field]) * float(row[energy]) for row in subset) / max(
            total, 1e-30
        )

    heldout = [row for row in rows if row["heldout"]]
    aggregate = {
        "heldout_current_functional_image_recovery": weighted(
            "functional_current_image_recovery",
            "functional_current_image_target_energy",
            heldout,
        ),
        "heldout_future_activated_functional_tangent_recovery": weighted(
            "activated_future_functional_tangent_recovery",
            "future_functional_target_energy",
            heldout,
        ),
        "heldout_future_identity_functional_tangent_recovery": weighted(
            "identity_future_functional_tangent_recovery",
            "future_functional_target_energy",
            heldout,
        ),
        "all_current_functional_image_recovery": weighted(
            "functional_current_image_recovery",
            "functional_current_image_target_energy",
            rows,
        ),
    }
    threshold = float(
        plan["decision_rule"]["thresholds"][
            "heldout_future_functional_tangent_recovery_minimum"
        ]
    )
    classification = classify(
        range_valid,
        aggregate["heldout_current_functional_image_recovery"],
        aggregate["heldout_future_activated_functional_tangent_recovery"],
        threshold,
    )
    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "paper_activation_oracle_rows.csv"
    result_path = args.output_dir / "paper_activation_oracle_result.json"
    write_csv(rows_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": classification,
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_mlp_cproj_paper_activation_oracle",
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "analyzer_sha256": file_sha256(Path(__file__)),
            "acquisition_result_sha256": file_sha256(args.acquisition_result),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "run_identity_sha256": run_identity,
        },
        "mechanism": {
            "activation": "signed_scaled_tanh",
            "scale": plan["analysis"]["activation_scale"],
            "scale_multiplier": scale_multiplier,
            "bias": "s*atanh(W_step0/s)",
            "jacobian": "diag(1-(g(z)/s)^2)*A_BlockFHT",
            "latent_ratio": latent_ratio,
            "block_fht_layers": fht_layers,
            "weight_scale": weight_scale,
            "oracle_coordinates": True,
            "oracle_future_tangent_coefficients": True,
        },
        "range_valid": range_valid,
        "aggregate": aggregate,
        "authorization": {
            "activation_family_capacity_supported": classification
            == "PAPER_ACTIVATION_ORACLE_PASS",
            "implement_production_candidate": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
        },
        "artifacts": {
            "rows": str(rows_path),
            "rows_sha256": file_sha256(rows_path),
        },
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
