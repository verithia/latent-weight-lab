#!/usr/bin/env python3
"""Test optimizer-state transport against its own exact c_proj trajectory."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    ActivationCollector,
)
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_mlp_cproj_multiscale_path import (
    cumulative_lr_coordinate,
    polynomial_predict,
)
from examples.nanogpt.analyze_mlp_cproj_optimizer_state_transport import (
    COMPONENTS,
    DISCOVERY_STEPS,
    HELDOUT_PROBE_STEPS,
    LAYERS,
    REFERENCE_POST_STEPS,
    TERMINAL_STEP,
    classify,
    functional_metric,
    metric,
    output_additive_projection,
    reconstruct_components,
    weighted,
)
from examples.nanogpt.analyze_mlp_cproj_polynomial_oracle_ce import (
    restore_radius,
)
from examples.nanogpt.analyze_mlp_cproj_predictive_manifold import (
    fit_through_origin_basis,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.parameter_trajectory import SCHEMA_VERSION
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    get_batch,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_mlp_cproj_same_run_optimizer_state_transport_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_same_run_optimizer_state_transport_result_v1"
SNAPSHOT_STEPS = (0, *range(99, 2373, 99), 2373)
PROBE_STEPS = (0, 98, 296, 593, 890, 1187, 1484, 1781, 2078, 2372)


def parameter_name(layer: int) -> str:
    return f"transformer.h.{layer}.mlp.c_proj.weight"


def phase_target_step(probe_index: int) -> int | None:
    """Return the next nontrivial phase endpoint for one pre-step probe."""
    if not 0 <= probe_index < len(PROBE_STEPS):
        raise IndexError("optimizer probe index is out of range")
    if probe_index >= len(REFERENCE_POST_STEPS):
        return None
    return REFERENCE_POST_STEPS[probe_index]


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("same-run optimizer-state transport plan mismatch")
    expected = {
        "parameter_updates": 0,
        "same_run_only": True,
        "layers": list(LAYERS),
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "probe_steps": list(PROBE_STEPS),
        "reference_post_steps": list(REFERENCE_POST_STEPS),
        "discovery_steps": list(DISCOVERY_STEPS),
        "terminal_step": TERMINAL_STEP,
        "polynomial_rank": 4,
        "polynomial_degree": 2,
        "activation_rows": 2048,
        "terminal_activations_from_same_run_checkpoint": True,
        "components": list(COMPONENTS),
        "heldout_probe_steps": list(HELDOUT_PROBE_STEPS),
        "output_additive_projection": True,
        "future_phase_target_by_probe": {
            str(step): phase_target_step(index)
            for index, step in enumerate(PROBE_STEPS)
        },
    }
    analysis = plan.get("analysis", {})
    for key, value in expected.items():
        if analysis.get(key) != value:
            raise ValueError(f"same-run analysis field changed: {key}")
    thresholds = plan.get("decision_rule", {}).get("thresholds", {})
    if thresholds != {
        "compression_reconstruction_max_relative_error": 1e-4,
        "causal_heldout_functional_line_recovery_minimum": 0.80,
    }:
        raise ValueError("same-run optimizer-state thresholds changed")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_state_transport_analysis") is not True:
        raise ValueError("same-run state-transport analysis is not authorized")
    for key in (
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def load_snapshot(
    path: Path, expected_hash: str, run_identity: str
) -> dict[str, Any]:
    if file_sha256(path) != expected_hash:
        raise ValueError(f"targeted snapshot SHA-256 mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("targeted snapshot schema mismatch")
    if payload.get("run_identity_sha256") != run_identity:
        raise ValueError("targeted snapshot run identity mismatch")
    expected_names = {parameter_name(layer) for layer in LAYERS}
    if set(payload.get("parameters", {})) != expected_names:
        raise ValueError("targeted snapshot parameter inventory mismatch")
    return payload


def scheduled_terminal_residual(
    snapshots: dict[int, dict[str, Any]],
    config: dict[str, Any],
    layer: int,
    device: str,
) -> torch.Tensor:
    steps = (*DISCOVERY_STEPS, TERMINAL_STEP)
    name = parameter_name(layer)
    weights = {
        step: snapshots[step]["parameters"][name].float().to(device)
        for step in steps
    }
    initial_norm = weights[0].norm().clamp_min(1e-30)
    normalized = {
        step: weight * (initial_norm / weight.norm().clamp_min(1e-30))
        for step, weight in weights.items()
    }
    discovery = torch.stack(
        [
            (normalized[step] - normalized[0]).reshape(-1)
            for step in DISCOVERY_STEPS
        ]
    )
    basis = fit_through_origin_basis(discovery[1:], 4)
    coordinates = discovery @ basis.T
    discovery_progress = torch.tensor(
        [cumulative_lr_coordinate(step, config) for step in DISCOVERY_STEPS],
        device=device,
    )
    terminal_progress = torch.tensor(
        [cumulative_lr_coordinate(TERMINAL_STEP, config)], device=device
    )
    prediction = polynomial_predict(
        coordinates, discovery_progress, terminal_progress, 2
    )
    predicted = normalized[0] + (prediction @ basis).reshape_as(normalized[0])
    predicted = restore_radius(predicted, weights[TERMINAL_STEP].norm())
    return weights[TERMINAL_STEP] - predicted


def terminal_post_gelu_activations(
    checkpoint_path: Path,
    config: dict[str, Any],
    fixed_indices: dict[str, torch.Tensor],
    sample_cap: int,
    device: str,
) -> dict[int, torch.Tensor]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
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
    collector = ActivationCollector(model, list(LAYERS), sample_cap)
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
            for layer in LAYERS
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
    }
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
        raise ValueError("same-run paired-state acquisition is not accepted")
    if acquisition["identity"]["checkpoint_sha256"] != identity["checkpoint_sha256"]:
        raise ValueError("accepted checkpoint identity changed")
    config = json.loads(args.config.read_text())
    manifest = Path(config["data_dir"]) / "manifest.json"
    if file_sha256(manifest) != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    require_block_fht_native_extension(
        bool(config["block_fht_native_extension_required"])
    )

    run_identity = acquisition["identity"]["run_identity_sha256"]
    snapshot_hashes = acquisition["inventory"]["snapshot_sha256_by_step"]
    required_steps = sorted(
        set(DISCOVERY_STEPS) | set(REFERENCE_POST_STEPS) | {TERMINAL_STEP}
    )
    snapshots = {
        step: load_snapshot(
            args.snapshot_dir / f"step_{step:06d}.pt",
            snapshot_hashes[str(step)],
            run_identity,
        )
        for step in required_steps
    }
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
    terminal_residuals = {
        layer: scheduled_terminal_residual(snapshots, config, layer, args.device)
        for layer in LAYERS
    }

    probe_hashes = acquisition["inventory"]["probe_sha256_by_step"]
    rows: list[dict[str, Any]] = []
    max_reconstruction_error = 0.0
    started = time.time()
    for index, probe_step in enumerate(PROBE_STEPS):
        path = args.probe_dir / f"step_{probe_step:06d}.pt"
        if file_sha256(path) != probe_hashes[str(probe_step)]:
            raise ValueError(f"probe SHA-256 mismatch at step {probe_step}")
        probe = torch.load(path, map_location="cpu", weights_only=False)
        if probe["run_identity_sha256"] != run_identity:
            raise ValueError("optimizer probe run identity changed")
        future_step = phase_target_step(index)
        for layer in LAYERS:
            name = parameter_name(layer)
            state = {
                key: value.to(args.device)
                for key, value in probe["parameters"][name].items()
            }
            hyper = probe["hyperparameters"][name]
            components, reconstruction_error = reconstruct_components(state, hyper)
            max_reconstruction_error = max(
                max_reconstruction_error, reconstruction_error
            )
            hidden = activations[layer].to(args.device)
            terminal_target = terminal_residuals[layer]
            future_target = None
            if future_step is not None:
                future_target = (
                    snapshots[future_step]["parameters"][name].float().to(args.device)
                    - state["weight_before_step"].float()
                )
            for component_name, component in components.items():
                raw_terminal = metric(terminal_target, component)
                functional_terminal = functional_metric(
                    terminal_target, component, hidden
                )
                projected = output_additive_projection(component, hidden)
                component_function = functional_metric(component, projected, hidden)
                row: dict[str, Any] = {
                    "probe_step": probe_step,
                    "layer": layer,
                    "component": component_name,
                    "heldout": probe_step in HELDOUT_PROBE_STEPS,
                    "compression_reconstruction_relative_error": reconstruction_error,
                    "terminal_raw_target_energy": raw_terminal["target_energy"],
                    "terminal_raw_cosine": raw_terminal["cosine"],
                    "terminal_raw_positive_line_recovery": raw_terminal[
                        "positive_line_recovery"
                    ],
                    "terminal_raw_fixed_scale_recovery": raw_terminal[
                        "fixed_scale_recovery"
                    ],
                    "terminal_functional_target_energy": functional_terminal[
                        "target_energy"
                    ],
                    "terminal_functional_cosine": functional_terminal["cosine"],
                    "terminal_functional_positive_line_recovery": functional_terminal[
                        "positive_line_recovery"
                    ],
                    "terminal_functional_fixed_scale_recovery": functional_terminal[
                        "fixed_scale_recovery"
                    ],
                    "component_output_additive_functional_recovery": component_function[
                        "fixed_scale_recovery"
                    ],
                }
                if future_target is not None:
                    raw_future = metric(future_target, component)
                    functional_future = functional_metric(
                        future_target, component, hidden
                    )
                    row.update(
                        {
                            "future_reference_step": future_step,
                            "future_raw_target_energy": raw_future["target_energy"],
                            "future_raw_positive_line_recovery": raw_future[
                                "positive_line_recovery"
                            ],
                            "future_functional_target_energy": functional_future[
                                "target_energy"
                            ],
                            "future_functional_positive_line_recovery": functional_future[
                                "positive_line_recovery"
                            ],
                        }
                    )
                rows.append(row)
        del probe

    aggregate: dict[str, dict[str, float]] = {}
    for name in COMPONENTS:
        component_rows = [row for row in rows if row["component"] == name]
        heldout = [row for row in component_rows if row["heldout"]]
        heldout_future = [
            row for row in heldout if "future_reference_step" in row
        ]
        aggregate[name] = {
            "heldout_terminal_raw_positive_line_recovery": weighted(
                heldout,
                "terminal_raw_positive_line_recovery",
                "terminal_raw_target_energy",
            ),
            "heldout_terminal_functional_positive_line_recovery": weighted(
                heldout,
                "terminal_functional_positive_line_recovery",
                "terminal_functional_target_energy",
            ),
            "heldout_future_raw_positive_line_recovery": weighted(
                heldout_future,
                "future_raw_positive_line_recovery",
                "future_raw_target_energy",
            ),
            "heldout_future_functional_positive_line_recovery": weighted(
                heldout_future,
                "future_functional_positive_line_recovery",
                "future_functional_target_energy",
            ),
            "mean_component_output_additive_functional_recovery": sum(
                float(row["component_output_additive_functional_recovery"])
                for row in component_rows
            )
            / len(component_rows),
        }
    threshold = float(
        plan["decision_rule"]["thresholds"][
            "causal_heldout_functional_line_recovery_minimum"
        ]
    )
    reconstruction_valid = max_reconstruction_error <= float(
        plan["decision_rule"]["thresholds"][
            "compression_reconstruction_max_relative_error"
        ]
    )
    classification, best = classify(
        aggregate, reconstruction_valid, threshold
    )
    args.output_dir.mkdir(parents=True)
    rows_path = args.output_dir / "same_run_optimizer_state_transport_rows.csv"
    result_path = args.output_dir / "same_run_optimizer_state_transport_result.json"
    write_csv(rows_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": classification,
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": (
                "examples.nanogpt.analyze_mlp_cproj_same_run_optimizer_state_transport"
            ),
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "analyzer_sha256": file_sha256(Path(__file__)),
            "acquisition_result_sha256": file_sha256(args.acquisition_result),
            "config_sha256": file_sha256(args.config),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "run_identity_sha256": run_identity,
        },
        "mechanical_reconstruction": {
            "passed": reconstruction_valid,
            "maximum_relative_error": max_reconstruction_error,
        },
        "aggregate": aggregate,
        "best_heldout_component": best,
        "authorization": {
            "compact_state_conditioned_mapper": classification
            == "CAUSAL_OPTIMIZER_STATE_TRANSPORT_SUFFICIENT",
            "implement_candidate_structure": False,
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
    if not reconstruction_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
