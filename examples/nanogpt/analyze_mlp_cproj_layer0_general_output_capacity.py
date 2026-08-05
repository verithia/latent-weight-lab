#!/usr/bin/env python3
"""Test a layer-0 general c_proj output action on fresh fixed windows."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_replay import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_124m_mlp_cproj_layer0_general_output_capacity_plan_v1"
OUTPUT32 = "hidden88_output32_full_carry"
DIRECTED16 = "hidden88_directed16_full_carry"
COMPONENTS = ("skew", "symmetric", "full")


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected layer-0 general-output plan schema")
    promotion = plan.get("promotion_basis", {})
    result_path = REPO_ROOT / promotion.get("layer_attribution_result", "")
    if file_sha256(result_path) != promotion.get("layer_attribution_result_sha256"):
        raise ValueError("layer-attribution promotion result hash mismatch")
    if promotion.get("classification") != "LAYER_LOCAL_DIRECTED_FUNCTIONAL_SIGNAL":
        raise ValueError("layer-attribution result did not authorize this capacity test")
    if promotion.get("selected_layer") != 0:
        raise ValueError("unexpected selected layer")
    analysis = plan.get("analysis", {})
    if analysis.get("parameter_updates") != 0 or analysis.get("selected_layer") != 0:
        raise ValueError("general-output capacity test must remain layer-0 zero-update")
    action = analysis.get("general_action", {})
    if (
        action.get("relative_ridge") != 1e-6
        or action.get("components_in_smallest_pass_order") != list(COMPONENTS)
        or action.get("persistent_coordinates") != 0
        or action.get("minimum_singular_value_i_plus_action") != 0.95
    ):
        raise ValueError("general-output action differs from the frozen contract")
    if plan.get("authorization", {}).get("implement_and_run_zero_update_capacity_oracle") is not True:
        raise ValueError("capacity oracle is not authorized")


def _minimum_singular_value(matrix: torch.Tensor) -> float:
    return float(torch.linalg.svdvals(matrix.float()).min())


def enforce_minimum_singular_value(
    action: torch.Tensor,
    *,
    minimum: float,
    bisection_steps: int = 32,
) -> tuple[torch.Tensor, float, float]:
    identity = torch.eye(action.shape[0], device=action.device, dtype=action.dtype)
    raw_minimum = _minimum_singular_value(identity + action)
    if raw_minimum >= minimum:
        return action, 1.0, raw_minimum
    low = 0.0
    high = 1.0
    for _ in range(bisection_steps):
        middle = (low + high) / 2.0
        if _minimum_singular_value(identity + middle * action) >= minimum:
            low = middle
        else:
            high = middle
    bounded = action * low
    bounded_minimum = _minimum_singular_value(identity + bounded)
    if bounded_minimum < minimum - 1e-5:
        raise ValueError("general-output trust bisection failed")
    return bounded, low, bounded_minimum


def fit_general_output_actions(
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    relative_ridge: float,
    minimum_singular_value: float,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float]]]:
    if weight.ndim != 2 or target.shape != weight.shape:
        raise ValueError("weight and target must be equal two-dimensional tensors")
    source = weight.float()
    residual = target.float() - source
    gram = source @ source.T
    ridge = float(relative_ridge) * float(torch.diagonal(gram).mean())
    regularized = gram + ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    cross = residual @ source.T
    full = torch.linalg.solve(regularized, cross.T).T
    raw_components = {
        "skew": 0.5 * (full - full.T),
        "symmetric": 0.5 * (full + full.T),
        "full": full,
    }
    candidates: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    residual_energy = float(residual.double().square().sum())
    for name in COMPONENTS:
        bounded, scale, minimum = enforce_minimum_singular_value(
            raw_components[name], minimum=minimum_singular_value
        )
        delta = bounded @ source
        candidate = source + delta
        error = target.float() - candidate
        values = {
            "ridge": ridge,
            "trust_scale": scale,
            "minimum_singular_value_i_plus_action": minimum,
            "action_fro": float(bounded.double().norm()),
            "delta_energy": float(delta.double().square().sum()),
            "target_residual_energy": residual_energy,
            "endpoint_recovery": 1.0
            - float(error.double().square().sum()) / max(residual_energy, 1e-30),
        }
        if not torch.isfinite(candidate).all() or not all(
            math.isfinite(value) for value in values.values()
        ):
            raise ValueError("general-output action produced nonfinite values")
        candidates[name] = candidate.contiguous()
        diagnostics[name] = values
    return candidates, diagnostics


def select_general_action(
    rows: list[dict[str, Any]],
    *,
    seeds: list[int],
    minimum_gain: float,
    minimum_dense_recovery: float,
    diagnostics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    by_name = {str(row["variant"]): row for row in rows}
    control = by_name["output32_all"]
    directed = by_name["directed16_layer0"]
    dense = by_name["dense_layer0"]
    comparisons: dict[str, Any] = {}
    for component in COMPONENTS:
        name = f"general_{component}_layer0"
        candidate = by_name[name]
        gains = {
            str(seed): float(control[f"val_ce_seed_{seed}"])
            - float(candidate[f"val_ce_seed_{seed}"])
            for seed in seeds
        }
        directed_gains = {
            str(seed): float(control[f"val_ce_seed_{seed}"])
            - float(directed[f"val_ce_seed_{seed}"])
            for seed in seeds
        }
        dense_gains = {
            str(seed): float(control[f"val_ce_seed_{seed}"])
            - float(dense[f"val_ce_seed_{seed}"])
            for seed in seeds
        }
        recoveries = {
            str(seed): gains[str(seed)] / max(dense_gains[str(seed)], 1e-30)
            for seed in seeds
        }
        all_finite = all(
            math.isfinite(value)
            for collection in (gains, directed_gains, dense_gains, recoveries)
            for value in collection.values()
        ) and all(math.isfinite(value) for value in diagnostics[component].values())
        gates = {
            "all_finite": all_finite,
            "gain_each_seed": all(value >= minimum_gain for value in gains.values()),
            "strictly_beats_directed16_each_seed": all(
                gains[str(seed)] > directed_gains[str(seed)] for seed in seeds
            ),
            "dense_recovery_each_seed": all(
                value >= minimum_dense_recovery for value in recoveries.values()
            ),
            "minimum_singular_value": diagnostics[component][
                "minimum_singular_value_i_plus_action"
            ]
            >= 0.95,
            "bounded_full_residual_energy": (
                component != "full"
                or diagnostics[component]["delta_energy"]
                <= diagnostics[component]["target_residual_energy"] * (1.0 + 1e-5)
            ),
        }
        comparisons[name] = {
            "validation_ce_gain_by_seed": gains,
            "directed16_validation_ce_gain_by_seed": directed_gains,
            "dense_layer0_validation_ce_gain_by_seed": dense_gains,
            "dense_layer0_gain_recovery_by_seed": recoveries,
            "diagnostics": diagnostics[component],
            "gates": gates,
            "passed": all(gates.values()),
        }
    selected = next(
        (
            f"general_{component}_layer0"
            for component in COMPONENTS
            if comparisons[f"general_{component}_layer0"]["passed"]
        ),
        None,
    )
    return {
        "comparisons": comparisons,
        "selected_variant": selected,
        "decision": (
            "LAYER0_GENERAL_OUTPUT_CAPACITY_PASS"
            if selected is not None
            else "REJECT_LAYER0_GENERAL_OUTPUT_ACTION"
        ),
        "authorization": {
            "production_implementation": selected is not None,
            "exact_config_mfu_preflight": selected is not None,
            "language_model_training": False,
            "larger_rung": False,
        },
    }


def install_layer0(
    model,
    *,
    states: dict[tuple[str, int], torch.Tensor],
    dense_weights: dict[int, torch.Tensor],
    selected_layers: list[int],
    layer0_weight: torch.Tensor,
) -> None:
    with torch.no_grad():
        for layer in selected_layers:
            source = states[(OUTPUT32, layer)]
            target = model.transformer.h[layer].mlp.c_proj.weight
            target.copy_(source.to(device=target.device, dtype=target.dtype))
        target = model.transformer.h[0].mlp.c_proj.weight
        target.copy_(layer0_weight.to(device=target.device, dtype=target.dtype))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--terminal-states", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    identity = plan["identity"]
    protocol = plan["analysis"]["evaluation_protocol"]
    layers = [0, 3, 6, 9, 11]
    seeds = protocol["eval_seeds"]
    if args.output.exists():
        raise ValueError("output directory already exists; rerun is not authorized")
    if file_sha256(args.checkpoint) != identity["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    if file_sha256(args.terminal_states) != identity["terminal_states_sha256"]:
        raise ValueError("terminal-state hash mismatch")
    if file_sha256(args.data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash mismatch")

    payload = torch.load(args.terminal_states, map_location="cpu", weights_only=False)
    if payload.get("trajectory_run_identity_sha256") != identity["trajectory_run_identity_sha256"]:
        raise ValueError("trajectory identity mismatch")
    states = {
        (name.rsplit(".", 1)[0], int(name.rsplit(".", 1)[1])): value.float()
        for name, value in payload["states"].items()
        if name.rsplit(".", 1)[0] in {OUTPUT32, DIRECTED16}
    }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dense_weights = {
        layer: checkpoint["model"][
            f"transformer.h.{layer}.mlp.c_proj.weight"
        ].float().clone()
        for layer in layers
    }
    del checkpoint, payload
    candidates, diagnostics = fit_general_output_actions(
        states[(OUTPUT32, 0)].to(args.device),
        dense_weights[0].to(args.device),
        relative_ridge=plan["analysis"]["general_action"]["relative_ridge"],
        minimum_singular_value=plan["analysis"]["general_action"][
            "minimum_singular_value_i_plus_action"
        ],
    )
    print(json.dumps({"general_action": diagnostics}, sort_keys=True), flush=True)

    fixed_by_seed = {
        seed: make_fixed_eval_indices(
            args.data_dir,
            protocol["eval_batch_size"],
            protocol["block_size"],
            protocol["eval_iters_per_seed"],
            seed,
        )
        for seed in seeds
    }
    digests = {
        str(seed): fixed_eval_indices_digest(indices)
        for seed, indices in fixed_by_seed.items()
    }
    model = load_model(args.checkpoint, args.device)
    source = TokenBatchSource(args.data_dir)
    layer0_by_variant = {
        "output32_all": states[(OUTPUT32, 0)],
        "directed16_layer0": states[(DIRECTED16, 0)],
        "dense_layer0": dense_weights[0],
        **{f"general_{name}_layer0": value for name, value in candidates.items()},
    }
    rows: list[dict[str, Any]] = []
    for name in plan["analysis"]["evaluation_variants"]:
        install_layer0(
            model,
            states=states,
            dense_weights=dense_weights,
            selected_layers=layers,
            layer0_weight=layer0_by_variant[name],
        )
        row: dict[str, Any] = {"variant": name}
        for seed in seeds:
            row[f"val_ce_seed_{seed}"] = evaluate_fixed_ce(
                model,
                data_dir=args.data_dir,
                fixed_indices=fixed_by_seed[seed],
                split="val",
                eval_iters=protocol["eval_iters_per_seed"],
                eval_batch_size=protocol["eval_batch_size"],
                block_size=protocol["block_size"],
                device=args.device,
                dtype="bfloat16",
                source=source,
            )
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    requirements = plan["decision_rule"]["candidate_requirements"]
    selection = select_general_action(
        rows,
        seeds=seeds,
        minimum_gain=requirements[
            "validation_ce_gain_over_output32_each_seed_minimum"
        ],
        minimum_dense_recovery=requirements[
            "dense_layer0_ce_gain_recovery_each_seed_minimum"
        ],
        diagnostics=diagnostics,
    )
    args.output.mkdir(parents=True)
    result_path = args.output / "cproj_layer0_general_output_capacity_result.json"
    result = {
        "schema_version": "nanogpt_cproj_layer0_general_output_capacity_v1",
        "rows": rows,
        "selection": selection,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_cproj_layer0_general_output_capacity_metadata_v1",
        "repository_commit": git_commit(REPO_ROOT),
        "entrypoint": str(script),
        "entrypoint_sha256": file_sha256(script),
        "command": sys.argv,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "device": args.device,
        "checkpoint": {"path": str(args.checkpoint), "sha256": file_sha256(args.checkpoint)},
        "terminal_states": {"path": str(args.terminal_states), "sha256": file_sha256(args.terminal_states)},
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "dataset_manifest": {"path": str(args.data_dir / "manifest.json"), "sha256": file_sha256(args.data_dir / "manifest.json")},
        "evaluation": {**protocol, "fixed_eval_indices_sha256": digests},
        "outputs": {"result_sha256": file_sha256(result_path)},
        "limitations": plan["analysis"]["not_tested"],
    }
    metadata_path = args.output / "cproj_layer0_general_output_capacity_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)
    print(f"metadata={metadata_path}", flush=True)


if __name__ == "__main__":
    main()
