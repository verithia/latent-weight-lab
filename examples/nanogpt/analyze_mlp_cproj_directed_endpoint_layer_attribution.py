#!/usr/bin/env python3
"""Attribute directed16 c_proj endpoint CE gains across replayed layers."""

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


PLAN_SCHEMA = "mai_124m_mlp_cproj_directed_endpoint_layer_attribution_plan_v1"
OUTPUT32 = "hidden88_output32_full_carry"
DIRECTED16 = "hidden88_directed16_full_carry"


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected directed endpoint attribution plan schema")
    promotion = plan.get("promotion_basis", {})
    result_path = REPO_ROOT / promotion.get("endpoint_result", "")
    if file_sha256(result_path) != promotion.get("endpoint_result_sha256"):
        raise ValueError("endpoint promotion result hash mismatch")
    if promotion.get("classification") != "REJECT_DIRECTED16_ENDPOINT_TASK_DIRECTION":
        raise ValueError("endpoint result does not authorize the attribution")
    if promotion.get("directed_output_product_family_closed") is not True:
        raise ValueError("directed-product family closure is not recorded")
    analysis = plan.get("analysis", {})
    if analysis.get("parameter_updates") != 0:
        raise ValueError("attribution must remain zero-update")
    if analysis.get("layers_in_fixed_depth_order") != [0, 3, 6, 9, 11]:
        raise ValueError("layer order differs from the frozen contract")
    if analysis.get("base_variant") != OUTPUT32 or analysis.get("candidate_variant") != DIRECTED16:
        raise ValueError("endpoint variants differ from the frozen contract")
    if plan.get("authorization", {}).get("implement_and_run_zero_update_attribution") is not True:
        raise ValueError("zero-update attribution is not authorized")


def classify_attribution(
    rows: list[dict[str, Any]],
    *,
    layers: list[int],
    seeds: list[int],
    minimum_mean_gain: float,
) -> dict[str, Any]:
    by_name = {str(row["variant"]): row for row in rows}
    control = by_name["output32_all"]
    candidate = by_name["directed16_all"]
    all_finite = all(
        math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key.startswith("val_ce_seed_")
    )
    all_gains = {
        str(seed): float(control[f"val_ce_seed_{seed}"])
        - float(candidate[f"val_ce_seed_{seed}"])
        for seed in seeds
    }
    layer_gains: dict[str, dict[str, float]] = {}
    for layer in layers:
        row = by_name[f"directed16_layer_{layer}"]
        layer_gains[str(layer)] = {
            str(seed): float(control[f"val_ce_seed_{seed}"])
            - float(row[f"val_ce_seed_{seed}"])
            for seed in seeds
        }
    mean_layer_gains = {
        layer: sum(values.values()) / len(values)
        for layer, values in layer_gains.items()
    }
    consistent_layers = [
        int(layer)
        for layer, values in layer_gains.items()
        if all(value > 0.0 for value in values.values())
    ]
    positive = sorted(
        (value for value in mean_layer_gains.values() if value > 0.0),
        reverse=True,
    )
    positive_sum = sum(positive)
    top_two_fraction = (
        sum(positive[:2]) / positive_sum if positive_sum > 0.0 else 1.0
    )
    replicated = all(value >= minimum_mean_gain for value in all_gains.values())
    if not all_finite or not replicated:
        classification = "DIRECTED_ENDPOINT_FUNCTIONALLY_NULL"
    elif top_two_fraction > 0.75:
        classification = "LAYER_LOCAL_DIRECTED_FUNCTIONAL_SIGNAL"
    elif len(consistent_layers) >= 4:
        classification = "DISTRIBUTED_DIRECTED_FUNCTIONAL_SIGNAL"
    else:
        classification = "MIXED_DIRECTED_FUNCTIONAL_SIGNAL"
    mean_all_gain = sum(all_gains.values()) / len(all_gains)
    single_sum = sum(mean_layer_gains.values())
    nonadditivity_ratio = abs(single_sum - mean_all_gain) / max(
        abs(single_sum), abs(mean_all_gain), 1e-30
    )
    return {
        "classification": classification,
        "all_finite": all_finite,
        "all_layer_ce_gain_by_seed": all_gains,
        "mean_all_layer_ce_gain": mean_all_gain,
        "single_layer_ce_gain_by_seed": layer_gains,
        "mean_single_layer_ce_gain": mean_layer_gains,
        "consistent_improving_layers": consistent_layers,
        "summed_positive_single_layer_gain": positive_sum,
        "top_two_positive_gain_fraction": top_two_fraction,
        "sum_single_layer_gain": single_sum,
        "nonadditivity_ratio": nonadditivity_ratio,
        "nonadditive_context": nonadditivity_ratio > 0.5,
        "minimum_replicated_gain": minimum_mean_gain,
        "authorization": {
            "directed_product_implementation": False,
            "performance_preflight": False,
            "language_model_training": False,
            "larger_rung": False,
        },
    }


def install_variant(
    model,
    *,
    dense_weights: dict[int, torch.Tensor],
    states: dict[tuple[str, int], torch.Tensor],
    layers: list[int],
    directed_layers: set[int] | None,
    dense: bool = False,
) -> None:
    with torch.no_grad():
        for layer in layers:
            if dense:
                source = dense_weights[layer]
            elif directed_layers is not None and layer in directed_layers:
                source = states[(DIRECTED16, layer)]
            else:
                source = states[(OUTPUT32, layer)]
            target = model.transformer.h[layer].mlp.c_proj.weight
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def variant_layers(name: str, layers: list[int]) -> tuple[set[int] | None, bool]:
    if name == "dense_endpoint":
        return None, True
    if name == "output32_all":
        return set(), False
    if name == "directed16_all":
        return set(layers), False
    if name.startswith("directed16_layer_"):
        return {int(name.rsplit("_", 1)[-1])}, False
    if name.startswith("directed16_cumulative_"):
        suffix = name.removeprefix("directed16_cumulative_")
        return {int(value) for value in suffix.split("_")}, False
    raise ValueError(f"unexpected attribution variant {name}")


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
    layers = plan["analysis"]["layers_in_fixed_depth_order"]
    variants = plan["analysis"]["evaluation_variants"]
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
    if payload.get("schema_version") != "cproj_directed_product_endpoint_states_v1":
        raise ValueError("unexpected terminal-state schema")
    if payload.get("layers") != layers:
        raise ValueError("terminal-state layer mismatch")
    if payload.get("trajectory_run_identity_sha256") != identity["trajectory_run_identity_sha256"]:
        raise ValueError("terminal-state trajectory identity mismatch")
    states = {
        (name.rsplit(".", 1)[0], int(name.rsplit(".", 1)[1])): value.float()
        for name, value in payload["states"].items()
        if name.rsplit(".", 1)[0] in {OUTPUT32, DIRECTED16}
    }
    expected_keys = {
        (variant, layer)
        for variant in (OUTPUT32, DIRECTED16)
        for layer in layers
    }
    if set(states) != expected_keys:
        raise ValueError("terminal-state variant inventory mismatch")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dense_weights = {
        layer: checkpoint["model"][
            f"transformer.h.{layer}.mlp.c_proj.weight"
        ].float().clone()
        for layer in layers
    }
    del checkpoint, payload
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
    print(json.dumps({"fixed_eval_indices_sha256": digests}, sort_keys=True), flush=True)

    model = load_model(args.checkpoint, args.device)
    source = TokenBatchSource(args.data_dir)
    rows: list[dict[str, Any]] = []
    for name in variants:
        directed_layers, dense = variant_layers(name, layers)
        install_variant(
            model,
            dense_weights=dense_weights,
            states=states,
            layers=layers,
            directed_layers=directed_layers,
            dense=dense,
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

    decision = classify_attribution(
        rows,
        layers=layers,
        seeds=seeds,
        minimum_mean_gain=plan["classification_rule"][
            "replicated_signal_minimum_mean_ce_gain"
        ],
    )
    args.output.mkdir(parents=True)
    result_path = args.output / "cproj_directed_endpoint_layer_attribution_result.json"
    result = {
        "schema_version": "nanogpt_cproj_directed_endpoint_layer_attribution_v1",
        "rows": rows,
        "decision": decision,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_cproj_directed_endpoint_layer_attribution_metadata_v1",
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
        "evaluation": {
            **protocol,
            "fixed_eval_indices_sha256": digests,
        },
        "outputs": {"result_sha256": file_sha256(result_path)},
        "limitations": plan["analysis"]["not_tested"],
    }
    metadata_path = args.output / "cproj_directed_endpoint_layer_attribution_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print(f"metadata={metadata_path}", flush=True)


if __name__ == "__main__":
    main()
