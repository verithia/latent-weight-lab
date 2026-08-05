#!/usr/bin/env python3
"""Evaluate directed-product c_proj replay endpoints on fixed LM windows."""

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
    EXPECTED_EVAL_DIGEST,
    evaluate_fixed_ce,
    install_variant,
)
from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_replay import (
    cosine_lr,
    file_sha256,
    git_commit,
    load_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_teacher_forced_directed_product_carry import (
    ARMS,
    structured_step,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_124m_mlp_cproj_directed_product_endpoint_fixed_eval_plan_v1"
VARIANTS = (
    "dense_endpoint",
    "hidden88_full_carry",
    "hidden88_output32_full_carry",
    "hidden88_directed16_full_carry",
)
REPLAY_ARMS = tuple(arm for arm in ARMS if arm.name in VARIANTS)
CONTROL = "hidden88_output32_full_carry"
CANDIDATE = "hidden88_directed16_full_carry"


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected directed-product endpoint plan schema")
    analysis = plan.get("analysis", {})
    expected = {
        "parameter_updates": 0,
        "teacher_forced_dense_updates": True,
        "layers": [0, 3, 6, 9, 11],
        "trajectory_steps": 238,
        "feedback_decay": 1.0,
        "hidden_parent_stages": [64, 24],
        "output32_control_stages": [32],
        "selected_directed_incoming_by_stage": [16],
        "selected_output_coordinate_budget_per_layer": 12_288,
        "models_in_evaluation_order": list(VARIANTS),
    }
    if {key: analysis.get(key) for key in expected} != expected:
        raise ValueError("directed-product endpoint plan does not match v1 contract")
    promotion = plan.get("promotion_basis", {})
    result_path = REPO_ROOT / promotion.get("geometry_result", "")
    if file_sha256(result_path) != promotion.get("geometry_result_sha256"):
        raise ValueError("geometry promotion result hash mismatch")
    if promotion.get("classification") != "DIRECTED_OUTPUT_PRODUCT_PASS":
        raise ValueError("geometry result did not authorize endpoint evaluation")
    if promotion.get("selected_arm") != CANDIDATE:
        raise ValueError("unexpected selected directed-product arm")
    if (
        plan.get("authorization", {}).get(
            "implement_and_run_zero_update_endpoint_oracle"
        )
        is not True
    ):
        raise ValueError("endpoint oracle is not authorized")


def select_directed_endpoint(
    rows: list[dict[str, Any]],
    *,
    minimum_val_improvement: float,
    maximum_val_gap_to_dense: float,
) -> dict[str, Any]:
    by_name = {str(row["variant"]): row for row in rows}
    if set(by_name) != set(VARIANTS):
        raise ValueError("endpoint rows do not match the frozen variant set")
    dense = by_name["dense_endpoint"]
    control = by_name[CONTROL]
    candidate = by_name[CANDIDATE]
    all_finite = all(
        math.isfinite(float(row[key]))
        for row in by_name.values()
        for key in ("train_ce", "val_ce")
    )
    train_improvement = float(control["train_ce"]) - float(candidate["train_ce"])
    val_improvement = float(control["val_ce"]) - float(candidate["val_ce"])
    val_gap_to_dense = abs(float(candidate["val_ce"]) - float(dense["val_ce"]))
    gates = {
        "all_finite": all_finite,
        "validation_ce_improvement_over_output32": (
            val_improvement >= minimum_val_improvement
        ),
        "train_ce_no_worse_than_output32": train_improvement >= 0.0,
        "validation_ce_no_worse_than_output32": val_improvement >= 0.0,
        "validation_ce_gap_to_dense": val_gap_to_dense <= maximum_val_gap_to_dense,
    }
    passed = all(gates.values())
    return {
        "candidate": CANDIDATE,
        "control": CONTROL,
        "metrics": {
            "train_ce_improvement_over_output32": train_improvement,
            "validation_ce_improvement_over_output32": val_improvement,
            "validation_ce_gap_to_dense": val_gap_to_dense,
        },
        "thresholds": {
            "validation_ce_improvement_over_output32_minimum": minimum_val_improvement,
            "validation_ce_gap_to_dense_maximum": maximum_val_gap_to_dense,
        },
        "gates": gates,
        "passed": passed,
        "decision": (
            "DIRECTED16_ENDPOINT_TASK_LOSS_PASS"
            if passed
            else "REJECT_DIRECTED16_ENDPOINT_TASK_DIRECTION"
        ),
        "authorization": {
            "production_implementation": passed,
            "exact_config_mfu_preflight": passed,
            "language_model_training": False,
        },
    }


def replay_terminal_states(
    *,
    snapshot_dir: Path,
    layers: list[int],
    neighbors: int,
    seed: int,
    device: str,
    expected_identity: str,
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[str, Any]]:
    paths = [snapshot_dir / f"step_{step:06d}.pt" for step in range(239)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} snapshots; first={missing[0]}")
    first = load_snapshot(paths[0])
    identity = first["run_identity_sha256"]
    if identity != expected_identity:
        raise ValueError("trajectory identity mismatch")
    config = first["run_identity"]["resolved_config"]
    names = [f"transformer.h.{layer}.mlp.c_proj.weight" for layer in layers]
    starts = {
        layer: first["parameters"][name].to(device).float()
        for layer, name in zip(layers, names, strict=True)
    }
    dense_previous = {layer: value.clone() for layer, value in starts.items()}
    states = {
        (arm.name, layer): starts[layer].clone()
        for arm in REPLAY_ARMS
        for layer in layers
    }
    feedback = {key: torch.zeros_like(value) for key, value in states.items()}
    for step in range(238):
        payload = load_snapshot(paths[step + 1])
        if payload["run_identity_sha256"] != identity:
            raise ValueError(f"trajectory identity mismatch at step {step + 1}")
        lr = cosine_lr(
            step,
            learning_rate=float(config["learning_rate"]),
            min_lr=float(config["min_lr"]),
            warmup_iters=int(config["warmup_iters"]),
            decay_iters=int(config["lr_decay_iters"]),
        )
        weight_decay = float(config["weight_decay"])
        for layer, name in zip(layers, names, strict=True):
            dense_before = dense_previous[layer]
            dense_after = payload["parameters"][name].to(device).float()
            dense_delta = dense_after - dense_before
            dense_nondecay = dense_delta + lr * weight_decay * dense_before
            for arm in REPLAY_ARMS:
                key = (arm.name, layer)
                candidate = states[key]
                requested = dense_nondecay - lr * weight_decay * candidate
                states[key], feedback[key], _, _ = structured_step(
                    candidate,
                    requested,
                    feedback[key],
                    arm=arm,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    neighbors=neighbors,
                    seed=seed + layer * 100000 + step * 10,
                )
            dense_previous[layer] = dense_after
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == 238:
            print(json.dumps({"replay_step": step + 1}), flush=True)
        del payload
    inventory = {
        "count": len(paths),
        "first_sha256": file_sha256(paths[0]),
        "last_sha256": file_sha256(paths[-1]),
        "total_bytes": sum(path.stat().st_size for path in paths),
    }
    return states, {"config": config, "snapshot_inventory": inventory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--eval-seed", type=int, default=20260715)
    parser.add_argument("--eval-iters", type=int, default=400)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--minimum-val-improvement", type=float, default=0.002)
    parser.add_argument("--maximum-val-gap-to-dense", type=float, default=0.0046)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    layers = [int(value) for value in args.layers.split(",")]
    protocol = plan["analysis"]["evaluation_protocol"]
    rule = plan["decision_rule"]["selected_candidate_requirements"]
    if layers != plan["analysis"]["layers"]:
        raise ValueError("runtime layer contract mismatch")
    if (
        args.eval_iters != protocol["eval_iters"]
        or args.eval_batch_size != protocol["eval_batch_size"]
        or args.block_size != protocol["block_size"]
        or args.dtype != "bfloat16"
        or args.seed != 20260805
        or args.eval_seed != 20260715
        or args.neighbors != 64
    ):
        raise ValueError("runtime evaluation or replay contract mismatch")
    if (
        args.minimum_val_improvement
        != rule["validation_ce_improvement_over_output32_minimum"]
        or args.maximum_val_gap_to_dense
        != rule["validation_ce_gap_to_dense_maximum"]
    ):
        raise ValueError("runtime decision thresholds differ from plan")
    if args.output.exists():
        raise ValueError("output directory already exists; rerun is not authorized")
    identity = plan["identity"]
    if file_sha256(args.checkpoint) != identity["trajectory_checkpoint_sha256"]:
        raise ValueError("trajectory checkpoint hash mismatch")
    if file_sha256(args.data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash mismatch")

    print("replaying directed-product terminal states", flush=True)
    states, replay_meta = replay_terminal_states(
        snapshot_dir=args.snapshot_dir,
        layers=layers,
        neighbors=args.neighbors,
        seed=args.seed,
        device=args.device,
        expected_identity=identity["trajectory_run_identity_sha256"],
    )
    inventory = replay_meta["snapshot_inventory"]
    if (
        inventory["first_sha256"] != identity["first_snapshot_sha256"]
        or inventory["last_sha256"] != identity["last_snapshot_sha256"]
    ):
        raise ValueError("snapshot inventory hash mismatch")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dense_weights = {
        layer: checkpoint["model"][
            f"transformer.h.{layer}.mlp.c_proj.weight"
        ].float().clone()
        for layer in layers
    }
    last = load_snapshot(args.snapshot_dir / "step_000238.pt")
    for layer in layers:
        name = f"transformer.h.{layer}.mlp.c_proj.weight"
        if not torch.equal(dense_weights[layer], last["parameters"][name].float()):
            raise ValueError(f"checkpoint and terminal snapshot differ at layer {layer}")
    del checkpoint, last

    fixed_indices = make_fixed_eval_indices(
        args.data_dir,
        args.eval_batch_size,
        args.block_size,
        args.eval_iters,
        args.eval_seed,
    )
    fixed_digest = fixed_eval_indices_digest(fixed_indices)
    if (
        fixed_digest != EXPECTED_EVAL_DIGEST
        or fixed_digest != identity["fixed_eval_indices_sha256"]
    ):
        raise ValueError("fixed evaluation index digest mismatch")
    print(f"fixed_eval_indices_sha256={fixed_digest}", flush=True)

    model = load_model(args.checkpoint, args.device)
    source = TokenBatchSource(args.data_dir)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        print(f"evaluating variant={variant}", flush=True)
        install_variant(
            model,
            dense_weights=dense_weights,
            states=states,
            layers=layers,
            variant=variant,
        )
        row: dict[str, Any] = {"variant": variant}
        for split in ("train", "val"):
            row[f"{split}_ce"] = evaluate_fixed_ce(
                model,
                data_dir=args.data_dir,
                fixed_indices=fixed_indices,
                split=split,
                eval_iters=args.eval_iters,
                eval_batch_size=args.eval_batch_size,
                block_size=args.block_size,
                device=args.device,
                dtype=args.dtype,
                source=source,
            )
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    selection = select_directed_endpoint(
        rows,
        minimum_val_improvement=args.minimum_val_improvement,
        maximum_val_gap_to_dense=args.maximum_val_gap_to_dense,
    )
    args.output.mkdir(parents=True)
    state_path = args.output / "cproj_directed_product_endpoint_states.pt"
    torch.save(
        {
            "schema_version": "cproj_directed_product_endpoint_states_v1",
            "layers": layers,
            "trajectory_run_identity_sha256": identity[
                "trajectory_run_identity_sha256"
            ],
            "states": {
                f"{arm}.{layer}": value.detach().cpu()
                for (arm, layer), value in states.items()
            },
        },
        state_path,
    )
    result_path = args.output / "cproj_directed_product_endpoint_fixed_eval_result.json"
    result = {
        "schema_version": "nanogpt_cproj_directed_product_endpoint_fixed_eval_v1",
        "rows": rows,
        "selection": selection,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_cproj_directed_product_endpoint_fixed_eval_metadata_v1",
        "repository_commit": git_commit(REPO_ROOT),
        "entrypoint": str(script),
        "entrypoint_sha256": file_sha256(script),
        "command": sys.argv,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "device": args.device,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
        },
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "dataset_manifest": {
            "path": str(args.data_dir / "manifest.json"),
            "sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        "evaluation": {
            "protocol": "mai_ladder_fixed_eval_indices_v2",
            "fixed_eval_indices_sha256": fixed_digest,
            "eval_seed": args.eval_seed,
            "eval_iters_per_split": args.eval_iters,
            "eval_batch_size": args.eval_batch_size,
            "block_size": args.block_size,
            "dtype": args.dtype,
        },
        "replay": replay_meta,
        "outputs": {
            "result_sha256": file_sha256(result_path),
            "terminal_states_sha256": file_sha256(state_path),
        },
        "limitations": [
            "Only the five recorded c_proj layers are replaced.",
            "Directions remain teacher-forced from the paired dense replay.",
            "This is a zero-update endpoint oracle, not causal training.",
        ],
    }
    metadata_path = args.output / "cproj_directed_product_endpoint_fixed_eval_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)
    print(f"metadata={metadata_path}", flush=True)


if __name__ == "__main__":
    main()
