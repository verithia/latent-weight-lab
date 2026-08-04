#!/usr/bin/env python3
"""Replay bilateral c_proj endpoints and evaluate exact fixed-window CE."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_full_carry import (
    ARMS,
    CONTROL,
    CANDIDATE_ORDER,
    structured_step,
)
from examples.nanogpt.analyze_mlp_cproj_teacher_forced_bilateral_replay import (
    cosine_lr,
    file_sha256,
    git_commit,
    load_snapshot,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    get_batch,
    make_fixed_eval_indices,
)


EXPECTED_EVAL_DIGEST = (
    "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
)


def select_variant(
    rows: list[dict[str, Any]],
    *,
    minimum_val_gain: float,
    minimum_train_gain: float,
    authorize_production_implementation: bool = True,
) -> dict[str, Any]:
    by_name = {str(row["variant"]): row for row in rows}
    dense = by_name["dense_endpoint"]
    control = by_name[CONTROL]
    comparisons: dict[str, dict[str, Any]] = {}
    for name in CANDIDATE_ORDER:
        candidate = by_name[name]
        val_gain = float(control["val_ce"]) - float(candidate["val_ce"])
        train_gain = float(control["train_ce"]) - float(candidate["train_ce"])
        dense_val_distance = abs(
            float(candidate["val_ce"]) - float(dense["val_ce"])
        )
        control_dense_val_distance = abs(
            float(control["val_ce"]) - float(dense["val_ce"])
        )
        all_finite = all(
            math.isfinite(float(row[key]))
            for row in (dense, control, candidate)
            for key in ("train_ce", "val_ce")
        )
        comparison = {
            "validation_ce_gain_vs_right_only": val_gain,
            "train_ce_gain_vs_right_only": train_gain,
            "validation_distance_from_dense": dense_val_distance,
            "right_only_validation_distance_from_dense": control_dense_val_distance,
            "closer_to_dense_on_validation": (
                dense_val_distance < control_dense_val_distance
            ),
            "all_finite": all_finite,
        }
        comparison["passed"] = bool(
            all_finite
            and val_gain >= minimum_val_gain
            and train_gain >= minimum_train_gain
            and comparison["closer_to_dense_on_validation"]
        )
        comparisons[name] = comparison
    selected = next(
        (name for name in CANDIDATE_ORDER if comparisons[name]["passed"]),
        None,
    )
    if selected is None:
        decision = "REJECT_BILATERAL_ENDPOINT_AS_TASK_DIRECTION"
    elif authorize_production_implementation:
        decision = f"SELECT_{selected.upper()}_FOR_PRODUCTION_IMPLEMENTATION"
    else:
        decision = f"SELECT_{selected.upper()}_AS_DIAGNOSTIC_ENDPOINT_PASS"
    return {
        "comparisons": comparisons,
        "selected_variant": selected if selected is not None else CONTROL,
        "decision": decision,
        "production_implementation_authorized": bool(
            selected is not None and authorize_production_implementation
        ),
        "language_model_training_authorized": False,
    }


def replay_terminal_states(
    *,
    snapshot_dir: Path,
    layers: list[int],
    neighbors: int,
    seed: int,
    device: str,
    expected_identity: str,
    connectivity_target: str,
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
        for arm in ARMS
        for layer in layers
    }
    feedback = {key: torch.zeros_like(value) for key, value in states.items()}
    for step in range(238):
        payload = load_snapshot(paths[step + 1])
        if payload["run_identity_sha256"] != identity:
            raise ValueError(f"run identity mismatch at step {step + 1}")
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
            for arm in ARMS:
                key = (arm.name, layer)
                candidate = states[key]
                requested = dense_nondecay - lr * weight_decay * candidate
                parent_connectivity_update = None
                if connectivity_target == "production_nondecay":
                    parent_connectivity_update = (
                        dense_nondecay + feedback[key]
                    )
                states[key], feedback[key], _ = structured_step(
                    candidate,
                    requested,
                    feedback[key],
                    parent_connectivity_update=parent_connectivity_update,
                    output_stages=arm.output_stages,
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
    return states, {
        "config": config,
        "connectivity_target": connectivity_target,
        "snapshot_inventory": inventory,
    }


def install_variant(
    model,
    *,
    dense_weights: dict[int, torch.Tensor],
    states: dict[tuple[str, int], torch.Tensor],
    layers: list[int],
    variant: str,
) -> None:
    with torch.no_grad():
        for layer in layers:
            source = (
                dense_weights[layer]
                if variant == "dense_endpoint"
                else states[(variant, layer)]
            )
            target = model.transformer.h[layer].mlp.c_proj.weight
            target.copy_(source.to(device=target.device, dtype=target.dtype))


@torch.no_grad()
def evaluate_fixed_ce(
    model,
    *,
    data_dir: Path,
    fixed_indices: dict[str, torch.Tensor],
    split: str,
    eval_iters: int,
    eval_batch_size: int,
    block_size: int,
    device: str,
    dtype: str,
    source: TokenBatchSource,
) -> float:
    if dtype == "bfloat16":
        autocast_dtype = torch.bfloat16
    elif dtype == "float16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32
    losses = torch.zeros(eval_iters, dtype=torch.float32)
    model.eval()
    for index in range(eval_iters):
        x, y = get_batch(
            data_dir,
            split,
            eval_batch_size,
            block_size,
            device,
            indices=fixed_indices[split][index],
            source=source,
        )
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if device.startswith("cuda") and dtype != "float32"
            else nullcontext()
        )
        with context:
            _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite {split} loss at batch {index}")
        losses[index] = loss.detach().float().cpu()
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == eval_iters:
            print(
                json.dumps(
                    {
                        "eval_split": split,
                        "eval_batch": index + 1,
                        "eval_batches": eval_iters,
                    }
                ),
                flush=True,
            )
    return float(losses.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--eval-seed", type=int, default=20260715)
    parser.add_argument("--eval-iters", type=int, default=400)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--minimum-val-gain", type=float, default=0.002)
    parser.add_argument("--minimum-train-gain", type=float, default=0.0)
    parser.add_argument(
        "--connectivity-target",
        choices=("legacy_full_requested", "production_nondecay"),
        default="legacy_full_requested",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    layers = [int(value) for value in args.layers.split(",")]
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    schema_version = plan.get("schema_version")
    if schema_version not in {
        "mai_124m_mlp_cproj_bilateral_endpoint_fixed_eval_plan_v1",
        "mai_124m_mlp_cproj_bilateral_endpoint_production_selector_plan_v1",
        "mai_124m_mlp_cproj_bilateral_endpoint_matched_parent_plan_v1",
    }:
        raise ValueError("unexpected plan schema")
    if (
        schema_version in {
            "mai_124m_mlp_cproj_bilateral_endpoint_production_selector_plan_v1",
            "mai_124m_mlp_cproj_bilateral_endpoint_matched_parent_plan_v1",
        }
        and args.connectivity_target != "production_nondecay"
    ):
        raise ValueError("production-selector plan requires production_nondecay")
    decision_rule = plan.get(
        "registered_decision_rule", plan.get("decision_rule")
    )
    if not isinstance(decision_rule, dict):
        raise ValueError("plan does not contain a decision rule")
    if file_sha256(args.checkpoint) != plan["inputs"]["dense_endpoint_checkpoint_sha256"]:
        raise ValueError("dense endpoint checkpoint hash mismatch")
    if file_sha256(args.data_dir / "manifest.json") != plan["inputs"]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash mismatch")
    if args.minimum_val_gain != decision_rule["minimum_validation_ce_gain_vs_right_only"]:
        raise ValueError("minimum validation gain differs from plan")
    if args.minimum_train_gain != decision_rule["minimum_train_ce_gain_vs_right_only"]:
        raise ValueError("minimum train gain differs from plan")

    print("replaying terminal states", flush=True)
    states, replay_meta = replay_terminal_states(
        snapshot_dir=args.snapshot_dir,
        layers=layers,
        neighbors=args.neighbors,
        seed=args.seed,
        device=args.device,
        expected_identity=plan["inputs"]["trajectory_run_identity_sha256"],
        connectivity_target=args.connectivity_target,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dense_weights = {
        layer: checkpoint["model"][f"transformer.h.{layer}.mlp.c_proj.weight"].float().clone()
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
    if fixed_digest != EXPECTED_EVAL_DIGEST or fixed_digest != plan["inputs"]["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation index digest mismatch")
    print(f"fixed_eval_indices_sha256={fixed_digest}", flush=True)

    model = load_model(args.checkpoint, args.device)
    source = TokenBatchSource(args.data_dir)
    rows: list[dict[str, Any]] = []
    variants = ("dense_endpoint", CONTROL, *CANDIDATE_ORDER)
    for variant in variants:
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

    selection = select_variant(
        rows,
        minimum_val_gain=args.minimum_val_gain,
        minimum_train_gain=args.minimum_train_gain,
        authorize_production_implementation=bool(
            decision_rule.get(
                "endpoint_pass_authorizes_production_implementation",
                True,
            )
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "cproj_bilateral_terminal_states.pt"
    torch.save(
        {
            "schema_version": "cproj_bilateral_terminal_states_v1",
            "layers": layers,
            "trajectory_run_identity_sha256": plan["inputs"]["trajectory_run_identity_sha256"],
            "states": {f"{arm}.{layer}": value.detach().cpu() for (arm, layer), value in states.items()},
        },
        state_path,
    )
    result_path = args.output / "cproj_bilateral_endpoint_fixed_eval_result.json"
    result = {
        "schema_version": "nanogpt_cproj_bilateral_endpoint_fixed_eval_v1",
        "rows": rows,
        "selection": selection,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": "nanogpt_cproj_bilateral_endpoint_fixed_eval_metadata_v1",
        "repository_commit": git_commit(REPO_ROOT),
        "entrypoint": str(script),
        "entrypoint_sha256": file_sha256(script),
        "command": sys.argv,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "device": args.device,
        "checkpoint": {"path": str(args.checkpoint), "sha256": file_sha256(args.checkpoint)},
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "dataset_manifest": {"path": str(args.data_dir / "manifest.json"), "sha256": file_sha256(args.data_dir / "manifest.json")},
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
        "limitations": plan.get(
            "limitations",
            [
                "Only the five recorded c_proj layers are replaced.",
                "Directions remain teacher-forced from the paired dense replay.",
                "This is an endpoint capacity/task-direction oracle, not causal training.",
            ],
        ),
    }
    metadata_path = args.output / "cproj_bilateral_endpoint_fixed_eval_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)
    print(f"metadata={metadata_path}", flush=True)


if __name__ == "__main__":
    main()
