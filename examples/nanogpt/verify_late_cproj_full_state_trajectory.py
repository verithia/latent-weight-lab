#!/usr/bin/env python3
"""Verify the accepted late-c_proj full-state trajectory acquisition."""

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

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.parameter_trajectory import FULL_STATE_SCHEMA_VERSION
from examples.nanogpt.train import (
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)
from examples.nanogpt.verify_full_state_functional_replay import (
    evaluate_validation_ce,
    parse_logged_losses,
)


PLAN_SCHEMA = (
    "mai_124m_repaired_attention_cfc_late_cproj_"
    "full_state_trajectory_acquisition_plan_v1"
)
RESULT_SCHEMA = "mai_124m_late_cproj_full_state_trajectory_verification_v1"


def expected_snapshot_steps(max_iters: int, interval: int) -> list[int]:
    steps = list(range(0, max_iters + 1, interval))
    if steps[-1] != max_iters:
        steps.append(max_iters)
    return steps


def validate_snapshot(
    payload: dict[str, Any],
    *,
    expected_step: int,
    expected_parameter_names: set[str] | None,
    expected_buffer_names: set[str] | None,
) -> tuple[set[str], set[str]]:
    if payload.get("schema_version") != FULL_STATE_SCHEMA_VERSION:
        raise ValueError("snapshot is not full-state trajectory v2")
    if payload.get("all_parameters") is not True:
        raise ValueError("snapshot does not contain all named parameters")
    if payload.get("all_buffers") is not True:
        raise ValueError("snapshot does not contain all persistent buffers")
    if int(payload.get("step", -1)) != expected_step:
        raise ValueError("snapshot step mismatch")
    parameters = payload.get("parameters")
    buffers = payload.get("buffers")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("snapshot parameter inventory is empty")
    if not isinstance(buffers, dict) or not buffers:
        raise ValueError("snapshot persistent-buffer inventory is empty")
    parameter_names = set(parameters)
    buffer_names = set(buffers)
    if expected_parameter_names is not None and parameter_names != expected_parameter_names:
        raise ValueError("snapshot parameter inventory changed")
    if expected_buffer_names is not None and buffer_names != expected_buffer_names:
        raise ValueError("snapshot persistent-buffer inventory changed")
    for name, value in parameters.items():
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"invalid parameter tensor: {name}")
    for name, value in buffers.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"non-finite persistent buffer: {name}")
    return parameter_names, buffer_names


def compare_terminal_checkpoint(
    snapshot: dict[str, Any], checkpoint_path: Path
) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint has no model state")
    compared_parameters = 0
    compared_buffers = 0
    for key, values, counter in (
        ("parameters", snapshot["parameters"], "parameters"),
        ("buffers", snapshot["buffers"], "buffers"),
    ):
        for name, value in values.items():
            observed = model_state.get(name)
            if not torch.is_tensor(observed) or not torch.equal(
                observed.detach().cpu(), value.detach().cpu()
            ):
                raise ValueError(f"terminal checkpoint mismatch: {name}")
            if counter == "parameters":
                compared_parameters += 1
            else:
                compared_buffers += 1
    return {
        "compared_parameters": compared_parameters,
        "compared_buffers": compared_buffers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accepted-result", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("trajectory verification plan schema mismatch")
    identity = plan["identity"]
    if file_sha256(args.config) != identity["candidate_config_sha256"]:
        raise ValueError("config SHA-256 mismatch")
    if file_sha256(args.accepted_result) != identity["source_result_sha256"]:
        raise ValueError("accepted result SHA-256 mismatch")
    if file_sha256(Path(__file__)) != identity["verifier_sha256"]:
        raise ValueError("verifier SHA-256 mismatch")
    config = json.loads(args.config.read_text())
    accepted = json.loads(args.accepted_result.read_text())
    require_block_fht_native_extension(
        bool(config["block_fht_native_extension_required"])
    )
    manifest = Path(config["data_dir"]) / "manifest.json"
    if file_sha256(manifest) != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    fixed = make_fixed_eval_indices(
        Path(config["data_dir"]),
        int(config["eval_batch_size"]),
        int(config["block_size"]),
        int(config["eval_iters"]),
        int(config["eval_seed"]),
    )
    if fixed_eval_indices_digest(fixed) != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed validation indices SHA-256 mismatch")

    registered_steps = plan["full_state_contract"]["expected_snapshot_steps"]
    computed_steps = expected_snapshot_steps(
        int(config["max_iters"]),
        int(config["trajectory_snapshot_interval"]),
    )
    if registered_steps != computed_steps:
        raise ValueError("registered snapshot cadence mismatch")
    observed_paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    expected_names = [f"step_{step:06d}.pt" for step in registered_steps]
    if [path.name for path in observed_paths] != expected_names:
        raise ValueError("snapshot file inventory mismatch")

    started = time.time()
    snapshot_hashes: dict[str, str] = {}
    phase_snapshots: dict[int, dict[str, Any]] = {}
    run_identity = None
    parameter_names: set[str] | None = None
    buffer_names: set[str] | None = None
    for step, path in zip(registered_steps, observed_paths):
        payload = load_snapshot(path)
        current_parameters, current_buffers = validate_snapshot(
            payload,
            expected_step=step,
            expected_parameter_names=parameter_names,
            expected_buffer_names=buffer_names,
        )
        if parameter_names is None:
            parameter_names = current_parameters
            buffer_names = current_buffers
        if run_identity is None:
            run_identity = payload["run_identity_sha256"]
        elif payload["run_identity_sha256"] != run_identity:
            raise ValueError("snapshot run identities disagree")
        snapshot_hashes[str(step)] = file_sha256(path)
        if step in plan["full_state_contract"]["required_functional_replay_steps"]:
            phase_snapshots[step] = payload

    logged = parse_logged_losses(args.training_log)
    accepted_curve = {
        int(step): float(value)
        for step, value in plan["acceptance"][
            "accepted_validation_ce_by_step"
        ].items()
    }
    curve_tolerance = float(plan["acceptance"]["curve_absolute_tolerance_ce"])
    curve_rows = []
    for step, accepted_ce in accepted_curve.items():
        observed_ce = float(logged[step]["val"])
        delta = observed_ce - accepted_ce
        curve_rows.append(
            {
                "step": step,
                "accepted_validation_ce": accepted_ce,
                "observed_validation_ce": observed_ce,
                "delta_ce": delta,
                "absolute_delta_ce": abs(delta),
                "within_tolerance": abs(delta) <= curve_tolerance,
            }
        )

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]
    eval_args = SimpleNamespace(**config)
    eval_args.device = args.device
    eval_args._ptdtype = dtype
    ctx = (
        contextlib.nullcontext()
        if "cuda" not in args.device
        else torch.amp.autocast(device_type="cuda", dtype=dtype)
    )
    replay_tolerance = float(
        plan["acceptance"]["functional_replay_absolute_tolerance_ce"]
    )
    replay_rows = []
    for step in plan["full_state_contract"]["required_functional_replay_steps"]:
        model = model_from_snapshot(phase_snapshots[step], args.device)
        replay_ce = evaluate_validation_ce(
            model,
            data_dir=Path(config["data_dir"]),
            args=eval_args,
            indices=fixed["val"],
            ctx=ctx,
        )
        delta = replay_ce - float(logged[step]["val"])
        replay_rows.append(
            {
                "step": step,
                "logged_validation_ce": float(logged[step]["val"]),
                "replayed_validation_ce": replay_ce,
                "delta_ce": delta,
                "absolute_delta_ce": abs(delta),
                "within_tolerance": abs(delta) <= replay_tolerance,
            }
        )
        del model
        torch.cuda.empty_cache()

    terminal_step = int(config["max_iters"])
    checkpoint_comparison = compare_terminal_checkpoint(
        phase_snapshots[terminal_step], args.checkpoint
    )
    curve_passed = all(row["within_tolerance"] for row in curve_rows)
    replay_passed = all(row["within_tolerance"] for row in replay_rows)
    passed = curve_passed and replay_passed
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "ACCEPTED_COADAPTED_LATE_CPROJ_FULL_STATE_TRAJECTORY"
            if passed
            else "REJECTED_COADAPTED_LATE_CPROJ_TRAJECTORY_REPLAY"
        ),
        "passed": passed,
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.verify_late_cproj_full_state_trajectory",
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "config_sha256": file_sha256(args.config),
            "accepted_result_sha256": file_sha256(args.accepted_result),
            "training_log_sha256": file_sha256(args.training_log),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "run_identity_sha256": run_identity,
            "dataset_manifest_sha256": file_sha256(manifest),
            "fixed_eval_indices_sha256": fixed_eval_indices_digest(fixed),
        },
        "inventory": {
            "snapshot_count": len(registered_steps),
            "parameter_count": len(parameter_names or ()),
            "buffer_count": len(buffer_names or ()),
            "snapshot_sha256_by_step": snapshot_hashes,
            **checkpoint_comparison,
        },
        "curve_reproduction": {
            "passed": curve_passed,
            "tolerance_ce": curve_tolerance,
            "rows": curve_rows,
        },
        "functional_replay": {
            "passed": replay_passed,
            "tolerance_ce": replay_tolerance,
            "rows": replay_rows,
        },
        "authorization": {
            "zero_update_coadapted_manifold_analysis": passed,
            "candidate_structure_implementation": False,
            "candidate_language_model_training": False,
            "larger_rung": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
