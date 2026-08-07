#!/usr/bin/env python3
"""Fail-closed verifier for the QK+c_fc 20TPP phase acquisition."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_activation_update_alignment import load_snapshot, model_from_snapshot
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import file_sha256, git_commit
from examples.nanogpt.parameter_trajectory import FULL_STATE_SCHEMA_VERSION
from examples.nanogpt.train import fixed_eval_indices_digest, make_fixed_eval_indices, require_block_fht_native_extension
from examples.nanogpt.verify_full_state_functional_replay import evaluate_validation_ce, parse_logged_losses
from examples.nanogpt.verify_late_cproj_full_state_trajectory import compare_terminal_checkpoint, expected_snapshot_steps, validate_snapshot
from examples.nanogpt.verify_resume_checkpoint_envelope import verify as verify_resume


PLAN_SCHEMA = "mai_124m_qk_cfc_20tpp_phase_acquisition_plan_v1"
RESULT_SCHEMA = "mai_124m_qk_cfc_20tpp_phase_acquisition_verification_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--terminal-audit", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("phase acquisition plan schema mismatch")
    identity = plan["identity"]
    checks = {
        "config": file_sha256(args.config) == identity["candidate_config_sha256"],
        "source_result": file_sha256(args.source_result) == identity["source_result_sha256"],
        "terminal_audit": file_sha256(args.terminal_audit) == identity["terminal_audit_sha256"],
        "verifier": file_sha256(Path(__file__)) == identity["verifier_sha256"],
    }
    status = json.loads(args.status.read_text())
    provenance = json.loads(args.provenance.read_text())
    checks.update({
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "provenance_config": provenance["config"]["sha256"] == identity["candidate_config_sha256"],
        "provenance_dataset": provenance["dataset_manifest"]["sha256"] == identity["dataset_manifest_sha256"],
    })
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("identity checks failed: " + ", ".join(failed))
    config = json.loads(args.config.read_text())
    require_block_fht_native_extension(bool(config["block_fht_native_extension_required"]))
    manifest = Path(config["data_dir"]) / "manifest.json"
    if file_sha256(manifest) != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest mismatch")
    fixed = make_fixed_eval_indices(
        Path(config["data_dir"]), int(config["eval_batch_size"]),
        int(config["block_size"]), int(config["eval_iters"]), int(config["eval_seed"]),
    )
    if fixed_eval_indices_digest(fixed) != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed indices mismatch")
    registered = plan["full_state_contract"]["expected_snapshot_steps"]
    if registered != expected_snapshot_steps(int(config["max_iters"]), int(config["trajectory_snapshot_interval"])):
        raise ValueError("snapshot cadence mismatch")
    paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    if [path.name for path in paths] != [f"step_{step:06d}.pt" for step in registered]:
        raise ValueError("snapshot inventory mismatch")
    started = time.time()
    snapshot_hashes: dict[str, str] = {}
    snapshots: dict[int, dict[str, Any]] = {}
    parameter_names = buffer_names = None
    run_identity = None
    for step, path in zip(registered, paths):
        payload = load_snapshot(path)
        current_parameters, current_buffers = validate_snapshot(
            payload, expected_step=step,
            expected_parameter_names=parameter_names,
            expected_buffer_names=buffer_names,
        )
        parameter_names = parameter_names or current_parameters
        buffer_names = buffer_names or current_buffers
        run_identity = run_identity or payload["run_identity_sha256"]
        if payload["run_identity_sha256"] != run_identity:
            raise ValueError("snapshot run identity changed")
        snapshot_hashes[str(step)] = file_sha256(path)
        snapshots[step] = payload
    logged = parse_logged_losses(args.training_log)
    tolerance = float(plan["acceptance"]["curve_absolute_tolerance_ce"])
    curve_rows = []
    for raw_step, accepted_ce in plan["acceptance"]["accepted_validation_ce_by_step"].items():
        step = int(raw_step)
        observed = float(logged[step]["val"])
        curve_rows.append({
            "step": step, "accepted_validation_ce": float(accepted_ce),
            "observed_validation_ce": observed, "delta_ce": observed - float(accepted_ce),
            "within_tolerance": abs(observed - float(accepted_ce)) <= tolerance,
        })
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    eval_args = SimpleNamespace(**config)
    eval_args.device = args.device
    eval_args._ptdtype = dtype
    ctx = contextlib.nullcontext() if "cuda" not in args.device else torch.amp.autocast("cuda", dtype=dtype)
    replay_tolerance = float(plan["acceptance"]["functional_replay_absolute_tolerance_ce"])
    replay_rows = []
    for step in registered:
        model = model_from_snapshot(snapshots[step], args.device)
        replay = evaluate_validation_ce(model, data_dir=Path(config["data_dir"]), args=eval_args, indices=fixed["val"], ctx=ctx)
        delta = replay - float(logged[step]["val"])
        replay_rows.append({
            "step": step, "logged_validation_ce": float(logged[step]["val"]),
            "replayed_validation_ce": replay, "delta_ce": delta,
            "within_tolerance": abs(delta) <= replay_tolerance,
        })
        del model
        torch.cuda.empty_cache()
    checkpoint = verify_resume(args.checkpoint)
    terminal_comparison = compare_terminal_checkpoint(snapshots[int(config["max_iters"])], args.checkpoint)
    curve_passed = all(row["within_tolerance"] for row in curve_rows)
    replay_passed = all(row["within_tolerance"] for row in replay_rows)
    passed = curve_passed and replay_passed and int(checkpoint["next_iter"]) == int(config["max_iters"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "ACCEPTED_QK_CFC_20TPP_PHASE_ACQUISITION" if passed else "REJECTED_QK_CFC_20TPP_PHASE_ACQUISITION",
        "passed": passed,
        "identity_checks": checks,
        "identity": {
            "plan_sha256": file_sha256(args.plan), "config_sha256": file_sha256(args.config),
            "source_result_sha256": file_sha256(args.source_result),
            "terminal_audit_sha256": file_sha256(args.terminal_audit),
            "training_log_sha256": file_sha256(args.training_log),
            "status_sha256": file_sha256(args.status), "provenance_sha256": file_sha256(args.provenance),
            "checkpoint_sha256": file_sha256(args.checkpoint), "run_identity_sha256": run_identity,
            "dataset_manifest_sha256": file_sha256(manifest),
            "fixed_eval_indices_sha256": fixed_eval_indices_digest(fixed),
        },
        "inventory": {
            "snapshot_count": len(paths), "parameter_count": len(parameter_names or ()),
            "buffer_count": len(buffer_names or ()), "snapshot_sha256_by_step": snapshot_hashes,
            **terminal_comparison,
        },
        "checkpoint": checkpoint,
        "curve_reproduction": {"passed": curve_passed, "tolerance_ce": tolerance, "rows": curve_rows},
        "functional_replay": {"passed": replay_passed, "tolerance_ce": replay_tolerance, "rows": replay_rows},
        "execution": {"git_commit": git_commit(REPO_ROOT), "parameter_updates": 0, "elapsed_seconds": time.time() - started},
        "authorization": {"phase_analysis": passed, "candidate_structure": False, "language_model_training": False, "larger_rung": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
