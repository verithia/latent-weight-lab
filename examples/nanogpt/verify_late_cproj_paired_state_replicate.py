#!/usr/bin/env python3
"""Verify one confirmatory paired-state replay against prospective intervals."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.verify_late_cproj_optimizer_state_trajectory import (
    parse_logged_losses,
    validate_probe,
)
from examples.nanogpt.verify_late_cproj_paired_state_trajectory import (
    assert_equal_parameters,
    validate_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = (
    "mai_124m_repaired_attention_cfc_late_cproj_"
    "paired_state_confirmatory_replicate_plan_v1"
)
RESULT_SCHEMA = "mai_124m_late_cproj_paired_state_confirmatory_verification_v1"


def prediction_interval_rows(
    plan: dict[str, Any], logged: dict[int, dict[str, float]]
) -> list[dict[str, Any]]:
    rows = []
    intervals = plan["acceptance"]["replicate_prediction_intervals_by_step"]
    for raw_step, interval in intervals.items():
        step = int(raw_step)
        observed = float(logged[step]["val"])
        lower = float(interval["lower"])
        upper = float(interval["upper"])
        rows.append(
            {
                "step": step,
                "observed_validation_ce": observed,
                "prediction_lower": lower,
                "prediction_upper": upper,
                "inside_interval": lower <= observed <= upper,
            }
        )
    return sorted(rows, key=lambda row: row["step"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"verification result already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("confirmatory paired-state plan schema mismatch")
    identity = plan["identity"]
    if file_sha256(args.config) != identity["candidate_config_sha256"]:
        raise ValueError("config SHA-256 mismatch")
    if file_sha256(Path(__file__)) != identity["verifier_sha256"]:
        raise ValueError("verifier SHA-256 mismatch")
    status = json.loads(args.status.read_text())
    if status.get("state") != "finished" or status.get("exit_code") != 0:
        raise ValueError("confirmatory paired-state run did not finish cleanly")

    snapshot_contract = plan["targeted_snapshot_contract"]
    snapshot_steps = snapshot_contract["steps"]
    snapshot_paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    if [path.name for path in snapshot_paths] != [
        f"step_{step:06d}.pt" for step in snapshot_steps
    ]:
        raise ValueError("targeted trajectory snapshot inventory mismatch")
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    snapshot_hashes: dict[str, str] = {}
    run_identities: set[str] = set()
    for step, path in zip(snapshot_steps, snapshot_paths, strict=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        run_identity, parameters = validate_snapshot(
            payload, expected_step=step, contract=snapshot_contract
        )
        run_identities.add(run_identity)
        snapshots[step] = parameters
        snapshot_hashes[str(step)] = file_sha256(path)

    optimizer_contract = plan["optimizer_state_contract"]
    probe_steps = optimizer_contract["pre_step_probe_steps"]
    probe_paths = sorted(args.probe_dir.glob("step_*.pt"))
    if [path.name for path in probe_paths] != [
        f"step_{step:06d}.pt" for step in probe_steps
    ]:
        raise ValueError("optimizer probe inventory mismatch")
    probes: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    probe_hashes: dict[str, str] = {}
    for step, path in zip(probe_steps, probe_paths, strict=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        run_identity, parameters = validate_probe(
            payload, expected_step=step, contract=optimizer_contract
        )
        run_identities.add(run_identity)
        probes[step] = parameters
        probe_hashes[str(step)] = file_sha256(path)
    if len(run_identities) != 1:
        raise ValueError("snapshots and probes do not share one run identity")

    assert_equal_parameters(
        {
            name: state["weight_before_step"]
            for name, state in probes[0].items()
        },
        snapshots[0],
    )
    pair_rows = [{"pre_step": 0, "snapshot_step": 0, "state": "before"}]
    for pre_step, post_step in zip(
        probe_steps[1:], optimizer_contract["post_step_snapshot_steps"], strict=True
    ):
        assert_equal_parameters(
            {
                name: state["weight_after_step"]
                for name, state in probes[pre_step].items()
            },
            snapshots[post_step],
        )
        pair_rows.append(
            {"pre_step": pre_step, "snapshot_step": post_step, "state": "after"}
        )

    terminal_step = int(snapshot_contract["terminal_step"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    terminal_names = set(snapshots[terminal_step])
    assert_equal_parameters(
        snapshots[terminal_step],
        {name: checkpoint["model"][name] for name in terminal_names},
    )
    curve_rows = prediction_interval_rows(
        plan, parse_logged_losses(args.training_log)
    )
    if not all(row["inside_interval"] for row in curve_rows):
        raise ValueError("confirmatory paired-state prediction interval failed")

    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "ACCEPTED_SAME_RUN_CPROJ_PARAMETER_OPTIMIZER_TRAJECTORY",
        "passed": True,
        "execution": {
            "host": "PRO6",
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.verify_late_cproj_paired_state_replicate",
            "parameter_updates": 0,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "config_sha256": file_sha256(args.config),
            "training_log_sha256": file_sha256(args.training_log),
            "status_sha256": file_sha256(args.status),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "provenance_sha256": file_sha256(args.provenance),
            "run_identity_sha256": next(iter(run_identities)),
        },
        "inventory": {
            "snapshot_count": len(snapshot_hashes),
            "probe_count": len(probe_hashes),
            "parameter_count_per_record": len(terminal_names),
            "snapshot_sha256_by_step": snapshot_hashes,
            "probe_sha256_by_step": probe_hashes,
        },
        "same_run_pairing": {
            "bitwise_equal": True,
            "rows": pair_rows,
            "terminal_snapshot_bitwise_equal_to_checkpoint": True,
        },
        "curve_reproduction": {
            "passed": True,
            "method": plan["calibration"]["method"],
            "rows": curve_rows,
        },
        "authorization": {
            "zero_update_state_transport_analysis": True,
            "candidate_structure_implementation": False,
            "candidate_language_model_training": False,
            "larger_rung": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
