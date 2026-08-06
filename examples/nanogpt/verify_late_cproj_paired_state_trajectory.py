#!/usr/bin/env python3
"""Verify same-run c_proj parameter snapshots and optimizer-state probes."""

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
from examples.nanogpt.parameter_trajectory import SCHEMA_VERSION
from examples.nanogpt.verify_late_cproj_optimizer_state_trajectory import (
    parse_logged_losses,
    validate_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = (
    "mai_124m_repaired_attention_cfc_late_cproj_"
    "paired_state_acquisition_plan_v1"
)
RESULT_SCHEMA = "mai_124m_late_cproj_paired_state_verification_v1"


def validate_snapshot(
    payload: dict[str, Any], *, expected_step: int, contract: dict[str, Any]
) -> tuple[str, dict[str, torch.Tensor]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("targeted trajectory snapshot schema mismatch")
    if int(payload.get("step", -1)) != expected_step:
        raise ValueError("targeted trajectory snapshot step mismatch")
    if payload.get("targets") != [contract["target"]]:
        raise ValueError("targeted trajectory snapshot target mismatch")
    if payload.get("layers") != contract["layers"]:
        raise ValueError("targeted trajectory snapshot layers mismatch")
    if payload.get("storage_dtype") != contract["storage_dtype"]:
        raise ValueError("targeted trajectory snapshot dtype mismatch")
    if payload.get("all_parameters") or payload.get("all_buffers"):
        raise ValueError("targeted snapshot unexpectedly became full-state")
    if payload.get("buffers"):
        raise ValueError("targeted snapshot must not duplicate buffers")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("targeted trajectory parameters missing")
    expected_names = {
        f"transformer.h.{layer}.mlp.c_proj.weight"
        for layer in contract["layers"]
    }
    if set(parameters) != expected_names:
        raise ValueError("targeted trajectory parameter inventory mismatch")
    for name, value in parameters.items():
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"nonfinite targeted trajectory tensor: {name}")
    return str(payload["run_identity_sha256"]), parameters


def assert_equal_parameters(
    observed: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> None:
    if set(observed) != set(reference):
        raise ValueError("paired parameter names disagree")
    for name in observed:
        if not torch.equal(observed[name].float(), reference[name].float()):
            raise ValueError(f"same-run paired state mismatch: {name}")


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
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("paired-state plan schema mismatch")
    identity = plan["identity"]
    if file_sha256(args.config) != identity["candidate_config_sha256"]:
        raise ValueError("config SHA-256 mismatch")
    if file_sha256(Path(__file__)) != identity["verifier_sha256"]:
        raise ValueError("verifier SHA-256 mismatch")
    status = json.loads(args.status.read_text())
    if status.get("state") != "finished" or status.get("exit_code") != 0:
        raise ValueError("paired-state training run did not finish cleanly")

    snapshot_contract = plan["targeted_snapshot_contract"]
    snapshot_steps = snapshot_contract["steps"]
    snapshot_paths = sorted(args.snapshot_dir.glob("step_*.pt"))
    expected_snapshot_names = [f"step_{step:06d}.pt" for step in snapshot_steps]
    if [path.name for path in snapshot_paths] != expected_snapshot_names:
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
    expected_probe_names = [f"step_{step:06d}.pt" for step in probe_steps]
    if [path.name for path in probe_paths] != expected_probe_names:
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

    before_zero = {
        name: state["weight_before_step"] for name, state in probes[0].items()
    }
    assert_equal_parameters(before_zero, snapshots[0])
    pair_rows = [{"pre_step": 0, "snapshot_step": 0, "state": "before"}]
    for pre_step, post_step in zip(
        probe_steps[1:], optimizer_contract["post_step_snapshot_steps"], strict=True
    ):
        after = {
            name: state["weight_after_step"]
            for name, state in probes[pre_step].items()
        }
        assert_equal_parameters(after, snapshots[post_step])
        pair_rows.append(
            {"pre_step": pre_step, "snapshot_step": post_step, "state": "after"}
        )

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    terminal_names = set(snapshots[snapshot_contract["terminal_step"]])
    checkpoint_terminal = {
        name: checkpoint["model"][name] for name in terminal_names
    }
    assert_equal_parameters(
        snapshots[snapshot_contract["terminal_step"]], checkpoint_terminal
    )

    logged = parse_logged_losses(args.training_log)
    tolerance = float(plan["acceptance"]["curve_absolute_tolerance_ce"])
    curve_rows = []
    for raw_step, accepted in plan["acceptance"][
        "accepted_validation_ce_by_step"
    ].items():
        step = int(raw_step)
        observed = float(logged[step]["val"])
        curve_rows.append(
            {
                "step": step,
                "accepted_validation_ce": float(accepted),
                "observed_validation_ce": observed,
                "delta_ce": observed - float(accepted),
                "within_tolerance": abs(observed - float(accepted)) <= tolerance,
            }
        )
    curve_passed = all(row["within_tolerance"] for row in curve_rows)
    if not curve_passed:
        raise ValueError("paired-state fixed CE curve did not reproduce")

    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "ACCEPTED_SAME_RUN_CPROJ_PARAMETER_OPTIMIZER_TRAJECTORY",
        "passed": True,
        "execution": {
            "host": "PRO6",
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.verify_late_cproj_paired_state_trajectory",
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
            "tolerance_ce": tolerance,
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
