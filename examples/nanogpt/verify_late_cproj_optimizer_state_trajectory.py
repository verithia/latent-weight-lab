#!/usr/bin/env python3
"""Verify the nonintervening structured-Muon optimizer-state replay."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    git_commit,
)
from examples.nanogpt.parameter_trajectory import OPTIMIZER_PROBE_SCHEMA_VERSION
from examples.nanogpt.verify_full_state_functional_replay import parse_logged_losses


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = (
    "mai_124m_repaired_attention_cfc_late_cproj_"
    "optimizer_state_acquisition_plan_v1"
)
RESULT_SCHEMA = "mai_124m_late_cproj_optimizer_state_verification_v1"


def validate_probe(
    payload: dict[str, Any], *, expected_step: int, contract: dict[str, Any]
) -> tuple[str, dict[str, dict[str, torch.Tensor]]]:
    if payload.get("schema_version") != OPTIMIZER_PROBE_SCHEMA_VERSION:
        raise ValueError("optimizer probe schema mismatch")
    if int(payload.get("step", -1)) != expected_step:
        raise ValueError("optimizer probe step mismatch")
    if payload.get("targets") != [contract["target"]]:
        raise ValueError("optimizer probe target mismatch")
    if payload.get("layers") != contract["layers"]:
        raise ValueError("optimizer probe layers mismatch")
    if payload.get("storage_dtype") != contract["storage_dtype"]:
        raise ValueError("optimizer probe dtype mismatch")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("optimizer probe parameter payload missing")
    expected_names = {
        f"transformer.h.{layer}.mlp.c_proj.weight"
        for layer in contract["layers"]
    }
    if set(parameters) != expected_names:
        raise ValueError("optimizer probe parameter inventory mismatch")
    required_fields = set(contract["required_tensor_fields"])
    for name, state in parameters.items():
        if set(state) != required_fields:
            raise ValueError(f"optimizer state field mismatch: {name}")
        for field, value in state.items():
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise ValueError(f"nonfinite optimizer state: {name}/{field}")
        hyper = payload["hyperparameters"][name]
        if hyper["optimizer_kind"] != contract["optimizer_kind"]:
            raise ValueError("optimizer kind mismatch")
        if bool(hyper["error_feedback"]) != contract["error_feedback"]:
            raise ValueError("error-feedback switch mismatch")
        if not math.isclose(
            float(hyper["error_feedback_decay"]),
            float(contract["error_feedback_decay"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("error-feedback decay mismatch")
        momentum_expected = (
            float(hyper["momentum"])
            * state["momentum_buffer_before_step"].float()
            + state["gradient_after_clip"].float()
        )
        torch.testing.assert_close(
            state["momentum_buffer_after_step"].float(),
            momentum_expected,
            rtol=2e-6,
            atol=2e-6,
        )
        combined_expected = (
            state["gradient_after_clip"].float()
            + float(hyper["momentum"])
            * state["momentum_buffer_after_step"].float()
        )
        torch.testing.assert_close(
            state["combined_momentum_update"].float(),
            combined_expected,
            rtol=2e-6,
            atol=2e-6,
        )
        realized_expected = (
            state["weight_after_step"].float()
            - state["weight_before_step"].float()
        ) / float(hyper["lr"])
        torch.testing.assert_close(
            state["applied_direction_per_lr"].float(),
            realized_expected,
            rtol=2e-6,
            atol=2e-6,
        )
    return str(payload["run_identity_sha256"]), parameters


def compare_post_step_to_reference(
    parameters: dict[str, dict[str, torch.Tensor]], reference: dict[str, Any]
) -> None:
    buffers = reference.get("buffers")
    if not isinstance(buffers, dict):
        raise ValueError("reference snapshot has no persistent buffers")
    for name, state in parameters.items():
        observed = buffers.get(name)
        if not torch.is_tensor(observed) or not torch.equal(
            state["weight_after_step"].float(), observed.float()
        ):
            raise ValueError(f"post-step trajectory mismatch: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--reference-snapshot-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("optimizer-state plan schema mismatch")
    identity = plan["identity"]
    if file_sha256(args.config) != identity["candidate_config_sha256"]:
        raise ValueError("config SHA-256 mismatch")
    if file_sha256(Path(__file__)) != identity["verifier_sha256"]:
        raise ValueError("verifier SHA-256 mismatch")
    contract = plan["optimizer_state_contract"]
    probe_steps = contract["pre_step_probe_steps"]
    observed = sorted(args.probe_dir.glob("step_*.pt"))
    expected_names = [f"step_{step:06d}.pt" for step in probe_steps]
    if [path.name for path in observed] != expected_names:
        raise ValueError("optimizer probe file inventory mismatch")
    probe_hashes: dict[str, str] = {}
    reference_hashes: dict[str, str] = {}
    run_identity: str | None = None
    reference_steps = iter(contract["post_step_reference_steps"])
    for probe_step, path in zip(probe_steps, observed, strict=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        current_identity, parameters = validate_probe(
            payload, expected_step=probe_step, contract=contract
        )
        if run_identity is None:
            run_identity = current_identity
        elif current_identity != run_identity:
            raise ValueError("optimizer probe run identities disagree")
        probe_hashes[str(probe_step)] = file_sha256(path)
        if probe_step == 0:
            continue
        reference_step = next(reference_steps)
        reference_path = args.reference_snapshot_dir / (
            f"step_{reference_step:06d}.pt"
        )
        reference = torch.load(
            reference_path, map_location="cpu", weights_only=False
        )
        compare_post_step_to_reference(parameters, reference)
        reference_hashes[str(reference_step)] = file_sha256(reference_path)
    logged = parse_logged_losses(args.training_log)
    tolerance = float(plan["acceptance"]["curve_absolute_tolerance_ce"])
    curve_rows = []
    for raw_step, accepted_ce in plan["acceptance"][
        "accepted_validation_ce_by_step"
    ].items():
        step = int(raw_step)
        observed_ce = float(logged[step]["val"])
        delta = observed_ce - float(accepted_ce)
        curve_rows.append(
            {
                "step": step,
                "accepted_validation_ce": float(accepted_ce),
                "observed_validation_ce": observed_ce,
                "delta_ce": delta,
                "within_tolerance": abs(delta) <= tolerance,
            }
        )
    passed = all(row["within_tolerance"] for row in curve_rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "ACCEPTED_COADAPTED_LATE_CPROJ_OPTIMIZER_STATE_TRAJECTORY"
            if passed
            else "REJECTED_COADAPTED_LATE_CPROJ_OPTIMIZER_STATE_TRAJECTORY"
        ),
        "passed": passed,
        "execution": {
            "host": "PRO6",
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.verify_late_cproj_optimizer_state_trajectory",
            "parameter_updates": 0,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "config_sha256": file_sha256(args.config),
            "training_log_sha256": file_sha256(args.training_log),
            "run_identity_sha256": run_identity,
        },
        "inventory": {
            "probe_count": len(probe_hashes),
            "probe_sha256_by_step": probe_hashes,
            "reference_snapshot_sha256_by_step": reference_hashes,
            "parameter_count_per_probe": len(contract["layers"]),
        },
        "curve_reproduction": {
            "passed": passed,
            "tolerance_ce": tolerance,
            "rows": curve_rows,
        },
        "authorization": {
            "zero_update_state_transport_analysis": passed,
            "candidate_structure_implementation": False,
            "candidate_language_model_training": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
