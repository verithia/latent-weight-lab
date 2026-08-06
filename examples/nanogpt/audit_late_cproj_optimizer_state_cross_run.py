#!/usr/bin/env python3
"""Seal why the optimizer-state replay cannot use another run's trajectory."""

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
    PLAN_SCHEMA,
    parse_logged_losses,
    validate_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_SCHEMA = "mai_124m_late_cproj_optimizer_state_cross_run_audit_v1"


def mismatch_metrics(observed: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    observed = observed.float()
    reference = reference.float()
    difference = (observed - reference).double()
    return {
        "bitwise_equal": bool(torch.equal(observed, reference)),
        "maximum_absolute_difference": float(difference.abs().max()),
        "relative_frobenius_difference": float(
            difference.norm() / reference.double().norm().clamp_min(1e-30)
        ),
        "mismatched_elements": int(torch.count_nonzero(difference)),
        "numel": int(difference.numel()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--reference-snapshot-dir", type=Path, required=True)
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
        raise ValueError("optimizer-state acquisition plan schema mismatch")
    if file_sha256(args.config) != plan["identity"]["candidate_config_sha256"]:
        raise ValueError("config SHA-256 mismatch")
    status = json.loads(args.status.read_text())
    if status.get("state") != "finished" or status.get("exit_code") != 0:
        raise ValueError("training run did not finish cleanly")

    contract = plan["optimizer_state_contract"]
    probe_steps = contract["pre_step_probe_steps"]
    reference_steps = contract["post_step_reference_steps"]
    expected_probe_names = [f"step_{step:06d}.pt" for step in probe_steps]
    observed_probe_paths = sorted(args.probe_dir.glob("step_*.pt"))
    if [path.name for path in observed_probe_paths] != expected_probe_names:
        raise ValueError("optimizer probe inventory mismatch")

    payloads: dict[int, dict[str, Any]] = {}
    probe_hashes: dict[str, str] = {}
    run_identities: set[str] = set()
    for step, path in zip(probe_steps, observed_probe_paths, strict=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        identity, _parameters = validate_probe(
            payload, expected_step=step, contract=contract
        )
        run_identities.add(identity)
        payloads[step] = payload
        probe_hashes[str(step)] = file_sha256(path)
    if len(run_identities) != 1:
        raise ValueError("optimizer probes do not share one run identity")

    names = sorted(payloads[0]["parameters"])
    reference_zero_path = args.reference_snapshot_dir / "step_000000.pt"
    reference_zero = torch.load(
        reference_zero_path, map_location="cpu", weights_only=False
    )
    initialization_rows = []
    for name in names:
        initialization_rows.append(
            {
                "parameter": name,
                **mismatch_metrics(
                    payloads[0]["parameters"][name]["weight_before_step"],
                    reference_zero["buffers"][name],
                ),
            }
        )

    phase_rows = []
    reference_hashes: dict[str, str] = {"0": file_sha256(reference_zero_path)}
    for pre_step, post_step in zip(
        probe_steps[1:], reference_steps, strict=True
    ):
        reference_path = args.reference_snapshot_dir / f"step_{post_step:06d}.pt"
        reference = torch.load(
            reference_path, map_location="cpu", weights_only=False
        )
        reference_hashes[str(post_step)] = file_sha256(reference_path)
        for name in names:
            phase_rows.append(
                {
                    "pre_step": pre_step,
                    "post_step": post_step,
                    "parameter": name,
                    **mismatch_metrics(
                        payloads[pre_step]["parameters"][name][
                            "weight_after_step"
                        ],
                        reference["buffers"][name],
                    ),
                }
            )

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    terminal_payload = payloads[probe_steps[-1]]["parameters"]
    terminal_checkpoint_rows = [
        {
            "parameter": name,
            **mismatch_metrics(
                terminal_payload[name]["weight_after_step"],
                checkpoint["model"][name],
            ),
        }
        for name in names
    ]
    logged = parse_logged_losses(args.training_log)
    curve_rows = []
    tolerance = float(plan["acceptance"]["curve_absolute_tolerance_ce"])
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

    initialization_equal = all(row["bitwise_equal"] for row in initialization_rows)
    cross_run_equal = all(row["bitwise_equal"] for row in phase_rows)
    terminal_checkpoint_equal = all(
        row["bitwise_equal"] for row in terminal_checkpoint_rows
    )
    curve_passed = all(row["within_tolerance"] for row in curve_rows)
    if not initialization_equal or cross_run_equal:
        raise ValueError("audit did not reproduce the registered failure mode")
    if not terminal_checkpoint_equal or not curve_passed:
        raise ValueError("internal acquisition integrity failed")

    relative_values = [row["relative_frobenius_difference"] for row in phase_rows]
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "INVALID_CROSS_RUN_TRAJECTORY_IDENTITY",
        "scientific_conclusion": (
            "The optimizer probes are internally valid, start in the exact "
            "accepted gauge, reproduce the fixed CE curve, and end bitwise at "
            "their own checkpoint; independent CUDA training nevertheless "
            "diverges from the prior run's tensor path. Optimizer state and "
            "parameter targets must therefore be captured in one run."
        ),
        "execution": {
            "host": "PRO6",
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": (
                "examples.nanogpt.audit_late_cproj_optimizer_state_cross_run"
            ),
            "parameter_updates": 0,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "config_sha256": file_sha256(args.config),
            "training_log_sha256": file_sha256(args.training_log),
            "status_sha256": file_sha256(args.status),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "provenance_sha256": file_sha256(args.provenance),
            "optimizer_probe_run_identity_sha256": next(iter(run_identities)),
        },
        "internal_acquisition": {
            "probe_count": len(payloads),
            "parameter_count_per_probe": len(names),
            "all_probe_contracts_valid": True,
            "initialization_bitwise_equal_to_reference": initialization_equal,
            "terminal_probe_bitwise_equal_to_own_checkpoint": terminal_checkpoint_equal,
            "fixed_curve_within_tolerance": curve_passed,
            "probe_sha256_by_step": probe_hashes,
        },
        "cross_run_failure": {
            "all_post_step_weights_bitwise_equal": cross_run_equal,
            "minimum_relative_frobenius_difference": min(relative_values),
            "maximum_relative_frobenius_difference": max(relative_values),
            "initialization_rows": initialization_rows,
            "phase_rows": phase_rows,
            "reference_snapshot_sha256_by_step": reference_hashes,
        },
        "terminal_checkpoint_rows": terminal_checkpoint_rows,
        "curve_reproduction": {
            "tolerance_ce": tolerance,
            "rows": curve_rows,
        },
        "authorization": {
            "same_run_paired_state_acquisition_after_exact_config_mfu": True,
            "zero_update_state_transport_analysis": False,
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
