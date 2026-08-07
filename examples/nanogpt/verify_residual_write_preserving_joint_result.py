#!/usr/bin/env python3
"""Fail-closed terminal verifier for the residual-write-preserving joint run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.verify_resume_checkpoint_envelope import verify as verify_resume


RESULT_SCHEMA = "mai_124m_residual_write_preserving_joint_5tpp_verification_v1"
EVAL_RE = re.compile(
    r"^step\s+(?P<step>\d+):\s+train loss\s+(?P<train>\S+),\s+"
    r"val loss\s+(?P<val>\S+)\s*$"
)
MUON_RE = re.compile(r"\bmatrix_tensors=(?P<count>\d+)\b")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_logged_losses(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = EVAL_RE.match(line.strip())
        if match is None:
            continue
        step = int(match.group("step"))
        train = float(match.group("train"))
        validation = float(match.group("val"))
        if not math.isfinite(train) or not math.isfinite(validation):
            raise ValueError(f"non-finite fixed loss at step {step}")
        rows[step] = {"train": train, "validation": validation}
    return rows


def parse_dense_muon_matrix_count(path: Path) -> int:
    counts = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("optimizer=muon "):
            continue
        match = MUON_RE.search(line)
        if match is not None:
            counts.append(int(match.group("count")))
    if len(counts) != 1:
        raise ValueError(f"expected one Muon ownership line, found {len(counts)}")
    return counts[0]


def fixed_curve_decision(
    plan: dict[str, Any], logged: dict[int, dict[str, float]]
) -> dict[str, Any]:
    candidate = plan["candidate"]
    rule = plan["decision_rule"]
    steps = [int(step) for step in candidate["fixed_evaluation_steps"]]
    parent = [float(value) for value in rule["qkv_parent_validation_ce"]]
    dense = [float(value) for value in rule["fair_blockwise_dense_validation_ce"]]
    if len(steps) != len(parent) or len(steps) != len(dense):
        raise ValueError("registered fixed curves differ in length")
    missing = [step for step in steps if step not in logged]
    if missing:
        raise ValueError(f"training log lacks fixed evaluations: {missing}")
    rows = []
    for step, parent_ce, dense_ce in zip(steps, parent, dense):
        candidate_ce = float(logged[step]["validation"])
        rows.append(
            {
                "step": step,
                "candidate_train_ce": float(logged[step]["train"]),
                "candidate_validation_ce": candidate_ce,
                "qkv_parent_validation_ce": parent_ce,
                "gap_to_qkv_parent_ce": candidate_ce - parent_ce,
                "fair_blockwise_dense_validation_ce": dense_ce,
                "gap_to_fair_blockwise_dense_ce": candidate_ce - dense_ce,
            }
        )
    terminal = rows[-1]
    terminal_maximum = float(rule["terminal_validation_ce_maximum"])
    terminal_gap_maximum = float(rule["terminal_gap_to_qkv_parent_maximum"])
    curve_gap_maximum = float(rule["maximum_fixed_curve_gap_to_qkv_parent"])
    terminal_passed = terminal["candidate_validation_ce"] <= terminal_maximum
    terminal_gap_passed = terminal["gap_to_qkv_parent_ce"] <= terminal_gap_maximum
    curve_passed = max(row["gap_to_qkv_parent_ce"] for row in rows) <= curve_gap_maximum
    return {
        "rows": rows,
        "terminal_validation_ce": terminal["candidate_validation_ce"],
        "terminal_gap_to_qkv_parent_ce": terminal["gap_to_qkv_parent_ce"],
        "terminal_gap_to_fair_blockwise_dense_ce": terminal[
            "gap_to_fair_blockwise_dense_ce"
        ],
        "maximum_fixed_curve_gap_to_qkv_parent_ce": max(
            row["gap_to_qkv_parent_ce"] for row in rows
        ),
        "terminal_maximum": terminal_maximum,
        "terminal_gap_maximum": terminal_gap_maximum,
        "curve_gap_maximum": curve_gap_maximum,
        "terminal_passed": terminal_passed,
        "terminal_gap_passed": terminal_gap_passed,
        "curve_passed": curve_passed,
        "scientific_gate_passed": (
            terminal_passed and terminal_gap_passed and curve_passed
        ),
    }


def tensor_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model")
    optimizer = payload.get("optimizer")
    if not isinstance(model, dict) or not isinstance(optimizer, dict):
        raise ValueError("checkpoint lacks model or optimizer state")
    model_tensors = [value for value in model.values() if torch.is_tensor(value)]
    optimizer_tensors: list[torch.Tensor] = []

    def visit(value: Any) -> None:
        if torch.is_tensor(value):
            optimizer_tensors.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(optimizer)
    model_finite = all(
        not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())
        for tensor in model_tensors
    )
    optimizer_finite = all(
        not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())
        for tensor in optimizer_tensors
    )
    if not model_finite or not optimizer_finite:
        raise ValueError("checkpoint contains non-finite model or optimizer state")
    return {
        "model_tensor_count": len(model_tensors),
        "optimizer_tensor_count": len(optimizer_tensors),
        "all_model_tensors_finite": model_finite,
        "all_optimizer_tensors_finite": optimizer_finite,
        "optimizer_state_present": bool(optimizer_tensors),
        "rng_state_present": all(
            payload.get(key) is not None
            for key in (
                "train_data_generator_state",
                "cpu_torch_rng_state",
                "cuda_rng_states",
                "python_random_state",
                "numpy_rng_state",
            )
        ),
    }


def architecture_checks(config: dict[str, Any], model_config: dict[str, Any]) -> dict[str, bool]:
    expected_targets = ["attn.c_attn.qk_headwise", "attn.c_attn.v"]
    expected_schedule = [22, 22, 22, 22, 22, 22]
    checks = {
        "generated_qkv_only": list(config.get("block_fht_targets", []))
        == expected_targets,
        "attention_cproj_dense": "attn.c_proj"
        not in list(config.get("block_fht_targets", [])),
        "mlp_cfc_directed_product": config.get("block_fht_mlp_cfc_directed_product")
        is True,
        "mlp_cfc_schedule": list(
            config.get("block_fht_mlp_cfc_directed_product_schedule", [])
        )
        == expected_schedule,
        "mlp_cproj_dense": config.get(
            "block_fht_mlp_cproj_muon_matched_givens", False
        )
        is False,
        "mlp_cproj_no_layer_mask": list(
            config.get("block_fht_mlp_cproj_muon_matched_givens_layers", [])
        )
        == [],
        "mlp_cproj_no_product_fht": int(
            config.get("block_fht_cproj_product_fht_factors", 0)
        )
        == 0,
        "mlp_cproj_no_lowrank": int(config.get("block_fht_cproj_lowrank_rank", 0))
        == 0,
        "checkpoint_generated_qkv_only": list(
            model_config.get("block_fht_targets", [])
        )
        == expected_targets,
        "checkpoint_mlp_cfc_directed_product": model_config.get(
            "block_fht_mlp_cfc_directed_product"
        )
        is True,
        "checkpoint_mlp_cproj_dense": model_config.get(
            "block_fht_mlp_cproj_muon_matched_givens"
        )
        is False,
    }
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--accounting-correction", type=Path, required=True)
    parser.add_argument("--mfu-result", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    correction = json.loads(args.accounting_correction.read_text())
    mfu = json.loads(args.mfu_result.read_text())
    metadata = json.loads(args.run_metadata.read_text())
    config = json.loads(args.config.read_text())
    status = json.loads(args.status.read_text())
    provenance = json.loads(args.provenance.read_text())

    immutable = metadata["immutable_inputs"]
    checks = {
        "plan": file_sha256(args.plan) == immutable["plan"]["sha256"],
        "accounting_correction": file_sha256(args.accounting_correction)
        == immutable["accounting_correction"]["sha256"],
        "mfu_result": file_sha256(args.mfu_result)
        == immutable["mfu_result"]["sha256"],
        "config": file_sha256(args.config) == immutable["config"]["sha256"],
        "provenance": file_sha256(args.provenance)
        == immutable["provenance"]["sha256"],
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "provenance_commit": provenance["repository"]["git_commit"]
        == metadata["scientific_identity"]["execution_git_commit"],
        "provenance_config": provenance["config"]["sha256"]
        == immutable["config"]["sha256"],
        "provenance_mfu": provenance["performance_preflight"]["sha256"]
        == immutable["performance_preflight"]["sha256"],
        "provenance_dataset": provenance["dataset_manifest"]["sha256"]
        == immutable["dataset_manifest"]["sha256"],
        "mfu_passed": mfu.get("passed") is True,
        "accounting_claim_pinned": config.get("accounting_correction_sha256")
        == immutable["accounting_correction"]["sha256"],
        "thresholds_unchanged": plan["decision_rule"]
        ["threshold_changed_after_measurement"]
        is False,
    }

    logged = parse_logged_losses(args.training_log)
    dense_muon_matrix_count = parse_dense_muon_matrix_count(args.training_log)
    checks["both_dense_residual_write_families_owned_by_muon"] = (
        dense_muon_matrix_count == 24
    )
    resume = verify_resume(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint.get("next_iter", -1)) != int(config["max_iters"]):
        raise ValueError("terminal checkpoint next_iter mismatch")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model config is absent")
    checks.update(architecture_checks(config, model_config))
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("terminal identity checks failed: " + ", ".join(failed))

    decision = fixed_curve_decision(plan, logged)
    inventory = tensor_inventory(checkpoint)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "PASS_RESIDUAL_WRITE_PRESERVING_JOINT_124M_5TPP"
            if decision["scientific_gate_passed"]
            else "FAIL_RESIDUAL_WRITE_PRESERVING_JOINT_124M_5TPP"
        ),
        "verified": True,
        "scientific_gate_passed": decision["scientific_gate_passed"],
        "identity_checks": checks,
        "optimizer_ownership": {
            "dense_muon_matrix_tensors": dense_muon_matrix_count,
            "interpretation": "12 attention c_proj plus 12 MLP c_proj dense matrices",
        },
        "fixed_evaluation": decision,
        "checkpoint": {
            **resume,
            **inventory,
            "sha256": file_sha256(args.checkpoint),
            "metadata_sha256": file_sha256(args.checkpoint.with_name("ckpt.meta.json")),
        },
        "artifacts": {
            "training_log_sha256": file_sha256(args.training_log),
            "status_sha256": file_sha256(args.status),
            "provenance_sha256": file_sha256(args.provenance),
            "config_sha256": file_sha256(args.config),
            "verifier_sha256": file_sha256(Path(__file__)),
        },
        "decision": {
            "residual_write_preserving_joint_accepted": decision[
                "scientific_gate_passed"
            ],
            "full_attention_replacement_accepted": False,
            "full_mlp_replacement_accepted": False,
            "registered_parameter_reduction_attributed_to_qkv_only": True,
            "cfc_component_registered_parameter_reduction": 0,
            "inference_parameter_reduction": 0,
            "automatic_rerun_authorized": False,
            "different_structure_authorized": False,
            "larger_rung_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
