#!/usr/bin/env python3
"""Fail-closed terminal verifier for the 124M QK-only functional-LWT run."""

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


RESULT_SCHEMA = "mai_124m_attention_qk_only_lwt_20tpp_verification_v1"
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
    rule = plan["decision_rule"]
    steps = [
        int(step)
        for step in plan["candidate"]["horizon"]["fixed_evaluation_steps"]
    ]
    parent = [float(value) for value in rule["qkv_parent_validation_ce"]]
    if len(steps) != len(parent):
        raise ValueError("registered fixed steps and parent curve differ in length")
    missing = [step for step in steps if step not in logged]
    if missing:
        raise ValueError(f"training log lacks fixed evaluations: {missing}")

    rows = []
    for step, parent_ce in zip(steps, parent):
        candidate_ce = float(logged[step]["validation"])
        rows.append(
            {
                "step": step,
                "candidate_train_ce": float(logged[step]["train"]),
                "candidate_validation_ce": candidate_ce,
                "qkv_parent_validation_ce": parent_ce,
                "gap_to_qkv_parent_ce": candidate_ce - parent_ce,
            }
        )

    terminal = rows[-1]
    terminal_maximum = float(rule["terminal_validation_ce_maximum"])
    minimum_improvement = float(rule["minimum_terminal_improvement_over_qkv"])
    maximum_curve_gap = float(rule["maximum_fixed_curve_gap_to_qkv"])
    terminal_improvement = (
        terminal["qkv_parent_validation_ce"]
        - terminal["candidate_validation_ce"]
    )
    maximum_observed_curve_gap = max(
        row["gap_to_qkv_parent_ce"] for row in rows
    )
    terminal_passed = terminal["candidate_validation_ce"] <= terminal_maximum
    improvement_passed = terminal_improvement >= minimum_improvement
    curve_passed = maximum_observed_curve_gap <= maximum_curve_gap
    return {
        "rows": rows,
        "terminal_validation_ce": terminal["candidate_validation_ce"],
        "terminal_gap_to_dense_ce": terminal["candidate_validation_ce"]
        - float(rule["dense_terminal_validation_ce"]),
        "terminal_improvement_over_qkv_ce": terminal_improvement,
        "maximum_fixed_curve_gap_to_qkv_ce": maximum_observed_curve_gap,
        "terminal_maximum": terminal_maximum,
        "minimum_terminal_improvement_over_qkv": minimum_improvement,
        "maximum_fixed_curve_gap_to_qkv": maximum_curve_gap,
        "terminal_passed": terminal_passed,
        "improvement_passed": improvement_passed,
        "curve_passed": curve_passed,
        "scientific_gate_passed": (
            terminal_passed and improvement_passed and curve_passed
        ),
    }


def _list(config: dict[str, Any], key: str) -> list[Any]:
    value = config.get(key)
    return [] if value is None else list(value)


def architecture_checks(
    config: dict[str, Any], model_config: dict[str, Any]
) -> dict[str, bool]:
    target = "attn.c_attn.qk_headwise"

    def checks_for(payload: dict[str, Any], prefix: str) -> dict[str, bool]:
        return {
            f"{prefix}_qk_only_generated": _list(payload, "block_fht_targets")
            == [target],
            f"{prefix}_qk_output_gain": _list(
                payload, "block_fht_output_gain_targets"
            )
            == [target],
            f"{prefix}_qk_cayley": _list(
                payload, "block_fht_attn_cayley_targets"
            )
            == [target],
            f"{prefix}_qk_output_cayley": _list(
                payload, "block_fht_attn_cayley_output_targets"
            )
            == [target],
            f"{prefix}_qk_bilateral_cayley": _list(
                payload, "block_fht_attn_cayley_bilateral_targets"
            )
            == [target],
            f"{prefix}_qk_rank_64": int(
                (payload.get("block_fht_attn_cayley_ranks") or {}).get(target, -1)
            )
            == 64,
            f"{prefix}_mlp_cfc_dense": payload.get(
                "block_fht_mlp_cfc_directed_product", False
            )
            is not True,
            f"{prefix}_mlp_cproj_no_givens": payload.get(
                "block_fht_mlp_cproj_muon_matched_givens", False
            )
            is not True,
            f"{prefix}_mlp_cproj_no_product_fht": int(
                payload.get("block_fht_cproj_product_fht_factors") or 0
            )
            == 0,
            f"{prefix}_mlp_cproj_no_lowrank": int(
                payload.get("block_fht_cproj_lowrank_rank") or 0
            )
            == 0,
        }

    return {**checks_for(config, "config"), **checks_for(model_config, "checkpoint")}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
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
    mfu = json.loads(args.mfu_result.read_text())
    metadata = json.loads(args.run_metadata.read_text())
    config = json.loads(args.config.read_text())
    status = json.loads(args.status.read_text())
    provenance = json.loads(args.provenance.read_text())
    immutable = metadata["immutable_inputs"]

    checks = {
        "plan": file_sha256(args.plan) == immutable["plan"]["sha256"],
        "mfu_result": file_sha256(args.mfu_result)
        == immutable["mfu_result"]["sha256"],
        "config": file_sha256(args.config) == immutable["config"]["sha256"],
        "provenance": file_sha256(args.provenance)
        == immutable["provenance"]["sha256"],
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "status_run_name": status.get("run_name")
        == metadata["scientific_identity"]["run_name"],
        "provenance_run_id": provenance.get("run_id")
        == metadata["scientific_identity"]["run_id"],
        "provenance_commit": provenance["repository"]["git_commit"]
        == metadata["scientific_identity"]["execution_git_commit"],
        "provenance_config": provenance["config"]["sha256"]
        == immutable["config"]["sha256"],
        "provenance_mfu": provenance["performance_preflight"]["sha256"]
        == immutable["performance_preflight"]["sha256"],
        "provenance_dataset": provenance["dataset_manifest"]["sha256"]
        == immutable["dataset_manifest"]["sha256"],
        "mfu_passed": mfu.get("passed") is True,
        "thresholds_unchanged": plan["decision_rule"][
            "threshold_changed_after_measurement"
        ]
        is False,
    }

    logged = parse_logged_losses(args.training_log)
    dense_muon_matrix_count = parse_dense_muon_matrix_count(args.training_log)
    checks["dense_v_cproj_and_full_mlp_owned_by_muon"] = (
        dense_muon_matrix_count == 48
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
            "PASS_FUNCTIONAL_LWT_QK_ONLY_124M_20TPP"
            if decision["scientific_gate_passed"]
            else "FAIL_FUNCTIONAL_LWT_QK_ONLY_124M_20TPP"
        ),
        "verified": True,
        "scientific_gate_passed": decision["scientific_gate_passed"],
        "identity_checks": checks,
        "optimizer_ownership": {
            "dense_muon_matrix_tensors": dense_muon_matrix_count,
            "interpretation": (
                "12 V plus 12 attention c_proj plus 12 c_fc plus 12 MLP c_proj"
            ),
        },
        "fixed_evaluation": decision,
        "checkpoint": {
            **resume,
            **inventory,
            "sha256": file_sha256(args.checkpoint),
            "metadata_sha256": file_sha256(
                args.checkpoint.with_name("ckpt.meta.json")
            ),
        },
        "artifacts": {
            "training_log_sha256": file_sha256(args.training_log),
            "status_sha256": file_sha256(args.status),
            "provenance_sha256": file_sha256(args.provenance),
            "config_sha256": file_sha256(args.config),
            "verifier_sha256": file_sha256(Path(__file__)),
        },
        "decision": {
            "functional_lwt_qk_only_partial_attention_accepted": decision[
                "scientific_gate_passed"
            ],
            "generated_v_accepted": False,
            "full_attention_replacement_accepted": False,
            "full_mlp_replacement_accepted": False,
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
