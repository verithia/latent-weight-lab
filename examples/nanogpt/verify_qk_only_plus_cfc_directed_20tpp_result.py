#!/usr/bin/env python3
"""Fail-closed verifier for QK-only plus procedural-c_fc at 124M/20TPP."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.verify_residual_write_preserving_joint_result import (
    file_sha256,
    parse_dense_muon_matrix_count,
    parse_logged_losses,
    tensor_inventory,
)
from examples.nanogpt.verify_resume_checkpoint_envelope import verify as verify_resume


RESULT_SCHEMA = "mai_124m_qk_only_plus_cfc_directed_20tpp_verification_v1"
PARAM_RE = re.compile(
    r"^parameters:\s+total=(?P<total>[\d,]+)\s+trainable=(?P<trainable>[\d,]+)$"
)
GENERATED_RE = re.compile(
    r"^block_fht:\s+modules=(?P<modules>\d+)\s+"
    r"generated=(?P<generated>[\d,]+)\s+latent=(?P<latent>[\d,]+)$"
)


def fixed_curve_decision(
    plan: dict[str, Any], logged: dict[int, dict[str, float]]
) -> dict[str, Any]:
    rule = plan["decision_rule"]
    steps = [int(step) for step in plan["candidate"]["horizon"]["fixed_evaluation_steps"]]
    parent = [float(value) for value in rule["qk_only_parent_validation_ce"]]
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
                "qk_only_parent_validation_ce": parent_ce,
                "gap_to_qk_only_parent_ce": candidate_ce - parent_ce,
            }
        )
    terminal = rows[-1]
    terminal_maximum = float(rule["terminal_validation_ce_maximum"])
    curve_gap_maximum = float(rule["maximum_fixed_curve_gap_to_qk_only_parent"])
    maximum_curve_gap = max(row["gap_to_qk_only_parent_ce"] for row in rows)
    terminal_passed = terminal["candidate_validation_ce"] <= terminal_maximum
    curve_passed = maximum_curve_gap <= curve_gap_maximum
    return {
        "rows": rows,
        "terminal_validation_ce": terminal["candidate_validation_ce"],
        "terminal_gap_to_qk_only_parent_ce": terminal["gap_to_qk_only_parent_ce"],
        "terminal_gap_to_dense_ce": terminal["candidate_validation_ce"]
        - float(rule["dense_terminal_validation_ce"]),
        "maximum_fixed_curve_gap_to_qk_only_parent_ce": maximum_curve_gap,
        "terminal_maximum": terminal_maximum,
        "curve_gap_maximum": curve_gap_maximum,
        "terminal_passed": terminal_passed,
        "curve_passed": curve_passed,
        "scientific_gate_passed": terminal_passed and curve_passed,
    }


def architecture_checks(
    config: dict[str, Any], model_config: dict[str, Any]
) -> dict[str, bool]:
    qk = "attn.c_attn.qk_headwise"
    schedule = [22] * 6

    def checks(payload: dict[str, Any], prefix: str) -> dict[str, bool]:
        targets = list(payload.get("block_fht_targets") or [])
        return {
            f"{prefix}_qk_only_generated": targets == [qk],
            f"{prefix}_v_dense": "attn.c_attn.v" not in targets,
            f"{prefix}_attention_cproj_dense": "attn.c_proj" not in targets,
            f"{prefix}_qk_rank64": int(
                (payload.get("block_fht_attn_cayley_ranks") or {}).get(qk, -1)
            )
            == 64,
            f"{prefix}_qk_output_gain": list(
                payload.get("block_fht_output_gain_targets") or []
            )
            == [qk],
            f"{prefix}_cfc_directed_product": payload.get(
                "block_fht_mlp_cfc_directed_product"
            )
            is True,
            f"{prefix}_cfc_schedule": list(
                payload.get("block_fht_mlp_cfc_directed_product_schedule") or []
            )
            == schedule,
            f"{prefix}_cfc_error_feedback": payload.get(
                "block_fht_mlp_cfc_directed_product_error_feedback"
            )
            is True,
            f"{prefix}_cfc_error_feedback_decay1": float(
                payload.get("block_fht_mlp_cfc_directed_product_error_feedback_decay", -1)
            )
            == 1.0,
            f"{prefix}_mlp_cproj_dense": payload.get(
                "block_fht_mlp_cproj_muon_matched_givens", False
            )
            is False,
            f"{prefix}_mlp_cproj_no_product_fht": int(
                payload.get("block_fht_cproj_product_fht_factors") or 0
            )
            == 0,
            f"{prefix}_mlp_cproj_no_lowrank": int(
                payload.get("block_fht_cproj_lowrank_rank") or 0
            )
            == 0,
        }

    return {**checks(config, "config"), **checks(model_config, "checkpoint")}


def parse_runtime_accounting(path: Path) -> dict[str, int]:
    parameter_rows = []
    generated_rows = []
    for line in path.read_text(errors="replace").splitlines():
        parameter = PARAM_RE.match(line.strip())
        if parameter:
            parameter_rows.append(
                {key: int(value.replace(",", "")) for key, value in parameter.groupdict().items()}
            )
        generated = GENERATED_RE.match(line.strip())
        if generated:
            generated_rows.append(
                {key: int(value.replace(",", "")) for key, value in generated.groupdict().items()}
            )
    if len(parameter_rows) != 1 or len(generated_rows) != 1:
        raise ValueError("expected one parameter and one BlockFHT accounting line")
    return {**parameter_rows[0], **generated_rows[0]}


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
        "mfu_result": file_sha256(args.mfu_result) == immutable["mfu_result"]["sha256"],
        "config": file_sha256(args.config) == immutable["config"]["sha256"],
        "provenance": file_sha256(args.provenance) == immutable["provenance"]["sha256"],
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "status_run_name": status.get("run_name") == metadata["scientific_identity"]["run_name"],
        "provenance_run_id": provenance.get("run_id") == metadata["scientific_identity"]["run_id"],
        "provenance_commit": provenance["repository"]["git_commit"]
        == metadata["scientific_identity"]["execution_git_commit"],
        "provenance_config": provenance["config"]["sha256"] == immutable["config"]["sha256"],
        "provenance_mfu": provenance["performance_preflight"]["sha256"]
        == immutable["archived_mfu_certificate"]["sha256"],
        "provenance_dataset": provenance["dataset_manifest"]["sha256"]
        == immutable["dataset_manifest_sha256"],
        "mfu_passed": mfu.get("passed") is True,
        "accounting_claim_pinned": config.get("accounting_correction_sha256")
        == immutable["accounting_correction"]["sha256"],
        "accounting_science_unchanged": correction["unchanged_science"]
        ["scientific_parameter_updates_before_correction"]
        == 0,
        "thresholds_unchanged": plan["decision_rule"]["threshold_changed_after_measurement"]
        is False,
    }
    logged = parse_logged_losses(args.training_log)
    dense_muon_count = parse_dense_muon_matrix_count(args.training_log)
    checks["dense_v_and_both_cproj_families_owned_by_muon"] = dense_muon_count == 36
    accounting = parse_runtime_accounting(args.training_log)
    checks["ordinary_registered_count"] = accounting["trainable"] == 85605360
    checks["generated_weight_count"] = accounting["generated"] == 42467328
    checks["procedural_coordinate_count"] = accounting["latent"] == 5007600

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
            "PASS_QK_ONLY_PLUS_CFC_DIRECTED_124M_20TPP"
            if decision["scientific_gate_passed"]
            else "FAIL_QK_ONLY_PLUS_CFC_DIRECTED_124M_20TPP"
        ),
        "verified": True,
        "scientific_gate_passed": decision["scientific_gate_passed"],
        "identity_checks": checks,
        "optimizer_ownership": {
            "dense_muon_matrix_tensors": dense_muon_count,
            "interpretation": "12 V plus 12 attention c_proj plus 12 MLP c_proj",
        },
        "runtime_accounting": accounting,
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
            "qk_plus_cfc_partial_architecture_accepted": decision["scientific_gate_passed"],
            "generated_v_accepted": False,
            "full_attention_replacement_accepted": False,
            "full_mlp_replacement_accepted": False,
            "cfc_realized_algorithmic_parameter_reduction": 0,
            "inference_parameter_reduction": 0,
            "inference_flop_reduction": 0,
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
