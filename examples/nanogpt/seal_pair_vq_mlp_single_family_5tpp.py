#!/usr/bin/env python3
"""Seal one arm of the 124M/5TPP c_fc-versus-c_proj Pair-VQ factorial."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Any

from examples.nanogpt.fixed_model_compute_equivalence import EvalPoint, summarize
from examples.nanogpt.seal_attention_pair_vq_vo_endpoint import (
    expected_eval_steps,
    file_sha256,
    integer,
    parse_fixed_losses,
    parse_json_lines,
    parse_key_value_line,
    parse_perf_tps,
)


PLAN_SCHEMA = "mai_124m_pair_vq_mlp_factorial_5tpp_localization_plan_v1"
CONFIG_SCHEMA = "mai_124m_qkonly_pairvq_single_mlp_family_5tpp_v1"
RESULT_SCHEMA = "mai_124m_pair_vq_single_mlp_family_5tpp_result_v1"
SUPPORTED_TARGETS = {"mlp.c_fc", "mlp.c_proj"}


def reference_points(result: dict[str, Any]) -> list[EvalPoint]:
    return [
        EvalPoint(float(row["step"]), float(row["validation_ce"]))
        for row in result["run"]["fixed_evaluations"]
    ]


def candidate_points(
    losses: dict[int, dict[str, float]], steps: list[int]
) -> list[EvalPoint]:
    return [
        EvalPoint(float(step), float(losses[step]["val"]))
        for step in steps
        if step > 0 and step in losses
    ]


def classification(valid: bool, target: str, gap: float, ceiling: float) -> str:
    token = target.removeprefix("mlp.").upper()
    if not valid:
        return f"INVALID_PAIR_VQ_{token}_124M_5TPP_LOCALIZATION"
    verdict = "NEGLIGIBLE" if gap <= ceiling else "MATERIAL"
    return f"PAIR_VQ_{token}_{verdict}_124M_5TPP"


def dense_control_is_unmodified(config: dict[str, Any], target: str) -> bool:
    block_targets = set(config.get("block_fht_targets", ()))
    int8_targets = set(config.get("block_fht_mlp_int8_lattice_targets", ()))
    if target in block_targets or target in int8_targets:
        return False
    if target == "mlp.c_fc":
        return not any(
            (
                config.get("block_fht_mlp_cfc_directed_product", False),
                config.get("block_fht_mlp_cfc_functional_shear", False),
                any(name.startswith("mlp.c_fc_group") for name in block_targets),
            )
        )
    if target == "mlp.c_proj":
        return not any(
            (
                config.get("block_fht_mlp_cproj_muon_matched_givens", False),
                any(
                    name.startswith("mlp.c_proj_")
                    for name in block_targets
                ),
            )
        )
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mfu-certificate", required=True, type=Path)
    parser.add_argument("--reference-result", required=True, type=Path)
    parser.add_argument("--qualification-result", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    config = json.loads(args.config.read_text())
    certificate = json.loads(args.mfu_certificate.read_text())
    reference = json.loads(args.reference_result.read_text())
    qualification = json.loads(args.qualification_result.read_text())
    status = json.loads(args.status.read_text())
    provenance = json.loads(args.provenance.read_text())
    log_text = args.training_log.read_text(errors="replace")
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("factorial localization plan schema mismatch")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("single-family config schema mismatch")

    config_sha = file_sha256(args.config)
    reference_sha = file_sha256(args.reference_result)
    qualification_sha = file_sha256(args.qualification_result)
    manifest = Path(config["data_dir"]) / "manifest.json"
    manifest_sha = file_sha256(manifest)
    targets = tuple(config.get("block_fht_mlp_pair_vq_targets", ()))
    if len(targets) != 1 or targets[0] not in SUPPORTED_TARGETS:
        raise ValueError("single-family seal requires exactly one Pair-VQ target")
    target = targets[0]
    dense_control = "mlp.c_proj" if target == "mlp.c_fc" else "mlp.c_fc"
    endpoint = config["endpoint_gate"]

    checks: dict[str, bool] = {
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "run_name_matches": status.get("run_name") == provenance.get("run_name"),
        "plan_matches_config": file_sha256(args.plan) == config["theory_plan_sha256"],
        "config_matches_provenance": provenance["config"]["sha256"] == config_sha,
        "config_matches_mfu": certificate["config"]["sha256"] == config_sha,
        "mfu_certificate_matches_provenance": provenance["performance_preflight"]["sha256"] == file_sha256(args.mfu_certificate),
        "mfu_passed": certificate.get("passed") is True,
        "mfu_floor": float(certificate["measurement"]["mfu_fraction"]) >= float(plan["performance_gate"]["minimum_mfu_fraction_each_arm"]),
        "persistent_state_matches_mfu": certificate["pair_vq_persistent_training_state"] == {
            "expected": int(endpoint["persistent_training_bytes_exact"]),
            "observed": [int(endpoint["persistent_training_bytes_exact"])],
            "passed": True,
        },
        "reference_matches_config": reference_sha == config["scientific_parent_sha256"],
        "qualification_matches_config": qualification_sha == config["qualification_dependency_sha256"],
        "qualification_is_material_mlp": qualification.get("classification") == "MLP_MATERIAL_QKONLY_PAIR_VQ_124M_5TPP",
        "manifest_matches_config": manifest_sha == config["data_manifest_sha256"],
        "manifest_matches_provenance": provenance["dataset_manifest"]["sha256"] == manifest_sha,
        "attention_pair_vq_disabled": config.get("block_fht_attn_pair_vq") is False,
        "single_pair_vq_target": list(targets) == endpoint["pair_vq_targets_required"],
        "dense_control_target_matches": dense_control == endpoint["dense_control_target_required"],
        "dense_control_is_unmodified": dense_control_is_unmodified(config, dense_control),
    }

    rng_rows = parse_json_lines(log_text, "rng_eval_metadata ")
    if len(rng_rows) != 1:
        raise ValueError(f"expected one rng_eval_metadata row, observed {len(rng_rows)}")
    rng = rng_rows[0]
    checks["fixed_eval_indices_match_reference"] = (
        rng["fixed_eval_indices_sha256"]
        == plan["matched_parent"]["fixed_eval_indices_sha256"]
    )
    checks["fixed_eval_protocol_match"] = (
        rng["eval_protocol_id"] == config["eval_protocol_id"]
    )

    required_steps = expected_eval_steps(
        int(config["max_iters"]), int(config["eval_interval"])
    )
    fixed_losses = parse_fixed_losses(log_text)
    checks["fixed_eval_inventory"] = sorted(fixed_losses) == required_steps
    fixed_values = [
        value
        for step in required_steps
        for value in fixed_losses.get(step, {}).values()
    ]
    checks["all_fixed_eval_losses_finite"] = (
        len(fixed_values) == 2 * len(required_steps)
        and all(math.isfinite(value) for value in fixed_values)
    )

    stats = parse_key_value_line(log_text, "mlp_pair_vq: ")
    checks.update(
        {
            "pair_vq_module_count": integer(stats, "modules") == int(endpoint["pair_vq_modules_required"]),
            "pair_vq_element_count": integer(stats, "elements") == int(endpoint["pair_vq_elements_required"]),
            "persistent_training_bytes_exact": integer(stats, "persistent_training_bytes") == int(endpoint["persistent_training_bytes_exact"]),
            "dense_master_weight_absent": stats["dense_master_weight"] == "disabled",
            "dense_momentum_absent": stats["dense_optimizer_momentum"] == "fp16_reserved_escape_capacity_ceiling",
            "raw_ambient_error_buffer_absent": stats["ambient_error_buffer"] == "disabled",
            "forward_visible_feedback": stats["forward_visible_feedback"] == "1",
        }
    )
    reserved = parse_json_lines(log_text, "pair_vq_reserved_escape ")
    if not reserved:
        raise ValueError("no pair_vq_reserved_escape events were logged")
    maximum_momentum = max(int(row["momentum_bytes"]) for row in reserved)
    checks["momentum_capacity"] = maximum_momentum <= int(
        endpoint["persistent_momentum_bytes_max_every_step"]
    )
    absent_scope = "c_proj" if target == "mlp.c_fc" else "c_fc"
    checks["absent_scope_has_zero_momentum"] = all(
        int(row[f"{absent_scope}_bytes"]) == 0 for row in reserved
    )

    terminal = fixed_losses.get(int(config["max_iters"]), {})
    candidate_ce = float(terminal.get("val", math.nan))
    reference_ce = float(plan["matched_parent"]["terminal_validation_ce"])
    gap_ce = candidate_ce - reference_ce
    checks["terminal_ce_finite"] = math.isfinite(candidate_ce)
    curve: dict[str, Any] | None = None
    curve_error: str | None = None
    try:
        curve = summarize(
            reference_points(reference),
            candidate_points(fixed_losses, required_steps),
        )
    except (KeyError, ValueError, ZeroDivisionError) as error:
        curve_error = str(error)
    checks["curve_analysis_available"] = curve is not None
    checks["common_loss_ratio_available"] = (
        curve is not None and "common_loss_ratio" in curve
    )

    tps = parse_perf_tps(log_text)
    if not tps:
        raise ValueError("no scientific perf rows were logged")
    valid = all(checks.values())
    threshold = float(
        plan["scientific_readout"]["per_arm_negligible_gap_ce_max"]
    )
    label = classification(valid, target, gap_ce, threshold)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": label,
        "valid": valid,
        "target": target,
        "dense_control_target": dense_control,
        "checks": checks,
        "terminal": {
            "train_ce": terminal.get("train"),
            "validation_ce": candidate_ce,
            "dense_mlp_parent_validation_ce": reference_ce,
            "candidate_minus_parent_ce": gap_ce,
            "single_family_negligible_gap_ce_max": threshold,
            "fixed_validation_curve": [
                {"step": step, "validation_ce": fixed_losses[step]["val"]}
                for step in required_steps
                if step in fixed_losses
            ],
        },
        "curve_analysis": curve,
        "curve_analysis_error": curve_error,
        "performance": {
            "preflight_mfu_fraction": certificate["measurement"]["mfu_fraction"],
            "preflight_tokens_per_second": certificate["measurement"]["tokens_per_second"],
            "scientific_median_tokens_per_second": statistics.median(tps),
            "scientific_mean_tokens_per_second": statistics.fmean(tps),
            "scientific_perf_samples": len(tps),
        },
        "capacity": {
            "pair_vq_modules": integer(stats, "modules"),
            "pair_vq_elements": integer(stats, "elements"),
            "codec_bytes": integer(stats, "codec_bytes"),
            "maximum_momentum_bytes": maximum_momentum,
            "momentum_byte_ceiling": int(endpoint["persistent_momentum_bytes_max_every_step"]),
            "compact_feedback_bytes": integer(stats, "compact_feedback_bytes"),
            "persistent_training_bytes": integer(stats, "persistent_training_bytes"),
        },
        "provenance": {
            "git_commit": provenance["repository"]["git_commit"],
            "plan_sha256": file_sha256(args.plan),
            "config_sha256": config_sha,
            "reference_result_sha256": reference_sha,
            "qualification_result_sha256": qualification_sha,
            "dataset_manifest_sha256": manifest_sha,
            "mfu_certificate_sha256": file_sha256(args.mfu_certificate),
            "runtime_provenance_sha256": file_sha256(args.provenance),
            "scientific_log_sha256": file_sha256(args.training_log),
            "terminal_status_sha256": file_sha256(args.status),
            "fixed_eval_indices_sha256": rng["fixed_eval_indices_sha256"],
        },
        "decision": {
            "single_family_negligible": valid and gap_ce <= threshold,
            "wait_for_other_preregistered_arm": True,
            "authorize_20tpp": False,
            "authorize_model_size_scale_up": False,
            "authorize_representation_change": False,
            "automatic_rerun": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
