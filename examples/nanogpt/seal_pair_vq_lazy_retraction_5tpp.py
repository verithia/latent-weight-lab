#!/usr/bin/env python3
"""Fail-closed seal for the preregistered 124M/5TPP lazy retraction test."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.fixed_model_compute_equivalence import EvalPoint, summarize
from examples.nanogpt.seal_attention_pair_vq_vo_endpoint import (
    expected_eval_steps,
    file_sha256,
    parse_fixed_losses,
    parse_json_lines,
    parse_perf_tps,
)


RESULT_SCHEMA = "mai_pair_vq_lazy_retraction_124m_5tpp_result_v1"
PLAN_SCHEMA = "mai_pair_vq_lazy_retraction_plan_v1"
REGISTRATION_SCHEMA = "mai_pair_vq_lazy_retraction_124m_5tpp_registration_v1"
PERF_RE = re.compile(
    r"^perf iter=(?P<step>\d+) tokens_per_s=(?P<tps>[-+0-9.eE]+) "
    r"iter_ms=(?P<iter_ms>[-+0-9.eE]+).*?opt_ms=(?P<opt>[-+0-9.eE]+) "
    r".*?peak_mib=(?P<peak>[-+0-9.eE]+)"
)


def classification(valid: bool, quality_pass: bool, compute_pass: bool) -> str:
    if not valid:
        return "INVALID_LAZY_RETRACTION_124M_5TPP"
    if quality_pass and compute_pass:
        return "PASS_LAZY_RETRACTION_124M_5TPP"
    return "REJECT_LAZY_RETRACTION_124M_5TPP"


def dense_mlp_tensors(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if tuple(value.shape) in {(768, 3072), (3072, 768)}:
                findings.append(
                    {"path": path, "shape": list(value.shape), "dtype": str(value.dtype)}
                )
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(checkpoint, "checkpoint")
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--qualification-result", required=True, type=Path)
    parser.add_argument("--causal-comparator", required=True, type=Path)
    parser.add_argument("--parent-result", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mfu-certificate", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    registration = json.loads(args.registration.read_text())
    qualification = json.loads(args.qualification_result.read_text())
    comparator = json.loads(args.causal_comparator.read_text())
    parent = json.loads(args.parent_result.read_text())
    config = json.loads(args.config.read_text())
    certificate = json.loads(args.mfu_certificate.read_text())
    status = json.loads(args.status.read_text())
    metadata = json.loads(args.checkpoint_metadata.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    log_text = args.training_log.read_text(errors="replace")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("lazy-retraction plan schema mismatch")
    if registration.get("schema_version") != REGISTRATION_SCHEMA:
        raise ValueError("lazy-retraction registration schema mismatch")

    hashes = {
        "plan": file_sha256(args.plan),
        "registration": file_sha256(args.registration),
        "qualification": file_sha256(args.qualification_result),
        "comparator": file_sha256(args.causal_comparator),
        "parent": file_sha256(args.parent_result),
        "config": file_sha256(args.config),
        "mfu": file_sha256(args.mfu_certificate),
        "log": file_sha256(args.training_log),
        "status": file_sha256(args.status),
        "checkpoint": file_sha256(args.checkpoint),
        "metadata": file_sha256(args.checkpoint_metadata),
    }
    manifest = Path(config["data_dir"]) / "manifest.json"
    hashes["manifest"] = file_sha256(manifest)

    required_steps = expected_eval_steps(
        int(config["max_iters"]), int(config["eval_interval"])
    )
    losses = parse_fixed_losses(log_text)
    fixed_values = [
        value
        for step in required_steps
        for value in losses.get(step, {}).values()
    ]
    terminal_log = losses.get(int(config["max_iters"]), {})
    terminal_ce = float(checkpoint.get("best_val_loss", math.nan))
    parent_ce = float(config["endpoint_gate"]["matched_parent_terminal_validation_ce"])
    quality_ceiling = float(
        config["endpoint_gate"]["terminal_candidate_validation_ce_max"]
    )
    gap_ce = terminal_ce - parent_ce

    parent_curve = [
        EvalPoint(float(row["step"]), float(row["validation_ce"]))
        for row in parent["run"]["fixed_evaluations"]
    ]
    candidate_curve = [
        EvalPoint(
            float(step),
            terminal_ce if step == int(config["max_iters"]) else float(losses[step]["val"]),
        )
        for step in required_steps
        if step > 0
    ]
    curve = summarize(parent_curve, candidate_curve)
    compute_penalty = float(
        curve["terminal_dense_penalty"]["candidate_over_dense_compute"]
    )
    compute_ceiling = float(
        config["endpoint_gate"]["fixed_model_compute_equivalent_penalty_max"]
    )
    quality_pass = math.isfinite(terminal_ce) and terminal_ce <= quality_ceiling
    compute_pass = math.isfinite(compute_penalty) and compute_penalty <= compute_ceiling

    refresh_rows = [
        row
        for row in parse_json_lines(log_text, "muon_matched_givens_refresh ")
        if int(row.get("lazy_retraction", 0)) == 1
        and int(row.get("compact_boundary", 0)) == 1
        and int(row.get("retracted", 0)) == 1
    ]
    normalized_boundaries = sorted(
        {int(row["optimizer_step"]) + 1 for row in refresh_rows}
    )
    expected_boundaries = sorted(
        set(range(8, int(config["max_iters"]) + 1, 8))
        | set(config["block_fht_mlp_pair_vq_lazy_retraction_forced_steps"])
    )
    reserved = parse_json_lines(log_text, "pair_vq_reserved_escape ")
    if not reserved:
        raise ValueError("no adaptive momentum records")
    maximum_momentum = max(int(row["momentum_bytes"]) for row in reserved)
    compact_findings = dense_mlp_tensors(checkpoint)

    rng_rows = parse_json_lines(log_text, "rng_eval_metadata ")
    if len(rng_rows) != 1:
        raise ValueError(f"expected one rng metadata row, got {len(rng_rows)}")
    rng = rng_rows[0]
    checks: dict[str, bool] = {
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "config_matches_status": status["config"]["sha256"] == hashes["config"],
        "config_matches_preflight": certificate["config"]["sha256"] == hashes["config"],
        "plan_matches_config": config["theory_plan_sha256"] == hashes["plan"],
        "registration_matches_config": config["preregistration_sha256"] == hashes["registration"],
        "qualification_matches_config": config["qualification_dependency_sha256"] == hashes["qualification"],
        "comparator_matches_config": config["causal_comparator_sha256"] == hashes["comparator"],
        "parent_matches_config": config["scientific_parent_sha256"] == hashes["parent"],
        "qualification_passed": qualification.get("passed") is True,
        "comparator_is_material_failure": comparator.get("classification")
        == "MLP_MATERIAL_QKONLY_PAIR_VQ_124M_5TPP",
        "manifest_matches_config": config["data_manifest_sha256"] == hashes["manifest"],
        "manifest_matches_status": status["dataset_manifest"]["sha256"] == hashes["manifest"],
        "log_matches_status": status["log_sha256"] == hashes["log"],
        "checkpoint_matches_status": status["checkpoint"]["sha256"] == hashes["checkpoint"],
        "metadata_matches_status": status["checkpoint_metadata"]["sha256"] == hashes["metadata"],
        "preflight_passed": certificate.get("passed") is True,
        "preflight_mfu_floor": float(certificate["measurement"]["mfu_fraction"]) >= 0.20,
        "fixed_eval_protocol_matches": rng["eval_protocol_id"] == config["eval_protocol_id"],
        "fixed_eval_indices_match_parent": rng["fixed_eval_indices_sha256"]
        == parent["run"]["fixed_eval_indices_sha256"],
        "fixed_eval_inventory": sorted(losses) == required_steps,
        "all_fixed_losses_finite": len(fixed_values) == 2 * len(required_steps)
        and all(math.isfinite(value) for value in fixed_values),
        "checkpoint_next_iter": int(metadata["next_iter"]) == int(config["max_iters"]),
        "checkpoint_best_val_finite": math.isfinite(terminal_ce),
        "checkpoint_and_log_terminal_agree": abs(
            terminal_ce - float(terminal_log.get("val", math.nan))
        ) <= 5.1e-4,
        "compact_checkpoint_has_no_dense_mlp_tensor": not compact_findings,
        "all_expected_retraction_boundaries": normalized_boundaries == expected_boundaries,
        "twenty_four_mlp_matrices_per_boundary": len(refresh_rows)
        == 24 * len(expected_boundaries),
        "all_boundary_weights_match_compact_state": all(
            int(row.get("virtual_weight_matches_compact_state", 0)) == 1
            for row in refresh_rows
        ),
        "no_persistent_raw_ambient_momentum": all(
            int(row.get("persistent_raw_ambient_momentum_tensors", -1)) == 0
            for row in refresh_rows
        ),
        "adaptive_momentum_within_ceiling": maximum_momentum
        <= int(config["persistent_momentum_bytes_max"]),
    }
    valid = all(checks.values())
    passed = valid and quality_pass and compute_pass

    perf_rows = []
    for line in log_text.splitlines():
        match = PERF_RE.match(line.strip())
        if match:
            perf_rows.append(
                {
                    "tokens_per_second": float(match.group("tps")),
                    "iteration_ms": float(match.group("iter_ms")),
                    "optimizer_ms": float(match.group("opt")),
                    "peak_memory_mib": float(match.group("peak")),
                }
            )
    tps = parse_perf_tps(log_text)
    if not perf_rows or not tps:
        raise ValueError("no scientific performance rows")
    old_ce = float(comparator["terminal"]["validation_ce"])
    old_tps = float(comparator["performance"]["scientific_median_tokens_per_second"])

    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": classification(valid, quality_pass, compute_pass),
        "valid": valid,
        "passed": passed,
        "checks": checks,
        "scientific_gates": {
            "quality_passed": quality_pass,
            "compute_equivalent_passed": compute_pass,
        },
        "terminal": {
            "train_ce": terminal_log.get("train"),
            "validation_ce_exact": terminal_ce,
            "validation_ce_logged": terminal_log.get("val"),
            "matched_parent_validation_ce": parent_ce,
            "candidate_minus_parent_validation_ce": gap_ce,
            "candidate_validation_ce_max": quality_ceiling,
            "fixed_model_compute_equivalent_penalty": compute_penalty,
            "fixed_model_compute_equivalent_penalty_max": compute_ceiling,
            "fixed_model_curve_analysis": curve,
            "fixed_validation_curve": [
                {"step": step, "validation_ce": losses[step]["val"]}
                for step in required_steps
            ],
        },
        "causal_comparison": {
            "per_step_retraction_validation_ce": old_ce,
            "lazy_minus_per_step_validation_ce": terminal_ce - old_ce,
            "per_step_retraction_median_tokens_per_second": old_tps,
            "lazy_median_tokens_per_second": statistics.median(tps),
            "lazy_throughput_ratio_to_per_step": statistics.median(tps) / old_tps,
            "interpretation": (
                "amortizing retraction improves throughput but does not close the "
                "late-horizon full-MLP quality gap"
            ),
        },
        "performance": {
            "elapsed_seconds": status["elapsed_seconds"],
            "preflight_mfu_fraction": certificate["measurement"]["mfu_fraction"],
            "preflight_tokens_per_second": certificate["measurement"]["tokens_per_second"],
            "scientific_median_tokens_per_second": statistics.median(tps),
            "scientific_mean_tokens_per_second": statistics.fmean(tps),
            "scientific_median_iteration_ms": statistics.median(
                row["iteration_ms"] for row in perf_rows
            ),
            "scientific_median_optimizer_ms": statistics.median(
                row["optimizer_ms"] for row in perf_rows
            ),
            "scientific_peak_memory_mib": max(
                row["peak_memory_mib"] for row in perf_rows
            ),
            "scientific_perf_samples": len(perf_rows),
        },
        "compact_boundary": {
            "lazy_retraction_interval": config[
                "block_fht_mlp_pair_vq_lazy_retraction_interval"
            ],
            "normalized_boundary_count": len(normalized_boundaries),
            "normalized_boundaries": normalized_boundaries,
            "boundary_matrix_records": len(refresh_rows),
            "minimum_feedback_codec_energy_recovery": min(
                float(row["feedback_codec_energy_recovery"]) for row in refresh_rows
            ),
            "minimum_conserved_requested_step_energy_recovery": min(
                float(row["conserved_requested_step_energy_recovery"])
                for row in refresh_rows
            ),
            "minimum_requested_update_cosine": min(
                float(row["requested_update_cosine"]) for row in refresh_rows
            ),
            "persisted_dense_mlp_tensors": compact_findings,
            "checkpoint_is_compact_boundary": not compact_findings,
        },
        "persistent_state": {
            "maximum_adaptive_momentum_bytes": maximum_momentum,
            "final_adaptive_momentum_bytes": int(reserved[-1]["momentum_bytes"]),
            "adaptive_momentum_byte_ceiling": config["persistent_momentum_bytes_max"],
            "persistent_raw_ambient_momentum_tensors": 0,
            "new_persistent_matrix_state": False,
        },
        "provenance": {
            "git_commit": status["git_commit"],
            **{f"{key}_sha256": value for key, value in hashes.items()},
            "entrypoint": status["entrypoint"],
            "literal_command": status["literal_command"],
        },
        "decision": {
            "authorize_compact_attention_vo_composition": passed,
            "reject_lazy_retraction_as_sufficient": valid and not passed,
            "authorize_interval_sweep": False,
            "authorize_automatic_rerun": False,
            "authorize_20tpp": False,
            "authorize_model_size_scale_up": False,
            "next_action": (
                "return to theory; no additional training is authorized by this result"
            ),
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
