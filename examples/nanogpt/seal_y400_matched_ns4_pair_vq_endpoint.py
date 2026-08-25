#!/usr/bin/env python3
"""Fail-closed seal for the Y400 124M matched-NS4 Pair-VQ MLP endpoint."""

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
    parse_key_value_line,
    parse_perf_tps,
)


RESULT_SCHEMA = "mai_124m_pair_vq_matched_ns4_quality_result_v1"
INTEGRATION_SCHEMA = (
    "mai_124m_pair_vq_matched_ns4_hierarchical_integration_passed_result_v1"
)
PARENT_SCHEMA = "mai_124m_pair_vq_matched_ns4_dense_result_v1"
LAUNCH_SCHEMA = "mai_124m_pair_vq_matched_ns4_quality_launch_v1"
PERF_RE = re.compile(
    r"^perf iter=(?P<step>\d+) tokens_per_s=(?P<tps>[-+0-9.eE]+) "
    r"iter_ms=(?P<iter_ms>[-+0-9.eE]+).*?opt_ms=(?P<opt_ms>[-+0-9.eE]+) "
    r".*?peak_mib=(?P<peak>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--parent-result", required=True, type=Path)
    parser.add_argument("--integration-result", required=True, type=Path)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--mfu-certificate", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def integer(values: dict[str, str], key: str) -> int:
    return int(values[key].replace(",", ""))


def json_equivalent(left: Any, right: Any) -> bool:
    """Compare checkpoint objects with their JSON-sidecar representation.

    PyTorch preserves tuples while JSON necessarily rehydrates them as lists.
    The sidecar was emitted from the same checkpoint identity, so comparison
    must use the canonical JSON domain rather than Python container types.
    """

    encode = lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return encode(left) == encode(right)


def checkpoint_dense_mlp_tensors(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    """Find persisted 768x3072/3072x768 matrices anywhere in the checkpoint."""

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


def classification(valid: bool, quality_passed: bool) -> str:
    if not valid:
        return "INVALID_MATCHED_NS4_PAIR_VQ_124M_0P5TPP"
    if quality_passed:
        return "PASS_MATCHED_NS4_PAIR_VQ_124M_0P5TPP"
    return "REJECT_MATCHED_NS4_PAIR_VQ_124M_0P5TPP"


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    config = json.loads(args.config.read_text())
    parent = json.loads(args.parent_result.read_text())
    integration = json.loads(args.integration_result.read_text())
    launch = json.loads(args.launch_receipt.read_text())
    certificate = json.loads(args.mfu_certificate.read_text())
    status = json.loads(args.status.read_text())
    metadata = json.loads(args.checkpoint_metadata.read_text())
    log_text = args.training_log.read_text(errors="replace")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if parent.get("schema_version") != PARENT_SCHEMA:
        raise ValueError("matched dense parent result schema mismatch")
    if integration.get("schema_version") != INTEGRATION_SCHEMA:
        raise ValueError("hierarchical integration result schema mismatch")
    if launch.get("schema_version") != LAUNCH_SCHEMA:
        raise ValueError("quality launch receipt schema mismatch")

    config_sha = file_sha256(args.config)
    parent_sha = file_sha256(args.parent_result)
    integration_sha = file_sha256(args.integration_result)
    launch_sha = file_sha256(args.launch_receipt)
    certificate_sha = file_sha256(args.mfu_certificate)
    status_sha = file_sha256(args.status)
    log_sha = file_sha256(args.training_log)
    checkpoint_sha = file_sha256(args.checkpoint)
    metadata_sha = file_sha256(args.checkpoint_metadata)
    manifest_path = Path(config["data_dir"]) / "manifest.json"
    manifest_sha = file_sha256(manifest_path)

    gate = config["endpoint_gate"]
    required_steps = expected_eval_steps(
        int(config["max_iters"]), int(config["eval_interval"])
    )
    fixed_losses = parse_fixed_losses(log_text)
    fixed_values = [
        value
        for step in required_steps
        for value in fixed_losses.get(step, {}).values()
    ]
    terminal = fixed_losses.get(int(config["max_iters"]), {})
    candidate_ce = float(checkpoint.get("best_val_loss", math.nan))
    parent_curve_values = parent["quality_gate"]["validation_cross_entropy"]
    parent_steps = parent["quality_gate"]["fixed_eval_steps"]
    parent_curve = [
        EvalPoint(float(step), float(loss))
        for step, loss in zip(parent_steps, parent_curve_values)
        if int(step) > 0
    ]
    candidate_curve = [
        EvalPoint(float(step), float(fixed_losses[step]["val"]))
        for step in required_steps
        if step > 0
    ]
    compute_analysis = summarize(parent_curve, candidate_curve)
    compute_penalty = float(
        compute_analysis["terminal_dense_penalty"]["candidate_over_dense_compute"]
    )
    parent_ce = float(gate["matched_dense_terminal_validation_ce"])
    gap_ce = candidate_ce - parent_ce

    stats = parse_key_value_line(log_text, "mlp_pair_vq: ")
    dense_findings = checkpoint_dense_mlp_tensors(checkpoint)
    perf_rows = []
    for line in log_text.splitlines():
        match = PERF_RE.match(line.strip())
        if match:
            perf_rows.append(
                {
                    "step": int(match.group("step")),
                    "tokens_per_second": float(match.group("tps")),
                    "iteration_ms": float(match.group("iter_ms")),
                    "optimizer_ms": float(match.group("opt_ms")),
                    "peak_memory_mib": float(match.group("peak")),
                }
            )
    tps = parse_perf_tps(log_text)
    if not perf_rows or not tps:
        raise ValueError("no scientific performance rows")

    optimizer_proof = 'pair_vq_optimizer_config {"hierarchical_feedback_fit":[true]}'
    metadata_identity = metadata.get("run_identity")
    checkpoint_identity = checkpoint.get("run_identity")
    certificate_source_sha = certificate["config"]["sha256"]
    integration_preflight = integration["exact_preflight"]
    preflight_gate = integration["frozen_gate_outcomes"]
    expected_persistent = int(gate["persistent_training_bytes_exact"])
    dense_state = int(gate["dense_fp32_weight_plus_muon_bytes"])
    observed_persistent = integer(stats, "persistent_training_bytes")
    compression = dense_state / observed_persistent

    checks: dict[str, bool] = {
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "status_commit_matches_launch": status.get("git_commit") == launch.get("git_commit"),
        "config_matches_status": status["config"]["sha256"] == config_sha,
        "config_matches_launch": launch["config"]["sha256"] == config_sha,
        "parent_matches_config": config["authorization_result_sha256"] == parent_sha,
        "parent_matches_endpoint_gate": gate["matched_dense_result_sha256"] == parent_sha,
        "integration_preregistered_current_config": integration["preregistered_endpoint"]["config_sha256"] == config_sha,
        "integration_governance_only_lineage": integration["preregistered_endpoint"]["scientific_runtime_fields_unchanged_after_preflight"] is True,
        "certificate_source_matches_integration": integration_preflight["source_config_sha256"] == certificate_source_sha,
        "certificate_matches_integration": integration_preflight["sha256"] == certificate_sha,
        "certificate_matches_launch": launch["mfu_preflight"]["sha256"] == certificate_sha,
        "manifest_matches_config": config["data_manifest_sha256"] == manifest_sha,
        "manifest_matches_status": status["dataset_manifest"]["sha256"] == manifest_sha,
        "manifest_matches_checkpoint": checkpoint_identity["data_manifest"]["sha256"] == manifest_sha,
        "log_matches_status": status["log_sha256"] == log_sha,
        "checkpoint_matches_status": status["checkpoint"]["sha256"] == checkpoint_sha,
        "metadata_matches_status": status["checkpoint_metadata"]["sha256"] == metadata_sha,
        "checkpoint_metadata_schema": metadata.get("checkpoint_schema_version") == "nanogpt_exact_resume_v2",
        "checkpoint_schema": checkpoint.get("schema_version") == "nanogpt_exact_resume_v2",
        "checkpoint_next_iter": int(checkpoint.get("next_iter", -1)) == int(config["max_iters"]),
        "metadata_next_iter": int(metadata.get("next_iter", -1)) == int(config["max_iters"]),
        "checkpoint_identity_matches_metadata": json_equivalent(checkpoint_identity, metadata_identity),
        "checkpoint_resolved_config_matches_metadata": json_equivalent(checkpoint_identity.get("resolved_config"), metadata_identity.get("resolved_config")),
        "checkpoint_fixed_eval_digest_matches_parent": checkpoint_identity["evaluation"]["fixed_eval_indices_sha256"] == parent["implementation"]["fixed_eval_indices_sha256"],
        "fixed_eval_inventory": sorted(fixed_losses) == required_steps,
        "all_fixed_losses_finite": len(fixed_values) == 2 * len(required_steps) and all(math.isfinite(value) for value in fixed_values),
        "checkpoint_best_val_finite": math.isfinite(candidate_ce),
        "checkpoint_and_log_terminal_agree": abs(candidate_ce - float(terminal.get("val", math.nan))) <= 5.1e-4,
        "native_block_fht_required": config["block_fht_native_extension_required"] is True,
        "native_block_fht_loaded": "block_fht_native_extension loaded=true" in log_text,
        "optimizer_hierarchical_fit_proved": optimizer_proof in log_text,
        "model_hierarchical_fit_enabled": config["block_fht_mlp_pair_vq_hierarchical_feedback_fit"] is True,
        "pair_vq_module_count": integer(stats, "modules") == 24,
        "pair_vq_element_count": integer(stats, "elements") == 56_623_104,
        "persistent_training_bytes_exact": observed_persistent == expected_persistent,
        "persistent_compression_gate": compression >= float(gate["minimum_persistent_mlp_state_compression"]),
        "dense_master_weight_absent": stats["dense_master_weight"] == "disabled",
        "dense_optimizer_momentum_absent": stats["dense_optimizer_momentum"] == "disabled",
        "raw_ambient_error_buffer_absent": stats["ambient_error_buffer"] == "disabled",
        "ambient_momentum_absent": integer(stats, "ambient_momentum_bytes") == 0,
        "compact_checkpoint_has_no_dense_mlp_tensor": not dense_findings,
        "preflight_passed": certificate.get("passed") is True,
        "preflight_all_frozen_gates_passed": all(bool(value) for value in preflight_gate.values()),
        "preflight_state_exact": certificate["pair_vq_persistent_training_state"]["observed"] == [expected_persistent],
        "preflight_mfu_floor": float(certificate["measurement"]["mfu_fraction"]) >= float(config["mfu_min_fraction"]),
        "preflight_optimizer_ceiling": float(certificate["measurement"]["timing_breakdown_ms"]["opt_ms"]) <= 495.0,
        "preflight_memory_ceiling": float(certificate["measurement"]["peak_mib"]) <= 35_803.43,
    }
    valid = all(checks.values())
    quality_checks = {
        "candidate_absolute_ce": candidate_ce <= float(gate["terminal_candidate_validation_ce_max"]),
        "candidate_minus_parent_ce": gap_ce <= float(gate["candidate_minus_matched_dense_validation_ce_max"]),
        "fixed_model_compute_penalty": compute_penalty <= float(gate["fixed_model_compute_penalty_max"]),
    }
    quality_passed = valid and all(quality_checks.values())

    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": classification(valid, quality_passed),
        "valid": valid,
        "passed": quality_passed,
        "checks": checks,
        "quality_checks": quality_checks,
        "terminal": {
            "train_ce": terminal.get("train"),
            "validation_ce_exact": candidate_ce,
            "validation_ce_logged": terminal.get("val"),
            "matched_dense_parent_validation_ce": parent_ce,
            "candidate_minus_parent_validation_ce": gap_ce,
            "candidate_validation_ce_max": gate["terminal_candidate_validation_ce_max"],
            "candidate_minus_parent_validation_ce_max": gate["candidate_minus_matched_dense_validation_ce_max"],
            "fixed_validation_curve": [
                {"step": step, "train_ce": fixed_losses[step]["train"], "validation_ce": fixed_losses[step]["val"]}
                for step in required_steps
            ],
            "matched_dense_fixed_validation_curve": [
                {"step": int(step), "validation_ce": float(loss)}
                for step, loss in zip(parent_steps, parent_curve_values)
            ],
            "fixed_model_compute_equivalent_penalty": compute_penalty,
            "fixed_model_compute_penalty_max": gate["fixed_model_compute_penalty_max"],
            "fixed_model_curve_analysis": compute_analysis,
        },
        "performance": {
            "elapsed_seconds": float(status["elapsed_seconds"]),
            "preflight_mfu_fraction": certificate["measurement"]["mfu_fraction"],
            "preflight_tokens_per_second": certificate["measurement"]["tokens_per_second"],
            "preflight_optimizer_ms": certificate["measurement"]["timing_breakdown_ms"]["opt_ms"],
            "scientific_median_tokens_per_second": statistics.median(tps),
            "scientific_mean_tokens_per_second": statistics.fmean(tps),
            "scientific_median_iteration_ms": statistics.median(row["iteration_ms"] for row in perf_rows),
            "scientific_median_optimizer_ms": statistics.median(row["optimizer_ms"] for row in perf_rows),
            "scientific_peak_memory_mib": max(row["peak_memory_mib"] for row in perf_rows),
            "scientific_perf_samples": len(perf_rows),
        },
        "compact_state": {
            "pair_vq_modules": integer(stats, "modules"),
            "pair_vq_elements": integer(stats, "elements"),
            "codec_bytes": integer(stats, "codec_bytes"),
            "compact_momentum_bytes": integer(stats, "compact_momentum_bytes"),
            "compact_feedback_bytes": integer(stats, "compact_feedback_bytes"),
            "persistent_training_bytes": observed_persistent,
            "dense_fp32_weight_plus_muon_bytes": dense_state,
            "compression": compression,
            "checkpoint_file_bytes": args.checkpoint.stat().st_size,
            "persisted_dense_mlp_tensors": dense_findings,
            "exact_compact_checkpoint_identity": not dense_findings,
        },
        "provenance": {
            "git_commit": status["git_commit"],
            "config_sha256": config_sha,
            "parent_result_sha256": parent_sha,
            "integration_result_sha256": integration_sha,
            "launch_receipt_sha256": launch_sha,
            "mfu_certificate_sha256": certificate_sha,
            "mfu_certificate_source_config_sha256": certificate_source_sha,
            "dataset_manifest_sha256": manifest_sha,
            "scientific_log_sha256": log_sha,
            "terminal_status_sha256": status_sha,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_metadata_sha256": metadata_sha,
            "entrypoint": status["entrypoint"],
            "literal_command": status["literal_command"],
        },
        "decision": {
            "authorize_5tpp": quality_passed,
            "authorize_sweep": False,
            "authorize_model_size_scale_up": False,
            "automatic_rerun": False,
            "next_action": (
                "Preregister one exact 124M/5TPP transfer only."
                if quality_passed
                else "Reject the matched-NS4 hierarchical Pair-VQ endpoint and return to theory; no sweep or scale-up."
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
