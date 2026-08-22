#!/usr/bin/env python3
"""Fail-closed seal for the 124M lazy Pair-VQ retraction endpoint."""

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


RESULT_SCHEMA = "mai_pair_vq_lazy_retraction_endpoint_result_v1"
PLAN_SCHEMA = "mai_pair_vq_lazy_retraction_plan_v1"
PARENT_CURVE = (
    EvalPoint(60.0, 6.2180),
    EvalPoint(120.0, 5.7258),
    EvalPoint(180.0, 5.4642),
    EvalPoint(238.0, 5.3610),
)
PERF_RE = re.compile(
    r"^perf iter=(?P<step>\d+) tokens_per_s=(?P<tps>[-+0-9.eE]+) "
    r"iter_ms=(?P<iter_ms>[-+0-9.eE]+).*?peak_mib=(?P<peak>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mfu-certificate", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def checkpoint_dense_mlp_tensors(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    """Find any persisted dense 768x3072 MLP matrix in the whole checkpoint."""

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


def classification(valid: bool, terminal_ce: float, ceiling: float) -> str:
    if not valid:
        return "INVALID_LAZY_RETRACTION_124M_0P5TPP"
    if terminal_ce <= ceiling:
        return "PASS_LAZY_RETRACTION_124M_0P5TPP"
    return "REJECT_LAZY_RETRACTION_124M_0P5TPP"


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    config = json.loads(args.config.read_text())
    certificate = json.loads(args.mfu_certificate.read_text())
    status = json.loads(args.status.read_text())
    metadata = json.loads(args.checkpoint_metadata.read_text())
    log_text = args.training_log.read_text(errors="replace")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("lazy-retraction plan schema mismatch")

    config_sha = file_sha256(args.config)
    plan_sha = file_sha256(args.plan)
    certificate_sha = file_sha256(args.mfu_certificate)
    status_sha = file_sha256(args.status)
    log_sha = file_sha256(args.training_log)
    checkpoint_sha = file_sha256(args.checkpoint)
    metadata_sha = file_sha256(args.checkpoint_metadata)
    manifest = Path(config["data_dir"]) / "manifest.json"
    manifest_sha = file_sha256(manifest)
    gate = config["endpoint_gate"]
    terminal_ceiling = float(gate["terminal_candidate_validation_ce_max"])
    parent_ce = float(gate["dense_parent_terminal_validation_ce"])

    required_steps = expected_eval_steps(
        int(config["max_iters"]), int(config["eval_interval"])
    )
    losses = parse_fixed_losses(log_text)
    terminal_log = losses.get(int(config["max_iters"]), {})
    terminal_ce_exact = float(checkpoint.get("best_val_loss", math.nan))
    terminal_gap = terminal_ce_exact - parent_ce
    dense_findings = checkpoint_dense_mlp_tensors(checkpoint)

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
    maximum_momentum = max(int(row["momentum_bytes"]) for row in reserved)
    final_momentum = int(reserved[-1]["momentum_bytes"])

    fixed_values = [
        value
        for step in required_steps
        for value in losses.get(step, {}).values()
    ]
    checks: dict[str, bool] = {
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "config_matches_status": status["config"]["sha256"] == config_sha,
        "config_matches_preflight": certificate["config"]["sha256"] == config_sha,
        "plan_matches_config": config["scientific_parent_sha256"] == plan_sha,
        "manifest_matches_config": config["data_manifest_sha256"] == manifest_sha,
        "manifest_matches_status": status["dataset_manifest"]["sha256"] == manifest_sha,
        "log_matches_status": status["log_sha256"] == log_sha,
        "checkpoint_matches_status": status["checkpoint"]["sha256"] == checkpoint_sha,
        "checkpoint_metadata_matches_status": status["checkpoint_metadata"]["sha256"] == metadata_sha,
        "preflight_passed": certificate.get("passed") is True,
        "preflight_mfu_floor": float(certificate["measurement"]["mfu_fraction"]) >= 0.20,
        "fixed_eval_inventory": sorted(losses) == required_steps,
        "all_fixed_losses_finite": len(fixed_values) == 2 * len(required_steps)
        and all(math.isfinite(value) for value in fixed_values),
        "checkpoint_next_iter": int(metadata["next_iter"]) == int(config["max_iters"]),
        "checkpoint_best_val_finite": math.isfinite(terminal_ce_exact),
        "checkpoint_and_log_terminal_agree": abs(
            terminal_ce_exact - float(terminal_log.get("val", math.nan))
        ) <= 5.1e-4,
        "compact_checkpoint_has_no_dense_mlp_tensor": not dense_findings,
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
    label = classification(valid, terminal_ce_exact, terminal_ceiling)
    passed = valid and terminal_ce_exact <= terminal_ceiling

    candidate_curve = [
        EvalPoint(float(step), float(losses[step]["val"]))
        for step in required_steps
        if step > 0
    ]
    curve = summarize(PARENT_CURVE, candidate_curve)
    fixed_model_penalty = float(
        curve["terminal_dense_penalty"]["candidate_over_dense_compute"]
    )
    perf_rows = []
    for line in log_text.splitlines():
        match = PERF_RE.match(line.strip())
        if match:
            perf_rows.append(
                {
                    "step": int(match.group("step")),
                    "tokens_per_second": float(match.group("tps")),
                    "iteration_ms": float(match.group("iter_ms")),
                    "peak_memory_mib": float(match.group("peak")),
                }
            )
    tps = parse_perf_tps(log_text)
    if not perf_rows or not tps:
        raise ValueError("no scientific performance rows")

    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": label,
        "valid": valid,
        "passed": passed,
        "checks": checks,
        "terminal": {
            "train_ce": terminal_log.get("train"),
            "validation_ce_exact": terminal_ce_exact,
            "validation_ce_logged": terminal_log.get("val"),
            "matched_parent_validation_ce": parent_ce,
            "candidate_minus_parent_validation_ce": terminal_gap,
            "candidate_validation_ce_max": terminal_ceiling,
            "fixed_validation_curve": [
                {"step": step, "validation_ce": losses[step]["val"]}
                for step in required_steps
            ],
            "fixed_model_compute_equivalent_penalty": fixed_model_penalty,
            "fixed_model_curve_analysis": curve,
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
            "scientific_peak_memory_mib": max(
                row["peak_memory_mib"] for row in perf_rows
            ),
            "scientific_perf_samples": len(perf_rows),
        },
        "compact_boundary": {
            "lazy_retraction_interval": config[
                "block_fht_mlp_pair_vq_lazy_retraction_interval"
            ],
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
            "persisted_dense_mlp_tensors": dense_findings,
            "checkpoint_is_compact_boundary": not dense_findings,
        },
        "persistent_state": {
            "maximum_adaptive_momentum_bytes": maximum_momentum,
            "final_adaptive_momentum_bytes": final_momentum,
            "adaptive_momentum_byte_ceiling": config[
                "persistent_momentum_bytes_max"
            ],
            "persistent_raw_ambient_momentum_tensors": 0,
            "new_persistent_matrix_state": False,
        },
        "provenance": {
            "git_commit": status["git_commit"],
            "plan_sha256": plan_sha,
            "config_sha256": config_sha,
            "mfu_certificate_sha256": certificate_sha,
            "dataset_manifest_sha256": manifest_sha,
            "scientific_log_sha256": log_sha,
            "terminal_status_sha256": status_sha,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_metadata_sha256": metadata_sha,
            "entrypoint": status["entrypoint"],
            "literal_command": status["literal_command"],
        },
        "decision": {
            "authorize_identical_124m_5tpp_preregistration": passed,
            "authorize_interval_sweep": False,
            "authorize_20tpp": False,
            "authorize_model_size_scale_up": False,
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
