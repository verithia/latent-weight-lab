#!/usr/bin/env python3
"""Fail-closed seal for the 124M compact attention V/output endpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "mai_124m_attention_pair_vq_vo_plan_v1"
RESULT_SCHEMA = "mai_124m_attention_pair_vq_vo_endpoint_result_v1"
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LOSS_PATTERN = re.compile(
    rf"^step (?P<step>\d+): train loss (?P<train>{FLOAT}), "
    rf"val loss (?P<val>{FLOAT})$"
)
PERF_PATTERN = re.compile(
    rf"^perf iter=(?P<step>\d+) tokens_per_s=(?P<tps>{FLOAT})\b"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_fixed_losses(text: str) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in text.splitlines():
        match = LOSS_PATTERN.match(line.strip())
        if match is None:
            continue
        rows[int(match.group("step"))] = {
            "train": float(match.group("train")),
            "val": float(match.group("val")),
        }
    return rows


def parse_perf_tps(text: str) -> list[float]:
    return [
        float(match.group("tps"))
        for line in text.splitlines()
        if (match := PERF_PATTERN.match(line.strip())) is not None
    ]


def parse_key_value_line(text: str, prefix: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one {prefix!r} line, observed {len(lines)}")
    values: dict[str, str] = {}
    for token in lines[0][len(prefix) :].strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value.rstrip(",")
    return values


def parse_json_lines(text: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith(prefix):
            value = json.loads(line[len(prefix) :])
            if not isinstance(value, dict):
                raise ValueError(f"{prefix!r} payload must be an object")
            rows.append(value)
    return rows


def expected_eval_steps(max_iters: int, eval_interval: int) -> list[int]:
    steps = list(range(0, max_iters + 1, eval_interval))
    if not steps or steps[-1] != max_iters:
        steps.append(max_iters)
    return steps


def compute_equivalent_penalty(
    gap_ce: float, parent_ce: float, alpha_eff: float
) -> float:
    if parent_ce <= 0.0 or alpha_eff <= 0.0:
        raise ValueError("parent CE and alpha_eff must be positive")
    return math.exp(gap_ce / (alpha_eff * parent_ce))


def integer(values: dict[str, str], key: str) -> int:
    return int(values[key].replace(",", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mfu-certificate", required=True, type=Path)
    parser.add_argument("--parent-result", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha-eff", type=float, default=0.0700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    config = json.loads(args.config.read_text())
    certificate = json.loads(args.mfu_certificate.read_text())
    parent = json.loads(args.parent_result.read_text())
    status = json.loads(args.status.read_text())
    provenance = json.loads(args.provenance.read_text())
    log_text = args.training_log.read_text(errors="replace")
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("attention Pair-VQ plan schema mismatch")

    plan_impl = plan["implementation_gates"]
    plan_science = plan["scientific_gates"]
    config_sha = file_sha256(args.config)
    parent_sha = file_sha256(args.parent_result)
    manifest = Path(config["data_dir"]) / "manifest.json"
    manifest_sha = file_sha256(manifest)
    checks = {
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "run_name_matches": status.get("run_name") == provenance.get("run_name"),
        "config_matches_provenance": provenance["config"]["sha256"] == config_sha,
        "config_matches_mfu": certificate["config"]["sha256"] == config_sha,
        "mfu_certificate_matches_provenance": provenance["performance_preflight"]["sha256"] == file_sha256(args.mfu_certificate),
        "mfu_passed": certificate.get("passed") is True,
        "mfu_floor": float(certificate["measurement"]["mfu_fraction"]) >= float(plan_impl["minimum_exact_config_mfu_fraction"]),
        "parent_matches_config": parent_sha == config["scientific_parent_sha256"],
        "manifest_matches_config": manifest_sha == config["data_manifest_sha256"],
        "manifest_matches_provenance": provenance["dataset_manifest"]["sha256"] == manifest_sha,
    }

    rng_rows = parse_json_lines(log_text, "rng_eval_metadata ")
    if len(rng_rows) != 1:
        raise ValueError(f"expected one rng_eval_metadata row, observed {len(rng_rows)}")
    rng = rng_rows[0]
    parent_fixed_digest = parent["terminal"]["fixed_eval_indices_sha256"]
    checks["fixed_eval_indices_match_parent"] = (
        rng["fixed_eval_indices_sha256"] == parent_fixed_digest
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
    required_modules = int(config["endpoint_gate"]["all_pair_vq_modules_required"])
    required_elements = int(plan["representation"]["all_pair_vq_matrix_elements"])
    required_persistent = int(
        config["mfu_preflight_pair_vq_persistent_training_bytes_exact"]
    )
    checks.update(
        {
            "pair_vq_module_count": integer(stats, "modules") == required_modules,
            "pair_vq_element_count": integer(stats, "elements") == required_elements,
            "persistent_training_bytes_exact": integer(stats, "persistent_training_bytes") == required_persistent,
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
    momentum_ceiling = int(plan_impl["maximum_momentum_bytes"])
    checks["momentum_capacity"] = maximum_momentum <= momentum_ceiling

    terminal = fixed_losses.get(int(config["max_iters"]), {})
    candidate_ce = float(terminal.get("val", math.nan))
    parent_ce = float(parent["terminal"]["validation_ce"])
    gap_ce = candidate_ce - parent_ce
    compute_penalty = compute_equivalent_penalty(gap_ce, parent_ce, args.alpha_eff)
    quality_limit = float(
        plan_science[
            "candidate_minus_matched_compact_parent_terminal_validation_ce_max"
        ]
    )
    compute_limit = float(
        plan_science["candidate_over_parent_compute_equivalent_max"]
    )
    checks["terminal_quality_gate"] = math.isfinite(gap_ce) and gap_ce <= quality_limit
    checks["compute_equivalent_gate"] = (
        math.isfinite(compute_penalty) and compute_penalty <= compute_limit
    )
    passed = all(checks.values())
    tps = parse_perf_tps(log_text)
    if not tps:
        raise ValueError("no scientific perf rows were logged")

    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": (
            "PASS_COMPACT_FULL_REPLACEMENT_124M_0P5TPP"
            if passed
            else "REJECT_COMPACT_FULL_REPLACEMENT_124M_0P5TPP"
        ),
        "passed": passed,
        "checks": checks,
        "terminal": {
            "train_ce": terminal.get("train"),
            "validation_ce": candidate_ce,
            "matched_parent_validation_ce": parent_ce,
            "gap_ce": gap_ce,
            "quality_gate_ce": quality_limit,
            "alpha_eff": args.alpha_eff,
            "compute_equivalent_penalty": compute_penalty,
            "compute_equivalent_gate": compute_limit,
            "fixed_validation_curve": [
                {"step": step, "validation_ce": fixed_losses[step]["val"]}
                for step in required_steps
                if step in fixed_losses
            ],
        },
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
            "momentum_byte_ceiling": momentum_ceiling,
            "compact_feedback_bytes": integer(stats, "compact_feedback_bytes"),
            "persistent_training_bytes": integer(stats, "persistent_training_bytes"),
            "persistent_raw_ambient_momentum_tensors": 0,
        },
        "provenance": {
            "git_commit": provenance["repository"]["git_commit"],
            "plan_sha256": file_sha256(args.plan),
            "config_sha256": config_sha,
            "parent_result_sha256": parent_sha,
            "dataset_manifest_sha256": manifest_sha,
            "mfu_certificate_sha256": file_sha256(args.mfu_certificate),
            "runtime_provenance_sha256": file_sha256(args.provenance),
            "scientific_log_sha256": file_sha256(args.training_log),
            "terminal_status_sha256": file_sha256(args.status),
            "fixed_eval_indices_sha256": rng["fixed_eval_indices_sha256"],
        },
        "decision": {
            "authorize_identical_124m_5tpp_registration": passed,
            "authorize_124m_20tpp": False,
            "authorize_model_size_scale_up": False,
            "next_on_failure": plan_science["next_on_failure"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
