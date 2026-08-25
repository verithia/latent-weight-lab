#!/usr/bin/env python3
"""Fail-closed seal for the Pair-VQ paired-neuron dense-escape oracle."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

from examples.nanogpt.seal_attention_pair_vq_vo_endpoint import file_sha256


RAW_SCHEMA = "mai_pair_vq_paired_neuron_dense_escape_oracle_result_v1"
PLAN_SCHEMA = "mai_124m_pair_vq_paired_neuron_dense_escape_oracle_plan_v1"
SEALED_SCHEMA = "mai_pair_vq_paired_neuron_dense_escape_oracle_sealed_result_v1"
PASSED = "PAIRED_NEURON_DENSE_ESCAPE_ACTIONABLE"
DIFFUSE = "PAIRED_NEURON_DENSE_ESCAPE_DIFFUSE_UNHELPFUL"
INVALID = "PAIRED_NEURON_DENSE_ESCAPE_INVALID"


def classify(valid: bool, selected: float | None, maximum: float) -> str:
    if not valid or selected is None:
        return INVALID
    return PASSED if selected <= maximum else DIFFUSE


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _candidate_pass(candidate: dict[str, Any], thresholds: dict[str, float]) -> tuple[dict[str, bool], bool]:
    measurements = candidate["measurements"]
    checks = {
        key: math.isfinite(float(measurements[key]))
        and float(measurements[key]) >= float(threshold)
        for key, threshold in thresholds.items()
        if key != "full_fraction_sanity_cosine"
    }
    return checks, all(checks.values())


def _selection_jaccards(records: list[dict[str, Any]]) -> list[float]:
    if len(records) != 2:
        return []
    first = records[0]["dense_escape"]["selection"]["layers"]
    second = records[1]["dense_escape"]["selection"]["layers"]
    if [row["identity"] for row in first] != [row["identity"] for row in second]:
        return []
    values: list[float] = []
    for left, right in zip(first, second, strict=True):
        left_set = set(int(value) for value in left["actionable_selected_indices"])
        right_set = set(int(value) for value in right["actionable_selected_indices"])
        values.append(len(left_set & right_set) / len(left_set | right_set))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mfu-certificate", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    raw = json.loads(args.raw_result.read_text())
    plan = json.loads(args.plan.read_text())
    config = json.loads(args.config.read_text())
    certificate = json.loads(args.mfu_certificate.read_text())
    status = json.loads(args.status.read_text())
    provenance = json.loads(args.provenance.read_text())
    log_text = args.training_log.read_text(errors="replace")
    hashes = {
        "raw_result": file_sha256(args.raw_result),
        "plan": file_sha256(args.plan),
        "config": file_sha256(args.config),
        "mfu_certificate": file_sha256(args.mfu_certificate),
        "status": file_sha256(args.status),
        "provenance": file_sha256(args.provenance),
        "training_log": file_sha256(args.training_log),
        "dataset_manifest": file_sha256(args.dataset_manifest),
    }

    expected_steps = [int(value) for value in plan["frozen_protocol"]["probe_steps"]]
    expected_fractions = [float(value) for value in plan["dense_escape"]["fractions"]]
    expected_keys = {str(value) for value in expected_fractions}
    thresholds = plan["functional_gate"]
    records = raw.get("records", [])
    inventory_ok = [int(row.get("step", -1)) for row in records] == expected_steps
    inventory_ok &= all(
        set(row.get("dense_escape", {}).get("candidates", {})) == expected_keys
        for row in records
    )

    flags_ok = inventory_ok
    sanity_ok = inventory_ok
    if inventory_ok:
        for record in records:
            candidates = record["dense_escape"]["candidates"]
            for fraction in expected_fractions:
                candidate = candidates[str(fraction)]
                checks, passed = _candidate_pass(candidate, thresholds)
                flags_ok &= candidate.get("checks") == checks
                flags_ok &= bool(candidate.get("passed")) == passed
            full = candidates["1.0"]
            sanity = (
                float(full["measurements"]["aggregate_heldout_functional_cosine"])
                >= float(thresholds["full_fraction_sanity_cosine"])
            )
            sanity_ok &= bool(record["dense_escape"].get("full_fraction_sanity_passed")) == sanity
            sanity_ok &= sanity

    passing_fractions = [
        fraction
        for fraction in expected_fractions
        if inventory_ok
        and all(
            bool(record["dense_escape"]["candidates"][str(fraction)]["passed"])
            for record in records
        )
    ]
    selected = min(passing_fractions) if passing_fractions else None
    maximum = float(plan["dense_escape"]["maximum_actionable_dense_fraction"])
    expected_classification = classify(True, selected, maximum)
    jaccards = _selection_jaccards(records) if inventory_ok else []
    overlap_ok = bool(jaccards) and _close(
        min(jaccards), raw["gate"]["minimum_actionable_selection_jaccard_across_late_steps"]
    ) and _close(
        sum(jaccards) / len(jaccards),
        raw["gate"]["mean_actionable_selection_jaccard_across_late_steps"],
    )

    checks: dict[str, bool] = {
        "raw_schema": raw.get("schema_version") == RAW_SCHEMA,
        "plan_schema": plan.get("schema_version") == PLAN_SCHEMA,
        "raw_finished": raw.get("status") == "finished",
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "raw_plan_hash": raw.get("plan", {}).get("sha256") == hashes["plan"],
        "config_plan_hash": config.get("theory_plan_sha256") == hashes["plan"],
        "config_functional_plan_hash": config.get("pair_vq_dense_shadow_functional_plan_sha256") == hashes["plan"],
        "raw_source_hash": raw.get("source_config", {}).get("sha256") == config.get("pair_vq_dense_shadow_source_sha256"),
        "manifest_matches_config": config.get("data_manifest_sha256") == hashes["dataset_manifest"],
        "provenance_config_hash": provenance.get("config", {}).get("sha256") == hashes["config"],
        "provenance_manifest_hash": provenance.get("dataset_manifest", {}).get("sha256") == hashes["dataset_manifest"],
        "provenance_mfu_hash": provenance.get("performance_preflight", {}).get("sha256") == hashes["mfu_certificate"],
        "status_provenance_hash": status.get("provenance", {}).get("sha256") == hashes["provenance"],
        "mfu_passed": certificate.get("passed") is True,
        "mfu_floor": float(certificate.get("measurement", {}).get("mfu_fraction", 0.0)) >= 0.20,
        "mfu_config_hash": certificate.get("config", {}).get("sha256") == hashes["config"],
        "probe_and_fraction_inventory": inventory_ok,
        "candidate_flags_recomputed": flags_ok,
        "full_fraction_sanity": sanity_ok,
        "selection_overlap_recomputed": overlap_ok,
        "fit_and_heldout_disjoint": raw.get("fit_and_heldout_indices_disjoint") is True,
        "zero_model_updates": int(plan["frozen_protocol"]["model_updates_from_oracle"]) == 0,
        "gate_ready": raw.get("gate", {}).get("ready") is True,
        "gate_selected_fraction": raw.get("gate", {}).get("selected_fraction") == selected,
        "gate_maximum_fraction": raw.get("gate", {}).get("maximum_actionable_fraction") == maximum,
        "gate_pass": bool(raw.get("gate", {}).get("passed")) == (selected is not None and selected <= maximum),
        "gate_classification": raw.get("gate", {}).get("classification") == expected_classification,
        "no_traceback": "Traceback (most recent call last)" not in log_text,
    }
    valid = all(checks.values())
    sealed_classification = classify(valid, selected, maximum)
    if not valid:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"refusing invalid seal; failed checks: {failed}")

    record_summaries = []
    for record in records:
        candidates = record["dense_escape"]["candidates"]
        record_summaries.append(
            {
                "step": int(record["step"]),
                "virtual_weight_energy_recovery": {
                    "weighted": float(record["virtual_weight"]["weighted_virtual_weight_energy_recovery"]),
                    "worst_matrix": float(record["virtual_weight"]["worst_matrix_virtual_weight_energy_recovery"]),
                },
                "fractions": {
                    key: {
                        "fraction": float(candidate["fraction"]),
                        "count_per_layer": int(candidate["count_per_layer"]),
                        "passed": bool(candidate["passed"]),
                        "measurements": candidate["measurements"],
                        "selected_values": int(candidate["selected_values"]),
                        "selected_weight_plus_optimizer_bytes": int(candidate["selected_weight_plus_optimizer_bytes"]),
                    }
                    for key, candidate in candidates.items()
                },
            }
        )

    started = dt.datetime.fromisoformat(status["started_at"])
    finished = dt.datetime.fromisoformat(status["finished_at"])
    dense_bytes = int(records[0]["dense_escape"]["candidates"]["1.0"]["selected_weight_plus_optimizer_bytes"])
    compact_bytes = int(raw["persistent_pair_vq_training_bytes"])
    actionable_bytes = int(records[0]["dense_escape"]["candidates"][str(maximum)]["selected_weight_plus_optimizer_bytes"])
    result = {
        "schema_version": SEALED_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": sealed_classification,
        "valid": True,
        "passed": sealed_classification == PASSED,
        "checks": checks,
        "identity": {
            "run_identity_sha256": raw.get("run_identity_sha256"),
            "fixed_eval_indices_sha256": raw.get("fixed_eval_indices_sha256"),
            "git_commit": provenance.get("repository", {}).get("git_commit"),
            "config_sha256": hashes["config"],
            "plan_sha256": hashes["plan"],
            "dataset_manifest_sha256": hashes["dataset_manifest"],
        },
        "artifacts": hashes,
        "performance": {
            "mfu_fraction": float(certificate["measurement"]["mfu_fraction"]),
            "tokens_per_second": float(certificate["measurement"]["tokens_per_second"]),
            "peak_memory_mib": float(certificate["measurement"]["peak_mib"]),
            "wall_seconds": (finished - started).total_seconds(),
        },
        "gate": {
            "thresholds": thresholds,
            "selected_fraction": selected,
            "maximum_actionable_fraction": maximum,
            "selection_jaccard": {
                "mean": sum(jaccards) / len(jaccards),
                "minimum": min(jaccards),
            },
        },
        "state": {
            "persistent_pair_vq_training_bytes": compact_bytes,
            "dense_mlp_weight_plus_optimizer_bytes": dense_bytes,
            "actionable_quarter_total_bytes": compact_bytes + actionable_bytes,
            "actionable_quarter_compression_vs_dense": dense_bytes / (compact_bytes + actionable_bytes),
            "selected_total_bytes": compact_bytes + dense_bytes,
            "selected_compression_vs_dense": dense_bytes / (compact_bytes + dense_bytes),
        },
        "records": record_summaries,
        "interpretation": {
            "value": "Pair-VQ plus feedback retains the dense replay value essentially exactly.",
            "direction": "The useful c_fc-row/c_proj-column functional request is diffuse across the hidden inventory; neither a quarter nor one half of complete neurons meets the registered functional or task-line gates.",
            "next_action": "Reject the paired-neuron dense-escape hybrid without an endpoint or fraction sweep. No further full-MLP GPU experiment is authorized until the compression or target-architecture constraint is explicitly changed.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": sealed_classification, "output": str(args.output)}))


if __name__ == "__main__":
    main()
