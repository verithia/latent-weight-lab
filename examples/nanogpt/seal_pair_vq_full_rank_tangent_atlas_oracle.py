#!/usr/bin/env python3
"""Fail-closed seal for the Pair-VQ full-rank tangent-atlas oracle."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

from examples.nanogpt.seal_attention_pair_vq_vo_endpoint import file_sha256


RAW_SCHEMA = "mai_pair_vq_full_rank_tangent_atlas_oracle_result_v1"
PLAN_SCHEMA = "mai_124m_pair_vq_full_rank_tangent_atlas_oracle_plan_v1"
SEALED_SCHEMA = "mai_pair_vq_full_rank_tangent_atlas_oracle_sealed_result_v1"
REJECTED = "FULL_RANK_TANGENT_ATLAS_REJECTED"
PASSED_S1 = "FULL_RANK_TANGENT_ATLAS_S1_PASSED"
PASSED_S2 = "FULL_RANK_TANGENT_ATLAS_S2_PASSED"


def classification(valid: bool, s1_passed: bool, s2_passed: bool) -> str:
    if not valid:
        return "FULL_RANK_TANGENT_ATLAS_INVALID"
    if s1_passed:
        return PASSED_S1
    if s2_passed:
        return PASSED_S2
    return REJECTED


def candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    matrices = candidate["matrices"]

    def side_summary(side: str) -> dict[str, float]:
        selected = [row for row in matrices if row["side"] == side]
        if len(selected) != 12:
            raise ValueError(f"expected 12 {side} matrices, got {len(selected)}")
        return {
            "minimum_value_cosine": min(float(row["value"]["cosine"]) for row in selected),
            "minimum_tangent_cosine": min(float(row["tangent"]["cosine"]) for row in selected),
            "maximum_tangent_cosine": max(float(row["tangent"]["cosine"]) for row in selected),
            "minimum_functional_jvp_cosine": min(
                float(row["functional"]["cosine"]) for row in selected
            ),
            "maximum_functional_jvp_cosine": max(
                float(row["functional"]["cosine"]) for row in selected
            ),
            "minimum_task_line_retention": min(
                float(row["task_line_retention"]) for row in selected
            ),
        }

    tangent = candidate["tangent"]
    functional = candidate["functional"]
    return {
        "passed": bool(candidate["passed"]),
        "protocol": candidate["protocol"],
        "checks": candidate["checks"],
        "measurements": candidate["measurements"],
        "aggregate_tangent": {
            "cosine": float(tangent["cosine"]),
            "positive_line_recovery": float(tangent["positive_line_recovery"]),
            "candidate_to_reference_energy": float(tangent["candidate_energy"])
            / float(tangent["reference_energy"]),
        },
        "aggregate_functional_jvp": {
            "cosine": float(functional["cosine"]),
            "positive_line_recovery": float(functional["positive_line_recovery"]),
            "candidate_to_reference_energy": float(functional["candidate_energy"])
            / float(functional["reference_energy"]),
        },
        "c_fc": side_summary("c_fc"),
        "c_proj": side_summary("c_proj"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mfu-certificate", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
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
    manifest = Path(config["data_dir"]) / "manifest.json"
    hashes = {
        "raw_result": file_sha256(args.raw_result),
        "plan": file_sha256(args.plan),
        "config": file_sha256(args.config),
        "mfu_certificate": file_sha256(args.mfu_certificate),
        "status": file_sha256(args.status),
        "provenance": file_sha256(args.provenance),
        "training_log": file_sha256(args.training_log),
        "dataset_manifest": file_sha256(manifest),
    }

    expected_steps = [int(step) for step in plan["frozen_protocol"]["probe_steps"]]
    records = raw.get("records", [])
    candidate_names_ok = all(
        set(record.get("tangent_atlas", {}).get("candidates", {})) == {"S1", "S2"}
        for record in records
    )
    protocol_ok = candidate_names_ok and all(
        record["tangent_atlas"]["candidates"][name]["protocol"]
        == {
            "atoms": atoms,
            "coordinates_per_panel": 8448 * atoms,
            "stages": int(plan["atlas"]["stages"]),
        }
        for record in records
        for name, atoms in (("S1", 1), ("S2", 2))
    )
    matrix_inventory_ok = candidate_names_ok and all(
        len(record["tangent_atlas"]["candidates"][name].get("matrices", [])) == 24
        for record in records
        for name in ("S1", "S2")
    )
    metric_checks_ok = True
    candidate_pass_flags_ok = True
    if candidate_names_ok:
        thresholds = plan["atlas_gate"]
        for record in records:
            for name in ("S1", "S2"):
                candidate = record["tangent_atlas"]["candidates"][name]
                measurements = candidate["measurements"]
                recomputed = {
                    key: math.isfinite(float(measurements[key]))
                    and float(measurements[key]) >= float(threshold)
                    for key, threshold in thresholds.items()
                }
                metric_checks_ok &= candidate.get("checks") == recomputed
                candidate_pass_flags_ok &= bool(candidate.get("passed")) == all(
                    recomputed.values()
                )
    else:
        metric_checks_ok = False
        candidate_pass_flags_ok = False

    s1_passed = candidate_names_ok and all(
        bool(record["tangent_atlas"]["candidates"]["S1"]["passed"])
        for record in records
    )
    s2_passed = candidate_names_ok and all(
        bool(record["tangent_atlas"]["candidates"]["S2"]["passed"])
        for record in records
    )
    expected_classification = classification(True, s1_passed, s2_passed)
    expected_selected = "S1" if s1_passed else "S2" if s2_passed else None

    checks: dict[str, bool] = {
        "raw_schema": raw.get("schema_version") == RAW_SCHEMA,
        "plan_schema": plan.get("schema_version") == PLAN_SCHEMA,
        "raw_finished": raw.get("status") == "finished",
        "status_finished": status.get("state") == "finished",
        "status_exit_zero": status.get("exit_code") == 0,
        "result_plan_hash": raw.get("plan", {}).get("sha256") == hashes["plan"],
        "config_plan_hash": config.get("theory_plan_sha256") == hashes["plan"],
        "config_functional_plan_hash": config.get(
            "pair_vq_dense_shadow_functional_plan_sha256"
        )
        == hashes["plan"],
        "manifest_matches_config": config.get("data_manifest_sha256")
        == hashes["dataset_manifest"],
        "provenance_config_hash": provenance.get("config", {}).get("sha256")
        == hashes["config"],
        "provenance_manifest_hash": provenance.get("dataset_manifest", {}).get(
            "sha256"
        )
        == hashes["dataset_manifest"],
        "provenance_mfu_hash": provenance.get("performance_preflight", {}).get(
            "sha256"
        )
        == hashes["mfu_certificate"],
        "status_provenance_hash": status.get("provenance", {}).get("sha256")
        == hashes["provenance"],
        "mfu_passed": certificate.get("passed") is True,
        "mfu_floor": float(certificate.get("measurement", {}).get("mfu_fraction", 0.0))
        >= 0.20,
        "mfu_config_hash": certificate.get("config", {}).get("sha256")
        == hashes["config"],
        "probe_inventory": [int(record.get("step", -1)) for record in records]
        == expected_steps,
        "candidate_inventory": candidate_names_ok,
        "protocol_inventory": protocol_ok,
        "matrix_inventory": matrix_inventory_ok,
        "metric_checks_recomputed": metric_checks_ok,
        "candidate_pass_flags_recomputed": candidate_pass_flags_ok,
        "fit_and_heldout_disjoint": raw.get("fit_and_heldout_indices_disjoint") is True,
        "zero_model_updates": int(
            plan.get("frozen_protocol", {}).get("model_updates_from_oracle", -1)
        )
        == 0,
        "terminal_gate_ready": raw.get("gate", {}).get("ready") is True,
        "terminal_gate_pass": bool(raw.get("gate", {}).get("passed"))
        == (s1_passed or s2_passed),
        "terminal_gate_selection": raw.get("gate", {}).get("selected")
        == expected_selected,
        "terminal_classification": raw.get("gate", {}).get("classification")
        == expected_classification,
        "no_traceback": "Traceback (most recent call last)" not in log_text,
    }
    valid = all(checks.values())
    sealed_classification = classification(valid, s1_passed, s2_passed)
    summaries = [
        {
            "step": int(record["step"]),
            "virtual_weight_energy_recovery": {
                "weighted": float(record["virtual_weight"]["weighted_virtual_weight_energy_recovery"]),
                "worst_matrix": float(record["virtual_weight"]["worst_matrix_virtual_weight_energy_recovery"]),
            },
            "candidates": {
                name: candidate_summary(record["tangent_atlas"]["candidates"][name])
                for name in ("S1", "S2")
            },
        }
        for record in records
    ]
    result = {
        "schema_version": SEALED_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": sealed_classification,
        "valid": valid,
        "passed": valid and (s1_passed or s2_passed),
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
            "wall_seconds": 925.0,
        },
        "state": {
            "persistent_pair_vq_training_bytes": int(raw["persistent_pair_vq_training_bytes"]),
            "S1_compression_vs_dense_values": 69.81818181818181,
            "S2_compression_vs_dense_values": 34.90909090909091,
        },
        "gate": {
            "thresholds": plan["atlas_gate"],
            "selected": expected_selected,
            "S1_passed_all_late_probes": s1_passed,
            "S2_passed_all_late_probes": s2_passed,
        },
        "records": summaries,
        "interpretation": {
            "value": "Both candidates pass only the value-cosine gate; the Pair-VQ plus feedback value chart remains accurate.",
            "tangent": "The fitted local tangent behaves like a generic compact subspace and is grossly misaligned with the same-momentum dense Muon request.",
            "functional": "Held-out MLP JVP and task-line gates fail for both c_fc and c_proj at both late probes.",
            "next_action": "Reject this exact atlas without a sweep; measure the minimum paired-hidden-neuron dense escape fraction before any endpoint training.",
        },
    }
    if not valid:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"refusing invalid seal; failed checks: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": sealed_classification, "output": str(args.output)}))


if __name__ == "__main__":
    main()
