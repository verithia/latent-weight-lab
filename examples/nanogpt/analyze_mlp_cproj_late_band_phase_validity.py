#!/usr/bin/env python3
"""Factor late-band and middle-band c_proj phase validity in one gauge."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    git_commit,
    parameter_name,
)
from examples.nanogpt.analyze_mlp_cproj_bounded_integrated_trajectory import (
    PHASES,
    fit_straight_chord,
)
from examples.nanogpt.analyze_mlp_cproj_diagonal_kfac_selector import (
    acquisition_artifact_hashes,
    require_full_state_snapshot,
)
from examples.nanogpt.analyze_mlp_cproj_same_gauge_lwt_allocation import (
    ALL_LAYERS,
    ALWAYS_PROCEDURAL_LAYERS,
    DIFFICULT_LAYERS,
    evaluate_variant,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.train import (
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)


PLAN_SCHEMA = "mai_124m_mlp_cproj_late_band_phase_validity_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_late_band_phase_validity_result_v1"
MIDDLE_LAYERS = tuple(range(4, 8))
EARLY_LAYERS = tuple(range(4))


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    observed = {
        "schema_version": plan.get("schema_version"),
        "parameter_updates": analysis.get("parameter_updates"),
        "phases": analysis.get("phases"),
        "early_layers": analysis.get("early_dense_layers"),
        "middle_layers": analysis.get("middle_test_layers"),
        "late_layers": analysis.get("late_test_layers"),
        "substeps": analysis.get("straight_chord_substeps"),
        "feedback_decay": analysis.get("feedback_decay"),
        "chart": analysis.get("chart"),
        "eval": analysis.get("fixed_validation"),
        "thresholds": plan.get("decision_rule", {}).get("thresholds"),
    }
    expected = {
        "schema_version": PLAN_SCHEMA,
        "parameter_updates": 0,
        "phases": [list(value) for value in PHASES],
        "early_layers": list(EARLY_LAYERS),
        "middle_layers": list(MIDDLE_LAYERS),
        "late_layers": list(ALWAYS_PROCEDURAL_LAYERS),
        "substeps": 8,
        "feedback_decay": 0.5,
        "chart": {
            "hidden_parent_stages": 64,
            "hidden_residual_stages": 24,
            "output_stages": 32,
            "neighbors": 64,
            "matching_seed": 20260807,
        },
        "eval": {
            "split": "validation",
            "eval_iters": 400,
            "eval_batch_size": 16,
            "block_size": 1024,
            "eval_seed": 20260715,
            "fixed_eval_indices_sha256": (
                "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
            ),
        },
        "thresholds": {
            "maximum_phase_validation_ce_gap": 0.01,
            "terminal_validation_ce_gap": 0.005,
        },
    }
    if observed != expected:
        raise ValueError("late-band phase-validity plan does not match v1 contract")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_band_factorization") is not True:
        raise ValueError("band factorization is not authorized")
    for key in (
        "run_coadapted_trajectory_acquisition",
        "implement_candidate_mask",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def band_passes(
    gaps: dict[int, float],
    *,
    maximum_phase_gap: float,
    terminal_gap: float,
) -> bool:
    terminal = PHASES[-1][1]
    return (
        max(gaps.values()) <= maximum_phase_gap
        and gaps[terminal] <= terminal_gap
    )


def classify(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    late_pass = band_passes(
        metrics["late_only_gap_by_end_step"],
        maximum_phase_gap=thresholds["maximum_phase_validation_ce_gap"],
        terminal_gap=thresholds["terminal_validation_ce_gap"],
    )
    middle_pass = band_passes(
        metrics["middle_only_gap_by_end_step"],
        maximum_phase_gap=thresholds["maximum_phase_validation_ce_gap"],
        terminal_gap=thresholds["terminal_validation_ce_gap"],
    )
    combined_pass = band_passes(
        metrics["combined_gap_by_end_step"],
        maximum_phase_gap=thresholds["maximum_phase_validation_ce_gap"],
        terminal_gap=thresholds["terminal_validation_ce_gap"],
    )
    if not late_pass:
        classification = "REQUIRE_COADAPTED_LATE_BAND_TRAJECTORY"
    elif not middle_pass:
        classification = "LOCALIZE_DENSE_PATH_FAILURE_TO_MIDDLE_4_7"
    elif not combined_pass:
        classification = "LOCALIZE_DENSE_PATH_FAILURE_TO_CROSS_BAND_INTERACTION"
    else:
        classification = "DENSE_PARENT_BAND_PATH_CAPACITY_PASS"
    return {
        "classification": classification,
        "gates": {
            "late_band_phase_valid": late_pass,
            "middle_band_phase_valid": middle_pass,
            "combined_band_phase_valid": combined_pass,
        },
        "authorization": {
            "run_coadapted_trajectory_acquisition": not late_pass,
            "implement_candidate_mask": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def verify_identity_inputs(
    plan: dict[str, Any],
    acquisition: dict[str, Any],
    *,
    acquisition_result: Path,
    config_path: Path,
) -> dict[str, Any]:
    identity = plan["identity"]
    for path_key in (
        "allocation_result",
        "bounded_result",
        "late_band_lwt_result",
    ):
        path = REPO_ROOT / identity[path_key]
        if file_sha256(path) != identity[f"{path_key}_sha256"]:
            raise ValueError(f"{path_key} SHA-256 mismatch")
    if file_sha256(acquisition_result) != identity["acquisition_result_sha256"]:
        raise ValueError("acquisition result SHA-256 mismatch")
    if acquisition.get("classification") != (
        "ACCEPTED_PARENT_EQUIVALENT_EXACT_FUNCTIONAL_REPLAY"
    ):
        raise ValueError("acquisition is not functionally accepted")
    replay = acquisition.get("functional_replay", {})
    if (
        replay.get("passed") is not True
        or replay.get("result_sha256")
        != identity["functional_replay_result_sha256"]
    ):
        raise ValueError("functional replay SHA-256 mismatch")
    if acquisition["identity"]["run_identity_sha256"] != identity[
        "run_identity_sha256"
    ]:
        raise ValueError("acquisition run identity mismatch")
    if file_sha256(config_path) != identity["config_sha256"]:
        raise ValueError("config SHA-256 mismatch")
    config = json.loads(config_path.read_text())
    if file_sha256(Path(config["data_dir"]) / "manifest.json") != identity[
        "dataset_manifest_sha256"
    ]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acquisition-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    acquisition = json.loads(args.acquisition_result.read_text())
    config = verify_identity_inputs(
        plan,
        acquisition,
        acquisition_result=args.acquisition_result,
        config_path=args.config,
    )
    require_block_fht_native_extension(
        bool(config["block_fht_native_extension_required"])
    )

    snapshot_hashes = acquisition_artifact_hashes(acquisition, "snapshots")
    snapshots: dict[int, dict[str, Any]] = {}
    dense_weights: dict[int, dict[int, torch.Tensor]] = {}
    approximate_weights: dict[int, dict[int, torch.Tensor]] = {}
    run_identity = plan["identity"]["run_identity_sha256"]
    chart = plan["analysis"]["chart"]
    for phase_index, (start_step, end_step) in enumerate(PHASES):
        for step in (start_step, end_step):
            if step in snapshots:
                continue
            path = args.snapshot_dir / f"step_{step:06d}.pt"
            if file_sha256(path) != snapshot_hashes[str(step)]:
                raise ValueError(f"snapshot SHA-256 mismatch at step {step}")
            snapshot = load_snapshot(path)
            require_full_state_snapshot(snapshot)
            if snapshot["run_identity_sha256"] != run_identity:
                raise ValueError("snapshot run identity mismatch")
            snapshots[step] = snapshot
            dense_weights[step] = {
                layer: snapshot["parameters"][parameter_name(layer)].float().clone()
                for layer in ALL_LAYERS
            }
        approximate_weights[end_step] = {}
        for layer in ALL_LAYERS:
            start = dense_weights[start_step][layer].to(args.device)
            end = dense_weights[end_step][layer].to(args.device)
            candidate, _feedback, _recoveries = fit_straight_chord(
                start,
                end - start,
                feedback_decay=0.5,
                layer=layer,
                phase_index=phase_index,
                neighbors=int(chart["neighbors"]),
                seed=int(chart["matching_seed"]),
            )
            approximate_weights[end_step][layer] = candidate.cpu()

    fixed_spec = plan["analysis"]["fixed_validation"]
    fixed = make_fixed_eval_indices(
        Path(config["data_dir"]),
        int(fixed_spec["eval_batch_size"]),
        int(fixed_spec["block_size"]),
        int(fixed_spec["eval_iters"]),
        int(fixed_spec["eval_seed"]),
    )
    if fixed_eval_indices_digest(fixed) != fixed_spec[
        "fixed_eval_indices_sha256"
    ]:
        raise ValueError("fixed validation indices SHA-256 mismatch")
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]
    eval_args = SimpleNamespace(**config)
    eval_args._ptdtype = dtype
    ctx = (
        contextlib.nullcontext()
        if "cuda" not in args.device
        else torch.amp.autocast(device_type="cuda", dtype=dtype)
    )
    parent_ce = {
        int(row["step"]): float(row["replayed_validation_ce"])
        for row in acquisition["functional_replay"]["rows"]
    }
    variants = {
        "late_only": set(range(8)),
        "middle_only": set(EARLY_LAYERS + ALWAYS_PROCEDURAL_LAYERS),
    }
    rows: list[dict[str, Any]] = []
    started = time.time()
    for _start_step, end_step in PHASES:
        model = model_from_snapshot(snapshots[end_step], args.device)
        for variant, dense_layers in variants.items():
            validation_ce = evaluate_variant(
                model,
                dense=dense_weights[end_step],
                approximate=approximate_weights[end_step],
                dense_layers=dense_layers,
                data_dir=Path(config["data_dir"]),
                eval_args=eval_args,
                indices=fixed["val"],
                ctx=ctx,
            )
            rows.append(
                {
                    "step": end_step,
                    "variant": variant,
                    "dense_layers": ",".join(
                        str(value) for value in sorted(dense_layers)
                    ),
                    "validation_ce": validation_ce,
                    "parent_validation_ce": parent_ce[end_step],
                    "validation_ce_gap": validation_ce - parent_ce[end_step],
                }
            )
        del model
        torch.cuda.empty_cache()

    gaps = {
        variant: {
            int(row["step"]): float(row["validation_ce_gap"])
            for row in rows
            if row["variant"] == variant
        }
        for variant in variants
    }
    allocation_result = json.loads(
        (REPO_ROOT / plan["identity"]["allocation_result"]).read_text()
    )
    combined_gaps = {
        int(step): float(value)
        for step, value in allocation_result[
            "selected_validation_ce_gap_by_end_step"
        ].items()
    }
    metrics: dict[str, Any] = {
        "late_only_gap_by_end_step": gaps["late_only"],
        "middle_only_gap_by_end_step": gaps["middle_only"],
        "combined_gap_by_end_step": combined_gaps,
        "factorial_interaction_gap_by_end_step": {
            step: combined_gaps[step]
            - gaps["late_only"][step]
            - gaps["middle_only"][step]
            for _start, step in PHASES
        },
    }
    thresholds = plan["decision_rule"]["thresholds"]
    decision = classify(metrics, thresholds)
    args.output.mkdir(parents=True)
    rows_path = args.output / "late_band_phase_validity_rows.csv"
    result_path = args.output / "late_band_phase_validity_result.json"
    write_csv(rows_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": decision["classification"],
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_mlp_cproj_late_band_phase_validity",
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "parameter_updates": 0,
            "elapsed_seconds": time.time() - started,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "identity": {
            "plan_sha256": file_sha256(args.plan),
            "allocation_result_sha256": file_sha256(
                REPO_ROOT / plan["identity"]["allocation_result"]
            ),
            "acquisition_result_sha256": file_sha256(args.acquisition_result),
            "functional_replay_result_sha256": acquisition[
                "functional_replay"
            ]["result_sha256"],
            "run_identity_sha256": run_identity,
            "dataset_manifest_sha256": plan["identity"][
                "dataset_manifest_sha256"
            ],
        },
        "metrics": metrics,
        "rows": rows,
        "decision": decision,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
