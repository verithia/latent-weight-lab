#!/usr/bin/env python3
"""Attribute same-gauge dense c_proj exceptions inside the difficult band."""

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
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.train import (
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
    require_block_fht_native_extension,
)
from examples.nanogpt.verify_full_state_functional_replay import (
    evaluate_validation_ce,
)


PLAN_SCHEMA = "mai_124m_mlp_cproj_same_gauge_lwt_allocation_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_same_gauge_lwt_allocation_result_v1"
DIFFICULT_LAYERS = tuple(range(8))
ALWAYS_PROCEDURAL_LAYERS = tuple(range(8, 12))
ALL_LAYERS = DIFFICULT_LAYERS + ALWAYS_PROCEDURAL_LAYERS


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    observed = {
        "schema_version": plan.get("schema_version"),
        "parameter_updates": analysis.get("parameter_updates"),
        "difficult_layers": analysis.get("difficult_layers"),
        "always_procedural_layers": analysis.get("always_procedural_layers"),
        "phases": analysis.get("phases"),
        "substeps": analysis.get("straight_chord_substeps"),
        "feedback_decay": analysis.get("feedback_decay"),
        "chart": analysis.get("chart"),
        "eval": analysis.get("fixed_validation"),
        "selection": analysis.get("selection"),
        "thresholds": plan.get("decision_rule", {}).get("thresholds"),
    }
    expected = {
        "schema_version": PLAN_SCHEMA,
        "parameter_updates": 0,
        "difficult_layers": list(DIFFICULT_LAYERS),
        "always_procedural_layers": list(ALWAYS_PROCEDURAL_LAYERS),
        "phases": [list(value) for value in PHASES],
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
            "discovery_slice": [0, 64],
            "holdout_slice": [64, 192],
            "confirmation_slice": [0, 400],
        },
        "selection": {
            "ranking": "terminal discovery single-dense repair descending",
            "prefix_sizes": [1, 2, 3, 4],
            "selected_prefix": "smallest discovery prefix within 0.005 CE of dense",
            "failure_fallback": "top-four prefix for diagnostic confirmation only",
        },
        "thresholds": {
            "maximum_dense_exceptions": 4,
            "discovery_terminal_validation_ce_gap": 0.005,
            "holdout_terminal_validation_ce_gap": 0.005,
            "confirmation_terminal_validation_ce_gap": 0.005,
            "confirmation_maximum_phase_validation_ce_gap": 0.01,
            "minimum_terminal_repair_over_all_approx": 0.002,
            "predecessor_must_fail_terminal_gap": 0.005,
        },
    }
    if observed != expected:
        raise ValueError("same-gauge LWT allocation plan does not match v1 contract")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_same_gauge_lwt_attribution") is not True:
        raise ValueError("same-gauge LWT attribution is not authorized")
    for key in (
        "implement_candidate_mask",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def choose_prefix(
    ranking: list[int],
    prefix_ce: dict[int, float],
    dense_ce: float,
    *,
    maximum_k: int,
    maximum_gap: float,
) -> tuple[list[int], int | None]:
    for k in range(1, maximum_k + 1):
        if prefix_ce[k] - dense_ce <= maximum_gap:
            return ranking[:k], k
    return ranking[:maximum_k], None


def classify(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    selected_k = metrics["selected_k"]
    predecessor_gap = metrics["confirmation_predecessor_terminal_gap"]
    gates = {
        "selection_found": selected_k is not None,
        "dense_budget": selected_k is not None
        and selected_k <= int(thresholds["maximum_dense_exceptions"]),
        "discovery": metrics["discovery_selected_terminal_gap"]
        <= thresholds["discovery_terminal_validation_ce_gap"],
        "holdout": metrics["holdout_selected_terminal_gap"]
        <= thresholds["holdout_terminal_validation_ce_gap"],
        "confirmation_terminal": metrics["confirmation_selected_terminal_gap"]
        <= thresholds["confirmation_terminal_validation_ce_gap"],
        "confirmation_phases": metrics["confirmation_maximum_phase_gap"]
        <= thresholds["confirmation_maximum_phase_validation_ce_gap"],
        "material_repair": metrics["confirmation_terminal_repair_over_all_approx"]
        >= thresholds["minimum_terminal_repair_over_all_approx"],
        "minimality": predecessor_gap
        > thresholds["predecessor_must_fail_terminal_gap"],
    }
    passed = all(gates.values())
    if passed:
        classification = "SAME_GAUGE_LWT_MASK_CAPACITY_PASS"
    elif all(value for key, value in gates.items() if key != "minimality"):
        classification = "MASK_OVERSELECTED_REQUIRES_FRESH_PREREGISTRATION"
    else:
        classification = "REJECT_EARLY_MIDDLE_LWT_EXPANSION_AT_DENSE_BUDGET"
    return {
        "classification": classification,
        "passed": passed,
        "gates": gates,
        "authorization": {
            "candidate_mask_theory": passed,
            "implement_candidate_mask": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def install_variant(
    model: torch.nn.Module,
    *,
    dense: dict[int, torch.Tensor],
    approximate: dict[int, torch.Tensor],
    dense_layers: set[int],
) -> None:
    with torch.no_grad():
        for layer in ALL_LAYERS:
            target = model.transformer.h[layer].mlp.c_proj.weight
            source = dense[layer] if layer in dense_layers else approximate[layer]
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def evaluate_variant(
    model: torch.nn.Module,
    *,
    dense: dict[int, torch.Tensor],
    approximate: dict[int, torch.Tensor],
    dense_layers: set[int],
    data_dir: Path,
    eval_args: SimpleNamespace,
    indices: torch.Tensor,
    ctx: Any,
) -> float:
    install_variant(
        model,
        dense=dense,
        approximate=approximate,
        dense_layers=dense_layers,
    )
    current_args = SimpleNamespace(**vars(eval_args))
    current_args.eval_iters = int(indices.shape[0])
    return evaluate_validation_ce(
        model,
        data_dir=data_dir,
        args=current_args,
        indices=indices,
        ctx=ctx,
    )


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
    identity_spec = plan["identity"]
    for path_key in ("bounded_result", "late_band_lwt_result"):
        upstream_path = REPO_ROOT / identity_spec[path_key]
        if file_sha256(upstream_path) != identity_spec[f"{path_key}_sha256"]:
            raise ValueError(f"{path_key} SHA-256 mismatch")
    if file_sha256(args.acquisition_result) != plan["identity"][
        "acquisition_result_sha256"
    ]:
        raise ValueError("acquisition result SHA-256 mismatch")
    if acquisition.get("classification") != (
        "ACCEPTED_PARENT_EQUIVALENT_EXACT_FUNCTIONAL_REPLAY"
    ):
        raise ValueError("acquisition is not functionally accepted")
    functional_replay = acquisition["functional_replay"]
    if (
        functional_replay.get("passed") is not True
        or functional_replay["result_sha256"]
        != identity_spec["functional_replay_result_sha256"]
    ):
        raise ValueError("functional replay SHA-256 mismatch")
    if acquisition["identity"]["run_identity_sha256"] != identity_spec[
        "run_identity_sha256"
    ]:
        raise ValueError("acquisition run identity mismatch")
    if file_sha256(args.config) != plan["identity"]["config_sha256"]:
        raise ValueError("config SHA-256 mismatch")
    config = json.loads(args.config.read_text())
    manifest = Path(config["data_dir"]) / "manifest.json"
    if file_sha256(manifest) != identity_spec["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    require_block_fht_native_extension(
        bool(config["block_fht_native_extension_required"])
    )

    snapshot_hashes = acquisition_artifact_hashes(acquisition, "snapshots")
    snapshots: dict[int, dict[str, Any]] = {}
    dense_weights: dict[int, dict[int, torch.Tensor]] = {}
    approximate_weights: dict[int, dict[int, torch.Tensor]] = {}
    identity = plan["identity"]["run_identity_sha256"]
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
            if snapshot["run_identity_sha256"] != identity:
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
    if fixed_eval_indices_digest(fixed) != fixed_spec["fixed_eval_indices_sha256"]:
        raise ValueError("fixed validation indices SHA-256 mismatch")
    val_indices = fixed["val"]
    discovery = val_indices[slice(*fixed_spec["discovery_slice"])]
    holdout = val_indices[slice(*fixed_spec["holdout_slice"])]
    confirmation = val_indices[slice(*fixed_spec["confirmation_slice"])]
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
    data_dir = Path(config["data_dir"])
    started = time.time()
    rows: list[dict[str, Any]] = []
    terminal = PHASES[-1][1]
    model = model_from_snapshot(snapshots[terminal], args.device)
    dense = dense_weights[terminal]
    approximate = approximate_weights[terminal]

    def score_terminal(
        name: str,
        layers: set[int],
        indices: torch.Tensor,
        split: str,
    ) -> float:
        value = evaluate_variant(
            model,
            dense=dense,
            approximate=approximate,
            dense_layers=layers,
            data_dir=data_dir,
            eval_args=eval_args,
            indices=indices,
            ctx=ctx,
        )
        rows.append(
            {
                "step": terminal,
                "split": split,
                "variant": name,
                "dense_layers": ",".join(str(value) for value in sorted(layers)),
                "dense_layer_count": len(layers),
                "validation_ce": value,
            }
        )
        return value

    all_dense = set(ALL_LAYERS)
    dense_discovery_ce = score_terminal(
        "all_dense", all_dense, discovery, "discovery"
    )
    all_approx_discovery_ce = score_terminal(
        "all_approx", set(), discovery, "discovery"
    )
    single_ce: dict[int, float] = {}
    for layer in DIFFICULT_LAYERS:
        single_ce[layer] = score_terminal(
            f"single_dense_{layer}", {layer}, discovery, "discovery"
        )
    ranking = sorted(
        DIFFICULT_LAYERS,
        key=lambda layer: (single_ce[layer], layer),
    )
    prefix_ce: dict[int, float] = {1: single_ce[ranking[0]]}
    for k in range(2, 5):
        prefix_ce[k] = score_terminal(
            f"dense_prefix_{k}", set(ranking[:k]), discovery, "discovery"
        )
    thresholds = plan["decision_rule"]["thresholds"]
    selected_layers, selected_k = choose_prefix(
        ranking,
        prefix_ce,
        dense_discovery_ce,
        maximum_k=int(thresholds["maximum_dense_exceptions"]),
        maximum_gap=float(thresholds["discovery_terminal_validation_ce_gap"]),
    )
    diagnostic_k = len(selected_layers)
    predecessor_layers = set(ranking[: max(diagnostic_k - 1, 0)])
    selected_set = set(selected_layers)
    selected_discovery_ce = prefix_ce[diagnostic_k]
    dense_holdout_ce = score_terminal("all_dense", all_dense, holdout, "holdout")
    selected_holdout_ce = score_terminal(
        "selected", selected_set, holdout, "holdout"
    )
    predecessor_holdout_ce = score_terminal(
        "predecessor", predecessor_layers, holdout, "holdout"
    )
    del model
    torch.cuda.empty_cache()

    parent_ce = {
        int(row["step"]): float(row["replayed_validation_ce"])
        for row in acquisition["functional_replay"]["rows"]
    }
    confirmation_rows: list[dict[str, Any]] = []
    for _start_step, end_step in PHASES:
        phase_model = model_from_snapshot(snapshots[end_step], args.device)
        selected_ce = evaluate_variant(
            phase_model,
            dense=dense_weights[end_step],
            approximate=approximate_weights[end_step],
            dense_layers=selected_set,
            data_dir=data_dir,
            eval_args=eval_args,
            indices=confirmation,
            ctx=ctx,
        )
        confirmation_rows.append(
            {
                "step": end_step,
                "split": "confirmation",
                "variant": "selected",
                "dense_layers": ",".join(str(value) for value in selected_layers),
                "dense_layer_count": diagnostic_k,
                "validation_ce": selected_ce,
                "parent_validation_ce": parent_ce[end_step],
                "validation_ce_gap": selected_ce - parent_ce[end_step],
            }
        )
        if end_step == terminal:
            all_approx_ce = evaluate_variant(
                phase_model,
                dense=dense_weights[end_step],
                approximate=approximate_weights[end_step],
                dense_layers=set(),
                data_dir=data_dir,
                eval_args=eval_args,
                indices=confirmation,
                ctx=ctx,
            )
            confirmation_rows.append(
                {
                    "step": end_step,
                    "split": "confirmation",
                    "variant": "all_approx",
                    "dense_layers": "",
                    "dense_layer_count": 0,
                    "validation_ce": all_approx_ce,
                    "parent_validation_ce": parent_ce[end_step],
                    "validation_ce_gap": all_approx_ce - parent_ce[end_step],
                }
            )
            predecessor_ce = evaluate_variant(
                phase_model,
                dense=dense_weights[end_step],
                approximate=approximate_weights[end_step],
                dense_layers=predecessor_layers,
                data_dir=data_dir,
                eval_args=eval_args,
                indices=confirmation,
                ctx=ctx,
            )
            confirmation_rows.append(
                {
                    "step": end_step,
                    "split": "confirmation",
                    "variant": "predecessor",
                    "dense_layers": ",".join(
                        str(value) for value in sorted(predecessor_layers)
                    ),
                    "dense_layer_count": len(predecessor_layers),
                    "validation_ce": predecessor_ce,
                    "parent_validation_ce": parent_ce[end_step],
                    "validation_ce_gap": predecessor_ce - parent_ce[end_step],
                }
            )
        del phase_model
        torch.cuda.empty_cache()

    rows.extend(confirmation_rows)
    confirmation_index = {
        (int(row["step"]), str(row["variant"])): row
        for row in confirmation_rows
    }
    terminal_selected = confirmation_index[(terminal, "selected")]
    terminal_predecessor = confirmation_index[(terminal, "predecessor")]
    terminal_all_approx = confirmation_index[(terminal, "all_approx")]
    metrics: dict[str, Any] = {
        "ranking": list(ranking),
        "selected_layers": selected_layers,
        "selected_k": selected_k,
        "diagnostic_k": diagnostic_k,
        "discovery_all_approx_terminal_gap": all_approx_discovery_ce
        - dense_discovery_ce,
        "discovery_selected_terminal_gap": selected_discovery_ce
        - dense_discovery_ce,
        "holdout_selected_terminal_gap": selected_holdout_ce - dense_holdout_ce,
        "holdout_predecessor_terminal_gap": predecessor_holdout_ce
        - dense_holdout_ce,
        "confirmation_selected_terminal_gap": float(
            terminal_selected["validation_ce_gap"]
        ),
        "confirmation_predecessor_terminal_gap": float(
            terminal_predecessor["validation_ce_gap"]
        ),
        "confirmation_maximum_phase_gap": max(
            float(row["validation_ce_gap"])
            for row in confirmation_rows
            if row["variant"] == "selected"
        ),
        "confirmation_all_approx_terminal_gap": float(
            terminal_all_approx["validation_ce_gap"]
        ),
        "confirmation_terminal_repair_over_all_approx": float(
            terminal_all_approx["validation_ce_gap"]
        )
        - float(terminal_selected["validation_ce_gap"]),
    }
    decision = classify(metrics, thresholds)
    args.output.mkdir(parents=True)
    rows_path = args.output / "same_gauge_lwt_allocation_rows.csv"
    result_path = args.output / "same_gauge_lwt_allocation_result.json"
    write_csv(rows_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": decision["classification"],
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_mlp_cproj_same_gauge_lwt_allocation",
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "parameter_updates": 0,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_path": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "acquisition_result_sha256": file_sha256(args.acquisition_result),
            "functional_replay_result_sha256": acquisition[
                "functional_replay"
            ]["result_sha256"],
            "bounded_result_sha256": file_sha256(
                REPO_ROOT / identity_spec["bounded_result"]
            ),
            "late_band_lwt_result_sha256": file_sha256(
                REPO_ROOT / identity_spec["late_band_lwt_result"]
            ),
            "run_identity_sha256": identity,
            "dataset_manifest_sha256": plan["identity"][
                "dataset_manifest_sha256"
            ],
        },
        "metrics": metrics,
        "allocation_contract": {
            "difficult_layers": list(DIFFICULT_LAYERS),
            "always_procedural_layers": list(ALWAYS_PROCEDURAL_LAYERS),
            "dense_exception_budget": int(
                thresholds["maximum_dense_exceptions"]
            ),
        },
        "single_dense_discovery_ce": {
            str(layer): single_ce[layer] for layer in DIFFICULT_LAYERS
        },
        "prefix_discovery_ce": {str(k): prefix_ce[k] for k in sorted(prefix_ce)},
        "confirmation": confirmation_rows,
        "decision": decision,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
