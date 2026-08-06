#!/usr/bin/env python3
"""Decompose the 5TPP c_proj chart residual across phase and matrix axes.

This is a zero-update same-gauge diagnostic.  At each phase boundary it
reconstructs the exact dense Muon update and the frozen production-sized
hidden64+24+output32 Frobenius chart.  Only the chart residual is analyzed.
Future phase chords and later residuals are scoring-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_activation_update_alignment import load_snapshot
from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    file_sha256,
    fit_frobenius_pass,
    git_commit,
    load_probe,
    parameter_name,
    shared_hidden_chart,
)
from examples.nanogpt.analyze_mlp_cproj_diagonal_kfac_selector import (
    acquisition_artifact_hashes,
    require_full_state_snapshot,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv


PLAN_SCHEMA = "mai_124m_mlp_cproj_5tpp_temporal_residual_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_5tpp_temporal_residual_result_v1"
PHASES = ((0, 594), (594, 1188), (1188, 1782), (1782, 2373))
LAYERS = tuple(range(8))


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-30)


def vector_metrics(target: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    target = target.double().reshape(-1)
    candidate = candidate.double().reshape(-1)
    target_energy = target.square().sum().clamp_min(1e-30)
    candidate_energy = candidate.square().sum().clamp_min(1e-30)
    dot = (target * candidate).sum()
    cosine = dot / (target_energy.sqrt() * candidate_energy.sqrt())
    positive_line = dot.clamp_min(0).square() / (
        target_energy * candidate_energy
    )
    fixed = 1.0 - (target - candidate).square().sum() / target_energy
    return {
        "cosine": float(cosine),
        "positive_line_recovery": float(positive_line),
        "fixed_scale_recovery": float(fixed),
        "target_energy": float(target_energy),
        "candidate_energy": float(candidate_energy),
    }


def top_fraction_energy(energy: torch.Tensor, fraction: float) -> float:
    energy = energy.double().flatten()
    count = max(1, math.ceil(fraction * energy.numel()))
    return float(energy.topk(count).values.sum() / energy.sum().clamp_min(1e-30))


def participation_rank(energy: torch.Tensor) -> float:
    energy = energy.double().flatten().clamp_min(0)
    return float(energy.sum().square() / energy.square().sum().clamp_min(1e-30))


def matrix_structure(residual: torch.Tensor) -> dict[str, float]:
    residual = residual.double()
    row_energy = residual.square().sum(dim=1)
    column_energy = residual.square().sum(dim=0)
    gram = residual @ residual.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1e-30)

    def spectral_fraction(rank: int) -> float:
        return float(eigenvalues[: min(rank, eigenvalues.numel())].sum() / total)

    return {
        "singular_top1_energy_fraction": spectral_fraction(1),
        "singular_top4_energy_fraction": spectral_fraction(4),
        "singular_top16_energy_fraction": spectral_fraction(16),
        "singular_top64_energy_fraction": spectral_fraction(64),
        "singular_participation_rank": participation_rank(eigenvalues),
        "stable_rank": float(total / eigenvalues[0].clamp_min(1e-30)),
        "output_top_quarter_energy_fraction": top_fraction_energy(row_energy, 0.25),
        "hidden_top_quarter_energy_fraction": top_fraction_energy(column_energy, 0.25),
        "output_channel_participation_rank": participation_rank(row_energy),
        "hidden_channel_participation_rank": participation_rank(column_energy),
    }


def span_recovery(target: torch.Tensor, bases: list[torch.Tensor]) -> float:
    if not bases:
        return 0.0
    target = target.double().reshape(-1)
    matrix = torch.stack([basis.double().reshape(-1) for basis in bases])
    gram = matrix @ matrix.T
    rhs = matrix @ target
    coefficients = torch.linalg.pinv(gram, rtol=1e-10) @ rhs
    projected_energy = rhs @ coefficients
    return float(
        (projected_energy / target.square().sum().clamp_min(1e-30)).clamp(0, 1)
    )


def temporal_structure(residuals: list[torch.Tensor]) -> dict[str, Any]:
    matrix = torch.stack([value.double().reshape(-1) for value in residuals])
    gram = matrix @ matrix.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1e-30)
    normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-30)
    cosines = normalized @ normalized.T
    previous_line = [
        vector_metrics(residuals[index], residuals[index - 1])[
            "positive_line_recovery"
        ]
        for index in range(1, len(residuals))
    ]
    prior_span = [
        span_recovery(residuals[index], residuals[:index])
        for index in range(1, len(residuals))
    ]
    residual_energy = [float(value.double().square().sum()) for value in residuals]
    return {
        "pc1_energy_fraction": float(eigenvalues[:1].sum() / total),
        "pc2_energy_fraction": float(eigenvalues[:2].sum() / total),
        "pc3_energy_fraction": float(eigenvalues[:3].sum() / total),
        "temporal_participation_rank": participation_rank(eigenvalues),
        "pairwise_cosines": cosines.tolist(),
        "previous_residual_positive_line_recovery": previous_line,
        "prior_residual_span_recovery": prior_span,
        "residual_energy_by_phase": residual_energy,
        "mean_previous_residual_positive_line_recovery": sum(previous_line)
        / len(previous_line),
        "mean_prior_residual_span_recovery": sum(prior_span) / len(prior_span),
        "total_residual_energy": float(total),
    }


def weighted_mean(rows: list[dict[str, Any]], field: str, weight: str) -> float:
    denominator = sum(float(row[weight]) for row in rows)
    return safe_ratio(
        sum(float(row[field]) * float(row[weight]) for row in rows), denominator
    )


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected temporal-residual plan schema")
    analysis = plan.get("analysis", {})
    observed = {
        "parameter_updates": analysis.get("parameter_updates"),
        "layers": analysis.get("layers"),
        "phases": analysis.get("phases"),
        "chart": analysis.get("chart"),
        "thresholds": plan.get("decision_rule", {}).get("thresholds"),
    }
    expected = {
        "parameter_updates": 0,
        "layers": list(LAYERS),
        "phases": [list(pair) for pair in PHASES],
        "chart": {
            "hidden_parent_stages": 64,
            "hidden_residual_stages": 24,
            "output_stages": 32,
            "neighbors": 64,
            "matching_seed": 20260806,
            "weight_decay_application": "identical production ordering",
        },
        "thresholds": {
            "temporal_pc2_energy_fraction_minimum": 0.8,
            "causal_previous_line_recovery_minimum": 0.1,
            "causal_prior_span_recovery_minimum": 0.25,
            "future_chord_residual_line_recovery_minimum": 0.05,
            "singular_top64_energy_fraction_minimum": 0.5,
            "channel_top_quarter_energy_fraction_minimum": 0.5,
        },
    }
    if observed != expected:
        raise ValueError("temporal-residual plan does not match the v1 contract")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_temporal_decomposition") is not True:
        raise ValueError("temporal decomposition is not authorized")
    for key in (
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def classify(aggregate: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    gates = {
        "temporally_low_dimensional": aggregate["temporal_pc2_energy_fraction"]
        >= thresholds["temporal_pc2_energy_fraction_minimum"],
        "previous_direction_predictive": aggregate[
            "causal_previous_line_recovery"
        ]
        >= thresholds["causal_previous_line_recovery_minimum"],
        "prior_span_predictive": aggregate["causal_prior_span_recovery"]
        >= thresholds["causal_prior_span_recovery_minimum"],
        "residual_predicts_future_chord": aggregate[
            "future_chord_residual_line_recovery"
        ]
        >= thresholds["future_chord_residual_line_recovery_minimum"],
        "matrix_low_rank": aggregate["singular_top64_energy_fraction"]
        >= thresholds["singular_top64_energy_fraction_minimum"],
        "output_channel_concentrated": aggregate[
            "output_top_quarter_energy_fraction"
        ]
        >= thresholds["channel_top_quarter_energy_fraction_minimum"],
        "hidden_channel_concentrated": aggregate[
            "hidden_top_quarter_energy_fraction"
        ]
        >= thresholds["channel_top_quarter_energy_fraction_minimum"],
    }
    if gates["temporally_low_dimensional"] and not gates["prior_span_predictive"]:
        conclusion = "LOW_DIMENSIONAL_BUT_PHASE_TRANSPORTED"
    elif gates["prior_span_predictive"]:
        conclusion = "CAUSALLY_COMPRESSIBLE_TEMPORAL_RESIDUAL"
    else:
        conclusion = "TEMPORALLY_DIFFUSE_RESIDUAL"
    return {
        "classification": conclusion,
        "gates": gates,
        "authorization": {
            "phase_local_transport_design": conclusion
            == "LOW_DIMENSIONAL_BUT_PHASE_TRANSPORTED",
            "compact_residual_state_design": conclusion
            == "CAUSALLY_COMPRESSIBLE_TEMPORAL_RESIDUAL",
            "low_rank_design": gates["matrix_low_rank"],
            "output_channel_modulation_design": gates[
                "output_channel_concentrated"
            ],
            "hidden_channel_modulation_design": gates[
                "hidden_channel_concentrated"
            ],
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acquisition-result", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    acquisition = json.loads(args.acquisition_result.read_text())
    if file_sha256(args.acquisition_result) != plan["identity"][
        "acquisition_result_sha256"
    ]:
        raise ValueError("acquisition result SHA-256 mismatch")
    if acquisition.get("classification") != (
        "ACCEPTED_PARENT_EQUIVALENT_EXACT_FUNCTIONAL_REPLAY"
    ):
        raise ValueError("acquisition is not functionally accepted")
    if acquisition["functional_replay"]["result_sha256"] != plan["identity"][
        "functional_replay_result_sha256"
    ]:
        raise ValueError("functional replay SHA-256 mismatch")
    run_identity = plan["identity"]["run_identity_sha256"]
    if acquisition["identity"]["run_identity_sha256"] != run_identity:
        raise ValueError("acquisition run identity mismatch")

    snapshot_hashes = acquisition_artifact_hashes(acquisition, "snapshots")
    probe_hashes = acquisition_artifact_hashes(acquisition, "optimizer_probes")
    weights: dict[int, dict[int, torch.Tensor]] = {}
    for step in sorted({value for phase in PHASES for value in phase}):
        path = args.snapshot_dir / f"step_{step:06d}.pt"
        if file_sha256(path) != snapshot_hashes[str(step)]:
            raise ValueError(f"snapshot SHA-256 mismatch at step {step}")
        snapshot = load_snapshot(path)
        require_full_state_snapshot(snapshot)
        if snapshot["run_identity_sha256"] != run_identity:
            raise ValueError("snapshot run identity mismatch")
        weights[step] = {
            layer: snapshot["parameters"][parameter_name(layer)].float().clone()
            for layer in LAYERS
        }

    residuals: dict[int, list[torch.Tensor]] = {layer: [] for layer in LAYERS}
    rows: list[dict[str, Any]] = []
    chart = plan["analysis"]["chart"]
    started = time.time()
    for phase_index, (start, end) in enumerate(PHASES):
        probe_path = args.probe_dir / f"step_{start:06d}.pt"
        if file_sha256(probe_path) != probe_hashes[str(start)]:
            raise ValueError(f"probe SHA-256 mismatch at step {start}")
        probe = load_probe(probe_path, start, run_identity)
        for layer in LAYERS:
            name = parameter_name(layer)
            state = probe["parameters"][name]
            hyper = probe["hyperparameters"][name]
            weight = weights[start][layer].to(args.device)
            torch.testing.assert_close(
                state["weight_before_step"], weight.cpu(), rtol=0.0, atol=0.0
            )
            learning_rate = float(hyper["lr"])
            weight_decay = float(hyper["weight_decay"])
            applied_per_lr = state["applied_direction_per_lr"].to(args.device)
            exact_update = learning_rate * applied_per_lr
            matching_direction = applied_per_lr + weight_decay * weight
            seed = int(chart["matching_seed"]) + layer * 100000 + phase_index * 10
            hidden_weight, output_residual, hidden_diagnostics = shared_hidden_chart(
                weight,
                exact_update,
                matching_direction,
                parent_stages=int(chart["hidden_parent_stages"]),
                residual_stages=int(chart["hidden_residual_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed,
            )
            fitted, output_diagnostics = fit_frobenius_pass(
                hidden_weight.T.contiguous(),
                output_residual.T.contiguous(),
                stages=int(chart["output_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed + 2,
            )
            final_weight = fitted.T.contiguous() * (
                1.0 - learning_rate * weight_decay
            )
            chart_update = final_weight - weight
            residual = exact_update - chart_update
            future_chord = weights[end][layer].to(args.device) - weight
            residuals[layer].append(residual.detach().cpu())
            exact_metrics = vector_metrics(future_chord, exact_update)
            chart_metrics = vector_metrics(future_chord, chart_update)
            residual_metrics = vector_metrics(future_chord, residual)
            update_metrics = vector_metrics(exact_update, chart_update)
            structure = matrix_structure(residual)
            coordinates = sum(
                int(value["coordinates"]) for value in hidden_diagnostics
            ) + int(output_diagnostics["coordinates"])
            if coordinates != 147456:
                raise ValueError("chart coordinate budget mismatch")
            row = {
                "phase_start": start,
                "phase_end": end,
                "layer": layer,
                "coordinates_per_layer": coordinates,
                "exact_update_energy": update_metrics["target_energy"],
                "chart_update_energy": update_metrics["candidate_energy"],
                "chart_fixed_scale_recovery": update_metrics[
                    "fixed_scale_recovery"
                ],
                "residual_energy": float(residual.double().square().sum()),
                "residual_energy_fraction_of_exact_update": safe_ratio(
                    float(residual.double().square().sum()),
                    update_metrics["target_energy"],
                ),
                "future_chord_energy": exact_metrics["target_energy"],
                "future_chord_exact_update_line_recovery": exact_metrics[
                    "positive_line_recovery"
                ],
                "future_chord_chart_line_recovery": chart_metrics[
                    "positive_line_recovery"
                ],
                "future_chord_residual_line_recovery": residual_metrics[
                    "positive_line_recovery"
                ],
                "future_chord_residual_cosine": residual_metrics["cosine"],
                **structure,
            }
            rows.append(row)
        del probe

    temporal_rows = []
    for layer in LAYERS:
        temporal_rows.append({"layer": layer, **temporal_structure(residuals[layer])})

    aggregate = {
        "chart_fixed_scale_recovery": weighted_mean(
            rows, "chart_fixed_scale_recovery", "exact_update_energy"
        ),
        "residual_energy_fraction_of_exact_update": safe_ratio(
            sum(float(row["residual_energy"]) for row in rows),
            sum(float(row["exact_update_energy"]) for row in rows),
        ),
        "future_chord_exact_update_line_recovery": weighted_mean(
            rows,
            "future_chord_exact_update_line_recovery",
            "future_chord_energy",
        ),
        "future_chord_chart_line_recovery": weighted_mean(
            rows,
            "future_chord_chart_line_recovery",
            "future_chord_energy",
        ),
        "future_chord_residual_line_recovery": weighted_mean(
            rows,
            "future_chord_residual_line_recovery",
            "future_chord_energy",
        ),
    }
    for field in (
        "singular_top1_energy_fraction",
        "singular_top4_energy_fraction",
        "singular_top16_energy_fraction",
        "singular_top64_energy_fraction",
        "singular_participation_rank",
        "stable_rank",
        "output_top_quarter_energy_fraction",
        "hidden_top_quarter_energy_fraction",
        "output_channel_participation_rank",
        "hidden_channel_participation_rank",
    ):
        aggregate[field] = weighted_mean(rows, field, "residual_energy")
    total_temporal_energy = sum(
        float(row["total_residual_energy"]) for row in temporal_rows
    )
    aggregate["temporal_pc1_energy_fraction"] = safe_ratio(
        sum(
            float(row["pc1_energy_fraction"])
            * float(row["total_residual_energy"])
            for row in temporal_rows
        ),
        total_temporal_energy,
    )
    aggregate["temporal_pc2_energy_fraction"] = safe_ratio(
        sum(
            float(row["pc2_energy_fraction"])
            * float(row["total_residual_energy"])
            for row in temporal_rows
        ),
        total_temporal_energy,
    )
    aggregate["causal_previous_line_recovery"] = safe_ratio(
        sum(
            sum(
                float(value) * float(row["residual_energy_by_phase"][index + 1])
                for index, value in enumerate(
                    row["previous_residual_positive_line_recovery"]
                )
            )
            for row in temporal_rows
        ),
        sum(
            sum(float(value) for value in row["residual_energy_by_phase"][1:])
            for row in temporal_rows
        ),
    )
    aggregate["causal_prior_span_recovery"] = safe_ratio(
        sum(
            sum(
                float(value) * float(row["residual_energy_by_phase"][index + 1])
                for index, value in enumerate(row["prior_residual_span_recovery"])
            )
            for row in temporal_rows
        ),
        sum(
            sum(float(value) for value in row["residual_energy_by_phase"][1:])
            for row in temporal_rows
        ),
    )
    decision = classify(aggregate, plan["decision_rule"]["thresholds"])

    args.output.mkdir(parents=True)
    detail_path = args.output / "temporal_residual_cells.csv"
    temporal_path = args.output / "temporal_residual_by_layer.json"
    result_path = args.output / "temporal_residual_result.json"
    write_csv(detail_path, rows)
    temporal_path.write_text(
        json.dumps(temporal_rows, indent=2, sort_keys=True) + "\n"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": decision["classification"],
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_mlp_cproj_temporal_residual",
            "parameter_updates": 0,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_path": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "acquisition_result_path": str(args.acquisition_result),
            "acquisition_result_sha256": file_sha256(args.acquisition_result),
            "run_identity_sha256": run_identity,
        },
        "aggregate": aggregate,
        "decision": decision,
        "by_layer": temporal_rows,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
