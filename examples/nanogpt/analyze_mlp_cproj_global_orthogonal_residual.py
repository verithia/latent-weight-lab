#!/usr/bin/env python3
"""Test whether the c_proj chart residual is sparse in a global FHT basis."""

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
from latent_weight_lab.block_fht import normalized_fht_last_dim


PLAN_SCHEMA = "mai_124m_mlp_cproj_5tpp_global_orthogonal_residual_plan_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_5tpp_global_orthogonal_residual_result_v1"
PHASES = ((0, 594), (594, 1188), (1188, 1782), (1782, 2373))
LAYERS = tuple(range(8))
BASES = ("identity", "local_block_fht256", "global_tensor_fht")


def deterministic_signs(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    bits = torch.randint(
        0,
        2,
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=torch.int8,
    )
    return bits.to(reference.dtype).mul_(2).sub_(1)


def deterministic_orthogonal3(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    diagonal_sign = torch.where(torch.diag(r) >= 0, 1.0, -1.0)
    q = q * diagonal_sign.reshape(1, -1)
    return q.to(device=device, dtype=torch.float32)


def local_block_fht256(values: torch.Tensor, seed: int) -> torch.Tensor:
    flat = values.float().reshape(-1)
    if flat.numel() % 256:
        raise ValueError("local block-FHT input must divide into 256-vectors")
    signed = flat * deterministic_signs(flat, seed)
    transformed = normalized_fht_last_dim(signed.reshape(-1, 256))
    transformed = transformed * deterministic_signs(transformed, seed + 1)
    transformed = normalized_fht_last_dim(transformed)
    return transformed.reshape(-1)


def global_tensor_fht(values: torch.Tensor, seed: int) -> torch.Tensor:
    """Exact orthogonal transform over 768=3x256 and 3072=3x1024."""
    if tuple(values.shape) != (768, 3072):
        raise ValueError("global tensor FHT expects a 768x3072 c_proj matrix")
    matrix = values.float() * deterministic_signs(values, seed)
    row_q = deterministic_orthogonal3(seed + 11, matrix.device)
    col_q = deterministic_orthogonal3(seed + 17, matrix.device)
    matrix = matrix.reshape(3, 256, 3072)
    matrix = normalized_fht_last_dim(matrix.transpose(1, 2)).transpose(1, 2)
    matrix = torch.einsum("ab,bic->aic", row_q, matrix).reshape(768, 3072)
    matrix = matrix.reshape(768, 3, 1024)
    matrix = normalized_fht_last_dim(matrix)
    matrix = torch.einsum("ab,rbi->rai", col_q, matrix)
    return matrix.reshape(-1)


def basis_coefficients(values: torch.Tensor, basis: str, seed: int) -> torch.Tensor:
    if basis == "identity":
        return values.float().reshape(-1)
    if basis == "local_block_fht256":
        return local_block_fht256(values, seed)
    if basis == "global_tensor_fht":
        return global_tensor_fht(values, seed)
    raise ValueError(f"unknown basis: {basis}")


def topk_energy(coefficients: torch.Tensor, count: int) -> tuple[float, torch.Tensor]:
    energy = coefficients.double().square()
    values, indices = torch.topk(energy, count, sorted=False)
    recovery = values.sum() / energy.sum().clamp_min(1e-30)
    return float(recovery), indices


def support_energy(coefficients: torch.Tensor, indices: torch.Tensor) -> float:
    energy = coefficients.double().square()
    return float(energy[indices].sum() / energy.sum().clamp_min(1e-30))


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected global-orthogonal plan schema")
    observed = {
        "parameter_updates": plan.get("analysis", {}).get("parameter_updates"),
        "layers": plan.get("analysis", {}).get("layers"),
        "phases": plan.get("analysis", {}).get("phases"),
        "coordinate_budget": plan.get("analysis", {}).get("coordinate_budget"),
        "seeds": plan.get("analysis", {}).get("orthogonal_seeds"),
        "chart": plan.get("analysis", {}).get("chart"),
        "thresholds": plan.get("decision_rule", {}).get("thresholds"),
    }
    expected = {
        "parameter_updates": 0,
        "layers": list(LAYERS),
        "phases": [list(value) for value in PHASES],
        "coordinate_budget": 147456,
        "seeds": [20260807, 20260808, 20260809],
        "chart": {
            "hidden_parent_stages": 64,
            "hidden_residual_stages": 24,
            "output_stages": 32,
            "neighbors": 64,
            "matching_seed": 20260806,
            "weight_decay_application": "identical production ordering",
        },
        "thresholds": {
            "global_oracle_recovery_minimum": 0.4,
            "global_over_gaussian_enrichment_minimum": 1.25,
            "global_over_local_ratio_minimum": 1.1,
            "global_over_local_absolute_minimum": 0.02,
            "previous_support_recovery_minimum": 0.1,
            "previous_support_over_random_minimum": 1.5,
            "combined_exact_update_recovery_minimum": 0.6,
        },
    }
    if observed != expected:
        raise ValueError("global-orthogonal plan does not match the v1 contract")
    authorization = plan.get("authorization", {})
    if authorization.get("run_zero_update_global_orthogonal_analysis") is not True:
        raise ValueError("global orthogonal analysis is not authorized")
    for key in (
        "implement_candidate_structure",
        "run_exact_config_mfu",
        "run_language_model_training",
        "larger_rung",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"plan must keep {key} false")


def classify(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    gates = {
        "oracle_recovery": metrics["global_oracle_recovery"]
        >= thresholds["global_oracle_recovery_minimum"],
        "gaussian_enrichment": metrics["global_over_gaussian_enrichment"]
        >= thresholds["global_over_gaussian_enrichment_minimum"],
        "locality_advantage_ratio": metrics["global_over_local_ratio"]
        >= thresholds["global_over_local_ratio_minimum"],
        "locality_advantage_absolute": metrics["global_over_local_absolute"]
        >= thresholds["global_over_local_absolute_minimum"],
        "previous_support_recovery": metrics["global_previous_support_recovery"]
        >= thresholds["previous_support_recovery_minimum"],
        "previous_support_enrichment": metrics[
            "global_previous_support_over_random"
        ]
        >= thresholds["previous_support_over_random_minimum"],
        "combined_exact_update_recovery": metrics[
            "global_combined_exact_update_recovery"
        ]
        >= thresholds["combined_exact_update_recovery_minimum"],
    }
    passed = all(gates.values())
    return {
        "classification": (
            "GLOBAL_ORTHOGONAL_LOCALITY_HYPOTHESIS_SUPPORTED"
            if passed
            else "REJECT_GLOBAL_ORTHOGONAL_LOCALITY_HYPOTHESIS"
        ),
        "passed": passed,
        "gates": gates,
        "authorization": {
            "global_orthogonal_structure_theory": passed,
            "implement_candidate_structure": False,
            "run_exact_config_mfu": False,
            "run_language_model_training": False,
            "larger_rung": False,
        },
    }


def reconstruct_residuals(
    *,
    plan: dict[str, Any],
    acquisition: dict[str, Any],
    snapshot_dir: Path,
    probe_dir: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], torch.Tensor]]:
    run_identity = plan["identity"]["run_identity_sha256"]
    snapshot_hashes = acquisition_artifact_hashes(acquisition, "snapshots")
    probe_hashes = acquisition_artifact_hashes(acquisition, "optimizer_probes")
    weights: dict[int, dict[int, torch.Tensor]] = {}
    for step in sorted({value for phase in PHASES for value in phase}):
        path = snapshot_dir / f"step_{step:06d}.pt"
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

    chart = plan["analysis"]["chart"]
    cells: list[dict[str, Any]] = []
    residuals: dict[tuple[int, int], torch.Tensor] = {}
    for phase_index, (start, _end) in enumerate(PHASES):
        path = probe_dir / f"step_{start:06d}.pt"
        if file_sha256(path) != probe_hashes[str(start)]:
            raise ValueError(f"probe SHA-256 mismatch at step {start}")
        probe = load_probe(path, start, run_identity)
        for layer in LAYERS:
            name = parameter_name(layer)
            state = probe["parameters"][name]
            hyper = probe["hyperparameters"][name]
            weight = weights[start][layer].to(device)
            torch.testing.assert_close(
                state["weight_before_step"], weight.cpu(), rtol=0.0, atol=0.0
            )
            lr = float(hyper["lr"])
            decay = float(hyper["weight_decay"])
            applied = state["applied_direction_per_lr"].to(device)
            exact = lr * applied
            seed = int(chart["matching_seed"]) + layer * 100000 + phase_index * 10
            hidden, output_residual, hidden_diagnostics = shared_hidden_chart(
                weight,
                exact,
                applied + decay * weight,
                parent_stages=int(chart["hidden_parent_stages"]),
                residual_stages=int(chart["hidden_residual_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed,
            )
            fitted, output_diagnostics = fit_frobenius_pass(
                hidden.T.contiguous(),
                output_residual.T.contiguous(),
                stages=int(chart["output_stages"]),
                neighbors=int(chart["neighbors"]),
                seed=seed + 2,
            )
            coordinates = sum(
                int(value["coordinates"]) for value in hidden_diagnostics
            ) + int(output_diagnostics["coordinates"])
            if coordinates != int(plan["analysis"]["coordinate_budget"]):
                raise ValueError("chart coordinate budget mismatch")
            chart_update = fitted.T.contiguous() * (1.0 - lr * decay) - weight
            residual = (exact - chart_update).detach()
            residual_energy = float(residual.double().square().sum())
            exact_energy = float(exact.double().square().sum())
            cells.append(
                {
                    "phase_start": start,
                    "layer": layer,
                    "residual_energy": residual_energy,
                    "exact_update_energy": exact_energy,
                    "chart_recovery": 1.0 - residual_energy / max(exact_energy, 1e-30),
                }
            )
            residuals[(start, layer)] = residual.cpu()
        del probe
    return cells, residuals


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
    if acquisition["identity"]["run_identity_sha256"] != plan["identity"][
        "run_identity_sha256"
    ]:
        raise ValueError("run identity mismatch")

    started = time.time()
    cells, residuals = reconstruct_residuals(
        plan=plan,
        acquisition=acquisition,
        snapshot_dir=args.snapshot_dir,
        probe_dir=args.probe_dir,
        device=args.device,
    )
    budget = int(plan["analysis"]["coordinate_budget"])
    seeds = [int(value) for value in plan["analysis"]["orthogonal_seeds"]]
    ambient = 768 * 3072
    random_fraction = budget / ambient
    basis_rows: list[dict[str, Any]] = []
    previous_support: dict[tuple[str, int, int], torch.Tensor] = {}
    gaussian_recovery: dict[int, float] = {}
    for seed in seeds:
        generator = torch.Generator(device=args.device).manual_seed(seed + 1000)
        gaussian = torch.randn(ambient, generator=generator, device=args.device)
        gaussian_recovery[seed], _indices = topk_energy(gaussian, budget)

    cell_by_key = {(row["phase_start"], row["layer"]): row for row in cells}
    for start, _end in PHASES:
        for layer in LAYERS:
            residual = residuals[(start, layer)].to(args.device)
            for basis in BASES:
                basis_seeds = [0] if basis == "identity" else seeds
                for seed in basis_seeds:
                    coefficients = basis_coefficients(residual, basis, seed)
                    before = float(residual.double().square().sum())
                    after = float(coefficients.double().square().sum())
                    if abs(after - before) / max(before, 1e-30) > 2e-5:
                        raise ValueError(f"{basis} is not energy preserving")
                    oracle, support = topk_energy(coefficients, budget)
                    key = (basis, seed, layer)
                    carry = (
                        None
                        if key not in previous_support
                        else support_energy(coefficients, previous_support[key])
                    )
                    previous_support[key] = support
                    gaussian = (
                        gaussian_recovery[seed]
                        if basis != "identity"
                        else sum(gaussian_recovery.values()) / len(gaussian_recovery)
                    )
                    basis_rows.append(
                        {
                            "phase_start": start,
                            "layer": layer,
                            "basis": basis,
                            "seed": seed,
                            "residual_energy": cell_by_key[(start, layer)][
                                "residual_energy"
                            ],
                            "exact_update_energy": cell_by_key[(start, layer)][
                                "exact_update_energy"
                            ],
                            "chart_recovery": cell_by_key[(start, layer)][
                                "chart_recovery"
                            ],
                            "oracle_topk_recovery": oracle,
                            "gaussian_topk_recovery": gaussian,
                            "oracle_over_gaussian": oracle / gaussian,
                            "previous_support_recovery": carry,
                            "random_support_recovery": random_fraction,
                            "previous_support_over_random": (
                                None if carry is None else carry / random_fraction
                            ),
                            "combined_exact_update_recovery": (
                                cell_by_key[(start, layer)]["chart_recovery"]
                                + (
                                    1.0
                                    - cell_by_key[(start, layer)]["chart_recovery"]
                                )
                                * oracle
                            ),
                        }
                    )

    def weighted_basis(field: str, basis: str, *, carry: bool = False) -> float:
        selected = [
            row
            for row in basis_rows
            if row["basis"] == basis and (not carry or row[field] is not None)
        ]
        numerator = sum(
            float(row[field]) * float(row["residual_energy"]) for row in selected
        )
        denominator = sum(float(row["residual_energy"]) for row in selected)
        return numerator / max(denominator, 1e-30)

    metrics = {
        "coordinate_fraction": random_fraction,
        "gaussian_oracle_recovery": sum(gaussian_recovery.values())
        / len(gaussian_recovery),
        "identity_oracle_recovery": weighted_basis(
            "oracle_topk_recovery", "identity"
        ),
        "local_oracle_recovery": weighted_basis(
            "oracle_topk_recovery", "local_block_fht256"
        ),
        "global_oracle_recovery": weighted_basis(
            "oracle_topk_recovery", "global_tensor_fht"
        ),
        "global_previous_support_recovery": weighted_basis(
            "previous_support_recovery", "global_tensor_fht", carry=True
        ),
        "global_combined_exact_update_recovery": weighted_basis(
            "combined_exact_update_recovery", "global_tensor_fht"
        ),
    }
    metrics["global_over_gaussian_enrichment"] = metrics[
        "global_oracle_recovery"
    ] / metrics["gaussian_oracle_recovery"]
    metrics["global_over_local_ratio"] = metrics["global_oracle_recovery"] / metrics[
        "local_oracle_recovery"
    ]
    metrics["global_over_local_absolute"] = metrics[
        "global_oracle_recovery"
    ] - metrics["local_oracle_recovery"]
    metrics["global_previous_support_over_random"] = metrics[
        "global_previous_support_recovery"
    ] / random_fraction
    decision = classify(metrics, plan["decision_rule"]["thresholds"])

    args.output.mkdir(parents=True)
    cells_path = args.output / "global_orthogonal_cells.csv"
    result_path = args.output / "global_orthogonal_result.json"
    write_csv(cells_path, basis_rows)
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": decision["classification"],
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_mlp_cproj_global_orthogonal_residual",
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
            "run_identity_sha256": plan["identity"]["run_identity_sha256"],
        },
        "metrics": metrics,
        "decision": decision,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
