#!/usr/bin/env python3
"""H54a unquantized capacity gate for a shared hard-routed row dictionary."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_lowbit_global_frame_dct_carrier_capacity import (
    DENSE_REPLACED_MLP_FP16_BYTES,
    DEPLOYED_NODES,
    ROWS,
    WIDTH,
    file_sha256,
    load_node_pc_inventory,
    role_summary,
    tensor_sha256,
    write_json,
)


SCHEMA_VERSION = "nanogpt_mlp_shared_int4_row_dictionary_capacity_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_shared_int4_row_dictionary_capacity_plan_v1"
ATOMS = 1024
SPARSITY = 3


def deployment_accounting(
    *,
    atoms: int = ATOMS,
    sparsity: int = SPARSITY,
    width: int = WIDTH,
    rows: int = ROWS,
    deployed_nodes: int = DEPLOYED_NODES,
) -> dict[str, int | float]:
    dense_bytes = deployed_nodes * rows * width * 2
    dictionary_values = atoms * width
    dictionary_bits = dictionary_values * 4
    if dictionary_bits % 8:
        raise ValueError("int4 dictionary must be byte aligned")
    dictionary_bytes = dictionary_bits // 8
    scale_values = atoms
    scale_bytes = 2 * scale_values
    index_values = deployed_nodes * rows * sparsity
    index_bits = index_values * math.ceil(math.log2(atoms))
    if index_bits % 8:
        raise ValueError("private indices must be byte aligned")
    index_bytes = index_bits // 8
    coefficient_values = index_values
    coefficient_bytes = 2 * coefficient_values
    total = dictionary_bytes + scale_bytes + index_bytes + coefficient_bytes
    return {
        "dense_replaced_mlp_fp16_bytes": dense_bytes,
        "int4_dictionary_values": dictionary_values,
        "int4_dictionary_bytes": dictionary_bytes,
        "fp16_dictionary_scale_values": scale_values,
        "fp16_dictionary_scale_bytes": scale_bytes,
        "private_index_values": index_values,
        "private_index_bits": index_bits,
        "private_index_bytes": index_bytes,
        "fp16_private_coefficient_values": coefficient_values,
        "fp16_private_coefficient_bytes": coefficient_bytes,
        "total_checkpoint_bytes": total,
        "checkpoint_byte_fraction": total / dense_bytes,
        "persistent_pca_or_per_node_basis_values": 0,
    }


def solve_codes(
    rows: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    sparsity: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unit_dictionary = F.normalize(dictionary, dim=1)
    correlations = rows @ unit_dictionary.T
    indices = correlations.abs().topk(sparsity, dim=1).indices
    selected = unit_dictionary[indices]
    gram = torch.bmm(selected, selected.transpose(1, 2))
    identity = torch.eye(sparsity, device=rows.device, dtype=rows.dtype)[None]
    rhs = torch.bmm(selected, rows[:, :, None]).squeeze(-1)
    coefficients = torch.linalg.solve(gram + ridge * identity, rhs)
    prediction = (coefficients[:, :, None] * selected).sum(dim=1)
    return prediction, indices, coefficients


def initialize_dictionary(
    targets: tuple[torch.Tensor, ...],
    *,
    atoms: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    target_rows = [target.reshape(-1, target.shape[-1]) for target in targets]
    total_targets = sum(target.shape[0] for target in targets)
    base = atoms // total_targets
    remainder = atoms % total_targets
    selected_rows = []
    flat_target = 0
    for node_target in targets:
        for component in range(node_target.shape[0]):
            count = base + int(flat_target < remainder)
            rows = node_target[component]
            energy = rows.square().sum(dim=1).detach().cpu()
            indices = torch.multinomial(
                energy.clamp_min(1e-30),
                count,
                replacement=False,
                generator=generator,
            ).to(rows.device)
            selected_rows.append(rows[indices])
            flat_target += 1
    dictionary = torch.cat(selected_rows, dim=0)
    if dictionary.shape != (atoms, targets[0].shape[-1]):
        raise AssertionError(dictionary.shape)
    permutation = torch.randperm(atoms, generator=generator).to(dictionary.device)
    return F.normalize(dictionary[permutation], dim=1)


def sample_training_rows(
    targets: tuple[torch.Tensor, ...],
    *,
    batch_rows: int,
    generator: torch.Generator,
) -> torch.Tensor:
    target_count = sum(target.shape[0] for target in targets)
    base = batch_rows // target_count
    remainder = batch_rows % target_count
    sampled = []
    flat_target = 0
    for node_target in targets:
        for component in range(node_target.shape[0]):
            count = base + int(flat_target < remainder)
            rows = node_target[component]
            energy = rows.square().sum(dim=1).detach().cpu()
            indices = torch.multinomial(
                energy.clamp_min(1e-30),
                count,
                replacement=True,
                generator=generator,
            ).to(rows.device)
            sampled.append(F.normalize(rows[indices], dim=1))
            flat_target += 1
    return torch.cat(sampled, dim=0)


def fit_dictionary(
    targets: tuple[torch.Tensor, ...],
    *,
    dictionary: torch.Tensor,
    steps: int,
    batch_rows: int,
    sparsity: int,
    ridge: float,
    learning_rate: float,
    gradient_clip_norm: float,
    seed: int,
    progress_callback: Any | None = None,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    value = torch.nn.Parameter(dictionary.detach().clone())
    optimizer = torch.optim.Adam([value], lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history = []
    record_steps = {0, 1, 2, 3, 7, 15, 31, 63, 127, 191, 255, steps - 1}
    for step in range(steps):
        rows = sample_training_rows(
            targets, batch_rows=batch_rows, generator=generator
        )
        optimizer.zero_grad(set_to_none=True)
        prediction, indices, _ = solve_codes(
            rows, value, sparsity=sparsity, ridge=ridge
        )
        residual = prediction - rows
        loss = residual.square().sum(dim=1).mean()
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_([value], gradient_clip_norm)
        )
        optimizer.step()
        with torch.no_grad():
            value.copy_(F.normalize(value, dim=1))
        if progress_callback is not None:
            progress_callback(step + 1, steps)
        if step in record_steps:
            used = torch.bincount(indices.flatten(), minlength=value.shape[0])
            history.append(
                {
                    "step": step + 1,
                    "sample_relative_squared_error": float(loss.detach()),
                    "sample_capture": float(
                        1.0 - residual.square().sum() / rows.square().sum()
                    ),
                    "gradient_norm": gradient_norm,
                    "active_dictionary_fraction": float((used > 0).float().mean()),
                }
            )
    return F.normalize(value.detach(), dim=1), history


def _row_energy_strata(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> list[float]:
    energy = target.square().sum(dim=1)
    order = energy.argsort()
    values = []
    for group in torch.tensor_split(order, 4):
        target_group = target[group].flatten()
        prediction_group = prediction[group].flatten()
        denominator = (
            target_group.square().sum() * prediction_group.square().sum()
        ).clamp_min(1e-30)
        values.append(float((target_group @ prediction_group).square() / denominator))
    return values


def evaluate_dictionary(
    targets: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    dictionary: torch.Tensor,
    sparsity: int,
    ridge: float,
    row_batch: int,
) -> dict[str, Any]:
    rows_out = []
    utilization = torch.zeros(
        dictionary.shape[0], device=dictionary.device, dtype=torch.int64
    )
    strata_sums = torch.zeros(4, device=dictionary.device, dtype=torch.float64)
    strata_count = 0
    with torch.no_grad():
        for node, (target_components, weight) in enumerate(
            zip(targets, weights, strict=True)
        ):
            captures = []
            for target in target_components:
                predictions = []
                for start in range(0, target.shape[0], row_batch):
                    prediction, indices, _ = solve_codes(
                        target[start : start + row_batch],
                        dictionary,
                        sparsity=sparsity,
                        ridge=ridge,
                    )
                    predictions.append(prediction)
                    utilization += torch.bincount(
                        indices.flatten(), minlength=dictionary.shape[0]
                    )
                prediction = torch.cat(predictions, dim=0)
                target_flat = target.flatten()
                prediction_flat = prediction.flatten()
                denominator = (
                    target_flat.square().sum() * prediction_flat.square().sum()
                ).clamp_min(1e-30)
                captures.append((target_flat @ prediction_flat).square() / denominator)
                strata_sums += torch.tensor(
                    _row_energy_strata(target, prediction),
                    device=dictionary.device,
                    dtype=torch.float64,
                )
                strata_count += 1
            capture = torch.stack(captures)
            rows_out.append(
                {
                    "index": node,
                    "weighted_top16_capture": float((capture * weight).sum()),
                    "uniform_mean_capture": float(capture.mean()),
                    "minimum_pc_capture": float(capture.min()),
                    "median_pc_capture": float(capture.median()),
                    "maximum_pc_capture": float(capture.max()),
                    "component_captures": [float(value) for value in capture],
                }
            )
    probabilities = utilization.double() / utilization.sum().clamp_min(1)
    nonzero = probabilities > 0
    entropy = float(
        -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        / math.log(dictionary.shape[0])
    )
    return {
        "mean_weighted_capture": sum(
            row["weighted_top16_capture"] for row in rows_out
        )
        / len(rows_out),
        "rows": rows_out,
        "role_summaries": {
            "c_fc": role_summary(rows_out, (0, 2, 4)),
            "c_proj": role_summary(rows_out, (1, 3, 5)),
        },
        "dictionary_utilization": {
            "active_fraction": float((utilization > 0).float().mean()),
            "normalized_entropy": entropy,
            "maximum_load_fraction": float(utilization.max() / utilization.sum()),
            "assignment_count": int(utilization.sum()),
        },
        "mean_capture_by_row_energy_quartile": [
            float(value) for value in strata_sums / max(1, strata_count)
        ],
    }


def self_test(device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(54)
    dictionary = F.normalize(
        torch.randn(32, 16, generator=generator).to(device), dim=1
    )
    indices = torch.randint(0, 32, (19, 3), generator=generator).to(device)
    coefficients = torch.randn(19, 3, generator=generator).to(device)
    target = (coefficients[:, :, None] * dictionary[indices]).sum(dim=1)
    own_prediction = (coefficients[:, :, None] * dictionary[indices]).sum(dim=1)
    own_capture = float(
        (target.flatten() @ own_prediction.flatten()).square()
        / (target.square().sum() * own_prediction.square().sum()).clamp_min(1e-30)
    )
    recovered, recovered_indices, _ = solve_codes(
        target, dictionary, sparsity=3, ridge=1e-5
    )
    recovered_capture = float(
        (target.flatten() @ recovered.flatten()).square()
        / (target.square().sum() * recovered.square().sum()).clamp_min(1e-30)
    )
    accounting = deployment_accounting()
    if own_capture < 0.999999:
        raise AssertionError(own_capture)
    if not math.isfinite(recovered_capture):
        raise AssertionError(recovered_capture)
    if accounting["total_checkpoint_bytes"] != 1_114_112:
        raise AssertionError(accounting)
    if accounting["dense_replaced_mlp_fp16_bytes"] != DENSE_REPLACED_MLP_FP16_BYTES:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "synthetic_own_family_capture": own_capture,
        "synthetic_hard_reencode_capture": recovered_capture,
        "synthetic_active_atoms": int(recovered_indices.unique().numel()),
        "accounting": accounting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.device), sort_keys=True))
        return

    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unexpected H54a plan schema")
    decoder = plan["frozen_decoder"]
    accounting = deployment_accounting(
        atoms=int(decoder["dictionary_atoms"]),
        sparsity=int(decoder["private_atoms_per_row"]),
    )
    if accounting != plan["exact_deployment_accounting"]:
        raise ValueError({"computed": accounting, "planned": plan["exact_deployment_accounting"]})
    if accounting["checkpoint_byte_fraction"] > 0.01:
        raise ValueError("H54 exceeds one-percent checkpoint budget")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device.index or 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()

    fit_plan = plan["unquantized_capacity_fit"]
    systems = plan["systems_preflight"]
    components = (
        int(systems["preflight_components_per_node"])
        if args.preflight
        else int(plan["frozen_inventory"]["components_per_node"])
    )
    steps = (
        int(systems["preflight_steps"])
        if args.preflight
        else int(fit_plan["steps"])
    )
    batch_rows = (
        int(systems["preflight_batch_rows"])
        if args.preflight
        else int(fit_plan["batch_rows"])
    )
    targets, weights, inventory, _ = load_node_pc_inventory(
        args.trajectory_dir, components=components, device=args.device
    )
    if inventory["trajectory_identity_sha256"] != plan["frozen_inventory"]["trajectory_identity_sha256"]:
        raise ValueError("H54 trajectory identity mismatch")
    progress_path = args.output / "progress.json"

    def progress(step: int, total: int) -> None:
        write_json(
            progress_path,
            {
                "schema_version": f"{SCHEMA_VERSION}_progress_v1",
                "stage": "unquantized_dictionary_fit",
                "stage_step": step,
                "stage_steps": total,
                "completed_updates": step,
                "total_updates": total,
                "fraction": step / total,
            },
        )

    progress(0, steps)
    initial = initialize_dictionary(
        targets,
        atoms=int(decoder["dictionary_atoms"]),
        seed=int(fit_plan["seed"]),
    )
    learned, history = fit_dictionary(
        targets,
        dictionary=initial,
        steps=steps,
        batch_rows=batch_rows,
        sparsity=int(decoder["private_atoms_per_row"]),
        ridge=1e-5,
        learning_rate=float(fit_plan["learning_rate"]),
        gradient_clip_norm=float(fit_plan["gradient_clip_norm"]),
        seed=int(fit_plan["seed"]) + 1,
        progress_callback=progress,
    )
    row_batch = int(fit_plan["evaluation_row_batch"])
    candidate = evaluate_dictionary(
        targets,
        weights,
        dictionary=learned,
        sparsity=int(decoder["private_atoms_per_row"]),
        ridge=1e-5,
        row_batch=row_batch,
    )
    top_one = evaluate_dictionary(
        targets,
        weights,
        dictionary=learned,
        sparsity=1,
        ridge=1e-5,
        row_batch=row_batch,
    )
    initial_control = evaluate_dictionary(
        targets,
        weights,
        dictionary=initial,
        sparsity=int(decoder["private_atoms_per_row"]),
        ridge=1e-5,
        row_batch=row_batch,
    )
    candidate["history"] = history
    gates = plan["capacity_gates"]
    weighted_pass = all(
        row["weighted_top16_capture"]
        >= float(gates["unquantized_weighted_top16_capture_min_every_node"])
        for row in candidate["rows"]
    )
    role_pass = all(
        summary["median_weighted_top16_capture"]
        >= float(gates["unquantized_weighted_top16_capture_median_each_role"])
        for summary in candidate["role_summaries"].values()
    )
    minimum_pass = all(
        row["minimum_pc_capture"]
        >= float(gates["unquantized_minimum_pc_capture_every_node"])
        for row in candidate["rows"]
    )
    finite_pass = all(
        math.isfinite(row["weighted_top16_capture"])
        and math.isfinite(row["minimum_pc_capture"])
        for row in candidate["rows"]
    )
    capacity_pass = weighted_pass and role_pass and minimum_pass and finite_pass
    classification = (
        "PREFLIGHT"
        if args.preflight
        else (
            "UNQUANTIZED_PASSED_INT4_PENDING"
            if capacity_pass
            else "UNQUANTIZED_REJECTED"
        )
    )
    gate = {
        "classification": classification,
        "unquantized_capacity_pass": capacity_pass,
        "weighted_capture_every_node_pass": weighted_pass,
        "role_median_pass": role_pass,
        "minimum_pc_every_node_pass": minimum_pass,
        "finite_pass": finite_pass,
        "int4_stage_authorized": (not args.preflight) and capacity_pass,
    }
    accounting_path = args.output / "accounting.json"
    write_json(accounting_path, accounting)
    metrics = {
        "unquantized_top_three": candidate,
        "learned_top_one": top_one,
        "fixed_initial_top_three": initial_control,
        "gate": gate,
        "learned_dictionary_sha256": tensor_sha256(learned),
        "initial_dictionary_sha256": tensor_sha256(initial),
    }
    metrics_path = args.output / "metrics.json"
    write_json(metrics_path, metrics)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.time() - started
    peak = (
        torch.cuda.max_memory_allocated(device.index or 0)
        if device.type == "cuda"
        else 0
    )
    projected = (
        runtime
        * int(fit_plan["steps"])
        / max(1, steps)
        * int(fit_plan["batch_rows"])
        / batch_rows
        * int(plan["frozen_inventory"]["components_per_node"])
        / components
        if args.preflight
        else runtime
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "preflight": args.preflight,
        "plan": plan,
        "inventory": inventory,
        "accounting": accounting,
        "metrics": metrics,
        "self_test": self_test(args.device),
        "execution": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "source_status": subprocess.check_output(
                ["git", "status", "--short"], cwd=REPO_ROOT, text=True
            ).splitlines(),
            "entrypoint": str(script),
            "entrypoint_sha256": file_sha256(script),
            "plan_path": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "command": [str(script), *sys.argv[1:]],
            "runtime_seconds": runtime,
            "projected_binding_runtime_seconds": projected,
            "peak_cuda_allocated_bytes": peak,
            "device": args.device,
        },
        "outputs": {
            "accounting": {"path": str(accounting_path), "sha256": file_sha256(accounting_path)},
            "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            "progress": {"path": str(progress_path), "sha256": file_sha256(progress_path)},
        },
        "limitations": [
            "This all-PC unquantized fit is a necessary dictionary-capacity gate, not a compact checkpoint.",
            "Per-PC row codes and coefficients are alternative manifold points and are not stored simultaneously.",
            "No int4 candidate is fitted unless every frozen unquantized gate passes.",
            "No function, CE, attention, or scale result is produced by H54a.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "classification": classification,
                "metadata": str(metadata_path),
                "candidate_mean_weighted_capture": candidate["mean_weighted_capture"],
                "candidate_role_summaries": candidate["role_summaries"],
                "initial_mean_weighted_capture": initial_control["mean_weighted_capture"],
                "runtime_seconds": runtime,
                "projected_binding_runtime_seconds": projected,
                "peak_cuda_allocated_bytes": peak,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
