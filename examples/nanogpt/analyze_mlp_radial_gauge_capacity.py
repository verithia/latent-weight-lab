#!/usr/bin/env python3
"""H56a paired O(32) radial-MLP gauge and fixed-rate capacity audit."""

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
    role_summary,
    tensor_sha256,
    write_json,
)
from examples.nanogpt.analyze_mlp_position_product_code_capacity import (
    BLOCK_WIDTH,
    CODEWORDS,
    _stack_target_blocks,
    evaluate_codebook,
    fit_codebook,
    initialize_global_codebook,
    pack_unsigned_codes,
    unpack_unsigned_codes,
)
from examples.nanogpt.analyze_mlp_synthetic_muon_program import TRAJECTORY_SCHEMA


SCHEMA_VERSION = "nanogpt_mlp_radial_gauge_capacity_v1"
PLAN_SCHEMA_VERSION = "nanogpt_mlp_radial_gauge_capacity_plan_v1"
GROUP_WIDTH = 32
GROUPS = ROWS // GROUP_WIDTH
RAW_H55_GLOBAL_CAPTURE = 0.13617485016584396


def deployment_accounting(
    *,
    block_width: int = BLOCK_WIDTH,
    codewords: int = CODEWORDS,
    code_bits: int = 5,
    deployed_nodes: int = DEPLOYED_NODES,
) -> dict[str, int | float]:
    if WIDTH % block_width:
        raise ValueError("readout block width must divide canonical width")
    if codewords > 2**code_bits:
        raise ValueError("readout code width cannot address codebook")
    blocks = deployed_nodes * ROWS * (WIDTH // block_width)
    private_bits = blocks * code_bits
    if private_bits % 8:
        raise ValueError("private product codes must be byte aligned")
    private_bytes = private_bits // 8
    dictionary_values = codewords * block_width
    dictionary_bits = 4 * dictionary_values
    if dictionary_bits % 8:
        raise ValueError("int4 global codebook must be byte aligned")
    dictionary_bytes = dictionary_bits // 8
    scale_values = codewords
    scale_bytes = 2 * scale_values
    total = private_bytes + dictionary_bytes + scale_bytes
    return {
        "dense_replaced_mlp_fp16_bytes": DENSE_REPLACED_MLP_FP16_BYTES,
        "deployed_block_values": blocks,
        "private_code_bits": private_bits,
        "private_code_bytes": private_bytes,
        "int4_dictionary_values": dictionary_values,
        "int4_dictionary_bytes": dictionary_bytes,
        "fp16_dictionary_scale_values": scale_values,
        "fp16_dictionary_scale_bytes": scale_bytes,
        "total_checkpoint_bytes": total,
        "checkpoint_byte_fraction": total / DENSE_REPLACED_MLP_FP16_BYTES,
        "persistent_pca_or_per_node_basis_values": 0,
    }


def radial_activation(
    value: torch.Tensor, *, tau: float = 0.0, epsilon: float = 1e-6
) -> torch.Tensor:
    norm = value.norm(dim=-1, keepdim=True)
    return F.gelu(norm - tau) * value / (norm + epsilon)


def load_pair_states(
    path: Path,
    *,
    fc_parameter: str,
    proj_parameter: str,
    state_limit: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    files = sorted(path.glob("step_*.pt"))
    if len(files) != 239:
        raise ValueError(f"expected 239 trajectory states, found {len(files)}")
    fc_rows = []
    proj_rows = []
    identity: str | None = None
    for expected_step, file in enumerate(files[:state_limit]):
        payload = torch.load(file, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != TRAJECTORY_SCHEMA:
            raise ValueError(f"unexpected trajectory schema in {file}")
        if int(payload["step"]) != expected_step:
            raise ValueError("trajectory step ordering mismatch")
        observed = str(payload["run_identity_sha256"])
        identity = observed if identity is None else identity
        if observed != identity:
            raise ValueError("trajectory identity changed")
        stored = payload["parameters"]
        fc_rows.append(stored[fc_parameter].detach().contiguous())
        proj_rows.append(stored[proj_parameter].detach().contiguous())
    return torch.stack(fc_rows), torch.stack(proj_rows), str(identity)


def _paired_cross(
    current_fc: torch.Tensor,
    current_proj: torch.Tensor,
    reference_fc: torch.Tensor,
    reference_proj: torch.Tensor,
    *,
    paired: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    fc_energy = reference_fc.square().sum(dim=(-2, -1))
    proj_energy = reference_proj.square().sum(dim=(-2, -1)).clamp_min(1e-30)
    weight = fc_energy / proj_energy
    cross = torch.einsum("gik,tgjk->tgij", reference_fc, current_fc)
    if paired:
        cross = cross + weight[None, :, None, None] * torch.einsum(
            "gik,tgjk->tgij", reference_proj, current_proj
        )
    return cross, weight


def procrustes_gauge(
    current_fc: torch.Tensor,
    current_proj: torch.Tensor,
    reference_fc: torch.Tensor,
    reference_proj: torch.Tensor,
    *,
    paired: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    cross, weight = _paired_cross(
        current_fc,
        current_proj,
        reference_fc,
        reference_proj,
        paired=paired,
    )
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    gauge = u @ vh
    identity = torch.eye(
        gauge.shape[-1], device=gauge.device, dtype=gauge.dtype
    )
    gauge[0] = identity
    return gauge, weight


def greedy_permutation_gauge(cross: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an exact group-local permutation using batched greedy matching."""
    batch_shape = cross.shape[:-2]
    size = cross.shape[-1]
    score = cross.reshape(-1, size, size).clone()
    mapping = torch.full(
        (score.shape[0], size), -1, device=score.device, dtype=torch.int64
    )
    negative = torch.finfo(score.dtype).min
    for _ in range(size):
        flat_index = score.flatten(1).argmax(dim=1)
        reference_row = flat_index // size
        current_row = flat_index % size
        batch = torch.arange(score.shape[0], device=score.device)
        mapping[batch, reference_row] = current_row
        score[batch, reference_row, :] = negative
        score[batch, :, current_row] = negative
    if bool((mapping < 0).any()):
        raise AssertionError("incomplete permutation")
    mapping = mapping.reshape(*batch_shape, size)
    identity = torch.arange(size, device=cross.device)
    mapping[0] = identity
    gauge = F.one_hot(mapping, num_classes=size).to(cross.dtype)
    return gauge, mapping


def apply_gauge(value: torch.Tensor, gauge: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tgij,tgjk->tgik", gauge, value)


def gauge_statistics(
    raw_fc: torch.Tensor,
    raw_proj: torch.Tensor,
    gauged_fc: torch.Tensor,
    gauged_proj: torch.Tensor,
    gauge: torch.Tensor,
    reference_fc: torch.Tensor,
    reference_proj: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, Any]:
    identity = torch.eye(
        gauge.shape[-1], device=gauge.device, dtype=gauge.dtype
    )
    orthogonality = (gauge @ gauge.transpose(-1, -2) - identity).norm(
        dim=(-2, -1)
    )
    before = (
        (raw_fc - reference_fc[None]).square().sum(dim=(-2, -1))
        + weight[None]
        * (raw_proj - reference_proj[None]).square().sum(dim=(-2, -1))
    )
    after = (
        (gauged_fc - reference_fc[None]).square().sum(dim=(-2, -1))
        + weight[None]
        * (gauged_proj - reference_proj[None]).square().sum(dim=(-2, -1))
    )
    determinant = torch.linalg.det(gauge)
    distance = (gauge - identity).norm(dim=(-2, -1))
    return {
        "maximum_orthogonality_error": float(orthogonality.max()),
        "mean_orthogonality_error": float(orthogonality.mean()),
        "step_zero_identity_error": float((gauge[0] - identity).abs().max()),
        "paired_objective_before": float(before.sum()),
        "paired_objective_after": float(after.sum()),
        "paired_objective_ratio": float(after.sum() / before.sum().clamp_min(1e-30)),
        "reflection_fraction": float((determinant < 0).float().mean()),
        "mean_distance_from_identity": float(distance.mean()),
        "maximum_distance_from_identity": float(distance.max()),
    }


def pca_node(
    states: torch.Tensor,
    *,
    components: int,
    chronological_discovery_states: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    centered = states - states.mean(dim=0, keepdim=True)
    flat = centered.flatten(1)
    gram = flat @ flat.T
    gram = (gram + gram.T) * 0.5
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    vectors = vectors[:, order]
    scales = eigenvalues[:components].sqrt().clamp_min(1e-20)
    pcs = vectors[:, :components].T @ flat
    pcs = pcs / scales[:, None]
    pcs = pcs.reshape(components, *states.shape[1:])
    pcs = pcs / pcs.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-20).view(
        -1, 1, 1
    )
    total = eigenvalues.sum().clamp_min(1e-30)
    selected = eigenvalues[:components]
    weights = selected / selected.sum().clamp_min(1e-30)
    effective_rank = total.square() / eigenvalues.square().sum().clamp_min(1e-30)
    chronological = None
    if (
        chronological_discovery_states is not None
        and states.shape[0] > chronological_discovery_states
        and chronological_discovery_states > components
    ):
        discovery = states[:chronological_discovery_states]
        discovery_mean = discovery.mean(dim=0, keepdim=True)
        discovery_flat = (discovery - discovery_mean).flatten(1)
        discovery_gram = (
            discovery_flat @ discovery_flat.T
            + (discovery_flat @ discovery_flat.T).T
        ) * 0.5
        discovery_eigenvalues, discovery_vectors = torch.linalg.eigh(discovery_gram)
        discovery_order = torch.argsort(discovery_eigenvalues, descending=True)
        discovery_vectors = discovery_vectors[:, discovery_order[:components]]
        discovery_scales = discovery_eigenvalues[
            discovery_order[:components]
        ].clamp_min(0.0).sqrt().clamp_min(1e-20)
        basis = discovery_vectors.T @ discovery_flat
        basis = basis / discovery_scales[:, None]
        basis = basis / basis.norm(dim=1, keepdim=True).clamp_min(1e-20)
        heldout = (states[chronological_discovery_states:] - discovery_mean).flatten(1)
        coefficients = heldout @ basis.T
        prediction = coefficients @ basis
        chronological = {
            "discovery_states": chronological_discovery_states,
            "heldout_states": states.shape[0] - chronological_discovery_states,
            "heldout_energy_capture": float(
                prediction.square().sum() / heldout.square().sum().clamp_min(1e-30)
            ),
            "heldout_squared_cosine": float(
                (prediction.flatten() @ heldout.flatten()).square()
                / (
                    prediction.square().sum()
                    * heldout.square().sum()
                ).clamp_min(1e-30)
            ),
        }
    return pcs.contiguous(), weights.contiguous(), {
        "state_count": states.shape[0],
        "top_component_energy_fraction": float(eigenvalues[0] / total),
        "top_k_energy_fraction": float(selected.sum() / total),
        "temporal_effective_rank": float(effective_rank),
        "eigenvalues": [float(value) for value in selected],
        "chronological": chronological,
    }


def fit_global_readout(
    pcs: tuple[torch.Tensor, ...],
    weights: tuple[torch.Tensor, ...],
    *,
    steps: int,
    batch_blocks: int,
    seed: int,
    block_batch: int,
    progress_callback: Any | None,
) -> tuple[dict[str, Any], torch.Tensor]:
    target_blocks = _stack_target_blocks(pcs, block_width=BLOCK_WIDTH)
    initial = initialize_global_codebook(
        target_blocks, codewords=CODEWORDS, seed=seed
    )
    learned, history = fit_codebook(
        target_blocks,
        initial=initial,
        steps=steps,
        batch_blocks=batch_blocks,
        ema_coefficient=0.25,
        seed=seed + 1,
        position_conditioned=False,
        progress_callback=progress_callback,
    )
    result = evaluate_codebook(
        pcs,
        weights,
        codebook=learned,
        block_width=BLOCK_WIDTH,
        block_batch=block_batch,
        position_conditioned=False,
    )
    result["history"] = history
    return result, learned


def self_test(device_name: str = "cpu") -> dict[str, Any]:
    device = torch.device(device_name)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(56)
    reference_fc = torch.randn(2, 4, 7, generator=generator).to(device)
    reference_proj = torch.randn(2, 4, 7, generator=generator).to(device)
    raw = torch.randn(2, 4, 4, generator=generator).to(device)
    q_true, _ = torch.linalg.qr(raw)
    current_fc = torch.einsum("gji,gjk->gik", q_true, reference_fc)[None]
    current_proj = torch.einsum("gji,gjk->gik", q_true, reference_proj)[None]
    gauge, _ = procrustes_gauge(
        current_fc,
        current_proj,
        reference_fc,
        reference_proj,
        paired=True,
    )
    # Step zero is deliberately identity, so test recovery on a two-state batch.
    current_fc = torch.cat((reference_fc[None], current_fc), dim=0)
    current_proj = torch.cat((reference_proj[None], current_proj), dim=0)
    gauge, _ = procrustes_gauge(
        current_fc,
        current_proj,
        reference_fc,
        reference_proj,
        paired=True,
    )
    recovered_fc = apply_gauge(current_fc, gauge)
    recovered_proj = apply_gauge(current_proj, gauge)
    recovery_error = float(
        (recovered_fc[1] - reference_fc).norm()
        / reference_fc.norm().clamp_min(1e-30)
        + (recovered_proj[1] - reference_proj).norm()
        / reference_proj.norm().clamp_min(1e-30)
    )
    value = torch.randn(11, 4, generator=generator).to(device)
    rotated = torch.einsum("ij,nj->ni", q_true[0], value)
    equivariance_error = float(
        (
            radial_activation(rotated)
            - torch.einsum("ij,nj->ni", q_true[0], radial_activation(value))
        ).norm()
        / radial_activation(value).norm().clamp_min(1e-30)
    )
    codes = torch.arange(32, dtype=torch.int64).repeat(7)
    packed = pack_unsigned_codes(codes, bits=5)
    unpacked = unpack_unsigned_codes(packed, values=codes.numel(), bits=5)
    accounting = deployment_accounting()
    if recovery_error > 1e-4:
        raise AssertionError(recovery_error)
    if equivariance_error > 1e-5:
        raise AssertionError(equivariance_error)
    if not torch.equal(codes, unpacked):
        raise AssertionError("five-bit packing failed")
    if accounting["total_checkpoint_bytes"] != 1_106_496:
        raise AssertionError(accounting)
    return {
        "status": "passed",
        "procrustes_recovery_relative_error_sum": recovery_error,
        "radial_equivariance_relative_error": equivariance_error,
        "packed_roundtrip": True,
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
        raise ValueError("unexpected H56a plan schema")
    accounting = deployment_accounting()
    planned_accounting = plan["fixed_product_code_readout"]["deployment_accounting"]
    if accounting != planned_accounting:
        raise ValueError({"computed": accounting, "planned": planned_accounting})
    if accounting["checkpoint_byte_fraction"] > 0.01:
        raise ValueError("H56 readout exceeds one-percent checkpoint budget")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device.index or 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()

    systems = plan["systems_preflight"]
    audit = plan["same_trajectory_audit"]
    state_count = int(systems["preflight_states"]) if args.preflight else 239
    components = (
        int(systems["preflight_components_per_node"])
        if args.preflight
        else int(plan["frozen_inventory"]["components_per_node"])
    )
    readout_steps = (
        int(systems["preflight_product_code_steps"])
        if args.preflight
        else 256
    )
    batch_blocks = (
        int(systems["preflight_batch_blocks"])
        if args.preflight
        else 8192
    )
    block_batch = 65536
    variants = ("paired", "c_fc_only", "permutation")
    pcs: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    weights: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    pca_manifest: dict[str, list[dict[str, Any]]] = {
        "raw": [],
        **{name: [] for name in variants},
    }
    gauge_manifest: dict[str, list[dict[str, Any]]] = {
        name: [] for name in variants
    }
    identity: str | None = None
    progress_path = args.output / "progress.json"
    total_updates = 3 + len(variants) * readout_steps

    def write_progress(stage: str, completed: int, stage_step: int, stage_steps: int) -> None:
        write_json(
            progress_path,
            {
                "schema_version": f"{SCHEMA_VERSION}_progress_v1",
                "stage": stage,
                "stage_step": stage_step,
                "stage_steps": stage_steps,
                "completed_updates": completed,
                "total_updates": total_updates,
                "fraction": completed / total_updates,
            },
        )

    write_progress("paired_gauge_acquisition", 0, 0, 3)
    paired_parameters = plan["frozen_inventory"]["paired_parameters"]
    chronological_discovery = (
        int(audit["chronological_discovery_states"])
        if not args.preflight
        else None
    )
    for layer_index, (fc_parameter, proj_parameter) in enumerate(paired_parameters):
        fc_cpu, proj_cpu, observed_identity = load_pair_states(
            args.trajectory_dir,
            fc_parameter=fc_parameter,
            proj_parameter=proj_parameter,
            state_limit=state_count,
        )
        identity = observed_identity if identity is None else identity
        if observed_identity != identity:
            raise ValueError("paired trajectory identity changed")
        raw_fc = fc_cpu.to(device=device, dtype=torch.float32).reshape(
            state_count, GROUPS, GROUP_WIDTH, WIDTH
        )
        raw_proj = proj_cpu.to(device=device, dtype=torch.float32).transpose(
            -1, -2
        ).reshape(state_count, GROUPS, GROUP_WIDTH, WIDTH)
        del fc_cpu, proj_cpu
        reference_fc = raw_fc[0].clone()
        reference_proj = raw_proj[0].clone()
        paired_gauge, weight = procrustes_gauge(
            raw_fc,
            raw_proj,
            reference_fc,
            reference_proj,
            paired=True,
        )
        cfc_gauge, _ = procrustes_gauge(
            raw_fc,
            raw_proj,
            reference_fc,
            reference_proj,
            paired=False,
        )
        paired_cross, _ = _paired_cross(
            raw_fc,
            raw_proj,
            reference_fc,
            reference_proj,
            paired=True,
        )
        permutation_gauge, permutation_mapping = greedy_permutation_gauge(
            paired_cross
        )
        gauges = {
            "paired": paired_gauge,
            "c_fc_only": cfc_gauge,
            "permutation": permutation_gauge,
        }
        raw_matrices = (
            raw_fc.reshape(state_count, ROWS, WIDTH),
            raw_proj.reshape(state_count, ROWS, WIDTH),
        )
        for raw_matrix in raw_matrices:
            _, _, manifest = pca_node(
                raw_matrix,
                components=components,
                chronological_discovery_states=chronological_discovery,
            )
            pca_manifest["raw"].append(manifest)
        for variant, gauge in gauges.items():
            gauged_fc = apply_gauge(raw_fc, gauge)
            gauged_proj = apply_gauge(raw_proj, gauge)
            stats = gauge_statistics(
                raw_fc,
                raw_proj,
                gauged_fc,
                gauged_proj,
                gauge,
                reference_fc,
                reference_proj,
                weight,
            )
            if variant == "permutation":
                identity_rows = torch.arange(GROUP_WIDTH, device=device)
                stats["fixed_row_fraction"] = float(
                    (permutation_mapping == identity_rows).float().mean()
                )
            gauge_manifest[variant].append(stats)
            for matrix in (
                gauged_fc.reshape(state_count, ROWS, WIDTH),
                gauged_proj.reshape(state_count, ROWS, WIDTH),
            ):
                component_rows, component_weights, manifest = pca_node(
                    matrix,
                    components=components,
                    chronological_discovery_states=chronological_discovery,
                )
                pcs[variant].append(component_rows)
                weights[variant].append(component_weights)
                pca_manifest[variant].append(manifest)
            del gauged_fc, gauged_proj
        del raw_fc, raw_proj, paired_gauge, cfc_gauge, paired_cross
        del permutation_gauge, permutation_mapping, gauges
        if device.type == "cuda":
            torch.cuda.empty_cache()
        write_progress(
            "paired_gauge_acquisition",
            layer_index + 1,
            layer_index + 1,
            3,
        )

    if identity != plan["frozen_inventory"]["trajectory_identity_sha256"]:
        raise ValueError("H56 trajectory identity mismatch")
    readouts: dict[str, Any] = {}
    codebook_hashes: dict[str, str] = {}
    seed = int(audit["seed"])
    for variant_index, variant in enumerate(variants):
        stage = f"{variant}_product_code_fit"
        offset = 3 + variant_index * readout_steps

        def callback(step: int, total: int, *, _stage: str = stage, _offset: int = offset) -> None:
            write_progress(_stage, _offset + step, step, total)

        result, codebook = fit_global_readout(
            tuple(pcs[variant]),
            tuple(weights[variant]),
            steps=readout_steps,
            batch_blocks=batch_blocks,
            seed=seed + 10 * variant_index,
            block_batch=block_batch,
            progress_callback=callback,
        )
        readouts[variant] = result
        codebook_hashes[variant] = tensor_sha256(codebook)
    paired_result = readouts["paired"]
    gates = plan["capacity_gates"]
    weighted_pass = all(
        row["weighted_top16_capture"]
        >= float(gates["weighted_top16_capture_min_every_node"])
        for row in paired_result["rows"]
    )
    role_pass = all(
        summary["median_weighted_top16_capture"]
        >= float(gates["weighted_top16_capture_median_each_role"])
        for summary in paired_result["role_summaries"].values()
    )
    minimum_pass = all(
        row["minimum_pc_capture"]
        >= float(gates["minimum_pc_capture_every_node"])
        for row in paired_result["rows"]
    )
    margin = paired_result["mean_weighted_capture"] - RAW_H55_GLOBAL_CAPTURE
    margin_pass = margin >= float(gates["minimum_absolute_margin_over_raw_h55"])
    maximum_orthogonality = max(
        row["maximum_orthogonality_error"]
        for row in gauge_manifest["paired"]
    )
    orthogonality_pass = maximum_orthogonality <= float(
        gates["maximum_orthogonality_error"]
    )
    identity_pass = all(
        row["step_zero_identity_error"] == 0.0
        for row in gauge_manifest["paired"]
    )
    finite_pass = all(
        math.isfinite(row["weighted_top16_capture"])
        and math.isfinite(row["minimum_pc_capture"])
        for row in paired_result["rows"]
    )
    capacity_pass = (
        weighted_pass
        and role_pass
        and minimum_pass
        and margin_pass
        and orthogonality_pass
        and identity_pass
        and finite_pass
    )
    classification = (
        "PREFLIGHT"
        if args.preflight
        else ("GAUGE_CAPACITY_PASSED_H56B_PENDING" if capacity_pass else "GAUGE_REJECTED")
    )
    gate = {
        "classification": classification,
        "capacity_pass": capacity_pass,
        "weighted_capture_every_node_pass": weighted_pass,
        "role_median_pass": role_pass,
        "minimum_pc_every_node_pass": minimum_pass,
        "absolute_margin_over_raw_h55": margin,
        "margin_pass": margin_pass,
        "maximum_orthogonality_error": maximum_orthogonality,
        "orthogonality_pass": orthogonality_pass,
        "step_zero_identity_pass": identity_pass,
        "finite_pass": finite_pass,
        "h56b_authorized": (not args.preflight) and capacity_pass,
    }
    accounting_path = args.output / "accounting.json"
    write_json(accounting_path, accounting)
    metrics = {
        "readouts": readouts,
        "raw_h55_global_unpositioned_mean_weighted_capture": RAW_H55_GLOBAL_CAPTURE,
        "pca_manifest": pca_manifest,
        "gauge_manifest": gauge_manifest,
        "gate": gate,
        "codebook_sha256": codebook_hashes,
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
        * 239
        / state_count
        * 256
        / readout_steps
        * 8192
        / batch_blocks
        if args.preflight
        else runtime
    )
    script = Path(__file__).resolve()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "preflight": args.preflight,
        "plan": plan,
        "inventory": {
            "trajectory_identity_sha256": identity,
            "state_count": state_count,
            "components": components,
            "paired_parameters": paired_parameters,
        },
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
            "The exact O(32) gauge applies to the proposed radial latent function, not coordinatewise GELU.",
            "The saved inventory contains weights only; a future radial module must rotate and account for c_fc bias with the same Q.",
            "The global K32 readout is a representation probe, not a compact trained language-model checkpoint.",
            "No H56b, function, CE, attention, or scale result is produced unless every capacity gate passes.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "classification": classification,
                "metadata": str(metadata_path),
                "paired_mean_weighted_capture": paired_result["mean_weighted_capture"],
                "paired_role_summaries": paired_result["role_summaries"],
                "c_fc_only_mean_weighted_capture": readouts["c_fc_only"]["mean_weighted_capture"],
                "permutation_mean_weighted_capture": readouts["permutation"]["mean_weighted_capture"],
                "absolute_margin_over_raw_h55": margin,
                "runtime_seconds": runtime,
                "projected_binding_runtime_seconds": projected,
                "peak_cuda_allocated_bytes": peak,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
