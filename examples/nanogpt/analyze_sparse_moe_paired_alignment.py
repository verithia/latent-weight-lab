#!/usr/bin/env python3
"""Audit sparse-MoE trajectory alignment in the complete expert gauge.

This is a zero-update diagnostic.  It uses a terminal model only to collect
two disjoint fixed activation frames, then evaluates historical compact expert
snapshots on those common frames.  It does not fit or train a decoder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.extract_moe_paired_snapshot import SCHEMA_VERSION
from examples.nanogpt.moe_paired_geometry import (
    functional_atom_similarity,
    maximum_weight_assignment,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


class SparseMLPInputCollector:
    def __init__(self, model: torch.nn.Module, layers: list[int], sample_cap: int) -> None:
        self.layers = set(layers)
        self.sample_cap = int(sample_cap)
        self.values: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.counts: dict[int, int] = defaultdict(int)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer, block in enumerate(model.transformer.h):
            if layer in self.layers:
                if not hasattr(block.mlp, "expert_c_fc"):
                    raise ValueError(f"layer {layer} is not a sparse complete-expert MLP")
                self.handles.append(
                    block.mlp.register_forward_pre_hook(self._hook(layer))
                )

    def _hook(self, layer: int):
        def hook(_module, inputs):
            remaining = self.sample_cap - self.counts[layer]
            if remaining <= 0:
                return
            rows = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            rows = rows[:remaining].cpu()
            self.values[layer].append(rows)
            self.counts[layer] += int(rows.shape[0])

        return hook

    def complete(self) -> bool:
        return all(self.counts[layer] >= self.sample_cap for layer in self.layers)

    def tensors(self) -> dict[int, torch.Tensor]:
        if not self.complete():
            raise RuntimeError("sparse MLP activation sample cap was not reached")
        return {
            layer: torch.cat(self.values[layer], dim=0)[: self.sample_cap]
            for layer in self.layers
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_inputs(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    layers: list[int],
    sample_cap: int,
    device: str,
) -> dict[int, torch.Tensor]:
    collector = SparseMLPInputCollector(model, layers, sample_cap)
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch.to(device), None)
                if collector.complete():
                    break
        return collector.tensors()
    finally:
        collector.close()


def load_paired_snapshot(path: Path, expected_layers: list[int]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"not a paired sparse-MoE snapshot: {path}")
    if payload.get("layers") != expected_layers:
        raise ValueError(f"paired snapshot layer identity mismatch: {path}")
    if not isinstance(payload.get("model"), dict):
        raise ValueError(f"paired snapshot has no model state: {path}")
    return payload


def _selected_similarity(similarity: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(similarity.shape[0], device=similarity.device)
    return similarity[rows, permutation.to(similarity.device)]


def _router_counts(router: torch.Tensor, values: torch.Tensor, top_k: int) -> torch.Tensor:
    logits = values.float() @ router.float().T
    tie = torch.arange(logits.shape[1], dtype=logits.dtype, device=logits.device)
    selected = torch.topk(
        logits - tie * torch.finfo(logits.dtype).eps,
        top_k,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    return torch.bincount(selected.flatten(), minlength=router.shape[0])


def paired_alignment_metrics(
    left_c_fc: torch.Tensor,
    left_c_proj: torch.Tensor,
    right_c_fc: torch.Tensor,
    right_c_proj: torch.Tensor,
    fit_activations: torch.Tensor,
    eval_activations: torch.Tensor,
) -> dict[str, float | torch.Tensor]:
    fit_similarity = functional_atom_similarity(
        left_c_fc,
        left_c_proj,
        right_c_fc,
        right_c_proj,
        fit_activations,
    )
    eval_similarity = functional_atom_similarity(
        left_c_fc,
        left_c_proj,
        right_c_fc,
        right_c_proj,
        eval_activations,
    )
    fit_permutation = maximum_weight_assignment(fit_similarity)
    eval_permutation = maximum_weight_assignment(eval_similarity)
    hidden = left_c_fc.shape[0]
    identity = torch.arange(hidden)
    permutation = fit_permutation
    device_permutation = permutation.to(right_c_fc.device)
    aligned_right_fc = right_c_fc.index_select(0, device_permutation)
    aligned_right_proj = right_c_proj.index_select(1, device_permutation)
    raw_chord = (right_c_fc - left_c_fc).float().square().sum() + (
        right_c_proj - left_c_proj
    ).float().square().sum()
    aligned_chord = (aligned_right_fc - left_c_fc).float().square().sum() + (
        aligned_right_proj - left_c_proj
    ).float().square().sum()

    epsilon = 1e-30
    left_fc_norm = left_c_fc.float().norm(dim=1).clamp_min(epsilon)
    right_fc_norm = aligned_right_fc.float().norm(dim=1).clamp_min(epsilon)
    left_proj_norm = left_c_proj.float().norm(dim=0).clamp_min(epsilon)
    right_proj_norm = aligned_right_proj.float().norm(dim=0).clamp_min(epsilon)
    fc_log_ratio = (right_fc_norm / left_fc_norm).log()
    proj_log_ratio = (right_proj_norm / left_proj_norm).log()
    centered_fc = fc_log_ratio - fc_log_ratio.mean()
    centered_proj = proj_log_ratio - proj_log_ratio.mean()
    gauge_correlation = (centered_fc * centered_proj).sum() / (
        centered_fc.norm() * centered_proj.norm()
    ).clamp_min(epsilon)

    with torch.no_grad():
        left_output = F.gelu(eval_activations.float() @ left_c_fc.float().T) @ left_c_proj.float().T
        right_output = F.gelu(eval_activations.float() @ right_c_fc.float().T) @ right_c_proj.float().T
        output_chord = (right_output - left_output).square().mean().sqrt()

    fit_selected_on_eval = _selected_similarity(eval_similarity, permutation)
    fit_selected = _selected_similarity(fit_similarity, permutation)
    eval_oracle = _selected_similarity(eval_similarity, eval_permutation)
    return {
        "fit_permutation": permutation,
        "eval_permutation": eval_permutation,
        "assignment_overlap": float((permutation == eval_permutation).float().mean()),
        "fit_identity_fraction": float((permutation == identity).float().mean()),
        "eval_identity_fraction": float((eval_permutation == identity).float().mean()),
        "fit_mean_similarity": float(fit_selected.mean()),
        "eval_mean_similarity_under_fit_assignment": float(fit_selected_on_eval.mean()),
        "eval_oracle_mean_similarity": float(eval_oracle.mean()),
        "eval_assignment_regret": float(eval_oracle.mean() - fit_selected_on_eval.mean()),
        "raw_chord_energy": float(raw_chord),
        "quotient_chord_energy": float(aligned_chord),
        "quotient_to_raw_chord_ratio": float(aligned_chord / raw_chord.clamp_min(epsilon)),
        "paired_log_norm_correlation": float(gauge_correlation),
        "expert_output_chord_rms": float(output_chord),
    }


def aggregate(rows: list[dict[str, Any]], occupancy_minimum: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty set of expert rows")
    metrics = [
        "assignment_overlap",
        "fit_identity_fraction",
        "eval_identity_fraction",
        "fit_mean_similarity",
        "eval_mean_similarity_under_fit_assignment",
        "eval_oracle_mean_similarity",
        "eval_assignment_regret",
        "quotient_to_raw_chord_ratio",
        "paired_log_norm_correlation",
        "expert_output_chord_rms",
    ]
    summary: dict[str, Any] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    summary["expert_rows"] = len(rows)
    summary["occupancy_minimum"] = occupancy_minimum
    summary["underoccupied_rows"] = sum(
        min(
            int(row["fit_left_count"]),
            int(row["fit_right_count"]),
            int(row["eval_left_count"]),
            int(row["eval_right_count"]),
        )
        < occupancy_minimum
        for row in rows
    )
    return summary


def minimum_occupancy(row: dict[str, Any]) -> int:
    return min(
        int(row["fit_left_count"]),
        int(row["fit_right_count"]),
        int(row["eval_left_count"]),
        int(row["eval_right_count"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--sample-cap", type=int, default=2048)
    parser.add_argument("--fit-seed", type=int, default=20260811)
    parser.add_argument("--eval-seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    source = plan["source_control"]
    layers = [int(layer) for layer in source["layers"]]
    steps = [int(step) for step in source["fixed_eval_steps"]]
    occupancy_minimum = 256
    snapshots: dict[int, tuple[Path, dict[str, Any]]] = {}
    for step in steps:
        path = args.snapshot_dir / f"step_{step:06d}_moe_paired_l0_l5_l11.pt"
        payload = load_paired_snapshot(path, layers)
        if int(payload["step"]) != step:
            raise ValueError(f"paired snapshot step mismatch: {path}")
        snapshots[step] = (path, payload)

    fit_batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        args.batches,
        args.fit_seed,
    )
    eval_batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        args.batches,
        args.eval_seed,
    )
    fit_tokens = torch.cat(fit_batches)
    eval_tokens = torch.cat(eval_batches)
    model = load_model(args.terminal_checkpoint, args.device)
    try:
        fit_inputs = collect_inputs(model, fit_batches, layers, args.sample_cap, args.device)
        eval_inputs = collect_inputs(model, eval_batches, layers, args.sample_cap, args.device)
    finally:
        del model
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    top_k = 2
    for start, end in zip(steps[:-1], steps[1:]):
        left = snapshots[start][1]["model"]
        right = snapshots[end][1]["model"]
        for layer in layers:
            prefix = f"transformer.h.{layer}.mlp."
            left_fc = left[prefix + "expert_c_fc"].to(args.device)
            left_proj = left[prefix + "expert_c_proj"].to(args.device)
            left_router = left[prefix + "router.weight"].to(args.device)
            right_fc = right[prefix + "expert_c_fc"].to(args.device)
            right_proj = right[prefix + "expert_c_proj"].to(args.device)
            right_router = right[prefix + "router.weight"].to(args.device)
            fit_layer = fit_inputs[layer].to(args.device)
            eval_layer = eval_inputs[layer].to(args.device)
            fit_counts_left = _router_counts(left_router, fit_layer, top_k)
            fit_counts_right = _router_counts(right_router, fit_layer, top_k)
            eval_counts_left = _router_counts(left_router, eval_layer, top_k)
            eval_counts_right = _router_counts(right_router, eval_layer, top_k)
            for expert in range(left_fc.shape[0]):
                metrics = paired_alignment_metrics(
                    left_fc[expert],
                    left_proj[expert],
                    right_fc[expert],
                    right_proj[expert],
                    fit_layer,
                    eval_layer,
                )
                metrics.pop("fit_permutation")
                metrics.pop("eval_permutation")
                rows.append(
                    {
                        "start_step": start,
                        "end_step": end,
                        "layer": layer,
                        "expert": expert,
                        "fit_left_count": int(fit_counts_left[expert]),
                        "fit_right_count": int(fit_counts_right[expert]),
                        "eval_left_count": int(eval_counts_left[expert]),
                        "eval_right_count": int(eval_counts_right[expert]),
                        **metrics,
                    }
                )
            del left_fc, left_proj, left_router
            del right_fc, right_proj, right_router, fit_layer, eval_layer

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "paired_alignment_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    eligible_rows = [row for row in rows if minimum_occupancy(row) >= occupancy_minimum]
    excluded_rows = [row for row in rows if minimum_occupancy(row) < occupancy_minimum]
    summary = aggregate(eligible_rows, occupancy_minimum)
    all_rows_summary = aggregate(rows, occupancy_minimum)
    assignment_gate = float(plan["frozen_diagnostic_gates"]["discovery_eval_assignment_overlap_min"])
    result = {
        "schema_version": "nanogpt_sparse_moe_paired_alignment_audit_v2",
        "scope": "zero-update alignment audit scored only on experts that satisfy the frozen occupancy rule; decoder recovery gates remain deferred to the denser seed-two trajectory",
        "plan": {"path": str(args.plan), "sha256": file_sha256(args.plan)},
        "terminal_checkpoint": {
            "path": str(args.terminal_checkpoint),
            "sha256": file_sha256(args.terminal_checkpoint),
        },
        "snapshots": {
            str(step): {"path": str(path), "sha256": file_sha256(path)}
            for step, (path, _payload) in snapshots.items()
        },
        "activation_frame": {
            "construction": "terminal-model MLP inputs applied as one common exogenous frame to every historical paired expert snapshot",
            "sample_cap_per_layer": args.sample_cap,
            "fit_seed": args.fit_seed,
            "eval_seed": args.eval_seed,
            "fit_token_sha256": tensor_sha256(fit_tokens),
            "eval_token_sha256": tensor_sha256(eval_tokens),
        },
        "summary": summary,
        "all_rows_summary": all_rows_summary,
        "occupancy": {
            "minimum_routed_tokens_per_split_and_endpoint": occupancy_minimum,
            "eligible_rows": len(eligible_rows),
            "excluded_rows": len(excluded_rows),
            "excluded": [
                {
                    "start_step": row["start_step"],
                    "end_step": row["end_step"],
                    "layer": row["layer"],
                    "expert": row["expert"],
                    "minimum_count": minimum_occupancy(row),
                }
                for row in excluded_rows
            ],
        },
        "gates": {
            "assignment_overlap_threshold": assignment_gate,
            "assignment_overlap_pass": summary["assignment_overlap"]["minimum"] >= assignment_gate,
            "occupancy_exclusion_rule_pass": (
                len(eligible_rows) > 0
                and len(eligible_rows) + len(excluded_rows) == len(rows)
                and summary["underoccupied_rows"] == 0
            ),
            "all_rows_occupancy_pass": len(excluded_rows) == 0,
            "decoder_recovery_gates_evaluated": False,
        },
        "rows": {"path": str(csv_path), "sha256": file_sha256(csv_path)},
        "source_sha256": file_sha256(Path(__file__)),
    }
    result_path = args.output / "paired_alignment_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
