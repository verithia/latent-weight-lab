#!/usr/bin/env python3
"""Fit gradient-selected paired neuron copies on the seven-trunk MLP.

This is a zero-LM-update representational capacity oracle.  It starts from
the exact trained seven-trunk endpoint, discovers a frozen set of conflicting
complete neurons on a separate state bank, then fits only the registered
shared groups and paired private copies to a dense teacher.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.analyze_shared_mlp_endpoint_function import (
    sha256_file,
    summarize,
    tensor_sha256,
    validate_core_configs,
)
from examples.nanogpt.analyze_shared_mlp_exact_family_teacher_fit import (
    GROUPS,
    atomic_json,
    build_group_data,
    classify,
    collect_stratified_inputs,
    full_objective,
    git_head,
    jvp_metrics,
    normalized_objective,
    output_metrics,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_seven_trunk_selected_atom_unsharing_plan_v1"
RESULT_SCHEMA = "mai_seven_trunk_selected_atom_unsharing_result_v1"
SEVEN_TRUNK_RESULT = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_shared_dense_mlp_trunk_groups7_1_1_1_1_1_3_4_5tpp_result.json"
)
TERMINAL_CONFLICT_RESULT = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_shared_mlp_seven_trunk_terminal_conflict_result.json"
)
SHARED_SPARSE_RESULT = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_shared_base_private_2of4_mlp_teacher_fit_result.json"
)


class PairedAtomGroupMLP(nn.Module):
    """One shared MLP with selected complete neurons private per layer."""

    def __init__(
        self,
        *,
        layers: tuple[int, ...],
        c_fc_weight: Tensor,
        c_proj_weight: Tensor,
        pre_gain: Tensor,
        output_log_gain: Tensor,
        private_indices: Tensor,
        private_fc: Tensor | None = None,
        private_proj: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.layers = tuple(int(layer) for layer in layers)
        self.c_fc_weight = nn.Parameter(c_fc_weight.detach().float().clone())
        self.c_proj_weight = nn.Parameter(
            c_proj_weight.detach().float().clone()
        )
        self.pre_gain = nn.Parameter(pre_gain.detach().float().clone())
        self.output_log_gain = nn.Parameter(
            output_log_gain.detach().float().clone()
        )
        indices = private_indices.detach().long().clone()
        self.register_buffer("private_indices", indices)
        if indices.ndim != 2 or indices.shape[0] != len(self.layers):
            raise ValueError("private atom index shape mismatch")
        hidden, width = self.c_fc_weight.shape
        if self.c_proj_weight.shape != (width, hidden):
            raise ValueError("shared MLP matrices do not pair")
        if self.pre_gain.shape != (len(self.layers), hidden):
            raise ValueError("pre-GELU gain shape mismatch")
        if self.output_log_gain.shape != (len(self.layers), width):
            raise ValueError("output gain shape mismatch")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= hidden
        ):
            raise ValueError("private atom index outside hidden width")
        if any(torch.unique(row).numel() != row.numel() for row in indices):
            raise ValueError("private atom indices must be unique per layer")
        if private_fc is None:
            private_fc = torch.stack(
                [self.c_fc_weight.detach().index_select(0, row) for row in indices]
            )
        if private_proj is None:
            private_proj = torch.stack(
                [self.c_proj_weight.detach().index_select(1, row) for row in indices]
            )
        self.private_fc = nn.Parameter(private_fc.detach().float().clone())
        self.private_proj = nn.Parameter(
            private_proj.detach().float().clone()
        )
        private_width = indices.shape[1]
        if self.private_fc.shape != (len(self.layers), private_width, width):
            raise ValueError("private c_fc row shape mismatch")
        if self.private_proj.shape != (
            len(self.layers),
            width,
            private_width,
        ):
            raise ValueError("private c_proj column shape mismatch")

    @property
    def private_width(self) -> int:
        return int(self.private_indices.shape[1])

    def forward_layer(self, layer_offset: int, values: Tensor) -> Tensor:
        offset = int(layer_offset)
        indices = self.private_indices[offset]
        leading = values.shape[:-1]
        flat = values.reshape(-1, values.shape[-1])
        shared_pre = F.linear(flat, self.c_fc_weight).reshape(*leading, -1)
        shared_hidden = F.gelu(shared_pre * self.pre_gain[offset])
        output = F.linear(
            shared_hidden.reshape(-1, shared_hidden.shape[-1]),
            self.c_proj_weight,
        ).reshape(*leading, -1)
        if self.private_width:
            private_pre = F.linear(flat, self.private_fc[offset]).reshape(
                *leading, -1
            )
            private_hidden = F.gelu(
                private_pre * self.pre_gain[offset].index_select(0, indices)
            )
            old_hidden = shared_hidden.index_select(-1, indices)
            private_write = F.linear(
                private_hidden.reshape(-1, self.private_width),
                self.private_proj[offset],
            ).reshape(*leading, -1)
            old_write = F.linear(
                old_hidden.reshape(-1, self.private_width),
                self.c_proj_weight.index_select(1, indices),
            ).reshape(*leading, -1)
            output = output + private_write - old_write
        return output * self.output_log_gain[offset].exp()

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 5 or values.shape[1] != len(self.layers):
            raise ValueError("unexpected grouped input shape")
        return torch.stack(
            [
                self.forward_layer(offset, values[:, offset])
                for offset in range(len(self.layers))
            ],
            dim=1,
        )

    @torch.no_grad()
    def effective_weights(self, layer_offset: int) -> tuple[Tensor, Tensor]:
        offset = int(layer_offset)
        indices = self.private_indices[offset]
        c_fc = self.c_fc_weight.clone()
        c_proj = self.c_proj_weight.clone()
        if self.private_width:
            c_fc.index_copy_(0, indices, self.private_fc[offset])
            c_proj.index_copy_(1, indices, self.private_proj[offset])
        c_fc.mul_(self.pre_gain[offset][:, None])
        c_proj.mul_(self.output_log_gain[offset].exp()[:, None])
        return c_fc, c_proj

    @torch.no_grad()
    def clamp_gains(self) -> None:
        self.pre_gain.clamp_(-4.0, 4.0)
        self.output_log_gain.clamp_(-4.0, 4.0)


def initialize_group(
    *,
    layers: tuple[int, ...],
    candidate: nn.Module,
    private_indices: Tensor,
    device: str,
) -> PairedAtomGroupMLP:
    root = candidate.transformer.h[layers[0]].mlp
    pre_gain = torch.stack(
        [candidate.transformer.h[layer].mlp.pregelu_gain for layer in layers]
    )
    output_log_gain = torch.stack(
        [
            candidate.transformer.h[layer].mlp.residual_output_log_gain
            * candidate.transformer.h[layer].mlp.residual_output_gain_scale
            for layer in layers
        ]
    )
    return PairedAtomGroupMLP(
        layers=layers,
        c_fc_weight=root.c_fc.weight,
        c_proj_weight=root.c_proj.weight,
        pre_gain=pre_gain,
        output_log_gain=output_log_gain,
        private_indices=private_indices,
    ).to(device)


def score_private_atoms(
    fc_gradients: Tensor,
    proj_gradients: Tensor,
    *,
    private_width: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Select paired neurons from the layer-private gradient component."""

    if fc_gradients.ndim != 3 or proj_gradients.ndim != 3:
        raise ValueError("expected layer-stacked matrix gradients")
    layers, hidden, width = fc_gradients.shape
    if proj_gradients.shape != (layers, width, hidden):
        raise ValueError("gradient matrices do not pair")
    if not 0 < int(private_width) <= hidden:
        raise ValueError("private width outside hidden dimension")
    fc_residual = fc_gradients - fc_gradients.mean(dim=0, keepdim=True)
    proj_residual = proj_gradients - proj_gradients.mean(dim=0, keepdim=True)
    fc_norm = fc_residual.square().sum(dim=-1).sqrt()
    proj_norm = proj_residual.square().sum(dim=1).sqrt()
    fc_rms = fc_norm.square().mean().sqrt().clamp_min(1e-30)
    proj_rms = proj_norm.square().mean().sqrt().clamp_min(1e-30)
    scores = ((fc_norm / fc_rms).square() + (proj_norm / proj_rms).square()).sqrt()
    selected = []
    for layer_scores in scores:
        ranked = torch.argsort(layer_scores, descending=True, stable=True)
        selected.append(ranked[: int(private_width)].sort().values)
    indices = torch.stack(selected)
    overlaps = []
    for left in range(layers):
        for right in range(left + 1, layers):
            overlap = torch.isin(indices[left], indices[right]).sum()
            overlaps.append(float(overlap) / float(private_width))
    diagnostics = {
        "fc_gradient_residual_rms": float(fc_rms),
        "proj_gradient_residual_rms": float(proj_rms),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "mean_pairwise_support_overlap_fraction": (
            sum(overlaps) / len(overlaps) if overlaps else 1.0
        ),
        "selected_score_mean_by_layer": [
            float(scores[layer].index_select(0, indices[layer]).mean())
            for layer in range(layers)
        ],
    }
    return indices, diagnostics


def select_group_support(
    *,
    layers: tuple[int, ...],
    candidate: nn.Module,
    selection: dict[str, Tensor],
    private_width: int,
) -> tuple[Tensor, dict[str, Any]]:
    root = candidate.transformer.h[layers[0]].mlp
    fc_gradients, proj_gradients = [], []
    losses = []
    for offset, layer in enumerate(layers):
        fc = root.c_fc.weight.detach().float().clone().requires_grad_(True)
        proj = root.c_proj.weight.detach().float().clone().requires_grad_(True)
        pre_gain = candidate.transformer.h[layer].mlp.pregelu_gain.detach().float()
        log_gain = (
            candidate.transformer.h[layer].mlp.residual_output_log_gain
            * candidate.transformer.h[layer].mlp.residual_output_gain_scale
        ).detach().float()
        values = selection["variants"][:, offset]
        leading = values.shape[:-1]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = F.linear(values.reshape(-1, values.shape[-1]), fc)
            hidden = hidden.reshape(*leading, -1) * pre_gain
            prediction = F.linear(
                F.gelu(hidden).reshape(-1, hidden.shape[-1]), proj
            ).reshape(*leading, -1)
            prediction = prediction * log_gain.exp()
            loss = normalized_objective(
                prediction, selection["targets"][:, offset]
            )
        grad_fc, grad_proj = torch.autograd.grad(loss, (fc, proj))
        fc_gradients.append(grad_fc.detach().float())
        proj_gradients.append(grad_proj.detach().float())
        losses.append(float(loss))
    indices, diagnostics = score_private_atoms(
        torch.stack(fc_gradients),
        torch.stack(proj_gradients),
        private_width=private_width,
    )
    diagnostics.update(
        {
            "layers": list(layers),
            "layer_objectives": losses,
            "support_sha256": tensor_sha256(indices.cpu()),
            "indices": indices.cpu().tolist(),
        }
    )
    return indices, diagnostics


def fit_group(
    module: PairedAtomGroupMLP,
    *,
    fit: dict[str, Tensor],
    holdout: dict[str, Tensor],
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    preflight: bool,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    optimizer = torch.optim.Adam(module.parameters(), lr=float(learning_rate))
    generator = torch.Generator(device=fit["clean"].device).manual_seed(int(seed))
    initial = full_objective(module, fit)
    best_objective = initial
    best_step = 0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    started = time.time()
    finite = math.isfinite(initial)
    for step in range(int(steps)):
        indices = torch.randint(
            fit["clean"].shape[2],
            (min(int(batch_size), fit["clean"].shape[2]),),
            generator=generator,
            device=fit["clean"].device,
        )
        variants = fit["variants"].index_select(-2, indices)
        targets = fit["targets"].index_select(-2, indices)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = normalized_objective(module(variants), targets)
        if not torch.isfinite(loss):
            finite = False
            break
        loss.backward()
        optimizer.step()
        module.clamp_gains()
        evaluate = preflight or (step + 1) % 100 == 0 or step + 1 == steps
        if evaluate:
            current = full_objective(module, fit)
            if current < best_objective:
                best_objective = current
                best_step = step + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
            print(
                json.dumps(
                    {
                        "layers": list(module.layers),
                        "fit_step": step + 1,
                        "fit_steps": steps,
                        "batch_loss": float(loss),
                        "full_fit_objective": current,
                        "best_step": best_step,
                        "best_objective": best_objective,
                    }
                ),
                flush=True,
            )
    module.load_state_dict(best_state)
    final = full_objective(module, fit) if finite else float("inf")
    holdout_objective = full_objective(module, holdout) if finite else float("inf")
    output_summary, output_rows = output_metrics(module, holdout)
    return (
        {
            "finite": bool(finite),
            "initial_fit_objective": initial,
            "final_fit_objective": final,
            "objective_reduction_fraction": (
                1.0 - final / max(initial, 1e-30) if finite else None
            ),
            "holdout_objective": holdout_objective,
            "holdout_output_summary": output_summary,
            "holdout_output_rows": output_rows,
            "best_step": best_step,
            "wall_seconds": time.time() - started,
        },
        best_state,
    )


@torch.no_grad()
def install_group(model: nn.Module, module: PairedAtomGroupMLP) -> None:
    for offset, layer in enumerate(module.layers):
        c_fc, c_proj = module.effective_weights(offset)
        mlp = model.transformer.h[layer].mlp
        mlp.c_fc.weight.copy_(c_fc.to(mlp.c_fc.weight))
        mlp.c_proj.weight.copy_(c_proj.to(mlp.c_proj.weight))


def validate_causal_results(plan: dict[str, Any]) -> None:
    causal = plan["causal_basis"]
    checks = (
        (SEVEN_TRUNK_RESULT, causal["seven_trunk_result_sha256"]),
        (TERMINAL_CONFLICT_RESULT, causal["terminal_conflict_result_sha256"]),
        (SHARED_SPARSE_RESULT, causal["shared_sparse_result_sha256"]),
    )
    for path, expected in checks:
        if sha256_file(path) != expected:
            raise ValueError(f"causal artifact identity mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=10)
    args = parser.parse_args()
    if args.device != "cuda":
        raise ValueError("the preregistered oracle requires CUDA")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected plan schema")
    validate_causal_results(plan)
    identity = plan["identities"]
    candidate_path = Path(identity["candidate_checkpoint"]["path"])
    teacher_path = Path(identity["dense_teacher_checkpoint"]["path"])
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    if sha256_file(candidate_path) != identity["candidate_checkpoint"]["sha256"]:
        raise ValueError("candidate checkpoint identity mismatch")
    if sha256_file(teacher_path) != identity["dense_teacher_checkpoint"]["sha256"]:
        raise ValueError("teacher checkpoint identity mismatch")
    if sha256_file(data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")

    selection_plan = plan["support_selection"]
    protocol = plan["fit_protocol"]
    measurement = plan["holdout_measurement"]
    torch.manual_seed(int(selection_plan["selection_token_seed"]))
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    candidate = load_model(candidate_path, args.device)
    teacher = load_model(teacher_path, args.device)
    validate_core_configs(candidate, teacher)
    if tuple(candidate.config.mlp_shared_dense_trunk_boundaries) != (
        1, 2, 3, 4, 5, 8, 12
    ):
        raise ValueError("candidate does not implement the frozen partition")
    if not candidate.config.block_fht_ffn_pregelu_gain:
        raise ValueError("candidate lacks signed pre-GELU gains")
    if not candidate.config.block_fht_mlp_residual_output_gain:
        raise ValueError("candidate lacks residual-output gains")

    def collect(seed: int, count: int, cap: int) -> dict[str, dict[int, Tensor]]:
        batches = fixed_validation_batches(
            data_dir,
            batch_size=int(protocol["token_batch_size"]),
            block_size=candidate.config.block_size,
            batches=int(count),
            seed=int(seed),
        )
        return {
            "candidate": collect_stratified_inputs(
                candidate,
                batches,
                sample_cap=int(cap),
                seed=int(seed),
                device=args.device,
            ),
            "teacher": collect_stratified_inputs(
                teacher,
                batches,
                sample_cap=int(cap),
                seed=int(seed),
                device=args.device,
            ),
        }

    print("collecting disjoint selection, fit, and holdout banks", flush=True)
    selection_banks = collect(
        int(selection_plan["selection_token_seed"]),
        int(selection_plan["selection_batches"]),
        int(selection_plan["sample_cap_per_layer_per_bank"]),
    )
    fit_banks = collect(
        int(protocol["fit_token_seed"]),
        int(protocol["fit_batches"]),
        int(protocol["sample_cap_per_layer_per_bank"]),
    )
    holdout_banks = collect(
        int(protocol["holdout_token_seed"]),
        int(protocol["holdout_batches"]),
        int(protocol["sample_cap_per_layer_per_bank"]),
    )

    group_data: dict[
        tuple[int, ...],
        tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]],
    ] = {}
    support_by_group: dict[tuple[int, ...], Tensor] = {}
    support_diagnostics: list[dict[str, Any]] = []
    for group_index, group in enumerate(GROUPS):
        selection = build_group_data(
            layers=group,
            banks=selection_banks,
            teacher=teacher,
            relative_rms=float(protocol["local_perturbation_relative_rms"]),
            seed=int(selection_plan["selection_token_seed"]) + group_index * 10_000,
            device=args.device,
        )
        fit = build_group_data(
            layers=group,
            banks=fit_banks,
            teacher=teacher,
            relative_rms=float(protocol["local_perturbation_relative_rms"]),
            seed=int(protocol["fit_token_seed"]) + group_index * 10_000,
            device=args.device,
        )
        holdout = build_group_data(
            layers=group,
            banks=holdout_banks,
            teacher=teacher,
            relative_rms=float(protocol["local_perturbation_relative_rms"]),
            seed=int(protocol["holdout_token_seed"]) + group_index * 10_000,
            device=args.device,
        )
        support, diagnostics = select_group_support(
            layers=group,
            candidate=candidate,
            selection=selection,
            private_width=int(plan["family"]["private_atoms_per_constrained_layer"]),
        )
        support_by_group[group] = support
        support_diagnostics.append(diagnostics)
        group_data[group] = (selection, fit, holdout)

    if args.preflight_only:
        group = GROUPS[0]
        module = initialize_group(
            layers=group,
            candidate=candidate,
            private_indices=support_by_group[group],
            device=args.device,
        )
        _, fit, holdout = group_data[group]
        row, _ = fit_group(
            module,
            fit=fit,
            holdout=holdout,
            steps=int(args.preflight_steps),
            batch_size=int(protocol["row_batch_size_per_layer_per_bank"]),
            learning_rate=float(protocol["learning_rate"]),
            seed=int(protocol["fit_token_seed"]),
            preflight=True,
        )
        seconds = row["wall_seconds"] / max(int(args.preflight_steps), 1)
        print(
            json.dumps(
                {
                    "preflight": "complete",
                    "seconds_per_fit_step_including_evaluation": seconds,
                    "conservative_estimated_total_fit_seconds": seconds
                    * int(protocol["steps"])
                    * len(GROUPS),
                    "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "support_sha256": support_diagnostics[0]["support_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    selected_modules: dict[tuple[int, ...], PairedAtomGroupMLP] = {}
    group_results: list[dict[str, Any]] = []
    output_records: list[dict[str, float]] = []
    jvp_records: list[dict[str, float]] = []
    for group_index, group in enumerate(GROUPS):
        _, fit, holdout = group_data[group]
        module = initialize_group(
            layers=group,
            candidate=candidate,
            private_indices=support_by_group[group],
            device=args.device,
        )
        row, _ = fit_group(
            module,
            fit=fit,
            holdout=holdout,
            steps=int(protocol["steps"]),
            batch_size=int(protocol["row_batch_size_per_layer_per_bank"]),
            learning_rate=float(protocol["learning_rate"]),
            seed=int(protocol["fit_token_seed"]) + group_index * 100_000,
            preflight=False,
        )
        jvp_summary, jvp_rows = jvp_metrics(
            module,
            holdout,
            directions=int(measurement["input_jvp_directions"]),
            seed=int(measurement["input_jvp_seed"]),
            teacher=teacher,
            device=args.device,
        )
        row["holdout_jvp_summary"] = jvp_summary
        row["holdout_jvp_rows"] = jvp_rows
        output_records.extend(row["holdout_output_rows"])
        jvp_records.extend(jvp_rows)
        selected_modules[group] = module
        group_results.append({"layers": list(group), **row})

    output_summary = summarize(output_records)
    jvp_summary = summarize(jvp_records)
    fixed = make_fixed_eval_indices(
        data_dir,
        int(measurement["fixed_eval_batch_size"]),
        int(measurement["fixed_eval_block_size"]),
        int(measurement["fixed_eval_batches"]),
        int(measurement["fixed_eval_seed"]),
    )
    fixed_digest = fixed_eval_indices_digest(fixed)
    if fixed_digest != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation digest mismatch")
    source = TokenBatchSource(data_dir)
    teacher_ce = evaluate_fixed_ce(
        teacher,
        data_dir=data_dir,
        fixed_indices=fixed,
        split="val",
        eval_iters=int(measurement["fixed_eval_batches"]),
        eval_batch_size=int(measurement["fixed_eval_batch_size"]),
        block_size=int(measurement["fixed_eval_block_size"]),
        device=args.device,
        dtype="bfloat16",
        source=source,
    )
    splice = load_model(teacher_path, args.device)
    for module in selected_modules.values():
        install_group(splice, module)
    splice_ce = evaluate_fixed_ce(
        splice,
        data_dir=data_dir,
        fixed_indices=fixed,
        split="val",
        eval_iters=int(measurement["fixed_eval_batches"]),
        eval_batch_size=int(measurement["fixed_eval_batch_size"]),
        block_size=int(measurement["fixed_eval_block_size"]),
        device=args.device,
        dtype="bfloat16",
        source=source,
    )
    ce_gap = splice_ce - teacher_ce
    healthy = all(
        bool(row["finite"])
        and float(row["objective_reduction_fraction"]) >= 0.5
        for row in group_results
    )
    classification = classify(
        output=output_summary,
        jvp=jvp_summary,
        ce_gap=ce_gap,
        healthy=healthy,
        gates=plan["frozen_gates"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "selected_atom_states.pt"
    torch.save(
        {
            "schema_version": "mai_seven_trunk_selected_atom_unsharing_state_v1",
            "groups": {
                ",".join(str(layer) for layer in group): {
                    key: value.detach().cpu()
                    for key, value in module.state_dict().items()
                }
                for group, module in selected_modules.items()
            },
        },
        state_path,
    )
    result_path = args.output / "result.json"
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identity,
        "resource_accounting": plan["family"],
        "support_diagnostics": support_diagnostics,
        "group_results": group_results,
        "summaries": {
            "holdout_output": output_summary,
            "holdout_input_jvp": jvp_summary,
        },
        "fixed_evaluation": {
            "indices_sha256": fixed_digest,
            "teacher_validation_cross_entropy": teacher_ce,
            "splice_validation_cross_entropy": splice_ce,
            "gap": ce_gap,
        },
        "optimization_healthy": healthy,
        "frozen_gates": plan["frozen_gates"],
        "state_artifact": {
            "path": str(state_path),
            "sha256": sha256_file(state_path),
        },
        "state_bank_sha256": {
            "selection_teacher": tensor_sha256(
                torch.stack(list(selection_banks["teacher"].values()))
            ),
            "selection_candidate": tensor_sha256(
                torch.stack(list(selection_banks["candidate"].values()))
            ),
            "fit_teacher": tensor_sha256(
                torch.stack(list(fit_banks["teacher"].values()))
            ),
            "fit_candidate": tensor_sha256(
                torch.stack(list(fit_banks["candidate"].values()))
            ),
            "holdout_teacher": tensor_sha256(
                torch.stack(list(holdout_banks["teacher"].values()))
            ),
            "holdout_candidate": tensor_sha256(
                torch.stack(list(holdout_banks["candidate"].values()))
            ),
        },
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.time() - started,
    }
    atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "classification": classification,
                "summaries": result["summaries"],
                "fixed_evaluation": result["fixed_evaluation"],
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
