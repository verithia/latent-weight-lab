#!/usr/bin/env python3
"""Fit a layer-private paired Kronecker MLP family to a fixed dense teacher."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import evaluate_fixed_ce
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_shared_mlp_endpoint_function import (
    module_jvp,
    pair_metrics,
    rademacher_tangent,
    sha256_file,
    summarize,
    validate_core_configs,
)
from examples.nanogpt.analyze_shared_mlp_exact_family_teacher_fit import (
    atomic_json,
    collect_stratified_inputs,
    git_head,
    normalized_objective,
)
from examples.nanogpt.train import TokenBatchSource, fixed_eval_indices_digest, make_fixed_eval_indices


PLAN_SCHEMA = "mai_layer_private_paired_kronecker_teacher_fit_plan_v1"
RESULT_SCHEMA = "mai_layer_private_paired_kronecker_teacher_fit_result_v1"


@dataclass(frozen=True)
class MatrixShape:
    outer_out: int
    inner_out: int
    outer_in: int
    inner_in: int

    @property
    def output_width(self) -> int:
        return self.outer_out * self.inner_out

    @property
    def input_width(self) -> int:
        return self.outer_in * self.inner_in


FC_SHAPE = MatrixShape(96, 32, 24, 32)
PROJ_SHAPE = MatrixShape(24, 32, 96, 32)


def materialize(group: Tensor, channel: Tensor) -> Tensor:
    """Materialize sum_r group[r] tensor-product channel[r]."""
    if group.ndim != 3 or channel.ndim != 3 or group.shape[0] != channel.shape[0]:
        raise ValueError("factor shapes must be [rank, rows, cols]")
    return torch.einsum("rik,rjl->ijkl", group, channel).reshape(
        group.shape[1] * channel.shape[1], group.shape[2] * channel.shape[2]
    )


def rearrange(weight: Tensor, shape: MatrixShape) -> Tensor:
    if tuple(weight.shape) != (shape.output_width, shape.input_width):
        raise ValueError("weight does not match registered Kronecker shape")
    return weight.reshape(
        shape.outer_out, shape.inner_out, shape.outer_in, shape.inner_in
    ).permute(0, 2, 1, 3).reshape(
        shape.outer_out * shape.outer_in, shape.inner_out * shape.inner_in
    )


def randomized_kronecker_svd(
    weight: Tensor, shape: MatrixShape, rank: int, *, seed: int, niter: int
) -> tuple[Tensor, Tensor]:
    torch.manual_seed(int(seed))
    matrix = rearrange(weight.float(), shape)
    u, singular, v = torch.svd_lowrank(matrix, q=int(rank), niter=int(niter))
    order = torch.argsort(singular, descending=True)
    u, singular, v = u[:, order], singular[order], v[:, order]
    root = singular.clamp_min(0).sqrt()
    group = (u * root[None, :]).T.reshape(
        rank, shape.outer_out, shape.outer_in
    )
    channel = (v * root[None, :]).T.reshape(
        rank, shape.inner_out, shape.inner_in
    )
    return group, channel


def recovery(prediction: Tensor, target: Tensor) -> float:
    denominator = target.float().square().sum().clamp_min(1e-30)
    return float(1.0 - (prediction.float() - target.float()).square().sum() / denominator)


class PairedKroneckerMLP(nn.Module):
    def __init__(
        self,
        *,
        fc_group: Tensor,
        fc_channel: Tensor,
        proj_group: Tensor,
        proj_channel: Tensor,
    ) -> None:
        super().__init__()
        self.fc_group = nn.Parameter(fc_group.detach().float().clone())
        self.fc_channel = nn.Parameter(fc_channel.detach().float().clone())
        self.proj_group = nn.Parameter(proj_group.detach().float().clone())
        self.proj_channel = nn.Parameter(proj_channel.detach().float().clone())

    def weights(self) -> tuple[Tensor, Tensor]:
        return materialize(self.fc_group, self.fc_channel), materialize(
            self.proj_group, self.proj_channel
        )

    def forward(self, values: Tensor) -> Tensor:
        c_fc, c_proj = self.weights()
        return F.linear(F.gelu(F.linear(values, c_fc)), c_proj)


def build_layer_data(
    *, layer: int, banks: dict[str, dict[int, Tensor]], teacher: nn.Module,
    relative_rms: float, seed: int, device: str,
) -> dict[str, Tensor]:
    clean = torch.stack([banks[name][layer] for name in ("teacher", "candidate")]).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed) + layer * 1009)
    signs = torch.randint(0, 2, clean.shape, generator=generator, device=device).float().mul_(2).sub_(1)
    delta = signs * float(relative_rms) * clean.square().mean(-1, keepdim=True).sqrt()
    variants = torch.stack((clean, clean + delta, clean - delta))
    with torch.no_grad():
        targets = teacher.transformer.h[layer].mlp(variants.reshape(-1, variants.shape[-1])).reshape(
            *variants.shape[:-1], -1
        )
    return {"clean": clean, "variants": variants, "targets": targets}


@torch.no_grad()
def objective(module: PairedKroneckerMLP, data: dict[str, Tensor], chunk: int = 128) -> float:
    total, rows = 0.0, 0
    for start in range(0, data["variants"].shape[-2], chunk):
        stop = min(start + chunk, data["variants"].shape[-2])
        prediction = module(data["variants"][..., start:stop, :])
        value = normalized_objective(prediction, data["targets"][..., start:stop, :])
        total += float(value) * (stop - start)
        rows += stop - start
    return total / max(rows, 1)


@torch.no_grad()
def output_metrics(
    module: PairedKroneckerMLP, data: dict[str, Tensor], layer: int
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records, rows = [], []
    for bank_index, bank in enumerate(("teacher", "candidate")):
        values = data["clean"][bank_index]
        target = data["targets"][0, bank_index]
        prediction = module(values)
        metric = pair_metrics(target.cpu(), prediction.cpu())
        records.append(metric)
        rows.append({"layer": layer, "bank": bank, **metric})
    return summarize(records), rows


def jvp_metrics(
    module: PairedKroneckerMLP, data: dict[str, Tensor], *, layer: int,
    teacher: nn.Module, directions: int, seed: int, device: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records, rows = [], []
    teacher_mlp = teacher.transformer.h[layer].mlp
    for bank_index, bank in enumerate(("teacher", "candidate")):
        values = data["clean"][bank_index]
        target_parts, prediction_parts = [], []
        for direction in range(int(directions)):
            tangent = rademacher_tangent(
                values.shape, device=device,
                seed=int(seed) + layer * 1000 + bank_index * 100_000 + direction,
            )
            target_parts.append(module_jvp(teacher_mlp, values, tangent).cpu())
            prediction_parts.append(module_jvp(module, values, tangent).cpu())
        metric = pair_metrics(torch.stack(target_parts), torch.stack(prediction_parts))
        records.append(metric)
        rows.append({"layer": layer, "bank": bank, **metric})
    return summarize(records), rows


def fit_layer(
    module: PairedKroneckerMLP, *, fit: dict[str, Tensor], holdout: dict[str, Tensor],
    steps: int, batch_size: int, learning_rate: float, seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    optimizer = torch.optim.Adam(module.parameters(), lr=float(learning_rate))
    generator = torch.Generator(device=fit["clean"].device).manual_seed(int(seed))
    initial = objective(module, fit)
    initial_holdout, _ = output_metrics(module, holdout, -1)
    best_loss, best_state = math.inf, None
    started, finite = time.time(), True
    for step in range(int(steps)):
        indices = torch.randint(
            fit["clean"].shape[1], (min(int(batch_size), fit["clean"].shape[1]),),
            generator=generator, device=fit["clean"].device,
        )
        optimizer.zero_grad(set_to_none=True)
        variants = fit["variants"].index_select(-2, indices)
        targets = fit["targets"].index_select(-2, indices)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = normalized_objective(module(variants), targets)
        if not torch.isfinite(loss):
            finite = False
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 10.0)
        optimizer.step()
        current = float(loss.detach())
        if current < best_loss:
            best_loss = current
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            print(json.dumps({"fit_step": step + 1, "fit_steps": steps, "loss": current}), flush=True)
    if best_state is not None:
        module.load_state_dict(best_state)
    final = objective(module, fit) if finite else float("inf")
    holdout_objective = objective(module, holdout) if finite else float("inf")
    final_holdout, final_rows = output_metrics(module, holdout, -1)
    state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    return {
        "finite": finite,
        "initial_fit_objective": initial,
        "final_fit_objective": final,
        "objective_reduction_fraction": 1.0 - final / max(initial, 1e-30) if finite else None,
        "holdout_objective": holdout_objective,
        "initial_holdout_output": initial_holdout,
        "final_holdout_output": final_holdout,
        "final_holdout_rows": final_rows,
        "wall_seconds": time.time() - started,
    }, state


def classify(summary: dict[str, Any], gap: float, healthy: bool, gates: dict[str, Any]) -> bool:
    return bool(
        healthy
        and summary["output"]["mean_explained_target_energy"] >= gates["minimum_mean_output_recovery"]
        and summary["output"]["minimum_explained_target_energy"] >= gates["minimum_worst_output_recovery"]
        and summary["input_jvp"]["mean_explained_target_energy"] >= gates["minimum_mean_input_jvp_recovery"]
        and summary["input_jvp"]["minimum_explained_target_energy"] >= gates["minimum_worst_input_jvp_recovery"]
        and gap <= gates["maximum_fixed_validation_cross_entropy_gap"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=10)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA or args.device != "cuda":
        raise ValueError("unexpected plan or device")
    identity, protocol, measurement = plan["identities"], plan["fit_protocol"], plan["measurement"]
    teacher_path = Path(identity["dense_teacher_checkpoint"]["path"])
    candidate_path = Path(identity["state_bank_checkpoint"]["path"])
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    for path, expected in (
        (teacher_path, identity["dense_teacher_checkpoint"]["sha256"]),
        (candidate_path, identity["state_bank_checkpoint"]["sha256"]),
        (data_dir / "manifest.json", identity["dataset_manifest_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"identity mismatch: {path}")
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    teacher, candidate = load_model(teacher_path, args.device), load_model(candidate_path, args.device)
    validate_core_configs(candidate, teacher)

    def collect(seed: int, count: int) -> dict[str, dict[int, Tensor]]:
        batches = fixed_validation_batches(data_dir, int(protocol["token_batch_size"]), teacher.config.block_size, int(count), int(seed))
        return {
            "teacher": collect_stratified_inputs(teacher, batches, sample_cap=int(protocol["sample_cap_per_layer_per_bank"]), seed=int(seed), device=args.device),
            "candidate": collect_stratified_inputs(candidate, batches, sample_cap=int(protocol["sample_cap_per_layer_per_bank"]), seed=int(seed), device=args.device),
        }

    print("collecting fit and holdout banks", flush=True)
    fit_banks = collect(int(protocol["fit_token_seed"]), int(protocol["fit_batches"]))
    holdout_banks = collect(int(protocol["holdout_token_seed"]), int(protocol["holdout_batches"]))
    fit_data = {layer: build_layer_data(layer=layer, banks=fit_banks, teacher=teacher, relative_rms=float(protocol["local_perturbation_relative_rms"]), seed=int(protocol["fit_token_seed"]), device=args.device) for layer in range(12)}
    holdout_data = {layer: build_layer_data(layer=layer, banks=holdout_banks, teacher=teacher, relative_rms=float(protocol["local_perturbation_relative_rms"]), seed=int(protocol["holdout_token_seed"]), device=args.device) for layer in range(12)}

    max_rank = max(int(rank) for rank in plan["family"]["ranks"])
    initial: dict[int, dict[str, Tensor]] = {}
    for layer in range(1 if args.preflight_only else 12):
        mlp = teacher.transformer.h[layer].mlp
        fc_group, fc_channel = randomized_kronecker_svd(mlp.c_fc.weight, FC_SHAPE, max_rank, seed=int(protocol["svd_seed"]) + layer * 2, niter=int(protocol["svd_power_iterations"]))
        proj_group, proj_channel = randomized_kronecker_svd(mlp.c_proj.weight, PROJ_SHAPE, max_rank, seed=int(protocol["svd_seed"]) + layer * 2 + 1, niter=int(protocol["svd_power_iterations"]))
        initial[layer] = {"fc_group": fc_group, "fc_channel": fc_channel, "proj_group": proj_group, "proj_channel": proj_channel}

    if args.preflight_only:
        rank = max_rank
        tensors = {k: v[:rank] for k, v in initial[0].items()}
        module = PairedKroneckerMLP(**tensors).to(args.device)
        row, _ = fit_layer(module, fit=fit_data[0], holdout=holdout_data[0], steps=int(args.preflight_steps), batch_size=int(protocol["row_batch_size_per_bank"]), learning_rate=float(protocol["learning_rate"]), seed=int(protocol["fit_token_seed"]))
        seconds_per_step = row["wall_seconds"] / max(int(args.preflight_steps), 1)
        print(json.dumps({"preflight": "complete", "seconds_per_fit_step": seconds_per_step, "estimated_fit_seconds": seconds_per_step * int(protocol["steps_per_layer_rank"]) * 12 * len(plan["family"]["ranks"]), "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())}, sort_keys=True))
        return

    rank_results, rank_states = {}, {}
    for rank_value in plan["family"]["ranks"]:
        rank = int(rank_value)
        layer_rows, output_records, jvp_records, states = [], [], [], {}
        for layer in range(12):
            print(json.dumps({"rank": rank, "layer": layer}), flush=True)
            tensors = {k: v[:rank] for k, v in initial[layer].items()}
            module = PairedKroneckerMLP(**tensors).to(args.device)
            dense_mlp = teacher.transformer.h[layer].mlp
            initial_weight_recovery = {
                "c_fc": recovery(module.weights()[0], dense_mlp.c_fc.weight),
                "c_proj": recovery(module.weights()[1], dense_mlp.c_proj.weight),
            }
            row, state = fit_layer(module, fit=fit_data[layer], holdout=holdout_data[layer], steps=int(protocol["steps_per_layer_rank"]), batch_size=int(protocol["row_batch_size_per_bank"]), learning_rate=float(protocol["learning_rate"]), seed=int(protocol["fit_token_seed"]) + rank * 100_000 + layer * 1000)
            output_summary, output_rows = output_metrics(module, holdout_data[layer], layer)
            jvp_summary, jvp_rows = jvp_metrics(module, holdout_data[layer], layer=layer, teacher=teacher, directions=int(measurement["input_jvp_directions"]), seed=int(measurement["input_jvp_seed"]), device=args.device)
            row.update({"layer": layer, "initial_weight_recovery": initial_weight_recovery, "holdout_output": output_summary, "holdout_input_jvp": jvp_summary})
            layer_rows.append(row)
            output_records.extend(output_rows)
            jvp_records.extend(jvp_rows)
            states[str(layer)] = state
        rank_states[str(rank)] = states
        rank_results[str(rank)] = {"layers": layer_rows, "summary": {"output": summarize(output_records), "input_jvp": summarize(jvp_records)}}

    fixed = make_fixed_eval_indices(data_dir, int(measurement["fixed_eval_batch_size"]), int(measurement["fixed_eval_block_size"]), int(measurement["fixed_eval_batches"]), int(measurement["fixed_eval_seed"]))
    digest = fixed_eval_indices_digest(fixed)
    if digest != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation digest mismatch")
    source = TokenBatchSource(data_dir)
    teacher_ce = evaluate_fixed_ce(teacher, data_dir=data_dir, fixed_indices=fixed, split="val", eval_iters=int(measurement["fixed_eval_batches"]), eval_batch_size=int(measurement["fixed_eval_batch_size"]), block_size=int(measurement["fixed_eval_block_size"]), device=args.device, dtype="bfloat16", source=source)
    for rank_value in plan["family"]["ranks"]:
        rank = int(rank_value)
        splice = load_model(teacher_path, args.device)
        for layer in range(12):
            tensors = {k: v.to(args.device) for k, v in rank_states[str(rank)][str(layer)].items()}
            module = PairedKroneckerMLP(**tensors).to(args.device)
            c_fc, c_proj = module.weights()
            splice.transformer.h[layer].mlp.c_fc.weight.data.copy_(c_fc.to(splice.transformer.h[layer].mlp.c_fc.weight))
            splice.transformer.h[layer].mlp.c_proj.weight.data.copy_(c_proj.to(splice.transformer.h[layer].mlp.c_proj.weight))
        value = evaluate_fixed_ce(splice, data_dir=data_dir, fixed_indices=fixed, split="val", eval_iters=int(measurement["fixed_eval_batches"]), eval_batch_size=int(measurement["fixed_eval_batch_size"]), block_size=int(measurement["fixed_eval_block_size"]), device=args.device, dtype="bfloat16", source=source)
        rows = rank_results[str(rank)]["layers"]
        healthy = all(row["finite"] and (row["objective_reduction_fraction"] >= 0.5 or row["initial_holdout_output"]["mean_explained_target_energy"] >= plan["frozen_gates"]["minimum_mean_output_recovery"]) for row in rows)
        rank_results[str(rank)].update({"fixed_validation_cross_entropy": value, "gap": value - teacher_ce, "optimization_healthy": healthy})
        rank_results[str(rank)]["passes"] = classify(rank_results[str(rank)]["summary"], value - teacher_ce, healthy, plan["frozen_gates"])
        del splice
        torch.cuda.empty_cache()

    if rank_results["16"]["passes"]:
        classification = "RANK16_PASS"
    elif rank_results["32"]["passes"]:
        classification = "RANK32_ONLY_PASS"
    elif rank_results["32"]["optimization_healthy"]:
        classification = "FAMILY_FAIL_AT_22X"
    else:
        classification = "OPTIMIZATION_INCONCLUSIVE"
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "fitted_states.pt"
    torch.save({"schema_version": "mai_layer_private_paired_kronecker_state_v1", "ranks": rank_states}, state_path)
    result = {"schema_version": RESULT_SCHEMA, "classification": classification, "repository_commit": git_head(Path(__file__).resolve().parents[2]), "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)}, "identities": identity, "rank_results": rank_results, "teacher_validation_cross_entropy": teacher_ce, "fixed_eval_indices_sha256": digest, "state_artifact": {"path": str(state_path), "sha256": sha256_file(state_path)}, "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()), "wall_seconds": time.time() - started}
    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    print(json.dumps({"classification": classification, "rank_results": {k: {x: v[x] for x in ("fixed_validation_cross_entropy", "gap", "optimization_healthy", "passes")} | {"summary": v["summary"]} for k, v in rank_results.items()}, "result": str(result_path), "result_sha256": sha256_file(result_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
