#!/usr/bin/env python3
"""Fit a shared MLP ridge dictionary plus private nonlinear layer atoms."""
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
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_shared_trunk_private_ridge_teacher_fit_plan_v1"
RESULT_SCHEMA = "mai_shared_trunk_private_ridge_teacher_fit_result_v1"


class SharedPrivateRidgeMLP(nn.Module):
    """One shared full-width MLP plus layer-private complete ridge atoms."""

    def __init__(
        self,
        *,
        shared_fc: Tensor,
        shared_proj: Tensor,
        pre_gain: Tensor,
        post_gain: Tensor,
        output_log_gain: Tensor,
        private_u: Tensor,
        private_bias: Tensor,
        private_v: Tensor,
    ) -> None:
        super().__init__()
        self.shared_fc = nn.Parameter(shared_fc.detach().float().clone())
        self.shared_proj = nn.Parameter(shared_proj.detach().float().clone())
        self.pre_gain = nn.Parameter(pre_gain.detach().float().clone())
        self.post_gain = nn.Parameter(post_gain.detach().float().clone())
        self.output_log_gain = nn.Parameter(
            output_log_gain.detach().float().clone()
        )
        self.private_u = nn.Parameter(private_u.detach().float().clone())
        self.private_bias = nn.Parameter(
            private_bias.detach().float().clone()
        )
        self.private_v = nn.Parameter(private_v.detach().float().clone())
        layers, hidden, width = (
            self.pre_gain.shape[0],
            self.shared_fc.shape[0],
            self.shared_fc.shape[1],
        )
        if self.shared_proj.shape != (width, hidden):
            raise ValueError("shared MLP matrix shapes do not pair")
        if self.pre_gain.shape != (layers, hidden):
            raise ValueError("pre_gain shape mismatch")
        if self.post_gain.shape != (layers, hidden):
            raise ValueError("post_gain shape mismatch")
        if self.output_log_gain.shape != (layers, width):
            raise ValueError("output_log_gain shape mismatch")
        private_width = self.private_u.shape[1]
        if self.private_u.shape != (layers, private_width, width):
            raise ValueError("private_u shape mismatch")
        if self.private_bias.shape != (layers, private_width):
            raise ValueError("private_bias shape mismatch")
        if self.private_v.shape != (layers, width, private_width):
            raise ValueError("private_v shape mismatch")

    @property
    def layers(self) -> int:
        return int(self.pre_gain.shape[0])

    @property
    def private_width(self) -> int:
        return int(self.private_u.shape[1])

    def _selected(self, value: Tensor, layer_indices: Tensor) -> Tensor:
        return value.index_select(0, layer_indices)

    def forward_selected(
        self, values: Tensor, layer_indices: Tensor
    ) -> Tensor:
        """Evaluate values shaped [variant, layer, bank, row, channel]."""

        if values.ndim != 5 or values.shape[1] != layer_indices.numel():
            raise ValueError("unexpected selected-layer input shape")
        leading = values.shape[:-1]
        hidden = F.linear(values.reshape(-1, values.shape[-1]), self.shared_fc)
        hidden = hidden.reshape(*leading, -1)
        pre = self._selected(self.pre_gain, layer_indices)
        post = self._selected(self.post_gain, layer_indices)
        shared = F.linear(
            (F.gelu(hidden * pre[None, :, None, None, :])
             * post[None, :, None, None, :]).reshape(-1, hidden.shape[-1]),
            self.shared_proj,
        ).reshape(*leading, -1)
        if self.private_width:
            u = self._selected(self.private_u, layer_indices)
            bias = self._selected(self.private_bias, layer_indices)
            v = self._selected(self.private_v, layer_indices)
            private_hidden = torch.einsum(
                "vlbri,lmi->vlbrm", values, u
            ) + bias[None, :, None, None, :]
            private = torch.einsum(
                "vlbrm,lom->vlbro", F.gelu(private_hidden), v
            )
            shared = shared + private
        output_gain = self._selected(
            self.output_log_gain, layer_indices
        ).exp()
        return shared * output_gain[None, :, None, None, :]

    def forward(self, values: Tensor) -> Tensor:
        indices = torch.arange(self.layers, device=values.device)
        return self.forward_selected(values, indices)

    def forward_layer(self, layer: int, values: Tensor) -> Tensor:
        hidden = F.linear(values, self.shared_fc)
        hidden = F.gelu(hidden * self.pre_gain[int(layer)])
        shared = F.linear(
            hidden * self.post_gain[int(layer)], self.shared_proj
        )
        if self.private_width:
            private = F.linear(
                F.gelu(
                    F.linear(
                        values,
                        self.private_u[int(layer)],
                        self.private_bias[int(layer)],
                    )
                ),
                self.private_v[int(layer)],
            )
            shared = shared + private
        return shared * self.output_log_gain[int(layer)].exp()

    @torch.no_grad()
    def clamp_charts(self) -> None:
        self.pre_gain.clamp_(-4.0, 4.0)
        self.post_gain.clamp_(-4.0, 4.0)
        self.output_log_gain.clamp_(-4.0, 4.0)


class InstalledRidgeMLP(nn.Module):
    """Layer view used only for fixed-checkpoint CE evaluation."""

    def __init__(self, family: SharedPrivateRidgeMLP, layer: int) -> None:
        super().__init__()
        self.family = family
        self.layer = int(layer)

    def forward(self, values: Tensor) -> Tensor:
        return self.family.forward_layer(self.layer, values)


def initial_family(checkpoint: nn.Module) -> SharedPrivateRidgeMLP:
    if not checkpoint.config.mlp_shared_dense_trunk:
        raise ValueError("initialization checkpoint is not a shared trunk")
    if int(checkpoint.config.mlp_shared_dense_trunk_groups) != 1:
        raise ValueError("initialization checkpoint is not a one-trunk model")
    blocks = list(checkpoint.transformer.h)
    root = blocks[0].mlp
    if root.c_fc.bias is not None or root.c_proj.bias is not None:
        raise ValueError("registered family assumes bias-free shared matrices")
    pre_gain = torch.stack([block.mlp.pregelu_gain for block in blocks])
    output_log_gain = torch.stack(
        [
            block.mlp.residual_output_log_gain
            * block.mlp.residual_output_gain_scale
            for block in blocks
        ]
    )
    layers, hidden = pre_gain.shape
    width = root.c_fc.weight.shape[1]
    return SharedPrivateRidgeMLP(
        shared_fc=root.c_fc.weight,
        shared_proj=root.c_proj.weight,
        pre_gain=pre_gain,
        post_gain=torch.ones_like(pre_gain),
        output_log_gain=output_log_gain,
        private_u=torch.empty(layers, 0, width, device=pre_gain.device),
        private_bias=torch.empty(layers, 0, device=pre_gain.device),
        private_v=torch.empty(layers, width, 0, device=pre_gain.device),
    )


def expand_private_width(
    module: SharedPrivateRidgeMLP, width: int, *, seed: int
) -> SharedPrivateRidgeMLP:
    width = int(width)
    if width < module.private_width:
        raise ValueError("private width cannot shrink along the nested path")
    added = width - module.private_width
    if not added:
        return module
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + width)
    new_u = torch.randn(
        module.layers,
        added,
        module.shared_fc.shape[1],
        generator=generator,
        dtype=torch.float32,
    ) * 0.02
    new_bias = torch.zeros(module.layers, added, dtype=torch.float32)
    new_v = torch.zeros(
        module.layers,
        module.shared_proj.shape[0],
        added,
        dtype=torch.float32,
    )
    device = module.shared_fc.device
    return SharedPrivateRidgeMLP(
        shared_fc=module.shared_fc,
        shared_proj=module.shared_proj,
        pre_gain=module.pre_gain,
        post_gain=module.post_gain,
        output_log_gain=module.output_log_gain,
        private_u=torch.cat((module.private_u.detach().cpu(), new_u), dim=1),
        private_bias=torch.cat(
            (module.private_bias.detach().cpu(), new_bias), dim=1
        ),
        private_v=torch.cat((module.private_v.detach().cpu(), new_v), dim=2),
    ).to(device)


def build_data(
    *,
    banks: dict[str, dict[int, Tensor]],
    teacher: nn.Module,
    relative_rms: float,
    seed: int,
    device: str,
) -> dict[str, Tensor]:
    layers = int(teacher.config.n_layer)
    clean = torch.stack(
        [
            torch.stack(
                [banks[bank][layer] for bank in ("teacher", "candidate")]
            )
            for layer in range(layers)
        ]
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    signs = (
        torch.randint(
            0,
            2,
            clean.shape,
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).float().mul_(2.0).sub_(1.0)
    )
    delta = signs * float(relative_rms) * clean.square().mean(
        dim=-1, keepdim=True
    ).sqrt()
    variants = torch.stack((clean, clean + delta, clean - delta))
    targets = torch.empty(
        *variants.shape[:-1],
        teacher.config.n_embd,
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for layer in range(layers):
            values = variants[:, layer].reshape(-1, variants.shape[-1])
            prediction = teacher.transformer.h[layer].mlp(values)
            targets[:, layer].copy_(
                prediction.reshape(*variants[:, layer].shape[:-1], -1)
            )
    return {"clean": clean, "variants": variants, "targets": targets}


@torch.no_grad()
def full_objective(
    module: SharedPrivateRidgeMLP,
    data: dict[str, Tensor],
    *,
    chunk: int = 128,
) -> float:
    total, rows = 0.0, 0
    for start in range(0, data["variants"].shape[-2], int(chunk)):
        stop = min(start + int(chunk), data["variants"].shape[-2])
        prediction = module(data["variants"][..., start:stop, :])
        value = normalized_objective(
            prediction, data["targets"][..., start:stop, :]
        )
        total += float(value) * (stop - start)
        rows += stop - start
    return total / max(rows, 1)


@torch.no_grad()
def output_metrics(
    module: SharedPrivateRidgeMLP, data: dict[str, Tensor]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records, rows = [], []
    for layer in range(module.layers):
        for bank_index, bank in enumerate(("teacher", "candidate")):
            target = data["targets"][0, layer, bank_index]
            prediction = module.forward_layer(
                layer, data["clean"][layer, bank_index]
            )
            metric = pair_metrics(target.cpu(), prediction.cpu())
            records.append(metric)
            rows.append({"layer": layer, "bank": bank, **metric})
    return summarize(records), rows


def jvp_metrics(
    module: SharedPrivateRidgeMLP,
    data: dict[str, Tensor],
    *,
    teacher: nn.Module,
    directions: int,
    seed: int,
    device: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records, rows = [], []
    for layer in range(module.layers):
        teacher_mlp = teacher.transformer.h[layer].mlp
        for bank_index, bank in enumerate(("teacher", "candidate")):
            values = data["clean"][layer, bank_index]
            teacher_parts, compact_parts = [], []
            for direction in range(int(directions)):
                tangent = rademacher_tangent(
                    values.shape,
                    device=device,
                    seed=(
                        int(seed)
                        + layer * 1000
                        + bank_index * 100_000
                        + direction
                    ),
                )
                teacher_parts.append(
                    module_jvp(teacher_mlp, values, tangent).cpu()
                )
                compact_parts.append(
                    module_jvp(
                        lambda x, layer=layer: module.forward_layer(layer, x),
                        values,
                        tangent,
                    ).cpu()
                )
            metric = pair_metrics(
                torch.stack(teacher_parts), torch.stack(compact_parts)
            )
            records.append(metric)
            rows.append({"layer": layer, "bank": bank, **metric})
    return summarize(records), rows


def fit_stage(
    module: SharedPrivateRidgeMLP,
    *,
    fit: dict[str, Tensor],
    holdout: dict[str, Tensor],
    steps: int,
    rows_per_layer: int,
    layers_per_update: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    optimizer = torch.optim.Adam(module.parameters(), lr=float(learning_rate))
    generator = torch.Generator(device=fit["clean"].device).manual_seed(
        int(seed)
    )
    initial = full_objective(module, fit)
    initial_holdout, _ = output_metrics(module, holdout)
    best_loss, best_state = math.inf, None
    started, finite = time.time(), True
    for step in range(int(steps)):
        layer_indices = torch.randperm(
            module.layers, generator=generator, device=fit["clean"].device
        )[: int(layers_per_update)]
        row_indices = torch.randint(
            fit["clean"].shape[2],
            (min(int(rows_per_layer), fit["clean"].shape[2]),),
            generator=generator,
            device=fit["clean"].device,
        )
        variants = fit["variants"].index_select(1, layer_indices)
        variants = variants.index_select(-2, row_indices)
        targets = fit["targets"].index_select(1, layer_indices)
        targets = targets.index_select(-2, row_indices)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = normalized_objective(
                module.forward_selected(variants, layer_indices), targets
            )
        if not torch.isfinite(loss):
            finite = False
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 10.0)
        optimizer.step()
        module.clamp_charts()
        current = float(loss.detach())
        if current < best_loss:
            best_loss = current
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in module.state_dict().items()
            }
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            print(
                json.dumps(
                    {
                        "private_width": module.private_width,
                        "fit_step": step + 1,
                        "fit_steps": steps,
                        "loss": current,
                    }
                ),
                flush=True,
            )
    if best_state is not None:
        module.load_state_dict(best_state)
    final = full_objective(module, fit) if finite else float("inf")
    holdout_value = full_objective(module, holdout) if finite else float("inf")
    final_holdout, final_rows = output_metrics(module, holdout)
    state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    return {
        "finite": finite,
        "initial_fit_objective": initial,
        "final_fit_objective": final,
        "objective_reduction_fraction": (
            1.0 - final / max(initial, 1e-30) if finite else None
        ),
        "holdout_objective": holdout_value,
        "initial_holdout_output": initial_holdout,
        "final_holdout_output": final_holdout,
        "final_holdout_rows": final_rows,
        "wall_seconds": time.time() - started,
    }, state


def family_from_state(state: dict[str, Tensor], device: str) -> SharedPrivateRidgeMLP:
    return SharedPrivateRidgeMLP(
        shared_fc=state["shared_fc"],
        shared_proj=state["shared_proj"],
        pre_gain=state["pre_gain"],
        post_gain=state["post_gain"],
        output_log_gain=state["output_log_gain"],
        private_u=state["private_u"],
        private_bias=state["private_bias"],
        private_v=state["private_v"],
    ).to(device)


def passes(
    row: dict[str, Any], gap: float, gates: dict[str, Any]
) -> bool:
    output, jvp = row["summary"]["output"], row["summary"]["input_jvp"]
    return bool(
        row["optimization_healthy"]
        and output["mean_explained_target_energy"]
        >= gates["minimum_mean_output_recovery"]
        and output["minimum_explained_target_energy"]
        >= gates["minimum_worst_output_recovery"]
        and jvp["mean_explained_target_energy"]
        >= gates["minimum_mean_input_jvp_recovery"]
        and jvp["minimum_explained_target_energy"]
        >= gates["minimum_worst_input_jvp_recovery"]
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
        raise ValueError("unexpected plan schema or device")
    identity = plan["identities"]
    paths = {
        key: Path(identity[key]["path"])
        for key in (
            "dense_teacher_checkpoint",
            "shared_trunk_initialization_checkpoint",
            "state_bank_checkpoint",
        )
    }
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    for key, path in paths.items():
        if sha256_file(path) != identity[key]["sha256"]:
            raise ValueError(f"checkpoint identity mismatch: {key}")
    if sha256_file(data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")
    protocol, measurement = plan["fit_protocol"], plan["measurement"]
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    teacher = load_model(paths["dense_teacher_checkpoint"], args.device)
    initializer = load_model(
        paths["shared_trunk_initialization_checkpoint"], args.device
    )
    candidate = load_model(paths["state_bank_checkpoint"], args.device)
    validate_core_configs(initializer, teacher)
    validate_core_configs(candidate, teacher)

    def collect(seed: int, count: int) -> dict[str, dict[int, Tensor]]:
        batches = fixed_validation_batches(
            data_dir,
            int(protocol["token_batch_size"]),
            teacher.config.block_size,
            int(count),
            int(seed),
        )
        return {
            "teacher": collect_stratified_inputs(
                teacher,
                batches,
                sample_cap=int(protocol["sample_cap_per_layer_per_bank"]),
                seed=int(seed),
                device=args.device,
            ),
            "candidate": collect_stratified_inputs(
                candidate,
                batches,
                sample_cap=int(protocol["sample_cap_per_layer_per_bank"]),
                seed=int(seed),
                device=args.device,
            ),
        }

    print("collecting fit and holdout banks", flush=True)
    fit_banks = collect(
        int(protocol["fit_token_seed"]), int(protocol["fit_batches"])
    )
    holdout_banks = collect(
        int(protocol["holdout_token_seed"]), int(protocol["holdout_batches"])
    )
    fit = build_data(
        banks=fit_banks,
        teacher=teacher,
        relative_rms=float(protocol["local_perturbation_relative_rms"]),
        seed=int(protocol["fit_token_seed"]),
        device=args.device,
    )
    holdout = build_data(
        banks=holdout_banks,
        teacher=teacher,
        relative_rms=float(protocol["local_perturbation_relative_rms"]),
        seed=int(protocol["holdout_token_seed"]),
        device=args.device,
    )
    module = initial_family(initializer).to(args.device)
    if args.preflight_only:
        module = expand_private_width(
            module,
            max(plan["family"]["nested_private_widths"]),
            seed=int(protocol["private_initialization_seed"]),
        )
        row, _ = fit_stage(
            module,
            fit=fit,
            holdout=holdout,
            steps=int(args.preflight_steps),
            rows_per_layer=int(protocol["row_batch_size_per_layer_per_bank"]),
            layers_per_update=int(protocol["layers_per_update"]),
            learning_rate=float(protocol["learning_rate"]),
            seed=int(protocol["fit_token_seed"]),
        )
        seconds = row["wall_seconds"] / max(int(args.preflight_steps), 1)
        print(
            json.dumps(
                {
                    "preflight": "complete",
                    "seconds_per_fit_step": seconds,
                    "estimated_fit_seconds": (
                        seconds
                        * int(protocol["steps_per_width"])
                        * len(plan["family"]["nested_private_widths"])
                    ),
                    "maximum_cuda_memory_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    width_results: dict[str, dict[str, Any]] = {}
    width_states: dict[str, dict[str, Tensor]] = {}
    for width_value in plan["family"]["nested_private_widths"]:
        width = int(width_value)
        module = expand_private_width(
            module,
            width,
            seed=int(protocol["private_initialization_seed"]),
        )
        row, state = fit_stage(
            module,
            fit=fit,
            holdout=holdout,
            steps=int(protocol["steps_per_width"]),
            rows_per_layer=int(protocol["row_batch_size_per_layer_per_bank"]),
            layers_per_update=int(protocol["layers_per_update"]),
            learning_rate=float(protocol["learning_rate"]),
            seed=int(protocol["fit_token_seed"]) + width * 1000,
        )
        output_summary, output_rows = output_metrics(module, holdout)
        jvp_summary, jvp_rows = jvp_metrics(
            module,
            holdout,
            teacher=teacher,
            directions=int(measurement["input_jvp_directions"]),
            seed=int(measurement["input_jvp_seed"]),
            device=args.device,
        )
        row.update(
            {
                "private_width": width,
                "compact_parameter_count": sum(
                    parameter.numel() for parameter in module.parameters()
                ),
                "summary": {
                    "output": output_summary,
                    "input_jvp": jvp_summary,
                },
                "holdout_output_rows": output_rows,
                "holdout_input_jvp_rows": jvp_rows,
            }
        )
        width_results[str(width)] = row
        width_states[str(width)] = state

    fixed = make_fixed_eval_indices(
        data_dir,
        int(measurement["fixed_eval_batch_size"]),
        int(measurement["fixed_eval_block_size"]),
        int(measurement["fixed_eval_batches"]),
        int(measurement["fixed_eval_seed"]),
    )
    digest = fixed_eval_indices_digest(fixed)
    if digest != identity["fixed_eval_indices_sha256"]:
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
    gates = plan["frozen_gates"]
    for width_value in plan["family"]["nested_private_widths"]:
        width = int(width_value)
        family = family_from_state(width_states[str(width)], args.device)
        splice = load_model(paths["dense_teacher_checkpoint"], args.device)
        for layer in range(family.layers):
            splice.transformer.h[layer].mlp = InstalledRidgeMLP(
                family, layer
            )
        value = evaluate_fixed_ce(
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
        row = width_results[str(width)]
        reduction = row["objective_reduction_fraction"]
        healthy = bool(
            row["finite"]
            and (
                row["final_fit_objective"] <= 0.5
                or (reduction is not None and reduction >= 0.25)
            )
        )
        row.update(
            {
                "fixed_validation_cross_entropy": value,
                "gap": value - teacher_ce,
                "optimization_healthy": healthy,
            }
        )
        row["passes"] = passes(row, value - teacher_ce, gates)
        del splice, family
        torch.cuda.empty_cache()

    passing = [
        int(width)
        for width in plan["family"]["nested_private_widths"]
        if width_results[str(width)]["passes"]
    ]
    if passing:
        classification = f"WIDTH{min(passing)}_PASS"
    elif width_results["256"]["optimization_healthy"]:
        classification = "FAMILY_FAIL_AT_WIDTH256"
    else:
        classification = "OPTIMIZATION_INCONCLUSIVE"
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "fitted_states.pt"
    torch.save(
        {
            "schema_version": "mai_shared_trunk_private_ridge_state_v1",
            "widths": width_states,
        },
        state_path,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identity,
        "width_results": width_results,
        "teacher_validation_cross_entropy": teacher_ce,
        "fixed_eval_indices_sha256": digest,
        "state_artifact": {
            "path": str(state_path),
            "sha256": sha256_file(state_path),
        },
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.time() - started,
    }
    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "classification": classification,
                "width_results": {
                    width: {
                        key: row[key]
                        for key in (
                            "fixed_validation_cross_entropy",
                            "gap",
                            "optimization_healthy",
                            "passes",
                            "summary",
                        )
                    }
                    for width, row in width_results.items()
                },
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
