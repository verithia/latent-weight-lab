#!/usr/bin/env python3
"""Fit four depth-shared full-width MLPs with layer-aware top-1 routing."""
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
    validate_core_configs,
)
from examples.nanogpt.analyze_shared_mlp_exact_family_teacher_fit import (
    atomic_json,
    collect_stratified_inputs,
    git_head,
    normalized_objective,
)
from examples.nanogpt.analyze_shared_trunk_private_ridge_teacher_fit import (
    build_data,
    jvp_metrics,
    output_metrics,
    passes,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_depth_shared_top1_mlp_teacher_fit_plan_v1"
RESULT_SCHEMA = "mai_depth_shared_top1_mlp_teacher_fit_result_v1"


class DepthSharedTop1MLP(nn.Module):
    """A small global pool of complete dense MLP fields with top-1 routing."""

    def __init__(
        self,
        *,
        expert_fc: Tensor,
        expert_proj: Tensor,
        pre_gain: Tensor,
        output_log_gain: Tensor,
        router_weight: Tensor,
        router_bias: Tensor,
    ) -> None:
        super().__init__()
        self.expert_fc = nn.Parameter(expert_fc.detach().float().clone())
        self.expert_proj = nn.Parameter(expert_proj.detach().float().clone())
        self.pre_gain = nn.Parameter(pre_gain.detach().float().clone())
        self.output_log_gain = nn.Parameter(
            output_log_gain.detach().float().clone()
        )
        self.router_weight = nn.Parameter(
            router_weight.detach().float().clone()
        )
        self.router_bias = nn.Parameter(router_bias.detach().float().clone())
        experts, hidden, width = self.expert_fc.shape
        layers = self.pre_gain.shape[0]
        if self.expert_proj.shape != (experts, width, hidden):
            raise ValueError("expert MLP matrix shapes do not pair")
        if self.pre_gain.shape != (layers, hidden):
            raise ValueError("pre_gain shape mismatch")
        if self.output_log_gain.shape != (layers, width):
            raise ValueError("output_log_gain shape mismatch")
        if self.router_weight.shape != (layers, experts, width):
            raise ValueError("router_weight shape mismatch")
        if self.router_bias.shape != (layers, experts):
            raise ValueError("router_bias shape mismatch")

    @property
    def layers(self) -> int:
        return int(self.pre_gain.shape[0])

    @property
    def experts(self) -> int:
        return int(self.expert_fc.shape[0])

    def set_trainable(self, *, router_only: bool) -> list[nn.Parameter]:
        for parameter in self.parameters():
            parameter.requires_grad_(not router_only)
        self.router_weight.requires_grad_(True)
        self.router_bias.requires_grad_(True)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _logits(self, values: Tensor, layer: int) -> Tensor:
        normalized = F.layer_norm(values.float(), (values.shape[-1],))
        return F.linear(
            normalized,
            self.router_weight[int(layer)],
            self.router_bias[int(layer)],
        )

    def _expert_outputs(self, values: Tensor, layer: int) -> Tensor:
        hidden = torch.einsum("...d,ehd->...eh", values, self.expert_fc)
        hidden = F.gelu(hidden * self.pre_gain[int(layer)])
        return torch.einsum("...eh,edh->...ed", hidden, self.expert_proj)

    @staticmethod
    def _route_weights(logits: Tensor, *, mode: str, temperature: float) -> Tensor:
        soft = torch.softmax(logits / float(temperature), dim=-1)
        if mode == "softmax":
            return soft
        if mode == "hard_top1":
            return F.one_hot(
                logits.argmax(dim=-1), num_classes=logits.shape[-1]
            ).to(logits.dtype)
        if mode == "hard_st":
            hard = F.one_hot(
                logits.argmax(dim=-1), num_classes=logits.shape[-1]
            ).to(logits.dtype)
            return hard + soft - soft.detach()
        raise ValueError(f"unknown routing mode: {mode}")

    def forward_selected(
        self,
        values: Tensor,
        layer_indices: Tensor,
        *,
        mode: str,
        temperature: float,
    ) -> Tensor:
        if values.ndim != 5 or values.shape[1] != layer_indices.numel():
            raise ValueError("unexpected selected-layer input shape")
        outputs = []
        for position, layer_tensor in enumerate(layer_indices):
            layer = int(layer_tensor)
            current = values[:, position]
            expert_outputs = self._expert_outputs(current, layer)
            weights = self._route_weights(
                self._logits(current, layer),
                mode=mode,
                temperature=temperature,
            )
            output = (expert_outputs * weights[..., None]).sum(dim=-2)
            output = output * self.output_log_gain[layer].exp()
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def forward_layer(self, layer: int, values: Tensor) -> Tensor:
        """Hard top-1 forward used by all terminal measurements."""

        layer = int(layer)
        logits = self._logits(values, layer)
        if torch.is_grad_enabled():
            weights = self._route_weights(
                logits, mode="hard_top1", temperature=1.0
            )
            output = (
                self._expert_outputs(values, layer) * weights[..., None]
            ).sum(dim=-2)
        else:
            flat = values.reshape(-1, values.shape[-1])
            assignment = logits.argmax(dim=-1).reshape(-1)
            flat_output = torch.empty_like(flat)
            for expert in range(self.experts):
                selected = assignment == expert
                if not bool(selected.any()):
                    continue
                hidden = F.linear(flat[selected], self.expert_fc[expert])
                hidden = F.gelu(hidden * self.pre_gain[layer])
                flat_output[selected] = F.linear(
                    hidden, self.expert_proj[expert]
                ).to(flat_output.dtype)
            output = flat_output.reshape(*values.shape[:-1], values.shape[-1])
        return output * self.output_log_gain[layer].exp()

    @torch.no_grad()
    def clamp_charts(self) -> None:
        self.pre_gain.clamp_(-4.0, 4.0)
        self.output_log_gain.clamp_(-4.0, 4.0)


class InstalledTop1MLP(nn.Module):
    """Layer view used only for fixed-checkpoint CE evaluation."""

    def __init__(self, family: DepthSharedTop1MLP, layer: int) -> None:
        super().__init__()
        self.family = family
        self.layer = int(layer)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    def forward(self, values: Tensor) -> Tensor:
        return self.family.forward_layer(self.layer, values)


def initial_family(checkpoint: nn.Module) -> DepthSharedTop1MLP:
    boundaries = tuple(
        int(value)
        for value in checkpoint.config.mlp_shared_dense_trunk_boundaries
    )
    if not checkpoint.config.mlp_shared_dense_trunk:
        raise ValueError("initialization checkpoint is not a shared trunk")
    if int(checkpoint.config.mlp_shared_dense_trunk_groups) != 4:
        raise ValueError("initialization checkpoint does not have four trunks")
    if boundaries != (2, 4, 8, 12):
        raise ValueError("unexpected four-trunk depth partition")
    roots = (0, 2, 4, 8)
    blocks = list(checkpoint.transformer.h)
    expert_fc = torch.stack([blocks[layer].mlp.c_fc.weight for layer in roots])
    expert_proj = torch.stack(
        [blocks[layer].mlp.c_proj.weight for layer in roots]
    )
    if any(
        blocks[layer].mlp.c_fc.bias is not None
        or blocks[layer].mlp.c_proj.bias is not None
        for layer in roots
    ):
        raise ValueError("registered expert family is bias free")
    pre_gain = torch.stack([block.mlp.pregelu_gain for block in blocks])
    output_log_gain = torch.stack(
        [
            block.mlp.residual_output_log_gain
            * block.mlp.residual_output_gain_scale
            for block in blocks
        ]
    )
    layers, hidden = pre_gain.shape
    width = expert_fc.shape[-1]
    router_weight = torch.zeros(
        layers, len(roots), width, device=expert_fc.device
    )
    router_bias = torch.zeros(layers, len(roots), device=expert_fc.device)
    assignment = (0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)
    for layer, expert in enumerate(assignment):
        router_bias[layer, expert] = 8.0
    return DepthSharedTop1MLP(
        expert_fc=expert_fc,
        expert_proj=expert_proj,
        pre_gain=pre_gain,
        output_log_gain=output_log_gain,
        router_weight=router_weight,
        router_bias=router_bias,
    )


def objective_parts(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    if prediction.shape[0] != 3 or target.shape[0] != 3:
        raise ValueError("value/direction objective expects clean/plus/minus")
    value = normalized_objective(prediction[0], target[0])
    direction = normalized_objective(
        prediction[1] - prediction[2], target[1] - target[2]
    )
    return {"value": value, "direction": direction, "total": value + direction}


@torch.no_grad()
def full_objective(
    module: DepthSharedTop1MLP,
    data: dict[str, Tensor],
    *,
    mode: str,
    temperature: float,
    chunk: int = 128,
) -> dict[str, float]:
    totals = {"value": 0.0, "direction": 0.0, "total": 0.0}
    rows = 0
    indices = torch.arange(module.layers, device=data["clean"].device)
    for start in range(0, data["variants"].shape[-2], int(chunk)):
        stop = min(start + int(chunk), data["variants"].shape[-2])
        prediction = module.forward_selected(
            data["variants"][..., start:stop, :],
            indices,
            mode=mode,
            temperature=temperature,
        )
        parts = objective_parts(
            prediction, data["targets"][..., start:stop, :]
        )
        for key in totals:
            totals[key] += float(parts[key]) * (stop - start)
        rows += stop - start
    return {key: value / max(rows, 1) for key, value in totals.items()}


def _temperature(stage: dict[str, Any], step: int) -> float:
    count = max(int(stage["steps"]) - 1, 1)
    fraction = min(max(int(step), 0), count) / count
    start = float(stage["temperature_start"])
    end = float(stage["temperature_end"])
    return start + fraction * (end - start)


def fit_stage(
    module: DepthSharedTop1MLP,
    *,
    fit: dict[str, Tensor],
    holdout: dict[str, Tensor],
    stage: dict[str, Any],
    rows_per_layer: int,
    layers_per_update: int,
    learning_rate: float,
    gradient_clip_norm: float,
    seed: int,
    steps_override: int | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    router_only = stage["name"] == "soft_router_only"
    parameters = module.set_trainable(router_only=router_only)
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    generator = torch.Generator(device=fit["clean"].device).manual_seed(int(seed))
    steps = int(stage["steps"] if steps_override is None else steps_override)
    mode = "hard_st" if stage["routing"].startswith("hard top-1") else "softmax"
    terminal_temperature = float(stage["temperature_end"])
    initial = full_objective(
        module, fit, mode=mode, temperature=terminal_temperature
    )
    best_loss, best_state = math.inf, None
    finite, started = True, time.time()
    last_parts: dict[str, Tensor] | None = None
    for step in range(steps):
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
        temperature = _temperature(stage, step)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = module.forward_selected(
                variants,
                layer_indices,
                mode=mode,
                temperature=temperature,
            )
            last_parts = objective_parts(prediction, targets)
        if not torch.isfinite(last_parts["total"]):
            finite = False
            break
        last_parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip_norm))
        optimizer.step()
        module.clamp_charts()
        checkpoint = (
            step == 0 or (step + 1) % 100 == 0 or step + 1 == steps
        )
        if checkpoint:
            evaluated = full_objective(
                module,
                fit,
                mode=mode,
                temperature=terminal_temperature,
            )
            if evaluated["total"] < best_loss:
                best_loss = evaluated["total"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
            print(
                json.dumps(
                    {
                        "stage": stage["name"],
                        "fit_step": step + 1,
                        "fit_steps": steps,
                        "temperature": temperature,
                        "minibatch_value": float(last_parts["value"].detach()),
                        "minibatch_direction": float(last_parts["direction"].detach()),
                        "evaluated_total": evaluated["total"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if best_state is not None:
        module.load_state_dict(best_state)
    final = (
        full_objective(module, fit, mode=mode, temperature=terminal_temperature)
        if finite
        else {"value": math.inf, "direction": math.inf, "total": math.inf}
    )
    holdout_value = (
        full_objective(
            module, holdout, mode="hard_top1", temperature=1.0
        )
        if finite
        else {"value": math.inf, "direction": math.inf, "total": math.inf}
    )
    state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    reduction = (
        1.0 - final["total"] / max(initial["total"], 1e-30)
        if finite
        else None
    )
    return {
        "finite": finite,
        "initial_fit_objective": initial,
        "final_fit_objective": final,
        "objective_reduction_fraction": reduction,
        "hard_holdout_objective": holdout_value,
        "wall_seconds": time.time() - started,
    }, state


def family_from_state(
    state: dict[str, Tensor], device: str
) -> DepthSharedTop1MLP:
    return DepthSharedTop1MLP(
        expert_fc=state["expert_fc"],
        expert_proj=state["expert_proj"],
        pre_gain=state["pre_gain"],
        output_log_gain=state["output_log_gain"],
        router_weight=state["router_weight"],
        router_bias=state["router_bias"],
    ).to(device)


@torch.no_grad()
def occupancy(
    module: DepthSharedTop1MLP, data: dict[str, Tensor]
) -> list[dict[str, Any]]:
    rows = []
    for layer in range(module.layers):
        for bank_index, bank in enumerate(("teacher", "candidate")):
            values = data["clean"][layer, bank_index]
            assignment = module._logits(values, layer).argmax(dim=-1)
            counts = torch.bincount(
                assignment.reshape(-1), minlength=module.experts
            )
            rows.append(
                {
                    "layer": layer,
                    "bank": bank,
                    "counts": [int(value) for value in counts.cpu()],
                    "fractions": [
                        float(value / max(int(assignment.numel()), 1))
                        for value in counts.cpu()
                    ],
                }
            )
    return rows


def evaluate_stage(
    module: DepthSharedTop1MLP,
    *,
    holdout: dict[str, Tensor],
    teacher: nn.Module,
    measurement: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    output_summary, output_rows = output_metrics(module, holdout)
    jvp_summary, jvp_rows = jvp_metrics(
        module,
        holdout,
        teacher=teacher,
        directions=int(measurement["input_jvp_directions"]),
        seed=int(measurement["input_jvp_seed"]),
        device=device,
    )
    return {
        "summary": {"output": output_summary, "input_jvp": jvp_summary},
        "holdout_output_rows": output_rows,
        "holdout_input_jvp_rows": jvp_rows,
        "hard_router_occupancy": occupancy(module, holdout),
    }


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
            "four_trunk_initialization_checkpoint",
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
        paths["four_trunk_initialization_checkpoint"], args.device
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
    fit = build_data(
        banks=collect(int(protocol["fit_token_seed"]), int(protocol["fit_batches"])),
        teacher=teacher,
        relative_rms=float(protocol["local_perturbation_relative_rms"]),
        seed=int(protocol["fit_token_seed"]),
        device=args.device,
    )
    holdout = build_data(
        banks=collect(
            int(protocol["holdout_token_seed"]), int(protocol["holdout_batches"])
        ),
        teacher=teacher,
        relative_rms=float(protocol["local_perturbation_relative_rms"]),
        seed=int(protocol["holdout_token_seed"]),
        device=args.device,
    )
    module = initial_family(initializer).to(args.device)
    if sum(parameter.numel() for parameter in module.parameters()) != int(
        plan["family"]["compact_parameters"]
    ):
        raise ValueError("compact parameter accounting mismatch")

    if args.preflight_only:
        stage = dict(protocol["stages"][-1])
        row, _ = fit_stage(
            module,
            fit=fit,
            holdout=holdout,
            stage=stage,
            rows_per_layer=int(protocol["row_batch_size_per_layer_per_bank"]),
            layers_per_update=int(protocol["layers_per_update"]),
            learning_rate=float(protocol["learning_rate"]),
            gradient_clip_norm=float(protocol["gradient_clip_norm"]),
            seed=int(protocol["fit_token_seed"]),
            steps_override=int(args.preflight_steps),
        )
        seconds = row["wall_seconds"] / max(int(args.preflight_steps), 1)
        print(
            json.dumps(
                {
                    "preflight": "complete",
                    "seconds_per_fit_step_including_preflight_evaluations": seconds,
                    "conservative_estimated_fit_seconds": seconds
                    * sum(int(stage["steps"]) for stage in protocol["stages"]),
                    "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    stage_results: dict[str, dict[str, Any]] = {}
    stage_states: dict[str, dict[str, Tensor]] = {}
    for stage_index, stage in enumerate(protocol["stages"]):
        row, state = fit_stage(
            module,
            fit=fit,
            holdout=holdout,
            stage=stage,
            rows_per_layer=int(protocol["row_batch_size_per_layer_per_bank"]),
            layers_per_update=int(protocol["layers_per_update"]),
            learning_rate=float(protocol["learning_rate"]),
            gradient_clip_norm=float(protocol["gradient_clip_norm"]),
            seed=int(protocol["fit_token_seed"]) + stage_index * 1000,
        )
        row.update(
            evaluate_stage(
                module,
                holdout=holdout,
                teacher=teacher,
                measurement=measurement,
                device=args.device,
            )
        )
        stage_results[stage["name"]] = row
        stage_states[stage["name"]] = state

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
    evaluated_names = ("soft_router_only", "hard_top1_joint")
    for name in evaluated_names:
        family = family_from_state(stage_states[name], args.device)
        splice = load_model(paths["dense_teacher_checkpoint"], args.device)
        for layer in range(family.layers):
            splice.transformer.h[layer].mlp = InstalledTop1MLP(family, layer)
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
        row = stage_results[name]
        reduction = row["objective_reduction_fraction"]
        healthy = bool(
            row["finite"]
            and (
                row["final_fit_objective"]["total"] <= 0.5
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

    router_pass = stage_results["soft_router_only"].get("passes", False)
    terminal = stage_results["hard_top1_joint"]
    if router_pass:
        classification = "ROUTER_ONLY_PASS"
    elif terminal.get("passes", False):
        classification = "JOINT_PASS"
    elif terminal.get("optimization_healthy", False):
        classification = "FOUR_EXPERT_TOP1_FAMILY_FAIL"
    else:
        classification = "OPTIMIZATION_INCONCLUSIVE"
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "fitted_states.pt"
    torch.save(
        {
            "schema_version": "mai_depth_shared_top1_mlp_state_v1",
            "stages": stage_states,
        },
        state_path,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identity,
        "compact_parameter_count": sum(
            parameter.numel() for parameter in module.parameters()
        ),
        "stage_results": stage_results,
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
                "evaluated_stages": {
                    name: {
                        "fixed_validation_cross_entropy": stage_results[name][
                            "fixed_validation_cross_entropy"
                        ],
                        "gap": stage_results[name]["gap"],
                        "optimization_healthy": stage_results[name][
                            "optimization_healthy"
                        ],
                        "passes": stage_results[name]["passes"],
                        "summary": stage_results[name]["summary"],
                    }
                    for name in evaluated_names
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
