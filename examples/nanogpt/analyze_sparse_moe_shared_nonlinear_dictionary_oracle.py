#!/usr/bin/env python3
"""Gate learned nonlinear sharing frontiers for complete sparse-MoE MLPs."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    action_cosine,
    collect_protocol_inputs,
    route_and_sample,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
    normalized_expert_loss,
    rademacher,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_shared_nonlinear_dictionary_oracle_plan_v1"
FAMILY_ORDER = (
    "global_shared_rank619",
    "layer_shared_rank60",
    "expert_local_rank7",
)


def coordinate_count(
    *, family: str, rank: int, tensor_layers: int, experts: int,
    input_width: int,
) -> int:
    """Count every learned direction and modulation value in one full model."""
    if family.startswith("global_shared"):
        basis = 2 * rank * input_width
    elif family.startswith("layer_shared"):
        basis = 2 * tensor_layers * rank * input_width
    elif family.startswith("expert_local"):
        basis = 2 * tensor_layers * experts * rank * input_width
    else:
        raise ValueError(f"unknown sharing family: {family}")
    modulation = 3 * tensor_layers * experts * rank
    return basis + modulation


def gelu_derivative(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0))) + (
        values * torch.exp(-0.5 * values.square()) / math.sqrt(2.0 * math.pi)
    )


def result_authorization(passed: bool) -> dict[str, bool]:
    return {
        "production_implementation": bool(passed),
        "initialization_and_mapping_loss_shadow": bool(passed),
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
        "full_attention_work": False,
        "automatic_retry_or_sweep": False,
    }


class SharedNonlinearDictionary(torch.nn.Module):
    """A counted learned dictionary that is the complete expert function."""

    def __init__(
        self,
        *,
        family: str,
        rank: int,
        tensor_layers: int,
        experts: int,
        input_width: int,
        seed: int,
        device: str,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.rank = int(rank)
        self.tensor_layers = int(tensor_layers)
        self.experts = int(experts)
        self.input_width = int(input_width)
        if min(self.rank, self.tensor_layers, self.experts, self.input_width) <= 0:
            raise ValueError("all nonlinear-dictionary dimensions must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            if self.family.startswith("global_shared"):
                shape = (self.rank, self.input_width)
            elif self.family.startswith("layer_shared"):
                shape = (self.tensor_layers, self.rank, self.input_width)
            elif self.family.startswith("expert_local"):
                shape = (
                    self.tensor_layers, self.experts, self.rank,
                    self.input_width,
                )
            else:
                raise ValueError(f"unknown sharing family: {self.family}")
            self.feature_basis = torch.nn.Parameter(torch.randn(shape) * 0.02)
            self.write_basis = torch.nn.Parameter(
                torch.randn(shape) * (0.02 / math.sqrt(2.0 * self.tensor_layers))
            )
        modulation_shape = (self.tensor_layers, self.experts, self.rank)
        self.input_gain = torch.nn.Parameter(torch.ones(modulation_shape))
        self.hidden_bias = torch.nn.Parameter(torch.zeros(modulation_shape))
        self.output_gain = torch.nn.Parameter(torch.zeros(modulation_shape))
        self.to(device=device, dtype=torch.float32)

    @property
    def device(self) -> str:
        return str(self.feature_basis.device)

    def compact_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _bases(
        self, layer: int, expert: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= int(layer) < self.tensor_layers:
            raise IndexError("layer index out of range")
        if self.family.startswith("global_shared"):
            return self.feature_basis, self.write_basis
        if self.family.startswith("layer_shared"):
            return self.feature_basis[layer], self.write_basis[layer]
        if expert is None:
            return self.feature_basis[layer], self.write_basis[layer]
        return (
            self.feature_basis[layer, expert : expert + 1],
            self.write_basis[layer, expert : expert + 1],
        )

    def function_and_jvp(
        self,
        inputs: torch.Tensor,
        directions: torch.Tensor,
        *,
        layer: int,
        expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.shape != directions.shape:
            raise ValueError("input and direction shapes disagree")
        if inputs.shape[-1] != self.input_width:
            raise ValueError("input width disagrees with dictionary")
        if expert is None:
            if inputs.shape[0] != self.experts:
                raise ValueError("batched input must contain every expert")
            selection = slice(None)
        else:
            if not 0 <= int(expert) < self.experts or inputs.shape[0] != 1:
                raise ValueError("single-expert input or index is invalid")
            selection = slice(int(expert), int(expert) + 1)
        feature, write = self._bases(int(layer), expert)
        if feature.ndim == 2:
            pre = torch.einsum("esd,rd->esr", inputs.float(), feature)
            pre_jvp = torch.einsum("esd,rd->esr", directions.float(), feature)
        else:
            pre = torch.einsum("esd,erd->esr", inputs.float(), feature)
            pre_jvp = torch.einsum("esd,erd->esr", directions.float(), feature)
        input_gain = self.input_gain[layer, selection, None, :]
        bias = self.hidden_bias[layer, selection, None, :]
        output_gain = self.output_gain[layer, selection, None, :]
        pre = pre * input_gain + bias
        pre_jvp = pre_jvp * input_gain
        hidden = F.gelu(pre) * output_gain
        hidden_jvp = gelu_derivative(pre) * pre_jvp * output_gain
        if write.ndim == 2:
            output = torch.einsum("esr,rd->esd", hidden, write)
            output_jvp = torch.einsum("esr,rd->esd", hidden_jvp, write)
        else:
            output = torch.einsum("esr,erd->esd", hidden, write)
            output_jvp = torch.einsum("esr,erd->esd", hidden_jvp, write)
        return output, output_jvp


def make_module(
    plan: dict[str, Any], family: str, device: str,
) -> SharedNonlinearDictionary:
    source = plan["source"]
    spec = plan["families"][family]
    return SharedNonlinearDictionary(
        family=family,
        rank=int(spec["rank"]),
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        input_width=int(source["input_width"]),
        seed=int(spec["fixed_seed"]),
        device=device,
    )


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def fit_joint(
    module: SharedNonlinearDictionary,
    samples: dict[int, torch.Tensor],
    states: dict[int, LayerState],
    *,
    layers: list[int],
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    jvp_weight: float,
    probe_seed: int,
) -> dict[str, Any]:
    live: dict[int, torch.Tensor] = {}
    directions: dict[int, torch.Tensor] = {}
    targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        live[layer] = samples[layer].to(module.device, dtype=torch.float32)
        directions[layer] = rademacher(
            tuple(live[layer].shape), probe_seed + 17 * layer, module.device
        )
        with torch.no_grad():
            targets[layer] = dense_function_and_jvp(
                live[layer],
                directions[layer],
                states[layer].c_fc.to(module.device, dtype=torch.float32),
                states[layer].c_proj.to(
                    module.device, dtype=torch.float32
                ).transpose(1, 2),
            )
    parameters = list(module.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    losses: list[float] = []
    output_losses: list[float] = []
    jvp_losses: list[float] = []
    maximum_gradient = 0.0
    gradients_finite = True
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        layer_output_losses = []
        layer_jvp_losses = []
        for layer in layers:
            output, output_jvp = module.function_and_jvp(
                live[layer], directions[layer], layer=layer
            )
            target_output, target_jvp = targets[layer]
            layer_output_losses.append(normalized_expert_loss(output, target_output))
            layer_jvp_losses.append(normalized_expert_loss(output_jvp, target_jvp))
        output_loss = torch.stack(layer_output_losses).mean()
        jvp_loss = torch.stack(layer_jvp_losses).mean()
        loss = output_loss + float(jvp_weight) * jvp_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite nonlinear-dictionary objective")
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in parameters
        )
        if not gradients_finite:
            raise RuntimeError("missing or non-finite nonlinear-dictionary gradient")
        gradient = float(
            torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
        )
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
        output_losses.append(float(output_loss.detach()))
        jvp_losses.append(float(jvp_loss.detach()))
    return {
        "steps": int(steps),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "initial_output_loss": output_losses[0],
        "final_output_loss": output_losses[-1],
        "initial_jvp_loss": jvp_losses[0],
        "final_jvp_loss": jvp_losses[-1],
        "maximum_preclip_gradient_norm": maximum_gradient,
        "all_gradients_finite": gradients_finite,
        "feature_basis_rms": float(module.feature_basis.detach().square().mean().sqrt()),
        "write_basis_rms": float(module.write_basis.detach().square().mean().sqrt()),
        "input_gain_rms": float(module.input_gain.detach().square().mean().sqrt()),
        "hidden_bias_rms": float(module.hidden_bias.detach().square().mean().sqrt()),
        "output_gain_rms": float(module.output_gain.detach().square().mean().sqrt()),
    }


@torch.no_grad()
def routed_evaluation(
    state: LayerState,
    activations: torch.Tensor,
    module: SharedNonlinearDictionary,
    *,
    layer: int,
    outer_top_k: int,
    probe_seed: int,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    state = state.to(module.device)
    all_directions = rademacher(tuple(activations.shape), probe_seed, "cpu")
    predicted_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    predicted_jvp_chunks: list[torch.Tensor] = []
    target_jvp_chunks: list[torch.Tensor] = []
    expert_error = torch.zeros(module.experts, dtype=torch.float64)
    expert_energy = torch.zeros(module.experts, dtype=torch.float64)
    for start in range(0, activations.shape[0], int(chunk_size)):
        stop = min(activations.shape[0], start + int(chunk_size))
        x = activations[start:stop].to(module.device, dtype=torch.float32)
        direction = all_directions[start:stop].to(module.device)
        logits = x @ state.router.T
        tie = torch.arange(logits.shape[-1], device=x.device, dtype=x.dtype)
        selected = torch.topk(
            logits - tie * torch.finfo(x.dtype).eps,
            int(outer_top_k), dim=-1, largest=True, sorted=True,
        ).indices
        probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
        predicted = torch.zeros_like(x)
        target = torch.zeros_like(x)
        predicted_jvp = torch.zeros_like(x)
        target_jvp = torch.zeros_like(x)
        for expert in range(module.experts):
            locations = (selected == expert).nonzero(as_tuple=False)
            if not locations.numel():
                continue
            token, slot = locations[:, 0], locations[:, 1]
            expert_input = x.index_select(0, token)[None]
            expert_direction = direction.index_select(0, token)[None]
            output, output_jvp = module.function_and_jvp(
                expert_input, expert_direction, layer=layer, expert=expert
            )
            target_output, dense_jvp = dense_function_and_jvp(
                expert_input,
                expert_direction,
                state.c_fc[expert : expert + 1],
                state.c_proj[expert : expert + 1].transpose(1, 2),
            )
            weight = probabilities[token, slot, None]
            predicted.index_add_(0, token, output[0] * weight)
            target.index_add_(0, token, target_output[0] * weight)
            predicted_jvp.index_add_(0, token, output_jvp[0] * weight)
            target_jvp.index_add_(0, token, dense_jvp[0] * weight)
            expert_error[expert] += float((output - target_output).square().sum())
            expert_energy[expert] += float(target_output.square().sum())
        predicted_chunks.append(predicted.cpu())
        target_chunks.append(target.cpu())
        predicted_jvp_chunks.append(predicted_jvp.cpu())
        target_jvp_chunks.append(target_jvp.cpu())
    predicted = torch.cat(predicted_chunks)
    target = torch.cat(target_chunks)
    predicted_jvp = torch.cat(predicted_jvp_chunks)
    target_jvp = torch.cat(target_jvp_chunks)
    return {
        "predicted": predicted,
        "target": target,
        "predicted_jvp": predicted_jvp,
        "target_jvp": target_jvp,
        "mixture_recovery": recovery_fraction(predicted, target),
        "jvp_recovery": recovery_fraction(predicted_jvp, target_jvp),
        "expert_recovery": [
            1.0 - float(error / max(energy, 1e-30))
            for error, energy in zip(expert_error, expert_energy)
        ],
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("shared nonlinear dictionary plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed")
    helpers = identity.get("helper_sha256")
    if not isinstance(helpers, dict) or not helpers:
        raise ValueError("helper hashes are not sealed")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in helpers.items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    source = plan["source"]
    for family in FAMILY_ORDER:
        spec = plan["families"][family]
        expected = coordinate_count(
            family=family,
            rank=int(spec["rank"]),
            tensor_layers=int(source["tensor_layers"]),
            experts=int(source["num_experts"]),
            input_width=int(source["input_width"]),
        )
        if expected != int(spec["total_coordinates_all_layers"]):
            raise ValueError(f"coordinate accounting drift: {family}")
        ratio = int(source["dense_paired_parameters_all_layers"]) / expected
        if not math.isclose(
            ratio, float(spec["paired_parameter_compression_ratio"]),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(f"compression accounting drift: {family}")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    layers = [int(value) for value in source["layers"]]
    generator = torch.Generator(device="cpu").manual_seed(20261211)
    samples: dict[int, torch.Tensor] = {}
    states: dict[int, LayerState] = {}
    for layer in layers:
        samples[layer] = torch.randn(
            int(source["num_experts"]), 8, int(source["input_width"]),
            generator=generator,
        ).contiguous()
        states[layer] = LayerState(
            router=torch.randn(
                int(source["num_experts"]), int(source["input_width"]),
                generator=generator,
            ).contiguous(),
            c_fc=(torch.randn(
                int(source["num_experts"]), int(source["expert_hidden_width"]),
                int(source["input_width"]), generator=generator,
            ) * 0.02).contiguous(),
            c_proj=(torch.randn(
                int(source["num_experts"]), int(source["input_width"]),
                int(source["expert_hidden_width"]), generator=generator,
            ) * (0.02 / math.sqrt(2.0 * int(source["tensor_layers"])))).contiguous(),
        )
    fit = plan["fit_protocol"]
    diagnostics: dict[str, Any] = {}
    initial_max = 0.0
    started = time.time()
    for family_index, family in enumerate(FAMILY_ORDER):
        module = make_module(plan, family, device)
        for layer in layers:
            live = samples[layer].to(device)
            direction = torch.randn_like(live)
            output, jvp = module.function_and_jvp(live, direction, layer=layer)
            initial_max = max(
                initial_max, float(output.abs().max()), float(jvp.abs().max())
            )
        diagnostics[family] = fit_joint(
            module, samples, states, layers=layers, steps=2,
            learning_rate=float(fit["learning_rate"]),
            weight_decay=float(fit["weight_decay"]),
            gradient_clip=float(fit["gradient_clip"]),
            jvp_weight=float(fit["jvp_weight"]),
            probe_seed=20261212 + 1009 * family_index,
        )
        if module.compact_parameter_count() != int(
            plan["families"][family]["total_coordinates_all_layers"]
        ):
            raise RuntimeError(f"live coordinate count drift: {family}")
        del module
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_shared_nonlinear_dictionary_preflight_v1",
        "device": device,
        "two_step_wall_seconds_all_three_families_one_bank": elapsed,
        "projected_full_protocol_seconds": elapsed * (int(fit["steps"]) / 2.0) * 2.0,
        "step_zero_output_and_jvp_max_abs": initial_max,
        "exact_step_zero_pass": initial_max == 0.0,
        "real_target_tensors_contiguous": all(
            state.c_fc.is_contiguous() and state.c_proj.is_contiguous()
            for state in states.values()
        ),
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
        "all_values_and_gradients_finite": all_finite(diagnostics),
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, args.plan)
    torch.set_float32_matmul_precision("high")
    if args.preflight_only:
        print(json.dumps(run_preflight(plan, args.device), indent=2, sort_keys=True))
        return
    if args.terminal_snapshot is None or args.data_dir is None or args.output is None:
        parser.error("oracle requires --terminal-snapshot, --data-dir, and --output")

    started = time.time()
    source = plan["source"]
    if file_sha256(args.terminal_snapshot) != source["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash drift")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != source["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash drift")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal snapshot step drift")
    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, plan, args.data_dir, args.device)
    mapping = dict(model.named_parameters())
    layers = [int(value) for value in source["layers"]]
    states = {layer: layer_state_from_mapping(mapping, layer) for layer in layers}
    del mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    banks = [row["name"] for row in plan["data_protocol"]["discovery_banks"]]
    samples: dict[str, dict[int, torch.Tensor]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    sample_count = int(plan["data_protocol"]["fit_samples_per_expert"])
    for bank_index, bank in enumerate(banks):
        samples[bank], occupancy[bank] = {}, {}
        for layer in layers:
            sampled, counts = route_and_sample(
                states[layer], inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=sample_count,
                seed=20261221 + 1009 * bank_index + 17 * layer,
            )
            samples[bank][layer] = sampled
            occupancy[bank][str(layer)] = counts

    fit = plan["fit_protocol"]
    summaries: dict[str, dict[str, Any]] = {family: {} for family in FAMILY_ORDER}
    diagnostics: dict[str, dict[str, Any]] = {family: {} for family in FAMILY_ORDER}
    saved: dict[str, dict[str, Any]] = {family: {} for family in FAMILY_ORDER}
    actions: dict[tuple[str, str, int], torch.Tensor] = {}
    for family_index, family in enumerate(FAMILY_ORDER):
        for bank_index, bank in enumerate(banks):
            module = make_module(plan, family, args.device)
            diagnostics[family][bank] = fit_joint(
                module, samples[bank], states, layers=layers,
                steps=int(fit["steps"]),
                learning_rate=float(fit["learning_rate"]),
                weight_decay=float(fit["weight_decay"]),
                gradient_clip=float(fit["gradient_clip"]),
                jvp_weight=float(fit["jvp_weight"]),
                probe_seed=20261222 + 1009 * bank_index + 10007 * family_index,
            )
            rows: dict[str, Any] = {}
            for layer in layers:
                evaluation = routed_evaluation(
                    states[layer], inputs["heldout"][layer], module,
                    layer=layer, outer_top_k=int(source["outer_moe_top_k"]),
                    probe_seed=20261223 + 17 * layer,
                )
                actions[(family, bank, layer)] = evaluation["predicted"]
                rows[str(layer)] = {
                    "mixture_recovery": evaluation["mixture_recovery"],
                    "jvp_recovery": evaluation["jvp_recovery"],
                    "minimum_expert_recovery": min(evaluation["expert_recovery"]),
                    "expert_recovery": evaluation["expert_recovery"],
                }
            static = float(plan["frozen_gates"][
                f"sealed_static_ceiling_recovery_{'a' if bank_index == 0 else 'b'}"
            ])
            aggregates = {
                "mixture_recovery_mean": sum(
                    float(row["mixture_recovery"]) for row in rows.values()
                ) / len(rows),
                "mixture_recovery_minimum_layer": min(
                    float(row["mixture_recovery"]) for row in rows.values()
                ),
                "jvp_recovery_mean": sum(
                    float(row["jvp_recovery"]) for row in rows.values()
                ) / len(rows),
                "minimum_expert_recovery": min(
                    float(row["minimum_expert_recovery"]) for row in rows.values()
                ),
                "minimum_discovery_assignments": min(
                    min(occupancy[bank][str(layer)]) for layer in layers
                ),
            }
            aggregates["minus_sealed_static_ceiling_recovery"] = (
                aggregates["mixture_recovery_mean"] - static
            )
            rows["aggregate"] = aggregates
            summaries[family][bank] = rows
            saved[family][bank] = cpu_state_dict(module)
            del module
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    frozen = plan["frozen_gates"]
    agreements: dict[str, Any] = {}
    gates: dict[str, dict[str, Any]] = {}
    passed_families: list[str] = []
    for family in FAMILY_ORDER:
        by_layer = {
            str(layer): action_cosine(
                actions[(family, banks[0], layer)],
                actions[(family, banks[1], layer)],
            ) for layer in layers
        }
        agreement = sum(by_layer.values()) / len(by_layer)
        agreements[family] = {"mean": agreement, "by_layer": by_layer}
        gates[family] = {}
        for bank in banks:
            aggregate = summaries[family][bank]["aggregate"]
            bank_gates = {
                "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_bank"]),
                "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_bank"]),
                "every_expert_pass": aggregate["minimum_expert_recovery"] >= float(frozen["heldout_expert_recovery_min_each_bank"]),
                "jvp_pass": aggregate["jvp_recovery_mean"] >= float(frozen["heldout_jvp_recovery_mean_min_each_bank"]),
                "action_agreement_pass": agreement >= float(frozen["heldout_bank_action_cosine_mean_min"]),
                "occupancy_pass": aggregate["minimum_discovery_assignments"] >= int(frozen["minimum_discovery_assignments_per_expert"]),
                "compression_pass": float(plan["families"][family]["paired_parameter_compression_ratio"]) >= float(frozen["paired_parameter_compression_ratio_min"]),
            }
            bank_gates["finite_pass"] = all_finite({
                "summary": summaries[family][bank],
                "diagnostics": diagnostics[family][bank],
                "agreement": agreement,
            }) and bool(diagnostics[family][bank]["all_gradients_finite"])
            bank_gates["all_pass"] = all(bank_gates.values())
            gates[family][bank] = bank_gates
        gates[family]["all_pass"] = all(gates[family][bank]["all_pass"] for bank in banks)
        if gates[family]["all_pass"]:
            passed_families.append(family)

    selected_family = None
    if passed_families:
        selected_family = max(
            passed_families,
            key=lambda family: (
                min(
                    float(summaries[family][bank]["aggregate"]["mixture_recovery_mean"])
                    for bank in banks
                ),
                min(
                    float(summaries[family][bank]["aggregate"]["jvp_recovery_mean"])
                    for bank in banks
                ),
                -int(plan["families"][family]["total_coordinates_all_layers"]),
            ),
        )
    passed = selected_family is not None
    finite = all_finite({
        "summaries": summaries, "diagnostics": diagnostics,
        "agreements": agreements,
    })

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_shared_nonlinear_dictionary_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_shared_nonlinear_dictionary_oracle_result_v1",
        "classification": (
            "SHARED_NONLINEAR_DICTIONARY_FRONTIER_PASSES"
            if passed else "SHARED_NONLINEAR_DICTIONARY_FRONTIER_REJECTED"
        ),
        "passed": passed,
        "selected_family": selected_family,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(manifest),
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "coordinates_path": str(coordinates_path),
            "coordinates_sha256": file_sha256(coordinates_path),
            "maximum_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        },
        "accounting": {
            family: {
                "dense_paired_parameters": int(source["dense_paired_parameters_all_layers"]),
                "compact_coordinates": int(plan["families"][family]["total_coordinates_all_layers"]),
                "compression_ratio": float(plan["families"][family]["paired_parameter_compression_ratio"]),
                "dense_base_or_residual": False,
                "all_learned_directions_counted": True,
            } for family in FAMILY_ORDER
        },
        "occupancy": occupancy,
        "fit_diagnostics": diagnostics,
        "summaries": summaries,
        "heldout_bank_action_cosine": agreements,
        "gates": gates,
        "all_values_finite": finite,
        "authorization": result_authorization(passed),
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
