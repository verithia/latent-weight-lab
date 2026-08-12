#!/usr/bin/env python3
"""Gate activation-routed complete procedural MLP atoms for sparse MoE."""
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
from latent_weight_lab.block_fht import normalized_fht_last_dim, signs_for


PLAN_SCHEMA = "nanogpt_sparse_moe_conditional_complete_atom_oracle_plan_v1"


def coordinate_count(
    *, experts: int, atoms: int, input_width: int, hidden_width: int,
    router_feature_width: int, conditional: bool,
) -> int:
    count = (
        experts * atoms * hidden_width
        + experts * atoms * input_width
        + experts * hidden_width
        + experts * atoms
    )
    if conditional:
        count += experts * atoms * router_feature_width
    return count


def result_authorization(passed: bool) -> dict[str, bool]:
    return {
        "implementation": bool(passed),
        "initialization_and_mapping_loss_shadow": bool(passed),
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
        "full_attention_work": False,
    }


class ConditionalCompleteAtoms(torch.nn.Module):
    """Input-conditioned mixture of complete fixed-FHT nonlinear MLP atoms."""

    def __init__(
        self,
        *,
        experts: int,
        atoms: int,
        input_width: int,
        hidden_width: int,
        padded_width: int,
        router_feature_width: int,
        tensor_layers: int,
        seed: int,
        layer: int,
        device: str,
    ) -> None:
        super().__init__()
        self.experts = int(experts)
        self.atoms = int(atoms)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.padded_width = int(padded_width)
        self.router_feature_width = int(router_feature_width)
        if self.padded_width & (self.padded_width - 1):
            raise ValueError("padded width must be a power of two")
        if self.padded_width < max(self.input_width, self.hidden_width):
            raise ValueError("padded width does not cover both matrix axes")
        if not 1 <= self.router_feature_width <= self.padded_width:
            raise ValueError("router feature width is outside padded width")

        reference = torch.empty(1, device=device, dtype=torch.float32)
        layer_seed = int(seed) + 1009 * int(layer)
        input_signs = []
        output_signs = []
        for expert in range(self.experts):
            expert_inputs = []
            expert_outputs = []
            for atom in range(self.atoms):
                expert_inputs.append(
                    signs_for(reference, expert, 3 * atom, layer_seed, self.padded_width)
                )
                expert_outputs.append(
                    signs_for(
                        reference, expert, 3 * atom + 1, layer_seed,
                        self.padded_width,
                    )
                )
            input_signs.append(torch.stack(expert_inputs))
            output_signs.append(torch.stack(expert_outputs))
        router_signs = torch.stack(
            [
                signs_for(
                    reference, expert, 3 * self.atoms + 2, layer_seed,
                    self.padded_width,
                )
                for expert in range(self.experts)
            ]
        )
        self.register_buffer(
            "input_signs", torch.stack(input_signs).float(), persistent=True
        )
        self.register_buffer(
            "output_signs", torch.stack(output_signs).float(), persistent=True
        )
        self.register_buffer("router_signs", router_signs.float(), persistent=True)

        self.hidden_gain_delta = torch.nn.Parameter(
            torch.zeros(self.experts, self.atoms, self.hidden_width)
        )
        self.output_gain_delta = torch.nn.Parameter(
            torch.zeros(self.experts, self.atoms, self.input_width)
        )
        self.hidden_bias = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width)
        )
        self.router_weight = torch.nn.Parameter(
            torch.zeros(
                self.experts, self.atoms, self.router_feature_width
            )
        )
        self.router_bias = torch.nn.Parameter(
            torch.zeros(self.experts, self.atoms)
        )
        self.c_fc_scale = math.sqrt(float(self.padded_width)) * 0.02
        self.c_proj_scale = (
            math.sqrt(float(self.padded_width))
            * 0.02
            / math.sqrt(2.0 * float(tensor_layers))
        )
        self.router_feature_scale = math.sqrt(
            float(self.padded_width) / float(self.input_width)
        )
        self.to(device=device, dtype=torch.float32)

    def trainable_parameters(self, *, conditional: bool) -> list[torch.nn.Parameter]:
        parameters = [
            self.hidden_gain_delta,
            self.output_gain_delta,
            self.hidden_bias,
            self.router_bias,
        ]
        if conditional:
            parameters.append(self.router_weight)
        return parameters

    def compact_parameter_count(self, *, conditional: bool) -> int:
        return sum(
            parameter.numel()
            for parameter in self.trainable_parameters(conditional=conditional)
        )

    def _selection(self, expert: int | None) -> slice:
        if expert is None:
            return slice(None)
        if not 0 <= int(expert) < self.experts:
            raise IndexError("expert index out of range")
        return slice(int(expert), int(expert) + 1)

    def function_and_jvp(
        self,
        inputs: torch.Tensor,
        directions: torch.Tensor,
        *,
        conditional: bool,
        expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.shape != directions.shape:
            raise ValueError("input and direction shapes disagree")
        selected = self._selection(expert)
        expected_experts = self.experts if expert is None else 1
        if inputs.shape[0] != expected_experts or inputs.shape[-1] != self.input_width:
            raise ValueError("input shape disagrees with complete-atom operator")
        values = F.pad(inputs.float(), (0, self.padded_width - self.input_width))
        tangent = F.pad(
            directions.float(), (0, self.padded_width - self.input_width)
        )

        input_signs = self.input_signs[selected]
        pre = normalized_fht_last_dim(
            values[:, None, :, :] * input_signs[:, :, None, :]
        )[..., : self.hidden_width]
        pre_jvp = normalized_fht_last_dim(
            tangent[:, None, :, :] * input_signs[:, :, None, :]
        )[..., : self.hidden_width]
        hidden_gain = 1.0 + self.hidden_gain_delta[selected, :, None, :]
        pre = (
            self.c_fc_scale * pre * hidden_gain
            + self.hidden_bias[selected, None, None, :]
        )
        pre_jvp = self.c_fc_scale * pre_jvp * hidden_gain
        hidden = F.gelu(pre)
        hidden_jvp = (
            0.5 * (1.0 + torch.erf(pre / math.sqrt(2.0)))
            + pre * torch.exp(-0.5 * pre.square()) / math.sqrt(2.0 * math.pi)
        ) * pre_jvp

        hidden = F.pad(hidden, (0, self.padded_width - self.hidden_width))
        hidden_jvp = F.pad(
            hidden_jvp, (0, self.padded_width - self.hidden_width)
        )
        output_signs = self.output_signs[selected]
        atom_output = normalized_fht_last_dim(
            hidden * output_signs[:, :, None, :]
        )[..., : self.input_width]
        atom_jvp = normalized_fht_last_dim(
            hidden_jvp * output_signs[:, :, None, :]
        )[..., : self.input_width]
        output_gain = 1.0 + self.output_gain_delta[selected, :, None, :]
        atom_output = self.c_proj_scale * atom_output * output_gain
        atom_jvp = self.c_proj_scale * atom_jvp * output_gain

        router_base = normalized_fht_last_dim(
            values * self.router_signs[selected, None, :]
        )[..., : self.router_feature_width] * self.router_feature_scale
        router_jvp = normalized_fht_last_dim(
            tangent * self.router_signs[selected, None, :]
        )[..., : self.router_feature_width] * self.router_feature_scale
        if conditional:
            router_weight = self.router_weight[selected]
            logits = torch.einsum("enr,ear->ena", router_base, router_weight)
            logits_jvp = torch.einsum("enr,ear->ena", router_jvp, router_weight)
        else:
            logits = torch.zeros(
                inputs.shape[0], inputs.shape[1], self.atoms,
                device=inputs.device, dtype=inputs.dtype,
            )
            logits_jvp = torch.zeros_like(logits)
        logits = logits + self.router_bias[selected, None, :]
        mixture = F.softmax(logits, dim=-1)
        mixture_jvp = mixture * (
            logits_jvp - (mixture * logits_jvp).sum(dim=-1, keepdim=True)
        )
        atom_output = atom_output.permute(0, 2, 1, 3)
        atom_jvp = atom_jvp.permute(0, 2, 1, 3)
        output = (mixture[..., None] * atom_output).sum(dim=2)
        output_jvp = (
            mixture[..., None] * atom_jvp
            + mixture_jvp[..., None] * atom_output
        ).sum(dim=2)
        return output, output_jvp


def fit_atoms(
    module: ConditionalCompleteAtoms,
    inputs: torch.Tensor,
    dense_c_fc: torch.Tensor,
    dense_c_proj: torch.Tensor,
    *,
    conditional: bool,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    jvp_weight: float,
    probe_seed: int,
) -> dict[str, Any]:
    device = str(module.hidden_bias.device)
    live_inputs = inputs.to(device=device, dtype=torch.float32)
    directions = rademacher(tuple(live_inputs.shape), probe_seed, device)
    with torch.no_grad():
        target_output, target_jvp = dense_function_and_jvp(
            live_inputs,
            directions,
            dense_c_fc.to(device=device, dtype=torch.float32),
            dense_c_proj.to(device=device, dtype=torch.float32).transpose(1, 2),
        )
    parameters = module.trainable_parameters(conditional=conditional)
    optimizer = torch.optim.AdamW(
        parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    losses: list[float] = []
    output_losses: list[float] = []
    jvp_losses: list[float] = []
    maximum_gradient = 0.0
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        output, output_jvp = module.function_and_jvp(
            live_inputs, directions, conditional=conditional
        )
        output_loss = normalized_expert_loss(output, target_output)
        jvp_loss = normalized_expert_loss(output_jvp, target_jvp)
        loss = output_loss + float(jvp_weight) * jvp_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite complete-atom objective")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise RuntimeError("non-finite or missing complete-atom gradient")
        gradient = float(
            torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
        )
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
        output_losses.append(float(output_loss.detach()))
        jvp_losses.append(float(jvp_loss.detach()))
    with torch.no_grad():
        entropy = F.softmax(module.router_bias, dim=-1)
        entropy = -(entropy * entropy.clamp_min(1e-30).log()).sum(dim=-1)
    return {
        "conditional": bool(conditional),
        "steps": int(steps),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "initial_output_loss": output_losses[0],
        "final_output_loss": output_losses[-1],
        "initial_jvp_loss": jvp_losses[0],
        "final_jvp_loss": jvp_losses[-1],
        "maximum_preclip_gradient_norm": maximum_gradient,
        "hidden_gain_delta_rms": float(
            module.hidden_gain_delta.detach().square().mean().sqrt()
        ),
        "output_gain_delta_rms": float(
            module.output_gain_delta.detach().square().mean().sqrt()
        ),
        "router_weight_rms": float(
            module.router_weight.detach().square().mean().sqrt()
        ),
        "static_router_entropy_mean": float(entropy.mean()),
    }


@torch.no_grad()
def routed_evaluation(
    state: LayerState,
    activations: torch.Tensor,
    module: ConditionalCompleteAtoms,
    *,
    conditional: bool,
    outer_top_k: int,
    probe_seed: int,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    device = str(module.hidden_bias.device)
    state = state.to(device)
    all_directions = rademacher(tuple(activations.shape), probe_seed, "cpu")
    predicted_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    predicted_jvp_chunks: list[torch.Tensor] = []
    target_jvp_chunks: list[torch.Tensor] = []
    expert_error = torch.zeros(module.experts, dtype=torch.float64)
    expert_energy = torch.zeros(module.experts, dtype=torch.float64)
    expert_jvp_error = torch.zeros(module.experts, dtype=torch.float64)
    expert_jvp_energy = torch.zeros(module.experts, dtype=torch.float64)
    for start in range(0, activations.shape[0], int(chunk_size)):
        stop = min(activations.shape[0], start + int(chunk_size))
        x = activations[start:stop].to(device=device, dtype=torch.float32)
        direction = all_directions[start:stop].to(device=device)
        logits = x @ state.router.T
        tie = torch.arange(logits.shape[-1], device=device, dtype=x.dtype)
        selected = torch.topk(
            logits - tie * torch.finfo(x.dtype).eps,
            int(outer_top_k), dim=-1, largest=True, sorted=True,
        ).indices
        probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
        predicted = torch.zeros_like(x)
        target = torch.zeros_like(x)
        predicted_jvp = torch.zeros_like(x)
        target_jvp_accumulator = torch.zeros_like(x)
        for expert in range(module.experts):
            locations = (selected == expert).nonzero(as_tuple=False)
            if not locations.numel():
                continue
            token = locations[:, 0]
            slot = locations[:, 1]
            expert_input = x.index_select(0, token)[None]
            expert_direction = direction.index_select(0, token)[None]
            output, output_jvp = module.function_and_jvp(
                expert_input,
                expert_direction,
                conditional=conditional,
                expert=expert,
            )
            target_output, dense_target_jvp = dense_function_and_jvp(
                expert_input,
                expert_direction,
                state.c_fc[expert : expert + 1],
                state.c_proj[expert : expert + 1].transpose(1, 2),
            )
            weight = probabilities[token, slot, None]
            predicted.index_add_(0, token, output[0] * weight)
            target.index_add_(0, token, target_output[0] * weight)
            predicted_jvp.index_add_(0, token, output_jvp[0] * weight)
            target_jvp_accumulator.index_add_(
                0, token, dense_target_jvp[0] * weight
            )
            expert_error[expert] += float((output - target_output).square().sum())
            expert_energy[expert] += float(target_output.square().sum())
            expert_jvp_error[expert] += float(
                (output_jvp - dense_target_jvp).square().sum()
            )
            expert_jvp_energy[expert] += float(dense_target_jvp.square().sum())
        predicted_chunks.append(predicted.cpu())
        target_chunks.append(target.cpu())
        predicted_jvp_chunks.append(predicted_jvp.cpu())
        target_jvp_chunks.append(target_jvp_accumulator.cpu())
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
        "expert_jvp_recovery": [
            1.0 - float(error / max(energy, 1e-30))
            for error, energy in zip(expert_jvp_error, expert_jvp_energy)
        ],
    }


def make_module(plan: dict[str, Any], layer: int, device: str) -> ConditionalCompleteAtoms:
    source = plan["source"]
    candidate = plan["candidate"]
    return ConditionalCompleteAtoms(
        experts=int(source["num_experts"]),
        atoms=int(candidate["atom_count"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        padded_width=int(candidate["padded_width"]),
        router_feature_width=int(candidate["router_feature_width"]),
        tensor_layers=int(source["tensor_layers"]),
        seed=int(candidate["fixed_operator_seed"]),
        layer=int(layer),
        device=device,
    )


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("conditional complete-atom plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    for control in plan["sealed_controls"].values():
        if file_sha256(root / control["path"]) != control["sha256"]:
            raise ValueError(f"sealed control hash drift: {control['path']}")
    source = plan["source"]
    candidate = plan["candidate"]
    expected = coordinate_count(
        experts=int(source["num_experts"]),
        atoms=int(candidate["atom_count"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        router_feature_width=int(candidate["router_feature_width"]),
        conditional=True,
    )
    if expected != int(candidate["total_coordinates_per_layer"]):
        raise ValueError("conditional complete-atom accounting drift")
    control_expected = coordinate_count(
        experts=int(source["num_experts"]),
        atoms=int(candidate["atom_count"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        router_feature_width=int(candidate["router_feature_width"]),
        conditional=False,
    )
    if control_expected != int(plan["same_run_control"]["total_coordinates_per_layer"]):
        raise ValueError("static control accounting drift")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    module = make_module(plan, 0, device)
    control = make_module(plan, 0, device)
    generator = torch.Generator(device="cpu").manual_seed(20261132)
    shape = (int(source["num_experts"]), 16, int(source["input_width"]))
    inputs = torch.randn(shape, generator=generator)
    c_fc = torch.randn(
        int(source["num_experts"]), int(source["expert_hidden_width"]),
        int(source["input_width"]), generator=generator,
    ) * 0.02
    c_proj = torch.randn(
        int(source["num_experts"]), int(source["input_width"]),
        int(source["expert_hidden_width"]), generator=generator,
    ) * (0.02 / math.sqrt(2.0 * int(source["tensor_layers"])))
    fit = plan["fit_protocol"]
    started = time.time()
    candidate_diag = fit_atoms(
        module, inputs, c_fc, c_proj,
        conditional=True, steps=2,
        learning_rate=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
        jvp_weight=float(fit["jvp_weight"]), probe_seed=20261133,
    )
    control_diag = fit_atoms(
        control, inputs, c_fc, c_proj,
        conditional=False, steps=2,
        learning_rate=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
        jvp_weight=float(fit["jvp_weight"]), probe_seed=20261133,
    )
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_conditional_complete_atom_preflight_v1",
        "device": device,
        "two_step_wall_seconds_candidate_plus_control": elapsed,
        "projected_full_protocol_seconds": (
            elapsed * (int(fit["steps"]) / 2.0) * 6.0
        ),
        "candidate_coordinate_count": module.compact_parameter_count(conditional=True),
        "control_coordinate_count": control.compact_parameter_count(conditional=False),
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
        "all_values_finite": all_finite(
            {"candidate": candidate_diag, "control": control_diag}
        ),
        "candidate_diagnostics": candidate_diag,
        "control_diagnostics": control_diag,
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

    fit = plan["fit_protocol"]
    banks = [row["name"] for row in plan["data_protocol"]["discovery_banks"]]
    samples_per_expert = int(plan["data_protocol"]["fit_samples_per_expert"])
    saved: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    actions: dict[tuple[str, int], torch.Tensor] = {}
    for bank_index, bank in enumerate(banks):
        saved[bank] = {}
        summaries[bank] = {}
        diagnostics[bank] = {}
        occupancy[bank] = {}
        for layer in layers:
            state = states[layer]
            sampled, counts = route_and_sample(
                state, inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=samples_per_expert,
                seed=20261134 + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            candidate = make_module(plan, layer, args.device)
            control = make_module(plan, layer, args.device)
            candidate_diag = fit_atoms(
                candidate, sampled, state.c_fc, state.c_proj,
                conditional=True, steps=int(fit["steps"]),
                learning_rate=float(fit["learning_rate"]),
                weight_decay=float(fit["weight_decay"]),
                gradient_clip=float(fit["gradient_clip"]),
                jvp_weight=float(fit["jvp_weight"]),
                probe_seed=20261135 + 1009 * bank_index + 17 * layer,
            )
            control_diag = fit_atoms(
                control, sampled, state.c_fc, state.c_proj,
                conditional=False, steps=int(fit["steps"]),
                learning_rate=float(fit["learning_rate"]),
                weight_decay=float(fit["weight_decay"]),
                gradient_clip=float(fit["gradient_clip"]),
                jvp_weight=float(fit["jvp_weight"]),
                probe_seed=20261135 + 1009 * bank_index + 17 * layer,
            )
            candidate_eval = routed_evaluation(
                state, inputs["heldout"][layer], candidate,
                conditional=True, outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=20261136 + 17 * layer,
            )
            control_eval = routed_evaluation(
                state, inputs["heldout"][layer], control,
                conditional=False, outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=20261136 + 17 * layer,
            )
            if not torch.equal(candidate_eval["target"], control_eval["target"]):
                raise RuntimeError("candidate and control target drift")
            actions[(bank, layer)] = candidate_eval["predicted"]
            summaries[bank][str(layer)] = {
                "mixture_recovery": candidate_eval["mixture_recovery"],
                "jvp_recovery": candidate_eval["jvp_recovery"],
                "minimum_expert_recovery": min(candidate_eval["expert_recovery"]),
                "minimum_expert_jvp_recovery": min(candidate_eval["expert_jvp_recovery"]),
                "static_control_recovery": control_eval["mixture_recovery"],
                "candidate_minus_static_control_recovery": (
                    candidate_eval["mixture_recovery"]
                    - control_eval["mixture_recovery"]
                ),
            }
            diagnostics[bank][str(layer)] = {
                "candidate": candidate_diag, "control": control_diag
            }
            saved[bank][str(layer)] = {
                "candidate": cpu_state_dict(candidate),
                "control": cpu_state_dict(control),
            }
            del candidate, control
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    frozen = plan["frozen_gates"]
    bank_gates: dict[str, dict[str, bool]] = {}
    for bank in banks:
        rows = [summaries[bank][str(layer)] for layer in layers]
        aggregate = {
            "mixture_recovery_mean": sum(float(row["mixture_recovery"]) for row in rows) / len(rows),
            "mixture_recovery_minimum_layer": min(float(row["mixture_recovery"]) for row in rows),
            "jvp_recovery_mean": sum(float(row["jvp_recovery"]) for row in rows) / len(rows),
            "minimum_expert_recovery": min(float(row["minimum_expert_recovery"]) for row in rows),
            "candidate_minus_static_control_recovery_mean": sum(
                float(row["candidate_minus_static_control_recovery"]) for row in rows
            ) / len(rows),
            "minimum_discovery_assignments": min(
                min(occupancy[bank][str(layer)]) for layer in layers
            ),
        }
        summaries[bank]["aggregate"] = aggregate
        bank_gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": aggregate["minimum_expert_recovery"] >= float(frozen["heldout_expert_recovery_min_each_bank"]),
            "jvp_pass": aggregate["jvp_recovery_mean"] >= float(frozen["heldout_jvp_recovery_mean_min_each_bank"]),
            "static_control_gain_pass": aggregate["candidate_minus_static_control_recovery_mean"] >= float(frozen["candidate_minus_static_control_recovery_mean_min_each_bank"]),
            "occupancy_pass": aggregate["minimum_discovery_assignments"] >= int(frozen["minimum_discovery_assignments_per_expert"]),
        }
    agreement_by_layer = {
        str(layer): action_cosine(actions[(banks[0], layer)], actions[(banks[1], layer)])
        for layer in layers
    }
    agreement_mean = sum(agreement_by_layer.values()) / len(agreement_by_layer)
    finite = all_finite(
        {"summaries": summaries, "diagnostics": diagnostics, "agreement": agreement_by_layer}
    )
    for bank in banks:
        bank_gates[bank]["action_agreement_pass"] = agreement_mean >= float(frozen["heldout_bank_action_cosine_mean_min"])
        bank_gates[bank]["finite_pass"] = finite
        bank_gates[bank]["all_pass"] = all(bank_gates[bank].values())
    passed = all(bank_gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save(
        {
            "schema_version": "nanogpt_sparse_moe_conditional_complete_atom_coordinates_v1",
            "states": saved,
        },
        coordinates_path,
    )
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_conditional_complete_atom_oracle_result_v1",
        "classification": (
            "CONDITIONAL_COMPLETE_ATOM_REPRESENTABILITY_PASSES"
            if passed else "CONDITIONAL_COMPLETE_ATOM_REPRESENTABILITY_REJECTED"
        ),
        "passed": passed,
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
            "dense_paired_parameters": int(plan["candidate"]["dense_paired_parameters_all_layers"]),
            "compact_coordinates": int(plan["candidate"]["total_coordinates_all_layers"]),
            "compression_ratio": float(plan["candidate"]["paired_parameter_compression_ratio"]),
            "materialized_dense_cfc": False,
            "materialized_dense_cproj": False,
            "fixed_full_matrix_storage": False,
            "conditional_complete_atoms": True,
        },
        "occupancy": occupancy,
        "fit_diagnostics": diagnostics,
        "summaries": summaries,
        "heldout_bank_action_cosine": {"mean": agreement_mean, "by_layer": agreement_by_layer},
        "gates": bank_gates,
        "all_values_finite": finite,
        "authorization": result_authorization(passed),
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
