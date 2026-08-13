#!/usr/bin/env python3
"""Gate a multiplicative full-rank procedural c_fc normal field for sparse MoE."""
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
from examples.nanogpt.analyze_sparse_moe_shared_nonlinear_dictionary_oracle import (
    gelu_derivative,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)
from latent_weight_lab.block_fht import normalized_fht_last_dim, signs_for


PLAN_SCHEMA = "nanogpt_sparse_moe_fullrank_gated_normal_field_oracle_plan_v1"


def coordinate_count(*, tensor_layers: int, experts: int, hidden_width: int) -> int:
    return int(tensor_layers) * int(experts) * (3 * int(hidden_width) + 1)


class FullRankGatedNormalField(torch.nn.Module):
    """Two fixed full-rank reads with learned diagonal/bias modulation."""

    def __init__(
        self,
        *,
        dense_cproj: torch.Tensor,
        input_width: int,
        hidden_width: int,
        padded_width: int,
        tensor_layers: int,
        experts: int,
        seed: int,
        layer: int,
        device: str,
    ) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.padded_width = int(padded_width)
        self.tensor_layers = int(tensor_layers)
        self.experts = int(experts)
        if self.padded_width < max(self.input_width, self.hidden_width):
            raise ValueError("padded width does not cover both axes")
        if self.padded_width & (self.padded_width - 1):
            raise ValueError("padded width must be a power of two")
        if tuple(dense_cproj.shape) != (
            self.experts, self.input_width, self.hidden_width
        ):
            raise ValueError("dense c_proj isolation tensor shape mismatch")

        reference = torch.empty(1, device=device, dtype=torch.float32)
        signs = []
        layer_seed = int(seed) + 1009 * int(layer)
        for expert in range(self.experts):
            signs.append(torch.stack([
                signs_for(reference, expert, 0, layer_seed, self.padded_width),
                signs_for(reference, expert, 1, layer_seed, self.padded_width),
            ]))
        self.register_buffer("signs", torch.stack(signs).float(), persistent=False)
        self.register_buffer(
            "dense_cproj", dense_cproj.detach().float().contiguous(), persistent=False
        )
        self.gate_gain_delta = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width)
        )
        self.value_gain_delta = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width)
        )
        self.gate_bias = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width)
        )
        self.log_scale = torch.nn.Parameter(torch.zeros(self.experts))
        self.c_fc_scale = math.sqrt(float(self.padded_width)) * 0.02
        self.to(device=device, dtype=torch.float32)

    def counted_coordinates(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.parameters())

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
        multiplicative: bool,
        expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.shape != directions.shape or inputs.shape[-1] != self.input_width:
            raise ValueError("input/direction shape mismatch")
        selected = self._selection(expert)
        expected = self.experts if expert is None else 1
        if inputs.shape[0] != expected:
            raise ValueError("expert batch mismatch")
        values = F.pad(inputs.float(), (0, self.padded_width - self.input_width))
        tangent = F.pad(
            directions.float(), (0, self.padded_width - self.input_width)
        )
        signs = self.signs[selected]
        mapped = normalized_fht_last_dim(
            values[:, None, :, :] * signs[:, :, None, :]
        )[..., : self.hidden_width]
        mapped_jvp = normalized_fht_last_dim(
            tangent[:, None, :, :] * signs[:, :, None, :]
        )[..., : self.hidden_width]
        gate = self.c_fc_scale * mapped[:, 0]
        gate_jvp = self.c_fc_scale * mapped_jvp[:, 0]
        value = self.c_fc_scale * mapped[:, 1]
        value_jvp = self.c_fc_scale * mapped_jvp[:, 1]
        gate_gain = 1.0 + self.gate_gain_delta[selected, None, :]
        value_gain = 1.0 + self.value_gain_delta[selected, None, :]
        gate = gate * gate_gain + self.gate_bias[selected, None, :]
        gate_jvp = gate_jvp * gate_gain
        value = value * value_gain
        value_jvp = value_jvp * value_gain
        activated = F.gelu(gate)
        activated_jvp = gelu_derivative(gate) * gate_jvp
        scale = self.log_scale[selected].exp()[:, None, None]
        if multiplicative:
            hidden = scale * activated * (1.0 + value)
            hidden_jvp = scale * (
                activated_jvp * (1.0 + value) + activated * value_jvp
            )
        else:
            hidden = scale * (activated + value)
            hidden_jvp = scale * (activated_jvp + value_jvp)
        cproj = self.dense_cproj[selected]
        output = torch.einsum("esh,edh->esd", hidden, cproj)
        output_jvp = torch.einsum("esh,edh->esd", hidden_jvp, cproj)
        return output, output_jvp


def fit_field(
    module: FullRankGatedNormalField,
    inputs: torch.Tensor,
    dense_c_fc: torch.Tensor,
    dense_c_proj: torch.Tensor,
    *,
    multiplicative: bool,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    jvp_weight: float,
    probe_seed: int,
) -> dict[str, Any]:
    device = str(module.gate_bias.device)
    live_inputs = inputs.to(device=device, dtype=torch.float32)
    directions = rademacher(tuple(live_inputs.shape), probe_seed, device)
    with torch.no_grad():
        target, target_jvp = dense_function_and_jvp(
            live_inputs,
            directions,
            dense_c_fc.to(device=device, dtype=torch.float32),
            dense_c_proj.to(device=device, dtype=torch.float32).transpose(1, 2),
        )
    parameters = module.trainable_parameters()
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
            live_inputs, directions, multiplicative=multiplicative
        )
        output_loss = normalized_expert_loss(output, target)
        jvp_loss = normalized_expert_loss(output_jvp, target_jvp)
        loss = output_loss + float(jvp_weight) * jvp_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite gated-normal objective")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise RuntimeError("non-finite or missing gated-normal gradient")
        gradient = float(
            torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
        )
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
        output_losses.append(float(output_loss.detach()))
        jvp_losses.append(float(jvp_loss.detach()))
    return {
        "multiplicative": bool(multiplicative),
        "steps": int(steps),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "initial_output_loss": output_losses[0],
        "final_output_loss": output_losses[-1],
        "initial_jvp_loss": jvp_losses[0],
        "final_jvp_loss": jvp_losses[-1],
        "maximum_preclip_gradient_norm": maximum_gradient,
        "gate_gain_delta_rms": float(
            module.gate_gain_delta.detach().square().mean().sqrt()
        ),
        "value_gain_delta_rms": float(
            module.value_gain_delta.detach().square().mean().sqrt()
        ),
        "gate_bias_rms": float(module.gate_bias.detach().square().mean().sqrt()),
        "log_scale_rms": float(module.log_scale.detach().square().mean().sqrt()),
    }


@torch.no_grad()
def routed_evaluation(
    state: LayerState,
    activations: torch.Tensor,
    module: FullRankGatedNormalField,
    *,
    multiplicative: bool,
    outer_top_k: int,
    probe_seed: int,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    device = str(module.gate_bias.device)
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
        target_jvp = torch.zeros_like(x)
        for expert in range(module.experts):
            locations = (selected == expert).nonzero(as_tuple=False)
            if not locations.numel():
                continue
            token, slot = locations[:, 0], locations[:, 1]
            expert_input = x.index_select(0, token)[None]
            expert_direction = direction.index_select(0, token)[None]
            output, output_jvp = module.function_and_jvp(
                expert_input, expert_direction,
                multiplicative=multiplicative, expert=expert,
            )
            dense_output, dense_jvp = dense_function_and_jvp(
                expert_input,
                expert_direction,
                state.c_fc[expert : expert + 1],
                state.c_proj[expert : expert + 1].transpose(1, 2),
            )
            weight = probabilities[token, slot, None]
            predicted.index_add_(0, token, output[0] * weight)
            target.index_add_(0, token, dense_output[0] * weight)
            predicted_jvp.index_add_(0, token, output_jvp[0] * weight)
            target_jvp.index_add_(0, token, dense_jvp[0] * weight)
            expert_error[expert] += float((output - dense_output).square().sum())
            expert_energy[expert] += float(dense_output.square().sum())
            expert_jvp_error[expert] += float((output_jvp - dense_jvp).square().sum())
            expert_jvp_energy[expert] += float(dense_jvp.square().sum())
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
        "expert_jvp_recovery": [
            1.0 - float(error / max(energy, 1e-30))
            for error, energy in zip(expert_jvp_error, expert_jvp_energy)
        ],
    }


def make_module(
    plan: dict[str, Any], layer: int, dense_cproj: torch.Tensor, device: str
) -> FullRankGatedNormalField:
    source, candidate = plan["source"], plan["candidate"]
    return FullRankGatedNormalField(
        dense_cproj=dense_cproj,
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        padded_width=int(candidate["padded_width"]),
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        seed=int(candidate["procedural_seed"]),
        layer=int(layer),
        device=device,
    )


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("full-rank gated-normal plan schema mismatch")
    source, candidate = plan["source"], plan["candidate"]
    expected = coordinate_count(
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        hidden_width=int(source["expert_hidden_width"]),
    )
    if expected != int(candidate["total_coordinates_all_layers"]):
        raise ValueError("gated-normal coordinate accounting drift")
    ratio = int(candidate["dense_cfc_parameters_all_layers"]) / expected
    if not math.isclose(
        ratio, float(candidate["cfc_coordinate_compression_ratio"]), rel_tol=1e-12
    ):
        raise ValueError("gated-normal compression accounting drift")
    identity = plan["identity"]
    expected_entry = identity.get("entrypoint_sha256")
    if expected_entry and expected_entry != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash drift")
    root = Path(__file__).resolve().parents[2]
    for relative, expected_hash in identity.get("helper_sha256", {}).items():
        if file_sha256(root / relative) != expected_hash:
            raise ValueError(f"helper hash drift: {relative}")
    if not file_sha256(plan_path):
        raise AssertionError("unreachable empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    generator = torch.Generator(device="cpu").manual_seed(20261741)
    experts = int(source["num_experts"])
    hidden = int(source["expert_hidden_width"])
    width = int(source["input_width"])
    inputs = torch.randn((experts, 16, width), generator=generator)
    c_fc = torch.randn((experts, hidden, width), generator=generator) * 0.02
    c_proj = torch.randn((experts, width, hidden), generator=generator) * (
        0.02 / math.sqrt(2.0 * int(source["tensor_layers"]))
    )
    candidate = make_module(plan, 0, c_proj, device)
    control = make_module(plan, 0, c_proj, device)
    fit = plan["fit_protocol"]
    started = time.time()
    candidate_diag = fit_field(
        candidate, inputs, c_fc, c_proj, multiplicative=True, steps=2,
        learning_rate=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
        jvp_weight=float(fit["jvp_weight"]), probe_seed=20261742,
    )
    control_diag = fit_field(
        control, inputs, c_fc, c_proj, multiplicative=False, steps=2,
        learning_rate=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
        jvp_weight=float(fit["jvp_weight"]), probe_seed=20261742,
    )
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_fullrank_gated_normal_preflight_v1",
        "device": device,
        "two_step_wall_seconds_candidate_plus_control": elapsed,
        "projected_full_protocol_seconds": elapsed * int(fit["steps"]) * 3.0,
        "candidate_coordinate_count_per_layer": candidate.counted_coordinates(),
        "control_coordinate_count_per_layer": control.counted_coordinates(),
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
    protocol = plan["data_protocol"]
    banks = [row["name"] for row in protocol["discovery_banks"]]
    saved: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    actions: dict[tuple[str, int], torch.Tensor] = {}
    for bank_index, bank in enumerate(banks):
        saved[bank], summaries[bank], diagnostics[bank], occupancy[bank] = {}, {}, {}, {}
        for layer in layers:
            state = states[layer]
            sampled, counts = route_and_sample(
                state, inputs[bank][layer], top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=int(protocol["fit_samples_per_expert"]),
                seed=int(protocol["sample_selection_seed_base"]) + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            candidate = make_module(plan, layer, state.c_proj, args.device)
            control = make_module(plan, layer, state.c_proj, args.device)
            probe_seed = int(protocol["fit_jvp_probe_seed_base"]) + 1009 * bank_index + 17 * layer
            candidate_diag = fit_field(
                candidate, sampled, state.c_fc, state.c_proj,
                multiplicative=True, steps=int(fit["steps"]),
                learning_rate=float(fit["learning_rate"]),
                weight_decay=float(fit["weight_decay"]),
                gradient_clip=float(fit["gradient_clip"]),
                jvp_weight=float(fit["jvp_weight"]), probe_seed=probe_seed,
            )
            control_diag = fit_field(
                control, sampled, state.c_fc, state.c_proj,
                multiplicative=False, steps=int(fit["steps"]),
                learning_rate=float(fit["learning_rate"]),
                weight_decay=float(fit["weight_decay"]),
                gradient_clip=float(fit["gradient_clip"]),
                jvp_weight=float(fit["jvp_weight"]), probe_seed=probe_seed,
            )
            heldout_seed = int(protocol["heldout_jvp_probe_seed_base"]) + 17 * layer
            candidate_eval = routed_evaluation(
                state, inputs["heldout"][layer], candidate,
                multiplicative=True, outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=heldout_seed,
            )
            control_eval = routed_evaluation(
                state, inputs["heldout"][layer], control,
                multiplicative=False, outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=heldout_seed,
            )
            if not torch.equal(candidate_eval["target"], control_eval["target"]):
                raise RuntimeError("candidate/control target drift")
            actions[(bank, layer)] = candidate_eval["predicted"]
            summaries[bank][str(layer)] = {
                "mixture_recovery": candidate_eval["mixture_recovery"],
                "jvp_recovery": candidate_eval["jvp_recovery"],
                "minimum_expert_recovery": min(candidate_eval["expert_recovery"]),
                "minimum_expert_jvp_recovery": min(candidate_eval["expert_jvp_recovery"]),
                "additive_control_recovery": control_eval["mixture_recovery"],
                "additive_control_jvp_recovery": control_eval["jvp_recovery"],
                "candidate_minus_additive_control_recovery": (
                    candidate_eval["mixture_recovery"] - control_eval["mixture_recovery"]
                ),
                "candidate_minus_additive_control_jvp_recovery": (
                    candidate_eval["jvp_recovery"] - control_eval["jvp_recovery"]
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
            "candidate_minus_additive_control_recovery_mean": sum(float(row["candidate_minus_additive_control_recovery"]) for row in rows) / len(rows),
            "candidate_minus_additive_control_jvp_recovery_mean": sum(float(row["candidate_minus_additive_control_jvp_recovery"]) for row in rows) / len(rows),
            "minimum_discovery_assignments": min(min(occupancy[bank][str(layer)]) for layer in layers),
        }
        summaries[bank]["aggregate"] = aggregate
        bank_gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["candidate_heldout_output_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["candidate_heldout_output_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": aggregate["minimum_expert_recovery"] >= float(frozen["candidate_heldout_output_recovery_minimum_expert_each_bank"]),
            "jvp_pass": aggregate["jvp_recovery_mean"] >= float(frozen["candidate_heldout_jvp_recovery_mean_min_each_bank"]),
            "output_control_gain_pass": aggregate["candidate_minus_additive_control_recovery_mean"] >= float(frozen["candidate_minus_additive_control_output_recovery_mean_min_each_bank"]),
            "jvp_control_gain_pass": aggregate["candidate_minus_additive_control_jvp_recovery_mean"] >= float(frozen["candidate_minus_additive_control_jvp_recovery_mean_min_each_bank"]),
            "occupancy_pass": aggregate["minimum_discovery_assignments"] >= int(frozen["minimum_discovery_assignments_per_expert"]),
        }
    agreement = {
        str(layer): action_cosine(actions[(banks[0], layer)], actions[(banks[1], layer)])
        for layer in layers
    }
    agreement_mean = sum(agreement.values()) / len(agreement)
    finite = all_finite({"summaries": summaries, "diagnostics": diagnostics, "agreement": agreement})
    for bank in banks:
        bank_gates[bank]["action_agreement_pass"] = agreement_mean >= float(frozen["cross_bank_candidate_action_cosine_min"])
        bank_gates[bank]["finite_pass"] = finite
        bank_gates[bank]["all_pass"] = all(bank_gates[bank].values())
    passed = all(bank_gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({"schema_version": "nanogpt_sparse_moe_fullrank_gated_normal_coordinates_v1", "states": saved}, coordinates_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_fullrank_gated_normal_oracle_result_v1",
        "classification": "FULLRANK_GATED_NORMAL_FIELD_PASSES" if passed else "FULLRANK_GATED_NORMAL_FIELD_REJECTED",
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
            "dense_cfc_parameters": int(plan["candidate"]["dense_cfc_parameters_all_layers"]),
            "compact_coordinates": int(plan["candidate"]["total_coordinates_all_layers"]),
            "compression_ratio": float(plan["candidate"]["cfc_coordinate_compression_ratio"]),
            "full_input_rank_procedural_maps": True,
            "dense_cproj_isolation_exception": True,
            "fixed_full_matrix_storage": False,
        },
        "occupancy": occupancy,
        "fit_diagnostics": diagnostics,
        "summaries": summaries,
        "cross_bank_action_cosine_by_layer": agreement,
        "cross_bank_action_cosine_mean": agreement_mean,
        "gates": bank_gates,
        "all_values_finite": finite,
        "authorization": {
            "paired_full_mlp_theory": bool(passed),
            "language_model_training": False,
            "larger_rung": False,
            "full_attention_work": False,
            "automatic_retry_or_sweep": False,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
