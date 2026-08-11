#!/usr/bin/env python3
"""Gate learned layer-shared butterfly directions for sparse-MoE c_fc."""
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

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    CompactCFCState,
    SpectralCFC,
    action_cosine,
    collect_protocol_inputs,
    dense_targets,
    fit_compact_state,
    normalized_fit_loss,
    route_and_sample,
    routed_outputs as spectral_routed_outputs,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)
from latent_weight_lab.block_fht import signs_for


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_learned_butterfly_frame_oracle_plan_v1"


def butterfly_angle_count(width: int) -> int:
    if width <= 0 or width & (width - 1):
        raise ValueError("butterfly width must be a positive power of two")
    return (width // 2) * int(math.log2(width))


def butterfly_transform(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply a full orthogonal butterfly whose stages contain Givens angles."""
    width = values.shape[-1]
    stages = int(math.log2(width))
    if width <= 0 or 1 << stages != width:
        raise ValueError("butterfly value width must be a power of two")
    if tuple(angles.shape) != (stages, width // 2):
        raise ValueError("butterfly angle tensor shape mismatch")
    output = values
    leading = [1] * (values.ndim - 1)
    for stage in range(stages):
        half = 1 << stage
        block = 2 * half
        groups = width // block
        blocked = output.reshape(*output.shape[:-1], groups, block)
        left = blocked[..., :half]
        right = blocked[..., half:]
        theta = angles[stage].reshape(groups, half).reshape(
            *leading, groups, half
        )
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        output = torch.cat(
            (cosine * left - sine * right, sine * left + cosine * right),
            dim=-1,
        ).reshape_as(output)
    return output


def candidate_coordinate_count(
    *,
    layers: int,
    experts: int,
    input_padded_width: int,
    hidden_padded_width: int,
    hidden_width: int,
) -> int:
    frames = layers * (
        butterfly_angle_count(input_padded_width)
        + butterfly_angle_count(hidden_padded_width)
    )
    modulation = layers * experts * (input_padded_width + hidden_width)
    return frames + modulation


@dataclass
class ButterflyCFCState:
    input_angles: torch.Tensor
    hidden_angles: torch.Tensor
    spectrum: torch.Tensor
    bias: torch.Tensor

    def cpu(self) -> "ButterflyCFCState":
        return ButterflyCFCState(
            self.input_angles.detach().cpu(),
            self.hidden_angles.detach().cpu(),
            self.spectrum.detach().cpu(),
            self.bias.detach().cpu(),
        )

    def expert(self, index: int) -> "ButterflyCFCState":
        return ButterflyCFCState(
            self.input_angles,
            self.hidden_angles,
            self.spectrum[index : index + 1],
            self.bias[index : index + 1],
        )


class LearnedButterflyCFC:
    def __init__(
        self,
        *,
        experts: int,
        input_width: int,
        hidden_width: int,
        input_padded_width: int,
        hidden_padded_width: int,
        seed: int,
        layer: int,
        device: str,
    ) -> None:
        if input_padded_width < input_width:
            raise ValueError("input padding does not cover input width")
        if hidden_padded_width < hidden_width:
            raise ValueError("hidden padding does not cover hidden width")
        if hidden_padded_width < input_padded_width:
            raise ValueError("hidden frame cannot embed input frame")
        butterfly_angle_count(input_padded_width)
        butterfly_angle_count(hidden_padded_width)
        self.experts = int(experts)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.input_padded_width = int(input_padded_width)
        self.hidden_padded_width = int(hidden_padded_width)
        self.device = device
        reference = torch.empty(1, device=device, dtype=torch.float32)
        self.input_sign = signs_for(
            reference,
            int(layer),
            0,
            int(seed),
            self.input_padded_width,
        ).to(device=device, dtype=torch.float32)
        self.hidden_sign = signs_for(
            reference,
            int(layer),
            1,
            int(seed),
            self.hidden_padded_width,
        ).to(device=device, dtype=torch.float32)
        self.base_scale = math.sqrt(float(hidden_padded_width)) * 0.02

    @property
    def input_stages(self) -> int:
        return int(math.log2(self.input_padded_width))

    @property
    def hidden_stages(self) -> int:
        return int(math.log2(self.hidden_padded_width))

    def initial_state(self, *, requires_grad: bool) -> ButterflyCFCState:
        input_angles = torch.full(
            (self.input_stages, self.input_padded_width // 2),
            math.pi / 4,
            device=self.device,
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        hidden_angles = torch.full(
            (self.hidden_stages, self.hidden_padded_width // 2),
            math.pi / 4,
            device=self.device,
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        spectrum = torch.zeros(
            self.experts,
            self.input_padded_width,
            device=self.device,
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        bias = torch.zeros(
            self.experts,
            self.hidden_width,
            device=self.device,
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        return ButterflyCFCState(input_angles, hidden_angles, spectrum, bias)

    def preactivation(
        self, inputs: torch.Tensor, state: ButterflyCFCState
    ) -> torch.Tensor:
        if inputs.shape[0] != state.spectrum.shape[0]:
            raise ValueError("expert axis and state spectrum disagree")
        if inputs.shape[-1] != self.input_width:
            raise ValueError("input width and butterfly operator disagree")
        values = F.pad(
            inputs.to(device=self.device, dtype=torch.float32),
            (0, self.input_padded_width - self.input_width),
        )
        values = butterfly_transform(
            values * self.input_sign,
            state.input_angles.to(self.device),
        )
        values = self.base_scale * values * (
            1.0 + state.spectrum.to(self.device)[:, None, :]
        )
        values = F.pad(
            values,
            (0, self.hidden_padded_width - self.input_padded_width),
        )
        values = butterfly_transform(
            values * self.hidden_sign,
            state.hidden_angles.to(self.device),
        )
        return values[..., : self.hidden_width] + state.bias.to(self.device)[
            :, None, :
        ]

    def expert_output(
        self,
        inputs: torch.Tensor,
        c_proj: torch.Tensor,
        state: ButterflyCFCState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preactivation = self.preactivation(inputs, state)
        output = torch.bmm(
            F.gelu(preactivation),
            c_proj.to(device=self.device, dtype=torch.float32).transpose(1, 2),
        )
        return preactivation, output


def fit_butterfly_state(
    operator: LearnedButterflyCFC,
    inputs: torch.Tensor,
    c_fc: torch.Tensor,
    c_proj: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
) -> tuple[ButterflyCFCState, dict[str, Any]]:
    target_pre, target_output = dense_targets(inputs, c_fc, c_proj, operator.device)
    state = operator.initial_state(requires_grad=True)
    parameters = [
        state.input_angles,
        state.hidden_angles,
        state.spectrum,
        state.bias,
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    losses: list[float] = []
    maximum_preclip_gradient_norm = 0.0
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        predicted_pre, predicted_output = operator.expert_output(
            inputs, c_proj, state
        )
        loss = normalized_fit_loss(
            predicted_pre,
            predicted_output,
            target_pre,
            target_output,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite learned-butterfly objective")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise RuntimeError("non-finite or missing learned-butterfly gradient")
        gradient = float(
            torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
        )
        maximum_preclip_gradient_norm = max(
            maximum_preclip_gradient_norm, gradient
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    initial_angle = math.pi / 4
    return state.cpu(), {
        "steps": int(steps),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_preclip_gradient_norm": maximum_preclip_gradient_norm,
        "input_angle_mean_absolute_displacement": float(
            (state.input_angles.detach() - initial_angle).abs().mean()
        ),
        "hidden_angle_mean_absolute_displacement": float(
            (state.hidden_angles.detach() - initial_angle).abs().mean()
        ),
    }


def routed_outputs(
    state: LayerState,
    activations: torch.Tensor,
    operator: LearnedButterflyCFC,
    compact: ButterflyCFCState,
    *,
    top_k: int,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float]]:
    state = state.to(operator.device)
    predicted_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    expert_error = torch.zeros(operator.experts, dtype=torch.float64)
    expert_energy = torch.zeros(operator.experts, dtype=torch.float64)
    pre_error = torch.zeros(operator.experts, dtype=torch.float64)
    pre_energy = torch.zeros(operator.experts, dtype=torch.float64)
    for start in range(0, activations.shape[0], int(chunk_size)):
        x = activations[start : start + int(chunk_size)].to(
            device=operator.device, dtype=torch.float32
        )
        logits = x @ state.router.T
        tie = torch.arange(logits.shape[-1], device=x.device, dtype=x.dtype)
        selected = torch.topk(
            logits - tie * torch.finfo(x.dtype).eps,
            int(top_k),
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
        predicted = torch.zeros_like(x)
        target = torch.zeros_like(x)
        for expert in range(operator.experts):
            locations = (selected == expert).nonzero(as_tuple=False)
            if not locations.numel():
                continue
            token = locations[:, 0]
            slot = locations[:, 1]
            expert_input = x.index_select(0, token)[None, :, :]
            candidate_pre, candidate_output = operator.expert_output(
                expert_input,
                state.c_proj[expert : expert + 1],
                compact.expert(expert),
            )
            target_pre, target_output = dense_targets(
                expert_input,
                state.c_fc[expert : expert + 1],
                state.c_proj[expert : expert + 1],
                operator.device,
            )
            weight = probabilities[token, slot, None]
            predicted.index_add_(0, token, candidate_output[0] * weight)
            target.index_add_(0, token, target_output[0] * weight)
            expert_error[expert] += float(
                (candidate_output - target_output).square().sum()
            )
            expert_energy[expert] += float(target_output.square().sum())
            pre_error[expert] += float((candidate_pre - target_pre).square().sum())
            pre_energy[expert] += float(target_pre.square().sum())
        predicted_chunks.append(predicted.cpu())
        target_chunks.append(target.cpu())
    return (
        torch.cat(predicted_chunks),
        torch.cat(target_chunks),
        [
            1.0 - float(error / max(energy, 1e-30))
            for error, energy in zip(expert_error, expert_energy)
        ],
        [
            1.0 - float(error / max(energy, 1e-30))
            for error, energy in zip(pre_error, pre_energy)
        ],
    )


def result_authorization(passed: bool) -> dict[str, bool]:
    return {
        "implementation": bool(passed),
        "initialization_and_mapping_loss_shadow": bool(passed),
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
        "generated_cproj": False,
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("learned-butterfly c_fc plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the frozen plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    candidate = plan["candidate"]
    expected = candidate_coordinate_count(
        layers=int(plan["source"]["tensor_layers"]),
        experts=int(plan["source"]["num_experts"]),
        input_padded_width=int(candidate["input_padded_width"]),
        hidden_padded_width=int(candidate["hidden_padded_width"]),
        hidden_width=int(plan["source"]["expert_hidden_width"]),
    )
    if expected != int(candidate["total_coordinates"]):
        raise ValueError("learned-butterfly coordinate accounting drift")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    candidate = plan["candidate"]
    source = plan["source"]
    fit = plan["fit_protocol"]
    operator = LearnedButterflyCFC(
        experts=int(source["num_experts"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        input_padded_width=int(candidate["input_padded_width"]),
        hidden_padded_width=int(candidate["hidden_padded_width"]),
        seed=20261014,
        layer=0,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(20261015)
    inputs = torch.randn(
        operator.experts,
        128,
        operator.input_width,
        generator=generator,
        device=device,
    )
    c_fc = torch.randn(
        operator.experts,
        operator.hidden_width,
        operator.input_width,
        generator=generator,
        device=device,
    ) * 0.02
    c_proj = torch.randn(
        operator.experts,
        operator.input_width,
        operator.hidden_width,
        generator=generator,
        device=device,
    ) * 0.02
    started = time.time()
    _state, diagnostics = fit_butterfly_state(
        operator,
        inputs,
        c_fc,
        c_proj,
        steps=5,
        learning_rate=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
    )
    return {
        "schema_version": "nanogpt_sparse_moe_cfc_learned_butterfly_preflight_v1",
        "device": device,
        "five_step_wall_seconds": time.time() - started,
        "projected_400_step_seconds_one_layer_bank": (
            (time.time() - started) * 80
        ),
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else 0
        ),
        "all_values_finite": all_finite(diagnostics),
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
    if args.preflight_only:
        print(json.dumps(run_preflight(plan, args.device), indent=2, sort_keys=True))
        return
    if args.terminal_snapshot is None or args.data_dir is None or args.output is None:
        parser.error("scientific oracle requires --terminal-snapshot, --data-dir, and --output")
    started = time.time()
    source = plan["source"]
    if file_sha256(args.terminal_snapshot) != source["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash disagrees with frozen plan")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != source["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash disagrees with frozen plan")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal snapshot step disagrees with frozen plan")
    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, plan, args.data_dir, args.device)
    terminal_mapping = dict(model.named_parameters())
    layers = [int(value) for value in source["layers"]]
    states = {
        layer: layer_state_from_mapping(terminal_mapping, layer) for layer in layers
    }
    del terminal_mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    candidate = plan["candidate"]
    fit = plan["fit_protocol"]
    samples_per_expert = int(plan["data_protocol"]["fit_samples_per_expert"])
    banks = [row["name"] for row in plan["data_protocol"]["discovery_banks"]]
    candidate_states: dict[str, dict[str, ButterflyCFCState]] = {}
    control_states: dict[str, dict[str, CompactCFCState]] = {}
    fit_diagnostics: dict[str, dict[str, Any]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    heldout_actions: dict[tuple[str, int], torch.Tensor] = {}

    for bank_index, bank in enumerate(banks):
        candidate_states[bank] = {}
        control_states[bank] = {}
        fit_diagnostics[bank] = {}
        occupancy[bank] = {}
        summaries[bank] = {}
        for layer in layers:
            state = states[layer]
            sampled, counts = route_and_sample(
                state,
                inputs[bank][layer],
                top_k=int(source["num_experts"] // 4),
                samples_per_expert=samples_per_expert,
                seed=20261016 + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            operator = LearnedButterflyCFC(
                experts=int(source["num_experts"]),
                input_width=int(source["input_width"]),
                hidden_width=int(source["expert_hidden_width"]),
                input_padded_width=int(candidate["input_padded_width"]),
                hidden_padded_width=int(candidate["hidden_padded_width"]),
                seed=20261014,
                layer=layer,
                device=args.device,
            )
            compact, compact_diagnostics = fit_butterfly_state(
                operator,
                sampled,
                state.c_fc,
                state.c_proj,
                steps=int(fit["steps"]),
                learning_rate=float(fit["learning_rate"]),
                weight_decay=float(fit["weight_decay"]),
                gradient_clip=float(fit["gradient_clip"]),
            )
            control_operator = SpectralCFC(
                experts=int(source["num_experts"]),
                input_width=int(source["input_width"]),
                hidden_width=int(source["expert_hidden_width"]),
                padded_width=2048,
                seed=20260931,
                layer=layer,
                device=args.device,
            )
            control_compact, control_diagnostics = fit_compact_state(
                control_operator,
                sampled,
                state.c_fc,
                state.c_proj,
                spectral=True,
                steps=int(fit["steps"]),
                learning_rate=float(fit["learning_rate"]),
                weight_decay=float(fit["weight_decay"]),
            )
            candidate_states[bank][str(layer)] = compact
            control_states[bank][str(layer)] = control_compact
            fit_diagnostics[bank][str(layer)] = {
                "candidate": compact_diagnostics,
                "control": control_diagnostics,
            }
            predicted, target, expert_recovery, pre_recovery = routed_outputs(
                state,
                inputs["heldout"][layer],
                operator,
                compact,
                top_k=2,
            )
            control_predicted, control_target, _, _ = spectral_routed_outputs(
                state,
                inputs["heldout"][layer],
                control_operator,
                control_compact,
                spectral=True,
                top_k=2,
            )
            if not torch.equal(target, control_target):
                raise RuntimeError("candidate and control target drift")
            mixture_recovery = recovery_fraction(predicted, target)
            control_recovery = recovery_fraction(control_predicted, target)
            summaries[bank][str(layer)] = {
                "mixture_recovery": mixture_recovery,
                "control_mixture_recovery": control_recovery,
                "candidate_minus_control_recovery": (
                    mixture_recovery - control_recovery
                ),
                "expert_recovery": expert_recovery,
                "minimum_expert_recovery": min(expert_recovery),
                "pregelu_recovery": pre_recovery,
                "minimum_pregelu_recovery": min(pre_recovery),
            }
            heldout_actions[(bank, layer)] = predicted

    frozen = plan["frozen_gates"]
    bank_gates: dict[str, dict[str, bool]] = {}
    for bank in banks:
        rows = [summaries[bank][str(layer)] for layer in layers]
        mixture = [float(row["mixture_recovery"]) for row in rows]
        improvements = [
            float(row["candidate_minus_control_recovery"]) for row in rows
        ]
        minimum_expert = min(float(row["minimum_expert_recovery"]) for row in rows)
        minimum_occupancy = min(
            min(occupancy[bank][str(layer)]) for layer in layers
        )
        aggregate = {
            "mixture_recovery_mean": sum(mixture) / len(mixture),
            "mixture_recovery_minimum_layer": min(mixture),
            "minimum_expert_recovery": minimum_expert,
            "candidate_minus_control_recovery_mean": (
                sum(improvements) / len(improvements)
            ),
            "minimum_discovery_assignments": minimum_occupancy,
        }
        summaries[bank]["aggregate"] = aggregate
        bank_gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"]
            >= float(frozen["heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"]
            >= float(frozen["heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": minimum_expert
            >= float(frozen["heldout_expert_recovery_min_each_bank"]),
            "learned_frame_gain_pass": aggregate[
                "candidate_minus_control_recovery_mean"
            ]
            >= float(frozen["candidate_minus_control_recovery_mean_min_each_bank"]),
            "occupancy_pass": minimum_occupancy
            >= int(frozen["minimum_discovery_assignments_per_expert"]),
        }

    agreement_by_layer = {
        str(layer): action_cosine(
            heldout_actions[(banks[0], layer)],
            heldout_actions[(banks[1], layer)],
        )
        for layer in layers
    }
    agreement_mean = sum(agreement_by_layer.values()) / len(agreement_by_layer)
    agreement_pass = agreement_mean >= float(
        frozen["heldout_bank_action_cosine_mean_min"]
    )
    finite = all_finite(
        {
            "summaries": summaries,
            "fit_diagnostics": fit_diagnostics,
            "agreement_by_layer": agreement_by_layer,
        }
    )
    for bank in banks:
        bank_gates[bank]["action_agreement_pass"] = agreement_pass
        bank_gates[bank]["finite_pass"] = finite
        bank_gates[bank]["all_pass"] = all(bank_gates[bank].values())
    passed = all(bank_gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save(
        {
            "schema_version": "nanogpt_sparse_moe_cfc_learned_butterfly_coordinates_v1",
            "candidate": candidate_states,
            "control": control_states,
        },
        coordinates_path,
    )
    result = {
        "schema_version": "nanogpt_sparse_moe_cfc_learned_butterfly_frame_oracle_result_v1",
        "classification": (
            "LEARNED_BUTTERFLY_CFC_REPRESENTABILITY_PASSES"
            if passed
            else "LEARNED_BUTTERFLY_CFC_REPRESENTABILITY_REJECTED"
        ),
        "passed": passed,
        "identity": {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
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
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda")
                else 0
            ),
        },
        "accounting": {
            "dense_cfc_parameters": int(candidate["dense_cfc_parameters"]),
            "candidate_coordinates": int(candidate["total_coordinates"]),
            "candidate_compression_ratio": float(candidate["cfc_compression_ratio"]),
            "control_coordinates": int(plan["control"]["total_coordinates"]),
            "control_compression_ratio": float(plan["control"]["cfc_compression_ratio"]),
            "materialized_dense_cfc_in_candidate": False,
            "dense_cproj_retained_as_exception": True,
        },
        "occupancy": occupancy,
        "fit_diagnostics": fit_diagnostics,
        "summaries": summaries,
        "heldout_bank_action_cosine": {
            "mean": agreement_mean,
            "by_layer": agreement_by_layer,
        },
        "gates": bank_gates,
        "all_values_finite": finite,
        "authorization": result_authorization(passed),
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
