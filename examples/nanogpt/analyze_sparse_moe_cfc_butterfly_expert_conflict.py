#!/usr/bin/env python3
"""Diagnose expert conflict versus topology capacity in butterfly c_fc maps."""
from __future__ import annotations

import argparse
import itertools
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
from examples.nanogpt.analyze_sparse_moe_cfc_learned_butterfly_frame_oracle import (
    ButterflyCFCState,
    LearnedButterflyCFC,
    result_authorization as parent_authorization,
    routed_outputs as shared_routed_outputs,
)
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    action_cosine,
    collect_protocol_inputs,
    dense_targets,
    normalized_fit_loss,
    route_and_sample,
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


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_butterfly_expert_conflict_audit_plan_v1"


def batched_butterfly_transform(
    values: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    """Apply independent orthogonal butterfly angles along the expert axis."""
    if values.ndim != 3:
        raise ValueError("batched butterfly values must be [expert, sample, width]")
    experts, _samples, width = values.shape
    stages = int(math.log2(width))
    if width <= 0 or 1 << stages != width:
        raise ValueError("batched butterfly width must be a power of two")
    if tuple(angles.shape) != (experts, stages, width // 2):
        raise ValueError("batched butterfly angle shape mismatch")
    output = values
    for stage in range(stages):
        half = 1 << stage
        block = 2 * half
        groups = width // block
        blocked = output.reshape(experts, output.shape[1], groups, block)
        left = blocked[..., :half]
        right = blocked[..., half:]
        theta = angles[:, stage].reshape(experts, 1, groups, half)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        output = torch.cat(
            (cosine * left - sine * right, sine * left + cosine * right),
            dim=-1,
        ).reshape_as(output)
    return output


@dataclass
class UnsharedButterflyState:
    input_angles: torch.Tensor
    hidden_angles: torch.Tensor
    spectrum: torch.Tensor
    bias: torch.Tensor

    @classmethod
    def from_shared(
        cls,
        shared: ButterflyCFCState,
        experts: int,
        device: str,
        *,
        requires_grad: bool,
    ) -> "UnsharedButterflyState":
        return cls(
            shared.input_angles.to(device).unsqueeze(0).repeat(experts, 1, 1)
            .detach().requires_grad_(requires_grad),
            shared.hidden_angles.to(device).unsqueeze(0).repeat(experts, 1, 1)
            .detach().requires_grad_(requires_grad),
            shared.spectrum.to(device).detach().clone().requires_grad_(requires_grad),
            shared.bias.to(device).detach().clone().requires_grad_(requires_grad),
        )

    def cpu(self) -> "UnsharedButterflyState":
        return UnsharedButterflyState(
            self.input_angles.detach().cpu(),
            self.hidden_angles.detach().cpu(),
            self.spectrum.detach().cpu(),
            self.bias.detach().cpu(),
        )

    def expert(self, index: int) -> "UnsharedButterflyState":
        return UnsharedButterflyState(
            self.input_angles[index : index + 1],
            self.hidden_angles[index : index + 1],
            self.spectrum[index : index + 1],
            self.bias[index : index + 1],
        )

    def all_finite(self) -> bool:
        return all(
            bool(torch.isfinite(value).all())
            for value in (
                self.input_angles,
                self.hidden_angles,
                self.spectrum,
                self.bias,
            )
        )


def unshared_preactivation(
    operator: LearnedButterflyCFC,
    inputs: torch.Tensor,
    state: UnsharedButterflyState,
) -> torch.Tensor:
    if inputs.shape[0] != state.input_angles.shape[0]:
        raise ValueError("expert axes disagree in unshared butterfly state")
    values = F.pad(
        inputs.to(device=operator.device, dtype=torch.float32),
        (0, operator.input_padded_width - operator.input_width),
    )
    values = batched_butterfly_transform(
        values * operator.input_sign,
        state.input_angles.to(operator.device),
    )
    values = operator.base_scale * values * (
        1.0 + state.spectrum.to(operator.device)[:, None, :]
    )
    values = F.pad(
        values,
        (0, operator.hidden_padded_width - operator.input_padded_width),
    )
    values = batched_butterfly_transform(
        values * operator.hidden_sign,
        state.hidden_angles.to(operator.device),
    )
    return values[..., : operator.hidden_width] + state.bias.to(operator.device)[
        :, None, :
    ]


def unshared_output(
    operator: LearnedButterflyCFC,
    inputs: torch.Tensor,
    c_proj: torch.Tensor,
    state: UnsharedButterflyState,
) -> tuple[torch.Tensor, torch.Tensor]:
    preactivation = unshared_preactivation(operator, inputs, state)
    output = torch.bmm(
        F.gelu(preactivation),
        c_proj.to(device=operator.device, dtype=torch.float32).transpose(1, 2),
    )
    return preactivation, output


def fit_unshared_state(
    operator: LearnedButterflyCFC,
    shared: ButterflyCFCState,
    inputs: torch.Tensor,
    c_fc: torch.Tensor,
    c_proj: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
) -> tuple[UnsharedButterflyState, dict[str, Any]]:
    target_pre, target_output = dense_targets(inputs, c_fc, c_proj, operator.device)
    state = UnsharedButterflyState.from_shared(
        shared,
        operator.experts,
        operator.device,
        requires_grad=True,
    )
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
    maximum_gradient = 0.0
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        predicted_pre, predicted_output = unshared_output(
            operator, inputs, c_proj, state
        )
        loss = normalized_fit_loss(
            predicted_pre, predicted_output, target_pre, target_output
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite unshared butterfly objective")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise RuntimeError("non-finite or missing unshared butterfly gradient")
        gradient = float(
            torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
        )
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
    return state.cpu(), {
        "steps": int(steps),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_preclip_gradient_norm": maximum_gradient,
    }


def gradient_conflict(
    operator: LearnedButterflyCFC,
    shared: ButterflyCFCState,
    inputs: torch.Tensor,
    c_fc: torch.Tensor,
    c_proj: torch.Tensor,
) -> dict[str, Any]:
    gradients: list[torch.Tensor] = []
    norms: list[float] = []
    losses: list[float] = []
    for expert in range(operator.experts):
        input_angles = shared.input_angles.to(operator.device).detach().clone()
        hidden_angles = shared.hidden_angles.to(operator.device).detach().clone()
        input_angles.requires_grad_(True)
        hidden_angles.requires_grad_(True)
        expert_state = ButterflyCFCState(
            input_angles,
            hidden_angles,
            shared.spectrum[expert : expert + 1].to(operator.device),
            shared.bias[expert : expert + 1].to(operator.device),
        )
        target_pre, target_output = dense_targets(
            inputs[expert : expert + 1],
            c_fc[expert : expert + 1],
            c_proj[expert : expert + 1],
            operator.device,
        )
        predicted_pre, predicted_output = operator.expert_output(
            inputs[expert : expert + 1],
            c_proj[expert : expert + 1],
            expert_state,
        )
        loss = normalized_fit_loss(
            predicted_pre,
            predicted_output,
            target_pre,
            target_output,
        )
        input_gradient, hidden_gradient = torch.autograd.grad(
            loss, (input_angles, hidden_angles)
        )
        gradient = torch.cat(
            (input_gradient.reshape(-1), hidden_gradient.reshape(-1))
        ).detach().float()
        gradients.append(gradient)
        norms.append(float(gradient.norm()))
        losses.append(float(loss.detach()))
    cosines = [
        float(
            gradients[left] @ gradients[right]
            / (gradients[left].norm() * gradients[right].norm()).clamp_min(1e-30)
        )
        for left, right in itertools.combinations(range(operator.experts), 2)
    ]
    stacked = torch.stack(gradients)
    cancellation = float(
        stacked.sum(dim=0).square().sum()
        / (operator.experts * stacked.square().sum()).clamp_min(1e-30)
    )
    return {
        "expert_objectives": losses,
        "expert_gradient_norms": norms,
        "finite_nonzero_gradient_count": sum(
            math.isfinite(value) and value > 1e-12 for value in norms
        ),
        "pairwise_cosine_mean": sum(cosines) / len(cosines),
        "pairwise_cosine_minimum": min(cosines),
        "pairwise_cosine_maximum": max(cosines),
        "negative_pair_fraction": sum(value < 0 for value in cosines) / len(cosines),
        "cancellation_ratio": cancellation,
    }


def routed_outputs(
    state: LayerState,
    activations: torch.Tensor,
    operator: LearnedButterflyCFC,
    compact: UnsharedButterflyState,
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
            candidate_pre, candidate_output = unshared_output(
                operator,
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


def classify(conflict_pass: bool, upper_bound_pass: bool) -> str:
    if conflict_pass and upper_bound_pass:
        return "EXPERT_SHARING_CONFLICT_CONFIRMED"
    if not upper_bound_pass:
        return "ONE_SWEEP_BUTTERFLY_TOPOLOGY_INSUFFICIENT"
    return "SHARING_DIAGNOSIS_AMBIGUOUS"


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("butterfly expert-conflict plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the frozen plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    if parent_authorization(True)["language_model_training"]:
        raise AssertionError("parent authorization unexpectedly permits training")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    constants = plan["replay_constants"]
    operator = LearnedButterflyCFC(
        experts=int(source["num_experts"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        input_padded_width=1024,
        hidden_padded_width=2048,
        seed=int(constants["operator_seed"]),
        layer=0,
        device=device,
    )
    shared = operator.initial_state(requires_grad=False).cpu()
    generator = torch.Generator(device=device)
    generator.manual_seed(20261020)
    inputs = torch.randn(8, 128, 768, generator=generator, device=device)
    c_fc = torch.randn(8, 1536, 768, generator=generator, device=device) * 0.02
    c_proj = torch.randn(8, 768, 1536, generator=generator, device=device) * 0.02
    started = time.time()
    fitted, diagnostics = fit_unshared_state(
        operator,
        shared,
        inputs,
        c_fc,
        c_proj,
        steps=5,
        learning_rate=float(constants["learning_rate"]),
        weight_decay=float(constants["weight_decay"]),
        gradient_clip=float(constants["gradient_clip"]),
    )
    elapsed = time.time() - started
    conflict = gradient_conflict(operator, shared, inputs, c_fc, c_proj)
    return {
        "schema_version": "nanogpt_sparse_moe_cfc_butterfly_expert_conflict_preflight_v1",
        "five_step_wall_seconds": elapsed,
        "projected_full_audit_seconds": elapsed * 80 * 6,
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else 0
        ),
        "all_values_finite": (
            fitted.all_finite()
            and all_finite({"fit": diagnostics, "conflict": conflict})
        ),
        "fit_diagnostics": diagnostics,
        "gradient_conflict": conflict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--parent-coordinates", type=Path)
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
    required = (
        args.parent_plan,
        args.parent_result,
        args.parent_coordinates,
        args.terminal_snapshot,
        args.data_dir,
        args.output,
    )
    if any(value is None for value in required):
        parser.error("scientific audit requires all parent, source, data, and output paths")
    started = time.time()
    parent = plan["causal_parent"]
    source = plan["source"]
    for path, expected, label in (
        (args.parent_plan, parent["plan_sha256"], "parent plan"),
        (args.parent_result, parent["remote_result_sha256"], "parent result"),
        (args.parent_coordinates, parent["coordinates_sha256"], "parent coordinates"),
        (args.terminal_snapshot, source["terminal_manifold_snapshot_sha256"], "snapshot"),
        (args.data_dir / "manifest.json", source["dataset_manifest_sha256"], "dataset manifest"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"{label} hash disagrees with frozen plan")
    parent_plan = json.loads(args.parent_plan.read_text(encoding="utf-8"))
    parent_result = json.loads(args.parent_result.read_text(encoding="utf-8"))
    if parent_result.get("passed"):
        raise ValueError("expert-conflict audit requires rejected parent")
    coordinates = torch.load(
        args.parent_coordinates, map_location="cpu", weights_only=False
    )
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("snapshot step disagrees with frozen plan")
    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, parent_plan, args.data_dir, args.device)
    terminal_mapping = dict(model.named_parameters())
    layers = [int(value) for value in source["layers"]]
    states = {
        layer: layer_state_from_mapping(terminal_mapping, layer) for layer in layers
    }
    del terminal_mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    constants = plan["replay_constants"]
    banks = [str(value) for value in source["discovery_banks"]]
    fitted_states: dict[str, dict[str, UnsharedButterflyState]] = {}
    fit_diagnostics: dict[str, dict[str, Any]] = {}
    conflict_rows: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    actions: dict[tuple[str, int], torch.Tensor] = {}

    for bank_index, bank in enumerate(banks):
        fitted_states[bank] = {}
        fit_diagnostics[bank] = {}
        conflict_rows[bank] = {}
        summaries[bank] = {}
        occupancy[bank] = {}
        for layer in layers:
            state = states[layer]
            sampled, counts = route_and_sample(
                state,
                inputs[bank][layer],
                top_k=int(constants["top_k"]),
                samples_per_expert=int(constants["fit_samples_per_expert"]),
                seed=(
                    int(constants["sampling_seed_base"])
                    + int(constants["sampling_seed_bank_stride"]) * bank_index
                    + int(constants["sampling_seed_layer_stride"]) * layer
                ),
            )
            occupancy[bank][str(layer)] = counts
            shared = coordinates["candidate"][bank][str(layer)]
            operator = LearnedButterflyCFC(
                experts=int(source["num_experts"]),
                input_width=int(source["input_width"]),
                hidden_width=int(source["expert_hidden_width"]),
                input_padded_width=1024,
                hidden_padded_width=2048,
                seed=int(constants["operator_seed"]),
                layer=layer,
                device=args.device,
            )
            conflict_rows[bank][str(layer)] = gradient_conflict(
                operator, shared, sampled, state.c_fc, state.c_proj
            )
            fitted, diagnostics = fit_unshared_state(
                operator,
                shared,
                sampled,
                state.c_fc,
                state.c_proj,
                steps=int(constants["fit_steps"]),
                learning_rate=float(constants["learning_rate"]),
                weight_decay=float(constants["weight_decay"]),
                gradient_clip=float(constants["gradient_clip"]),
            )
            fitted_states[bank][str(layer)] = fitted
            fit_diagnostics[bank][str(layer)] = diagnostics
            predicted, target, expert_recovery, pre_recovery = routed_outputs(
                state,
                inputs["heldout"][layer],
                operator,
                fitted,
                top_k=int(constants["top_k"]),
            )
            shared_predicted, shared_target, _, _ = shared_routed_outputs(
                state,
                inputs["heldout"][layer],
                operator,
                shared,
                top_k=int(constants["top_k"]),
            )
            if not torch.equal(target, shared_target):
                raise RuntimeError("upper-bound and shared targets drift")
            mixture = recovery_fraction(predicted, target)
            shared_mixture = recovery_fraction(shared_predicted, target)
            summaries[bank][str(layer)] = {
                "mixture_recovery": mixture,
                "shared_parent_mixture_recovery": shared_mixture,
                "gain_over_shared_parent": mixture - shared_mixture,
                "expert_recovery": expert_recovery,
                "minimum_expert_recovery": min(expert_recovery),
                "pregelu_recovery": pre_recovery,
                "minimum_pregelu_recovery": min(pre_recovery),
            }
            actions[(bank, layer)] = predicted

    upper_thresholds = plan["nondeployable_upper_bound"]["diagnostic_thresholds"]
    upper_gates: dict[str, dict[str, bool]] = {}
    for bank in banks:
        rows = [summaries[bank][str(layer)] for layer in layers]
        recoveries = [float(row["mixture_recovery"]) for row in rows]
        gains = [float(row["gain_over_shared_parent"]) for row in rows]
        aggregate = {
            "mixture_recovery_mean": sum(recoveries) / len(recoveries),
            "mixture_recovery_minimum_layer": min(recoveries),
            "minimum_expert_recovery": min(
                float(row["minimum_expert_recovery"]) for row in rows
            ),
            "gain_over_shared_parent_mean": sum(gains) / len(gains),
        }
        summaries[bank]["aggregate"] = aggregate
        upper_gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"]
            >= float(upper_thresholds["heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"]
            >= float(upper_thresholds["heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": aggregate["minimum_expert_recovery"]
            >= float(upper_thresholds["heldout_expert_recovery_min_each_bank"]),
            "gain_pass": aggregate["gain_over_shared_parent_mean"]
            >= float(upper_thresholds["gain_over_shared_parent_mean_min_each_bank"]),
        }
    agreement_by_layer = {
        str(layer): action_cosine(
            actions[(banks[0], layer)], actions[(banks[1], layer)]
        )
        for layer in layers
    }
    agreement_mean = sum(agreement_by_layer.values()) / len(agreement_by_layer)
    agreement_pass = agreement_mean >= float(
        upper_thresholds["heldout_bank_action_cosine_mean_min"]
    )
    finite = all(
        fitted_states[bank][str(layer)].all_finite()
        for bank in banks
        for layer in layers
    ) and all_finite(
        {
            "fit": fit_diagnostics,
            "conflict": conflict_rows,
            "summaries": summaries,
            "agreement": agreement_by_layer,
        }
    )
    for bank in banks:
        upper_gates[bank]["agreement_pass"] = agreement_pass
        upper_gates[bank]["finite_pass"] = finite
        upper_gates[bank]["all_pass"] = all(upper_gates[bank].values())
    upper_bound_pass = all(upper_gates[bank]["all_pass"] for bank in banks)

    conflict_thresholds = plan["shared_endpoint_gradient_audit"][
        "conflict_thresholds"
    ]
    all_conflicts = [
        conflict_rows[bank][str(layer)] for bank in banks for layer in layers
    ]
    mean_pairwise = sum(
        float(row["pairwise_cosine_mean"]) for row in all_conflicts
    ) / len(all_conflicts)
    mean_cancellation = sum(
        float(row["cancellation_ratio"]) for row in all_conflicts
    ) / len(all_conflicts)
    minimum_nonzero = min(
        int(row["finite_nonzero_gradient_count"]) for row in all_conflicts
    )
    conflict_gates = {
        "mean_pairwise_cosine_pass": mean_pairwise
        <= float(conflict_thresholds["mean_pairwise_cosine_max"]),
        "mean_cancellation_ratio_pass": mean_cancellation
        <= float(conflict_thresholds["mean_cancellation_ratio_max"]),
        "nonzero_gradient_count_pass": minimum_nonzero
        >= int(conflict_thresholds["minimum_finite_nonzero_expert_gradients_per_layer"]),
        "finite_pass": finite,
    }
    conflict_gates["all_pass"] = all(conflict_gates.values())
    classification = classify(conflict_gates["all_pass"], upper_bound_pass)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "nondeployable_unshared_coordinates.pt"
    torch.save(
        {
            "schema_version": "nanogpt_sparse_moe_cfc_unshared_butterfly_coordinates_v1",
            "states": fitted_states,
        },
        coordinates_path,
    )
    result = {
        "schema_version": "nanogpt_sparse_moe_cfc_butterfly_expert_conflict_result_v1",
        "classification": classification,
        "identity": {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "parent_plan_sha256": file_sha256(args.parent_plan),
            "parent_result_sha256": file_sha256(args.parent_result),
            "parent_coordinates_sha256": file_sha256(args.parent_coordinates),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
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
            "upper_bound_coordinates": int(
                plan["nondeployable_upper_bound"]["coordinates"]
            ),
            "upper_bound_compression_ratio": float(
                plan["nondeployable_upper_bound"]["cfc_compression_ratio"]
            ),
            "deployable": False,
            "dense_cproj_retained_as_exception": True,
        },
        "gradient_conflict": {
            "by_bank_and_layer": conflict_rows,
            "mean_pairwise_cosine": mean_pairwise,
            "mean_cancellation_ratio": mean_cancellation,
            "minimum_finite_nonzero_gradient_count": minimum_nonzero,
            "gates": conflict_gates,
        },
        "fit_diagnostics": fit_diagnostics,
        "summaries": summaries,
        "heldout_bank_action_cosine": {
            "mean": agreement_mean,
            "by_layer": agreement_by_layer,
        },
        "upper_bound_gates": upper_gates,
        "upper_bound_passed": upper_bound_pass,
        "all_values_finite": finite,
        "authorization": {
            "implementation": False,
            "initialization_fit_shadow": False,
            "mfu_preflight": False,
            "language_model_training": False,
            "larger_rung": False,
            "generated_cproj": False,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
