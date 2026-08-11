#!/usr/bin/env python3
"""Gate a compact learned-spectral c_fc feature map for sparse MoE experts."""
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
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
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
from latent_weight_lab.block_fht import (
    normalized_fht_last_dim,
    signs_for,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_spectral_feature_oracle_plan_v1"


def action_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denominator = left.norm() * right.norm()
    return float((left @ right) / denominator.clamp_min(1e-30))


def _seeded_signs(
    reference: torch.Tensor,
    *,
    experts: int,
    padded_width: int,
    seed: int,
    layer: int,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack(
                [
                    signs_for(
                        reference,
                        expert,
                        stage,
                        int(seed) + 1009 * int(layer),
                        padded_width,
                    )
                    for stage in range(3)
                ]
            )
            for expert in range(experts)
        ]
    )


@dataclass
class CompactCFCState:
    spectral_1: torch.Tensor
    spectral_2: torch.Tensor
    bias: torch.Tensor
    scale: torch.Tensor | None = None

    def cpu(self) -> "CompactCFCState":
        return CompactCFCState(
            self.spectral_1.detach().cpu(),
            self.spectral_2.detach().cpu(),
            self.bias.detach().cpu(),
            None if self.scale is None else self.scale.detach().cpu(),
        )


class SpectralCFC:
    """Two learned diagonal spectra between three fixed global FHT mixers."""

    def __init__(
        self,
        *,
        experts: int,
        input_width: int,
        hidden_width: int,
        padded_width: int,
        seed: int,
        layer: int,
        device: str,
    ) -> None:
        if padded_width & (padded_width - 1):
            raise ValueError("padded width must be a power of two")
        if padded_width < max(input_width, hidden_width):
            raise ValueError("padded width does not cover feature widths")
        self.experts = int(experts)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.padded_width = int(padded_width)
        self.device = device
        reference = torch.empty(1, device=device, dtype=torch.float32)
        self.signs = _seeded_signs(
            reference,
            experts=experts,
            padded_width=padded_width,
            seed=seed,
            layer=layer,
        ).to(device=device, dtype=torch.float32)
        self.base_scale = math.sqrt(float(padded_width)) * 0.02

    @property
    def coordinates_per_expert(self) -> int:
        return 2 * self.padded_width + self.hidden_width

    def fixed_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape[:1]) != (self.experts,):
            raise ValueError("expert axis disagrees with spectral operator")
        if inputs.shape[-1] != self.input_width:
            raise ValueError("input width disagrees with spectral operator")
        values = F.pad(
            inputs.to(device=self.device, dtype=torch.float32),
            (0, self.padded_width - self.input_width),
        )
        values = normalized_fht_last_dim(values * self.signs[:, 0, None, :])
        return values

    def preactivation(
        self,
        inputs: torch.Tensor,
        state: CompactCFCState,
        *,
        spectral: bool,
    ) -> torch.Tensor:
        values = self.fixed_features(inputs)
        if spectral:
            first = 1.0 + state.spectral_1.to(self.device)[:, None, :]
            second = 1.0 + state.spectral_2.to(self.device)[:, None, :]
        else:
            first = torch.ones_like(values[:, :1, :])
            second = torch.ones_like(values[:, :1, :])
        values = normalized_fht_last_dim(
            values * first * self.signs[:, 1, None, :]
        )
        values = normalized_fht_last_dim(
            values * second * self.signs[:, 2, None, :]
        )
        preactivation = self.base_scale * values[..., : self.hidden_width]
        if state.scale is not None:
            preactivation = preactivation * state.scale.to(self.device)[:, None, :]
        return preactivation + state.bias.to(self.device)[:, None, :]

    def expert_output(
        self,
        inputs: torch.Tensor,
        c_proj: torch.Tensor,
        state: CompactCFCState,
        *,
        spectral: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preactivation = self.preactivation(inputs, state, spectral=spectral)
        output = torch.bmm(
            F.gelu(preactivation),
            c_proj.to(self.device).transpose(1, 2),
        )
        return preactivation, output


def route_and_sample(
    state: LayerState,
    activations: torch.Tensor,
    *,
    top_k: int,
    samples_per_expert: int,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    x = activations.float().cpu()
    logits = x @ state.router.float().cpu().T
    tie = torch.arange(logits.shape[-1], dtype=logits.dtype)
    selected = torch.topk(
        logits - tie * torch.finfo(logits.dtype).eps,
        top_k,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    samples: list[torch.Tensor] = []
    counts: list[int] = []
    for expert in range(state.c_fc.shape[0]):
        indices = (selected == expert).any(dim=-1).nonzero(as_tuple=False).flatten()
        count = int(indices.numel())
        counts.append(count)
        if count < samples_per_expert:
            raise RuntimeError(
                f"expert {expert} has {count} assignments, below "
                f"required {samples_per_expert}"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 131 * expert)
        chosen = indices.index_select(
            0, torch.randperm(count, generator=generator)[:samples_per_expert]
        )
        samples.append(x.index_select(0, chosen))
    return torch.stack(samples), counts


def dense_targets(
    inputs: torch.Tensor,
    c_fc: torch.Tensor,
    c_proj: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = inputs.to(device=device, dtype=torch.float32)
    c_fc = c_fc.to(device=device, dtype=torch.float32)
    c_proj = c_proj.to(device=device, dtype=torch.float32)
    preactivation = torch.bmm(x, c_fc.transpose(1, 2))
    output = torch.bmm(F.gelu(preactivation), c_proj.transpose(1, 2))
    return preactivation, output


def normalized_fit_loss(
    predicted_pre: torch.Tensor,
    predicted_output: torch.Tensor,
    target_pre: torch.Tensor,
    target_output: torch.Tensor,
) -> torch.Tensor:
    pre_scale = target_pre.square().mean(dim=(1, 2)).clamp_min(1e-12)
    output_scale = target_output.square().mean(dim=(1, 2)).clamp_min(1e-12)
    pre_error = (predicted_pre - target_pre).square().mean(dim=(1, 2)) / pre_scale
    output_error = (
        (predicted_output - target_output).square().mean(dim=(1, 2))
        / output_scale
    )
    return (output_error + 0.25 * pre_error).mean()


def fit_compact_state(
    operator: SpectralCFC,
    inputs: torch.Tensor,
    c_fc: torch.Tensor,
    c_proj: torch.Tensor,
    *,
    spectral: bool,
    steps: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[CompactCFCState, dict[str, float | int]]:
    device = operator.device
    target_pre, target_output = dense_targets(inputs, c_fc, c_proj, device)
    spectral_1 = torch.zeros(
        operator.experts, operator.padded_width, device=device, requires_grad=spectral
    )
    spectral_2 = torch.zeros_like(spectral_1, requires_grad=spectral)
    bias = torch.zeros(
        operator.experts, operator.hidden_width, device=device, requires_grad=True
    )
    scale = None
    parameters: list[torch.Tensor] = [bias]
    if spectral:
        parameters.extend((spectral_1, spectral_2))
    else:
        scale = torch.ones(
            operator.experts, 1, device=device, requires_grad=True
        )
        parameters.append(scale)
    state = CompactCFCState(spectral_1, spectral_2, bias, scale)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    losses: list[float] = []
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        predicted_pre, predicted_output = operator.expert_output(
            inputs, c_proj, state, spectral=spectral
        )
        loss = normalized_fit_loss(
            predicted_pre, predicted_output, target_pre, target_output
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite compact c_fc fit loss")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise RuntimeError("non-finite or missing compact c_fc gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    return state.cpu(), {
        "steps": int(steps),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
    }


def routed_outputs(
    state: LayerState,
    activations: torch.Tensor,
    operator: SpectralCFC,
    compact: CompactCFCState,
    *,
    spectral: bool,
    top_k: int,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float]]:
    state = state.to(operator.device)
    predicted_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    expert_error = torch.zeros(operator.experts, dtype=torch.float64)
    expert_energy = torch.zeros(operator.experts, dtype=torch.float64)
    expert_pre_error = torch.zeros(operator.experts, dtype=torch.float64)
    expert_pre_energy = torch.zeros(operator.experts, dtype=torch.float64)
    for start in range(0, activations.shape[0], chunk_size):
        x = activations[start : start + chunk_size].to(
            device=operator.device, dtype=torch.float32
        )
        logits = x @ state.router.T
        tie = torch.arange(logits.shape[-1], device=x.device, dtype=x.dtype)
        selected = torch.topk(
            logits - tie * torch.finfo(x.dtype).eps,
            top_k,
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
            compact_expert = CompactCFCState(
                compact.spectral_1[expert : expert + 1],
                compact.spectral_2[expert : expert + 1],
                compact.bias[expert : expert + 1],
                None if compact.scale is None else compact.scale[expert : expert + 1],
            )
            expert_operator = SpectralCFC(
                experts=1,
                input_width=operator.input_width,
                hidden_width=operator.hidden_width,
                padded_width=operator.padded_width,
                seed=0,
                layer=0,
                device=operator.device,
            )
            expert_operator.signs = operator.signs[expert : expert + 1]
            predicted_pre, predicted_output = expert_operator.expert_output(
                expert_input,
                state.c_proj[expert : expert + 1],
                compact_expert,
                spectral=spectral,
            )
            target_pre, target_output = dense_targets(
                expert_input,
                state.c_fc[expert : expert + 1],
                state.c_proj[expert : expert + 1],
                operator.device,
            )
            weight = probabilities[token, slot, None]
            predicted.index_add_(0, token, predicted_output[0] * weight)
            target.index_add_(0, token, target_output[0] * weight)
            expert_error[expert] += float(
                (predicted_output - target_output).square().sum()
            )
            expert_energy[expert] += float(target_output.square().sum())
            expert_pre_error[expert] += float(
                (predicted_pre - target_pre).square().sum()
            )
            expert_pre_energy[expert] += float(target_pre.square().sum())
        predicted_chunks.append(predicted.cpu())
        target_chunks.append(target.cpu())
    expert_recovery = [
        1.0 - float(error / max(energy, 1e-30))
        for error, energy in zip(expert_error, expert_energy)
    ]
    pre_recovery = [
        1.0 - float(error / max(energy, 1e-30))
        for error, energy in zip(expert_pre_error, expert_pre_energy)
    ]
    return (
        torch.cat(predicted_chunks),
        torch.cat(target_chunks),
        expert_recovery,
        pre_recovery,
    )


def collect_protocol_inputs(
    model: torch.nn.Module,
    plan: dict[str, Any],
    data_dir: Path,
    device: str,
) -> dict[str, dict[int, torch.Tensor]]:
    layers = [int(layer) for layer in plan["source"]["layers"]]
    result: dict[str, dict[int, torch.Tensor]] = {}
    for spec in plan["data_protocol"]["discovery_banks"]:
        batches = fixed_validation_batches(
            data_dir,
            int(spec["batch_size"]),
            int(spec["block_size"]),
            int(spec["batches"]),
            int(spec["seed"]),
        )
        result[spec["name"]] = collect_inputs(
            model, batches, layers, int(spec["tokens"]), device
        )
    spec = plan["data_protocol"]["heldout"]
    batches = fixed_validation_batches(
        data_dir,
        int(spec["batch_size"]),
        int(spec["block_size"]),
        int(spec["batches"]),
        int(spec["seed"]),
    )
    result["heldout"] = collect_inputs(
        model, batches, layers, int(spec["tokens"]), device
    )
    return result


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("spectral c_fc oracle plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the frozen plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    candidate = plan["candidate"]
    expected = (
        2 * int(candidate["padded_width"])
        + int(candidate["expert_hidden_width"])
    )
    if expected != int(candidate["trainable_coordinates_per_expert"]):
        raise ValueError("compact c_fc coordinate accounting drift")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--terminal-snapshot", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, args.plan)
    source = plan["source"]
    if file_sha256(args.terminal_snapshot) != source["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash disagrees with frozen plan")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal snapshot step disagrees with frozen plan")
    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, plan, args.data_dir, args.device)
    terminal_mapping = dict(model.named_parameters())
    layers = [int(layer) for layer in source["layers"]]
    states = {
        layer: layer_state_from_mapping(terminal_mapping, layer) for layer in layers
    }
    del terminal_mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    candidate_spec = plan["candidate"]
    fit_spec = plan["fit_protocol"]
    samples_per_expert = int(plan["data_protocol"]["fit_samples_per_expert"])
    banks = [spec["name"] for spec in plan["data_protocol"]["discovery_banks"]]
    compact_states: dict[str, dict[str, CompactCFCState]] = {}
    control_states: dict[str, dict[str, CompactCFCState]] = {}
    fit_diagnostics: dict[str, dict[str, Any]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    heldout_actions: dict[tuple[str, int], torch.Tensor] = {}

    for bank_index, bank in enumerate(banks):
        compact_states[bank] = {}
        control_states[bank] = {}
        fit_diagnostics[bank] = {}
        occupancy[bank] = {}
        summaries[bank] = {}
        for layer in layers:
            state = states[layer]
            sampled, counts = route_and_sample(
                state,
                inputs[bank][layer],
                top_k=int(candidate_spec["top_k"]),
                samples_per_expert=samples_per_expert,
                seed=20260932 + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            operator = SpectralCFC(
                experts=int(candidate_spec["num_experts"]),
                input_width=int(candidate_spec["input_width"]),
                hidden_width=int(candidate_spec["expert_hidden_width"]),
                padded_width=int(candidate_spec["padded_width"]),
                seed=int(candidate_spec["fixed_operator_seed"]),
                layer=layer,
                device=args.device,
            )
            compact, compact_diag = fit_compact_state(
                operator,
                sampled,
                state.c_fc,
                state.c_proj,
                spectral=True,
                steps=int(fit_spec["steps"]),
                learning_rate=float(fit_spec["learning_rate"]),
                weight_decay=float(fit_spec["weight_decay"]),
            )
            control, control_diag = fit_compact_state(
                operator,
                sampled,
                state.c_fc,
                state.c_proj,
                spectral=False,
                steps=int(fit_spec["steps"]),
                learning_rate=float(fit_spec["learning_rate"]),
                weight_decay=float(fit_spec["weight_decay"]),
            )
            compact_states[bank][str(layer)] = compact
            control_states[bank][str(layer)] = control
            fit_diagnostics[bank][str(layer)] = {
                "spectral": compact_diag,
                "control": control_diag,
            }
            predicted, target, expert_recovery, pre_recovery = routed_outputs(
                state,
                inputs["heldout"][layer],
                operator,
                compact,
                spectral=True,
                top_k=int(candidate_spec["top_k"]),
            )
            control_predicted, control_target, _, _ = routed_outputs(
                state,
                inputs["heldout"][layer],
                operator,
                control,
                spectral=False,
                top_k=int(candidate_spec["top_k"]),
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
        layer_rows = [summaries[bank][str(layer)] for layer in layers]
        mixture = [float(row["mixture_recovery"]) for row in layer_rows]
        improvements = [
            float(row["candidate_minus_control_recovery"]) for row in layer_rows
        ]
        minimum_expert = min(
            float(row["minimum_expert_recovery"]) for row in layer_rows
        )
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
            "spectral_gain_pass": aggregate["candidate_minus_control_recovery_mean"]
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
            "schema_version": "nanogpt_sparse_moe_cfc_spectral_coordinates_v1",
            "candidate": compact_states,
            "control": control_states,
        },
        coordinates_path,
    )
    result = {
        "schema_version": "nanogpt_sparse_moe_cfc_spectral_feature_oracle_result_v1",
        "classification": (
            "SPECTRAL_CFC_REPRESENTABILITY_PASSES"
            if passed
            else "SPECTRAL_CFC_REPRESENTABILITY_REJECTED"
        ),
        "passed": passed,
        "identity": {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": source["dataset_manifest_sha256"],
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "coordinates_path": str(coordinates_path),
            "coordinates_sha256": file_sha256(coordinates_path),
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
        "authorization": {
            "implementation": passed,
            "initialization_fit_shadow": passed,
            "mfu_preflight": false,
            "language_model_training": false,
            "larger_rung": false,
            "generated_cproj": false,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = {
        "status": "finished",
        "exit_code": 0,
        "classification": result["classification"],
        "result": str(result_path),
        "result_sha256": file_sha256(result_path),
        "finished_at_epoch": time.time(),
    }
    (args.output / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
