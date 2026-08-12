#!/usr/bin/env python3
"""Gate a nonlinear paired-neuron coordinate field for sparse-MoE MLPs."""
from __future__ import annotations

import argparse
import copy
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
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_paired_coordinate_field_oracle_plan_v1"


def decoder_parameter_count(input_width: int, hidden_width: int) -> int:
    return (
        input_width * hidden_width
        + hidden_width
        + hidden_width * hidden_width
        + hidden_width
        + hidden_width * 2
        + 2
    )


def coordinate_count(
    *, experts: int, hidden_width: int, code_width: int, decoder_input: int,
    decoder_hidden: int,
) -> int:
    return (
        experts * hidden_width * code_width
        + experts * hidden_width
        + decoder_parameter_count(decoder_input, decoder_hidden)
        + 2
    )


def channel_encoding(width: int, frequencies: int, device: str) -> torch.Tensor:
    if width < 2 or frequencies < 1:
        raise ValueError("coordinate encoding needs width>=2 and frequencies>=1")
    coordinate = torch.linspace(-1.0, 1.0, width, device=device)
    powers = 2.0 ** torch.arange(frequencies, device=device)
    phase = math.pi * coordinate[:, None] * powers[None, :]
    return torch.cat((coordinate[:, None], torch.sin(phase), torch.cos(phase)), dim=1)


def gelu_derivative(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0))) + (
        values * torch.exp(-0.5 * values.square()) / math.sqrt(2.0 * math.pi)
    )


def rademacher(shape: tuple[int, ...], seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
    values = values.to(device=device, dtype=torch.float32).mul_(2).sub_(1)
    return values / math.sqrt(float(shape[-1]))


class PairedCoordinateField(torch.nn.Module):
    """Jointly generate c_fc rows and c_proj columns from neuron codes."""

    def __init__(
        self,
        *,
        experts: int,
        input_width: int,
        hidden_width: int,
        code_width: int,
        encoding_frequencies: int,
        decoder_hidden_width: int,
        layer: int,
        tensor_layers: int,
        seed: int,
        device: str,
        channel_chunk: int = 96,
    ) -> None:
        super().__init__()
        self.experts = int(experts)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.code_width = int(code_width)
        self.encoding_frequencies = int(encoding_frequencies)
        self.encoding_width = 1 + 2 * self.encoding_frequencies
        self.decoder_hidden_width = int(decoder_hidden_width)
        self.channel_chunk = int(channel_chunk)
        if self.channel_chunk <= 0:
            raise ValueError("channel chunk must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 1009 * int(layer))
            self.codes = torch.nn.Parameter(
                0.5 * torch.randn(self.experts, self.hidden_width, self.code_width)
            )
            self.hidden_bias = torch.nn.Parameter(
                torch.zeros(self.experts, self.hidden_width)
            )
            self.log_scales = torch.nn.Parameter(
                torch.tensor(
                    [math.log(0.02), math.log(0.02 / math.sqrt(2 * tensor_layers))]
                )
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(self.code_width + self.encoding_width, self.decoder_hidden_width),
                torch.nn.GELU(),
                torch.nn.Linear(self.decoder_hidden_width, self.decoder_hidden_width),
                torch.nn.GELU(),
                torch.nn.Linear(self.decoder_hidden_width, 2),
            )
            for module in self.decoder:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    torch.nn.init.zeros_(module.bias)
        self.register_buffer(
            "encoding",
            channel_encoding(self.input_width, self.encoding_frequencies, "cpu"),
            persistent=True,
        )
        self.to(device=device, dtype=torch.float32)

    def compact_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def decoder_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.decoder.parameters())

    def coordinate_parameters(self) -> list[torch.nn.Parameter]:
        return [self.codes, self.hidden_bias, self.log_scales]

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        pieces: list[torch.Tensor] = []
        codes = self.codes[:, :, None, :]
        for start in range(0, self.input_width, self.channel_chunk):
            stop = min(self.input_width, start + self.channel_chunk)
            count = stop - start
            live_codes = codes.expand(-1, -1, count, -1)
            live_encoding = self.encoding[start:stop][None, None, :, :].expand(
                self.experts, self.hidden_width, -1, -1
            )
            fields = torch.cat((live_codes, live_encoding), dim=-1)
            pieces.append(self.decoder(fields.reshape(-1, fields.shape[-1])).reshape(
                self.experts, self.hidden_width, count, 2
            ))
        paired = torch.cat(pieces, dim=2) * self.log_scales.exp()[None, None, None, :]
        return paired[..., 0], paired[..., 1]

    def function_and_jvp(
        self, inputs: torch.Tensor, directions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_fc, c_proj_atoms = self.materialize()
        return function_and_jvp(
            inputs, directions, c_fc, c_proj_atoms, self.hidden_bias
        )


def function_and_jvp(
    inputs: torch.Tensor,
    directions: torch.Tensor,
    c_fc: torch.Tensor,
    c_proj_atoms: torch.Tensor,
    hidden_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.shape != directions.shape:
        raise ValueError("input and JVP direction shapes disagree")
    preactivation = torch.bmm(inputs, c_fc.transpose(1, 2))
    if hidden_bias is not None:
        preactivation = preactivation + hidden_bias[:, None, :]
    preactivation_jvp = torch.bmm(directions, c_fc.transpose(1, 2))
    output = torch.bmm(F.gelu(preactivation), c_proj_atoms)
    output_jvp = torch.bmm(
        gelu_derivative(preactivation) * preactivation_jvp,
        c_proj_atoms,
    )
    return output, output_jvp


def normalized_expert_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (predicted - target).square().sum(dim=(1, 2))
    energy = target.square().sum(dim=(1, 2)).clamp_min(1e-30)
    return (error / energy).mean()


def fit_field(
    field: PairedCoordinateField,
    inputs: torch.Tensor,
    dense_c_fc: torch.Tensor,
    dense_c_proj: torch.Tensor,
    *,
    steps: int,
    decoder_learning_rate: float,
    coordinate_learning_rate: float,
    decoder_weight_decay: float,
    code_weight_decay: float,
    gradient_clip: float,
    jvp_weight: float,
    probe_seed: int,
    train_decoder: bool,
) -> dict[str, Any]:
    device = str(field.codes.device)
    live_inputs = inputs.to(device=device, dtype=torch.float32)
    target_fc = dense_c_fc.to(device=device, dtype=torch.float32)
    target_proj_atoms = dense_c_proj.to(device=device, dtype=torch.float32).transpose(1, 2)
    directions = rademacher(tuple(live_inputs.shape), probe_seed, device)
    with torch.no_grad():
        target_output, target_jvp = function_and_jvp(
            live_inputs, directions, target_fc, target_proj_atoms
        )
    for parameter in field.decoder_parameters():
        parameter.requires_grad_(bool(train_decoder))
    groups: list[dict[str, Any]] = [
        {
            "params": field.coordinate_parameters(),
            "lr": float(coordinate_learning_rate),
            "weight_decay": float(code_weight_decay),
        }
    ]
    if train_decoder:
        groups.append(
            {
                "params": field.decoder_parameters(),
                "lr": float(decoder_learning_rate),
                "weight_decay": float(decoder_weight_decay),
            }
        )
    optimizer = torch.optim.AdamW(groups)
    losses: list[float] = []
    output_losses: list[float] = []
    jvp_losses: list[float] = []
    maximum_gradient = 0.0
    parameters = [parameter for group in groups for parameter in group["params"]]
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        predicted_output, predicted_jvp = field.function_and_jvp(live_inputs, directions)
        output_loss = normalized_expert_loss(predicted_output, target_output)
        jvp_loss = normalized_expert_loss(predicted_jvp, target_jvp)
        loss = output_loss + float(jvp_weight) * jvp_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite paired-coordinate objective")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise RuntimeError("non-finite or missing paired-coordinate gradient")
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip)))
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
        output_losses.append(float(output_loss.detach()))
        jvp_losses.append(float(jvp_loss.detach()))
    return {
        "steps": int(steps),
        "train_decoder": bool(train_decoder),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "initial_output_loss": output_losses[0],
        "final_output_loss": output_losses[-1],
        "initial_jvp_loss": jvp_losses[0],
        "final_jvp_loss": jvp_losses[-1],
        "maximum_preclip_gradient_norm": maximum_gradient,
        "code_norm_mean": float(field.codes.detach().norm(dim=-1).mean()),
        "hidden_bias_rms": float(field.hidden_bias.detach().square().mean().sqrt()),
        "scales": [float(value) for value in field.log_scales.detach().exp()],
    }


@torch.no_grad()
def routed_evaluation(
    state: LayerState,
    activations: torch.Tensor,
    field: PairedCoordinateField,
    *,
    top_k: int,
    probe_seed: int,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    device = str(field.codes.device)
    state = state.to(device)
    candidate_fc, candidate_proj_atoms = field.materialize()
    target_fc = state.c_fc.float()
    target_proj_atoms = state.c_proj.float().transpose(1, 2)
    all_directions = rademacher(tuple(activations.shape), probe_seed, device="cpu")
    predicted_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    predicted_jvp_chunks: list[torch.Tensor] = []
    target_jvp_chunks: list[torch.Tensor] = []
    expert_error = torch.zeros(field.experts, dtype=torch.float64)
    expert_energy = torch.zeros(field.experts, dtype=torch.float64)
    expert_jvp_error = torch.zeros(field.experts, dtype=torch.float64)
    expert_jvp_energy = torch.zeros(field.experts, dtype=torch.float64)
    for start in range(0, activations.shape[0], int(chunk_size)):
        stop = min(activations.shape[0], start + int(chunk_size))
        x = activations[start:stop].to(device=device, dtype=torch.float32)
        direction = all_directions[start:stop].to(device=device)
        logits = x @ state.router.T
        tie = torch.arange(logits.shape[-1], device=device, dtype=x.dtype)
        selected = torch.topk(
            logits - tie * torch.finfo(x.dtype).eps,
            int(top_k), dim=-1, largest=True, sorted=True,
        ).indices
        probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
        predicted = torch.zeros_like(x)
        target = torch.zeros_like(x)
        predicted_jvp = torch.zeros_like(x)
        target_jvp = torch.zeros_like(x)
        for expert in range(field.experts):
            locations = (selected == expert).nonzero(as_tuple=False)
            if not locations.numel():
                continue
            token = locations[:, 0]
            slot = locations[:, 1]
            expert_input = x.index_select(0, token)[None]
            expert_direction = direction.index_select(0, token)[None]
            candidate_output, candidate_jvp = function_and_jvp(
                expert_input,
                expert_direction,
                candidate_fc[expert : expert + 1],
                candidate_proj_atoms[expert : expert + 1],
                field.hidden_bias[expert : expert + 1],
            )
            dense_output, dense_jvp = function_and_jvp(
                expert_input,
                expert_direction,
                target_fc[expert : expert + 1],
                target_proj_atoms[expert : expert + 1],
            )
            weight = probabilities[token, slot, None]
            predicted.index_add_(0, token, candidate_output[0] * weight)
            target.index_add_(0, token, dense_output[0] * weight)
            predicted_jvp.index_add_(0, token, candidate_jvp[0] * weight)
            target_jvp.index_add_(0, token, dense_jvp[0] * weight)
            expert_error[expert] += float((candidate_output - dense_output).square().sum())
            expert_energy[expert] += float(dense_output.square().sum())
            expert_jvp_error[expert] += float((candidate_jvp - dense_jvp).square().sum())
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


def result_authorization(passed: bool) -> dict[str, bool]:
    return {
        "implementation": bool(passed),
        "initialization_and_mapping_loss_shadow": bool(passed),
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
        "full_attention_work": False,
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("paired-coordinate plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    source = plan["source"]
    candidate = plan["candidate"]
    expected = coordinate_count(
        experts=int(source["num_experts"]),
        hidden_width=int(source["expert_hidden_width"]),
        code_width=int(candidate["neuron_code_width"]),
        decoder_input=int(candidate["decoder_input_width"]),
        decoder_hidden=int(candidate["decoder_hidden_width"]),
    )
    if expected != int(candidate["total_coordinates_per_layer"]):
        raise ValueError("paired-coordinate accounting drift")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def make_field(plan: dict[str, Any], layer: int, device: str, seed: int) -> PairedCoordinateField:
    source = plan["source"]
    candidate = plan["candidate"]
    return PairedCoordinateField(
        experts=int(source["num_experts"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        code_width=int(candidate["neuron_code_width"]),
        encoding_frequencies=(int(candidate["channel_encoding_width"]) - 1) // 2,
        decoder_hidden_width=int(candidate["decoder_hidden_width"]),
        layer=layer,
        tensor_layers=int(source["tensor_layers"]),
        seed=seed,
        device=device,
    )


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    field = make_field(plan, 0, device, 20261104)
    generator = torch.Generator(device="cpu").manual_seed(20261105)
    source = plan["source"]
    shape = (int(source["num_experts"]), 16, int(source["input_width"]))
    inputs = torch.randn(shape, generator=generator)
    c_fc = torch.randn(
        int(source["num_experts"]), int(source["expert_hidden_width"]),
        int(source["input_width"]), generator=generator,
    ) * 0.02
    c_proj = torch.randn(
        int(source["num_experts"]), int(source["input_width"]),
        int(source["expert_hidden_width"]), generator=generator,
    ) * 0.02
    fit = plan["fit_protocol"]
    started = time.time()
    diagnostics = fit_field(
        field, inputs, c_fc, c_proj,
        steps=2,
        decoder_learning_rate=float(fit["decoder_learning_rate"]),
        coordinate_learning_rate=float(fit["code_bias_scale_learning_rate"]),
        decoder_weight_decay=float(fit["decoder_weight_decay"]),
        code_weight_decay=float(fit["code_weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
        jvp_weight=0.10,
        probe_seed=20261106,
        train_decoder=True,
    )
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_paired_coordinate_field_preflight_v1",
        "device": device,
        "two_step_wall_seconds_one_fit": elapsed,
        "projected_full_protocol_seconds": elapsed * (int(fit["steps"]) / 2.0) * 12.0,
        "compact_parameter_count": field.compact_parameter_count(),
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
        "all_values_finite": all_finite(diagnostics),
        "diagnostics": diagnostics,
    }


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


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
    samples_per_expert = int(plan["data_protocol"]["fit_samples_per_expert"])
    banks = [row["name"] for row in plan["data_protocol"]["discovery_banks"]]
    saved: dict[str, dict[str, Any]] = {}
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
                state, inputs[bank][layer], top_k=int(source["moe_top_k"]),
                samples_per_expert=samples_per_expert,
                seed=20261107 + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            candidate = make_field(plan, layer, args.device, 20261104)
            control = make_field(plan, layer, args.device, 20261104)
            control.load_state_dict(copy.deepcopy(candidate.state_dict()))
            candidate_diagnostics = fit_field(
                candidate, sampled, state.c_fc, state.c_proj,
                steps=int(fit["steps"]),
                decoder_learning_rate=float(fit["decoder_learning_rate"]),
                coordinate_learning_rate=float(fit["code_bias_scale_learning_rate"]),
                decoder_weight_decay=float(fit["decoder_weight_decay"]),
                code_weight_decay=float(fit["code_weight_decay"]),
                gradient_clip=float(fit["gradient_clip"]),
                jvp_weight=0.10,
                probe_seed=20261108 + 1009 * bank_index + 17 * layer,
                train_decoder=True,
            )
            control_diagnostics = fit_field(
                control, sampled, state.c_fc, state.c_proj,
                steps=int(fit["steps"]),
                decoder_learning_rate=float(fit["decoder_learning_rate"]),
                coordinate_learning_rate=float(fit["code_bias_scale_learning_rate"]),
                decoder_weight_decay=float(fit["decoder_weight_decay"]),
                code_weight_decay=float(fit["code_weight_decay"]),
                gradient_clip=float(fit["gradient_clip"]),
                jvp_weight=0.10,
                probe_seed=20261108 + 1009 * bank_index + 17 * layer,
                train_decoder=False,
            )
            candidate_eval = routed_evaluation(
                state, inputs["heldout"][layer], candidate,
                top_k=int(source["moe_top_k"]),
                probe_seed=20261109 + 17 * layer,
            )
            control_eval = routed_evaluation(
                state, inputs["heldout"][layer], control,
                top_k=int(source["moe_top_k"]),
                probe_seed=20261109 + 17 * layer,
            )
            if not torch.equal(candidate_eval["target"], control_eval["target"]):
                raise RuntimeError("candidate/control target drift")
            actions[(bank, layer)] = candidate_eval["predicted"]
            summaries[bank][str(layer)] = {
                "mixture_recovery": candidate_eval["mixture_recovery"],
                "jvp_recovery": candidate_eval["jvp_recovery"],
                "minimum_expert_recovery": min(candidate_eval["expert_recovery"]),
                "minimum_expert_jvp_recovery": min(candidate_eval["expert_jvp_recovery"]),
                "control_mixture_recovery": control_eval["mixture_recovery"],
                "candidate_minus_control_recovery": (
                    candidate_eval["mixture_recovery"] - control_eval["mixture_recovery"]
                ),
            }
            diagnostics[bank][str(layer)] = {
                "candidate": candidate_diagnostics,
                "control": control_diagnostics,
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
            "candidate_minus_control_recovery_mean": sum(
                float(row["candidate_minus_control_recovery"]) for row in rows
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
            "control_gain_pass": aggregate["candidate_minus_control_recovery_mean"] >= float(frozen["candidate_minus_control_recovery_mean_min_each_bank"]),
            "occupancy_pass": aggregate["minimum_discovery_assignments"] >= int(frozen["minimum_discovery_assignments_per_expert"]),
        }
    agreement_by_layer = {
        str(layer): action_cosine(actions[(banks[0], layer)], actions[(banks[1], layer)])
        for layer in layers
    }
    agreement_mean = sum(agreement_by_layer.values()) / len(agreement_by_layer)
    finite = all_finite({"summaries": summaries, "diagnostics": diagnostics, "agreement": agreement_by_layer})
    for bank in banks:
        bank_gates[bank]["action_agreement_pass"] = agreement_mean >= float(frozen["heldout_bank_action_cosine_mean_min"])
        bank_gates[bank]["finite_pass"] = finite
        bank_gates[bank]["all_pass"] = all(bank_gates[bank].values())
    passed = all(bank_gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({"schema_version": "nanogpt_sparse_moe_paired_coordinate_field_coordinates_v1", "fields": saved}, coordinates_path)
    result = {
        "schema_version": "nanogpt_sparse_moe_paired_coordinate_field_oracle_result_v1",
        "classification": "PAIRED_COORDINATE_FIELD_REPRESENTABILITY_PASSES" if passed else "PAIRED_COORDINATE_FIELD_REPRESENTABILITY_REJECTED",
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
            "maximum_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        },
        "accounting": {
            "dense_paired_parameters": int(plan["candidate"]["dense_paired_parameters_all_layers"]),
            "compact_coordinates": int(plan["candidate"]["total_coordinates_all_layers"]),
            "compression_ratio": float(plan["candidate"]["paired_parameter_compression_ratio"]),
            "materialized_dense_cfc_in_candidate": False,
            "materialized_dense_cproj_in_candidate": False,
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
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
