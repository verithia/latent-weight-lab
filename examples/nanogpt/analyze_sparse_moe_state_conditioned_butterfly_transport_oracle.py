#!/usr/bin/env python3
"""Gate full-rank state-conditioned orthogonal transport for complete MLPs."""
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
from examples.nanogpt.analyze_sparse_moe_conditional_complete_atom_oracle import (
    cpu_state_dict,
    result_authorization,
    routed_evaluation,
    signs_for,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
    normalized_expert_loss,
    rademacher,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import LayerState
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)
from latent_weight_lab.block_fht import normalized_fht_last_dim


PLAN_SCHEMA = "nanogpt_sparse_moe_state_conditioned_butterfly_transport_oracle_plan_v1"


def angle_count_per_side_expert(width: int) -> int:
    if width != 768:
        raise ValueError("registered mixed-radix flow requires width 768")
    return 3 * (256 // 2) * 8 + 3 * 256


def coordinate_count(*, experts: int, input_width: int, hidden_width: int) -> int:
    atom = experts * (2 * hidden_width + input_width)
    angles = 2 * experts * angle_count_per_side_expert(input_width)
    return atom + angles


def _gelu_derivative(values: torch.Tensor) -> torch.Tensor:
    return (
        0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))
        + values * torch.exp(-0.5 * values.square())
        / math.sqrt(2.0 * math.pi)
    )


def _givens_with_jvp(
    left: torch.Tensor,
    right: torch.Tensor,
    left_jvp: torch.Tensor,
    right_jvp: torch.Tensor,
    theta: torch.Tensor,
    theta_jvp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cosine, sine = torch.cos(theta), torch.sin(theta)
    rotated_left = cosine * left - sine * right
    rotated_right = sine * left + cosine * right
    rotated_left_jvp = (
        cosine * left_jvp - sine * right_jvp - theta_jvp * rotated_right
    )
    rotated_right_jvp = (
        sine * left_jvp + cosine * right_jvp + theta_jvp * rotated_left
    )
    return rotated_left, rotated_right, rotated_left_jvp, rotated_right_jvp


def _mixed_radix_flow_with_jvp(
    values: torch.Tensor,
    tangents: torch.Tensor,
    binary_angles: torch.Tensor,
    binary_angle_jvp: torch.Tensor,
    cross_angles: torch.Tensor,
    cross_angle_jvp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an exact 768=3*256 orthogonal flow and its input JVP."""
    if values.shape != tangents.shape or values.shape[-1] != 768:
        raise ValueError("mixed-radix value/tangent shape mismatch")
    expected_binary = (*values.shape[:-1], 8, 384)
    expected_cross = (*values.shape[:-1], 3, 256)
    if binary_angles.shape != expected_binary or binary_angle_jvp.shape != expected_binary:
        raise ValueError("binary angle field shape mismatch")
    if cross_angles.shape != expected_cross or cross_angle_jvp.shape != expected_cross:
        raise ValueError("cross angle field shape mismatch")

    result = values.reshape(*values.shape[:-1], 3, 256)
    result_jvp = tangents.reshape_as(result)
    for stage in range(8):
        half = 1 << stage
        block = 2 * half
        groups = 256 // block
        blocked = result.reshape(*result.shape[:-2], 3, groups, block)
        blocked_jvp = result_jvp.reshape_as(blocked)
        left, right = blocked[..., :half], blocked[..., half:]
        left_jvp, right_jvp = blocked_jvp[..., :half], blocked_jvp[..., half:]
        theta = binary_angles[..., stage, :].reshape(
            *values.shape[:-1], 3, groups, half
        )
        theta_jvp = binary_angle_jvp[..., stage, :].reshape_as(theta)
        left, right, left_jvp, right_jvp = _givens_with_jvp(
            left, right, left_jvp, right_jvp, theta, theta_jvp
        )
        result = torch.cat((left, right), dim=-1).reshape_as(result)
        result_jvp = torch.cat((left_jvp, right_jvp), dim=-1).reshape_as(result_jvp)

    pairs = ((0, 1), (1, 2), (0, 2))
    for stage, (first, second) in enumerate(pairs):
        left, right = result[..., first, :], result[..., second, :]
        left_jvp, right_jvp = result_jvp[..., first, :], result_jvp[..., second, :]
        left, right, left_jvp, right_jvp = _givens_with_jvp(
            left,
            right,
            left_jvp,
            right_jvp,
            cross_angles[..., stage, :],
            cross_angle_jvp[..., stage, :],
        )
        updated = result.clone()
        updated_jvp = result_jvp.clone()
        updated[..., first, :], updated[..., second, :] = left, right
        updated_jvp[..., first, :], updated_jvp[..., second, :] = left_jvp, right_jvp
        result, result_jvp = updated, updated_jvp
    return result.reshape_as(values), result_jvp.reshape_as(tangents)


class StateConditionedButterflyAtom(torch.nn.Module):
    """One complete procedural atom with input/output mixed-radix flows."""

    def __init__(
        self,
        *,
        experts: int,
        input_width: int,
        hidden_width: int,
        tensor_layers: int,
        seed: int,
        layer: int,
        device: str,
        conditional: bool,
        beta: float,
        feature_shift_scale: float,
        raw_angle_initial_tanh: float,
    ) -> None:
        super().__init__()
        self.experts = int(experts)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.tensor_layers = int(tensor_layers)
        self.conditional = bool(conditional)
        self.beta = float(beta) if conditional else 0.0
        self.feature_shift_scale = float(feature_shift_scale)
        if self.input_width != 768 or self.hidden_width != 1536:
            raise ValueError("registered flow requires 768/1536 MLP widths")
        if not 0.0 < float(raw_angle_initial_tanh) < 1.0:
            raise ValueError("raw angle tanh initialization must be in (0,1)")

        reference = torch.empty(1, device=device, dtype=torch.float32)
        layer_seed = int(seed) + 1009 * int(layer)
        self.register_buffer(
            "input_signs",
            torch.stack([
                signs_for(reference, expert, 0, layer_seed, 2048)
                for expert in range(self.experts)
            ]).float(),
        )
        self.register_buffer(
            "output_signs",
            torch.stack([
                signs_for(reference, expert, 1, layer_seed, 2048)
                for expert in range(self.experts)
            ]).float(),
        )
        self.register_buffer(
            "input_feature_signs",
            torch.stack([
                signs_for(reference, expert, 20, layer_seed, 1024)
                for expert in range(self.experts)
            ]).float(),
        )
        self.register_buffer(
            "output_feature_signs",
            torch.stack([
                signs_for(reference, expert, 40, layer_seed, 2048)
                for expert in range(self.experts)
            ]).float(),
        )
        input_binary, input_cross = self._feature_indices(layer_seed + 61, 1024)
        output_binary, output_cross = self._feature_indices(layer_seed + 79, 2048)
        self.register_buffer("input_binary_indices", input_binary)
        self.register_buffer("input_cross_indices", input_cross)
        self.register_buffer("output_binary_indices", output_binary)
        self.register_buffer("output_cross_indices", output_cross)

        self.hidden_gain_delta = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width)
        )
        self.output_gain_delta = torch.nn.Parameter(
            torch.zeros(self.experts, self.input_width)
        )
        self.hidden_bias = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width)
        )
        initial_raw = math.atanh(float(raw_angle_initial_tanh))
        self.input_binary_raw = torch.nn.Parameter(
            torch.full((self.experts, 8, 384), initial_raw)
        )
        self.input_cross_raw = torch.nn.Parameter(
            torch.full((self.experts, 3, 256), initial_raw)
        )
        self.output_binary_raw = torch.nn.Parameter(
            torch.full((self.experts, 8, 384), initial_raw)
        )
        self.output_cross_raw = torch.nn.Parameter(
            torch.full((self.experts, 3, 256), initial_raw)
        )
        self.c_fc_scale = math.sqrt(2048.0) * 0.02
        self.c_proj_scale = math.sqrt(2048.0) * 0.02 / math.sqrt(
            2.0 * float(self.tensor_layers)
        )
        self.to(device=device, dtype=torch.float32)

    @staticmethod
    def _feature_indices(seed: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        binary = torch.stack([
            torch.randperm(width, generator=generator)[:384] for _ in range(8)
        ])
        cross = torch.stack([
            torch.randperm(width, generator=generator)[:256] for _ in range(3)
        ])
        return binary.long(), cross.long()

    def _selection(self, expert: int | None) -> slice:
        if expert is None:
            return slice(None)
        if not 0 <= int(expert) < self.experts:
            raise IndexError("expert index out of range")
        return slice(int(expert), int(expert) + 1)

    def trainable_parameters(self, *, conditional: bool) -> list[torch.nn.Parameter]:
        if bool(conditional) != self.conditional:
            raise ValueError("conditional flag disagrees with constructed module")
        return [
            self.hidden_gain_delta,
            self.output_gain_delta,
            self.hidden_bias,
            self.input_binary_raw,
            self.input_cross_raw,
            self.output_binary_raw,
            self.output_cross_raw,
        ]

    def compact_parameter_count(self, *, conditional: bool) -> int:
        return sum(
            parameter.numel()
            for parameter in self.trainable_parameters(conditional=conditional)
        )

    def _angle_field(
        self,
        state: torch.Tensor,
        state_jvp: torch.Tensor,
        *,
        selected: slice,
        side: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if side == "input":
            padded_width = 1024
            signs = self.input_feature_signs[selected]
            binary_indices, cross_indices = (
                self.input_binary_indices,
                self.input_cross_indices,
            )
            binary_raw, cross_raw = (
                self.input_binary_raw[selected],
                self.input_cross_raw[selected],
            )
        elif side == "output":
            padded_width = 2048
            signs = self.output_feature_signs[selected]
            binary_indices, cross_indices = (
                self.output_binary_indices,
                self.output_cross_indices,
            )
            binary_raw, cross_raw = (
                self.output_binary_raw[selected],
                self.output_cross_raw[selected],
            )
        else:
            raise ValueError("unknown flow side")
        padded = F.pad(state, (0, padded_width - state.shape[-1]))
        padded_jvp = F.pad(state_jvp, (0, padded_width - state.shape[-1]))
        scale = math.sqrt(float(padded_width) / float(state.shape[-1]))
        features = normalized_fht_last_dim(padded * signs[:, None, :]) * scale
        features_jvp = normalized_fht_last_dim(
            padded_jvp * signs[:, None, :]
        ) * scale
        binary_feature = features[..., binary_indices]
        binary_feature_jvp = features_jvp[..., binary_indices]
        cross_feature = features[..., cross_indices]
        cross_feature_jvp = features_jvp[..., cross_indices]

        def angles(
            raw: torch.Tensor,
            feature: torch.Tensor,
            feature_jvp: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            squashed_feature = torch.tanh(feature)
            squashed_feature_jvp = (1.0 - squashed_feature.square()) * feature_jvp
            inside = raw[:, None, ...] + (
                self.feature_shift_scale * self.beta * squashed_feature
            )
            tanh_inside = torch.tanh(inside)
            theta = (math.pi / 2.0) * tanh_inside
            theta_jvp = (
                (math.pi / 2.0)
                * (1.0 - tanh_inside.square())
                * self.feature_shift_scale
                * self.beta
                * squashed_feature_jvp
            )
            return theta, theta_jvp

        binary_theta, binary_theta_jvp = angles(
            binary_raw, binary_feature, binary_feature_jvp
        )
        cross_theta, cross_theta_jvp = angles(
            cross_raw, cross_feature, cross_feature_jvp
        )
        return binary_theta, binary_theta_jvp, cross_theta, cross_theta_jvp

    def _transport(
        self,
        value: torch.Tensor,
        value_jvp: torch.Tensor,
        *,
        state: torch.Tensor,
        state_jvp: torch.Tensor,
        selected: slice,
        side: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angle_field = self._angle_field(
            state, state_jvp, selected=selected, side=side
        )
        return _mixed_radix_flow_with_jvp(value, value_jvp, *angle_field)

    def function_and_jvp(
        self,
        inputs: torch.Tensor,
        directions: torch.Tensor,
        *,
        conditional: bool,
        expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(conditional) != self.conditional:
            raise ValueError("conditional flag disagrees with constructed module")
        if inputs.shape != directions.shape:
            raise ValueError("input and direction shapes disagree")
        selected = self._selection(expert)
        expected_experts = self.experts if expert is None else 1
        if inputs.shape[0] != expected_experts or inputs.shape[-1] != self.input_width:
            raise ValueError("input shape disagrees with butterfly operator")
        original, original_jvp = inputs.float(), directions.float()
        transported, transported_jvp = self._transport(
            original,
            original_jvp,
            state=original,
            state_jvp=original_jvp,
            selected=selected,
            side="input",
        )
        padded = F.pad(transported, (0, 2048 - self.input_width))
        padded_jvp = F.pad(transported_jvp, (0, 2048 - self.input_width))
        pre = normalized_fht_last_dim(
            padded * self.input_signs[selected, None, :]
        )[..., : self.hidden_width]
        pre_jvp = normalized_fht_last_dim(
            padded_jvp * self.input_signs[selected, None, :]
        )[..., : self.hidden_width]
        hidden_gain = 1.0 + self.hidden_gain_delta[selected, None, :]
        pre = self.c_fc_scale * pre * hidden_gain + self.hidden_bias[selected, None, :]
        pre_jvp = self.c_fc_scale * pre_jvp * hidden_gain
        hidden = F.gelu(pre)
        hidden_jvp = _gelu_derivative(pre) * pre_jvp
        hidden_padded = F.pad(hidden, (0, 2048 - self.hidden_width))
        hidden_jvp_padded = F.pad(hidden_jvp, (0, 2048 - self.hidden_width))
        output = normalized_fht_last_dim(
            hidden_padded * self.output_signs[selected, None, :]
        )[..., : self.input_width]
        output_jvp = normalized_fht_last_dim(
            hidden_jvp_padded * self.output_signs[selected, None, :]
        )[..., : self.input_width]
        output_gain = 1.0 + self.output_gain_delta[selected, None, :]
        output = self.c_proj_scale * output * output_gain
        output_jvp = self.c_proj_scale * output_jvp * output_gain
        return self._transport(
            output,
            output_jvp,
            state=hidden,
            state_jvp=hidden_jvp,
            selected=selected,
            side="output",
        )


def fit_atom(
    module: StateConditionedButterflyAtom,
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
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        output, output_jvp = module.function_and_jvp(
            live_inputs, directions, conditional=conditional
        )
        output_loss = normalized_expert_loss(output, target_output)
        jvp_loss = normalized_expert_loss(output_jvp, target_jvp)
        loss = output_loss + float(jvp_weight) * jvp_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite state-conditioned butterfly objective")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise RuntimeError("non-finite or missing butterfly gradient")
        gradient = float(
            torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
        )
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
        output_losses.append(float(output_loss.detach()))
        jvp_losses.append(float(jvp_loss.detach()))
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
        "input_angle_rms": float(
            ((math.pi / 2.0) * torch.tanh(module.input_binary_raw.detach()))
            .square().mean().sqrt()
        ),
        "output_angle_rms": float(
            ((math.pi / 2.0) * torch.tanh(module.output_binary_raw.detach()))
            .square().mean().sqrt()
        ),
    }


def make_module(
    plan: dict[str, Any], layer: int, device: str, *, conditional: bool
) -> StateConditionedButterflyAtom:
    source, candidate = plan["source"], plan["candidate"]
    return StateConditionedButterflyAtom(
        experts=int(source["num_experts"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        tensor_layers=int(source["tensor_layers"]),
        seed=int(candidate["fixed_feature_seed"]),
        layer=int(layer),
        device=device,
        conditional=conditional,
        beta=float(candidate["beta"]),
        feature_shift_scale=float(candidate["feature_shift_scale"]),
        raw_angle_initial_tanh=float(candidate["raw_angle_initial_tanh"]),
    )


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("state-conditioned butterfly plan schema mismatch")
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
    source, candidate = plan["source"], plan["candidate"]
    expected = coordinate_count(
        experts=int(source["num_experts"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
    )
    if expected != int(candidate["total_coordinates_per_layer"]):
        raise ValueError("candidate coordinate accounting drift")
    if expected != int(plan["same_run_control"]["total_coordinates_per_layer"]):
        raise ValueError("control coordinate accounting drift")
    ratio = float(candidate["dense_paired_parameters_per_layer"]) / float(expected)
    if abs(ratio - float(candidate["paired_parameter_compression_ratio"])) > 1e-12:
        raise ValueError("candidate compression ratio drift")
    if ratio < 200.0:
        raise ValueError("candidate is outside compression budget")
    if not file_sha256(plan_path):
        raise AssertionError("empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    candidate = make_module(plan, 0, device, conditional=True)
    control = make_module(plan, 0, device, conditional=False)
    generator = torch.Generator(device="cpu").manual_seed(20261190)
    shape = (int(source["num_experts"]), 16, int(source["input_width"]))
    inputs = torch.randn(shape, generator=generator)
    c_fc = torch.randn(
        int(source["num_experts"]),
        int(source["expert_hidden_width"]),
        int(source["input_width"]),
        generator=generator,
    ) * 0.02
    c_proj = torch.randn(
        int(source["num_experts"]),
        int(source["input_width"]),
        int(source["expert_hidden_width"]),
        generator=generator,
    ) * (0.02 / math.sqrt(2.0 * int(source["tensor_layers"])))
    fit = plan["fit_protocol"]
    started = time.time()
    candidate_diag = fit_atom(
        candidate,
        inputs,
        c_fc,
        c_proj,
        conditional=True,
        steps=2,
        learning_rate=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
        jvp_weight=float(fit["jvp_weight"]),
        probe_seed=20261191,
    )
    control_diag = fit_atom(
        control,
        inputs,
        c_fc,
        c_proj,
        conditional=False,
        steps=2,
        learning_rate=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
        jvp_weight=float(fit["jvp_weight"]),
        probe_seed=20261191,
    )
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_state_conditioned_butterfly_transport_preflight_v1",
        "device": device,
        "two_step_wall_seconds_candidate_plus_control": elapsed,
        "projected_full_protocol_seconds": elapsed * (int(fit["steps"]) / 2.0) * 6.0,
        "candidate_coordinate_count": candidate.compact_parameter_count(conditional=True),
        "control_coordinate_count": control.compact_parameter_count(conditional=False),
        "maximum_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
        "all_values_finite": all_finite({"candidate": candidate_diag, "control": control_diag}),
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
    states: dict[int, LayerState] = {
        layer: layer_state_from_mapping(mapping, layer) for layer in layers
    }
    del mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    root = Path(__file__).resolve().parents[2]
    biplane = json.loads(
        (root / plan["sealed_controls"]["coupled_biplane_result"]["path"])
        .read_text()
    )
    fit = plan["fit_protocol"]
    banks = [row["name"] for row in plan["data_protocol"]["discovery_banks"]]
    samples_per_expert = int(plan["data_protocol"]["fit_samples_per_expert"])
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
                state,
                inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=samples_per_expert,
                seed=20261192 + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            candidate = make_module(plan, layer, args.device, conditional=True)
            control = make_module(plan, layer, args.device, conditional=False)
            common_fit = {
                "steps": int(fit["steps"]),
                "learning_rate": float(fit["learning_rate"]),
                "weight_decay": float(fit["weight_decay"]),
                "gradient_clip": float(fit["gradient_clip"]),
                "jvp_weight": float(fit["jvp_weight"]),
                "probe_seed": 20261193 + 1009 * bank_index + 17 * layer,
            }
            candidate_diag = fit_atom(
                candidate,
                sampled,
                state.c_fc,
                state.c_proj,
                conditional=True,
                **common_fit,
            )
            control_diag = fit_atom(
                control,
                sampled,
                state.c_fc,
                state.c_proj,
                conditional=False,
                **common_fit,
            )
            candidate_eval = routed_evaluation(
                state,
                inputs["heldout"][layer],
                candidate,
                conditional=True,
                outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=20261194 + 17 * layer,
            )
            control_eval = routed_evaluation(
                state,
                inputs["heldout"][layer],
                control,
                conditional=False,
                outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=20261194 + 17 * layer,
            )
            if not torch.equal(candidate_eval["target"], control_eval["target"]):
                raise RuntimeError("candidate and control target drift")
            actions[(bank, layer)] = candidate_eval["predicted"]
            sealed_layer = float(
                biplane["summaries"][bank][str(layer)]["mixture_recovery"]
            )
            summaries[bank][str(layer)] = {
                "mixture_recovery": candidate_eval["mixture_recovery"],
                "jvp_recovery": candidate_eval["jvp_recovery"],
                "minimum_expert_recovery": min(candidate_eval["expert_recovery"]),
                "minimum_expert_jvp_recovery": min(candidate_eval["expert_jvp_recovery"]),
                "static_control_recovery": control_eval["mixture_recovery"],
                "candidate_minus_static_control_recovery": candidate_eval["mixture_recovery"] - control_eval["mixture_recovery"],
                "sealed_biplane_recovery": sealed_layer,
                "candidate_minus_sealed_biplane_recovery": candidate_eval["mixture_recovery"] - sealed_layer,
            }
            diagnostics[bank][str(layer)] = {
                "candidate": candidate_diag,
                "control": control_diag,
            }
            saved[bank][str(layer)] = {
                "candidate": cpu_state_dict(candidate),
                "control": cpu_state_dict(control),
            }
            del candidate, control
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    frozen = plan["frozen_gates"]
    gates: dict[str, dict[str, bool]] = {}
    for bank in banks:
        rows = [summaries[bank][str(layer)] for layer in layers]
        aggregate = {
            "mixture_recovery_mean": sum(float(row["mixture_recovery"]) for row in rows) / len(rows),
            "mixture_recovery_minimum_layer": min(float(row["mixture_recovery"]) for row in rows),
            "jvp_recovery_mean": sum(float(row["jvp_recovery"]) for row in rows) / len(rows),
            "minimum_expert_recovery": min(float(row["minimum_expert_recovery"]) for row in rows),
            "candidate_minus_static_control_recovery_mean": sum(float(row["candidate_minus_static_control_recovery"]) for row in rows) / len(rows),
            "candidate_minus_sealed_biplane_recovery_mean": sum(float(row["candidate_minus_sealed_biplane_recovery"]) for row in rows) / len(rows),
            "minimum_discovery_assignments": min(min(occupancy[bank][str(layer)]) for layer in layers),
        }
        summaries[bank]["aggregate"] = aggregate
        gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": aggregate["minimum_expert_recovery"] >= float(frozen["heldout_expert_recovery_min_each_bank"]),
            "jvp_pass": aggregate["jvp_recovery_mean"] >= float(frozen["heldout_jvp_recovery_mean_min_each_bank"]),
            "static_control_gain_pass": aggregate["candidate_minus_static_control_recovery_mean"] >= float(frozen["candidate_minus_static_control_recovery_mean_min_each_bank"]),
            "sealed_biplane_gain_pass": aggregate["candidate_minus_sealed_biplane_recovery_mean"] >= float(frozen["candidate_minus_sealed_biplane_recovery_mean_min_each_bank"]),
            "occupancy_pass": aggregate["minimum_discovery_assignments"] >= int(frozen["minimum_discovery_assignments_per_expert"]),
        }
    agreement_by_layer = {
        str(layer): action_cosine(actions[(banks[0], layer)], actions[(banks[1], layer)])
        for layer in layers
    }
    agreement_mean = sum(agreement_by_layer.values()) / len(agreement_by_layer)
    finite = all_finite({
        "summaries": summaries,
        "diagnostics": diagnostics,
        "agreement": agreement_by_layer,
    })
    for bank in banks:
        gates[bank]["action_agreement_pass"] = agreement_mean >= float(frozen["heldout_bank_action_cosine_mean_min"])
        gates[bank]["finite_pass"] = finite
        gates[bank]["all_pass"] = all(gates[bank].values())
    passed = all(gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_state_conditioned_butterfly_transport_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    result = {
        "schema_version": "nanogpt_sparse_moe_state_conditioned_butterfly_transport_oracle_result_v1",
        "classification": "STATE_CONDITIONED_BUTTERFLY_TRANSPORT_REPRESENTABILITY_PASSES" if passed else "STATE_CONDITIONED_BUTTERFLY_TRANSPORT_REPRESENTABILITY_REJECTED",
        "passed": passed,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(manifest),
            "coupled_biplane_result_sha256": file_sha256(root / plan["sealed_controls"]["coupled_biplane_result"]["path"]),
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
            "state_conditioned_full_rank_butterfly_transport": True,
        },
        "occupancy": occupancy,
        "fit_diagnostics": diagnostics,
        "summaries": summaries,
        "heldout_bank_action_cosine": {
            "mean": agreement_mean,
            "by_layer": agreement_by_layer,
        },
        "gates": gates,
        "all_values_finite": finite,
        "authorization": result_authorization(passed),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
