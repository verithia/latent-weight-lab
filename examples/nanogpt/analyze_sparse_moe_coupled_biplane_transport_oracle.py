#!/usr/bin/env python3
"""Gate coupled state-conditioned biplane transport for complete MLP replacement."""
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


PLAN_SCHEMA = "nanogpt_sparse_moe_coupled_biplane_transport_oracle_plan_v1"


def coordinate_count(
    *, experts: int, planes: int, input_width: int, hidden_width: int,
    conditional: bool,
) -> int:
    static = experts * hidden_width + experts * input_width + experts * hidden_width
    if not conditional:
        return static
    plane_vectors = 2 * experts * planes * 2 * input_width
    angles = 2 * experts * planes
    return static + plane_vectors + angles


def _normalize_with_jvp(
    value: torch.Tensor, tangent: torch.Tensor, eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = value.square().sum(dim=-1, keepdim=True).add(eps * eps).sqrt()
    unit = value / norm
    unit_tangent = (
        tangent - unit * (unit * tangent).sum(dim=-1, keepdim=True)
    ) / norm
    return unit, unit_tangent


def _orthonormal_plane_with_jvp(
    first: torch.Tensor, first_jvp: torch.Tensor,
    second: torch.Tensor, second_jvp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    v, v_jvp = _normalize_with_jvp(first, first_jvp)
    projection = (v * second).sum(dim=-1, keepdim=True)
    projection_jvp = (
        (v_jvp * second).sum(dim=-1, keepdim=True)
        + (v * second_jvp).sum(dim=-1, keepdim=True)
    )
    orthogonal = second - v * projection
    orthogonal_jvp = second_jvp - v_jvp * projection - v * projection_jvp
    w, w_jvp = _normalize_with_jvp(orthogonal, orthogonal_jvp)
    return v, v_jvp, w, w_jvp


def _rotate_with_jvp(
    value: torch.Tensor, tangent: torch.Tensor,
    v: torch.Tensor, v_jvp: torch.Tensor,
    w: torch.Tensor, w_jvp: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = torch.cos(theta)[..., None, None]
    sine = torch.sin(theta)[..., None, None]
    along_v = (value * v).sum(dim=-1, keepdim=True)
    along_w = (value * w).sum(dim=-1, keepdim=True)
    along_v_jvp = (
        (tangent * v).sum(dim=-1, keepdim=True)
        + (value * v_jvp).sum(dim=-1, keepdim=True)
    )
    along_w_jvp = (
        (tangent * w).sum(dim=-1, keepdim=True)
        + (value * w_jvp).sum(dim=-1, keepdim=True)
    )
    coefficient_v = (cosine - 1.0) * along_v - sine * along_w
    coefficient_w = sine * along_v + (cosine - 1.0) * along_w
    coefficient_v_jvp = (
        (cosine - 1.0) * along_v_jvp - sine * along_w_jvp
    )
    coefficient_w_jvp = (
        sine * along_v_jvp + (cosine - 1.0) * along_w_jvp
    )
    rotated = value + coefficient_v * v + coefficient_w * w
    rotated_jvp = (
        tangent
        + coefficient_v_jvp * v
        + coefficient_v * v_jvp
        + coefficient_w_jvp * w
        + coefficient_w * w_jvp
    )
    return rotated, rotated_jvp


class CoupledBiplaneTransportAtom(torch.nn.Module):
    """One complete procedural atom with optional moving input/output planes."""

    def __init__(
        self, *, experts: int, planes: int, input_width: int,
        hidden_width: int, padded_width: int, tensor_layers: int,
        seed: int, layer: int, device: str, conditional: bool,
    ) -> None:
        super().__init__()
        self.experts = int(experts)
        self.planes = int(planes)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.padded_width = int(padded_width)
        self.conditional = bool(conditional)
        if self.padded_width & (self.padded_width - 1):
            raise ValueError("padded width must be a power of two")
        if self.padded_width < max(self.input_width, self.hidden_width):
            raise ValueError("padded width does not cover both axes")

        reference = torch.empty(1, device=device, dtype=torch.float32)
        layer_seed = int(seed) + 1009 * int(layer)
        self.register_buffer(
            "input_signs",
            torch.stack([
                signs_for(reference, expert, 0, layer_seed, self.padded_width)
                for expert in range(self.experts)
            ]).float(),
        )
        self.register_buffer(
            "output_signs",
            torch.stack([
                signs_for(reference, expert, 1, layer_seed, self.padded_width)
                for expert in range(self.experts)
            ]).float(),
        )
        self.hidden_gain_delta = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width)
        )
        self.output_gain_delta = torch.nn.Parameter(
            torch.zeros(self.experts, self.input_width)
        )
        self.hidden_bias = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width)
        )
        if self.conditional:
            input_plane_signs = []
            output_plane_signs = []
            for expert in range(self.experts):
                input_plane_signs.append(torch.stack([
                    torch.stack([
                        signs_for(
                            reference, expert, 10 + 4 * plane + axis,
                            layer_seed, self.padded_width,
                        )
                        for axis in range(2)
                    ])
                    for plane in range(self.planes)
                ]))
                output_plane_signs.append(torch.stack([
                    torch.stack([
                        signs_for(
                            reference, expert, 100 + 4 * plane + axis,
                            layer_seed, self.padded_width,
                        )
                        for axis in range(2)
                    ])
                    for plane in range(self.planes)
                ]))
            self.register_buffer(
                "input_plane_signs", torch.stack(input_plane_signs).float()
            )
            self.register_buffer(
                "output_plane_signs", torch.stack(output_plane_signs).float()
            )
            self.input_plane_gain_delta = torch.nn.Parameter(
                torch.zeros(self.experts, self.planes, 2, self.input_width)
            )
            self.output_plane_gain_delta = torch.nn.Parameter(
                torch.zeros(self.experts, self.planes, 2, self.input_width)
            )
            initial_angle = math.atanh(0.25)
            self.input_angle_raw = torch.nn.Parameter(
                torch.full((self.experts, self.planes), initial_angle)
            )
            self.output_angle_raw = torch.nn.Parameter(
                torch.full((self.experts, self.planes), initial_angle)
            )
        else:
            self.register_buffer("input_plane_signs", None)
            self.register_buffer("output_plane_signs", None)
            self.register_parameter("input_plane_gain_delta", None)
            self.register_parameter("output_plane_gain_delta", None)
            self.register_parameter("input_angle_raw", None)
            self.register_parameter("output_angle_raw", None)

        self.c_fc_scale = math.sqrt(float(self.padded_width)) * 0.02
        self.c_proj_scale = (
            math.sqrt(float(self.padded_width)) * 0.02
            / math.sqrt(2.0 * float(tensor_layers))
        )
        self.feature_scale = math.sqrt(
            float(self.padded_width) / float(self.input_width)
        )
        self.to(device=device, dtype=torch.float32)

    def _selection(self, expert: int | None) -> slice:
        if expert is None:
            return slice(None)
        if not 0 <= int(expert) < self.experts:
            raise IndexError("expert index out of range")
        return slice(int(expert), int(expert) + 1)

    def trainable_parameters(self, *, conditional: bool) -> list[torch.nn.Parameter]:
        if bool(conditional) != self.conditional:
            raise ValueError("conditional flag disagrees with constructed module")
        result = [self.hidden_gain_delta, self.output_gain_delta, self.hidden_bias]
        if self.conditional:
            result.extend([
                self.input_plane_gain_delta,
                self.output_plane_gain_delta,
                self.input_angle_raw,
                self.output_angle_raw,
            ])
        return result

    def compact_parameter_count(self, *, conditional: bool) -> int:
        return sum(
            parameter.numel()
            for parameter in self.trainable_parameters(conditional=conditional)
        )

    def _plane_features(
        self, values: torch.Tensor, tangent: torch.Tensor, *,
        selected: slice, side: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        signs = (
            self.input_plane_signs[selected]
            if side == "input" else self.output_plane_signs[selected]
        )
        gains = (
            self.input_plane_gain_delta[selected]
            if side == "input" else self.output_plane_gain_delta[selected]
        )
        padded = F.pad(values, (0, self.padded_width - self.input_width))
        padded_jvp = F.pad(tangent, (0, self.padded_width - self.input_width))
        raw = normalized_fht_last_dim(
            padded[:, None, None, :, :] * signs[:, :, :, None, :]
        )[..., : self.input_width] * self.feature_scale
        raw_jvp = normalized_fht_last_dim(
            padded_jvp[:, None, None, :, :] * signs[:, :, :, None, :]
        )[..., : self.input_width] * self.feature_scale
        base = torch.tanh(raw)
        base_jvp = (1.0 - base.square()) * raw_jvp
        feature = base * (1.0 + gains[:, :, :, None, :])
        feature_jvp = base_jvp * (1.0 + gains[:, :, :, None, :])
        return _orthonormal_plane_with_jvp(
            feature[:, :, 0], feature_jvp[:, :, 0],
            feature[:, :, 1], feature_jvp[:, :, 1],
        )

    def _transport(
        self, value: torch.Tensor, tangent: torch.Tensor, *,
        original: torch.Tensor, original_jvp: torch.Tensor,
        selected: slice, side: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v, v_jvp, w, w_jvp = self._plane_features(
            original, original_jvp, selected=selected, side=side
        )
        raw_angles = (
            self.input_angle_raw[selected]
            if side == "input" else self.output_angle_raw[selected]
        )
        angles = (math.pi / 2.0) * torch.tanh(raw_angles)
        result, result_jvp = value, tangent
        for plane in range(self.planes):
            result, result_jvp = _rotate_with_jvp(
                result, result_jvp,
                v[:, plane], v_jvp[:, plane],
                w[:, plane], w_jvp[:, plane], angles[:, plane],
            )
        return result, result_jvp

    def function_and_jvp(
        self, inputs: torch.Tensor, directions: torch.Tensor, *,
        conditional: bool, expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(conditional) != self.conditional:
            raise ValueError("conditional flag disagrees with constructed module")
        if inputs.shape != directions.shape:
            raise ValueError("input and direction shapes disagree")
        selected = self._selection(expert)
        expected_experts = self.experts if expert is None else 1
        if inputs.shape[0] != expected_experts or inputs.shape[-1] != self.input_width:
            raise ValueError("input shape disagrees with biplane operator")
        original = inputs.float()
        original_jvp = directions.float()
        transported, transported_jvp = original, original_jvp
        if self.conditional:
            transported, transported_jvp = self._transport(
                transported, transported_jvp,
                original=original, original_jvp=original_jvp,
                selected=selected, side="input",
            )
        padded = F.pad(
            transported, (0, self.padded_width - self.input_width)
        )
        padded_jvp = F.pad(
            transported_jvp, (0, self.padded_width - self.input_width)
        )
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
        hidden_jvp = (
            0.5 * (1.0 + torch.erf(pre / math.sqrt(2.0)))
            + pre * torch.exp(-0.5 * pre.square()) / math.sqrt(2.0 * math.pi)
        ) * pre_jvp
        hidden = F.pad(hidden, (0, self.padded_width - self.hidden_width))
        hidden_jvp = F.pad(
            hidden_jvp, (0, self.padded_width - self.hidden_width)
        )
        output = normalized_fht_last_dim(
            hidden * self.output_signs[selected, None, :]
        )[..., : self.input_width]
        output_jvp = normalized_fht_last_dim(
            hidden_jvp * self.output_signs[selected, None, :]
        )[..., : self.input_width]
        output_gain = 1.0 + self.output_gain_delta[selected, None, :]
        output = self.c_proj_scale * output * output_gain
        output_jvp = self.c_proj_scale * output_jvp * output_gain
        if self.conditional:
            output, output_jvp = self._transport(
                output, output_jvp,
                original=original, original_jvp=original_jvp,
                selected=selected, side="output",
            )
        return output, output_jvp


def fit_atom(
    module: CoupledBiplaneTransportAtom,
    inputs: torch.Tensor,
    dense_c_fc: torch.Tensor,
    dense_c_proj: torch.Tensor,
    *, conditional: bool, steps: int, learning_rate: float,
    weight_decay: float, gradient_clip: float, jvp_weight: float,
    probe_seed: int,
) -> dict[str, Any]:
    device = str(module.hidden_bias.device)
    live_inputs = inputs.to(device=device, dtype=torch.float32)
    directions = rademacher(tuple(live_inputs.shape), probe_seed, device)
    with torch.no_grad():
        target_output, target_jvp = dense_function_and_jvp(
            live_inputs, directions,
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
            raise RuntimeError("non-finite biplane objective")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise RuntimeError("non-finite or missing biplane gradient")
        gradient = float(
            torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
        )
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
        output_losses.append(float(output_loss.detach()))
        jvp_losses.append(float(jvp_loss.detach()))
    diagnostics: dict[str, Any] = {
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
    }
    if conditional:
        diagnostics.update({
            "input_plane_gain_delta_rms": float(
                module.input_plane_gain_delta.detach().square().mean().sqrt()
            ),
            "output_plane_gain_delta_rms": float(
                module.output_plane_gain_delta.detach().square().mean().sqrt()
            ),
            "input_angle_rms": float(
                ((math.pi / 2.0) * torch.tanh(module.input_angle_raw.detach()))
                .square().mean().sqrt()
            ),
            "output_angle_rms": float(
                ((math.pi / 2.0) * torch.tanh(module.output_angle_raw.detach()))
                .square().mean().sqrt()
            ),
        })
    return diagnostics


def make_module(
    plan: dict[str, Any], layer: int, device: str, *, conditional: bool,
) -> CoupledBiplaneTransportAtom:
    source, candidate = plan["source"], plan["candidate"]
    return CoupledBiplaneTransportAtom(
        experts=int(source["num_experts"]),
        planes=int(candidate["planes_per_side"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        padded_width=int(candidate["padded_width"]),
        tensor_layers=int(source["tensor_layers"]),
        seed=int(candidate["fixed_operator_seed"]),
        layer=int(layer), device=device, conditional=conditional,
    )


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("coupled-biplane plan schema mismatch")
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
    common = {
        "experts": int(source["num_experts"]),
        "planes": int(candidate["planes_per_side"]),
        "input_width": int(source["input_width"]),
        "hidden_width": int(source["expert_hidden_width"]),
    }
    if coordinate_count(**common, conditional=True) != int(
        candidate["total_coordinates_per_layer"]
    ):
        raise ValueError("candidate accounting drift")
    if coordinate_count(**common, conditional=False) != int(
        plan["same_run_control"]["total_coordinates_per_layer"]
    ):
        raise ValueError("control accounting drift")
    expected_ratio = (
        float(candidate["dense_paired_parameters_per_layer"])
        / float(candidate["total_coordinates_per_layer"])
    )
    if abs(expected_ratio - float(candidate["paired_parameter_compression_ratio"])) > 1e-12:
        raise ValueError("candidate compression ratio drift")
    if expected_ratio < 200.0:
        raise ValueError("candidate is outside compression budget")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    candidate = make_module(plan, 0, device, conditional=True)
    control = make_module(plan, 0, device, conditional=False)
    generator = torch.Generator(device="cpu").manual_seed(20261172)
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
    candidate_diag = fit_atom(
        candidate, inputs, c_fc, c_proj, conditional=True, steps=2,
        learning_rate=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
        jvp_weight=float(fit["jvp_weight"]), probe_seed=20261173,
    )
    control_diag = fit_atom(
        control, inputs, c_fc, c_proj, conditional=False, steps=2,
        learning_rate=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
        jvp_weight=float(fit["jvp_weight"]), probe_seed=20261173,
    )
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_coupled_biplane_transport_preflight_v1",
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
    bilateral = json.loads(
        (root / plan["sealed_controls"]["bilateral_coordinate_result"]["path"])
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
                state, inputs[bank][layer], top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=samples_per_expert,
                seed=20261174 + 1009 * bank_index + 17 * layer,
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
                "probe_seed": 20261175 + 1009 * bank_index + 17 * layer,
            }
            candidate_diag = fit_atom(
                candidate, sampled, state.c_fc, state.c_proj,
                conditional=True, **common_fit,
            )
            control_diag = fit_atom(
                control, sampled, state.c_fc, state.c_proj,
                conditional=False, **common_fit,
            )
            candidate_eval = routed_evaluation(
                state, inputs["heldout"][layer], candidate, conditional=True,
                outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=20261176 + 17 * layer,
            )
            control_eval = routed_evaluation(
                state, inputs["heldout"][layer], control, conditional=False,
                outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=20261176 + 17 * layer,
            )
            if not torch.equal(candidate_eval["target"], control_eval["target"]):
                raise RuntimeError("candidate and control target drift")
            actions[(bank, layer)] = candidate_eval["predicted"]
            sealed_layer = float(
                bilateral["summaries"][bank][str(layer)]["mixture_recovery"]
            )
            summaries[bank][str(layer)] = {
                "mixture_recovery": candidate_eval["mixture_recovery"],
                "jvp_recovery": candidate_eval["jvp_recovery"],
                "minimum_expert_recovery": min(candidate_eval["expert_recovery"]),
                "minimum_expert_jvp_recovery": min(candidate_eval["expert_jvp_recovery"]),
                "static_control_recovery": control_eval["mixture_recovery"],
                "candidate_minus_static_control_recovery": candidate_eval["mixture_recovery"] - control_eval["mixture_recovery"],
                "sealed_bilateral_recovery": sealed_layer,
                "candidate_minus_sealed_bilateral_recovery": candidate_eval["mixture_recovery"] - sealed_layer,
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
            "candidate_minus_static_control_recovery_mean": sum(float(row["candidate_minus_static_control_recovery"]) for row in rows) / len(rows),
            "candidate_minus_sealed_bilateral_recovery_mean": sum(float(row["candidate_minus_sealed_bilateral_recovery"]) for row in rows) / len(rows),
            "minimum_discovery_assignments": min(min(occupancy[bank][str(layer)]) for layer in layers),
        }
        summaries[bank]["aggregate"] = aggregate
        bank_gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": aggregate["minimum_expert_recovery"] >= float(frozen["heldout_expert_recovery_min_each_bank"]),
            "jvp_pass": aggregate["jvp_recovery_mean"] >= float(frozen["heldout_jvp_recovery_mean_min_each_bank"]),
            "static_control_gain_pass": aggregate["candidate_minus_static_control_recovery_mean"] >= float(frozen["candidate_minus_static_control_recovery_mean_min_each_bank"]),
            "sealed_bilateral_gain_pass": aggregate["candidate_minus_sealed_bilateral_recovery_mean"] >= float(frozen["candidate_minus_sealed_bilateral_recovery_mean_min_each_bank"]),
            "occupancy_pass": aggregate["minimum_discovery_assignments"] >= int(frozen["minimum_discovery_assignments_per_expert"]),
        }
    agreement_by_layer = {
        str(layer): action_cosine(actions[(banks[0], layer)], actions[(banks[1], layer)])
        for layer in layers
    }
    agreement_mean = sum(agreement_by_layer.values()) / len(agreement_by_layer)
    finite = all_finite({
        "summaries": summaries, "diagnostics": diagnostics,
        "agreement": agreement_by_layer,
    })
    for bank in banks:
        bank_gates[bank]["action_agreement_pass"] = agreement_mean >= float(frozen["heldout_bank_action_cosine_mean_min"])
        bank_gates[bank]["finite_pass"] = finite
        bank_gates[bank]["all_pass"] = all(bank_gates[bank].values())
    passed = all(bank_gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_coupled_biplane_transport_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    result = {
        "schema_version": "nanogpt_sparse_moe_coupled_biplane_transport_oracle_result_v1",
        "classification": "COUPLED_BIPLANE_TRANSPORT_REPRESENTABILITY_PASSES" if passed else "COUPLED_BIPLANE_TRANSPORT_REPRESENTABILITY_REJECTED",
        "passed": passed,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(manifest),
            "bilateral_coordinate_result_sha256": file_sha256(root / plan["sealed_controls"]["bilateral_coordinate_result"]["path"]),
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
            "coupled_state_conditioned_biplane_transport": True,
        },
        "occupancy": occupancy,
        "fit_diagnostics": diagnostics,
        "summaries": summaries,
        "heldout_bank_action_cosine": {
            "mean": agreement_mean, "by_layer": agreement_by_layer,
        },
        "gates": bank_gates,
        "all_values_finite": finite,
        "authorization": result_authorization(passed),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
