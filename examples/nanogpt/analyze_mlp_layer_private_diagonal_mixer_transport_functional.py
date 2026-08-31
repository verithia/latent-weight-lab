#!/usr/bin/env python3
"""Frozen H66 layer-private full-rank diagonal--mixer MLP audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    file_sha256,
    load_snapshot,
    model_from_snapshot,
)
from examples.nanogpt.analyze_mlp_hard_productcode_value_affine_jet_functional import (
    HardProductCodeValueAffineJet,
)
from examples.nanogpt.analyze_mlp_layer_private_householder_transport_functional import (
    fit_student,
)
from examples.nanogpt.analyze_mlp_lowbit_complete_neuron_functional import (
    canonical_sha256,
    collect_functional_bank,
    evaluate_function,
    extract_teacher_atoms,
    git_commit,
    summarize,
    teacher_jvp,
    tensor_sha256,
    write_json,
)
from examples.nanogpt.analyze_mlp_noncommuting_hardcell_transport_functional import (
    deterministic_gaussian,
)
from examples.nanogpt.analyze_mlp_token_router_residual_carrier_functional import (
    dense_teacher_function,
)


SCHEMA_VERSION = "nanogpt_mlp_layer_private_diagonal_mixer_transport_functional_v1"
PLAN_SCHEMA_VERSION = (
    "nanogpt_mlp_layer_private_diagonal_mixer_transport_functional_plan_v1"
)
CANDIDATE_NAME = "learned_layer_private_diagonal_mixer_transport_atlas"


def normalized_hadamard_last_dim(values: torch.Tensor) -> torch.Tensor:
    """Apply a normalized Walsh--Hadamard transform on a power-of-two axis."""
    width = values.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    result = values
    stride = 1
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    while stride < width:
        grouped = result.reshape(*result.shape[:-1], -1, 2, stride)
        left = grouped[..., 0, :]
        right = grouped[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1)
        result = result.reshape(*values.shape) * inv_sqrt_two
        stride *= 2
    return result


def mixed_radix_involution(
    values: torch.Tensor,
    *,
    groups: int,
    group_width: int,
) -> torch.Tensor:
    """Apply Q_groups tensor H_group_width without storing either matrix."""
    if groups != 3:
        raise ValueError("H66 freezes the three-group Q3 mixer")
    if values.shape[-1] != groups * group_width:
        raise ValueError("mixed-radix width mismatch")
    shaped = values.reshape(*values.shape[:-1], groups, group_width)
    shaped = normalized_hadamard_last_dim(shaped)
    group_sum = shaped.sum(dim=-2, keepdim=True)
    shaped = shaped - (2.0 / groups) * group_sum
    return shaped.reshape_as(values)


def apply_diagonal_mixer_transport(
    values: torch.Tensor,
    diagonals: torch.Tensor,
    *,
    groups: int,
    group_width: int,
    mode: str,
) -> torch.Tensor:
    """Apply the frozen three-diagonal transport or one of its controls."""
    if mode == "identity":
        return values
    if diagonals.shape != (3, values.shape[-1]):
        raise ValueError("H66 transport-diagonal shape mismatch")
    scales = 1.0 + diagonals
    if mode == "coordinate_only":
        return values * scales[0] * scales[1] * scales[2]
    if mode == "middle_only":
        result = mixed_radix_involution(
            values, groups=groups, group_width=group_width
        )
        result = result * scales[1]
        return mixed_radix_involution(
            result, groups=groups, group_width=group_width
        )
    if mode != "full":
        raise ValueError(f"unknown H66 transport mode: {mode}")
    result = values * scales[0]
    result = mixed_radix_involution(
        result, groups=groups, group_width=group_width
    )
    result = result * scales[1]
    result = mixed_radix_involution(
        result, groups=groups, group_width=group_width
    )
    return result * scales[2]


def deployment_accounting(
    *,
    layers: int = 12,
    width: int = 768,
    banks: int = 3,
    codes_per_bank: int = 64,
    router_rank: int = 64,
    tangent_rank: int = 160,
    transport_diagonals_per_side: int = 3,
) -> dict[str, int | float]:
    total_codes = banks * codes_per_bank
    factored_router = width * router_rank + total_codes * router_rank
    ambient_codebook = total_codes * width
    private_code = layers * 2 * total_codes
    tangent_factors = 2 * width * tangent_rank
    layer_diagonal = layers * tangent_rank
    transport_diagonal = layers * 2 * transport_diagonals_per_side * width
    anchor_offset = layers * 2 * width
    total_values = (
        factored_router
        + ambient_codebook
        + private_code
        + tangent_factors
        + layer_diagonal
        + transport_diagonal
        + anchor_offset
    )
    dense_values = layers * 2 * 4 * width * width
    return {
        "dense_replaced_mlp_fp16_values": dense_values,
        "dense_replaced_mlp_fp16_bytes": 2 * dense_values,
        "fp16_factored_router_values": factored_router,
        "fp16_ambient_codebook_values": ambient_codebook,
        "fp16_private_code_values": private_code,
        "fp16_tangent_factor_values": tangent_factors,
        "fp16_layer_diagonal_values": layer_diagonal,
        "fp16_transport_diagonal_values": transport_diagonal,
        "fp16_anchor_offset_values": anchor_offset,
        "total_latent_values": total_values,
        "total_checkpoint_payload_bytes": 2 * total_values,
        "latent_value_fraction": total_values / dense_values,
        "checkpoint_byte_fraction": total_values / dense_values,
        "extra_mlp_matmul_fraction": (
            width * router_rank
            + router_rank * total_codes
            + 2 * width * tangent_rank
        )
        / (8 * width * width),
        "extra_transport_operation_upper_bound_fraction": 44_544
        / (8 * width * width),
        "cached_procedural_endpoint_bytes": 2 * dense_values,
    }


class LayerPrivateDiagonalMixerTransport(HardProductCodeValueAffineJet):
    def __init__(
        self,
        base_detector: dict[int, torch.Tensor],
        base_write: dict[int, torch.Tensor],
        layers: list[int],
        anchors: torch.Tensor,
        *,
        banks: int,
        codes_per_bank: int,
        router_rank: int,
        tangent_rank: int,
        transport_diagonals_per_side: int,
        mixer_groups: int,
        mixer_group_width: int,
        router_q_seed: int,
        router_p_seed: int,
        tangent_v_seed: int,
        code_gain_initial: float,
        router_bias_initial: float,
        temperature: float,
        layer_diagonal_initial: float,
        transport_diagonal_initial: float,
        offset_initial: float,
    ) -> None:
        super().__init__(
            base_detector,
            base_write,
            layers,
            anchors,
            banks=banks,
            codes_per_bank=codes_per_bank,
            tangent_rank=tangent_rank,
            router_seed=router_p_seed,
            tangent_v_seed=tangent_v_seed,
            code_gain_initial=code_gain_initial,
            router_bias_initial=router_bias_initial,
            temperature=temperature,
            tangent_initial=layer_diagonal_initial,
            offset_initial=offset_initial,
        )
        if transport_diagonals_per_side != 3:
            raise ValueError("H66 freezes three transport diagonals per side")
        if mixer_groups * mixer_group_width != self.width:
            raise ValueError("H66 mixer does not cover model width")
        del self.router
        del self.initial_router
        self.router_rank = int(router_rank)
        self.transport_diagonals_per_side = int(
            transport_diagonals_per_side
        )
        self.mixer_groups = int(mixer_groups)
        self.mixer_group_width = int(mixer_group_width)
        initial_q = deterministic_gaussian(
            self.width, self.router_rank, router_q_seed, self.width
        )
        initial_p = deterministic_gaussian(
            self.total_codes, self.router_rank, router_p_seed, self.router_rank
        )
        self.register_buffer("initial_router_q", initial_q.clone())
        self.register_buffer("initial_router_p", initial_p.clone())
        self.router_q = torch.nn.Parameter(initial_q)
        self.router_p = torch.nn.Parameter(initial_p)
        shape = (
            len(layers),
            self.transport_diagonals_per_side,
            self.width,
        )
        self.input_transport_diagonals = torch.nn.Parameter(
            torch.full(shape, float(transport_diagonal_initial))
        )
        self.output_transport_diagonals = torch.nn.Parameter(
            torch.full(shape, float(transport_diagonal_initial))
        )

    @property
    def input_reflectors(self) -> torch.nn.Parameter:
        """Compatibility alias for the frozen H65 optimizer routine."""
        return self.input_transport_diagonals

    @property
    def output_reflectors(self) -> torch.nn.Parameter:
        """Compatibility alias for the frozen H65 optimizer routine."""
        return self.output_transport_diagonals

    def _parts(self, row: int, mode: str) -> dict[str, Any]:
        parts: dict[str, Any] = {
            "router_q": self.router_q,
            "router_p": self.router_p,
            "codebook": self.codebook,
            "router_bias": self.router_bias[row],
            "code_gain": self.code_gain[row],
            "tangent_u": self.tangent_u,
            "tangent_v": self.tangent_v,
            "layer_diagonal": self.tangent_coordinate[row],
            "input_diagonals": self.input_transport_diagonals[row],
            "output_diagonals": self.output_transport_diagonals[row],
            "input_mode": "full",
            "output_mode": "full",
            "anchor": self.anchors[row],
            "offset": self.output_offset[row],
        }
        if mode == "step_zero_parent":
            parts["codebook"] = torch.zeros_like(self.codebook)
            parts["tangent_u"] = torch.zeros_like(self.tangent_u)
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "identity_transports":
            parts["input_mode"] = "identity"
            parts["output_mode"] = "identity"
        elif mode == "input_transport_only":
            parts["output_mode"] = "identity"
        elif mode == "output_transport_only":
            parts["input_mode"] = "identity"
        elif mode == "one_conjugated_diagonal_stage":
            parts["input_mode"] = "middle_only"
            parts["output_mode"] = "middle_only"
        elif mode == "coordinate_diagonal_only":
            parts["input_mode"] = "coordinate_only"
            parts["output_mode"] = "coordinate_only"
        elif mode == "shared_across_layers":
            parts["input_diagonals"] = self.input_transport_diagonals.mean(dim=0)
            parts["output_diagonals"] = self.output_transport_diagonals.mean(dim=0)
        elif mode == "piecewise_tangent_only":
            parts["codebook"] = torch.zeros_like(self.codebook)
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "hard_value_only":
            parts["tangent_u"] = torch.zeros_like(self.tangent_u)
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "initial_random_router":
            parts["router_q"] = self.initial_router_q
            parts["router_p"] = self.initial_router_p
        elif mode != "full":
            raise ValueError(f"unknown H66 mode: {mode}")
        return parts

    def _transport(
        self,
        values: torch.Tensor,
        diagonals: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        return apply_diagonal_mixer_transport(
            values,
            diagonals,
            groups=self.mixer_groups,
            group_width=self.mixer_group_width,
            mode=mode,
        )

    def forward_function(
        self,
        layer: int,
        inputs: torch.Tensor,
        directions: torch.Tensor | None,
        *,
        mode: str = "full",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        row = self.layer_to_row[layer]
        detector = self.base_detector[row]
        write = self.base_write[row]
        p = self._parts(row, mode)
        hidden = F.gelu(inputs @ detector.T)
        base_output = hidden @ write
        logits = (base_output @ p["router_q"]) @ p["router_p"].T + p[
            "router_bias"
        ]
        assignment = self._assignment(logits)
        flat_assignment = assignment.reshape(-1, self.total_codes)
        value_correction = (flat_assignment * p["code_gain"]) @ p["codebook"]
        transported_input = self._transport(
            inputs - p["anchor"], p["input_diagonals"], p["input_mode"]
        )
        latent = (transported_input @ p["tangent_v"]) * p["layer_diagonal"]
        raw_tangent = latent @ p["tangent_u"].T
        tangent_correction = self._transport(
            raw_tangent, p["output_diagonals"], p["output_mode"]
        )
        output = base_output + value_correction + p["offset"] + tangent_correction
        if directions is None:
            return output, None
        base_action = teacher_jvp(inputs, directions, detector, write)
        transported_direction = self._transport(
            directions, p["input_diagonals"], p["input_mode"]
        )
        latent_action = (
            transported_direction @ p["tangent_v"]
        ) * p["layer_diagonal"]
        raw_action = latent_action @ p["tangent_u"].T
        tangent_action = self._transport(
            raw_action, p["output_diagonals"], p["output_mode"]
        )
        return output, base_action + tangent_action


def terminal_artifact(
    student: LayerPrivateDiagonalMixerTransport,
    accounting: dict[str, Any],
) -> dict[str, Any]:
    values = {
        "router_q": student.router_q.detach().to(torch.float16).cpu(),
        "router_p": student.router_p.detach().to(torch.float16).cpu(),
        "codebook": student.codebook.detach().to(torch.float16).cpu(),
        "router_bias": student.router_bias.detach().to(torch.float16).cpu(),
        "code_gain": student.code_gain.detach().to(torch.float16).cpu(),
        "tangent_u": student.tangent_u.detach().to(torch.float16).cpu(),
        "tangent_v": student.tangent_v.detach().to(torch.float16).cpu(),
        "layer_diagonal": student.tangent_coordinate.detach().to(torch.float16).cpu(),
        "input_transport_diagonals": student.input_transport_diagonals.detach().to(torch.float16).cpu(),
        "output_transport_diagonals": student.output_transport_diagonals.detach().to(torch.float16).cpu(),
        "anchors": student.anchors.detach().to(torch.float16).cpu(),
        "output_offset": student.output_offset.detach().to(torch.float16).cpu(),
    }
    payload_bytes = 2 * sum(value.numel() for value in values.values())
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise AssertionError((payload_bytes, accounting))
    return {
        "schema_version": "layer_private_diagonal_mixer_checkpoint_v1",
        **values,
        "layers": student.layers,
        "banks": student.banks,
        "codes_per_bank": student.codes_per_bank,
        "router_rank": student.router_rank,
        "transport_diagonals_per_side": student.transport_diagonals_per_side,
        "mixer_groups": student.mixer_groups,
        "mixer_group_width": student.mixer_group_width,
        "accounted_payload_bytes": payload_bytes,
        "base_detector_sha256": tensor_sha256(student.base_detector),
        "base_write_sha256": tensor_sha256(student.base_write),
    }


def artifact_student(
    artifact: dict[str, Any],
    base_detector: dict[int, torch.Tensor],
    base_write: dict[int, torch.Tensor],
    plan: dict[str, Any],
    device: str,
) -> LayerPrivateDiagonalMixerTransport:
    frozen = plan["frozen_representation"]
    init = plan["initialization"]
    student = LayerPrivateDiagonalMixerTransport(
        base_detector,
        base_write,
        [int(value) for value in artifact["layers"]],
        artifact["anchors"].float(),
        banks=int(frozen["banks"]),
        codes_per_bank=int(frozen["codes_per_bank"]),
        router_rank=int(frozen["router_rank"]),
        tangent_rank=int(frozen["tangent_rank"]),
        transport_diagonals_per_side=int(
            frozen["transport_diagonals_per_side"]
        ),
        mixer_groups=int(frozen["mixer_groups"]),
        mixer_group_width=int(frozen["mixer_group_width"]),
        router_q_seed=int(init["router_q_seed"]),
        router_p_seed=int(init["router_p_seed"]),
        tangent_v_seed=int(init["tangent_v_seed"]),
        code_gain_initial=float(init["code_gain"]),
        router_bias_initial=float(init["router_bias"]),
        temperature=float(init["straight_through_temperature"]),
        layer_diagonal_initial=float(init["layer_diagonal"]),
        transport_diagonal_initial=float(init["input_transport_diagonals"]),
        offset_initial=float(init["output_offset"]),
    )
    with torch.no_grad():
        for source_name, target_name in (
            ("router_q", "router_q"),
            ("router_p", "router_p"),
            ("codebook", "codebook"),
            ("router_bias", "router_bias"),
            ("code_gain", "code_gain"),
            ("tangent_u", "tangent_u"),
            ("tangent_v", "tangent_v"),
            ("layer_diagonal", "tangent_coordinate"),
            ("input_transport_diagonals", "input_transport_diagonals"),
            ("output_transport_diagonals", "output_transport_diagonals"),
            ("anchors", "anchors"),
            ("output_offset", "output_offset"),
        ):
            getattr(student, target_name).copy_(artifact[source_name].float())
    student.to(device)
    student.eval()
    return student


def student_function(
    student: LayerPrivateDiagonalMixerTransport, mode: str
) -> Callable[
    [int, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]:
    def function(
        layer: int, inputs: torch.Tensor, directions: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return student.forward_function(layer, inputs, directions, mode=mode)

    return function


def gate_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["candidate"] == CANDIDATE_NAME and row["split"] == "holdout"
    ]
    layer_gates = [
        {
            "layer": row["layer"],
            "output_pass": row["relative_output_rmse"] <= 0.10,
            "jvp_pass": row["relative_jvp_rmse"] <= 0.15,
            "covariance_pass": row[
                "retained_centered_output_covariance_energy"
            ]
            >= 0.90,
            "finite_pass": bool(row["finite"]),
        }
        for row in selected
    ]
    parent = {
        (row["split"], row["layer"]): row["relative_jvp_rmse"]
        for row in rows
        if row["candidate"] == "step_zero_parent"
    }
    hard_value = {
        (row["split"], row["layer"]): row["relative_jvp_rmse"]
        for row in rows
        if row["candidate"] == "hard_value_only"
    }
    identity_error = max(
        abs(hard_value[key] - value) for key, value in parent.items()
    )
    identity_pass = identity_error <= 1e-6
    representation_pass = (
        len(layer_gates) == 12
        and all(
            all(value for key, value in row.items() if key != "layer")
            for row in layer_gates
        )
        and identity_pass
    )
    return {
        "layer_gates": layer_gates,
        "hard_value_jvp_identity_max_absolute_error": identity_error,
        "hard_value_jvp_identity_pass": identity_pass,
        "representation_pass": representation_pass,
        "decision": (
            "PROMOTE_H66_TO_EXACT_SYSTEMS_GATE"
            if representation_pass
            else "REJECT_H66_LAYER_PRIVATE_DIAGONAL_MIXER_TRANSPORT"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-cap", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=3072)
    parser.add_argument("--minibatch-rows", type=int, default=128)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unexpected H66 plan schema")
    if args.preflight:
        args.sample_cap = min(args.sample_cap, 128)
        args.iterations = min(args.iterations, 24)
    else:
        fit = plan["fit"]
        inventory = plan["frozen_function_inventory"]
        if args.sample_cap != int(inventory["samples_per_layer_per_split"]):
            raise ValueError("binding sample cap differs from frozen plan")
        if args.iterations != int(fit["iterations"]):
            raise ValueError("binding iterations differ from frozen plan")
        if args.minibatch_rows != int(fit["minibatch_rows"]):
            raise ValueError("binding minibatch differs from frozen plan")

    teacher_plan = plan["frozen_teacher"]
    layers = [int(value) for value in teacher_plan["layers"]]
    width = int(teacher_plan["model_width"])
    frozen = plan["frozen_representation"]
    accounting = deployment_accounting(
        layers=len(layers),
        width=width,
        banks=int(frozen["banks"]),
        codes_per_bank=int(frozen["codes_per_bank"]),
        router_rank=int(frozen["router_rank"]),
        tangent_rank=int(frozen["tangent_rank"]),
        transport_diagonals_per_side=int(
            frozen["transport_diagonals_per_side"]
        ),
    )
    expected = plan["exact_deployment_accounting"]
    for key in (
        "dense_replaced_mlp_fp16_values",
        "dense_replaced_mlp_fp16_bytes",
        "fp16_factored_router_values",
        "fp16_ambient_codebook_values",
        "fp16_private_code_values",
        "fp16_tangent_factor_values",
        "fp16_layer_diagonal_values",
        "fp16_transport_diagonal_values",
        "fp16_anchor_offset_values",
        "total_latent_values",
        "cached_procedural_endpoint_bytes",
    ):
        if accounting[key] != expected[key]:
            raise AssertionError((key, accounting[key], expected[key]))
    if accounting["total_checkpoint_payload_bytes"] != expected[
        "total_checkpoint_bytes"
    ]:
        raise AssertionError("H66 checkpoint-byte accounting mismatch")

    initial_snapshot = args.snapshot_dir / teacher_plan["initial_snapshot"]
    terminal_snapshot = args.snapshot_dir / teacher_plan["terminal_snapshot"]
    if file_sha256(initial_snapshot) != teacher_plan["initial_snapshot_sha256"]:
        raise ValueError("step-zero snapshot identity mismatch")
    if file_sha256(terminal_snapshot) != teacher_plan["terminal_snapshot_sha256"]:
        raise ValueError("terminal snapshot identity mismatch")
    if file_sha256(args.data_dir / "manifest.json") != plan[
        "frozen_function_inventory"
    ]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")

    terminal_payload = load_snapshot(terminal_snapshot)
    terminal_model = model_from_snapshot(terminal_payload, args.device)
    terminal_model.eval()
    teacher_detector, teacher_write = extract_teacher_atoms(terminal_model, layers)
    inventory = plan["frozen_function_inventory"]
    banks_data = {
        "train": collect_functional_bank(
            terminal_model,
            args.data_dir,
            layers,
            args.sample_cap,
            int(inventory["train_seed"]),
            2,
            int(terminal_model.config.block_size),
            args.device,
        ),
        "holdout": collect_functional_bank(
            terminal_model,
            args.data_dir,
            layers,
            args.sample_cap,
            int(inventory["holdout_seed"]),
            2,
            int(terminal_model.config.block_size),
            args.device,
        ),
    }
    del terminal_model
    initial_payload = load_snapshot(initial_snapshot)
    initial_model = model_from_snapshot(initial_payload, args.device)
    initial_model.eval()
    base_detector, base_write = extract_teacher_atoms(initial_model, layers)
    del initial_model
    anchors = torch.stack(
        [banks_data["train"].inputs[layer].float().mean(dim=0) for layer in layers]
    )
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    init = plan["initialization"]
    student = LayerPrivateDiagonalMixerTransport(
        base_detector,
        base_write,
        layers,
        anchors,
        banks=int(frozen["banks"]),
        codes_per_bank=int(frozen["codes_per_bank"]),
        router_rank=int(frozen["router_rank"]),
        tangent_rank=int(frozen["tangent_rank"]),
        transport_diagonals_per_side=int(
            frozen["transport_diagonals_per_side"]
        ),
        mixer_groups=int(frozen["mixer_groups"]),
        mixer_group_width=int(frozen["mixer_group_width"]),
        router_q_seed=int(init["router_q_seed"]),
        router_p_seed=int(init["router_p_seed"]),
        tangent_v_seed=int(init["tangent_v_seed"]),
        code_gain_initial=float(init["code_gain"]),
        router_bias_initial=float(init["router_bias"]),
        temperature=float(init["straight_through_temperature"]),
        layer_diagonal_initial=float(init["layer_diagonal"]),
        transport_diagonal_initial=float(init["input_transport_diagonals"]),
        offset_initial=float(init["output_offset"]),
    )
    history = fit_student(
        student,
        banks_data["train"],
        teacher_detector,
        teacher_write,
        plan,
        layers=layers,
        iterations=args.iterations,
        minibatch_rows=args.minibatch_rows,
        jvp_seed=int(inventory["jvp_seed"]),
        device=args.device,
    )
    artifact = terminal_artifact(student, accounting)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output / "layer_private_diagonal_mixer_checkpoint.pt"
    torch.save(artifact, checkpoint_path)
    fitted = artifact_student(artifact, base_detector, base_write, plan, args.device)
    modes = (
        "full",
        "step_zero_parent",
        "identity_transports",
        "input_transport_only",
        "output_transport_only",
        "one_conjugated_diagonal_stage",
        "coordinate_diagonal_only",
        "shared_across_layers",
        "piecewise_tangent_only",
        "hard_value_only",
        "initial_random_router",
    )
    candidates = {
        (CANDIDATE_NAME if mode == "full" else mode): student_function(
            fitted, mode
        )
        for mode in modes
    }
    candidates["dense_teacher_identity"] = dense_teacher_function(
        teacher_detector, teacher_write, args.device
    )
    rows: list[dict[str, Any]] = []
    for name, function in candidates.items():
        rows.extend(
            evaluate_function(
                name,
                function,
                banks_data,
                teacher_detector,
                teacher_write,
                layers=layers,
                jvp_seed=int(inventory["jvp_seed"]),
                directions=int(inventory["jvp_directions_per_layer_per_split"]),
                device=args.device,
            )
        )
    summary = summarize(rows)
    gate = gate_outcome(rows)
    gate["all_gates_pass"] = bool(gate["representation_pass"])
    detail_path = args.output / "functional_metrics.json"
    summary_path = args.output / "functional_summary.json"
    history_path = args.output / "fit_history.json"
    gate_path = args.output / "gate.json"
    write_json(detail_path, rows)
    write_json(summary_path, summary)
    write_json(history_path, history)
    write_json(gate_path, gate)
    source = Path(__file__).resolve()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "preflight": bool(args.preflight),
        "source_commit": git_commit(source.parents[2]),
        "source_sha256": file_sha256(source),
        "plan": str(args.plan),
        "plan_sha256": file_sha256(args.plan),
        "initial_snapshot": str(initial_snapshot),
        "initial_snapshot_sha256": file_sha256(initial_snapshot),
        "initial_snapshot_run_identity_sha256": str(
            initial_payload["run_identity_sha256"]
        ),
        "initial_snapshot_tensor_inventory_sha256": canonical_sha256(
            initial_payload["tensor_inventory"]
        ),
        "terminal_snapshot": str(terminal_snapshot),
        "terminal_snapshot_sha256": file_sha256(terminal_snapshot),
        "terminal_snapshot_run_identity_sha256": str(
            terminal_payload["run_identity_sha256"]
        ),
        "terminal_snapshot_tensor_inventory_sha256": canonical_sha256(
            terminal_payload["tensor_inventory"]
        ),
        "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        "sample_cap": args.sample_cap,
        "iterations": args.iterations,
        "layers": layers,
        "base_detector_sha256": tensor_sha256(
            torch.stack([base_detector[layer] for layer in layers])
        ),
        "base_write_sha256": tensor_sha256(
            torch.stack([base_write[layer] for layer in layers])
        ),
        "activation_anchor_sha256": tensor_sha256(anchors),
        "accounting": accounting,
        "temporary_training_state": {
            "fp32_coordinate_master_bytes": accounting["total_latent_values"] * 4,
            "fp32_coordinate_gradient_upper_bound_bytes": accounting[
                "total_latent_values"
            ]
            * 4,
            "adam_coordinate_moment_upper_bound_bytes": accounting[
                "total_latent_values"
            ]
            * 8,
            "fixed_fp32_procedural_endpoint_bytes": accounting[
                "dense_replaced_mlp_fp16_bytes"
            ]
            * 2,
            "optimizer_state_is_not_deployment_state": True,
        },
        "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "detail_sha256": file_sha256(detail_path),
        "summary_sha256": file_sha256(summary_path),
        "history_sha256": file_sha256(history_path),
        "gate_sha256": file_sha256(gate_path),
        "runtime_seconds": time.time() - started,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated())
            if args.device.startswith("cuda")
            else 0
        ),
        "gate": gate,
        "limitations": [
            "Single 124M step-zero parent, terminal teacher, and dataset.",
            "Function/JVP representation audit, not CE training.",
            "Hard-routing JVP excludes routing-boundary derivatives.",
            "Step-zero endpoints are exact audit inputs but excluded from compact state.",
            "Activation anchors are data-derived and fully charged.",
            "FP32 coordinate optimizer state is transient training state.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    result_bytes = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    if result_bytes > int(plan["runtime_gates"]["maximum_result_directory_bytes"]):
        raise RuntimeError("H66 result directory exceeds frozen storage gate")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
