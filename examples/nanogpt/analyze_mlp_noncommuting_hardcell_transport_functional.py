#!/usr/bin/env python3
"""Frozen H64 non-commuting hard-cell MLP functional audit."""

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
from examples.nanogpt.analyze_mlp_lowbit_complete_neuron_functional import (
    FunctionalBank,
    canonical_sha256,
    collect_functional_bank,
    evaluate_function,
    extract_teacher_atoms,
    git_commit,
    rademacher_direction,
    summarize,
    teacher_jvp,
    tensor_sha256,
    write_json,
)
from examples.nanogpt.analyze_mlp_token_router_residual_carrier_functional import (
    dense_teacher_function,
)


SCHEMA_VERSION = "nanogpt_mlp_noncommuting_hardcell_transport_functional_v1"
PLAN_SCHEMA_VERSION = (
    "nanogpt_mlp_noncommuting_hardcell_transport_functional_plan_v1"
)
CANDIDATE_NAME = "learned_noncommuting_hardcell_transport_atlas"


def deterministic_gaussian(
    rows: int, columns: int, seed: int, denominator: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(rows, columns, generator=generator) / math.sqrt(denominator)


def deployment_accounting(
    *,
    layers: int = 12,
    width: int = 768,
    banks: int = 3,
    codes_per_bank: int = 64,
    router_rank: int = 64,
    tangent_rank: int = 160,
) -> dict[str, int | float]:
    total_codes = banks * codes_per_bank
    factored_router = width * router_rank + total_codes * router_rank
    ambient_codebook = total_codes * width
    private_code = layers * 2 * total_codes
    tangent_factors = 2 * width * tangent_rank
    layer_diagonal = layers * tangent_rank
    code_transport = 2 * total_codes * tangent_rank
    anchor_offset = layers * 2 * width
    total_values = (
        factored_router
        + ambient_codebook
        + private_code
        + tangent_factors
        + layer_diagonal
        + code_transport
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
        "fp16_code_transport_values": code_transport,
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
        "cached_procedural_endpoint_bytes": 2 * dense_values,
    }


class NoncommutingHardCellTransport(HardProductCodeValueAffineJet):
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
        router_q_seed: int,
        router_p_seed: int,
        tangent_v_seed: int,
        transport_a_seed: int,
        transport_b_seed: int,
        code_gain_initial: float,
        router_bias_initial: float,
        temperature: float,
        layer_diagonal_initial: float,
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
        del self.router
        del self.initial_router
        self.router_rank = int(router_rank)
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
        self.transport_a = torch.nn.Parameter(
            deterministic_gaussian(
                self.total_codes,
                self.tangent_rank,
                transport_a_seed,
                self.tangent_rank,
            )
        )
        self.transport_b = torch.nn.Parameter(
            deterministic_gaussian(
                self.total_codes,
                self.tangent_rank,
                transport_b_seed,
                self.tangent_rank,
            )
        )

    def _parts(self, row: int, mode: str) -> dict[str, torch.Tensor | bool]:
        parts: dict[str, torch.Tensor | bool] = {
            "router_q": self.router_q,
            "router_p": self.router_p,
            "codebook": self.codebook,
            "router_bias": self.router_bias[row],
            "code_gain": self.code_gain[row],
            "tangent_u": self.tangent_u,
            "tangent_v": self.tangent_v,
            "layer_diagonal": self.tangent_coordinate[row],
            "transport_a": self.transport_a,
            "transport_b": self.transport_b,
            "anchor": self.anchors[row],
            "offset": self.output_offset[row],
            "single_bank": False,
        }
        if mode == "step_zero_parent":
            parts["codebook"] = torch.zeros_like(self.codebook)
            parts["tangent_u"] = torch.zeros_like(self.tangent_u)
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "diagonal_and_value_no_rankone":
            parts["transport_a"] = torch.zeros_like(self.transport_a)
        elif mode == "rankone_tangent_only":
            parts["layer_diagonal"] = torch.zeros_like(
                self.tangent_coordinate[row]
            )
            parts["codebook"] = torch.zeros_like(self.codebook)
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "piecewise_tangent_only":
            parts["codebook"] = torch.zeros_like(self.codebook)
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "hard_value_only":
            parts["tangent_u"] = torch.zeros_like(self.tangent_u)
            parts["offset"] = torch.zeros_like(self.output_offset[row])
        elif mode == "global_code_coordinates":
            parts["router_bias"] = self.router_bias.mean(dim=0)
            parts["code_gain"] = self.code_gain.mean(dim=0)
        elif mode == "initial_random_router":
            parts["router_q"] = self.initial_router_q
            parts["router_p"] = self.initial_router_p
        elif mode == "single_bank_only":
            parts["single_bank"] = True
        elif mode == "tied_symmetric_transport":
            parts["transport_b"] = self.transport_a
        elif mode != "full":
            raise ValueError(f"unknown H64 mode: {mode}")
        return parts

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
        tensors = {key: value for key, value in p.items() if key != "single_bank"}
        if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("H64 tensor part is not a tensor")
        router_q = tensors["router_q"]
        router_p = tensors["router_p"]
        codebook = tensors["codebook"]
        router_bias = tensors["router_bias"]
        code_gain = tensors["code_gain"]
        tangent_u = tensors["tangent_u"]
        tangent_v = tensors["tangent_v"]
        layer_diagonal = tensors["layer_diagonal"]
        transport_a = tensors["transport_a"]
        transport_b = tensors["transport_b"]
        anchor = tensors["anchor"]
        offset = tensors["offset"]

        hidden = F.gelu(inputs @ detector.T)
        base_output = hidden @ write
        logits = (base_output @ router_q) @ router_p.T + router_bias
        assignment = self._assignment(logits)
        if bool(p["single_bank"]):
            mask = torch.zeros_like(assignment)
            mask[:, 0, :] = assignment[:, 0, :]
            assignment = mask
        flat_assignment = assignment.reshape(-1, self.total_codes)
        value_correction = (flat_assignment * code_gain) @ codebook
        projected = (inputs - anchor) @ tangent_v
        transport_score = projected @ transport_b.T
        latent_correction = projected * layer_diagonal
        latent_correction = latent_correction + (
            flat_assignment * transport_score
        ) @ transport_a
        tangent_correction = latent_correction @ tangent_u.T
        output = base_output + value_correction + offset + tangent_correction
        if directions is None:
            return output, None
        base_action = teacher_jvp(inputs, directions, detector, write)
        direction_projected = directions @ tangent_v
        direction_score = direction_projected @ transport_b.T
        latent_action = direction_projected * layer_diagonal
        latent_action = latent_action + (
            flat_assignment * direction_score
        ) @ transport_a
        return output, base_action + latent_action @ tangent_u.T


def optimizer_for(
    student: NoncommutingHardCellTransport, plan: dict[str, Any]
) -> torch.optim.Optimizer:
    fit = plan["fit"]
    return torch.optim.AdamW(
        [
            {
                "params": [
                    student.router_q,
                    student.router_p,
                    student.codebook,
                    student.tangent_u,
                    student.tangent_v,
                    student.transport_a,
                    student.transport_b,
                ],
                "lr": float(fit["learning_rate_shared_factors"]),
                "weight_decay": float(fit["weight_decay_shared"]),
            },
            {
                "params": [
                    student.router_bias,
                    student.code_gain,
                    student.tangent_coordinate,
                    student.output_offset,
                ],
                "lr": float(fit["learning_rate_coordinates"]),
                "weight_decay": float(fit["weight_decay_coordinates"]),
            },
        ]
    )


def fit_student(
    student: NoncommutingHardCellTransport,
    bank: FunctionalBank,
    teacher_detector: dict[int, torch.Tensor],
    teacher_write: dict[int, torch.Tensor],
    plan: dict[str, Any],
    *,
    layers: list[int],
    iterations: int,
    minibatch_rows: int,
    jvp_seed: int,
    device: str,
) -> list[dict[str, float | int]]:
    student.to(device)
    student.train()
    optimizer = optimizer_for(student, plan)
    gradient_clip_norm = float(plan["fit"]["gradient_clip_norm"])
    history: list[dict[str, float | int]] = []
    for iteration in range(iterations):
        layer = layers[iteration % len(layers)]
        rows = bank.inputs[layer].shape[0]
        cycle = iteration // len(layers)
        start = (cycle * minibatch_rows) % rows
        index = (torch.arange(minibatch_rows) + start) % rows
        inputs = bank.inputs[layer][index].to(device=device, dtype=torch.float32)
        target = bank.outputs[layer][index].to(device=device, dtype=torch.float32)
        direction_index = cycle % 4
        direction = rademacher_direction(
            tuple(inputs.shape),
            jvp_seed
            + 1009 * layer
            + 100_003 * direction_index
            + 1_000_003 * start,
            device,
        )
        with torch.no_grad():
            target_action = teacher_jvp(
                inputs,
                direction,
                teacher_detector[layer].to(device=device, dtype=torch.float32),
                teacher_write[layer].to(device=device, dtype=torch.float32),
            )
        prediction, action = student.forward_function(layer, inputs, direction)
        if action is None:
            raise RuntimeError("H64 student JVP is missing")
        output_loss = (
            (prediction - target).square().mean()
            / target.square().mean().clamp_min(1e-30)
        )
        jvp_loss = (
            (action - target_action).square().mean()
            / target_action.square().mean().clamp_min(1e-30)
        )
        loss = output_loss + 0.25 * jvp_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), gradient_clip_norm
        )
        optimizer.step()
        if iteration in {
            0,
            iterations // 4,
            iterations // 2,
            3 * iterations // 4,
            iterations - 1,
        }:
            record = {
                "iteration": iteration + 1,
                "layer": layer,
                "loss": float(loss.detach()),
                "relative_output_mse": float(output_loss.detach()),
                "relative_jvp_mse": float(jvp_loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    student.eval()
    return history


def terminal_artifact(
    student: NoncommutingHardCellTransport, accounting: dict[str, Any]
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
        "transport_a": student.transport_a.detach().to(torch.float16).cpu(),
        "transport_b": student.transport_b.detach().to(torch.float16).cpu(),
        "anchors": student.anchors.detach().to(torch.float16).cpu(),
        "output_offset": student.output_offset.detach().to(torch.float16).cpu(),
    }
    payload_bytes = 2 * sum(value.numel() for value in values.values())
    if payload_bytes != accounting["total_checkpoint_payload_bytes"]:
        raise AssertionError((payload_bytes, accounting))
    return {
        "schema_version": "noncommuting_hardcell_transport_checkpoint_v1",
        **values,
        "layers": student.layers,
        "banks": student.banks,
        "codes_per_bank": student.codes_per_bank,
        "router_rank": student.router_rank,
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
) -> NoncommutingHardCellTransport:
    frozen = plan["frozen_representation"]
    init = plan["initialization"]
    student = NoncommutingHardCellTransport(
        base_detector,
        base_write,
        [int(value) for value in artifact["layers"]],
        artifact["anchors"].float(),
        banks=int(frozen["banks"]),
        codes_per_bank=int(frozen["codes_per_bank"]),
        router_rank=int(frozen["router_rank"]),
        tangent_rank=int(frozen["tangent_rank"]),
        router_q_seed=int(init["router_q_seed"]),
        router_p_seed=int(init["router_p_seed"]),
        tangent_v_seed=int(init["tangent_v_seed"]),
        transport_a_seed=int(init["code_transport_a_seed"]),
        transport_b_seed=int(init["code_transport_b_seed"]),
        code_gain_initial=float(init["code_gain"]),
        router_bias_initial=float(init["router_bias"]),
        temperature=float(init["straight_through_temperature"]),
        layer_diagonal_initial=float(init["layer_diagonal"]),
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
            ("transport_a", "transport_a"),
            ("transport_b", "transport_b"),
            ("anchors", "anchors"),
            ("output_offset", "output_offset"),
        ):
            getattr(student, target_name).copy_(artifact[source_name].float())
    student.to(device)
    student.eval()
    return student


def student_function(
    student: NoncommutingHardCellTransport, mode: str
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
    jvp_identity_error = max(
        abs(hard_value[key] - value) for key, value in parent.items()
    )
    hard_value_jvp_identity_pass = jvp_identity_error <= 1e-6
    representation_pass = (
        len(layer_gates) == 12
        and all(
            all(value for key, value in row.items() if key != "layer")
            for row in layer_gates
        )
        and hard_value_jvp_identity_pass
    )
    return {
        "layer_gates": layer_gates,
        "hard_value_jvp_identity_max_absolute_error": jvp_identity_error,
        "hard_value_jvp_identity_pass": hard_value_jvp_identity_pass,
        "representation_pass": representation_pass,
        "decision": (
            "PROMOTE_H64_TO_EXACT_SYSTEMS_GATE"
            if representation_pass
            else "REJECT_H64_NONCOMMUTING_HARDCELL_TRANSPORT"
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
        raise ValueError("unexpected H64 plan schema")
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
        "fp16_code_transport_values",
        "fp16_anchor_offset_values",
        "total_latent_values",
        "cached_procedural_endpoint_bytes",
    ):
        if accounting[key] != expected[key]:
            raise AssertionError((key, accounting[key], expected[key]))
    if accounting["total_checkpoint_payload_bytes"] != expected[
        "total_checkpoint_bytes"
    ]:
        raise AssertionError("H64 checkpoint-byte accounting mismatch")

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
    student = NoncommutingHardCellTransport(
        base_detector,
        base_write,
        layers,
        anchors,
        banks=int(frozen["banks"]),
        codes_per_bank=int(frozen["codes_per_bank"]),
        router_rank=int(frozen["router_rank"]),
        tangent_rank=int(frozen["tangent_rank"]),
        router_q_seed=int(init["router_q_seed"]),
        router_p_seed=int(init["router_p_seed"]),
        tangent_v_seed=int(init["tangent_v_seed"]),
        transport_a_seed=int(init["code_transport_a_seed"]),
        transport_b_seed=int(init["code_transport_b_seed"]),
        code_gain_initial=float(init["code_gain"]),
        router_bias_initial=float(init["router_bias"]),
        temperature=float(init["straight_through_temperature"]),
        layer_diagonal_initial=float(init["layer_diagonal"]),
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
    checkpoint_path = args.output / "noncommuting_hardcell_checkpoint.pt"
    torch.save(artifact, checkpoint_path)
    fitted = artifact_student(artifact, base_detector, base_write, plan, args.device)
    modes = (
        "full",
        "step_zero_parent",
        "diagonal_and_value_no_rankone",
        "rankone_tangent_only",
        "piecewise_tangent_only",
        "hard_value_only",
        "global_code_coordinates",
        "initial_random_router",
        "single_bank_only",
        "tied_symmetric_transport",
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
            "Hard-routing JVP is the deployed piecewise-affine derivative, not the straight-through routing derivative.",
            "Step-zero endpoints are exact audit inputs but excluded from the compact artifact; regeneration remains a later mandatory gate.",
            "Activation anchors are data-derived but fully charged in persistent state.",
            "FP32 compact-coordinate optimizer state is transient training state and is reported separately.",
        ],
    }
    metadata_path = args.output / "metadata.json"
    write_json(metadata_path, metadata)
    result_bytes = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    if result_bytes > int(plan["runtime_gates"]["maximum_result_directory_bytes"]):
        raise RuntimeError("H64 result directory exceeds frozen storage gate")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
