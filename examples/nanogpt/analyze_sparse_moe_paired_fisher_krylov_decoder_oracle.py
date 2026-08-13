#!/usr/bin/env python3
"""Gate a paired sparse-MoE Fisher-Krylov procedural decoder without updates."""
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
    action_cosine,
    collect_protocol_inputs,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    file_sha256,
    tensor_sha256,
)
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
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
    model_from_exact_stepzero,
    selected_stepzero_hashes,
)
from latent_weight_lab.block_fht import (
    block_fht_grad_latent,
    block_fht_slice,
    next_power_of_two,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_paired_fisher_krylov_decoder_oracle_plan_v1"
RESULT_SCHEMA = "nanogpt_sparse_moe_paired_fisher_krylov_decoder_oracle_result_v1"
COORDINATE_SCHEMA = "nanogpt_sparse_moe_paired_fisher_krylov_coordinates_v1"


@dataclass
class PairedTensor:
    c_fc: torch.Tensor
    c_proj: torch.Tensor

    def to(self, device: str) -> "PairedTensor":
        return PairedTensor(
            self.c_fc.to(device=device, dtype=torch.float32),
            self.c_proj.to(device=device, dtype=torch.float32),
        )

    def cpu(self) -> "PairedTensor":
        return PairedTensor(self.c_fc.detach().cpu(), self.c_proj.detach().cpu())


@dataclass
class LatentTuple:
    branches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def cpu(self) -> "LatentTuple":
        return LatentTuple(tuple(value.detach().cpu() for value in self.branches))


def pair_from_state(state: LayerState) -> PairedTensor:
    return PairedTensor(state.c_fc.float(), state.c_proj.float())


def pair_zeros_like(reference: PairedTensor) -> PairedTensor:
    return PairedTensor(torch.zeros_like(reference.c_fc), torch.zeros_like(reference.c_proj))


def pair_add(left: PairedTensor, right: PairedTensor) -> PairedTensor:
    return PairedTensor(left.c_fc + right.c_fc, left.c_proj + right.c_proj)


def pair_subtract(left: PairedTensor, right: PairedTensor) -> PairedTensor:
    return PairedTensor(left.c_fc - right.c_fc, left.c_proj - right.c_proj)


def pair_scale(value: PairedTensor, scale: torch.Tensor | float) -> PairedTensor:
    return PairedTensor(value.c_fc * scale, value.c_proj * scale)


def pair_dot(left: PairedTensor, right: PairedTensor) -> torch.Tensor:
    return (left.c_fc * right.c_fc).sum() + (left.c_proj * right.c_proj).sum()


def pair_norm(value: PairedTensor) -> torch.Tensor:
    return pair_dot(value, value).clamp_min(0.0).sqrt()


def latent_zeros(widths: list[int], experts: int, device: str) -> LatentTuple:
    return LatentTuple(
        tuple(torch.zeros(experts, width, device=device) for width in widths)
    )


def latent_add(left: LatentTuple, right: LatentTuple) -> LatentTuple:
    return LatentTuple(tuple(a + b for a, b in zip(left.branches, right.branches)))


def latent_subtract(left: LatentTuple, right: LatentTuple) -> LatentTuple:
    return LatentTuple(tuple(a - b for a, b in zip(left.branches, right.branches)))


def latent_scale(value: LatentTuple, scale: torch.Tensor | float) -> LatentTuple:
    return LatentTuple(tuple(branch * scale for branch in value.branches))


def latent_dot(left: LatentTuple, right: LatentTuple) -> torch.Tensor:
    return sum((a * b).sum() for a, b in zip(left.branches, right.branches))


def latent_numel(value: LatentTuple) -> int:
    return sum(branch.numel() for branch in value.branches)


def latent_rademacher(
    widths: list[int], experts: int, seed: int, device: str
) -> LatentTuple:
    return LatentTuple(
        tuple(
            rademacher((experts, width), int(seed) + 104729 * order, device)
            * math.sqrt(float(width))
            for order, width in enumerate(widths)
        )
    )


def pair_rademacher(reference: PairedTensor, seed: int, device: str) -> PairedTensor:
    return PairedTensor(
        rademacher(tuple(reference.c_fc.shape), seed, device)
        * math.sqrt(float(reference.c_fc.shape[-1])),
        rademacher(tuple(reference.c_proj.shape), seed + 104729, device)
        * math.sqrt(float(reference.c_proj.shape[-1])),
    )


def coordinate_accounting(plan: dict[str, Any]) -> dict[str, float | int]:
    source, candidate = plan["source"], plan["candidate"]
    per_expert = sum(int(value) for value in candidate["coordinate_split_by_polynomial_order"])
    compact = int(source["tensor_layers"]) * int(source["num_experts"]) * per_expert
    paired_per_expert = 2 * int(source["input_width"]) * int(source["expert_hidden_width"])
    dense = int(source["tensor_layers"]) * int(source["num_experts"]) * paired_per_expert
    return {
        "per_expert": per_expert,
        "per_layer": int(source["num_experts"]) * per_expert,
        "compact": compact,
        "dense": dense,
        "compression": dense / compact,
        "paired_parameters_per_expert": paired_per_expert,
    }


def route_and_sample_with_probabilities(
    state: LayerState,
    activations: torch.Tensor,
    *,
    top_k: int,
    samples_per_expert: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    x = activations.float().cpu()
    logits = x @ state.router.float().cpu().T
    tie = torch.arange(logits.shape[-1], dtype=logits.dtype)
    selected = torch.topk(
        logits - tie * torch.finfo(logits.dtype).eps,
        int(top_k),
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
    sampled_inputs: list[torch.Tensor] = []
    sampled_probabilities: list[torch.Tensor] = []
    counts: list[int] = []
    for expert in range(state.c_fc.shape[0]):
        locations = (selected == expert).nonzero(as_tuple=False)
        count = int(locations.shape[0])
        counts.append(count)
        if count < int(samples_per_expert):
            raise RuntimeError(
                f"expert {expert} has {count} assignments, below required "
                f"{samples_per_expert}"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 131 * expert)
        chosen = torch.randperm(count, generator=generator)[: int(samples_per_expert)]
        picked = locations.index_select(0, chosen)
        tokens, slots = picked[:, 0], picked[:, 1]
        sampled_inputs.append(x.index_select(0, tokens))
        sampled_probabilities.append(probabilities[tokens, slots])
    return torch.stack(sampled_inputs), torch.stack(sampled_probabilities), counts


class PairedEmpiricalFisher:
    """Matrix-free routed paired expert F=J^T J / samples."""

    def __init__(
        self,
        inputs: torch.Tensor,
        probabilities: torch.Tensor,
        teacher: PairedTensor,
        *,
        device: str,
    ) -> None:
        self.device = device
        self.inputs = inputs.to(device=device, dtype=torch.float32)
        self.probabilities = probabilities.to(device=device, dtype=torch.float32)
        self.teacher = teacher.to(device)
        if self.inputs.ndim != 3 or self.probabilities.shape != self.inputs.shape[:2]:
            raise ValueError("Fisher input/probability shapes disagree")
        experts, _samples, width = self.inputs.shape
        if self.teacher.c_fc.shape[0] != experts or self.teacher.c_fc.shape[-1] != width:
            raise ValueError("Fisher teacher shape disagrees with routed inputs")
        pre = torch.bmm(self.inputs, self.teacher.c_fc.transpose(1, 2))
        self.hidden = F.gelu(pre)
        self.gelu_prime = gelu_derivative(pre)
        self.samples = int(self.inputs.shape[1])
        self.largest_eigenvalue = 1.0

    def jvp(self, direction: PairedTensor) -> torch.Tensor:
        direction = direction.to(self.device)
        delta_pre = torch.bmm(self.inputs, direction.c_fc.transpose(1, 2))
        from_fc = torch.bmm(
            self.gelu_prime * delta_pre,
            self.teacher.c_proj.transpose(1, 2),
        )
        from_proj = torch.bmm(self.hidden, direction.c_proj.transpose(1, 2))
        return self.probabilities[:, :, None] * (from_fc + from_proj)

    def vjp(self, cotangent: torch.Tensor) -> PairedTensor:
        live = cotangent.to(device=self.device, dtype=torch.float32)
        routed = self.probabilities[:, :, None] * live
        hidden_cotangent = torch.bmm(routed, self.teacher.c_proj) * self.gelu_prime
        c_fc = torch.bmm(hidden_cotangent.transpose(1, 2), self.inputs)
        c_proj = torch.bmm(routed.transpose(1, 2), self.hidden)
        return PairedTensor(c_fc, c_proj)

    def apply(self, direction: PairedTensor) -> PairedTensor:
        return pair_scale(self.vjp(self.jvp(direction)), 1.0 / float(self.samples))

    def apply_normalized(self, direction: PairedTensor) -> PairedTensor:
        return pair_scale(self.apply(direction), 1.0 / float(self.largest_eigenvalue))

    def estimate_largest_eigenvalue(
        self, reference: PairedTensor, *, seed: int, iterations: int, epsilon: float
    ) -> dict[str, float | int]:
        vector = pair_rademacher(reference, seed, self.device)
        vector = pair_scale(vector, 1.0 / pair_norm(vector).clamp_min(float(epsilon)))
        rayleigh = torch.tensor(0.0, device=self.device)
        for _ in range(int(iterations)):
            image = self.apply(vector)
            rayleigh = pair_dot(vector, image) / pair_dot(vector, vector).clamp_min(float(epsilon))
            vector = pair_scale(image, 1.0 / pair_norm(image).clamp_min(float(epsilon)))
        self.largest_eigenvalue = max(float(rayleigh), float(epsilon))
        return {
            "iterations": int(iterations),
            "largest_eigenvalue": self.largest_eigenvalue,
        }


class ProceduralPairedMap:
    """A normalized repeated-block BlockFHT map for all experts in one layer."""

    def __init__(
        self,
        *,
        experts: int,
        input_width: int,
        hidden_width: int,
        latent_width: int,
        layers: int,
        seed: int,
        layer: int,
        bank_index: int,
        polynomial_order: int,
        device: str,
    ) -> None:
        self.experts = int(experts)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.latent_width = int(latent_width)
        self.layers = int(layers)
        self.base_seed = int(seed)
        self.layer = int(layer)
        self.bank_index = int(bank_index)
        self.polynomial_order = int(polynomial_order)
        self.device = device
        self.fc_size = self.input_width * self.hidden_width
        self.pair_size = 2 * self.fc_size
        self.block_size = next_power_of_two(self.latent_width)
        self.scale = math.sqrt(float(self.block_size) / float(self.pair_size))
        self.reference = torch.zeros(self.latent_width, device=device)

    def seed(self, expert: int) -> int:
        return (
            self.base_seed
            + 1000003 * self.bank_index
            + 1009 * self.layer
            + 131 * int(expert)
            + 104729 * self.polynomial_order
        )

    def apply(self, coordinates: torch.Tensor) -> PairedTensor:
        if tuple(coordinates.shape) != (self.experts, self.latent_width):
            raise ValueError("procedural-map coordinate shape drift")
        c_fc, c_proj = [], []
        for expert in range(self.experts):
            flat = self.scale * block_fht_slice(
                coordinates[expert], self.pair_size, self.layers,
                self.seed(expert), 0, self.pair_size,
            )
            c_fc.append(flat[: self.fc_size].reshape(self.hidden_width, self.input_width))
            c_proj.append(flat[self.fc_size :].reshape(self.input_width, self.hidden_width))
        return PairedTensor(torch.stack(c_fc), torch.stack(c_proj))

    def adjoint(self, cotangent: PairedTensor) -> torch.Tensor:
        cotangent = cotangent.to(self.device)
        result = []
        for expert in range(self.experts):
            flat = torch.cat(
                (cotangent.c_fc[expert].reshape(-1), cotangent.c_proj[expert].reshape(-1))
            )
            result.append(
                block_fht_grad_latent(
                    self.reference,
                    self.scale * flat,
                    self.pair_size,
                    self.layers,
                    self.seed(expert),
                )
            )
        return torch.stack(result)


class FisherKrylovDecoder:
    def __init__(
        self,
        plan: dict[str, Any],
        fisher: PairedEmpiricalFisher,
        *,
        bank_index: int,
        layer: int,
        candidate: bool,
        device: str,
    ) -> None:
        source, config = plan["source"], plan["candidate"]
        amendment = plan["implementation_protocol_amendment_before_implementation_or_candidate_values"]
        self.fisher = fisher
        self.candidate = bool(candidate)
        self.widths = [int(value) for value in config["coordinate_split_by_polynomial_order"]]
        self.experts = int(source["num_experts"])
        self.device = device
        self.maps = tuple(
            ProceduralPairedMap(
                experts=self.experts,
                input_width=int(source["input_width"]),
                hidden_width=int(source["expert_hidden_width"]),
                latent_width=width,
                layers=int(amendment["procedural_map"]["block_fht_layers"]),
                seed=int(config["procedural_seed_base"]),
                layer=int(layer),
                bank_index=int(bank_index),
                polynomial_order=order,
                device=device,
            )
            for order, width in enumerate(self.widths)
        )

    def zeros(self) -> LatentTuple:
        return latent_zeros(self.widths, self.experts, self.device)

    def apply(self, coordinates: LatentTuple) -> PairedTensor:
        raw = [mapping.apply(value) for mapping, value in zip(self.maps, coordinates.branches)]
        if not self.candidate:
            return pair_add(pair_add(raw[0], raw[1]), raw[2])
        first = self.fisher.apply_normalized(raw[1])
        second = self.fisher.apply_normalized(raw[2])
        second = self.fisher.apply_normalized(second)
        return pair_add(pair_add(raw[0], first), second)

    def adjoint(self, cotangent: PairedTensor) -> LatentTuple:
        if not self.candidate:
            return LatentTuple(tuple(mapping.adjoint(cotangent) for mapping in self.maps))
        first = self.fisher.apply_normalized(cotangent)
        second = self.fisher.apply_normalized(first)
        return LatentTuple(
            (
                self.maps[0].adjoint(cotangent),
                self.maps[1].adjoint(first),
                self.maps[2].adjoint(second),
            )
        )

    def normal_without_ridge(self, coordinates: LatentTuple) -> LatentTuple:
        return self.adjoint(self.fisher.apply_normalized(self.apply(coordinates)))


def estimate_average_normal_diagonal(
    decoder: FisherKrylovDecoder, *, seed: int, probes: int = 2
) -> float:
    estimates = []
    for probe in range(int(probes)):
        vector = latent_rademacher(
            decoder.widths, decoder.experts, int(seed) + 104729 * probe, decoder.device
        )
        image = decoder.normal_without_ridge(vector)
        estimates.append(float(latent_dot(vector, image) / latent_numel(vector)))
    return max(sum(estimates) / len(estimates), 1e-30)


def fit_decoder(
    decoder: FisherKrylovDecoder,
    target_delta: PairedTensor,
    *,
    relative_damping: float,
    maximum_iterations: int,
    tolerance: float,
    trace_seed: int,
) -> tuple[LatentTuple, dict[str, float | int]]:
    metric_target = decoder.fisher.apply_normalized(target_delta)
    rhs = decoder.adjoint(metric_target)
    average_diagonal = estimate_average_normal_diagonal(decoder, seed=trace_seed)
    ridge = float(relative_damping) * average_diagonal
    solution = decoder.zeros()
    residual = rhs
    search = residual
    residual_squared = latent_dot(residual, residual)
    initial_squared = residual_squared.clamp_min(1e-30)
    relative = 1.0
    iterations = 0
    for iteration in range(int(maximum_iterations)):
        normal = latent_add(
            decoder.normal_without_ridge(search), latent_scale(search, ridge)
        )
        denominator = latent_dot(search, normal)
        if not torch.isfinite(denominator) or float(denominator) <= 0.0:
            raise RuntimeError("non-positive or non-finite latent normal curvature")
        step = residual_squared / denominator
        solution = latent_add(solution, latent_scale(search, step))
        updated = latent_subtract(residual, latent_scale(normal, step))
        updated_squared = latent_dot(updated, updated)
        relative = float(torch.sqrt(updated_squared / initial_squared))
        iterations = iteration + 1
        residual = updated
        if relative <= float(tolerance):
            residual_squared = updated_squared
            break
        beta = updated_squared / residual_squared.clamp_min(1e-30)
        search = latent_add(residual, latent_scale(search, beta))
        residual_squared = updated_squared
    predicted = decoder.apply(solution)
    error = pair_subtract(predicted, target_delta)
    error_energy = pair_dot(error, decoder.fisher.apply_normalized(error))
    target_energy = pair_dot(target_delta, metric_target).clamp_min(1e-30)
    return solution, {
        "iterations": iterations,
        "relative_normal_residual": relative,
        "average_normal_diagonal": average_diagonal,
        "ridge": ridge,
        "fisher_metric_recovery": float(1.0 - error_energy / target_energy),
        "coordinate_l2_norm": float(torch.sqrt(latent_dot(solution, solution))),
    }


@torch.no_grad()
def routed_evaluation(
    route_state: LayerState,
    target_state: LayerState,
    predicted: PairedTensor,
    activations: torch.Tensor,
    *,
    outer_top_k: int,
    probe_seed: int,
    device: str,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    route_state, target_state = route_state.to(device), target_state.to(device)
    predicted = predicted.to(device)
    all_directions = rademacher(tuple(activations.shape), probe_seed, "cpu")
    predicted_chunks, target_chunks = [], []
    predicted_jvp_chunks, target_jvp_chunks = [], []
    experts = int(target_state.c_fc.shape[0])
    expert_error = torch.zeros(experts, dtype=torch.float64)
    expert_energy = torch.zeros_like(expert_error)
    expert_jvp_error = torch.zeros_like(expert_error)
    expert_jvp_energy = torch.zeros_like(expert_error)
    for start in range(0, activations.shape[0], int(chunk_size)):
        stop = min(activations.shape[0], start + int(chunk_size))
        x = activations[start:stop].to(device=device, dtype=torch.float32)
        direction = all_directions[start:stop].to(device=device, dtype=torch.float32)
        logits = x @ route_state.router.T
        tie = torch.arange(logits.shape[-1], device=device, dtype=x.dtype)
        selected = torch.topk(
            logits - tie * torch.finfo(x.dtype).eps,
            int(outer_top_k), dim=-1, largest=True, sorted=True,
        ).indices
        probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
        output, target = torch.zeros_like(x), torch.zeros_like(x)
        output_jvp, target_jvp = torch.zeros_like(x), torch.zeros_like(x)
        for expert in range(experts):
            locations = (selected == expert).nonzero(as_tuple=False)
            if not locations.numel():
                continue
            token, slot = locations[:, 0], locations[:, 1]
            expert_input = x.index_select(0, token)[None]
            expert_direction = direction.index_select(0, token)[None]
            value, value_jvp = dense_function_and_jvp(
                expert_input,
                expert_direction,
                predicted.c_fc[expert : expert + 1],
                predicted.c_proj[expert : expert + 1].transpose(1, 2),
            )
            teacher, teacher_jvp = dense_function_and_jvp(
                expert_input,
                expert_direction,
                target_state.c_fc[expert : expert + 1],
                target_state.c_proj[expert : expert + 1].transpose(1, 2),
            )
            weight = probabilities[token, slot, None]
            output.index_add_(0, token, value[0] * weight)
            target.index_add_(0, token, teacher[0] * weight)
            output_jvp.index_add_(0, token, value_jvp[0] * weight)
            target_jvp.index_add_(0, token, teacher_jvp[0] * weight)
            expert_error[expert] += float((value - teacher).square().sum())
            expert_energy[expert] += float(teacher.square().sum())
            expert_jvp_error[expert] += float((value_jvp - teacher_jvp).square().sum())
            expert_jvp_energy[expert] += float(teacher_jvp.square().sum())
        predicted_chunks.append(output.cpu())
        target_chunks.append(target.cpu())
        predicted_jvp_chunks.append(output_jvp.cpu())
        target_jvp_chunks.append(target_jvp.cpu())
    output, target = torch.cat(predicted_chunks), torch.cat(target_chunks)
    output_jvp, target_jvp = torch.cat(predicted_jvp_chunks), torch.cat(target_jvp_chunks)
    return {
        "predicted": output,
        "target": target,
        "predicted_jvp": output_jvp,
        "target_jvp": target_jvp,
        "mixture_recovery": recovery_fraction(output, target),
        "jvp_recovery": recovery_fraction(output_jvp, target_jvp),
        "expert_recovery": [
            1.0 - float(error / max(energy, 1e-30))
            for error, energy in zip(expert_error, expert_energy)
        ],
        "expert_jvp_recovery": [
            1.0 - float(error / max(energy, 1e-30))
            for error, energy in zip(expert_jvp_error, expert_jvp_energy)
        ],
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("paired Fisher-Krylov plan schema mismatch")
    counts = coordinate_accounting(plan)
    candidate, source = plan["candidate"], plan["source"]
    if counts["per_expert"] != int(candidate["coordinates_per_expert"]):
        raise ValueError("per-expert coordinate accounting drift")
    if counts["per_layer"] != int(candidate["coordinates_per_layer"]):
        raise ValueError("per-layer coordinate accounting drift")
    if counts["compact"] != int(candidate["total_coordinates_all_layers"]):
        raise ValueError("total compact coordinate accounting drift")
    if counts["dense"] != int(candidate["dense_paired_parameters_all_layers"]):
        raise ValueError("dense paired accounting drift")
    if not math.isclose(
        float(counts["compression"]),
        float(candidate["paired_coordinate_compression_ratio"]),
        rel_tol=1e-12,
    ):
        raise ValueError("paired compression accounting drift")
    root = Path(__file__).resolve().parents[2]
    for key, path_key, hash_key in (
        ("fullrank", "fullrank_gated_result", "fullrank_gated_result_sha256"),
        ("kfac", "matrix_normal_kfac_result", "matrix_normal_kfac_result_sha256"),
    ):
        del key
        if file_sha256(root / source[path_key]) != source[hash_key]:
            raise ValueError(f"sealed predecessor hash drift: {path_key}")
    identity = plan.get("identity", {})
    if identity.get("entrypoint_sha256") and identity["entrypoint_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash drift")
    for relative, expected in identity.get("helper_sha256", {}).items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    if not file_sha256(plan_path):
        raise AssertionError("empty plan hash")


def verify_stepzero_identity(
    plan: dict[str, Any], stepzero_hashes: dict[str, str]
) -> None:
    root = Path(__file__).resolve().parents[2]
    predecessor = json.loads(
        (root / plan["source"]["matrix_normal_kfac_result"]).read_text(encoding="utf-8")
    )
    expected = predecessor["identity"]["stepzero_parameter_hashes"]
    if stepzero_hashes != expected:
        raise ValueError("exact step-zero reconstruction hash drift")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    experts = int(source["num_experts"])
    width = int(source["input_width"])
    hidden = int(source["expert_hidden_width"])
    samples = int(plan["data_protocol"]["fit_samples_per_expert"])
    generator = torch.Generator(device="cpu").manual_seed(20261861)
    inputs = torch.randn(experts, samples, width, generator=generator)
    probabilities = torch.rand(experts, samples, generator=generator).mul_(0.5).add_(0.5)
    teacher = PairedTensor(
        torch.randn(experts, hidden, width, generator=generator) * 0.02,
        torch.randn(experts, width, hidden, generator=generator)
        * (0.02 / math.sqrt(2.0 * int(source["tensor_layers"]))),
    )
    fisher = PairedEmpiricalFisher(inputs, probabilities, teacher, device=device)
    reference = pair_rademacher(teacher, 20261862, device)
    started = time.time()
    image = fisher.apply(reference)
    fisher_seconds = time.time() - started
    fisher.largest_eigenvalue = max(
        float(pair_dot(reference, image) / pair_dot(reference, reference).clamp_min(1e-30)),
        float(plan["paired_functional_operator"]["normalization_epsilon"]),
    )
    timings, adjoint_errors = {}, {}
    for family_index, candidate in enumerate((True, False)):
        name = "candidate" if candidate else "control"
        decoder = FisherKrylovDecoder(
            plan, fisher, bank_index=0, layer=0, candidate=candidate, device=device
        )
        coordinates = latent_rademacher(decoder.widths, decoder.experts, 20261863, device)
        cotangent = pair_rademacher(teacher, 20261864, device)
        left = pair_dot(decoder.apply(coordinates), cotangent)
        right = latent_dot(coordinates, decoder.adjoint(cotangent))
        adjoint_errors[name] = float((left - right).abs() / left.abs().clamp_min(1e-30))
        started = time.time()
        normal = decoder.normal_without_ridge(coordinates)
        timings[name + "_normal_seconds"] = time.time() - started
        if not all(torch.isfinite(value).all() for value in normal.branches):
            raise RuntimeError("non-finite preflight normal action")
        del decoder, coordinates, cotangent, normal
    candidate_normal = timings["candidate_normal_seconds"]
    control_normal = timings["control_normal_seconds"]
    maximum_iterations = int(plan["fit_protocol"]["maximum_iterations"])
    projected = 6.0 * (
        int(plan["paired_functional_operator"]["power_iterations"]) * fisher_seconds
        + (maximum_iterations + 3) * (candidate_normal + control_normal)
    )
    return {
        "schema_version": "nanogpt_sparse_moe_paired_fisher_krylov_preflight_v1",
        "device": device,
        "registered_samples_per_expert": samples,
        "fisher_action_seconds": fisher_seconds,
        **timings,
        "projected_full_protocol_seconds_upper_bound": projected,
        "adjoint_relative_errors": adjoint_errors,
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
        "all_values_finite": all_finite(
            {"fisher_eigenvalue": fisher.largest_eigenvalue, "adjoint": adjoint_errors}
        ),
        "passed": max(adjoint_errors.values()) <= 2e-4,
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
    torch.set_grad_enabled(False)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, args.plan)
    if args.preflight_only:
        print(json.dumps(run_preflight(plan, args.device), indent=2, sort_keys=True))
        return
    if args.terminal_snapshot is None or args.data_dir is None or args.output is None:
        parser.error("oracle requires --terminal-snapshot, --data-dir, and --output")
    if args.output.exists():
        raise FileExistsError("paired Fisher-Krylov output already exists")

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
    layers = [int(value) for value in source["layers"]]

    stepzero_model = model_from_exact_stepzero(
        payload, int(source["model_seed"]), args.device
    )
    stepzero_hashes = selected_stepzero_hashes(stepzero_model, layers)
    verify_stepzero_identity(plan, stepzero_hashes)
    initial_mapping = dict(stepzero_model.named_parameters())
    initial = {layer: layer_state_from_mapping(initial_mapping, layer) for layer in layers}
    del initial_mapping, stepzero_model
    terminal = {layer: layer_state_from_mapping(payload["model"], layer) for layer in layers}

    terminal_model = load_model(args.terminal_snapshot, args.device)
    terminal_model.eval()
    inputs = collect_protocol_inputs(terminal_model, plan, args.data_dir, args.device)
    del terminal_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    banks = [row["name"] for row in plan["data_protocol"]["discovery_banks"]]
    fit = plan["fit_protocol"]
    operator_plan = plan["paired_functional_operator"]
    protocol = plan["data_protocol"]
    saved: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    actions: dict[tuple[str, int], torch.Tensor] = {}

    for bank_index, bank in enumerate(banks):
        saved[bank], summaries[bank], diagnostics[bank], occupancy[bank] = {}, {}, {}, {}
        for layer in layers:
            sampled, probabilities, counts = route_and_sample_with_probabilities(
                terminal[layer], inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=int(protocol["fit_samples_per_expert"]),
                seed=int(protocol["sample_selection_seed_base"]) + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            fisher = PairedEmpiricalFisher(
                sampled, probabilities, pair_from_state(terminal[layer]), device=args.device
            )
            reference = pair_subtract(
                pair_from_state(terminal[layer]).to(args.device),
                pair_from_state(initial[layer]).to(args.device),
            )
            power = fisher.estimate_largest_eigenvalue(
                reference,
                seed=int(operator_plan["power_seed_base"]) + 1009 * bank_index + 17 * layer,
                iterations=int(operator_plan["power_iterations"]),
                epsilon=float(operator_plan["normalization_epsilon"]),
            )
            layer_saved, layer_diagnostics, layer_evaluations = {}, {}, {}
            for family_index, candidate in enumerate((True, False)):
                family = "candidate" if candidate else "control"
                decoder = FisherKrylovDecoder(
                    plan, fisher, bank_index=bank_index, layer=layer,
                    candidate=candidate, device=args.device,
                )
                trace_seed = (
                    int(operator_plan["power_seed_base"]) + 5000003
                    + 1009 * bank_index + 17 * layer + 104729 * family_index
                )
                coordinates, solve = fit_decoder(
                    decoder,
                    reference,
                    relative_damping=float(fit["relative_damping"]),
                    maximum_iterations=int(fit["maximum_iterations"]),
                    tolerance=float(fit["relative_residual_tolerance"]),
                    trace_seed=trace_seed,
                )
                predicted_delta = decoder.apply(coordinates)
                predicted = pair_add(pair_from_state(initial[layer]).to(args.device), predicted_delta)
                evaluation = routed_evaluation(
                    terminal[layer], terminal[layer], predicted, inputs["heldout"][layer],
                    outer_top_k=int(source["outer_moe_top_k"]),
                    probe_seed=int(protocol["heldout_jvp_probe_seed_base"]) + 17 * layer,
                    device=args.device,
                )
                layer_saved[family] = {
                    "coordinates": coordinates.cpu().branches,
                    "predicted_c_fc_sha256": tensor_sha256(predicted.c_fc.cpu()),
                    "predicted_c_proj_sha256": tensor_sha256(predicted.c_proj.cpu()),
                }
                layer_diagnostics[family] = solve
                layer_evaluations[family] = evaluation
                del decoder, coordinates, predicted_delta, predicted
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            candidate_eval, control_eval = (
                layer_evaluations["candidate"], layer_evaluations["control"]
            )
            if not torch.equal(candidate_eval["target"], control_eval["target"]):
                raise RuntimeError("candidate/control heldout target drift")
            actions[(bank, layer)] = candidate_eval["predicted"]
            summaries[bank][str(layer)] = {
                "mixture_recovery": candidate_eval["mixture_recovery"],
                "jvp_recovery": candidate_eval["jvp_recovery"],
                "minimum_expert_recovery": min(candidate_eval["expert_recovery"]),
                "minimum_expert_jvp_recovery": min(candidate_eval["expert_jvp_recovery"]),
                "unfiltered_control_recovery": control_eval["mixture_recovery"],
                "unfiltered_control_jvp_recovery": control_eval["jvp_recovery"],
                "candidate_minus_unfiltered_control_recovery": (
                    candidate_eval["mixture_recovery"] - control_eval["mixture_recovery"]
                ),
                "candidate_minus_unfiltered_control_jvp_recovery": (
                    candidate_eval["jvp_recovery"] - control_eval["jvp_recovery"]
                ),
            }
            diagnostics[bank][str(layer)] = {
                "power": power,
                "candidate": layer_diagnostics["candidate"],
                "control": layer_diagnostics["control"],
            }
            saved[bank][str(layer)] = layer_saved
            del fisher, reference, layer_evaluations
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    frozen = plan["frozen_gates"]
    cross_bank = {
        str(layer): action_cosine(actions[(banks[0], layer)], actions[(banks[1], layer)])
        for layer in layers
    }
    cross_bank_mean = sum(cross_bank.values()) / len(cross_bank)
    finite = all_finite(
        {"summaries": summaries, "diagnostics": diagnostics, "cross_bank": cross_bank}
    )
    gates: dict[str, dict[str, bool]] = {}
    for bank in banks:
        rows = [summaries[bank][str(layer)] for layer in layers]
        aggregate = {
            "mixture_recovery_mean": sum(float(row["mixture_recovery"]) for row in rows) / len(rows),
            "mixture_recovery_minimum_layer": min(float(row["mixture_recovery"]) for row in rows),
            "jvp_recovery_mean": sum(float(row["jvp_recovery"]) for row in rows) / len(rows),
            "minimum_expert_recovery": min(float(row["minimum_expert_recovery"]) for row in rows),
            "candidate_minus_unfiltered_control_recovery_mean": sum(float(row["candidate_minus_unfiltered_control_recovery"]) for row in rows) / len(rows),
            "candidate_minus_unfiltered_control_jvp_recovery_mean": sum(float(row["candidate_minus_unfiltered_control_jvp_recovery"]) for row in rows) / len(rows),
            "minimum_discovery_assignments": min(min(occupancy[bank][str(layer)]) for layer in layers),
            "maximum_latent_normal_relative_residual": max(
                max(
                    float(diagnostics[bank][str(layer)][family]["relative_normal_residual"])
                    for family in ("candidate", "control")
                )
                for layer in layers
            ),
        }
        summaries[bank]["aggregate"] = aggregate
        gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["candidate_heldout_output_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["candidate_heldout_output_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": aggregate["minimum_expert_recovery"] >= float(frozen["candidate_heldout_output_recovery_minimum_expert_each_bank"]),
            "jvp_pass": aggregate["jvp_recovery_mean"] >= float(frozen["candidate_heldout_jvp_recovery_mean_min_each_bank"]),
            "output_control_gain_pass": aggregate["candidate_minus_unfiltered_control_recovery_mean"] >= float(frozen["candidate_minus_unfiltered_control_output_recovery_mean_min_each_bank"]),
            "jvp_control_gain_pass": aggregate["candidate_minus_unfiltered_control_jvp_recovery_mean"] >= float(frozen["candidate_minus_unfiltered_control_jvp_recovery_mean_min_each_bank"]),
            "occupancy_pass": aggregate["minimum_discovery_assignments"] >= int(frozen["minimum_discovery_assignments_per_expert"]),
            "solve_pass": aggregate["maximum_latent_normal_relative_residual"] <= float(frozen["maximum_latent_normal_relative_residual"]),
            "cross_bank_action_pass": cross_bank_mean >= float(frozen["cross_bank_candidate_action_cosine_min"]),
            "finite_pass": finite,
        }
        gates[bank]["all_pass"] = all(gates[bank].values())
    counts = coordinate_accounting(plan)
    compression_pass = float(counts["compression"]) >= float(
        frozen["paired_coordinate_compression_ratio_min"]
    )
    passed = compression_pass and all(gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({"schema_version": COORDINATE_SCHEMA, "states": saved}, coordinates_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": (
            "PAIRED_FISHER_KRYLOV_DECODER_PASSES"
            if passed else "PAIRED_FISHER_KRYLOV_DECODER_REJECTED"
        ),
        "passed": passed,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(manifest),
            "stepzero_parameter_hashes": stepzero_hashes,
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "coordinates_path": str(coordinates_path),
            "coordinates_sha256": file_sha256(coordinates_path),
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
            ),
        },
        "accounting": {
            **counts,
            "compression_pass": compression_pass,
            "fixed_full_matrix_storage": False,
            "learned_dense_basis": False,
            "dense_optimizer_state": False,
            "terminal_teacher_operator_oracle_exception": True,
            "terminal_router_isolation_exception": True,
        },
        "occupancy": occupancy,
        "fit_diagnostics": diagnostics,
        "summaries": summaries,
        "cross_bank_action_cosine_by_layer": cross_bank,
        "cross_bank_action_cosine_mean": cross_bank_mean,
        "gates": gates,
        "all_values_finite": finite,
        "authorization": {
            "causal_transport_theory": bool(passed),
            "causal_decoder_implementation": False,
            "language_model_training": False,
            "larger_rung": False,
            "full_attention_work": False,
            "automatic_retry_or_sweep": False,
        },
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
