#!/usr/bin/env python3
"""Gate a quotient-aware sparse-write chart for complete sparse-MoE MLPs."""
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
    SpectralCFC,
    action_cosine,
    collect_protocol_inputs,
    route_and_sample,
)
from examples.nanogpt.analyze_sparse_moe_conditional_complete_atom_oracle import (
    cpu_state_dict,
    result_authorization,
    routed_evaluation,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
    normalized_expert_loss,
    rademacher,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import LayerState
from examples.nanogpt.analyze_sparse_moe_state_conditioned_butterfly_transport_oracle import (
    _gelu_derivative,
    _mixed_radix_flow_with_jvp,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)
from latent_weight_lab.block_fht import normalized_fht_last_dim


PLAN_SCHEMA = "nanogpt_sparse_moe_paired_sparse_write_chart_oracle_plan_v1"


def support_indices(hidden_width: int = 1536, output_width: int = 768) -> torch.Tensor:
    if hidden_width != 1536 or output_width != 768:
        raise ValueError("registered sparse-write support requires 1536/768 widths")
    triples = ((1, 0, 0), (437, 363, 292), (721, 606, 474))
    result = []
    for hidden in range(hidden_width):
        quotient, remainder = divmod(hidden, output_width)
        result.append([
            (a * remainder + b + quotient * c) % output_width
            for a, b, c in triples
        ])
    indices = torch.tensor(result, dtype=torch.long)
    if any(len(set(row)) != 3 for row in indices.tolist()):
        raise AssertionError("registered sparse-write supports collided")
    for stream in range(3):
        counts = torch.bincount(indices[:, stream], minlength=output_width)
        if not torch.equal(counts, torch.full_like(counts, 2)):
            raise AssertionError("registered sparse-write stream is not balanced")
    return indices


def coordinate_count(
    *, tensor_layers: int, experts: int, hidden_width: int, padded_width: int
) -> dict[str, int | float]:
    cfc = tensor_layers * experts * (2 * padded_width + hidden_width)
    sparse_write = tensor_layers * experts * 3 * hidden_width
    output_flow = tensor_layers * 3840
    compact = cfc + sparse_write + output_flow
    dense = tensor_layers * experts * 2 * 768 * hidden_width
    return {
        "cfc": cfc,
        "sparse_write": sparse_write,
        "output_flow": output_flow,
        "compact": compact,
        "dense": dense,
        "compression": dense / compact,
    }


def _normalized_tensor_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    axes = tuple(range(1, target.ndim))
    scale = target.square().mean(dim=axes).clamp_min(1e-12)
    error = (predicted - target).square().mean(dim=axes)
    return (error / scale).mean()


def _static_flow_with_jvp(
    values: torch.Tensor,
    tangents: torch.Tensor,
    binary_raw: torch.Tensor,
    cross_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = values.shape[:-1]
    binary = ((math.pi / 2.0) * torch.tanh(binary_raw)).expand(
        *prefix, 8, 384
    )
    cross = ((math.pi / 2.0) * torch.tanh(cross_raw)).expand(
        *prefix, 3, 256
    )
    return _mixed_radix_flow_with_jvp(
        values,
        tangents,
        binary,
        torch.zeros_like(binary),
        cross,
        torch.zeros_like(cross),
    )


class PairedSparseWriteChart(torch.nn.Module):
    """Two-spectrum c_fc plus three-sparse writes in a layer-wise flow."""

    def __init__(
        self,
        *,
        experts: int,
        input_width: int,
        hidden_width: int,
        padded_width: int,
        tensor_layers: int,
        seed: int,
        layer: int,
        device: str,
        paired: bool,
        raw_angle_initial_tanh: float = 0.125,
    ) -> None:
        super().__init__()
        self.experts = int(experts)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.padded_width = int(padded_width)
        self.tensor_layers = int(tensor_layers)
        self.paired = bool(paired)
        if self.input_width != 768 or self.hidden_width != 1536:
            raise ValueError("registered paired chart requires 768/1536 widths")
        self.operator = SpectralCFC(
            experts=self.experts,
            input_width=self.input_width,
            hidden_width=self.hidden_width,
            padded_width=self.padded_width,
            seed=int(seed),
            layer=int(layer),
            device=device,
            context_beta=0.0,
        )
        self.spectral_1 = torch.nn.Parameter(
            torch.zeros(self.experts, self.padded_width)
        )
        self.spectral_2 = torch.nn.Parameter(
            torch.zeros(self.experts, self.padded_width)
        )
        self.hidden_bias = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width)
        )
        self.write_coefficients = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width, 3)
        )
        initial_raw = math.atanh(float(raw_angle_initial_tanh))
        self.output_binary_raw = torch.nn.Parameter(
            torch.full((1, 1, 8, 384), initial_raw)
        )
        self.output_cross_raw = torch.nn.Parameter(
            torch.full((1, 1, 3, 256), initial_raw)
        )
        self.register_buffer(
            "write_support", support_indices(self.hidden_width, self.input_width)
        )
        self.to(device=device, dtype=torch.float32)

    def trainable_parameters(self, *, paired: bool) -> list[torch.nn.Parameter]:
        if bool(paired) != self.paired:
            raise ValueError("paired flag disagrees with constructed chart")
        return [
            self.spectral_1,
            self.spectral_2,
            self.hidden_bias,
            self.write_coefficients,
            self.output_binary_raw,
            self.output_cross_raw,
        ]

    def compact_parameter_count(self, *, paired: bool) -> int:
        return sum(
            parameter.numel()
            for parameter in self.trainable_parameters(paired=paired)
        )

    def _selection(self, expert: int | None) -> slice:
        if expert is None:
            return slice(None)
        if not 0 <= int(expert) < self.experts:
            raise IndexError("expert index out of range")
        return slice(int(expert), int(expert) + 1)

    def preactivation_and_jvp(
        self,
        inputs: torch.Tensor,
        directions: torch.Tensor,
        *,
        selected: slice,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signs = self.operator.signs[selected]
        values = F.pad(inputs.to(dtype=torch.float32), (0, self.padded_width - self.input_width))
        tangent = F.pad(directions.to(dtype=torch.float32), (0, self.padded_width - self.input_width))
        values = normalized_fht_last_dim(values * signs[:, 0, None, :])
        tangent = normalized_fht_last_dim(tangent * signs[:, 0, None, :])
        first = 1.0 + self.spectral_1[selected, None, :]
        second = 1.0 + self.spectral_2[selected, None, :]
        values = normalized_fht_last_dim(values * first * signs[:, 1, None, :])
        tangent = normalized_fht_last_dim(tangent * first * signs[:, 1, None, :])
        values = normalized_fht_last_dim(values * second * signs[:, 2, None, :])
        tangent = normalized_fht_last_dim(tangent * second * signs[:, 2, None, :])
        pre = self.operator.base_scale * values[..., : self.hidden_width]
        pre_jvp = self.operator.base_scale * tangent[..., : self.hidden_width]
        return pre + self.hidden_bias[selected, None, :], pre_jvp

    def _scatter_writes(
        self,
        hidden: torch.Tensor,
        hidden_jvp: torch.Tensor,
        *,
        selected: slice,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count, samples = hidden.shape[:2]
        indices = self.write_support.reshape(1, 1, -1).expand(
            count, samples, -1
        )
        coefficients = self.write_coefficients[selected, None, :, :]
        values = (hidden[..., None] * coefficients).reshape(count, samples, -1)
        tangent_values = (
            hidden_jvp[..., None] * coefficients
        ).reshape(count, samples, -1)
        canonical = torch.zeros(
            count, samples, self.input_width,
            device=hidden.device, dtype=hidden.dtype,
        )
        canonical_jvp = torch.zeros_like(canonical)
        canonical.scatter_add_(-1, indices, values)
        canonical_jvp.scatter_add_(-1, indices, tangent_values)
        return canonical, canonical_jvp

    def function_details(
        self,
        inputs: torch.Tensor,
        directions: torch.Tensor,
        *,
        paired: bool,
        expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bool(paired) != self.paired:
            raise ValueError("paired flag disagrees with constructed chart")
        if inputs.shape != directions.shape:
            raise ValueError("input and direction shape mismatch")
        selected = self._selection(expert)
        expected = self.experts if expert is None else 1
        if inputs.shape[0] != expected or inputs.shape[-1] != self.input_width:
            raise ValueError("input shape disagrees with chart")
        pre, pre_jvp = self.preactivation_and_jvp(
            inputs.float(), directions.float(), selected=selected
        )
        hidden = F.gelu(pre)
        hidden_jvp = _gelu_derivative(pre) * pre_jvp
        canonical, canonical_jvp = self._scatter_writes(
            hidden, hidden_jvp, selected=selected
        )
        output, output_jvp = _static_flow_with_jvp(
            canonical,
            canonical_jvp,
            self.output_binary_raw,
            self.output_cross_raw,
        )
        return output, output_jvp, hidden

    def function_and_jvp(
        self,
        inputs: torch.Tensor,
        directions: torch.Tensor,
        *,
        conditional: bool,
        expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, output_jvp, _hidden = self.function_details(
            inputs, directions, paired=conditional, expert=expert
        )
        return output, output_jvp

    def generated_write_columns(self) -> torch.Tensor:
        canonical = torch.zeros(
            self.experts,
            self.hidden_width,
            self.input_width,
            device=self.write_coefficients.device,
            dtype=self.write_coefficients.dtype,
        )
        indices = self.write_support[None, :, :].expand(self.experts, -1, -1)
        canonical.scatter_add_(-1, indices, self.write_coefficients)
        generated, _ = _static_flow_with_jvp(
            canonical,
            torch.zeros_like(canonical),
            self.output_binary_raw,
            self.output_cross_raw,
        )
        return generated


def identity_permutation(experts: int, hidden_width: int, device: str) -> torch.Tensor:
    return torch.arange(hidden_width, device=device).expand(experts, -1).clone()


@torch.no_grad()
def dense_atom_targets(
    inputs: torch.Tensor,
    dense_c_fc: torch.Tensor,
    dense_c_proj: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    live = inputs.to(device=device, dtype=torch.float32)
    pre = torch.bmm(
        live, dense_c_fc.to(device=device, dtype=torch.float32).transpose(1, 2)
    )
    hidden = F.gelu(pre)
    writes = dense_c_proj.to(device=device, dtype=torch.float32).transpose(1, 2)
    return hidden, writes


def permute_atoms(values: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or values.shape[0] != permutation.shape[0]:
        raise ValueError("atom tensor and permutation disagree")
    if values.shape[1] == permutation.shape[1]:
        # [expert, hidden, feature]
        return values.gather(
            1, permutation[..., None].expand(-1, -1, values.shape[-1])
        )
    if values.shape[2] == permutation.shape[1]:
        # [expert, sample, hidden]
        return values.gather(
            2, permutation[:, None, :].expand(-1, values.shape[1], -1)
        )
    raise ValueError("registered atom axis does not match the permutation")


def fit_chart(
    module: PairedSparseWriteChart,
    inputs: torch.Tensor,
    dense_c_fc: torch.Tensor,
    dense_c_proj: torch.Tensor,
    permutation: torch.Tensor,
    *,
    paired: bool,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    jvp_weight: float,
    activation_alignment_weight: float,
    write_alignment_weight: float,
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
        dense_hidden, dense_writes = dense_atom_targets(
            live_inputs, dense_c_fc, dense_c_proj, device
        )
        target_hidden = permute_atoms(dense_hidden, permutation)
        target_writes = permute_atoms(dense_writes, permutation)
    parameters = module.trainable_parameters(paired=paired)
    optimizer = torch.optim.AdamW(
        parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    traces: dict[str, list[float]] = {
        "loss": [], "output": [], "jvp": [], "activation": [], "write": []
    }
    maximum_gradient = 0.0
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        output, output_jvp, hidden = module.function_details(
            live_inputs, directions, paired=paired
        )
        writes = module.generated_write_columns()
        output_loss = normalized_expert_loss(output, target_output)
        jvp_loss = normalized_expert_loss(output_jvp, target_jvp)
        activation_loss = _normalized_tensor_loss(hidden, target_hidden)
        write_loss = _normalized_tensor_loss(writes, target_writes)
        loss = (
            output_loss
            + float(jvp_weight) * jvp_loss
            + float(activation_alignment_weight) * activation_loss
            + float(write_alignment_weight) * write_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite paired sparse-write objective")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise RuntimeError("non-finite or missing paired-chart gradient")
        gradient = float(
            torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
        )
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        for name, value in (
            ("loss", loss), ("output", output_loss), ("jvp", jvp_loss),
            ("activation", activation_loss), ("write", write_loss),
        ):
            traces[name].append(float(value.detach()))
    result: dict[str, Any] = {
        "paired": bool(paired),
        "steps": int(steps),
        "maximum_preclip_gradient_norm": maximum_gradient,
    }
    for name, values in traces.items():
        result[f"initial_{name}_loss"] = values[0]
        result[f"final_{name}_loss"] = values[-1]
        result[f"minimum_{name}_loss"] = min(values)
    return result


@torch.no_grad()
def derive_joint_permutation(
    module: PairedSparseWriteChart,
    inputs: torch.Tensor,
    dense_c_fc: torch.Tensor,
    dense_c_proj: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - PRO6 runtime dependency
        raise RuntimeError("scipy is required for the frozen exact assignment") from error
    device = str(module.hidden_bias.device)
    live_inputs = inputs.to(device=device, dtype=torch.float32)
    zeros = torch.zeros_like(live_inputs)
    _out, _jvp, compact_hidden = module.function_details(
        live_inputs, zeros, paired=module.paired
    )
    compact_writes = module.generated_write_columns()
    dense_hidden, dense_writes = dense_atom_targets(
        live_inputs, dense_c_fc, dense_c_proj, device
    )
    permutations: list[torch.Tensor] = []
    scores: list[float] = []
    for expert in range(module.experts):
        target_activation = dense_hidden[expert].transpose(0, 1)
        compact_activation = compact_hidden[expert].transpose(0, 1)
        target_activation = F.normalize(target_activation, dim=-1, eps=1e-12)
        compact_activation = F.normalize(compact_activation, dim=-1, eps=1e-12)
        activation_score = target_activation @ compact_activation.T
        target_write = F.normalize(dense_writes[expert], dim=-1, eps=1e-12)
        compact_write = F.normalize(compact_writes[expert], dim=-1, eps=1e-12)
        write_score = target_write @ compact_write.T
        score = 0.5 * (activation_score + write_score)
        size = score.shape[0]
        row = torch.arange(size, device=device, dtype=torch.float64)[:, None]
        column = torch.arange(size, device=device, dtype=torch.float64)[None, :]
        tie = 1e-12 * torch.remainder((row + 1.0) * (column + 1.0), 104729.0)
        row_index, column_index = linear_sum_assignment(
            (-score.double() + tie).cpu().numpy()
        )
        row_tensor = torch.from_numpy(row_index).long()
        column_tensor = torch.from_numpy(column_index).long()
        permutation = torch.empty(size, dtype=torch.long)
        permutation[column_tensor] = row_tensor
        permutations.append(permutation)
        scores.append(float(score[
            row_tensor.to(device), column_tensor.to(device)
        ].mean()))
    result = torch.stack(permutations).to(device)
    expected = torch.arange(module.hidden_width, device=device)
    if any(not torch.equal(torch.sort(row).values, expected) for row in result):
        raise RuntimeError("assignment solver did not produce a permutation")
    return result, {
        "mean_matched_signature_cosine": sum(scores) / len(scores),
        "minimum_expert_matched_signature_cosine": min(scores),
        "fraction_moved": float(
            (result != expected[None, :]).float().mean()
        ),
    }


def make_module(
    plan: dict[str, Any], layer: int, device: str, *, paired: bool
) -> PairedSparseWriteChart:
    source = plan["source"]
    return PairedSparseWriteChart(
        experts=int(source["num_experts"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        padded_width=2048,
        tensor_layers=int(source["tensor_layers"]),
        seed=20261213,
        layer=int(layer),
        device=device,
        paired=paired,
        raw_angle_initial_tanh=0.125,
    )


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("paired sparse-write plan schema mismatch")
    root = Path(__file__).resolve().parents[2]
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed")
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    for control in plan["sealed_controls"].values():
        if file_sha256(root / control["path"]) != control["sha256"]:
            raise ValueError(f"sealed control hash drift: {control['path']}")
    source, candidate = plan["source"], plan["candidate"]
    counts = coordinate_count(
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        hidden_width=int(source["expert_hidden_width"]),
        padded_width=2048,
    )
    for key, plan_key in (
        ("cfc", "learned_cfc_coordinates_all_layers"),
        ("sparse_write", "learned_sparse_cproj_coordinates_all_layers"),
        ("output_flow", "learned_output_flow_coordinates_all_layers"),
        ("compact", "total_coordinates_all_layers"),
        ("dense", "dense_paired_parameters_all_layers"),
    ):
        if int(counts[key]) != int(candidate[plan_key]):
            raise ValueError(f"paired chart {key} accounting drift")
    if abs(float(counts["compression"]) - float(candidate["paired_parameter_compression_ratio"])) > 1e-12:
        raise ValueError("paired chart compression drift")
    if float(counts["compression"]) < 200.0:
        raise ValueError("paired chart is outside compression budget")
    support_indices()
    if not file_sha256(plan_path):
        raise AssertionError("empty plan hash")


def _fit_kwargs(plan: dict[str, Any], *, steps: int, probe_seed: int) -> dict[str, Any]:
    fit = plan["fit_protocol"]
    return {
        "steps": int(steps),
        "learning_rate": float(fit["learning_rate"]),
        "weight_decay": float(fit["weight_decay"]),
        "gradient_clip": float(fit["gradient_clip"]),
        "jvp_weight": float(fit["jvp_weight"]),
        "activation_alignment_weight": float(fit["activation_alignment_weight"]),
        "write_alignment_weight": float(fit["write_alignment_weight"]),
        "probe_seed": int(probe_seed),
    }


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    generator = torch.Generator(device="cpu").manual_seed(20261214)
    inputs = torch.randn(
        int(source["num_experts"]), 16, int(source["input_width"]),
        generator=generator,
    )
    c_fc = torch.randn(
        int(source["num_experts"]), int(source["expert_hidden_width"]),
        int(source["input_width"]), generator=generator,
    ) * 0.02
    c_proj = torch.randn(
        int(source["num_experts"]), int(source["input_width"]),
        int(source["expert_hidden_width"]), generator=generator,
    ) * (0.02 / math.sqrt(2.0 * int(source["tensor_layers"])))
    identity = identity_permutation(
        int(source["num_experts"]), int(source["expert_hidden_width"]), device
    )
    started = time.time()
    warmup = make_module(plan, 0, device, paired=False)
    warmup_diag = fit_chart(
        warmup, inputs, c_fc, c_proj, identity, paired=False,
        **_fit_kwargs(plan, steps=2, probe_seed=20261215),
    )
    permutation, assignment = derive_joint_permutation(
        warmup, inputs, c_fc, c_proj
    )
    candidate = make_module(plan, 0, device, paired=True)
    control = make_module(plan, 0, device, paired=False)
    if any(not torch.equal(a, b) for a, b in zip(candidate.state_dict().values(), control.state_dict().values())):
        raise RuntimeError("candidate/control initialization drift")
    candidate_diag = fit_chart(
        candidate, inputs, c_fc, c_proj, permutation, paired=True,
        **_fit_kwargs(plan, steps=2, probe_seed=20261216),
    )
    control_diag = fit_chart(
        control, inputs, c_fc, c_proj, identity, paired=False,
        **_fit_kwargs(plan, steps=2, probe_seed=20261216),
    )
    elapsed = time.time() - started
    full_steps = int(plan["fit_protocol"]["warmup_steps"]) + 2 * int(
        plan["fit_protocol"]["post_assignment_steps"]
    )
    count = candidate.compact_parameter_count(paired=True)
    return {
        "schema_version": "nanogpt_sparse_moe_paired_sparse_write_chart_preflight_v1",
        "device": device,
        "two_step_wall_seconds_warmup_candidate_control_and_assignment": elapsed,
        "projected_full_protocol_seconds": elapsed * full_steps / 6.0 * 6.0,
        "candidate_coordinate_count_per_layer": count,
        "control_coordinate_count_per_layer": control.compact_parameter_count(paired=False),
        "expected_coordinate_count_per_layer": int(plan["candidate"]["total_coordinates_all_layers"]) // int(source["tensor_layers"]),
        "maximum_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
        "assignment": assignment,
        "all_values_finite": all_finite({
            "warmup": warmup_diag,
            "candidate": candidate_diag,
            "control": control_diag,
            "assignment": assignment,
        }),
        "warmup_diagnostics": warmup_diag,
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
    if args.output.exists():
        raise FileExistsError("paired sparse-write output already exists")

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
    static_result = json.loads(
        (root / plan["sealed_controls"]["state_conditioned_full_rank_result"]["path"]).read_text()
    )
    banks = [row["name"] for row in plan["data_protocol"]["discovery_banks"]]
    samples_per_expert = int(plan["data_protocol"]["fit_samples_per_expert"])
    saved: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    assignments: dict[str, dict[str, dict[str, Any]]] = {}
    actions: dict[tuple[str, int], torch.Tensor] = {}
    for bank_index, bank in enumerate(banks):
        saved[bank], summaries[bank], diagnostics[bank] = {}, {}, {}
        occupancy[bank], assignments[bank] = {}, {}
        for layer in layers:
            state = states[layer]
            sampled, counts = route_and_sample(
                state,
                inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=samples_per_expert,
                seed=20261217 + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            identity = identity_permutation(
                int(source["num_experts"]), int(source["expert_hidden_width"]), args.device
            )
            warmup = make_module(plan, layer, args.device, paired=False)
            warmup_diag = fit_chart(
                warmup, sampled, state.c_fc, state.c_proj, identity, paired=False,
                **_fit_kwargs(
                    plan,
                    steps=int(plan["fit_protocol"]["warmup_steps"]),
                    probe_seed=20261218 + 1009 * bank_index + 17 * layer,
                ),
            )
            permutation, assignment_diag = derive_joint_permutation(
                warmup, sampled, state.c_fc, state.c_proj
            )
            del warmup
            candidate = make_module(plan, layer, args.device, paired=True)
            control = make_module(plan, layer, args.device, paired=False)
            if any(not torch.equal(a, b) for a, b in zip(candidate.state_dict().values(), control.state_dict().values())):
                raise RuntimeError("candidate/control initialization drift")
            fit_steps = int(plan["fit_protocol"]["post_assignment_steps"])
            candidate_diag = fit_chart(
                candidate, sampled, state.c_fc, state.c_proj, permutation,
                paired=True,
                **_fit_kwargs(
                    plan, steps=fit_steps,
                    probe_seed=20261219 + 1009 * bank_index + 17 * layer,
                ),
            )
            control_diag = fit_chart(
                control, sampled, state.c_fc, state.c_proj, identity,
                paired=False,
                **_fit_kwargs(
                    plan, steps=fit_steps,
                    probe_seed=20261219 + 1009 * bank_index + 17 * layer,
                ),
            )
            candidate_eval = routed_evaluation(
                state, inputs["heldout"][layer], candidate,
                conditional=True,
                outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=20261220 + 17 * layer,
            )
            control_eval = routed_evaluation(
                state, inputs["heldout"][layer], control,
                conditional=False,
                outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=20261220 + 17 * layer,
            )
            if not torch.equal(candidate_eval["target"], control_eval["target"]):
                raise RuntimeError("candidate/control dense function target drift")
            actions[(bank, layer)] = candidate_eval["predicted"]
            sealed_static = float(
                static_result["summaries"][bank][str(layer)]["static_control_recovery"]
            )
            summaries[bank][str(layer)] = {
                "mixture_recovery": candidate_eval["mixture_recovery"],
                "jvp_recovery": candidate_eval["jvp_recovery"],
                "minimum_expert_recovery": min(candidate_eval["expert_recovery"]),
                "minimum_expert_jvp_recovery": min(candidate_eval["expert_jvp_recovery"]),
                "identity_gauge_control_recovery": control_eval["mixture_recovery"],
                "candidate_minus_identity_gauge_control_recovery": candidate_eval["mixture_recovery"] - control_eval["mixture_recovery"],
                "sealed_static_ceiling_recovery": sealed_static,
                "candidate_minus_sealed_static_ceiling_recovery": candidate_eval["mixture_recovery"] - sealed_static,
            }
            diagnostics[bank][str(layer)] = {
                "warmup": warmup_diag,
                "candidate": candidate_diag,
                "control": control_diag,
            }
            assignments[bank][str(layer)] = assignment_diag
            saved[bank][str(layer)] = {
                "candidate": cpu_state_dict(candidate),
                "control": cpu_state_dict(control),
                "permutation": permutation.detach().cpu(),
            }
            del candidate, control, permutation
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
            "candidate_minus_identity_gauge_control_recovery_mean": sum(float(row["candidate_minus_identity_gauge_control_recovery"]) for row in rows) / len(rows),
            "candidate_minus_sealed_static_ceiling_recovery_mean": sum(float(row["candidate_minus_sealed_static_ceiling_recovery"]) for row in rows) / len(rows),
            "minimum_discovery_assignments": min(min(occupancy[bank][str(layer)]) for layer in layers),
        }
        summaries[bank]["aggregate"] = aggregate
        gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": aggregate["minimum_expert_recovery"] >= float(frozen["heldout_expert_recovery_min_each_bank"]),
            "jvp_pass": aggregate["jvp_recovery_mean"] >= float(frozen["heldout_jvp_recovery_mean_min_each_bank"]),
            "identity_gauge_gain_pass": aggregate["candidate_minus_identity_gauge_control_recovery_mean"] >= float(frozen["candidate_minus_identity_gauge_control_recovery_mean_min_each_bank"]),
            "static_ceiling_gain_pass": aggregate["candidate_minus_sealed_static_ceiling_recovery_mean"] >= float(frozen["candidate_minus_sealed_static_ceiling_recovery_mean_min_each_bank"]),
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
        "assignments": assignments,
        "agreement": agreement_by_layer,
    })
    for bank in banks:
        gates[bank]["action_agreement_pass"] = agreement_mean >= float(frozen["heldout_bank_action_cosine_mean_min"])
        gates[bank]["finite_pass"] = finite
        gates[bank]["compression_pass"] = float(plan["candidate"]["paired_parameter_compression_ratio"]) >= float(frozen["exact_full_mlp_compression_min"])
        gates[bank]["all_pass"] = all(gates[bank].values())
    passed = all(gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_paired_sparse_write_chart_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    result = {
        "schema_version": "nanogpt_sparse_moe_paired_sparse_write_chart_oracle_result_v1",
        "classification": "PAIRED_SPARSE_WRITE_CHART_REPRESENTABILITY_PASSES" if passed else "PAIRED_SPARSE_WRITE_CHART_REPRESENTABILITY_REJECTED",
        "passed": passed,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(manifest),
            "sealed_control_sha256": {
                name: file_sha256(root / row["path"])
                for name, row in plan["sealed_controls"].items()
            },
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
            "dense_learned_basis": False,
            "additive_lora_residual": False,
            "teacher_permutation_deployed": False,
        },
        "occupancy": occupancy,
        "assignment_diagnostics": assignments,
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
