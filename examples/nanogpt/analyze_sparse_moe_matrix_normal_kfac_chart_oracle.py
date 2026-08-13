#!/usr/bin/env python3
"""Gate a full two-sided KFAC matrix-normal chart for complete sparse-MoE MLPs."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_update_alignment import git_commit
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    action_cosine,
    route_and_sample,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
    tensor_sha256,
)
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
    normalized_expert_loss,
    rademacher,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_state_conditioned_butterfly_transport_oracle import (
    _gelu_derivative,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_kfac_factor_oracle import (
    collect_geometry,
    route_assignments,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
    model_from_exact_stepzero,
    selected_stepzero_hashes,
)
from latent_weight_lab.block_fht import block_fht_slice


PLAN_SCHEMA = "nanogpt_sparse_moe_matrix_normal_kfac_chart_oracle_plan_v1"
ROOT_NAMES = ("fc_left", "fc_right", "proj_left", "proj_right")


def coordinate_count(
    *, tensor_layers: int, experts: int, hidden_width: int, latent_width: int
) -> dict[str, int | float]:
    per_expert = 2 * int(latent_width) + int(hidden_width)
    compact = int(tensor_layers) * int(experts) * per_expert
    dense = int(tensor_layers) * int(experts) * 2 * 768 * int(hidden_width)
    return {
        "per_expert": per_expert,
        "compact": compact,
        "dense": dense,
        "compression": dense / compact,
    }


def normalized_covariance_root(
    rows: torch.Tensor,
    weights: torch.Tensor,
    *,
    ridge_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the registered unit-mean-eigenvalue symmetric PSD root."""
    if rows.ndim != 2 or weights.ndim != 1 or rows.shape[0] != weights.numel():
        raise ValueError("covariance rows and weights disagree")
    positive = weights.float().clamp_min(0.0)
    denominator = positive.sum().clamp_min(1e-30)
    values = rows.float()
    covariance = values.T @ (positive[:, None] * values) / denominator
    covariance = 0.5 * (covariance + covariance.T)
    width = covariance.shape[0]
    raw_trace = covariance.diagonal().sum().clamp_min(1e-30)
    normalized = float(width) * covariance / raw_trace
    normalized = normalized + float(ridge_ratio) * torch.eye(
        width, device=normalized.device, dtype=normalized.dtype
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(normalized)
    eigenvalues = eigenvalues.clamp_min(0.0)
    root = (eigenvectors * eigenvalues.sqrt()[None, :]) @ eigenvectors.T
    root = 0.5 * (root + root.T)
    energy = eigenvalues.sum().clamp_min(1e-30)
    probabilities = eigenvalues / energy
    effective_rank = torch.exp(
        -(probabilities.clamp_min(1e-30) * probabilities.clamp_min(1e-30).log()).sum()
    )
    return root, {
        "rows": int(rows.shape[0]),
        "weight_sum": float(denominator),
        "raw_trace": float(raw_trace),
        "normalized_trace": float(eigenvalues.sum()),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "effective_rank": float(effective_rank),
    }


def build_full_kfac_roots(
    state: LayerState,
    inputs: torch.Tensor,
    errors: torch.Tensor,
    *,
    ridge_ratio: float,
    minimum_assignments: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Acquire all four expert-specific KFAC roots from the step-zero state."""
    state = state.to(device)
    inputs = inputs.to(device=device, dtype=torch.float32)
    errors = errors.to(device=device, dtype=torch.float32)
    indices, probabilities = route_assignments(state, inputs, top_k=2)
    roots: dict[str, list[torch.Tensor]] = {name: [] for name in ROOT_NAMES}
    rows: list[dict[str, Any]] = []
    for expert in range(state.c_fc.shape[0]):
        locations = (indices == expert).nonzero(as_tuple=False)
        if int(locations.shape[0]) < int(minimum_assignments):
            raise RuntimeError(
                f"expert {expert} has {locations.shape[0]} assignments, below "
                f"registered minimum {minimum_assignments}"
            )
        token, slot = locations[:, 0], locations[:, 1]
        x = inputs.index_select(0, token)
        p = probabilities[token, slot]
        routed_error = p[:, None] * errors.index_select(0, token)
        pre = x @ state.c_fc[expert].T
        hidden = F.gelu(pre)
        hidden_error = (routed_error @ state.c_proj[expert]) * _gelu_derivative(pre)
        specifications = {
            "fc_left": (hidden_error, torch.ones_like(p)),
            "fc_right": (x, p),
            "proj_left": (routed_error, torch.ones_like(p)),
            "proj_right": (hidden, p),
        }
        row: dict[str, Any] = {
            "expert": expert,
            "assignments": int(locations.shape[0]),
        }
        for name, (factor_rows, factor_weights) in specifications.items():
            root, stats = normalized_covariance_root(
                factor_rows, factor_weights, ridge_ratio=ridge_ratio
            )
            roots[name].append(root.cpu())
            row[name] = stats
        rows.append(row)
    stacked = {name: torch.stack(values) for name, values in roots.items()}
    return stacked, rows


def factor_operator_cosine(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, float]:
    values = {}
    for name in ROOT_NAMES:
        a, b = left[name].double().reshape(-1), right[name].double().reshape(-1)
        values[name] = float(
            torch.dot(a, b) / (a.norm().clamp_min(1e-30) * b.norm().clamp_min(1e-30))
        )
    values["mean"] = sum(values.values()) / len(ROOT_NAMES)
    return values


class MatrixNormalKFACChart(torch.nn.Module):
    """Fixed step-zero base plus compact full-rank BlockFHT displacements."""

    def __init__(
        self,
        *,
        base: LayerState,
        roots: dict[str, torch.Tensor] | None,
        latent_width: int,
        fht_layers: int,
        fc_scale: float,
        proj_scale: float,
        seed: int,
        layer: int,
        device: str,
        shaped: bool,
    ) -> None:
        super().__init__()
        self.experts, self.hidden_width, self.input_width = base.c_fc.shape
        self.latent_width = int(latent_width)
        self.fht_layers = int(fht_layers)
        self.fc_scale = float(fc_scale)
        self.proj_scale = float(proj_scale)
        self.seed = int(seed)
        self.layer = int(layer)
        self.shaped = bool(shaped)
        self.register_buffer("base_c_fc", base.c_fc.float().to(device))
        self.register_buffer("base_c_proj", base.c_proj.float().to(device))
        if shaped:
            if roots is None or set(roots) != set(ROOT_NAMES):
                raise ValueError("shaped chart requires all four KFAC roots")
            for name in ROOT_NAMES:
                self.register_buffer(name, roots[name].float().to(device))
        self.fc_latent = torch.nn.Parameter(
            torch.zeros(self.experts, self.latent_width, device=device)
        )
        self.proj_latent = torch.nn.Parameter(
            torch.zeros(self.experts, self.latent_width, device=device)
        )
        self.hidden_bias = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width, device=device)
        )

    def trainable_parameters(self, *, conditional: bool) -> list[torch.nn.Parameter]:
        if bool(conditional) != self.shaped:
            raise ValueError("conditional flag disagrees with chart family")
        return [self.fc_latent, self.proj_latent, self.hidden_bias]

    def compact_parameter_count(self, *, conditional: bool) -> int:
        return sum(p.numel() for p in self.trainable_parameters(conditional=conditional))

    def _selection(self, expert: int | None) -> slice:
        if expert is None:
            return slice(None)
        if not 0 <= int(expert) < self.experts:
            raise IndexError("expert index out of range")
        return slice(int(expert), int(expert) + 1)

    def _raw_weights(self, selected: slice) -> tuple[torch.Tensor, torch.Tensor]:
        indices = list(range(self.experts))[selected]
        fc, proj = [], []
        for expert in indices:
            fc_seed = self.seed + 1009 * self.layer + 131 * expert
            proj_seed = self.seed + 104729 + 1009 * self.layer + 131 * expert
            fc.append(
                block_fht_slice(
                    self.fc_latent[expert], self.hidden_width * self.input_width,
                    self.fht_layers, fc_seed, 0, self.hidden_width * self.input_width,
                ).reshape(self.hidden_width, self.input_width)
            )
            proj.append(
                block_fht_slice(
                    self.proj_latent[expert], self.input_width * self.hidden_width,
                    self.fht_layers, proj_seed, 0, self.input_width * self.hidden_width,
                ).reshape(self.input_width, self.hidden_width)
            )
        return self.fc_scale * torch.stack(fc), self.proj_scale * torch.stack(proj)

    @staticmethod
    def _apply_metric(
        values: torch.Tensor,
        raw: torch.Tensor,
        right: torch.Tensor,
        left: torch.Tensor,
    ) -> torch.Tensor:
        return torch.bmm(torch.bmm(torch.bmm(values, right), raw.transpose(1, 2)), left)

    def function_and_jvp(
        self,
        inputs: torch.Tensor,
        directions: torch.Tensor,
        *,
        conditional: bool,
        expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(conditional) != self.shaped:
            raise ValueError("conditional flag disagrees with chart family")
        if inputs.shape != directions.shape:
            raise ValueError("input and direction shapes disagree")
        selected = self._selection(expert)
        raw_fc, raw_proj = self._raw_weights(selected)
        base_fc, base_proj = self.base_c_fc[selected], self.base_c_proj[selected]
        pre = torch.bmm(inputs, base_fc.transpose(1, 2))
        pre_jvp = torch.bmm(directions, base_fc.transpose(1, 2))
        if self.shaped:
            pre = pre + self._apply_metric(
                inputs, raw_fc, self.fc_right[selected], self.fc_left[selected]
            )
            pre_jvp = pre_jvp + self._apply_metric(
                directions, raw_fc, self.fc_right[selected], self.fc_left[selected]
            )
        else:
            pre = pre + torch.bmm(inputs, raw_fc.transpose(1, 2))
            pre_jvp = pre_jvp + torch.bmm(directions, raw_fc.transpose(1, 2))
        pre = pre + self.hidden_bias[selected, None, :]
        hidden = F.gelu(pre)
        hidden_jvp = _gelu_derivative(pre) * pre_jvp
        output = torch.bmm(hidden, base_proj.transpose(1, 2))
        output_jvp = torch.bmm(hidden_jvp, base_proj.transpose(1, 2))
        if self.shaped:
            output = output + self._apply_metric(
                hidden, raw_proj, self.proj_right[selected], self.proj_left[selected]
            )
            output_jvp = output_jvp + self._apply_metric(
                hidden_jvp, raw_proj, self.proj_right[selected], self.proj_left[selected]
            )
        else:
            output = output + torch.bmm(hidden, raw_proj.transpose(1, 2))
            output_jvp = output_jvp + torch.bmm(hidden_jvp, raw_proj.transpose(1, 2))
        return output, output_jvp


def make_module(
    plan: dict[str, Any],
    layer: int,
    base: LayerState,
    roots: dict[str, torch.Tensor] | None,
    device: str,
    *,
    shaped: bool,
) -> MatrixNormalKFACChart:
    return MatrixNormalKFACChart(
        base=base,
        roots=roots,
        latent_width=int(plan["candidate"]["c_fc_latent_coordinates_per_expert"]),
        fht_layers=int(plan["candidate"]["block_fht_layers"]),
        fc_scale=float(plan["candidate"]["raw_c_fc_scale"]),
        proj_scale=float(plan["candidate"]["raw_c_proj_scale"]),
        seed=int(plan["candidate"]["block_fht_seed"]),
        layer=int(layer),
        device=device,
        shaped=shaped,
    )


def fit_chart(
    module: MatrixNormalKFACChart,
    inputs: torch.Tensor,
    terminal: LayerState,
    *,
    conditional: bool,
    plan: dict[str, Any],
    probe_seed: int,
    steps: int | None = None,
) -> dict[str, Any]:
    fit = plan["fit_protocol"]
    device = str(module.hidden_bias.device)
    live = inputs.to(device=device, dtype=torch.float32)
    directions = rademacher(tuple(live.shape), probe_seed, device)
    with torch.no_grad():
        target, target_jvp = dense_function_and_jvp(
            live, directions, terminal.c_fc.to(device), terminal.c_proj.to(device).transpose(1, 2)
        )
    parameters = module.trainable_parameters(conditional=conditional)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
    )
    count = int(fit["steps"] if steps is None else steps)
    losses, output_losses, jvp_losses = [], [], []
    maximum_gradient = 0.0
    for _ in range(count):
        optimizer.zero_grad(set_to_none=True)
        output, output_jvp = module.function_and_jvp(
            live, directions, conditional=conditional
        )
        output_loss = normalized_expert_loss(output, target)
        jvp_loss = normalized_expert_loss(output_jvp, target_jvp)
        loss = output_loss + float(fit["jvp_weight"]) * jvp_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite matrix-normal objective")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError("missing or non-finite matrix-normal gradient")
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, float(fit["gradient_clip"])))
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
        output_losses.append(float(output_loss.detach()))
        jvp_losses.append(float(jvp_loss.detach()))
    return {
        "conditional": conditional,
        "steps": count,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "initial_output_loss": output_losses[0],
        "final_output_loss": output_losses[-1],
        "initial_jvp_loss": jvp_losses[0],
        "final_jvp_loss": jvp_losses[-1],
        "maximum_preclip_gradient_norm": maximum_gradient,
    }


@torch.no_grad()
def routed_evaluation(
    route_state: LayerState,
    target_state: LayerState,
    activations: torch.Tensor,
    module: MatrixNormalKFACChart,
    *,
    conditional: bool,
    outer_top_k: int,
    probe_seed: int,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    device = str(module.hidden_bias.device)
    route_state, target_state = route_state.to(device), target_state.to(device)
    all_directions = rademacher(tuple(activations.shape), probe_seed, "cpu")
    predicted_chunks, target_chunks = [], []
    predicted_jvp_chunks, target_jvp_chunks = [], []
    expert_error = torch.zeros(module.experts, dtype=torch.float64)
    expert_energy = torch.zeros_like(expert_error)
    expert_jvp_error = torch.zeros_like(expert_error)
    expert_jvp_energy = torch.zeros_like(expert_error)
    for start in range(0, activations.shape[0], int(chunk_size)):
        stop = min(activations.shape[0], start + int(chunk_size))
        x = activations[start:stop].to(device=device, dtype=torch.float32)
        direction = all_directions[start:stop].to(device=device)
        logits = x @ route_state.router.T
        tie = torch.arange(logits.shape[-1], device=device, dtype=x.dtype)
        selected = torch.topk(
            logits - tie * torch.finfo(x.dtype).eps,
            int(outer_top_k), dim=-1, largest=True, sorted=True,
        ).indices
        probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
        predicted, target = torch.zeros_like(x), torch.zeros_like(x)
        predicted_jvp, target_jvp = torch.zeros_like(x), torch.zeros_like(x)
        for expert in range(module.experts):
            locations = (selected == expert).nonzero(as_tuple=False)
            if not locations.numel():
                continue
            token, slot = locations[:, 0], locations[:, 1]
            expert_input = x.index_select(0, token)[None]
            expert_direction = direction.index_select(0, token)[None]
            output, output_j = module.function_and_jvp(
                expert_input, expert_direction, conditional=conditional, expert=expert
            )
            target_output, target_j = dense_function_and_jvp(
                expert_input,
                expert_direction,
                target_state.c_fc[expert : expert + 1],
                target_state.c_proj[expert : expert + 1].transpose(1, 2),
            )
            weight = probabilities[token, slot, None]
            predicted.index_add_(0, token, output[0] * weight)
            target.index_add_(0, token, target_output[0] * weight)
            predicted_jvp.index_add_(0, token, output_j[0] * weight)
            target_jvp.index_add_(0, token, target_j[0] * weight)
            expert_error[expert] += float((output - target_output).square().sum())
            expert_energy[expert] += float(target_output.square().sum())
            expert_jvp_error[expert] += float((output_j - target_j).square().sum())
            expert_jvp_energy[expert] += float(target_j.square().sum())
        predicted_chunks.append(predicted.cpu())
        target_chunks.append(target.cpu())
        predicted_jvp_chunks.append(predicted_jvp.cpu())
        target_jvp_chunks.append(target_jvp.cpu())
    predicted, target = torch.cat(predicted_chunks), torch.cat(target_chunks)
    predicted_jvp, target_jvp = torch.cat(predicted_jvp_chunks), torch.cat(target_jvp_chunks)
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


def trainable_state(module: MatrixNormalKFACChart) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.named_parameters()}


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("matrix-normal plan schema mismatch")
    root = Path(__file__).resolve().parents[2]
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") is not None:
        if identity["entrypoint_sha256"] != file_sha256(Path(__file__)):
            raise ValueError("entrypoint hash drift")
        for relative, expected in identity.get("helper_sha256", {}).items():
            if file_sha256(root / relative) != expected:
                raise ValueError(f"helper hash drift: {relative}")
    source, candidate = plan["source"], plan["candidate"]
    counts = coordinate_count(
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        hidden_width=int(source["expert_hidden_width"]),
        latent_width=int(candidate["c_fc_latent_coordinates_per_expert"]),
    )
    if int(counts["compact"]) != int(candidate["total_coordinates_all_layers"]):
        raise ValueError("compact coordinate accounting drift")
    if int(counts["dense"]) != int(candidate["dense_paired_parameters_all_layers"]):
        raise ValueError("dense coordinate accounting drift")
    if abs(float(counts["compression"]) - float(candidate["paired_parameter_compression_ratio"])) > 1e-12:
        raise ValueError("compression accounting drift")
    if float(counts["compression"]) < float(plan["frozen_gates"]["paired_parameter_compression_ratio_min"]):
        raise ValueError("registered matrix-normal chart is outside compression budget")
    if not file_sha256(plan_path):
        raise AssertionError("empty plan hash")


def _identity_roots(experts: int, input_width: int, hidden_width: int) -> dict[str, torch.Tensor]:
    # Materialize the expert batch.  ``expand`` would create zero-stride roots
    # and make the performance preflight radically cheaper than the real,
    # expert-specific dense KFAC factors.
    return {
        "fc_left": torch.eye(hidden_width).expand(experts, -1, -1).clone(),
        "fc_right": torch.eye(input_width).expand(experts, -1, -1).clone(),
        "proj_left": torch.eye(input_width).expand(experts, -1, -1).clone(),
        "proj_right": torch.eye(hidden_width).expand(experts, -1, -1).clone(),
    }


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    experts = int(source["num_experts"])
    width = int(source["input_width"])
    hidden = int(source["expert_hidden_width"])
    generator = torch.Generator(device="cpu").manual_seed(20260961)
    base = LayerState(
        torch.randn(experts, width, generator=generator) * 0.02,
        torch.randn(experts, hidden, width, generator=generator) * 0.02,
        torch.randn(experts, width, hidden, generator=generator) * 0.01,
    )
    terminal = LayerState(
        base.router,
        base.c_fc + torch.randn(base.c_fc.shape, generator=generator) * 0.002,
        base.c_proj + torch.randn(base.c_proj.shape, generator=generator) * 0.001,
    )
    roots = _identity_roots(experts, width, hidden)
    candidate = make_module(plan, 0, base, roots, device, shaped=True)
    control = make_module(plan, 0, base, None, device, shaped=False)
    inputs = torch.randn(
        experts, int(plan["geometry_protocol"]["fit_samples_per_expert"]),
        width, generator=generator,
    )
    directions = torch.randn(inputs.shape, generator=generator)
    with torch.no_grad():
        left = candidate.function_and_jvp(inputs.to(device), directions.to(device), conditional=True)
        right = control.function_and_jvp(inputs.to(device), directions.to(device), conditional=False)
        initial_difference = max(float((a - b).abs().max()) for a, b in zip(left, right))
    if initial_difference != 0.0:
        raise RuntimeError("candidate/control step-zero function or JVP differs")
    started = time.time()
    candidate_diag = fit_chart(
        candidate, inputs, terminal, conditional=True, plan=plan,
        probe_seed=20260962, steps=2,
    )
    control_diag = fit_chart(
        control, inputs, terminal, conditional=False, plan=plan,
        probe_seed=20260962, steps=2,
    )
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_matrix_normal_kfac_chart_preflight_v1",
        "device": device,
        "candidate_control_two_step_seconds": elapsed,
        "projected_fit_seconds": elapsed * int(plan["fit_protocol"]["steps"]) * 6 / 2,
        "candidate_coordinates_per_layer": candidate.compact_parameter_count(conditional=True),
        "control_coordinates_per_layer": control.compact_parameter_count(conditional=False),
        "step_zero_function_and_jvp_max_abs_difference": initial_difference,
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
    if args.output.exists():
        raise FileExistsError("matrix-normal output already exists")
    started = time.time()
    source, geometry = plan["source"], plan["geometry_protocol"]
    if file_sha256(args.terminal_snapshot) != source["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash drift")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != source["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash drift")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal snapshot step drift")
    layers = [int(value) for value in source["layers"]]
    model = model_from_exact_stepzero(payload, int(source["model_seed"]), args.device)
    stepzero_hashes = selected_stepzero_hashes(model, layers)
    initial_mapping = dict(model.named_parameters())
    initial = {layer: layer_state_from_mapping(initial_mapping, layer) for layer in layers}
    terminal = {layer: layer_state_from_mapping(payload["model"], layer) for layer in layers}
    roots: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    factor_rows: dict[str, dict[str, Any]] = {}
    discovery_inputs: dict[str, dict[int, torch.Tensor]] = {}
    for bank in geometry["discovery_banks"]:
        name = bank["name"]
        batches = fixed_validation_batches(
            args.data_dir, int(bank["batch_size"]), int(bank["block_size"]) + 1,
            int(bank["batches"]), int(bank["seed"]),
        )
        inputs, errors, discovery_loss = collect_geometry(model, batches, layers, args.device)
        discovery_inputs[name] = inputs
        roots[name], factor_rows[name] = {}, {"loss": discovery_loss, "layers": {}}
        for layer in layers:
            layer_roots, rows = build_full_kfac_roots(
                initial[layer], inputs[layer], errors[layer],
                ridge_ratio=float(geometry["expert_factors"]["ridge_ratio"]),
                minimum_assignments=int(geometry["expert_factors"]["minimum_assignments_per_expert_each_bank"]),
                device=args.device,
            )
            roots[name][layer] = layer_roots
            factor_rows[name]["layers"][str(layer)] = rows
    heldout_spec = geometry["heldout"]
    heldout_batches = fixed_validation_batches(
        args.data_dir, int(heldout_spec["batch_size"]), int(heldout_spec["block_size"]) + 1,
        int(heldout_spec["batches"]), int(heldout_spec["seed"]),
    )
    model.eval()
    heldout_inputs = collect_inputs(
        model, heldout_batches, layers,
        int(heldout_spec["activation_sample_cap_per_layer"]), args.device,
    )
    del initial_mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    banks = [row["name"] for row in geometry["discovery_banks"]]
    random = plan["random_protocol"]
    saved, summaries, diagnostics, occupancy = {}, {}, {}, {}
    actions: dict[tuple[str, int], torch.Tensor] = {}
    root_hashes: dict[str, dict[str, dict[str, str]]] = {}
    base_summaries: dict[str, Any] = {}
    for layer in layers:
        base_module = make_module(plan, layer, initial[layer], None, args.device, shaped=False)
        base_eval = routed_evaluation(
            initial[layer], terminal[layer], heldout_inputs[layer], base_module,
            conditional=False, outer_top_k=int(source["outer_moe_top_k"]),
            probe_seed=int(random["heldout_jvp_seed_base"]) + 17 * layer,
        )
        base_summaries[str(layer)] = {
            "mixture_recovery": base_eval["mixture_recovery"],
            "jvp_recovery": base_eval["jvp_recovery"],
            "minimum_expert_recovery": min(base_eval["expert_recovery"]),
        }
        del base_module
    for bank_index, bank in enumerate(banks):
        saved[bank], summaries[bank], diagnostics[bank], occupancy[bank] = {}, {}, {}, {}
        root_hashes[bank] = {}
        for layer in layers:
            sampled, counts = route_and_sample(
                initial[layer], discovery_inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=int(geometry["fit_samples_per_expert"]),
                seed=int(random["fit_sample_seed_base"]) + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            candidate = make_module(
                plan, layer, initial[layer], roots[bank][layer], args.device, shaped=True
            )
            control = make_module(plan, layer, initial[layer], None, args.device, shaped=False)
            for left, right in zip(
                candidate.trainable_parameters(conditional=True),
                control.trainable_parameters(conditional=False),
            ):
                if not torch.equal(left, right):
                    raise RuntimeError("candidate/control initialization drift")
            probe_seed = int(random["fit_jvp_seed_base"]) + 1009 * bank_index + 17 * layer
            candidate_diag = fit_chart(
                candidate, sampled, terminal[layer], conditional=True,
                plan=plan, probe_seed=probe_seed,
            )
            control_diag = fit_chart(
                control, sampled, terminal[layer], conditional=False,
                plan=plan, probe_seed=probe_seed,
            )
            eval_seed = int(random["heldout_jvp_seed_base"]) + 17 * layer
            candidate_eval = routed_evaluation(
                initial[layer], terminal[layer], heldout_inputs[layer], candidate,
                conditional=True, outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=eval_seed,
            )
            control_eval = routed_evaluation(
                initial[layer], terminal[layer], heldout_inputs[layer], control,
                conditional=False, outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=eval_seed,
            )
            if not torch.equal(candidate_eval["target"], control_eval["target"]):
                raise RuntimeError("candidate/control target drift")
            actions[(bank, layer)] = candidate_eval["predicted"]
            base_recovery = float(base_summaries[str(layer)]["mixture_recovery"])
            static = float(plan["frozen_gates"][f"sealed_static_ceiling_recovery_{'a' if bank_index == 0 else 'b'}"])
            summaries[bank][str(layer)] = {
                "mixture_recovery": candidate_eval["mixture_recovery"],
                "jvp_recovery": candidate_eval["jvp_recovery"],
                "minimum_expert_recovery": min(candidate_eval["expert_recovery"]),
                "minimum_expert_jvp_recovery": min(candidate_eval["expert_jvp_recovery"]),
                "isotropic_control_recovery": control_eval["mixture_recovery"],
                "candidate_minus_isotropic_control_recovery": candidate_eval["mixture_recovery"] - control_eval["mixture_recovery"],
                "exact_stepzero_base_recovery": base_recovery,
                "candidate_minus_exact_stepzero_base_recovery": candidate_eval["mixture_recovery"] - base_recovery,
                "sealed_static_ceiling_recovery": static,
                "candidate_minus_sealed_static_ceiling_recovery": candidate_eval["mixture_recovery"] - static,
            }
            diagnostics[bank][str(layer)] = {
                "candidate": candidate_diag,
                "control": control_diag,
            }
            saved[bank][str(layer)] = {
                "candidate": trainable_state(candidate),
                "control": trainable_state(control),
            }
            root_hashes[bank][str(layer)] = {
                name: tensor_sha256(value) for name, value in roots[bank][layer].items()
            }
            del candidate, control
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    factor_cosines = {
        str(layer): factor_operator_cosine(roots[banks[0]][layer], roots[banks[1]][layer])
        for layer in layers
    }
    factor_cosine_mean = sum(row["mean"] for row in factor_cosines.values()) / len(layers)
    action_cosines = {
        str(layer): action_cosine(actions[(banks[0], layer)], actions[(banks[1], layer)])
        for layer in layers
    }
    action_cosine_mean = sum(action_cosines.values()) / len(layers)
    gates, frozen = {}, plan["frozen_gates"]
    for bank in banks:
        rows = [summaries[bank][str(layer)] for layer in layers]
        aggregate = {
            "mixture_recovery_mean": sum(float(row["mixture_recovery"]) for row in rows) / len(rows),
            "mixture_recovery_minimum_layer": min(float(row["mixture_recovery"]) for row in rows),
            "jvp_recovery_mean": sum(float(row["jvp_recovery"]) for row in rows) / len(rows),
            "minimum_expert_recovery": min(float(row["minimum_expert_recovery"]) for row in rows),
            "candidate_minus_isotropic_control_recovery_mean": sum(float(row["candidate_minus_isotropic_control_recovery"]) for row in rows) / len(rows),
            "candidate_minus_exact_stepzero_base_recovery_mean": sum(float(row["candidate_minus_exact_stepzero_base_recovery"]) for row in rows) / len(rows),
            "candidate_minus_sealed_static_ceiling_recovery_mean": sum(float(row["candidate_minus_sealed_static_ceiling_recovery"]) for row in rows) / len(rows),
            "minimum_discovery_assignments": min(min(occupancy[bank][str(layer)]) for layer in layers),
        }
        summaries[bank]["aggregate"] = aggregate
        gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["candidate_heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["candidate_heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": aggregate["minimum_expert_recovery"] >= float(frozen["candidate_heldout_expert_recovery_min_each_bank"]),
            "jvp_pass": aggregate["jvp_recovery_mean"] >= float(frozen["candidate_heldout_jvp_recovery_mean_min_each_bank"]),
            "isotropic_control_gain_pass": aggregate["candidate_minus_isotropic_control_recovery_mean"] >= float(frozen["candidate_minus_isotropic_control_recovery_mean_min_each_bank"]),
            "stepzero_base_gain_pass": aggregate["candidate_minus_exact_stepzero_base_recovery_mean"] >= float(frozen["candidate_minus_exact_stepzero_base_recovery_mean_min_each_bank"]),
            "static_ceiling_gain_pass": aggregate["candidate_minus_sealed_static_ceiling_recovery_mean"] >= float(frozen["candidate_minus_sealed_static_ceiling_recovery_mean_min_each_bank"]),
            "minimum_occupancy_pass": aggregate["minimum_discovery_assignments"] >= int(frozen["minimum_expert_assignments_each_bank"]),
            "action_agreement_pass": action_cosine_mean >= float(frozen["candidate_cross_bank_action_cosine_mean_min"]),
            "factor_agreement_pass": factor_cosine_mean >= float(frozen["candidate_cross_bank_factor_operator_cosine_mean_min"]),
        }
    finite = all_finite({
        "summaries": summaries, "diagnostics": diagnostics,
        "factor_cosines": factor_cosines, "action_cosines": action_cosines,
        "factor_rows": factor_rows,
    })
    for bank in banks:
        gates[bank]["finite_pass"] = finite
        gates[bank]["compression_pass"] = float(plan["candidate"]["paired_parameter_compression_ratio"]) >= float(frozen["paired_parameter_compression_ratio_min"])
        gates[bank]["all_pass"] = all(gates[bank].values())
    passed = all(gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_matrix_normal_kfac_chart_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    fixed_factor_elements_one_model = int(source["tensor_layers"]) * int(source["num_experts"]) * 2 * (
        int(source["expert_hidden_width"]) ** 2 + int(source["input_width"]) ** 2
    )
    fixed_base_elements_one_model = int(plan["candidate"]["dense_paired_parameters_all_layers"])
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_matrix_normal_kfac_chart_oracle_result_v1",
        "classification": "MATRIX_NORMAL_KFAC_CHART_PASSES" if passed else "MATRIX_NORMAL_KFAC_CHART_REJECTED",
        "passed": passed,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(manifest),
            "stepzero_parameter_hashes": stepzero_hashes,
            "factor_root_sha256": root_hashes,
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
            "coordinate_compression_ratio": float(plan["candidate"]["paired_parameter_compression_ratio"]),
            "fixed_dense_stepzero_base_elements_oracle_only": fixed_base_elements_one_model,
            "fixed_dense_factor_elements_oracle_only": fixed_factor_elements_one_model,
            "fixed_dense_state_bytes_float32_oracle_only": 4 * (fixed_base_elements_one_model + fixed_factor_elements_one_model),
            "dense_fixed_state_is_deployable": False,
            "dense_learned_basis": False,
            "additive_lora_residual": False,
        },
        "factor_acquisition": factor_rows,
        "factor_operator_cosine": {"mean": factor_cosine_mean, "by_layer": factor_cosines},
        "heldout_bank_action_cosine": {"mean": action_cosine_mean, "by_layer": action_cosines},
        "base_summaries": base_summaries,
        "occupancy": occupancy,
        "fit_diagnostics": diagnostics,
        "summaries": summaries,
        "gates": gates,
        "all_values_finite": finite,
        "authorization": {
            "procedural_factor_theory": passed,
            "initialization_and_mapping_loss_shadow": passed,
            "dense_factor_or_base_storage_in_model": False,
            "language_model_training": False,
            "larger_rung": False,
            "full_attention_work": False,
            "automatic_retry_or_sweep": False,
        },
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
