#!/usr/bin/env python3
"""Gate normalized global features with compact Givens coefficient mixing."""
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
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
    normalized_expert_loss,
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
)


PLAN_SCHEMA = "nanogpt_sparse_moe_global_write_givens_feature_oracle_plan_v1"
BASIS_SCHEMA = "nanogpt_sparse_moe_write_subspace_ceiling_bases_v1"
BANKS = ("discovery_a", "discovery_b")


def coordinate_count(
    *, rank: int, input_width: int, tensor_layers: int, experts: int,
    stages: int,
) -> int:
    return (
        2 * rank * input_width
        + 3 * tensor_layers * experts * rank
        + stages * tensor_layers * experts * (rank // 2)
    )


def fixed_matchings(rank: int, stages: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.stack(
        [torch.randperm(int(rank), generator=generator) for _ in range(int(stages))]
    ) if stages else torch.empty(0, int(rank), dtype=torch.long)


def apply_givens_stages(
    values: torch.Tensor,
    angles: torch.Tensor | None,
    matchings: torch.Tensor,
) -> torch.Tensor:
    if angles is None:
        return values
    if values.shape[0] != angles.shape[0] or angles.shape[1] != matchings.shape[0]:
        raise ValueError("Givens expert/stage inventory mismatch")
    result = values
    pairs = angles.shape[-1]
    for stage in range(matchings.shape[0]):
        permutation = matchings[stage]
        inverse = torch.argsort(permutation)
        ordered = result.index_select(-1, permutation)
        paired = ordered[..., : 2 * pairs].reshape(
            *ordered.shape[:-1], pairs, 2
        )
        theta = angles[:, stage, :]
        while theta.ndim < paired[..., 0].ndim:
            theta = theta.unsqueeze(1)
        cosine, sine = theta.cos(), theta.sin()
        left, right = paired[..., 0], paired[..., 1]
        rotated = torch.stack(
            (cosine * left - sine * right, sine * left + cosine * right),
            dim=-1,
        ).flatten(-2)
        if 2 * pairs < ordered.shape[-1]:
            rotated = torch.cat((rotated, ordered[..., 2 * pairs :]), dim=-1)
        result = rotated.index_select(-1, inverse)
    return result


class GlobalWriteGivensFeatures(torch.nn.Module):
    """Complete expert function with normalized features and fixed global V."""

    def __init__(
        self,
        *,
        write_basis: torch.Tensor,
        tensor_layers: int,
        experts: int,
        feature_seed: int,
        matching_seed: int,
        stages: int,
        device: str,
    ) -> None:
        super().__init__()
        if write_basis.ndim != 2:
            raise ValueError("write basis must be [input_width, rank]")
        self.input_width = int(write_basis.shape[0])
        self.rank = int(write_basis.shape[1])
        self.tensor_layers = int(tensor_layers)
        self.experts = int(experts)
        self.stages = int(stages)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(feature_seed))
            self.raw_feature = torch.nn.Parameter(
                torch.randn(self.rank, self.input_width) * 0.02
            )
        shape = (self.tensor_layers, self.experts, self.rank)
        self.input_gain = torch.nn.Parameter(torch.ones(shape))
        self.hidden_bias = torch.nn.Parameter(torch.zeros(shape))
        self.output_gain = torch.nn.Parameter(torch.zeros(shape))
        if self.stages:
            self.angles = torch.nn.Parameter(torch.zeros(
                self.tensor_layers, self.experts, self.stages, self.rank // 2
            ))
        else:
            self.register_parameter("angles", None)
        self.register_buffer(
            "write_basis", write_basis.detach().float().contiguous(), persistent=True
        )
        self.register_buffer(
            "matchings", fixed_matchings(self.rank, self.stages, matching_seed),
            persistent=True,
        )
        self.feature_scale = 0.02 * math.sqrt(float(self.input_width))
        self.to(device=device, dtype=torch.float32)

    @property
    def device(self) -> str:
        return str(self.raw_feature.device)

    def counted_coordinates(self) -> int:
        return self.write_basis.numel() + sum(
            parameter.numel() for parameter in self.parameters()
        )

    def feature_basis(self) -> torch.Tensor:
        return F.normalize(self.raw_feature, dim=-1) * self.feature_scale

    def _selection(self, expert: int | None) -> slice:
        if expert is None:
            return slice(None)
        if not 0 <= int(expert) < self.experts:
            raise IndexError("expert index out of range")
        return slice(int(expert), int(expert) + 1)

    def coefficients_and_jvp(
        self,
        inputs: torch.Tensor,
        directions: torch.Tensor,
        *,
        layer: int,
        expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.shape != directions.shape or inputs.shape[-1] != self.input_width:
            raise ValueError("feature input/direction shape mismatch")
        expected = self.experts if expert is None else 1
        if inputs.shape[0] != expected:
            raise ValueError("feature expert batch mismatch")
        selected = self._selection(expert)
        feature = self.feature_basis()
        pre = torch.einsum("esd,rd->esr", inputs.float(), feature)
        pre_jvp = torch.einsum("esd,rd->esr", directions.float(), feature)
        gain = self.input_gain[layer, selected, None, :]
        pre = pre * gain + self.hidden_bias[layer, selected, None, :]
        pre_jvp = pre_jvp * gain
        hidden = F.gelu(pre)
        hidden_jvp = gelu_derivative(pre) * pre_jvp
        angles = None if self.angles is None else self.angles[layer, selected]
        mixed = apply_givens_stages(hidden, angles, self.matchings)
        mixed_jvp = apply_givens_stages(hidden_jvp, angles, self.matchings)
        output_gain = self.output_gain[layer, selected, None, :]
        return mixed * output_gain, mixed_jvp * output_gain

    def function_and_jvp(
        self,
        inputs: torch.Tensor,
        directions: torch.Tensor,
        *,
        layer: int,
        expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients, coefficient_jvp = self.coefficients_and_jvp(
            inputs, directions, layer=layer, expert=expert
        )
        return coefficients @ self.write_basis.T, coefficient_jvp @ self.write_basis.T


def load_write_bases(path: Path, plan: dict[str, Any]) -> dict[str, torch.Tensor]:
    if file_sha256(path) != plan["source"]["write_basis_artifact_sha256"]:
        raise ValueError("write basis artifact hash drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != BASIS_SCHEMA:
        raise ValueError("write basis artifact schema mismatch")
    rank = int(plan["candidate"]["rank"])
    result = {}
    for bank in BANKS:
        basis = payload["bases"]["global_shared_rank619"][bank]
        if basis.shape[0] != 1 or basis.shape[1] != int(plan["source"]["input_width"]):
            raise ValueError("global write basis shape drift")
        result[bank] = basis[0, :, :rank].float().contiguous()
    return result


def make_module(
    plan: dict[str, Any], write_basis: torch.Tensor, *, candidate: bool,
    device: str,
) -> GlobalWriteGivensFeatures:
    source, spec = plan["source"], plan["candidate"]
    return GlobalWriteGivensFeatures(
        write_basis=write_basis,
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        feature_seed=int(spec["feature_seed"]),
        matching_seed=int(spec["matching_seed"]),
        stages=int(spec["givens_stages"]) if candidate else 0,
        device=device,
    )


def trainable_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def fit_joint(
    module: GlobalWriteGivensFeatures,
    samples: dict[int, torch.Tensor],
    states: dict[int, LayerState],
    *,
    layers: list[int],
    plan: dict[str, Any],
    probe_seed: int,
) -> dict[str, Any]:
    fit = plan["fit_protocol"]
    live: dict[int, torch.Tensor] = {}
    directions: dict[int, torch.Tensor] = {}
    targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        live[layer] = samples[layer].to(module.device, dtype=torch.float32)
        directions[layer] = rademacher(
            tuple(live[layer].shape), probe_seed + 17 * layer, module.device
        )
        state = states[layer].to(module.device)
        with torch.no_grad():
            output, jvp = dense_function_and_jvp(
                live[layer], directions[layer],
                state.c_fc, state.c_proj.transpose(1, 2),
            )
            targets[layer] = (
                output @ module.write_basis,
                jvp @ module.write_basis,
            )
    parameters = list(module.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
    )
    losses, output_losses, jvp_losses = [], [], []
    maximum_gradient = 0.0
    gradients_finite = True
    for _step in range(int(fit["steps"])):
        optimizer.zero_grad(set_to_none=True)
        output_rows, jvp_rows = [], []
        for layer in layers:
            predicted, predicted_jvp = module.coefficients_and_jvp(
                live[layer], directions[layer], layer=layer
            )
            target, target_jvp = targets[layer]
            output_rows.append(normalized_expert_loss(predicted, target))
            jvp_rows.append(normalized_expert_loss(predicted_jvp, target_jvp))
        output_loss = torch.stack(output_rows).mean()
        jvp_loss = torch.stack(jvp_rows).mean()
        loss = output_loss + float(fit["jvp_weight"]) * jvp_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Givens feature objective")
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in parameters
        )
        if not gradients_finite:
            raise RuntimeError("missing or non-finite Givens feature gradient")
        gradient = float(torch.nn.utils.clip_grad_norm_(
            parameters, float(fit["gradient_clip"])
        ))
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
        output_losses.append(float(output_loss.detach()))
        jvp_losses.append(float(jvp_loss.detach()))
    angle_rms = 0.0 if module.angles is None else float(
        module.angles.detach().square().mean().sqrt()
    )
    return {
        "steps": int(fit["steps"]),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "initial_output_loss": output_losses[0],
        "final_output_loss": output_losses[-1],
        "initial_jvp_loss": jvp_losses[0],
        "final_jvp_loss": jvp_losses[-1],
        "maximum_preclip_gradient_norm": maximum_gradient,
        "all_gradients_finite": gradients_finite,
        "raw_feature_rms": float(module.raw_feature.detach().square().mean().sqrt()),
        "normalized_feature_row_norm_mean": float(module.feature_basis().detach().norm(dim=-1).mean()),
        "input_gain_rms": float(module.input_gain.detach().square().mean().sqrt()),
        "hidden_bias_rms": float(module.hidden_bias.detach().square().mean().sqrt()),
        "output_gain_rms": float(module.output_gain.detach().square().mean().sqrt()),
        "angle_rms_radians": angle_rms,
    }


@torch.no_grad()
def routed_evaluation(
    state: LayerState,
    activations: torch.Tensor,
    module: GlobalWriteGivensFeatures | None,
    write_basis: torch.Tensor,
    *,
    layer: int,
    outer_top_k: int,
    probe_seed: int,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    device = str(write_basis.device)
    state = state.to(device)
    directions = rademacher(tuple(activations.shape), probe_seed, "cpu")
    predicted_rows, target_rows = [], []
    predicted_jvp_rows, target_jvp_rows = [], []
    expert_error = torch.zeros(state.c_fc.shape[0], dtype=torch.float64)
    expert_energy = torch.zeros_like(expert_error)
    for start in range(0, activations.shape[0], int(chunk_size)):
        stop = min(activations.shape[0], start + int(chunk_size))
        x = activations[start:stop].to(device=device, dtype=torch.float32)
        direction = directions[start:stop].to(device=device)
        logits = x @ state.router.T
        tie = torch.arange(logits.shape[-1], device=device, dtype=x.dtype)
        selected = torch.topk(
            logits - tie * torch.finfo(x.dtype).eps,
            int(outer_top_k), dim=-1, largest=True, sorted=True,
        ).indices
        probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
        predicted, target = torch.zeros_like(x), torch.zeros_like(x)
        predicted_jvp, target_jvp = torch.zeros_like(x), torch.zeros_like(x)
        for expert in range(state.c_fc.shape[0]):
            locations = (selected == expert).nonzero(as_tuple=False)
            if not locations.numel():
                continue
            token, slot = locations[:, 0], locations[:, 1]
            expert_input = x.index_select(0, token)[None]
            expert_direction = direction.index_select(0, token)[None]
            target_output, dense_jvp = dense_function_and_jvp(
                expert_input, expert_direction,
                state.c_fc[expert : expert + 1],
                state.c_proj[expert : expert + 1].transpose(1, 2),
            )
            if module is None:
                output = (target_output @ write_basis) @ write_basis.T
                output_jvp = (dense_jvp @ write_basis) @ write_basis.T
            else:
                output, output_jvp = module.function_and_jvp(
                    expert_input, expert_direction, layer=layer, expert=expert
                )
            weight = probabilities[token, slot, None]
            predicted.index_add_(0, token, output[0] * weight)
            target.index_add_(0, token, target_output[0] * weight)
            predicted_jvp.index_add_(0, token, output_jvp[0] * weight)
            target_jvp.index_add_(0, token, dense_jvp[0] * weight)
            expert_error[expert] += float((output - target_output).square().sum())
            expert_energy[expert] += float(target_output.square().sum())
        predicted_rows.append(predicted.cpu())
        target_rows.append(target.cpu())
        predicted_jvp_rows.append(predicted_jvp.cpu())
        target_jvp_rows.append(target_jvp.cpu())
    predicted = torch.cat(predicted_rows)
    target = torch.cat(target_rows)
    predicted_jvp = torch.cat(predicted_jvp_rows)
    target_jvp = torch.cat(target_jvp_rows)
    return {
        "predicted": predicted,
        "target": target,
        "mixture_recovery": recovery_fraction(predicted, target),
        "jvp_recovery": recovery_fraction(predicted_jvp, target_jvp),
        "expert_recovery": [
            1.0 - float(error / max(energy, 1e-30))
            for error, energy in zip(expert_error, expert_energy)
        ],
    }


def aggregate_rows(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    return {
        "mixture_recovery_mean": sum(float(row["mixture_recovery"]) for row in values) / len(values),
        "mixture_recovery_minimum_layer": min(float(row["mixture_recovery"]) for row in values),
        "jvp_recovery_mean": sum(float(row["jvp_recovery"]) for row in values) / len(values),
        "minimum_expert_recovery": min(min(row["expert_recovery"]) for row in values),
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("global-write Givens plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed")
    helpers = identity.get("helper_sha256")
    if not isinstance(helpers, dict) or not helpers:
        raise ValueError("helper hashes are not sealed")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in helpers.items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    if file_sha256(root / plan["source"]["write_ceiling_result"]) != plan["source"]["write_ceiling_result_sha256"]:
        raise ValueError("write ceiling result hash drift")
    source, candidate = plan["source"], plan["candidate"]
    expected = coordinate_count(
        rank=int(candidate["rank"]), input_width=int(source["input_width"]),
        tensor_layers=int(source["tensor_layers"]), experts=int(source["num_experts"]),
        stages=int(candidate["givens_stages"]),
    )
    if expected != int(candidate["total_coordinates_all_layers"]):
        raise ValueError("candidate coordinate accounting drift")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source, fit = plan["source"], plan["fit_protocol"]
    rank = int(plan["candidate"]["rank"])
    generator = torch.Generator(device="cpu").manual_seed(20261421)
    write, _ = torch.linalg.qr(torch.randn(
        int(source["input_width"]), rank, generator=generator
    ))
    layers = [int(value) for value in source["layers"]]
    samples, states = {}, {}
    for layer in layers:
        samples[layer] = torch.randn(
            int(source["num_experts"]), 1024, int(source["input_width"]),
            generator=generator,
        ).contiguous()
        states[layer] = LayerState(
            router=torch.randn(int(source["num_experts"]), int(source["input_width"]), generator=generator),
            c_fc=(torch.randn(int(source["num_experts"]), int(source["expert_hidden_width"]), int(source["input_width"]), generator=generator) * 0.02).contiguous(),
            c_proj=(torch.randn(int(source["num_experts"]), int(source["input_width"]), int(source["expert_hidden_width"]), generator=generator) * (0.02 / math.sqrt(2.0 * int(source["tensor_layers"])))).contiguous(),
        )
    candidate = make_module(plan, write, candidate=True, device=device)
    control = make_module(plan, write, candidate=False, device=device)
    live = samples[layers[0]].to(device)
    direction = torch.randn_like(live)
    initial = []
    for module in (candidate, control):
        output, jvp = module.function_and_jvp(live, direction, layer=layers[0])
        initial.extend((float(output.abs().max()), float(jvp.abs().max())))
    original_steps = fit["steps"]
    fit["steps"] = 2
    started = time.time()
    try:
        candidate_diag = fit_joint(candidate, samples, states, layers=layers, plan=plan, probe_seed=20261422)
        control_diag = fit_joint(control, samples, states, layers=layers, plan=plan, probe_seed=20261422)
    finally:
        fit["steps"] = original_steps
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_global_write_givens_feature_preflight_v1",
        "device": device,
        "candidate_control_two_step_seconds": elapsed,
        "projected_full_two_bank_seconds": elapsed * (int(original_steps) / 2.0) * 2.0,
        "step_zero_output_and_jvp_max_abs": max(initial),
        "exact_step_zero_pass": max(initial) == 0.0,
        "candidate_coordinates": candidate.counted_coordinates(),
        "control_coordinates": control.counted_coordinates(),
        "all_values_and_gradients_finite": all_finite({"candidate": candidate_diag, "control": control_diag}),
        "maximum_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
        "candidate_diagnostics": candidate_diag,
        "control_diagnostics": control_diag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--write-bases", type=Path)
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
    if None in (args.write_bases, args.terminal_snapshot, args.data_dir, args.output):
        parser.error("oracle requires --write-bases, --terminal-snapshot, --data-dir, and --output")

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
    write_bases = load_write_bases(args.write_bases, plan)
    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, plan, args.data_dir, args.device)
    mapping = dict(model.named_parameters())
    layers = [int(value) for value in source["layers"]]
    states = {layer: layer_state_from_mapping(mapping, layer) for layer in layers}
    del mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    prerequisite_rows, prerequisite_actions = {}, {}
    for bank in BANKS:
        prerequisite_rows[bank] = {}
        basis = write_bases[bank].to(args.device)
        for layer in layers:
            evaluation = routed_evaluation(
                states[layer], inputs["heldout"][layer], None, basis,
                layer=layer, outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=int(plan["data_protocol"]["heldout_jvp_probe_seed_base"]) + 17 * layer,
            )
            prerequisite_actions[(bank, layer)] = evaluation["predicted"]
            prerequisite_rows[bank][str(layer)] = {
                key: value for key, value in evaluation.items()
                if key not in {"predicted", "target"}
            }
        prerequisite_rows[bank]["aggregate"] = aggregate_rows(prerequisite_rows[bank])
    prerequisite_agreement = sum(
        action_cosine(prerequisite_actions[(BANKS[0], layer)], prerequisite_actions[(BANKS[1], layer)])
        for layer in layers
    ) / len(layers)
    prereq = plan["prerequisite_gates"]
    prerequisite_gates = {}
    for bank in BANKS:
        row = prerequisite_rows[bank]["aggregate"]
        prerequisite_gates[bank] = {
            "mean_output_pass": row["mixture_recovery_mean"] >= float(prereq["rank561_fixed_write_mixture_recovery_mean_min_each_bank"]),
            "jvp_pass": row["jvp_recovery_mean"] >= float(prereq["rank561_fixed_write_jvp_recovery_mean_min_each_bank"]),
            "every_layer_pass": row["mixture_recovery_minimum_layer"] >= float(prereq["rank561_fixed_write_every_layer_min_each_bank"]),
            "every_expert_pass": row["minimum_expert_recovery"] >= float(prereq["rank561_fixed_write_every_expert_min_each_bank"]),
            "action_agreement_pass": prerequisite_agreement >= float(prereq["rank561_cross_bank_projected_action_cosine_min"]),
        }
        prerequisite_gates[bank]["all_pass"] = all(prerequisite_gates[bank].values())
    prerequisite_passed = all(prerequisite_gates[bank]["all_pass"] for bank in BANKS)
    if not prerequisite_passed:
        raise RuntimeError("registered rank-561 write prerequisite failed")

    samples, occupancy = {}, {}
    sample_count = int(plan["data_protocol"]["fit_samples_per_expert"])
    for bank_index, bank in enumerate(BANKS):
        samples[bank], occupancy[bank] = {}, {}
        for layer in layers:
            sampled, counts = route_and_sample(
                states[layer], inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]), samples_per_expert=sample_count,
                seed=20261431 + 1009 * bank_index + 17 * layer,
            )
            samples[bank][layer] = sampled
            occupancy[bank][str(layer)] = counts

    summaries, diagnostics, saved, actions = {}, {}, {}, {}
    for bank_index, bank in enumerate(BANKS):
        basis = write_bases[bank]
        candidate = make_module(plan, basis, candidate=True, device=args.device)
        control = make_module(plan, basis, candidate=False, device=args.device)
        for name in ("raw_feature", "input_gain", "hidden_bias", "output_gain"):
            if not torch.equal(getattr(candidate, name), getattr(control, name)):
                raise RuntimeError("candidate/control initialization drift")
        diagnostics[bank] = {
            "candidate": fit_joint(
                candidate, samples[bank], states, layers=layers, plan=plan,
                probe_seed=int(plan["data_protocol"]["fit_jvp_probe_seed_base"]) + 1009 * bank_index,
            ),
            "control": fit_joint(
                control, samples[bank], states, layers=layers, plan=plan,
                probe_seed=int(plan["data_protocol"]["fit_jvp_probe_seed_base"]) + 1009 * bank_index,
            ),
        }
        summaries[bank] = {"candidate": {}, "control": {}}
        for layer in layers:
            for name, module in (("candidate", candidate), ("control", control)):
                evaluation = routed_evaluation(
                    states[layer], inputs["heldout"][layer], module,
                    module.write_basis, layer=layer,
                    outer_top_k=int(source["outer_moe_top_k"]),
                    probe_seed=int(plan["data_protocol"]["heldout_jvp_probe_seed_base"]) + 17 * layer,
                )
                if name == "candidate":
                    actions[(bank, layer)] = evaluation["predicted"]
                summaries[bank][name][str(layer)] = {
                    key: value for key, value in evaluation.items()
                    if key not in {"predicted", "target"}
                }
        for name in ("candidate", "control"):
            summaries[bank][name]["aggregate"] = aggregate_rows(summaries[bank][name])
        saved[bank] = {
            "candidate": trainable_state(candidate),
            "control": trainable_state(control),
        }
        del candidate, control
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    agreement = sum(
        action_cosine(actions[(BANKS[0], layer)], actions[(BANKS[1], layer)])
        for layer in layers
    ) / len(layers)
    frozen = plan["candidate_gates"]
    gates = {}
    for bank in BANKS:
        candidate = summaries[bank]["candidate"]["aggregate"]
        control = summaries[bank]["control"]["aggregate"]
        gain = candidate["mixture_recovery_mean"] - control["mixture_recovery_mean"]
        gates[bank] = {
            "mean_recovery_pass": candidate["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": candidate["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": candidate["minimum_expert_recovery"] >= float(frozen["heldout_expert_recovery_min_each_bank"]),
            "jvp_pass": candidate["jvp_recovery_mean"] >= float(frozen["heldout_jvp_recovery_mean_min_each_bank"]),
            "action_agreement_pass": agreement >= float(frozen["heldout_bank_action_cosine_mean_min"]),
            "control_gain_pass": gain >= float(frozen["candidate_minus_same_rank_control_recovery_mean_min_each_bank"]),
            "occupancy_pass": min(min(values) for values in occupancy[bank].values()) >= int(frozen["minimum_discovery_assignments_per_expert"]),
            "finite_pass": all_finite({"summary": summaries[bank], "diagnostics": diagnostics[bank]}),
            "compression_pass": float(plan["candidate"]["paired_parameter_compression_ratio"]) >= float(frozen["paired_parameter_compression_ratio_min"]),
        }
        gates[bank]["candidate_minus_control_recovery_mean"] = gain
        gates[bank]["all_pass"] = all(
            value for key, value in gates[bank].items()
            if key not in {"candidate_minus_control_recovery_mean", "all_pass"}
        )
    passed = all(gates[bank]["all_pass"] for bank in BANKS)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_global_write_givens_feature_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_global_write_givens_feature_oracle_result_v1",
        "classification": "GLOBAL_WRITE_GIVENS_FEATURE_PASSES" if passed else "GLOBAL_WRITE_GIVENS_FEATURE_REJECTED",
        "passed": passed,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "write_basis_artifact_sha256": file_sha256(args.write_bases),
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
            "candidate_coordinates": int(plan["candidate"]["total_coordinates_all_layers"]),
            "candidate_compression_ratio": float(plan["candidate"]["paired_parameter_compression_ratio"]),
            "control_coordinates": int(plan["same_rank_control"]["total_coordinates_all_layers"]),
            "control_compression_ratio": float(plan["same_rank_control"]["paired_parameter_compression_ratio"]),
            "dense_base_or_residual": False,
            "terminal_derived_write_atlas_is_deployable": False,
        },
        "write_prerequisite": {
            "summaries": prerequisite_rows,
            "cross_bank_action_cosine": prerequisite_agreement,
            "gates": prerequisite_gates,
            "passed": prerequisite_passed,
        },
        "occupancy": occupancy,
        "fit_diagnostics": diagnostics,
        "summaries": summaries,
        "candidate_cross_bank_action_cosine": agreement,
        "gates": gates,
        "all_values_finite": all_finite({"summaries": summaries, "diagnostics": diagnostics, "agreement": agreement}),
        "authorization": {
            "causal_write_atlas_theory": passed,
            "language_model_training": False,
            "larger_rung": False,
            "full_attention_work": False,
            "automatic_retry_or_sweep": False,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
