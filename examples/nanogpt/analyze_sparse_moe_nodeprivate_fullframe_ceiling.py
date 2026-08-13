#!/usr/bin/env python3
"""Fit a nondeployable node-private full-frame ceiling for sparse-MoE MLPs."""
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
from examples.nanogpt.analyze_sparse_moe_cfc_global_conditional_tangent_audit import (
    all_tensor_values_finite,
)
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    action_cosine,
    collect_protocol_inputs,
    route_and_sample,
)
from examples.nanogpt.analyze_sparse_moe_global_write_givens_feature_oracle import (
    BANKS,
    aggregate_rows,
    routed_evaluation,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
    function_and_jvp as dense_function_and_jvp,
    normalized_expert_loss,
    rademacher,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import LayerState
from examples.nanogpt.analyze_sparse_moe_sharedframe_fullrank_pregelu_oracle import (
    SharedFrameFullRankPreGelu,
    load_write_bases,
    make_module,
)
from examples.nanogpt.analyze_sparse_moe_shared_nonlinear_dictionary_oracle import (
    gelu_derivative,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    layer_state_from_mapping,
    load_terminal_snapshot,
)
from latent_weight_lab.block_fht import normalized_fht_last_dim


PLAN_SCHEMA = "nanogpt_sparse_moe_nodeprivate_fullframe_ceiling_plan_v1"
COORDINATE_SCHEMA = "nanogpt_sparse_moe_sharedframe_fullrank_pregelu_coordinates_v1"


class NodePrivateFullFrame(torch.nn.Module):
    """Parent topology with either selected-node private or frozen shared frame."""

    def __init__(
        self,
        parent: SharedFrameFullRankPreGelu,
        *,
        layers: list[int],
        private_frame: bool,
        device: str,
    ) -> None:
        super().__init__()
        self.input_width = parent.input_width
        self.write_rank = parent.write_rank
        self.hidden_width = parent.hidden_width
        self.padded_width = parent.padded_width
        self.tensor_layers = parent.tensor_layers
        self.experts = parent.experts
        self.layers = tuple(int(value) for value in layers)
        self.private_frame = bool(private_frame)
        if self.private_frame:
            self.raw_frames = torch.nn.ParameterDict({
                str(layer): torch.nn.Parameter(
                    parent.raw_frame.detach().clone()[None].repeat(
                        self.experts, 1, 1
                    )
                )
                for layer in self.layers
            })
        else:
            self.register_buffer(
                "shared_raw_frame", parent.raw_frame.detach().clone(), persistent=True
            )
        self.hidden_bias = torch.nn.Parameter(parent.hidden_bias.detach().clone())
        self.output_modulation = torch.nn.Parameter(
            parent.output_modulation.detach().clone()
        )
        self.register_buffer(
            "write_basis", parent.write_basis.detach().clone(), persistent=True
        )
        self.register_buffer(
            "procedural_signs",
            parent.procedural_signs.detach().clone(),
            persistent=False,
        )
        self.frame_scale = parent.frame_scale
        self.expansion_scale = parent.expansion_scale
        self.contraction_scale = parent.contraction_scale
        self.to(device=device, dtype=torch.float32)

    @property
    def device(self) -> str:
        return str(self.hidden_bias.device)

    def _selection(self, expert: int | None) -> slice:
        if expert is None:
            return slice(None)
        if not 0 <= int(expert) < self.experts:
            raise IndexError("expert index out of range")
        return slice(int(expert), int(expert) + 1)

    def input_frame(self, layer: int, expert: int | None = None) -> torch.Tensor:
        if int(layer) not in self.layers:
            raise KeyError("layer was not selected for the ceiling")
        selected = self._selection(expert)
        if self.private_frame:
            raw = self.raw_frames[str(int(layer))][selected]
            return F.normalize(raw, dim=-1) * self.frame_scale
        return F.normalize(self.shared_raw_frame, dim=-1) * self.frame_scale

    def _fht_pair(
        self,
        values: torch.Tensor,
        tangents: torch.Tensor,
        *,
        layer: int,
        selected: slice,
        input_sign_index: int,
        output_sign_index: int,
        output_width: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = torch.stack((values, tangents), dim=0)
        pair = F.pad(pair, (0, self.padded_width - pair.shape[-1]))
        signs_in = self.procedural_signs[
            layer, selected, input_sign_index
        ].to(dtype=pair.dtype)[None, :, None, :]
        signs_out = self.procedural_signs[
            layer, selected, output_sign_index
        ].to(dtype=pair.dtype)[None, :, None, :]
        pair = normalized_fht_last_dim(pair * signs_in) * signs_out
        pair = pair[..., : int(output_width)] * float(scale)
        return pair[0], pair[1]

    def coefficients_and_jvp(
        self,
        inputs: torch.Tensor,
        directions: torch.Tensor,
        *,
        layer: int,
        expert: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.shape != directions.shape or inputs.shape[-1] != self.input_width:
            raise ValueError("input and direction shape mismatch")
        selected = self._selection(expert)
        frame = self.input_frame(layer, expert)
        if self.private_frame:
            compact = torch.einsum("esd,erd->esr", inputs.float(), frame)
            compact_jvp = torch.einsum("esd,erd->esr", directions.float(), frame)
        else:
            compact = torch.einsum("esd,rd->esr", inputs.float(), frame)
            compact_jvp = torch.einsum("esd,rd->esr", directions.float(), frame)
        pre, pre_jvp = self._fht_pair(
            compact,
            compact_jvp,
            layer=layer,
            selected=selected,
            input_sign_index=0,
            output_sign_index=1,
            output_width=self.hidden_width,
            scale=self.expansion_scale,
        )
        pre = pre + self.hidden_bias[layer, selected, None, :]
        hidden = F.gelu(pre)
        hidden_jvp = gelu_derivative(pre) * pre_jvp
        compact, compact_jvp = self._fht_pair(
            hidden,
            hidden_jvp,
            layer=layer,
            selected=selected,
            input_sign_index=2,
            output_sign_index=3,
            output_width=self.write_rank,
            scale=self.contraction_scale,
        )
        modulation = self.output_modulation[layer, selected, None, :]
        return compact * modulation, compact_jvp * modulation

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


def make_candidate_control(
    plan: dict[str, Any],
    parent_plan: dict[str, Any],
    parent_state: dict[str, torch.Tensor],
    write_basis: torch.Tensor,
    *,
    device: str,
) -> tuple[NodePrivateFullFrame, NodePrivateFullFrame]:
    parent = make_module(parent_plan, write_basis, candidate=True, device=device)
    parent.load_state_dict(parent_state, strict=True)
    layers = [int(value) for value in plan["source"]["layers"]]
    candidate = NodePrivateFullFrame(
        parent, layers=layers, private_frame=True, device=device
    )
    control = NodePrivateFullFrame(
        parent, layers=layers, private_frame=False, device=device
    )
    del parent
    return candidate, control


def fit_joint(
    module: NodePrivateFullFrame,
    samples: dict[int, torch.Tensor],
    states: dict[int, LayerState],
    *,
    layers: list[int],
    plan: dict[str, Any],
    probe_seed: int,
) -> dict[str, Any]:
    fit = plan["fit_protocol"]
    live, directions, targets = {}, {}, {}
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
            targets[layer] = (output @ module.write_basis, jvp @ module.write_basis)
    parameters = list(module.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
    )
    losses, output_losses, jvp_losses = [], [], []
    maximum_gradient = 0.0
    gradients_finite = True
    initial_frames = {
        key: value.detach().clone() for key, value in module.raw_frames.items()
    } if module.private_frame else {}
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
            raise RuntimeError("non-finite node-private objective")
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in parameters
        )
        if not gradients_finite:
            raise RuntimeError("missing or non-finite node-private gradient")
        gradient = float(torch.nn.utils.clip_grad_norm_(
            parameters, float(fit["gradient_clip"])
        ))
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
        output_losses.append(float(output_loss.detach()))
        jvp_losses.append(float(jvp_loss.detach()))
    frame_movement = 0.0
    if module.private_frame:
        frame_movement = float(torch.stack([
            (module.raw_frames[key].detach() - initial).square().mean()
            for key, initial in initial_frames.items()
        ]).mean().sqrt())
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
        "private_raw_frame_movement_rms": frame_movement,
        "hidden_bias_rms": float(module.hidden_bias.detach().square().mean().sqrt()),
        "output_modulation_rms": float(
            module.output_modulation.detach().square().mean().sqrt()
        ),
    }


def compact_state(module: NodePrivateFullFrame) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def validate_plan(plan: dict[str, Any], path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("node-private ceiling plan schema mismatch")
    root = Path(__file__).resolve().parents[2]
    source = plan["source"]
    causal = plan["causal_parent"]
    for relative, expected, label in (
        (causal["group_tangent_result"], causal["group_tangent_result_sha256"], "causal result"),
        (source["parent_plan"], source["parent_plan_sha256"], "parent plan"),
        (source["parent_result"], source["parent_result_sha256"], "parent result"),
    ):
        if file_sha256(root / relative) != expected:
            raise ValueError(f"{label} hash drift")
    spec = plan["candidate"]
    expected = (
        int(source["tensor_layers"]) * int(source["num_experts"])
        * int(source["input_width"]) ** 2
        + int(source["input_width"]) * int(source["write_rank"])
        + int(source["tensor_layers"]) * int(source["num_experts"])
        * (int(source["expert_hidden_width"]) + int(source["write_rank"]))
    )
    if expected != int(spec["total_extrapolated_coordinates"]):
        raise ValueError("node-private coordinate accounting drift")
    if abs(
        int(spec["dense_paired_parameters_all_layers"]) / expected
        - float(spec["compression_ratio"])
    ) > 1e-12:
        raise ValueError("node-private compression accounting drift")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") is not None:
        if identity["entrypoint_sha256"] != file_sha256(Path(__file__)):
            raise ValueError("entrypoint hash drift")
        for relative, expected_hash in identity["helper_sha256"].items():
            if file_sha256(root / relative) != expected_hash:
                raise ValueError(f"helper hash drift: {relative}")
    if not file_sha256(path):
        raise AssertionError("unreachable empty plan hash")


def preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    parent_plan = json.loads(
        (Path(__file__).resolve().parents[2] / source["parent_plan"]).read_text()
    )
    generator = torch.Generator(device="cpu").manual_seed(20261751)
    basis, _ = torch.linalg.qr(torch.randn(
        int(source["input_width"]), int(source["write_rank"]), generator=generator
    ))
    parent = make_module(parent_plan, basis, candidate=True, device=device)
    with torch.no_grad():
        parent.output_modulation.fill_(0.05)
        parent.hidden_bias.copy_(
            torch.randn(parent.hidden_bias.shape, generator=generator).to(device) * 0.1
        )
    state = {key: value.detach().cpu() for key, value in parent.state_dict().items()}
    candidate, control = make_candidate_control(
        plan, parent_plan, state, basis, device=device
    )
    layers = [int(value) for value in source["layers"]]
    samples, states = {}, {}
    for layer in layers:
        samples[layer] = torch.randn(
            int(source["num_experts"]), 1024, int(source["input_width"]),
            generator=generator,
        )
        states[layer] = LayerState(
            router=torch.randn(
                int(source["num_experts"]), int(source["input_width"]),
                generator=generator,
            ),
            c_fc=torch.randn(
                int(source["num_experts"]), int(source["expert_hidden_width"]),
                int(source["input_width"]), generator=generator,
            ) * 0.02,
            c_proj=torch.randn(
                int(source["num_experts"]), int(source["input_width"]),
                int(source["expert_hidden_width"]), generator=generator,
            ) * (0.02 / math.sqrt(2.0 * int(source["tensor_layers"]))),
        )
    initial_x = samples[layers[0]].to(device)
    initial_direction = torch.randn_like(initial_x)
    with torch.no_grad():
        initial_difference = max(
            float((a - b).abs().max())
            for a, b in zip(
                candidate.function_and_jvp(
                    initial_x, initial_direction, layer=layers[0]
                ),
                control.function_and_jvp(
                    initial_x, initial_direction, layer=layers[0]
                ),
            )
        )
    original_steps = int(plan["fit_protocol"]["steps"])
    plan["fit_protocol"]["steps"] = 2
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        candidate_diag = fit_joint(
            candidate, samples, states, layers=layers, plan=plan, probe_seed=20261752
        )
        control_diag = fit_joint(
            control, samples, states, layers=layers, plan=plan, probe_seed=20261752
        )
    finally:
        plan["fit_protocol"]["steps"] = original_steps
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "schema_version": "nanogpt_sparse_moe_nodeprivate_fullframe_ceiling_preflight_v1",
        "device": device,
        "candidate_control_two_step_seconds": elapsed,
        "projected_full_two_bank_seconds": elapsed * (original_steps / 2.0) * 2.0,
        "candidate_control_initial_max_abs_difference": initial_difference,
        "candidate_selected_frame_coordinates": sum(
            value.numel() for value in candidate.raw_frames.values()
        ),
        "all_values_and_gradients_finite": all_tensor_values_finite({
            "candidate": candidate_diag, "control": control_diag,
        }),
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
        "candidate_diagnostics": candidate_diag,
        "control_diagnostics": control_diag,
    }


def absolute_output_pass(row: dict[str, Any], gates: dict[str, Any]) -> bool:
    return (
        row["mixture_recovery_mean"] >= float(
            gates["heldout_mixture_recovery_mean_min_each_bank"]
        )
        and row["mixture_recovery_minimum_layer"] >= float(
            gates["heldout_mixture_recovery_every_layer_min_each_bank"]
        )
        and row["minimum_expert_recovery"] >= float(
            gates["heldout_expert_recovery_min_each_bank"]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--parent-coordinates", type=Path)
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
        print(json.dumps(preflight(plan, args.device), indent=2, sort_keys=True))
        return
    required = (
        args.parent_plan, args.parent_coordinates, args.write_bases,
        args.terminal_snapshot, args.data_dir, args.output,
    )
    if any(value is None for value in required):
        parser.error("full ceiling requires all source and output paths")

    started = time.time()
    source = plan["source"]
    for path, expected, label in (
        (args.parent_plan, source["parent_plan_sha256"], "parent plan"),
        (args.parent_coordinates, source["parent_coordinates_sha256"], "coordinates"),
        (args.write_bases, source["write_basis_artifact_sha256"], "write bases"),
        (args.terminal_snapshot, source["terminal_manifold_snapshot_sha256"], "snapshot"),
        (args.data_dir / "manifest.json", source["dataset_manifest_sha256"], "manifest"),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"{label} hash drift")
    parent_plan = json.loads(args.parent_plan.read_text(encoding="utf-8"))
    coordinates = torch.load(args.parent_coordinates, map_location="cpu", weights_only=False)
    if coordinates.get("schema_version") != COORDINATE_SCHEMA:
        raise ValueError("parent coordinate schema drift")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal step drift")
    write_bases = load_write_bases(args.write_bases, parent_plan)
    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, parent_plan, args.data_dir, args.device)
    mapping = dict(model.named_parameters())
    layers = [int(value) for value in source["layers"]]
    states = {layer: layer_state_from_mapping(mapping, layer) for layer in layers}
    del mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    samples, occupancy = {}, {}
    data = plan["data_protocol"]
    for bank_index, bank in enumerate(BANKS):
        samples[bank], occupancy[bank] = {}, {}
        for layer in layers:
            values, counts = route_and_sample(
                states[layer], inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=int(data["fit_samples_per_expert"]),
                seed=(
                    int(data["sample_selection_seed_base"])
                    + int(data["sample_selection_seed_bank_stride"]) * bank_index
                    + int(data["sample_selection_seed_layer_stride"]) * layer
                ),
            )
            samples[bank][layer] = values
            occupancy[bank][str(layer)] = counts

    diagnostics, summaries, saved = {}, {}, {}
    actions: dict[tuple[str, str, int], torch.Tensor] = {}
    for bank_index, bank in enumerate(BANKS):
        candidate, control = make_candidate_control(
            plan, parent_plan, coordinates["states"][bank]["candidate"],
            write_bases[bank], device=args.device,
        )
        live = samples[bank][layers[0]].to(args.device)
        direction = rademacher(tuple(live.shape), 20261761, args.device)
        with torch.no_grad():
            initial_difference = max(
                float((left - right).abs().max())
                for left, right in zip(
                    candidate.function_and_jvp(live, direction, layer=layers[0]),
                    control.function_and_jvp(live, direction, layer=layers[0]),
                )
            )
        if initial_difference > 1e-6:
            raise RuntimeError("candidate/control initial function drift")
        probe_seed = (
            int(data["fit_jvp_probe_seed_base"])
            + int(data["fit_jvp_probe_seed_bank_stride"]) * bank_index
        )
        diagnostics[bank] = {
            "candidate": fit_joint(
                candidate, samples[bank], states, layers=layers,
                plan=plan, probe_seed=probe_seed,
            ),
            "control": fit_joint(
                control, samples[bank], states, layers=layers,
                plan=plan, probe_seed=probe_seed,
            ),
            "candidate_control_initial_max_abs_difference": initial_difference,
        }
        summaries[bank] = {"candidate": {}, "control": {}}
        for layer in layers:
            for name, module in (("candidate", candidate), ("control", control)):
                evaluation = routed_evaluation(
                    states[layer], inputs["heldout"][layer], module,
                    module.write_basis, layer=layer,
                    outer_top_k=int(source["outer_moe_top_k"]),
                    probe_seed=int(data["heldout_jvp_probe_seed_base"]) + 17 * layer,
                )
                actions[(name, bank, layer)] = evaluation["predicted"]
                summaries[bank][name][str(layer)] = {
                    key: value for key, value in evaluation.items()
                    if key not in {"predicted", "target"}
                }
        for name in ("candidate", "control"):
            summaries[bank][name]["aggregate"] = aggregate_rows(
                summaries[bank][name]
            )
        saved[bank] = {
            "candidate": compact_state(candidate),
            "control": compact_state(control),
        }
        del candidate, control, live, direction
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    agreement = {
        name: sum(
            action_cosine(
                actions[(name, BANKS[0], layer)],
                actions[(name, BANKS[1], layer)],
            )
            for layer in layers
        ) / len(layers)
        for name in ("candidate", "control")
    }
    frozen = plan["gates"]
    gates = {}
    for bank in BANKS:
        candidate_row = summaries[bank]["candidate"]["aggregate"]
        control_row = summaries[bank]["control"]["aggregate"]
        output_gain = (
            candidate_row["mixture_recovery_mean"]
            - control_row["mixture_recovery_mean"]
        )
        jvp_gain = (
            candidate_row["jvp_recovery_mean"]
            - control_row["jvp_recovery_mean"]
        )
        row = {
            "absolute_output_pass": absolute_output_pass(candidate_row, frozen),
            "absolute_jvp_pass": candidate_row["jvp_recovery_mean"] >= float(
                frozen["heldout_jvp_recovery_mean_min_each_bank"]
            ),
            "action_agreement_pass": agreement["candidate"] >= float(
                frozen["heldout_bank_action_cosine_mean_min"]
            ),
            "output_gain": output_gain,
            "output_gain_pass": output_gain >= float(
                frozen["candidate_minus_continued_control_recovery_mean_min_each_bank"]
            ),
            "jvp_gain": jvp_gain,
            "jvp_gain_pass": jvp_gain >= float(
                frozen["candidate_minus_continued_control_jvp_recovery_mean_min_each_bank"]
            ),
            "occupancy_pass": min(min(values) for values in occupancy[bank].values()) >= int(
                frozen["minimum_discovery_assignments_per_expert"]
            ),
            "finite_pass": all_tensor_values_finite({
                "summary": summaries[bank], "diagnostics": diagnostics[bank]
            }),
        }
        row["all_pass"] = all(
            value for key, value in row.items() if key.endswith("_pass")
        )
        gates[bank] = row
    passed = all(gates[bank]["all_pass"] for bank in BANKS)
    output_pass = all(
        gates[bank]["absolute_output_pass"]
        and gates[bank]["output_gain_pass"]
        and gates[bank]["action_agreement_pass"]
        and gates[bank]["occupancy_pass"]
        and gates[bank]["finite_pass"]
        for bank in BANKS
    )
    if passed:
        classification = "NODEPRIVATE_FULLFRAME_CEILING_PASSES"
    elif output_pass:
        classification = "NODEPRIVATE_OUTPUT_ONLY_JVP_REJECTED"
    else:
        classification = "NODEPRIVATE_FULLFRAME_CEILING_REJECTED"

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "nodeprivate_coordinates.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_nodeprivate_fullframe_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_nodeprivate_fullframe_ceiling_result_v1",
        "classification": classification,
        "passed": passed,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "parent_plan_sha256": file_sha256(args.parent_plan),
            "parent_coordinates_sha256": file_sha256(args.parent_coordinates),
            "write_basis_artifact_sha256": file_sha256(args.write_bases),
            "terminal_snapshot_sha256": file_sha256(args.terminal_snapshot),
            "dataset_manifest_sha256": file_sha256(args.data_dir / "manifest.json"),
        },
        "execution": {
            "device": args.device,
            "wall_seconds": time.time() - started,
            "checkpoint_updates": 0,
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
            ),
            "coordinates_path": str(coordinates_path),
            "coordinates_sha256": file_sha256(coordinates_path),
        },
        "accounting": plan["candidate"],
        "occupancy": occupancy,
        "diagnostics": diagnostics,
        "summaries": summaries,
        "cross_bank_action_cosine": agreement,
        "gates": gates,
        "all_values_finite": all_tensor_values_finite({
            "diagnostics": diagnostics, "summaries": summaries,
            "agreement": agreement, "gates": gates,
        }),
        "authorization": {
            "private_frame_implementation": False,
            "decomposition_audit": passed,
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
