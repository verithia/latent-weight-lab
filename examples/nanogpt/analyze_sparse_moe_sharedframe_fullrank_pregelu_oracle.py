#!/usr/bin/env python3
"""Gate a shared learned full-rank pre-GELU frame for complete sparse-MoE MLPs."""
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
from examples.nanogpt.analyze_sparse_moe_shared_nonlinear_dictionary_oracle import (
    gelu_derivative,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)
from latent_weight_lab.block_fht import normalized_fht_last_dim


PLAN_SCHEMA = "nanogpt_sparse_moe_sharedframe_fullrank_pregelu_oracle_plan_v1"
BASIS_SCHEMA = "nanogpt_sparse_moe_write_subspace_ceiling_bases_v1"


def coordinate_count(
    *, input_width: int, write_rank: int, hidden_width: int,
    tensor_layers: int, experts: int,
) -> int:
    nodes = int(tensor_layers) * int(experts)
    return (
        int(input_width) * int(input_width)
        + int(input_width) * int(write_rank)
        + nodes * (int(hidden_width) + int(write_rank))
    )


def procedural_signs(
    *, tensor_layers: int, experts: int, padded_width: int, base_seed: int,
) -> torch.Tensor:
    signs = torch.empty(
        int(tensor_layers), int(experts), 4, int(padded_width), dtype=torch.int8
    )
    for layer in range(int(tensor_layers)):
        for expert in range(int(experts)):
            for map_index in range(4):
                seed = (
                    int(base_seed)
                    + 10007 * layer
                    + 1009 * expert
                    + 97 * map_index
                )
                generator = torch.Generator(device="cpu").manual_seed(seed)
                bits = torch.randint(
                    0, 2, (int(padded_width),), generator=generator,
                    dtype=torch.int8,
                )
                signs[layer, expert, map_index] = bits.mul(2).sub(1)
    return signs


class SharedFrameFullRankPreGelu(torch.nn.Module):
    """Complete expert with a shared square input frame and compact write frame."""

    def __init__(
        self,
        *,
        write_basis: torch.Tensor,
        hidden_width: int,
        padded_width: int,
        tensor_layers: int,
        experts: int,
        input_frame_seed: int,
        procedural_map_seed: int,
        learn_frame: bool,
        device: str,
    ) -> None:
        super().__init__()
        if write_basis.ndim != 2:
            raise ValueError("write basis must be [input_width, write_rank]")
        self.input_width = int(write_basis.shape[0])
        self.write_rank = int(write_basis.shape[1])
        self.hidden_width = int(hidden_width)
        self.padded_width = int(padded_width)
        self.tensor_layers = int(tensor_layers)
        self.experts = int(experts)
        self.learn_frame = bool(learn_frame)
        if self.padded_width < max(
            self.input_width, self.hidden_width, self.write_rank
        ):
            raise ValueError("procedural padded width is too small")
        if self.padded_width & (self.padded_width - 1):
            raise ValueError("procedural padded width must be a power of two")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(input_frame_seed))
            raw_frame = torch.randn(self.input_width, self.input_width) * 0.02
        if self.learn_frame:
            self.raw_frame = torch.nn.Parameter(raw_frame)
        else:
            self.register_buffer("raw_frame", raw_frame, persistent=True)
        self.hidden_bias = torch.nn.Parameter(torch.zeros(
            self.tensor_layers, self.experts, self.hidden_width
        ))
        self.output_modulation = torch.nn.Parameter(torch.zeros(
            self.tensor_layers, self.experts, self.write_rank
        ))
        self.register_buffer(
            "write_basis", write_basis.detach().float().contiguous(), persistent=True
        )
        self.register_buffer(
            "procedural_signs",
            procedural_signs(
                tensor_layers=self.tensor_layers,
                experts=self.experts,
                padded_width=self.padded_width,
                base_seed=procedural_map_seed,
            ),
            persistent=False,
        )
        self.frame_scale = 0.02 * math.sqrt(float(self.input_width))
        self.expansion_scale = math.sqrt(
            float(self.padded_width) / float(self.input_width)
        )
        self.contraction_scale = math.sqrt(
            float(self.padded_width) / float(self.hidden_width)
        )
        self.to(device=device, dtype=torch.float32)

    @property
    def device(self) -> str:
        return str(self.hidden_bias.device)

    def counted_coordinates(self) -> int:
        return coordinate_count(
            input_width=self.input_width,
            write_rank=self.write_rank,
            hidden_width=self.hidden_width,
            tensor_layers=self.tensor_layers,
            experts=self.experts,
        )

    def input_frame(self) -> torch.Tensor:
        return F.normalize(self.raw_frame, dim=-1) * self.frame_scale

    def _selection(self, expert: int | None) -> slice:
        if expert is None:
            return slice(None)
        if not 0 <= int(expert) < self.experts:
            raise IndexError("expert index out of range")
        return slice(int(expert), int(expert) + 1)

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
        expected = self.experts if expert is None else 1
        if inputs.shape[0] != expected:
            raise ValueError("expert batch mismatch")
        selected = self._selection(expert)
        frame = self.input_frame()
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


def load_write_bases(path: Path, plan: dict[str, Any]) -> dict[str, torch.Tensor]:
    if file_sha256(path) != plan["source"]["write_basis_artifact_sha256"]:
        raise ValueError("write basis artifact hash drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != BASIS_SCHEMA:
        raise ValueError("write basis artifact schema mismatch")
    rank = int(plan["candidate"]["write_rank"])
    width = int(plan["source"]["input_width"])
    result = {}
    for bank in BANKS:
        basis = payload["bases"]["global_shared_rank619"][bank]
        if basis.shape[0] != 1 or basis.shape[1] != width:
            raise ValueError("global write basis shape drift")
        result[bank] = basis[0, :, :rank].float().contiguous()
    return result


def make_module(
    plan: dict[str, Any], write_basis: torch.Tensor, *, candidate: bool,
    device: str,
) -> SharedFrameFullRankPreGelu:
    source, spec = plan["source"], plan["candidate"]
    return SharedFrameFullRankPreGelu(
        write_basis=write_basis,
        hidden_width=int(spec["hidden_width"]),
        padded_width=int(source["procedural_padded_width"]),
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        input_frame_seed=int(spec["input_frame_seed"]),
        procedural_map_seed=int(spec["procedural_map_seed"]),
        learn_frame=bool(candidate),
        device=device,
    )


def compact_state(module: SharedFrameFullRankPreGelu) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def fit_joint(
    module: SharedFrameFullRankPreGelu,
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
            targets[layer] = (
                output @ module.write_basis,
                jvp @ module.write_basis,
            )
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
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
            raise RuntimeError("non-finite shared-frame objective")
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in parameters
        )
        if not gradients_finite:
            raise RuntimeError("missing or non-finite shared-frame gradient")
        gradient = float(torch.nn.utils.clip_grad_norm_(
            parameters, float(fit["gradient_clip"])
        ))
        maximum_gradient = max(maximum_gradient, gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
        output_losses.append(float(output_loss.detach()))
        jvp_losses.append(float(jvp_loss.detach()))
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
        "input_frame_raw_rms": float(module.raw_frame.detach().square().mean().sqrt()),
        "input_frame_row_norm_mean": float(module.input_frame().detach().norm(dim=-1).mean()),
        "hidden_bias_rms": float(module.hidden_bias.detach().square().mean().sqrt()),
        "output_modulation_rms": float(
            module.output_modulation.detach().square().mean().sqrt()
        ),
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("shared-frame plan schema mismatch")
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
    source = plan["source"]
    for key in (
        "write_ceiling_result", "input_atlas_result", "gelu_boundary_result",
        "fullwidth_result", "fixed_fullrank_result", "fisher_krylov_result",
    ):
        if file_sha256(root / source[key]) != source[f"{key}_sha256"]:
            raise ValueError(f"source result hash drift: {key}")
    candidate = plan["candidate"]
    expected = coordinate_count(
        input_width=int(source["input_width"]),
        write_rank=int(candidate["write_rank"]),
        hidden_width=int(candidate["hidden_width"]),
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
    )
    if expected != int(candidate["total_coordinates_all_layers"]):
        raise ValueError("candidate coordinate accounting drift")
    dense = int(source["dense_paired_parameters_all_layers"])
    if abs(dense / expected - float(candidate["paired_parameter_compression_ratio"])) > 1e-12:
        raise ValueError("candidate compression accounting drift")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source, fit = plan["source"], plan["fit_protocol"]
    generator = torch.Generator(device="cpu").manual_seed(20261621)
    write, _ = torch.linalg.qr(torch.randn(
        int(source["input_width"]), int(plan["candidate"]["write_rank"]),
        generator=generator,
    ))
    layers = [int(value) for value in source["layers"]]
    samples, states = {}, {}
    for layer in layers:
        samples[layer] = torch.randn(
            int(source["num_experts"]),
            int(plan["data_protocol"]["fit_samples_per_expert"]),
            int(source["input_width"]), generator=generator,
        ).contiguous()
        states[layer] = LayerState(
            router=torch.randn(
                int(source["num_experts"]), int(source["input_width"]),
                generator=generator,
            ),
            c_fc=(torch.randn(
                int(source["num_experts"]), int(source["expert_hidden_width"]),
                int(source["input_width"]), generator=generator,
            ) * 0.02).contiguous(),
            c_proj=(torch.randn(
                int(source["num_experts"]), int(source["input_width"]),
                int(source["expert_hidden_width"]), generator=generator,
            ) * (0.02 / math.sqrt(2.0 * int(source["tensor_layers"])))).contiguous(),
        )
    candidate = make_module(plan, write, candidate=True, device=device)
    control = make_module(plan, write, candidate=False, device=device)
    if not torch.equal(candidate.raw_frame, control.raw_frame):
        raise RuntimeError("candidate/control initial frame drift")
    if not torch.equal(candidate.procedural_signs, control.procedural_signs):
        raise RuntimeError("candidate/control procedural map drift")
    live = samples[layers[0]].to(device)
    direction = torch.randn_like(live)
    initial = []
    for module in (candidate, control):
        output, jvp = module.function_and_jvp(live, direction, layer=layers[0])
        initial.extend((float(output.abs().max()), float(jvp.abs().max())))
    original_steps = int(fit["steps"])
    fit["steps"] = 2
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        candidate_diag = fit_joint(
            candidate, samples, states, layers=layers, plan=plan,
            probe_seed=20261622,
        )
        control_diag = fit_joint(
            control, samples, states, layers=layers, plan=plan,
            probe_seed=20261622,
        )
    finally:
        fit["steps"] = original_steps
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "schema_version": "nanogpt_sparse_moe_sharedframe_fullrank_pregelu_preflight_v1",
        "device": device,
        "candidate_control_two_step_seconds": elapsed,
        "projected_full_two_bank_seconds": elapsed * (original_steps / 2.0) * 2.0,
        "step_zero_output_and_jvp_max_abs": max(initial),
        "exact_step_zero_pass": max(initial) == 0.0,
        "candidate_coordinates": candidate.counted_coordinates(),
        "control_materialized_coordinates": control.counted_coordinates(),
        "procedural_sign_storage_bytes": candidate.procedural_signs.numel(),
        "procedural_signs_persistent_in_coordinate_state": (
            "procedural_signs" in candidate.state_dict()
        ),
        "all_values_and_gradients_finite": all_finite({
            "candidate": candidate_diag, "control": control_diag
        }),
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
        "candidate_diagnostics": candidate_diag,
        "control_diagnostics": control_diag,
    }


def _absolute_gates(
    row: dict[str, Any], *, agreement: float, occupancy: dict[str, list[int]],
    diagnostics: dict[str, Any], compression: float, frozen: dict[str, Any],
) -> dict[str, Any]:
    gates = {
        "mean_recovery_pass": row["mixture_recovery_mean"] >= float(
            frozen["heldout_mixture_recovery_mean_min_each_bank"]
        ),
        "every_layer_pass": row["mixture_recovery_minimum_layer"] >= float(
            frozen["heldout_mixture_recovery_every_layer_min_each_bank"]
        ),
        "every_expert_pass": row["minimum_expert_recovery"] >= float(
            frozen["heldout_expert_recovery_min_each_bank"]
        ),
        "jvp_pass": row["jvp_recovery_mean"] >= float(
            frozen["heldout_jvp_recovery_mean_min_each_bank"]
        ),
        "action_agreement_pass": agreement >= float(
            frozen["heldout_bank_action_cosine_mean_min"]
        ),
        "occupancy_pass": min(min(values) for values in occupancy.values()) >= int(
            frozen["minimum_discovery_assignments_per_expert"]
        ),
        "finite_pass": all_finite({"summary": row, "diagnostics": diagnostics}),
        "compression_pass": compression >= float(
            frozen["paired_parameter_compression_ratio_min"]
        ),
    }
    gates["absolute_all_pass"] = all(gates.values())
    return gates


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
        parser.error(
            "oracle requires --write-bases, --terminal-snapshot, --data-dir, and --output"
        )

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
                layer=layer,
                outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=int(plan["data_protocol"]["heldout_jvp_probe_seed_base"]) + 17 * layer,
            )
            prerequisite_actions[(bank, layer)] = evaluation["predicted"]
            prerequisite_rows[bank][str(layer)] = {
                key: value for key, value in evaluation.items()
                if key not in {"predicted", "target"}
            }
        prerequisite_rows[bank]["aggregate"] = aggregate_rows(prerequisite_rows[bank])
    prerequisite_agreement = sum(
        action_cosine(
            prerequisite_actions[(BANKS[0], layer)],
            prerequisite_actions[(BANKS[1], layer)],
        ) for layer in layers
    ) / len(layers)
    prereq = plan["prerequisite_gates"]
    prerequisite_gates = {}
    for bank in BANKS:
        row = prerequisite_rows[bank]["aggregate"]
        prerequisite_gates[bank] = {
            "mean_output_pass": row["mixture_recovery_mean"] >= float(
                prereq["rank448_fixed_write_mixture_recovery_mean_min_each_bank"]
            ),
            "jvp_pass": row["jvp_recovery_mean"] >= float(
                prereq["rank448_fixed_write_jvp_recovery_mean_min_each_bank"]
            ),
            "every_layer_pass": row["mixture_recovery_minimum_layer"] >= float(
                prereq["rank448_fixed_write_every_layer_min_each_bank"]
            ),
            "every_expert_pass": row["minimum_expert_recovery"] >= float(
                prereq["rank448_fixed_write_every_expert_min_each_bank"]
            ),
            "action_agreement_pass": prerequisite_agreement >= float(
                prereq["rank448_cross_bank_projected_action_cosine_min"]
            ),
        }
        prerequisite_gates[bank]["all_pass"] = all(prerequisite_gates[bank].values())
    prerequisite_passed = all(prerequisite_gates[bank]["all_pass"] for bank in BANKS)
    if not prerequisite_passed:
        raise RuntimeError("registered rank-448 write prerequisite failed")

    samples, occupancy = {}, {}
    sample_count = int(plan["data_protocol"]["fit_samples_per_expert"])
    sample_seed = int(plan["data_protocol"]["sample_selection_seed_base"])
    for bank_index, bank in enumerate(BANKS):
        samples[bank], occupancy[bank] = {}, {}
        for layer in layers:
            sampled, counts = route_and_sample(
                states[layer], inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=sample_count,
                seed=sample_seed + 1009 * bank_index + 17 * layer,
            )
            samples[bank][layer] = sampled
            occupancy[bank][str(layer)] = counts

    summaries, diagnostics, saved = {}, {}, {}
    actions: dict[tuple[str, str, int], torch.Tensor] = {}
    for bank_index, bank in enumerate(BANKS):
        basis = write_bases[bank]
        candidate = make_module(plan, basis, candidate=True, device=args.device)
        control = make_module(plan, basis, candidate=False, device=args.device)
        for name in ("raw_frame", "hidden_bias", "output_modulation"):
            if not torch.equal(getattr(candidate, name), getattr(control, name)):
                raise RuntimeError("candidate/control initialization drift")
        if not torch.equal(candidate.procedural_signs, control.procedural_signs):
            raise RuntimeError("candidate/control procedural map drift")
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
                actions[(name, bank, layer)] = evaluation["predicted"]
                summaries[bank][name][str(layer)] = {
                    key: value for key, value in evaluation.items()
                    if key not in {"predicted", "target"}
                }
        for name in ("candidate", "control"):
            summaries[bank][name]["aggregate"] = aggregate_rows(summaries[bank][name])
        saved[bank] = {
            "candidate": compact_state(candidate),
            "control": compact_state(control),
        }
        del candidate, control
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    agreements = {
        name: sum(
            action_cosine(
                actions[(name, BANKS[0], layer)],
                actions[(name, BANKS[1], layer)],
            ) for layer in layers
        ) / len(layers)
        for name in ("candidate", "control")
    }
    frozen = plan["candidate_gates"]
    gates = {}
    for bank in BANKS:
        candidate_row = summaries[bank]["candidate"]["aggregate"]
        control_row = summaries[bank]["control"]["aggregate"]
        candidate_gates = _absolute_gates(
            candidate_row,
            agreement=agreements["candidate"],
            occupancy=occupancy[bank],
            diagnostics=diagnostics[bank]["candidate"],
            compression=float(plan["candidate"]["paired_parameter_compression_ratio"]),
            frozen=frozen,
        )
        output_gain = candidate_row["mixture_recovery_mean"] - control_row["mixture_recovery_mean"]
        jvp_gain = candidate_row["jvp_recovery_mean"] - control_row["jvp_recovery_mean"]
        candidate_gates.update({
            "candidate_minus_control_recovery_mean": output_gain,
            "candidate_minus_control_jvp_recovery_mean": jvp_gain,
            "output_frame_gain_pass": output_gain >= float(
                frozen["candidate_minus_fixed_frame_control_recovery_mean_min_each_bank"]
            ),
            "jvp_frame_gain_pass": jvp_gain >= float(
                frozen["candidate_minus_fixed_frame_control_jvp_recovery_mean_min_each_bank"]
            ),
        })
        candidate_gates["candidate_all_pass"] = (
            candidate_gates["absolute_all_pass"]
            and candidate_gates["output_frame_gain_pass"]
            and candidate_gates["jvp_frame_gain_pass"]
        )
        gates[bank] = candidate_gates
    passed = all(gates[bank]["candidate_all_pass"] for bank in BANKS)
    classification = (
        "SHAREDFRAME_FULLRANK_PRE_GELU_PASSES"
        if passed else "SHAREDFRAME_FULLRANK_PRE_GELU_REJECTED"
    )

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_sharedframe_fullrank_pregelu_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_sharedframe_fullrank_pregelu_oracle_result_v1",
        "classification": classification,
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
            "maximum_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
            ),
        },
        "accounting": {
            "candidate_coordinates": int(plan["candidate"]["total_coordinates_all_layers"]),
            "candidate_compression_ratio": float(plan["candidate"]["paired_parameter_compression_ratio"]),
            "control_materialized_coordinates": int(plan["fixed_frame_control"]["counted_materialized_coordinates"]),
            "dense_base_or_residual": False,
            "fixed_dense_procedural_map_storage": False,
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
        "cross_bank_action_cosine": agreements,
        "gates": gates,
        "all_values_finite": all_finite({
            "summaries": summaries, "diagnostics": diagnostics,
            "agreements": agreements,
        }),
        "authorization": {
            "causal_acquisition_theory": passed,
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
