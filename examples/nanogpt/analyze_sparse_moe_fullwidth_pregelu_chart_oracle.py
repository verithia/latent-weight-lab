#!/usr/bin/env python3
"""Gate a full-width pre-GELU procedural chart for complete sparse-MoE MLPs."""
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
    apply_givens_stages,
    fit_joint,
    fixed_matchings,
    load_write_bases,
    routed_evaluation,
    trainable_state,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import file_sha256
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


PLAN_SCHEMA = "nanogpt_sparse_moe_fullwidth_pregelu_chart_oracle_plan_v1"


def coordinate_count(
    *, rank: int, input_width: int, hidden_width: int, tensor_layers: int,
    experts: int, pre_stages: int, post_stages: int,
) -> int:
    nodes = int(tensor_layers) * int(experts)
    return (
        2 * int(rank) * int(input_width)
        + nodes
        * (
            (int(pre_stages) + int(post_stages)) * (int(rank) // 2)
            + 2 * int(hidden_width)
            + int(rank)
        )
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
                    0, 2, (int(padded_width),), generator=generator, dtype=torch.int8
                )
                signs[layer, expert, map_index] = bits.mul(2).sub(1)
    return signs


class FullWidthPreGeluChart(torch.nn.Module):
    """Complete MLP with procedural full-width GELU features and fixed global V."""

    def __init__(
        self,
        *,
        write_basis: torch.Tensor,
        hidden_width: int,
        padded_width: int,
        tensor_layers: int,
        experts: int,
        feature_seed: int,
        pre_matching_seed: int,
        post_matching_seed: int,
        procedural_map_seed: int,
        learn_angles: bool,
        device: str,
    ) -> None:
        super().__init__()
        if write_basis.ndim != 2:
            raise ValueError("write basis must be [input_width, rank]")
        self.input_width = int(write_basis.shape[0])
        self.rank = int(write_basis.shape[1])
        self.hidden_width = int(hidden_width)
        self.padded_width = int(padded_width)
        self.tensor_layers = int(tensor_layers)
        self.experts = int(experts)
        self.learn_angles = bool(learn_angles)
        if self.padded_width < max(self.rank, self.hidden_width):
            raise ValueError("procedural padded width is too small")
        if self.padded_width & (self.padded_width - 1):
            raise ValueError("procedural padded width must be a power of two")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(feature_seed))
            self.raw_feature = torch.nn.Parameter(
                torch.randn(self.rank, self.input_width) * 0.02
            )
        self.input_gain = torch.nn.Parameter(torch.ones(
            self.tensor_layers, self.experts, self.hidden_width
        ))
        self.hidden_bias = torch.nn.Parameter(torch.zeros(
            self.tensor_layers, self.experts, self.hidden_width
        ))
        self.output_gain = torch.nn.Parameter(torch.zeros(
            self.tensor_layers, self.experts, self.rank
        ))
        if self.learn_angles:
            self.pre_angles = torch.nn.Parameter(torch.zeros(
                self.tensor_layers, self.experts, 1, self.rank // 2
            ))
            self.post_angles = torch.nn.Parameter(torch.zeros_like(self.pre_angles))
        else:
            self.register_parameter("pre_angles", None)
            self.register_parameter("post_angles", None)

        self.register_buffer(
            "write_basis", write_basis.detach().float().contiguous(), persistent=True
        )
        self.register_buffer(
            "pre_matching", fixed_matchings(self.rank, 1, pre_matching_seed),
            persistent=False,
        )
        self.register_buffer(
            "post_matching", fixed_matchings(self.rank, 1, post_matching_seed),
            persistent=False,
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
        self.feature_scale = 0.02 * math.sqrt(float(self.input_width))
        self.expansion_scale = math.sqrt(float(self.padded_width) / float(self.rank))
        self.contraction_scale = math.sqrt(
            float(self.padded_width) / float(self.hidden_width)
        )
        self.to(device=device, dtype=torch.float32)

    @property
    def device(self) -> str:
        return str(self.raw_feature.device)

    @property
    def angles(self) -> torch.Tensor | None:
        if self.pre_angles is None:
            return None
        return torch.cat((self.pre_angles, self.post_angles), dim=2)

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
            raise ValueError("feature input/direction shape mismatch")
        expected = self.experts if expert is None else 1
        if inputs.shape[0] != expected:
            raise ValueError("feature expert batch mismatch")
        selected = self._selection(expert)
        feature = self.feature_basis()
        compact = torch.einsum("esd,rd->esr", inputs.float(), feature)
        compact_jvp = torch.einsum("esd,rd->esr", directions.float(), feature)
        pre_angles = None if self.pre_angles is None else self.pre_angles[layer, selected]
        compact = apply_givens_stages(compact, pre_angles, self.pre_matching)
        compact_jvp = apply_givens_stages(
            compact_jvp, pre_angles, self.pre_matching
        )
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
        gain = self.input_gain[layer, selected, None, :]
        pre = pre * gain + self.hidden_bias[layer, selected, None, :]
        pre_jvp = pre_jvp * gain
        hidden = F.gelu(pre)
        hidden_jvp = gelu_derivative(pre) * pre_jvp
        compact, compact_jvp = self._fht_pair(
            hidden,
            hidden_jvp,
            layer=layer,
            selected=selected,
            input_sign_index=2,
            output_sign_index=3,
            output_width=self.rank,
            scale=self.contraction_scale,
        )
        post_angles = None if self.post_angles is None else self.post_angles[layer, selected]
        compact = apply_givens_stages(compact, post_angles, self.post_matching)
        compact_jvp = apply_givens_stages(
            compact_jvp, post_angles, self.post_matching
        )
        output_gain = self.output_gain[layer, selected, None, :]
        return compact * output_gain, compact_jvp * output_gain

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


def make_module(
    plan: dict[str, Any], write_basis: torch.Tensor, *, candidate: bool,
    device: str,
) -> FullWidthPreGeluChart:
    source, spec = plan["source"], plan["candidate"]
    return FullWidthPreGeluChart(
        write_basis=write_basis,
        hidden_width=int(spec["hidden_width"]),
        padded_width=int(source["procedural_padded_width"]),
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        feature_seed=int(spec["feature_seed"]),
        pre_matching_seed=int(spec["pre_gelu_matching_seed"]),
        post_matching_seed=int(spec["post_gelu_matching_seed"]),
        procedural_map_seed=int(spec["procedural_map_seed"]),
        learn_angles=bool(candidate),
        device=device,
    )


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("full-width pre-GELU plan schema mismatch")
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
    source, candidate, control = (
        plan["source"], plan["candidate"], plan["location_control"]
    )
    for key in ("global_write_givens_result", "write_ceiling_result"):
        expected = source[f"{key}_sha256"]
        if file_sha256(root / source[key]) != expected:
            raise ValueError(f"source result hash drift: {key}")
    expected_candidate = coordinate_count(
        rank=int(candidate["rank"]),
        input_width=int(source["input_width"]),
        hidden_width=int(candidate["hidden_width"]),
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        pre_stages=int(candidate["pre_gelu_givens_stages"]),
        post_stages=int(candidate["post_gelu_givens_stages"]),
    )
    expected_control = coordinate_count(
        rank=int(candidate["rank"]),
        input_width=int(source["input_width"]),
        hidden_width=int(candidate["hidden_width"]),
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        pre_stages=0,
        post_stages=0,
    )
    if expected_candidate != int(candidate["total_coordinates_all_layers"]):
        raise ValueError("candidate coordinate accounting drift")
    if expected_control != int(control["total_coordinates_all_layers"]):
        raise ValueError("control coordinate accounting drift")
    if not file_sha256(plan_path):
        raise AssertionError("unreachable empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source, fit = plan["source"], plan["fit_protocol"]
    rank = int(plan["candidate"]["rank"])
    generator = torch.Generator(device="cpu").manual_seed(20261521)
    write, _ = torch.linalg.qr(torch.randn(
        int(source["input_width"]), rank, generator=generator
    ))
    layers = [int(value) for value in source["layers"]]
    samples, states = {}, {}
    for layer in layers:
        samples[layer] = torch.randn(
            int(source["num_experts"]),
            int(plan["data_protocol"]["fit_samples_per_expert"]),
            int(source["input_width"]),
            generator=generator,
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
        candidate_diag = fit_joint(
            candidate, samples, states, layers=layers, plan=plan,
            probe_seed=20261522,
        )
        control_diag = fit_joint(
            control, samples, states, layers=layers, plan=plan,
            probe_seed=20261522,
        )
    finally:
        fit["steps"] = original_steps
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_fullwidth_pregelu_chart_preflight_v1",
        "device": device,
        "candidate_control_two_step_seconds": elapsed,
        "projected_full_two_bank_seconds": elapsed * (int(original_steps) / 2.0) * 2.0,
        "step_zero_output_and_jvp_max_abs": max(initial),
        "exact_step_zero_pass": max(initial) == 0.0,
        "candidate_coordinates": candidate.counted_coordinates(),
        "control_coordinates": control.counted_coordinates(),
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

    prerequisite_rows: dict[str, Any] = {}
    prerequisite_actions: dict[tuple[str, int], torch.Tensor] = {}
    for bank in BANKS:
        prerequisite_rows[bank] = {}
        basis = write_bases[bank].to(args.device)
        for layer in layers:
            evaluation = routed_evaluation(
                states[layer], inputs["heldout"][layer], None, basis,
                layer=layer,
                outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=(
                    int(plan["data_protocol"]["heldout_jvp_probe_seed_base"])
                    + 17 * layer
                ),
            )
            prerequisite_actions[(bank, layer)] = evaluation["predicted"]
            prerequisite_rows[bank][str(layer)] = {
                key: value for key, value in evaluation.items()
                if key not in {"predicted", "target"}
            }
        prerequisite_rows[bank]["aggregate"] = aggregate_rows(
            prerequisite_rows[bank]
        )
    prerequisite_agreement = sum(
        action_cosine(
            prerequisite_actions[(BANKS[0], layer)],
            prerequisite_actions[(BANKS[1], layer)],
        )
        for layer in layers
    ) / len(layers)
    prereq = plan["prerequisite_gates"]
    prerequisite_gates = {}
    for bank in BANKS:
        row = prerequisite_rows[bank]["aggregate"]
        prerequisite_gates[bank] = {
            "mean_output_pass": row["mixture_recovery_mean"] >= float(
                prereq["rank480_fixed_write_mixture_recovery_mean_min_each_bank"]
            ),
            "jvp_pass": row["jvp_recovery_mean"] >= float(
                prereq["rank480_fixed_write_jvp_recovery_mean_min_each_bank"]
            ),
            "every_layer_pass": row["mixture_recovery_minimum_layer"] >= float(
                prereq["rank480_fixed_write_every_layer_min_each_bank"]
            ),
            "every_expert_pass": row["minimum_expert_recovery"] >= float(
                prereq["rank480_fixed_write_every_expert_min_each_bank"]
            ),
            "action_agreement_pass": prerequisite_agreement >= float(
                prereq["rank480_cross_bank_projected_action_cosine_min"]
            ),
        }
        prerequisite_gates[bank]["all_pass"] = all(
            prerequisite_gates[bank].values()
        )
    prerequisite_passed = all(
        prerequisite_gates[bank]["all_pass"] for bank in BANKS
    )
    if not prerequisite_passed:
        raise RuntimeError("registered rank-480 write prerequisite failed")

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
        for name in ("raw_feature", "input_gain", "hidden_bias", "output_gain"):
            if not torch.equal(getattr(candidate, name), getattr(control, name)):
                raise RuntimeError("candidate/control initialization drift")
        if not torch.equal(candidate.procedural_signs, control.procedural_signs):
            raise RuntimeError("candidate/control procedural-map drift")
        diagnostics[bank] = {
            "candidate": fit_joint(
                candidate, samples[bank], states, layers=layers, plan=plan,
                probe_seed=(
                    int(plan["data_protocol"]["fit_jvp_probe_seed_base"])
                    + 1009 * bank_index
                ),
            ),
            "control": fit_joint(
                control, samples[bank], states, layers=layers, plan=plan,
                probe_seed=(
                    int(plan["data_protocol"]["fit_jvp_probe_seed_base"])
                    + 1009 * bank_index
                ),
            ),
        }
        summaries[bank] = {"candidate": {}, "control": {}}
        for layer in layers:
            for name, module in (("candidate", candidate), ("control", control)):
                evaluation = routed_evaluation(
                    states[layer], inputs["heldout"][layer], module,
                    module.write_basis,
                    layer=layer,
                    outer_top_k=int(source["outer_moe_top_k"]),
                    probe_seed=(
                        int(plan["data_protocol"]["heldout_jvp_probe_seed_base"])
                        + 17 * layer
                    ),
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
            "candidate": trainable_state(candidate),
            "control": trainable_state(control),
        }
        del candidate, control
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    agreements = {
        name: sum(
            action_cosine(
                actions[(name, BANKS[0], layer)],
                actions[(name, BANKS[1], layer)],
            )
            for layer in layers
        ) / len(layers)
        for name in ("candidate", "control")
    }
    frozen = plan["candidate_gates"]
    gates: dict[str, Any] = {}
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
        output_gain = (
            candidate_row["mixture_recovery_mean"]
            - control_row["mixture_recovery_mean"]
        )
        jvp_gain = candidate_row["jvp_recovery_mean"] - control_row["jvp_recovery_mean"]
        candidate_gates.update({
            "candidate_minus_control_recovery_mean": output_gain,
            "candidate_minus_control_jvp_recovery_mean": jvp_gain,
            "output_causal_gain_pass": output_gain >= float(
                frozen["candidate_minus_location_control_recovery_mean_min_each_bank"]
            ),
            "jvp_causal_gain_pass": jvp_gain >= float(
                frozen["candidate_minus_location_control_jvp_recovery_mean_min_each_bank"]
            ),
        })
        candidate_gates["candidate_all_pass"] = (
            candidate_gates["absolute_all_pass"]
            and candidate_gates["output_causal_gain_pass"]
            and candidate_gates["jvp_causal_gain_pass"]
        )
        control_gates = _absolute_gates(
            control_row,
            agreement=agreements["control"],
            occupancy=occupancy[bank],
            diagnostics=diagnostics[bank]["control"],
            compression=float(plan["location_control"]["paired_parameter_compression_ratio"]),
            frozen=frozen,
        )
        control_gates["control_escape_all_pass"] = control_gates["absolute_all_pass"]
        gates[bank] = {"candidate": candidate_gates, "control": control_gates}
    candidate_passed = all(
        gates[bank]["candidate"]["candidate_all_pass"] for bank in BANKS
    )
    control_escape_passed = all(
        gates[bank]["control"]["control_escape_all_pass"] for bank in BANKS
    )
    passed = candidate_passed or control_escape_passed
    classification = (
        "FULLWIDTH_PRE_GELU_CHART_PASSES"
        if candidate_passed
        else "FULLWIDTH_PRE_GELU_CONTROL_ESCAPE_PASSES"
        if control_escape_passed
        else "FULLWIDTH_PRE_GELU_CHART_REJECTED"
    )

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_fullwidth_pregelu_chart_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_fullwidth_pregelu_chart_oracle_result_v1",
        "classification": classification,
        "passed": passed,
        "candidate_passed": candidate_passed,
        "control_escape_passed": control_escape_passed,
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
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda") else 0
            ),
        },
        "accounting": {
            "candidate_coordinates": int(
                plan["candidate"]["total_coordinates_all_layers"]
            ),
            "candidate_compression_ratio": float(
                plan["candidate"]["paired_parameter_compression_ratio"]
            ),
            "control_coordinates": int(
                plan["location_control"]["total_coordinates_all_layers"]
            ),
            "control_compression_ratio": float(
                plan["location_control"]["paired_parameter_compression_ratio"]
            ),
            "dense_base_or_residual": False,
            "fixed_dense_procedural_map_storage": False,
            "terminal_derived_input_or_write_atlas_is_deployable": False,
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
            "summaries": summaries,
            "diagnostics": diagnostics,
            "agreements": agreements,
        }),
        "authorization": {
            "causal_chart_theory": candidate_passed,
            "simpler_control_theory": control_escape_passed,
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
