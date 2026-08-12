#!/usr/bin/env python3
"""Gate a multi-atom feature/write chart for complete sparse-MoE MLPs."""
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
from examples.nanogpt.analyze_sparse_moe_paired_sparse_write_chart_oracle import (
    _static_flow_with_jvp,
    fit_chart,
    identity_permutation,
    support_indices,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import LayerState
from examples.nanogpt.analyze_sparse_moe_state_conditioned_butterfly_transport_oracle import (
    _gelu_derivative,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)
from latent_weight_lab.block_fht import normalized_fht_last_dim


PLAN_SCHEMA = "nanogpt_sparse_moe_multiatom_feature_write_oracle_plan_v1"


def coordinate_count(
    *, tensor_layers: int, experts: int, hidden_width: int, padded_width: int
) -> dict[str, int | float]:
    feature = tensor_layers * experts * 3 * padded_width
    bias = tensor_layers * experts * hidden_width
    sparse_write = tensor_layers * experts * 2 * hidden_width
    output_flow = tensor_layers * 3840
    compact = feature + bias + sparse_write + output_flow
    dense = tensor_layers * experts * 2 * 768 * hidden_width
    return {
        "feature": feature,
        "bias": bias,
        "sparse_write": sparse_write,
        "output_flow": output_flow,
        "compact": compact,
        "dense": dense,
        "compression": dense / compact,
    }


class MultiAtomFeatureWriteChart(torch.nn.Module):
    """Three additive signed-FHT feature atoms plus two-sparse writes."""

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
        independent: bool,
        raw_angle_initial_tanh: float = 0.125,
    ) -> None:
        super().__init__()
        self.experts = int(experts)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.padded_width = int(padded_width)
        self.tensor_layers = int(tensor_layers)
        self.paired = bool(independent)
        if self.input_width != 768 or self.hidden_width != 1536:
            raise ValueError("registered multi-atom chart requires 768/1536 widths")
        atom_signs = []
        base_scale = None
        for atom in range(3):
            atom_seed = int(seed) if atom == 0 or not independent else int(seed) + 104729 * atom
            operator = SpectralCFC(
                experts=self.experts,
                input_width=self.input_width,
                hidden_width=self.hidden_width,
                padded_width=self.padded_width,
                seed=atom_seed,
                layer=int(layer),
                device=device,
                context_beta=0.0,
            )
            atom_signs.append(operator.signs)
            if base_scale is None:
                base_scale = float(operator.base_scale)
            elif float(operator.base_scale) != base_scale:
                raise RuntimeError("feature atom base-scale drift")
        self.register_buffer("atom_signs", torch.stack(atom_signs))
        self.base_scale = float(base_scale)
        self.feature_spectra = torch.nn.Parameter(
            torch.zeros(self.experts, 3, self.padded_width)
        )
        self.hidden_bias = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width)
        )
        self.write_coefficients = torch.nn.Parameter(
            torch.zeros(self.experts, self.hidden_width, 2)
        )
        initial_raw = math.atanh(float(raw_angle_initial_tanh))
        self.output_binary_raw = torch.nn.Parameter(
            torch.full((1, 1, 8, 384), initial_raw)
        )
        self.output_cross_raw = torch.nn.Parameter(
            torch.full((1, 1, 3, 256), initial_raw)
        )
        self.register_buffer(
            "write_support", support_indices(self.hidden_width, self.input_width)[:, :2]
        )
        self.to(device=device, dtype=torch.float32)

    def trainable_parameters(self, *, paired: bool) -> list[torch.nn.Parameter]:
        if bool(paired) != self.paired:
            raise ValueError("independent flag disagrees with constructed chart")
        return [
            self.feature_spectra,
            self.hidden_bias,
            self.write_coefficients,
            self.output_binary_raw,
            self.output_cross_raw,
        ]

    def compact_parameter_count(self, *, paired: bool) -> int:
        return sum(p.numel() for p in self.trainable_parameters(paired=paired))

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
        padded_inputs = F.pad(
            inputs.to(dtype=torch.float32), (0, self.padded_width - self.input_width)
        )
        padded_directions = F.pad(
            directions.to(dtype=torch.float32), (0, self.padded_width - self.input_width)
        )
        total = torch.zeros_like(padded_inputs)
        total_jvp = torch.zeros_like(padded_directions)
        for atom in range(3):
            signs = self.atom_signs[atom, selected]
            values = normalized_fht_last_dim(
                padded_inputs * signs[:, 0, None, :]
            )
            tangent = normalized_fht_last_dim(
                padded_directions * signs[:, 0, None, :]
            )
            diagonal = self.feature_spectra[selected, atom, None, :]
            if atom == 0:
                diagonal = 1.0 + diagonal
            values = normalized_fht_last_dim(
                values * diagonal * signs[:, 1, None, :]
            )
            tangent = normalized_fht_last_dim(
                tangent * diagonal * signs[:, 1, None, :]
            )
            values = normalized_fht_last_dim(values * signs[:, 2, None, :])
            tangent = normalized_fht_last_dim(tangent * signs[:, 2, None, :])
            total = total + values
            total_jvp = total_jvp + tangent
        pre = self.base_scale * total[..., : self.hidden_width]
        pre_jvp = self.base_scale * total_jvp[..., : self.hidden_width]
        return pre + self.hidden_bias[selected, None, :], pre_jvp

    def _scatter_writes(
        self,
        hidden: torch.Tensor,
        hidden_jvp: torch.Tensor,
        *,
        selected: slice,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count, samples = hidden.shape[:2]
        indices = self.write_support.reshape(1, 1, -1).expand(count, samples, -1)
        coefficients = self.write_coefficients[selected, None, :, :]
        values = (hidden[..., None] * coefficients).reshape(count, samples, -1)
        tangent_values = (hidden_jvp[..., None] * coefficients).reshape(
            count, samples, -1
        )
        canonical = torch.zeros(
            count,
            samples,
            self.input_width,
            device=hidden.device,
            dtype=hidden.dtype,
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
            raise ValueError("independent flag disagrees with constructed chart")
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


def make_module(
    plan: dict[str, Any], layer: int, device: str, *, independent: bool
) -> MultiAtomFeatureWriteChart:
    source = plan["source"]
    return MultiAtomFeatureWriteChart(
        experts=int(source["num_experts"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        padded_width=int(source["padded_width"]),
        tensor_layers=int(source["tensor_layers"]),
        seed=int(plan["candidate"]["candidate_sign_seed"]),
        layer=int(layer),
        device=device,
        independent=independent,
        raw_angle_initial_tanh=0.125,
    )


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("multi-atom plan schema mismatch")
    root = Path(__file__).resolve().parents[2]
    identity = plan.get("identity")
    if identity is not None:
        if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
            raise ValueError("entrypoint hash is not sealed")
        for relative, expected in identity.get("helper_sha256", {}).items():
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
        padded_width=int(source["padded_width"]),
    )
    if int(counts["compact"]) != int(candidate["total_coordinates_all_layers"]):
        raise ValueError("multi-atom compact accounting drift")
    if int(counts["dense"]) != int(candidate["dense_paired_parameters_all_layers"]):
        raise ValueError("multi-atom dense accounting drift")
    if abs(float(counts["compression"]) - float(candidate["paired_parameter_compression_ratio"])) > 1e-12:
        raise ValueError("multi-atom compression drift")
    if float(counts["compression"]) < 200.0:
        raise ValueError("multi-atom chart is outside compression budget")
    support = support_indices()[:, :2]
    if torch.any(support[:, 0] == support[:, 1]):
        raise ValueError("registered two-support chart has collisions")
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
        "activation_alignment_weight": 0.0,
        "write_alignment_weight": 0.0,
        "probe_seed": int(probe_seed),
    }


def _trainable_equal(
    candidate: MultiAtomFeatureWriteChart,
    control: MultiAtomFeatureWriteChart,
) -> bool:
    return all(
        torch.equal(left, right)
        for left, right in zip(
            candidate.trainable_parameters(paired=True),
            control.trainable_parameters(paired=False),
        )
    )


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    generator = torch.Generator(device="cpu").manual_seed(20261242)
    inputs = torch.randn(
        int(source["num_experts"]), 16, int(source["input_width"]), generator=generator
    )
    c_fc = torch.randn(
        int(source["num_experts"]),
        int(source["expert_hidden_width"]),
        int(source["input_width"]),
        generator=generator,
    ) * 0.02
    c_proj = torch.randn(
        int(source["num_experts"]),
        int(source["input_width"]),
        int(source["expert_hidden_width"]),
        generator=generator,
    ) * (0.02 / math.sqrt(2.0 * int(source["tensor_layers"])))
    identity = identity_permutation(
        int(source["num_experts"]), int(source["expert_hidden_width"]), device
    )
    candidate = make_module(plan, 0, device, independent=True)
    control = make_module(plan, 0, device, independent=False)
    if not _trainable_equal(candidate, control):
        raise RuntimeError("candidate/control trainable initialization drift")
    with torch.no_grad():
        zeros = torch.zeros_like(inputs, device=device)
        live = inputs.to(device)
        candidate_initial = candidate.function_and_jvp(
            live, zeros, conditional=True
        )[0]
        control_initial = control.function_and_jvp(
            live, zeros, conditional=False
        )[0]
        initial_max_abs_difference = float(
            (candidate_initial - control_initial).abs().max()
        )
    if initial_max_abs_difference != 0.0:
        raise RuntimeError("candidate/control step-zero functions differ")
    started = time.time()
    candidate_diag = fit_chart(
        candidate,
        inputs,
        c_fc,
        c_proj,
        identity,
        paired=True,
        **_fit_kwargs(plan, steps=2, probe_seed=20261243),
    )
    control_diag = fit_chart(
        control,
        inputs,
        c_fc,
        c_proj,
        identity,
        paired=False,
        **_fit_kwargs(plan, steps=2, probe_seed=20261243),
    )
    elapsed = time.time() - started
    count = candidate.compact_parameter_count(paired=True)
    return {
        "schema_version": "nanogpt_sparse_moe_multiatom_feature_write_preflight_v1",
        "device": device,
        "two_step_wall_seconds_candidate_control": elapsed,
        "projected_full_protocol_seconds": elapsed * int(plan["fit_protocol"]["steps"]) * 6.0 / 2.0,
        "candidate_coordinate_count_per_layer": count,
        "control_coordinate_count_per_layer": control.compact_parameter_count(paired=False),
        "expected_coordinate_count_per_layer": int(plan["candidate"]["total_coordinates_per_layer"]),
        "step_zero_function_max_abs_difference": initial_max_abs_difference,
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
        raise FileExistsError("multi-atom output already exists")

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
    actions: dict[tuple[str, int], torch.Tensor] = {}
    for bank_index, bank in enumerate(banks):
        saved[bank], summaries[bank], diagnostics[bank], occupancy[bank] = {}, {}, {}, {}
        for layer in layers:
            state = states[layer]
            sampled, counts = route_and_sample(
                state,
                inputs[bank][layer],
                top_k=int(source["outer_moe_top_k"]),
                samples_per_expert=samples_per_expert,
                seed=20261244 + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            identity = identity_permutation(
                int(source["num_experts"]), int(source["expert_hidden_width"]), args.device
            )
            candidate = make_module(plan, layer, args.device, independent=True)
            control = make_module(plan, layer, args.device, independent=False)
            if not _trainable_equal(candidate, control):
                raise RuntimeError("candidate/control trainable initialization drift")
            fit_steps = int(plan["fit_protocol"]["steps"])
            candidate_diag = fit_chart(
                candidate,
                sampled,
                state.c_fc,
                state.c_proj,
                identity,
                paired=True,
                **_fit_kwargs(
                    plan,
                    steps=fit_steps,
                    probe_seed=20261245 + 1009 * bank_index + 17 * layer,
                ),
            )
            control_diag = fit_chart(
                control,
                sampled,
                state.c_fc,
                state.c_proj,
                identity,
                paired=False,
                **_fit_kwargs(
                    plan,
                    steps=fit_steps,
                    probe_seed=20261245 + 1009 * bank_index + 17 * layer,
                ),
            )
            candidate_eval = routed_evaluation(
                state,
                inputs["heldout"][layer],
                candidate,
                conditional=True,
                outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=20261246 + 17 * layer,
            )
            control_eval = routed_evaluation(
                state,
                inputs["heldout"][layer],
                control,
                conditional=False,
                outer_top_k=int(source["outer_moe_top_k"]),
                probe_seed=20261246 + 17 * layer,
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
                "aliased_control_recovery": control_eval["mixture_recovery"],
                "candidate_minus_aliased_control_recovery": candidate_eval["mixture_recovery"] - control_eval["mixture_recovery"],
                "sealed_static_ceiling_recovery": sealed_static,
                "candidate_minus_sealed_static_ceiling_recovery": candidate_eval["mixture_recovery"] - sealed_static,
            }
            diagnostics[bank][str(layer)] = {
                "candidate": candidate_diag,
                "control": control_diag,
            }
            saved[bank][str(layer)] = {
                "candidate": cpu_state_dict(candidate),
                "control": cpu_state_dict(control),
            }
            del candidate, control
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
            "candidate_minus_aliased_control_recovery_mean": sum(float(row["candidate_minus_aliased_control_recovery"]) for row in rows) / len(rows),
            "candidate_minus_sealed_static_ceiling_recovery_mean": sum(float(row["candidate_minus_sealed_static_ceiling_recovery"]) for row in rows) / len(rows),
            "minimum_discovery_assignments": min(min(occupancy[bank][str(layer)]) for layer in layers),
        }
        summaries[bank]["aggregate"] = aggregate
        gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": aggregate["minimum_expert_recovery"] >= float(frozen["heldout_expert_recovery_min_each_bank"]),
            "jvp_pass": aggregate["jvp_recovery_mean"] >= float(frozen["heldout_jvp_recovery_mean_min_each_bank"]),
            "aliased_control_gain_pass": aggregate["candidate_minus_aliased_control_recovery_mean"] >= float(frozen["candidate_minus_aliased_control_recovery_mean_min_each_bank"]),
            "static_ceiling_gain_pass": aggregate["candidate_minus_sealed_static_ceiling_recovery_mean"] >= float(frozen["candidate_minus_sealed_static_ceiling_recovery_mean_min_each_bank"]),
        }
    agreement_by_layer = {
        str(layer): action_cosine(actions[(banks[0], layer)], actions[(banks[1], layer)])
        for layer in layers
    }
    agreement_mean = sum(agreement_by_layer.values()) / len(agreement_by_layer)
    finite = all_finite({
        "summaries": summaries,
        "diagnostics": diagnostics,
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
        "schema_version": "nanogpt_sparse_moe_multiatom_feature_write_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    result = {
        "schema_version": "nanogpt_sparse_moe_multiatom_feature_write_oracle_result_v1",
        "classification": "MULTIATOM_FEATURE_WRITE_REPRESENTABILITY_PASSES" if passed else "MULTIATOM_FEATURE_WRITE_REPRESENTABILITY_REJECTED",
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
            "feature_atoms": 3,
            "write_supports_per_neuron": 2,
            "materialized_dense_cfc": False,
            "materialized_dense_cproj": False,
            "dense_learned_basis": False,
            "additive_lora_residual": False,
        },
        "occupancy": occupancy,
        "fit_diagnostics": diagnostics,
        "summaries": summaries,
        "heldout_bank_action_cosine": {"mean": agreement_mean, "by_layer": agreement_by_layer},
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
