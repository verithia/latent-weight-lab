#!/usr/bin/env python3
"""Gate paired Householder and Monarch transports for complete sparse-MoE MLPs."""
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
from examples.nanogpt.analyze_sparse_moe_fullwidth_pregelu_chart_oracle import (
    FullWidthPreGeluChart,
)
from examples.nanogpt.analyze_sparse_moe_global_write_givens_feature_oracle import (
    BANKS,
    aggregate_rows,
    apply_givens_stages,
    fit_joint,
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


PLAN_SCHEMA = "nanogpt_sparse_moe_paired_ambient_transport_oracle_plan_v1"
TRIALS = ("householder", "monarch", "control")


def base_coordinate_count(
    *, rank: int, input_width: int, hidden_width: int,
    tensor_layers: int, experts: int,
) -> int:
    nodes = int(tensor_layers) * int(experts)
    return (
        2 * int(rank) * int(input_width)
        + nodes * (2 * int(hidden_width) + int(rank))
    )


def transport_coordinate_count(
    *, kind: str, tensor_layers: int, hidden_width: int,
    householder_reflectors: int, monarch_block_width: int,
) -> int:
    if kind == "control":
        return 0
    if kind == "householder":
        return (
            int(tensor_layers) * int(householder_reflectors) * int(hidden_width)
        )
    if kind == "monarch":
        return 2 * int(tensor_layers) * int(hidden_width) * int(monarch_block_width)
    raise ValueError(f"unknown transport kind: {kind}")


def _apply_householder(
    values: torch.Tensor, vectors: torch.Tensor, *, transpose: bool,
) -> torch.Tensor:
    sequence = reversed(vectors.unbind(0)) if transpose else vectors.unbind(0)
    result = values
    for vector in sequence:
        unit = F.normalize(vector, dim=0, eps=1e-8)
        projection = torch.einsum("...h,h->...", result, unit)
        result = result - 2.0 * projection[..., None] * unit
    return result


def _apply_blocks(
    values: torch.Tensor, blocks: torch.Tensor, *, transpose: bool,
) -> torch.Tensor:
    width = int(blocks.shape[-1])
    shaped = values.reshape(*values.shape[:-1], -1, width)
    matrix = blocks.transpose(-1, -2) if transpose else blocks
    return torch.einsum("...ni,nji->...nj", shaped, matrix).reshape_as(values)


class PairedAmbientTransportChart(FullWidthPreGeluChart):
    """Full-width procedural MLP with a paired hidden-space transport."""

    def __init__(
        self,
        *,
        transport_kind: str,
        householder_reflectors: int,
        monarch_block_width: int,
        monarch_permutation_seed: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(learn_angles=False, **kwargs)
        self.transport_kind = str(transport_kind)
        self.householder_reflectors = int(householder_reflectors)
        self.monarch_block_width = int(monarch_block_width)
        if self.transport_kind == "householder":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(monarch_permutation_seed) + 1)
                self.householder_vectors = torch.nn.Parameter(torch.randn(
                    self.tensor_layers,
                    self.householder_reflectors,
                    self.hidden_width,
                ))
            self.register_parameter("monarch_blocks", None)
        elif self.transport_kind == "monarch":
            if self.hidden_width % self.monarch_block_width:
                raise ValueError("Monarch block width must divide hidden width")
            blocks = self.hidden_width // self.monarch_block_width
            identity = torch.eye(self.monarch_block_width).expand(
                self.tensor_layers, 2, blocks,
                self.monarch_block_width, self.monarch_block_width,
            ).clone()
            self.monarch_blocks = torch.nn.Parameter(identity)
            self.register_parameter("householder_vectors", None)
        elif self.transport_kind == "control":
            self.register_parameter("householder_vectors", None)
            self.register_parameter("monarch_blocks", None)
        else:
            raise ValueError(f"unknown transport kind: {self.transport_kind}")

        generator = torch.Generator(device="cpu").manual_seed(
            int(monarch_permutation_seed)
        )
        permutation = torch.randperm(self.hidden_width, generator=generator)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(self.hidden_width)
        self.register_buffer("monarch_permutation", permutation, persistent=False)
        self.register_buffer("monarch_inverse", inverse, persistent=False)
        self.to(device=kwargs["device"], dtype=torch.float32)

    def transport_coordinates(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name in {"householder_vectors", "monarch_blocks"}
        )

    def _transport(self, values: torch.Tensor, *, layer: int, transpose: bool) -> torch.Tensor:
        if self.transport_kind == "control":
            return values
        if self.transport_kind == "householder":
            return _apply_householder(
                values, self.householder_vectors[layer], transpose=transpose
            )
        first, second = self.monarch_blocks[layer].unbind(0)
        if not transpose:
            result = values.index_select(-1, self.monarch_inverse)
            result = _apply_blocks(result, first, transpose=False)
            result = result.index_select(-1, self.monarch_permutation)
            return _apply_blocks(result, second, transpose=False)
        result = _apply_blocks(values, second, transpose=True)
        result = result.index_select(-1, self.monarch_inverse)
        result = _apply_blocks(result, first, transpose=True)
        return result.index_select(-1, self.monarch_permutation)

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
        compact = apply_givens_stages(compact, None, self.pre_matching)
        compact_jvp = apply_givens_stages(compact_jvp, None, self.pre_matching)
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
        pre = self._transport(pre, layer=layer, transpose=False)
        pre_jvp = self._transport(pre_jvp, layer=layer, transpose=False)
        gain = self.input_gain[layer, selected, None, :]
        pre = pre * gain + self.hidden_bias[layer, selected, None, :]
        pre_jvp = pre_jvp * gain
        hidden = F.gelu(pre)
        hidden_jvp = gelu_derivative(pre) * pre_jvp
        hidden = self._transport(hidden, layer=layer, transpose=True)
        hidden_jvp = self._transport(hidden_jvp, layer=layer, transpose=True)
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
        compact = apply_givens_stages(compact, None, self.post_matching)
        compact_jvp = apply_givens_stages(compact_jvp, None, self.post_matching)
        output_gain = self.output_gain[layer, selected, None, :]
        return compact * output_gain, compact_jvp * output_gain


def make_module(
    plan: dict[str, Any], write_basis: torch.Tensor, *, kind: str, device: str,
) -> PairedAmbientTransportChart:
    source, chart, transport = (
        plan["source"], plan["shared_chart"], plan["transports"]
    )
    return PairedAmbientTransportChart(
        write_basis=write_basis,
        hidden_width=int(source["expert_hidden_width"]),
        padded_width=int(source["procedural_padded_width"]),
        tensor_layers=int(source["tensor_layers"]),
        experts=int(source["num_experts"]),
        feature_seed=int(chart["feature_seed"]),
        pre_matching_seed=int(chart["pre_matching_seed"]),
        post_matching_seed=int(chart["post_matching_seed"]),
        procedural_map_seed=int(chart["procedural_map_seed"]),
        transport_kind=kind,
        householder_reflectors=int(transport["householder"]["reflectors"]),
        monarch_block_width=int(transport["monarch"]["block_width"]),
        monarch_permutation_seed=int(transport["monarch"]["permutation_seed"]),
        device=device,
    )


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("paired ambient transport plan schema mismatch")
    root = Path(__file__).resolve().parents[2]
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed")
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    for relative, expected in identity["source_artifact_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"source artifact hash drift: {relative}")
    source, chart, transports = (
        plan["source"], plan["shared_chart"], plan["transports"]
    )
    base = base_coordinate_count(
        rank=int(chart["rank"]), input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        tensor_layers=int(source["tensor_layers"]), experts=int(source["num_experts"]),
    )
    if base != int(chart["base_coordinates_all_layers"]):
        raise ValueError("base coordinate accounting drift")
    for kind in TRIALS:
        extra = transport_coordinate_count(
            kind=kind,
            tensor_layers=int(source["tensor_layers"]),
            hidden_width=int(source["expert_hidden_width"]),
            householder_reflectors=int(transports["householder"]["reflectors"]),
            monarch_block_width=int(transports["monarch"]["block_width"]),
        )
        spec = transports[kind]
        if extra != int(spec["transport_coordinates_all_layers"]):
            raise ValueError(f"{kind} transport coordinate accounting drift")
        if base + extra != int(spec["total_coordinates_all_layers"]):
            raise ValueError(f"{kind} total coordinate accounting drift")
    if not file_sha256(plan_path):
        raise AssertionError("empty plan hash")


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source, fit = plan["source"], plan["fit_protocol"]
    rank = int(plan["shared_chart"]["rank"])
    generator = torch.Generator(device="cpu").manual_seed(20261901)
    write, _ = torch.linalg.qr(torch.randn(
        int(source["input_width"]), rank, generator=generator
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
    modules = {kind: make_module(plan, write, kind=kind, device=device) for kind in TRIALS}
    live = samples[layers[0]].to(device)
    direction = torch.randn_like(live)
    zero_values = []
    for module in modules.values():
        output, jvp = module.function_and_jvp(live, direction, layer=layers[0])
        zero_values.extend((float(output.abs().max()), float(jvp.abs().max())))
    original_steps = fit["steps"]
    fit["steps"] = 2
    started = time.time()
    diagnostics = {}
    try:
        for kind in TRIALS:
            diagnostics[kind] = fit_joint(
                modules[kind], samples, states, layers=layers, plan=plan,
                probe_seed=20261902,
            )
    finally:
        fit["steps"] = original_steps
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_paired_ambient_transport_preflight_v1",
        "device": device,
        "three_trial_two_step_seconds": elapsed,
        "projected_full_two_bank_seconds": elapsed * (int(original_steps) / 2.0) * 2.0,
        "step_zero_output_and_jvp_max_abs": max(zero_values),
        "exact_step_zero_pass": max(zero_values) == 0.0,
        "coordinates": {kind: modules[kind].counted_coordinates() for kind in TRIALS},
        "transport_coordinates": {
            kind: modules[kind].transport_coordinates() for kind in TRIALS
        },
        "all_values_and_gradients_finite": all_finite(diagnostics),
        "diagnostics": diagnostics,
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
    }


def _gate(
    row: dict[str, Any], *, agreement: float, occupancy: dict[str, list[int]],
    diagnostics: dict[str, Any], frozen: dict[str, Any],
) -> dict[str, Any]:
    gates = {
        "mean_output_pass": row["mixture_recovery_mean"] >= float(
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
    }
    gates["all_pass"] = all(gates.values())
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
        parser.error("oracle requires write bases, terminal snapshot, data dir, and output")

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
        modules = {
            kind: make_module(plan, basis, kind=kind, device=args.device)
            for kind in TRIALS
        }
        base_names = ("raw_feature", "input_gain", "hidden_bias", "output_gain")
        for name in base_names:
            reference = getattr(modules["control"], name)
            for kind in ("householder", "monarch"):
                if not torch.equal(reference, getattr(modules[kind], name)):
                    raise RuntimeError(f"{kind}/control initialization drift: {name}")
        diagnostics[bank], summaries[bank], saved[bank] = {}, {}, {}
        for index, kind in enumerate(TRIALS):
            module = modules[kind]
            diagnostics[bank][kind] = fit_joint(
                module, samples[bank], states, layers=layers, plan=plan,
                probe_seed=(
                    int(plan["data_protocol"]["fit_jvp_probe_seed_base"])
                    + 1009 * bank_index
                ),
            )
            summaries[bank][kind] = {}
            for layer in layers:
                evaluation = routed_evaluation(
                    states[layer], inputs["heldout"][layer], module,
                    module.write_basis, layer=layer,
                    outer_top_k=int(source["outer_moe_top_k"]),
                    probe_seed=(
                        int(plan["data_protocol"]["heldout_jvp_probe_seed_base"])
                        + 17 * layer
                    ),
                )
                actions[(kind, bank, layer)] = evaluation["predicted"]
                summaries[bank][kind][str(layer)] = {
                    key: value for key, value in evaluation.items()
                    if key not in {"predicted", "target"}
                }
            summaries[bank][kind]["aggregate"] = aggregate_rows(
                summaries[bank][kind]
            )
            saved[bank][kind] = trainable_state(module)
        del modules
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    agreements = {
        kind: sum(
            action_cosine(
                actions[(kind, BANKS[0], layer)],
                actions[(kind, BANKS[1], layer)],
            )
            for layer in layers
        ) / len(layers)
        for kind in TRIALS
    }
    frozen = plan["candidate_gates"]
    gates = {}
    for bank in BANKS:
        gates[bank] = {}
        control = summaries[bank]["control"]["aggregate"]
        for kind in TRIALS:
            row = summaries[bank][kind]["aggregate"]
            gates[bank][kind] = _gate(
                row, agreement=agreements[kind], occupancy=occupancy[bank],
                diagnostics=diagnostics[bank][kind], frozen=frozen,
            )
            gates[bank][kind]["output_gain_over_control"] = (
                row["mixture_recovery_mean"] - control["mixture_recovery_mean"]
            )
            gates[bank][kind]["jvp_gain_over_control"] = (
                row["jvp_recovery_mean"] - control["jvp_recovery_mean"]
            )
            if kind != "control":
                gates[bank][kind]["causal_output_gain_pass"] = (
                    gates[bank][kind]["output_gain_over_control"] >= float(
                        frozen["candidate_minus_control_recovery_mean_min_each_bank"]
                    )
                )
                gates[bank][kind]["causal_jvp_gain_pass"] = (
                    gates[bank][kind]["jvp_gain_over_control"] >= float(
                        frozen["candidate_minus_control_jvp_recovery_mean_min_each_bank"]
                    )
                )
                gates[bank][kind]["all_pass"] = (
                    gates[bank][kind]["all_pass"]
                    and gates[bank][kind]["causal_output_gain_pass"]
                    and gates[bank][kind]["causal_jvp_gain_pass"]
                )
    passed = {
        kind: all(gates[bank][kind]["all_pass"] for bank in BANKS)
        for kind in ("householder", "monarch")
    }
    classification = (
        "PAIRED_AMBIENT_TRANSPORT_PASSES"
        if any(passed.values()) else "PAIRED_AMBIENT_TRANSPORT_REJECTED"
    )

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save({
        "schema_version": "nanogpt_sparse_moe_paired_ambient_transport_coordinates_v1",
        "states": saved,
    }, coordinates_path)
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "nanogpt_sparse_moe_paired_ambient_transport_oracle_result_v1",
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
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda") else 0
            ),
        },
        "accounting": {
            kind: {
                "total_coordinates": int(plan["transports"][kind]["total_coordinates_all_layers"]),
                "transport_coordinates": int(plan["transports"][kind]["transport_coordinates_all_layers"]),
                "compression_ratio": float(plan["transports"][kind]["paired_parameter_compression_ratio"]),
                "fused_extra_flops_per_active_expert_token": int(plan["transports"][kind]["fused_extra_flops_per_active_expert_token"]),
                "materialization_flops_per_expert_update": int(plan["transports"][kind]["materialization_flops_per_expert_update"]),
            }
            for kind in TRIALS
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
            "householder_training": passed["householder"],
            "monarch_training": passed["monarch"],
            "language_model_training": any(passed.values()),
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
