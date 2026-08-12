#!/usr/bin/env python3
"""Gate expert-specific paired-neuron coordinate fields for sparse-MoE MLPs."""
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
    channel_encoding,
    cpu_state_dict,
    fit_field,
    routed_evaluation,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_expert_paired_coordinate_field_oracle_plan_v1"


def decoder_parameter_count(input_width: int, hidden_width: int) -> int:
    return (
        input_width * hidden_width
        + hidden_width
        + hidden_width * hidden_width
        + hidden_width
        + hidden_width * 2
        + 2
    )


def coordinate_count(
    *, experts: int, hidden_width: int, code_width: int, decoder_input: int,
    decoder_hidden: int,
) -> int:
    return (
        experts * hidden_width * code_width
        + experts * hidden_width
        + experts * decoder_parameter_count(decoder_input, decoder_hidden)
        + experts * 2
    )


class ExpertPairedCoordinateField(torch.nn.Module):
    """Co-generate paired expert neurons without sharing decoder orientation."""

    def __init__(
        self,
        *,
        experts: int,
        input_width: int,
        hidden_width: int,
        code_width: int,
        encoding_frequencies: int,
        decoder_hidden_width: int,
        layer: int,
        tensor_layers: int,
        seed: int,
        device: str,
        channel_chunk: int = 96,
    ) -> None:
        super().__init__()
        self.experts = int(experts)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.code_width = int(code_width)
        self.encoding_frequencies = int(encoding_frequencies)
        self.encoding_width = 1 + 2 * self.encoding_frequencies
        self.decoder_hidden_width = int(decoder_hidden_width)
        self.channel_chunk = int(channel_chunk)
        if self.experts < 1 or self.channel_chunk <= 0:
            raise ValueError("experts and channel chunk must be positive")
        decoder_input = self.code_width + self.encoding_width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 1009 * int(layer))
            self.codes = torch.nn.Parameter(
                0.5 * torch.randn(self.experts, self.hidden_width, self.code_width)
            )
            self.hidden_bias = torch.nn.Parameter(
                torch.zeros(self.experts, self.hidden_width)
            )
            self.log_scales = torch.nn.Parameter(
                torch.tensor(
                    [
                        [math.log(0.02), math.log(0.02 / math.sqrt(2 * tensor_layers))]
                        for _ in range(self.experts)
                    ]
                )
            )
            self.decoder_weight_1 = torch.nn.Parameter(
                torch.empty(self.experts, decoder_input, self.decoder_hidden_width)
            )
            self.decoder_bias_1 = torch.nn.Parameter(
                torch.zeros(self.experts, self.decoder_hidden_width)
            )
            self.decoder_weight_2 = torch.nn.Parameter(
                torch.empty(
                    self.experts, self.decoder_hidden_width, self.decoder_hidden_width
                )
            )
            self.decoder_bias_2 = torch.nn.Parameter(
                torch.zeros(self.experts, self.decoder_hidden_width)
            )
            self.decoder_weight_3 = torch.nn.Parameter(
                torch.empty(self.experts, self.decoder_hidden_width, 2)
            )
            self.decoder_bias_3 = torch.nn.Parameter(
                torch.zeros(self.experts, 2)
            )
            for expert in range(self.experts):
                torch.nn.init.xavier_uniform_(self.decoder_weight_1[expert])
                torch.nn.init.xavier_uniform_(self.decoder_weight_2[expert])
                torch.nn.init.xavier_uniform_(self.decoder_weight_3[expert])
        self.register_buffer(
            "encoding",
            channel_encoding(self.input_width, self.encoding_frequencies, "cpu"),
            persistent=True,
        )
        self.to(device=device, dtype=torch.float32)

    def compact_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def decoder_parameters(self) -> list[torch.nn.Parameter]:
        return [
            self.decoder_weight_1,
            self.decoder_bias_1,
            self.decoder_weight_2,
            self.decoder_bias_2,
            self.decoder_weight_3,
            self.decoder_bias_3,
        ]

    def coordinate_parameters(self) -> list[torch.nn.Parameter]:
        return [self.codes, self.hidden_bias, self.log_scales]

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        pieces: list[torch.Tensor] = []
        codes = self.codes[:, :, None, :]
        for start in range(0, self.input_width, self.channel_chunk):
            stop = min(self.input_width, start + self.channel_chunk)
            count = stop - start
            fields = torch.cat(
                (
                    codes.expand(-1, -1, count, -1),
                    self.encoding[start:stop][None, None, :, :].expand(
                        self.experts, self.hidden_width, -1, -1
                    ),
                ),
                dim=-1,
            ).reshape(self.experts, self.hidden_width * count, -1)
            hidden = F.gelu(
                torch.bmm(fields, self.decoder_weight_1)
                + self.decoder_bias_1[:, None, :]
            )
            hidden = F.gelu(
                torch.bmm(hidden, self.decoder_weight_2)
                + self.decoder_bias_2[:, None, :]
            )
            paired = (
                torch.bmm(hidden, self.decoder_weight_3)
                + self.decoder_bias_3[:, None, :]
            ).reshape(self.experts, self.hidden_width, count, 2)
            pieces.append(paired)
        paired = torch.cat(pieces, dim=2) * self.log_scales.exp()[:, None, None, :]
        return paired[..., 0], paired[..., 1]

    def function_and_jvp(
        self, inputs: torch.Tensor, directions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from examples.nanogpt.analyze_sparse_moe_paired_coordinate_field_oracle import (
            function_and_jvp,
        )

        c_fc, c_proj_atoms = self.materialize()
        return function_and_jvp(
            inputs, directions, c_fc, c_proj_atoms, self.hidden_bias
        )


def result_authorization(passed: bool) -> dict[str, bool]:
    return {
        "implementation": bool(passed),
        "initialization_and_mapping_loss_shadow": bool(passed),
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
        "full_attention_work": False,
    }


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("expert paired-coordinate plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    control = plan["sealed_favorable_control"]
    if file_sha256(root / control["result_path"]) != control["result_sha256"]:
        raise ValueError("favorable shared control hash drift")
    source = plan["source"]
    candidate = plan["candidate"]
    expected = coordinate_count(
        experts=int(source["num_experts"]),
        hidden_width=int(source["expert_hidden_width"]),
        code_width=int(candidate["neuron_code_width"]),
        decoder_input=int(candidate["decoder_input_width"]),
        decoder_hidden=int(candidate["decoder_hidden_width"]),
    )
    if expected != int(candidate["total_coordinates_per_layer"]):
        raise ValueError("expert paired-coordinate accounting drift")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def make_field(
    plan: dict[str, Any], layer: int, device: str, seed: int,
) -> ExpertPairedCoordinateField:
    source = plan["source"]
    candidate = plan["candidate"]
    return ExpertPairedCoordinateField(
        experts=int(source["num_experts"]),
        input_width=int(source["input_width"]),
        hidden_width=int(source["expert_hidden_width"]),
        code_width=int(candidate["neuron_code_width"]),
        encoding_frequencies=(int(candidate["channel_encoding_width"]) - 1) // 2,
        decoder_hidden_width=int(candidate["decoder_hidden_width"]),
        layer=layer,
        tensor_layers=int(source["tensor_layers"]),
        seed=seed,
        device=device,
    )


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    field = make_field(plan, 0, device, 20261114)
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    source = plan["source"]
    shape = (int(source["num_experts"]), 16, int(source["input_width"]))
    inputs = torch.randn(shape, generator=generator)
    c_fc = torch.randn(
        int(source["num_experts"]), int(source["expert_hidden_width"]),
        int(source["input_width"]), generator=generator,
    ) * 0.02
    c_proj = torch.randn(
        int(source["num_experts"]), int(source["input_width"]),
        int(source["expert_hidden_width"]), generator=generator,
    ) * 0.02
    fit = plan["fit_protocol"]
    started = time.time()
    diagnostics = fit_field(
        field, inputs, c_fc, c_proj,
        steps=2,
        decoder_learning_rate=float(fit["decoder_learning_rate"]),
        coordinate_learning_rate=float(fit["code_bias_scale_learning_rate"]),
        decoder_weight_decay=float(fit["decoder_weight_decay"]),
        code_weight_decay=float(fit["code_weight_decay"]),
        gradient_clip=float(fit["gradient_clip"]),
        jvp_weight=0.10,
        probe_seed=20261116,
        train_decoder=True,
    )
    elapsed = time.time() - started
    return {
        "schema_version": "nanogpt_sparse_moe_expert_paired_coordinate_field_preflight_v1",
        "device": device,
        "two_step_wall_seconds_one_fit": elapsed,
        "projected_full_protocol_seconds": elapsed * (int(fit["steps"]) / 2.0) * 6.0,
        "compact_parameter_count": field.compact_parameter_count(),
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
        "all_values_finite": all_finite(diagnostics),
        "diagnostics": diagnostics,
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
    root = Path(__file__).resolve().parents[2]
    control_path = root / plan["sealed_favorable_control"]["result_path"]
    shared = json.loads(control_path.read_text(encoding="utf-8"))

    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    inputs = collect_protocol_inputs(model, plan, args.data_dir, args.device)
    mapping = dict(model.named_parameters())
    layers = [int(value) for value in source["layers"]]
    states = {layer: layer_state_from_mapping(mapping, layer) for layer in layers}
    del mapping, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    fit = plan["fit_protocol"]
    samples_per_expert = int(plan["data_protocol"]["fit_samples_per_expert"])
    banks = [row["name"] for row in plan["data_protocol"]["discovery_banks"]]
    saved: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    occupancy: dict[str, dict[str, list[int]]] = {}
    actions: dict[tuple[str, int], torch.Tensor] = {}
    for bank_index, bank in enumerate(banks):
        saved[bank] = {}
        summaries[bank] = {}
        diagnostics[bank] = {}
        occupancy[bank] = {}
        for layer in layers:
            state = states[layer]
            sampled, counts = route_and_sample(
                state, inputs[bank][layer], top_k=int(source["moe_top_k"]),
                samples_per_expert=samples_per_expert,
                seed=20261117 + 1009 * bank_index + 17 * layer,
            )
            occupancy[bank][str(layer)] = counts
            candidate = make_field(plan, layer, args.device, 20261114)
            candidate_diagnostics = fit_field(
                candidate, sampled, state.c_fc, state.c_proj,
                steps=int(fit["steps"]),
                decoder_learning_rate=float(fit["decoder_learning_rate"]),
                coordinate_learning_rate=float(fit["code_bias_scale_learning_rate"]),
                decoder_weight_decay=float(fit["decoder_weight_decay"]),
                code_weight_decay=float(fit["code_weight_decay"]),
                gradient_clip=float(fit["gradient_clip"]),
                jvp_weight=0.10,
                probe_seed=20261118 + 1009 * bank_index + 17 * layer,
                train_decoder=True,
            )
            candidate_eval = routed_evaluation(
                state, inputs["heldout"][layer], candidate,
                top_k=int(source["moe_top_k"]),
                probe_seed=20261119 + 17 * layer,
            )
            shared_recovery = float(shared["summaries"][bank][str(layer)]["mixture_recovery"])
            actions[(bank, layer)] = candidate_eval["predicted"]
            summaries[bank][str(layer)] = {
                "mixture_recovery": candidate_eval["mixture_recovery"],
                "jvp_recovery": candidate_eval["jvp_recovery"],
                "minimum_expert_recovery": min(candidate_eval["expert_recovery"]),
                "minimum_expert_jvp_recovery": min(candidate_eval["expert_jvp_recovery"]),
                "favorable_shared_recovery": shared_recovery,
                "candidate_minus_favorable_shared_recovery": (
                    candidate_eval["mixture_recovery"] - shared_recovery
                ),
            }
            diagnostics[bank][str(layer)] = candidate_diagnostics
            saved[bank][str(layer)] = cpu_state_dict(candidate)
            del candidate
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    frozen = plan["frozen_gates"]
    bank_gates: dict[str, dict[str, bool]] = {}
    for bank in banks:
        rows = [summaries[bank][str(layer)] for layer in layers]
        aggregate = {
            "mixture_recovery_mean": sum(float(row["mixture_recovery"]) for row in rows) / len(rows),
            "mixture_recovery_minimum_layer": min(float(row["mixture_recovery"]) for row in rows),
            "jvp_recovery_mean": sum(float(row["jvp_recovery"]) for row in rows) / len(rows),
            "minimum_expert_recovery": min(float(row["minimum_expert_recovery"]) for row in rows),
            "candidate_minus_favorable_shared_recovery_mean": sum(
                float(row["candidate_minus_favorable_shared_recovery"]) for row in rows
            ) / len(rows),
            "minimum_discovery_assignments": min(
                min(occupancy[bank][str(layer)]) for layer in layers
            ),
        }
        summaries[bank]["aggregate"] = aggregate
        bank_gates[bank] = {
            "mean_recovery_pass": aggregate["mixture_recovery_mean"] >= float(frozen["heldout_mixture_recovery_mean_min_each_bank"]),
            "every_layer_pass": aggregate["mixture_recovery_minimum_layer"] >= float(frozen["heldout_mixture_recovery_every_layer_min_each_bank"]),
            "every_expert_pass": aggregate["minimum_expert_recovery"] >= float(frozen["heldout_expert_recovery_min_each_bank"]),
            "jvp_pass": aggregate["jvp_recovery_mean"] >= float(frozen["heldout_jvp_recovery_mean_min_each_bank"]),
            "favorable_control_gain_pass": aggregate["candidate_minus_favorable_shared_recovery_mean"] >= float(frozen["candidate_minus_favorable_shared_recovery_mean_min_each_bank"]),
            "occupancy_pass": aggregate["minimum_discovery_assignments"] >= int(frozen["minimum_discovery_assignments_per_expert"]),
        }
    agreement_by_layer = {
        str(layer): action_cosine(actions[(banks[0], layer)], actions[(banks[1], layer)])
        for layer in layers
    }
    agreement_mean = sum(agreement_by_layer.values()) / len(agreement_by_layer)
    finite = all_finite(
        {"summaries": summaries, "diagnostics": diagnostics, "agreement": agreement_by_layer}
    )
    for bank in banks:
        bank_gates[bank]["action_agreement_pass"] = agreement_mean >= float(frozen["heldout_bank_action_cosine_mean_min"])
        bank_gates[bank]["finite_pass"] = finite
        bank_gates[bank]["all_pass"] = all(bank_gates[bank].values())
    passed = all(bank_gates[bank]["all_pass"] for bank in banks)

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "compact_coordinates.pt"
    torch.save(
        {
            "schema_version": "nanogpt_sparse_moe_expert_paired_coordinate_field_coordinates_v1",
            "fields": saved,
        },
        coordinates_path,
    )
    result = {
        "schema_version": "nanogpt_sparse_moe_expert_paired_coordinate_field_oracle_result_v1",
        "classification": (
            "EXPERT_PAIRED_COORDINATE_FIELD_REPRESENTABILITY_PASSES"
            if passed
            else "EXPERT_PAIRED_COORDINATE_FIELD_REPRESENTABILITY_REJECTED"
        ),
        "passed": passed,
        "identity": {
            "git_commit": git_commit(root),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "favorable_control_sha256": file_sha256(control_path),
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
            "dense_paired_parameters": int(plan["candidate"]["dense_paired_parameters_all_layers"]),
            "compact_coordinates": int(plan["candidate"]["total_coordinates_all_layers"]),
            "compression_ratio": float(plan["candidate"]["paired_parameter_compression_ratio"]),
            "materialized_dense_cfc_in_candidate": False,
            "materialized_dense_cproj_in_candidate": False,
            "expert_specific_decoders": True,
        },
        "occupancy": occupancy,
        "fit_diagnostics": diagnostics,
        "summaries": summaries,
        "heldout_bank_action_cosine": {"mean": agreement_mean, "by_layer": agreement_by_layer},
        "gates": bank_gates,
        "all_values_finite": finite,
        "authorization": result_authorization(passed),
    }
    result_path = args.output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
