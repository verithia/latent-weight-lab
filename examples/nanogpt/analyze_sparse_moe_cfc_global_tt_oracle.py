#!/usr/bin/env python3
"""Gate a global tensor-train representation of every sparse-MoE c_fc."""
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
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_sparse_moe_cfc_spectral_feature_oracle import (
    action_cosine,
)
from examples.nanogpt.analyze_sparse_moe_paired_alignment import (
    collect_inputs,
    file_sha256,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import (
    LayerState,
    recovery_fraction,
)
from examples.nanogpt.analyze_sparse_moe_stepzero_task_gradient_oracle import (
    all_finite,
    layer_state_from_mapping,
    load_terminal_snapshot,
)


PLAN_SCHEMA = "nanogpt_sparse_moe_cfc_global_tt_oracle_plan_v1"


def capped_bond_ranks(modes: list[int], rank_cap: int) -> list[int]:
    if not modes or any(int(mode) <= 0 for mode in modes):
        raise ValueError("TT modes must be positive and nonempty")
    total = math.prod(modes)
    left = 1
    ranks = [1]
    for mode in modes[:-1]:
        left *= int(mode)
        ranks.append(min(int(rank_cap), left, total // left))
    ranks.append(1)
    return ranks


def coordinate_count(modes: list[int], ranks: list[int]) -> int:
    if len(ranks) != len(modes) + 1:
        raise ValueError("TT ranks must bracket every mode")
    return sum(
        int(ranks[index]) * int(mode) * int(ranks[index + 1])
        for index, mode in enumerate(modes)
    )


def dense_to_physical(
    dense: torch.Tensor,
    hidden_modes: list[int],
    input_modes: list[int],
    *,
    interleaved: bool,
) -> torch.Tensor:
    if dense.ndim != 4:
        raise ValueError("dense c_fc tensor must be [layer, expert, hidden, input]")
    if math.prod(hidden_modes) != dense.shape[2]:
        raise ValueError("hidden modes do not match dense c_fc")
    if math.prod(input_modes) != dense.shape[3]:
        raise ValueError("input modes do not match dense c_fc")
    physical = dense.reshape(
        dense.shape[0], dense.shape[1], *hidden_modes, *input_modes
    )
    if not interleaved:
        return physical
    hidden_start = 2
    input_start = 2 + len(hidden_modes)
    permutation = [0, 1]
    for index in range(len(hidden_modes)):
        permutation.extend((hidden_start + index, input_start + index))
    return physical.permute(permutation).contiguous()


def physical_to_dense(
    physical: torch.Tensor,
    hidden_modes: list[int],
    input_modes: list[int],
    *,
    interleaved: bool,
) -> torch.Tensor:
    if len(hidden_modes) != len(input_modes):
        raise ValueError("hidden and input mode counts must agree")
    if interleaved:
        hidden_positions = [2 + 2 * index for index in range(len(hidden_modes))]
        input_positions = [3 + 2 * index for index in range(len(input_modes))]
        physical = physical.permute([0, 1, *hidden_positions, *input_positions])
    return physical.reshape(
        physical.shape[0],
        physical.shape[1],
        math.prod(hidden_modes),
        math.prod(input_modes),
    )


def randomized_left_factor(
    matrix: torch.Tensor,
    rank: int,
    *,
    seed: int,
    oversample: int,
    power_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return a rank-truncated left factor and propagated remainder."""
    if matrix.ndim != 2:
        raise ValueError("TT unfolding must be a matrix")
    rows, columns = matrix.shape
    maximum = min(rows, columns)
    if rank <= 0 or rank > maximum:
        raise ValueError("invalid requested unfolding rank")
    exact = rank == maximum
    if exact and rows <= columns:
        left = torch.eye(rows, device=matrix.device, dtype=matrix.dtype)
    elif exact:
        left = torch.linalg.qr(matrix, mode="reduced").Q[:, :rank]
    else:
        sketch_rank = min(maximum, rank + max(0, int(oversample)))
        generator = torch.Generator(device=matrix.device)
        generator.manual_seed(int(seed))
        omega = torch.randn(
            columns,
            sketch_rank,
            generator=generator,
            device=matrix.device,
            dtype=matrix.dtype,
        )
        q = torch.linalg.qr(matrix @ omega, mode="reduced").Q
        for _ in range(int(power_iterations)):
            q = torch.linalg.qr(matrix @ (matrix.T @ q), mode="reduced").Q
        projected = q.T @ matrix
        gram = projected @ projected.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        ordering = torch.argsort(eigenvalues, descending=True)[:rank]
        left = (q @ eigenvectors[:, ordering]).contiguous()
    remainder = left.T @ matrix
    total_energy = matrix.square().sum().clamp_min(1e-30)
    retained_energy = remainder.square().sum()
    reconstruction_error = float(
        (total_energy - retained_energy).clamp_min(0) / total_energy
    )
    return left, remainder, {
        "rows": int(rows),
        "columns": int(columns),
        "retained_rank": int(rank),
        "exact_split": bool(exact),
        "local_relative_error_squared": reconstruction_error,
    }


def randomized_tt_svd(
    physical: torch.Tensor,
    modes: list[int],
    ranks: list[int],
    *,
    seed: int,
    oversample: int,
    power_iterations: int,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    if list(physical.shape) != [int(mode) for mode in modes]:
        raise ValueError("physical tensor shape and registered modes disagree")
    if ranks != capped_bond_ranks(modes, max(ranks)):
        raise ValueError("registered TT ranks are not the capped ranks")
    remainder = physical.float()
    cores: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    for index, mode in enumerate(modes[:-1]):
        unfolding = remainder.reshape(int(ranks[index]) * int(mode), -1)
        left, propagated, row = randomized_left_factor(
            unfolding,
            int(ranks[index + 1]),
            seed=int(seed) + 1009 * index,
            oversample=int(oversample),
            power_iterations=int(power_iterations),
        )
        cores.append(
            left.reshape(int(ranks[index]), int(mode), int(ranks[index + 1]))
        )
        diagnostics.append({"split": index, **row})
        remainder = propagated.reshape(
            int(ranks[index + 1]), *[int(value) for value in modes[index + 1 :]]
        )
    cores.append(
        remainder.reshape(int(ranks[-2]), int(modes[-1]), int(ranks[-1]))
    )
    if sum(core.numel() for core in cores) != coordinate_count(modes, ranks):
        raise RuntimeError("materialized TT core count drift")
    return cores, diagnostics


def materialize_tt(cores: list[torch.Tensor]) -> torch.Tensor:
    if not cores or cores[0].shape[0] != 1 or cores[-1].shape[-1] != 1:
        raise ValueError("TT cores must have scalar boundary bonds")
    state = cores[0][0]
    for core in cores[1:]:
        if state.shape[-1] != core.shape[0]:
            raise ValueError("adjacent TT core bonds disagree")
        state = torch.tensordot(state, core, dims=([-1], [0]))
    return state[..., 0]


def materialize_expert_matrix(
    cores: list[torch.Tensor],
    layer: int,
    expert: int,
    hidden_modes: list[int],
    input_modes: list[int],
    *,
    interleaved: bool,
) -> torch.Tensor:
    if cores[0].shape[1] <= layer or cores[1].shape[1] <= expert:
        raise ValueError("layer or expert index outside TT modes")
    state = cores[0][0, int(layer)] @ cores[1][:, int(expert)]
    for core in cores[2:]:
        state = torch.tensordot(state, core, dims=([-1], [0]))
    physical = state[..., 0]
    if interleaved:
        permutation = [2 * index for index in range(len(hidden_modes))]
        permutation.extend(1 + 2 * index for index in range(len(input_modes)))
        physical = physical.permute(permutation)
    return physical.reshape(math.prod(hidden_modes), math.prod(input_modes))


def parameter_recovery(
    target: torch.Tensor,
    cores: list[torch.Tensor],
    hidden_modes: list[int],
    input_modes: list[int],
    *,
    interleaved: bool,
) -> dict[str, Any]:
    total_error = 0.0
    total_energy = 0.0
    by_layer: dict[str, float] = {}
    for layer in range(target.shape[0]):
        layer_error = 0.0
        layer_energy = 0.0
        for expert in range(target.shape[1]):
            candidate = materialize_expert_matrix(
                cores,
                layer,
                expert,
                hidden_modes,
                input_modes,
                interleaved=interleaved,
            )
            dense = target[layer, expert]
            layer_error += float((candidate - dense).square().sum())
            layer_energy += float(dense.square().sum())
        by_layer[str(layer)] = 1.0 - layer_error / max(layer_energy, 1e-30)
        total_error += layer_error
        total_energy += layer_energy
    return {
        "global": 1.0 - total_error / max(total_energy, 1e-30),
        "by_layer": by_layer,
    }


def routed_outputs(
    state: LayerState,
    activations: torch.Tensor,
    candidate_c_fc: torch.Tensor,
    *,
    top_k: int,
    chunk_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float]]:
    candidate_c_fc = candidate_c_fc.to(device=device, dtype=torch.float32)
    dense_c_fc = state.c_fc.to(device=device, dtype=torch.float32)
    dense_c_proj = state.c_proj.to(device=device, dtype=torch.float32)
    router = state.router.to(device=device, dtype=torch.float32)
    expert_error = [0.0 for _ in range(candidate_c_fc.shape[0])]
    expert_energy = [0.0 for _ in range(candidate_c_fc.shape[0])]
    pre_error = [0.0 for _ in range(candidate_c_fc.shape[0])]
    pre_energy = [0.0 for _ in range(candidate_c_fc.shape[0])]
    predicted_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for start in range(0, activations.shape[0], int(chunk_size)):
        x = activations[start : start + int(chunk_size)].to(
            device=device, dtype=torch.float32
        )
        logits = x @ router.T
        tie = torch.arange(logits.shape[-1], device=x.device, dtype=x.dtype)
        selected = torch.topk(
            logits - tie * torch.finfo(x.dtype).eps,
            int(top_k),
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        probabilities = F.softmax(logits.gather(-1, selected), dim=-1)
        predicted = torch.zeros_like(x)
        target = torch.zeros_like(x)
        for expert in range(candidate_c_fc.shape[0]):
            locations = (selected == expert).nonzero(as_tuple=False)
            if not locations.numel():
                continue
            token = locations[:, 0]
            slot = locations[:, 1]
            expert_input = x.index_select(0, token)
            candidate_pre = expert_input @ candidate_c_fc[expert].T
            target_pre = expert_input @ dense_c_fc[expert].T
            candidate_output = F.gelu(candidate_pre) @ dense_c_proj[expert].T
            target_output = F.gelu(target_pre) @ dense_c_proj[expert].T
            weight = probabilities[token, slot, None]
            predicted.index_add_(0, token, candidate_output * weight)
            target.index_add_(0, token, target_output * weight)
            expert_error[expert] += float(
                (candidate_output - target_output).square().sum()
            )
            expert_energy[expert] += float(target_output.square().sum())
            pre_error[expert] += float((candidate_pre - target_pre).square().sum())
            pre_energy[expert] += float(target_pre.square().sum())
        predicted_chunks.append(predicted.cpu())
        target_chunks.append(target.cpu())
    return (
        torch.cat(predicted_chunks),
        torch.cat(target_chunks),
        [
            1.0 - error / max(energy, 1e-30)
            for error, energy in zip(expert_error, expert_energy)
        ],
        [
            1.0 - error / max(energy, 1e-30)
            for error, energy in zip(pre_error, pre_energy)
        ],
    )


def collect_heldout_inputs(
    model: torch.nn.Module,
    plan: dict[str, Any],
    data_dir: Path,
    device: str,
) -> dict[str, dict[int, torch.Tensor]]:
    layers = [int(value) for value in plan["source"]["functional_probe_layers"]]
    result: dict[str, dict[int, torch.Tensor]] = {}
    for specification in plan["functional_protocol"]["heldout_banks"]:
        batches = fixed_validation_batches(
            data_dir,
            int(specification["batch_size"]),
            int(specification["block_size"]),
            int(specification["batches"]),
            int(specification["seed"]),
        )
        result[specification["name"]] = collect_inputs(
            model,
            batches,
            layers,
            int(specification["tokens"]),
            device,
        )
    return result


def validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("global c_fc TT plan schema mismatch")
    identity = plan["identity"]
    if identity.get("entrypoint_sha256") != file_sha256(Path(__file__)):
        raise ValueError("entrypoint hash is not sealed in the frozen plan")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in identity["helper_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"helper hash drift: {relative}")
    mechanism = plan["mechanism"]
    modes = [int(value) for value in mechanism["interleaved_modes"]]
    ranks = capped_bond_ranks(modes, int(mechanism["bond_rank_cap"]))
    if ranks != [int(value) for value in mechanism["actual_bond_ranks"]]:
        raise ValueError("interleaved TT bond accounting drift")
    if coordinate_count(modes, ranks) != int(mechanism["coordinates"]):
        raise ValueError("interleaved TT coordinate accounting drift")
    control = plan["equal_coordinate_control"]
    control_modes = [int(value) for value in control["modes"]]
    control_ranks = capped_bond_ranks(control_modes, int(control["bond_rank_cap"]))
    if control_ranks != [int(value) for value in control["actual_bond_ranks"]]:
        raise ValueError("control TT bond accounting drift")
    if coordinate_count(control_modes, control_ranks) != int(control["coordinates"]):
        raise ValueError("control TT coordinate accounting drift")
    if int(control["coordinates"]) != int(mechanism["coordinates"]):
        raise ValueError("candidate and control coordinate counts differ")
    if file_sha256(plan_path) == "":
        raise AssertionError("unreachable empty plan hash")


def result_authorization(passed: bool) -> dict[str, bool]:
    return {
        "implementation": bool(passed),
        "initialization_fit_shadow": bool(passed),
        "mfu_preflight": False,
        "language_model_training": False,
        "larger_rung": False,
        "generated_cproj": False,
    }


def run_preflight(plan: dict[str, Any], device: str) -> dict[str, Any]:
    source = plan["source"]
    mechanism = plan["mechanism"]
    hidden_modes = [int(value) for value in mechanism["hidden_modes"]]
    input_modes = [int(value) for value in mechanism["input_modes"]]
    modes = [int(value) for value in mechanism["interleaved_modes"]]
    ranks = [int(value) for value in mechanism["actual_bond_ranks"]]
    generator = torch.Generator(device=device)
    generator.manual_seed(20261000)
    started = time.time()
    dense = torch.randn(
        int(source["tensor_layers"]),
        int(source["num_experts"]),
        int(source["expert_hidden_width"]),
        int(source["input_width"]),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    physical = dense_to_physical(
        dense, hidden_modes, input_modes, interleaved=True
    )
    cores, diagnostics = randomized_tt_svd(
        physical,
        modes,
        ranks,
        seed=int(plan["decomposition_protocol"]["independent_seeds"][0]),
        oversample=int(plan["decomposition_protocol"]["oversample"]),
        power_iterations=int(plan["decomposition_protocol"]["power_iterations"]),
    )
    matrix = materialize_expert_matrix(
        cores, 0, 0, hidden_modes, input_modes, interleaved=True
    )
    return {
        "schema_version": "nanogpt_sparse_moe_cfc_global_tt_preflight_v1",
        "wall_seconds": time.time() - started,
        "device": device,
        "all_values_finite": bool(torch.isfinite(matrix).all()),
        "coordinates": sum(core.numel() for core in cores),
        "maximum_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else 0
        ),
        "split_diagnostics": diagnostics,
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
        parser.error("scientific oracle requires --terminal-snapshot, --data-dir, and --output")
    started = time.time()
    source = plan["source"]
    if file_sha256(args.terminal_snapshot) != source["terminal_manifold_snapshot_sha256"]:
        raise ValueError("terminal snapshot hash disagrees with frozen plan")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != source["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash disagrees with frozen plan")
    payload = load_terminal_snapshot(args.terminal_snapshot)
    if int(payload["next_iter"]) != int(source["next_iter"]):
        raise ValueError("terminal snapshot step disagrees with frozen plan")
    all_layers = list(range(int(source["tensor_layers"])))
    terminal_states = {
        layer: layer_state_from_mapping(payload["model"], layer)
        for layer in all_layers
    }
    dense = torch.stack(
        [terminal_states[layer].c_fc for layer in all_layers]
    ).to(device=args.device, dtype=torch.float32)

    model = load_model(args.terminal_snapshot, args.device)
    model.eval()
    heldout_inputs = collect_heldout_inputs(model, plan, args.data_dir, args.device)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    mechanism = plan["mechanism"]
    hidden_modes = [int(value) for value in mechanism["hidden_modes"]]
    input_modes = [int(value) for value in mechanism["input_modes"]]
    candidate_modes = [int(value) for value in mechanism["interleaved_modes"]]
    candidate_ranks = [int(value) for value in mechanism["actual_bond_ranks"]]
    control = plan["equal_coordinate_control"]
    control_modes = [int(value) for value in control["modes"]]
    control_ranks = [int(value) for value in control["actual_bond_ranks"]]
    protocol = plan["decomposition_protocol"]
    seeds = [int(value) for value in protocol["independent_seeds"]]
    probe_layers = [int(value) for value in source["functional_probe_layers"]]
    bank_names = [row["name"] for row in plan["functional_protocol"]["heldout_banks"]]
    physical_candidate = dense_to_physical(
        dense, hidden_modes, input_modes, interleaved=True
    )
    physical_control = dense_to_physical(
        dense, hidden_modes, input_modes, interleaved=False
    )
    decompositions: dict[str, dict[str, Any]] = {}
    stored_cores: dict[str, dict[str, list[torch.Tensor]]] = {}
    actions: dict[tuple[str, str, int], torch.Tensor] = {}

    for seed in seeds:
        seed_key = str(seed)
        decompositions[seed_key] = {}
        stored_cores[seed_key] = {}
        for name, physical, modes, ranks, interleaved in (
            (
                "interleaved",
                physical_candidate,
                candidate_modes,
                candidate_ranks,
                True,
            ),
            ("separated_control", physical_control, control_modes, control_ranks, False),
        ):
            cores, split_diagnostics = randomized_tt_svd(
                physical,
                modes,
                ranks,
                seed=seed,
                oversample=int(protocol["oversample"]),
                power_iterations=int(protocol["power_iterations"]),
            )
            stored_cores[seed_key][name] = [core.detach().cpu() for core in cores]
            parameter = parameter_recovery(
                dense,
                cores,
                hidden_modes,
                input_modes,
                interleaved=interleaved,
            )
            summaries: dict[str, Any] = {}
            for bank in bank_names:
                summaries[bank] = {}
                for layer in probe_layers:
                    candidate_c_fc = torch.stack(
                        [
                            materialize_expert_matrix(
                                cores,
                                layer,
                                expert,
                                hidden_modes,
                                input_modes,
                                interleaved=interleaved,
                            )
                            for expert in range(int(source["num_experts"]))
                        ]
                    )
                    predicted, target, expert_recovery, pre_recovery = routed_outputs(
                        terminal_states[layer],
                        heldout_inputs[bank][layer],
                        candidate_c_fc,
                        top_k=int(plan["functional_protocol"]["top_k"]),
                        chunk_size=int(plan["functional_protocol"]["chunk_size"]),
                        device=args.device,
                    )
                    summaries[bank][str(layer)] = {
                        "mixture_recovery": recovery_fraction(predicted, target),
                        "expert_recovery": expert_recovery,
                        "minimum_expert_recovery": min(expert_recovery),
                        "pregelu_recovery": pre_recovery,
                        "minimum_pregelu_recovery": min(pre_recovery),
                    }
                    if name == "interleaved":
                        actions[(seed_key, bank, layer)] = predicted
            decompositions[seed_key][name] = {
                "split_diagnostics": split_diagnostics,
                "parameter_recovery": parameter,
                "functional": summaries,
            }
            del cores
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    frozen = plan["frozen_gates"]
    gates: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        seed_key = str(seed)
        gates[seed_key] = {}
        for bank in bank_names:
            candidate_rows = [
                decompositions[seed_key]["interleaved"]["functional"][bank][str(layer)]
                for layer in probe_layers
            ]
            control_rows = [
                decompositions[seed_key]["separated_control"]["functional"][bank][str(layer)]
                for layer in probe_layers
            ]
            recoveries = [float(row["mixture_recovery"]) for row in candidate_rows]
            improvements = [
                float(candidate["mixture_recovery"])
                - float(control_row["mixture_recovery"])
                for candidate, control_row in zip(candidate_rows, control_rows)
            ]
            minimum_expert = min(
                float(row["minimum_expert_recovery"]) for row in candidate_rows
            )
            aggregate = {
                "mixture_recovery_mean": sum(recoveries) / len(recoveries),
                "mixture_recovery_minimum_layer": min(recoveries),
                "minimum_expert_recovery": minimum_expert,
                "interleaved_minus_separated_recovery_mean": (
                    sum(improvements) / len(improvements)
                ),
            }
            decompositions[seed_key]["interleaved"]["functional"][bank][
                "aggregate"
            ] = aggregate
            gates[seed_key][bank] = {
                "mean_recovery_pass": aggregate["mixture_recovery_mean"]
                >= float(
                    frozen["heldout_mixture_recovery_mean_min_each_seed_and_bank"]
                ),
                "every_layer_pass": aggregate["mixture_recovery_minimum_layer"]
                >= float(
                    frozen[
                        "heldout_mixture_recovery_every_layer_min_each_seed_and_bank"
                    ]
                ),
                "every_expert_pass": aggregate["minimum_expert_recovery"]
                >= float(frozen["heldout_expert_recovery_min_each_seed_and_bank"]),
                "interleaved_gain_pass": aggregate[
                    "interleaved_minus_separated_recovery_mean"
                ]
                >= float(
                    frozen[
                        "interleaved_minus_separated_recovery_mean_min_each_seed_and_bank"
                    ]
                ),
            }

    agreement_by_bank_layer: dict[str, dict[str, float]] = {}
    all_agreements: list[float] = []
    for bank in bank_names:
        agreement_by_bank_layer[bank] = {}
        for layer in probe_layers:
            value = action_cosine(
                actions[(str(seeds[0]), bank, layer)],
                actions[(str(seeds[1]), bank, layer)],
            )
            agreement_by_bank_layer[bank][str(layer)] = value
            all_agreements.append(value)
    agreement_mean = sum(all_agreements) / len(all_agreements)
    agreement_pass = agreement_mean >= float(
        frozen["same_bank_action_cosine_between_decomposition_seeds_mean_min"]
    )
    finite = all_finite(
        {
            "decompositions": decompositions,
            "agreement": agreement_by_bank_layer,
        }
    )
    for seed_key in gates:
        for bank in bank_names:
            gates[seed_key][bank]["action_agreement_pass"] = agreement_pass
            gates[seed_key][bank]["finite_pass"] = finite
            gates[seed_key][bank]["all_pass"] = all(
                gates[seed_key][bank].values()
            )
    passed = all(
        gates[str(seed)][bank]["all_pass"] for seed in seeds for bank in bank_names
    )

    args.output.mkdir(parents=True, exist_ok=False)
    coordinates_path = args.output / "tt_cores.pt"
    torch.save(
        {
            "schema_version": "nanogpt_sparse_moe_cfc_global_tt_coordinates_v1",
            "cores": stored_cores,
            "candidate_modes": candidate_modes,
            "control_modes": control_modes,
        },
        coordinates_path,
    )
    result = {
        "schema_version": "nanogpt_sparse_moe_cfc_global_tt_oracle_result_v1",
        "classification": (
            "GLOBAL_TT_CFC_REPRESENTABILITY_PASSES"
            if passed
            else "GLOBAL_TT_CFC_REPRESENTABILITY_REJECTED"
        ),
        "passed": passed,
        "identity": {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "plan_sha256": file_sha256(args.plan),
            "entrypoint_sha256": file_sha256(Path(__file__)),
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
                if args.device.startswith("cuda")
                else 0
            ),
        },
        "accounting": {
            "dense_cfc_parameters": int(mechanism["dense_cfc_parameters"]),
            "coordinates": int(mechanism["coordinates"]),
            "cfc_compression_ratio": float(mechanism["cfc_compression_ratio"]),
            "materialized_dense_cfc_in_candidate": False,
            "dense_cproj_retained_as_exception": True,
        },
        "decompositions": decompositions,
        "same_bank_action_cosine_between_decomposition_seeds": {
            "mean": agreement_mean,
            "by_bank_and_layer": agreement_by_bank_layer,
        },
        "gates": gates,
        "all_values_finite": finite,
        "authorization": result_authorization(passed),
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
