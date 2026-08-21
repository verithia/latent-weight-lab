#!/usr/bin/env python3
"""Capacity oracle for atomwise recombination of two late-depth MLP banks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_layer_axis_basis7_mlp_teacher_fit import (
    ASSIGNMENT,
    BOUNDARIES,
    ROOTS,
    fit_stage,
    full_objective,
)
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_shared_mlp_endpoint_function import (
    sha256_file,
    tensor_sha256,
    validate_core_configs,
)
from examples.nanogpt.analyze_shared_mlp_exact_family_teacher_fit import (
    atomic_json,
    collect_stratified_inputs,
    git_head,
)
from examples.nanogpt.analyze_shared_trunk_private_ridge_teacher_fit import (
    build_data,
    jvp_metrics,
    output_metrics,
    passes,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_two_bank_atomwise_recombination_plan_v1"
RESULT_SCHEMA = "mai_two_bank_atomwise_recombination_result_v1"
SEVEN_TRUNK_RESULT = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_shared_dense_mlp_trunk_groups7_1_1_1_1_1_3_4_5tpp_result.json"
)
LATE_LAYERS = tuple(range(5, 12))


def semantic_signatures(c_fc: Tensor, c_proj: Tensor) -> Tensor:
    """Equal-weight complete-neuron signatures for function-preserving matching."""

    fc = F.normalize(c_fc.float(), dim=1)
    proj = F.normalize(c_proj.float().T, dim=1)
    return torch.cat((fc, proj), dim=1) / math.sqrt(2.0)


@torch.no_grad()
def deterministic_pairing(
    a_fc: Tensor,
    a_proj: Tensor,
    b_fc: Tensor,
    b_proj: Tensor,
    *,
    top_k: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Greedily choose a deterministic one-to-one A-to-B atom alignment."""

    a = semantic_signatures(a_fc, a_proj)
    b = semantic_signatures(b_fc, b_proj)
    if a.shape != b.shape:
        raise ValueError("late-bank semantic signature shapes differ")
    hidden = a.shape[0]
    k = min(int(top_k), hidden)
    similarity = a @ b.T
    scores, candidates = torch.topk(
        similarity, k=k, dim=1, largest=True, sorted=True
    )
    flat_scores = scores.flatten().cpu()
    flat_a = torch.arange(hidden).repeat_interleave(k)
    flat_b = candidates.cpu().flatten()
    order = torch.argsort(flat_scores, descending=True, stable=True)
    permutation = torch.full((hidden,), -1, dtype=torch.long)
    used = torch.zeros(hidden, dtype=torch.bool)
    for edge in order.tolist():
        left = int(flat_a[edge])
        right = int(flat_b[edge])
        if permutation[left] < 0 and not used[right]:
            permutation[left] = right
            used[right] = True
    remaining_right = (~used).nonzero(as_tuple=False).flatten()
    for left in (permutation < 0).nonzero(as_tuple=False).flatten().tolist():
        row = similarity[left].index_select(0, remaining_right.to(similarity.device))
        local = int(torch.argmax(row))
        permutation[left] = remaining_right[local]
        remaining_right = torch.cat(
            (remaining_right[:local], remaining_right[local + 1 :])
        )
    if torch.unique(permutation).numel() != hidden:
        raise RuntimeError("atom matching did not produce a permutation")
    permutation_device = permutation.to(similarity.device)
    selected = similarity[
        torch.arange(hidden, device=similarity.device), permutation_device
    ].float().cpu()
    identity = similarity.diag().float().cpu()
    digest = hashlib.sha256(
        permutation.numpy().astype("<i8", copy=False).tobytes()
    ).hexdigest()
    diagnostics = {
        "algorithm": "global_score_greedy_topk_then_exact_unmatched_completion",
        "top_k": k,
        "permutation_sha256": digest,
        "matched_similarity_mean": float(selected.mean()),
        "matched_similarity_minimum": float(selected.min()),
        "matched_similarity_maximum": float(selected.max()),
        "identity_similarity_mean": float(identity.mean()),
        "identity_similarity_minimum": float(identity.min()),
        "fixed_point_fraction": float(
            (permutation == torch.arange(hidden)).float().mean()
        ),
    }
    return permutation.to(a_fc.device), diagnostics


class TwoBankAtomwiseMLP(nn.Module):
    """Five singleton banks plus two atomwise-recombined late banks."""

    def __init__(
        self,
        *,
        basis_fc: Tensor,
        basis_proj: Tensor,
        alpha_fc: Tensor,
        alpha_proj: Tensor,
        pre_gain: Tensor,
        output_log_gain: Tensor,
        pairing: Tensor,
    ) -> None:
        super().__init__()
        self.basis_fc = nn.Parameter(basis_fc.detach().float().clone())
        self.basis_proj = nn.Parameter(basis_proj.detach().float().clone())
        self.alpha_fc = nn.Parameter(alpha_fc.detach().float().clone())
        self.alpha_proj = nn.Parameter(alpha_proj.detach().float().clone())
        self.pre_gain = nn.Parameter(pre_gain.detach().float().clone())
        self.output_log_gain = nn.Parameter(output_log_gain.detach().float().clone())
        self.register_buffer("pairing", pairing.detach().long().clone())
        banks, hidden, width = self.basis_fc.shape
        if banks != 7 or self.basis_proj.shape != (banks, width, hidden):
            raise ValueError("expected seven paired full-width banks")
        if self.alpha_fc.shape != (len(LATE_LAYERS), hidden):
            raise ValueError("late c_fc coefficient shape mismatch")
        if self.alpha_proj.shape != self.alpha_fc.shape:
            raise ValueError("late c_proj coefficient shape mismatch")
        if self.pre_gain.shape != (12, hidden):
            raise ValueError("pre-GELU gain shape mismatch")
        if self.output_log_gain.shape != (12, width):
            raise ValueError("residual-output gain shape mismatch")
        if torch.unique(self.pairing).numel() != hidden:
            raise ValueError("pairing must be a complete permutation")

    @property
    def layers(self) -> int:
        return 12

    def set_trainable(self, *, coefficients_only: bool) -> list[nn.Parameter]:
        for parameter in self.parameters():
            parameter.requires_grad_(not coefficients_only)
        self.alpha_fc.requires_grad_(True)
        self.alpha_proj.requires_grad_(True)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def weights(self, layer: int) -> tuple[Tensor, Tensor]:
        layer = int(layer)
        if layer < 5:
            return self.basis_fc[layer], self.basis_proj[layer]
        offset = layer - 5
        a_fc = self.alpha_fc[offset][:, None]
        a_proj = self.alpha_proj[offset][None, :]
        c_fc = a_fc * self.basis_fc[5] + (1.0 - a_fc) * self.basis_fc[6]
        c_proj = (
            a_proj * self.basis_proj[5]
            + (1.0 - a_proj) * self.basis_proj[6]
        )
        return c_fc, c_proj

    def forward_layer(self, layer: int, values: Tensor) -> Tensor:
        layer = int(layer)
        c_fc, c_proj = self.weights(layer)
        hidden = F.gelu(F.linear(values, c_fc) * self.pre_gain[layer])
        output = F.linear(hidden, c_proj)
        return output * self.output_log_gain[layer].exp()

    def forward_selected(self, values: Tensor, layer_indices: Tensor) -> Tensor:
        if values.ndim != 5 or values.shape[1] != layer_indices.numel():
            raise ValueError("unexpected selected-layer input shape")
        return torch.stack(
            [
                self.forward_layer(int(layer), values[:, position])
                for position, layer in enumerate(layer_indices)
            ],
            dim=1,
        )

    @torch.no_grad()
    def clamp_charts(self) -> None:
        self.alpha_fc.clamp_(-1.0, 2.0)
        self.alpha_proj.clamp_(-1.0, 2.0)
        self.pre_gain.clamp_(-4.0, 4.0)
        self.output_log_gain.clamp_(-4.0, 4.0)


class InstalledTwoBankAtomwiseMLP(nn.Module):
    def __init__(self, family: TwoBankAtomwiseMLP, layer: int) -> None:
        super().__init__()
        self.family = family
        self.layer = int(layer)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    def forward(self, values: Tensor) -> Tensor:
        return self.family.forward_layer(self.layer, values)


@torch.no_grad()
def initial_family(
    checkpoint: nn.Module, *, top_k: int
) -> tuple[TwoBankAtomwiseMLP, dict[str, Any]]:
    config = checkpoint.config
    boundaries = tuple(int(value) for value in config.mlp_shared_dense_trunk_boundaries)
    if not config.mlp_shared_dense_trunk:
        raise ValueError("initialization checkpoint is not a shared trunk model")
    if int(config.mlp_shared_dense_trunk_groups) != 7 or boundaries != BOUNDARIES:
        raise ValueError("unexpected seven-trunk depth partition")
    blocks = list(checkpoint.transformer.h)
    basis_fc = torch.stack([blocks[layer].mlp.c_fc.weight for layer in ROOTS])
    basis_proj = torch.stack([blocks[layer].mlp.c_proj.weight for layer in ROOTS])
    pairing, diagnostics = deterministic_pairing(
        basis_fc[5], basis_proj[5], basis_fc[6], basis_proj[6], top_k=top_k
    )
    basis_fc[6] = basis_fc[6].index_select(0, pairing)
    basis_proj[6] = basis_proj[6].index_select(1, pairing)
    pre_gain = torch.stack([block.mlp.pregelu_gain for block in blocks])
    for layer in range(8, 12):
        pre_gain[layer] = pre_gain[layer].index_select(0, pairing)
    output_log_gain = torch.stack(
        [
            block.mlp.residual_output_log_gain
            * block.mlp.residual_output_gain_scale
            for block in blocks
        ]
    )
    alpha = torch.zeros(
        len(LATE_LAYERS), basis_fc.shape[1], device=basis_fc.device
    )
    alpha[:3].fill_(1.0)
    family = TwoBankAtomwiseMLP(
        basis_fc=basis_fc,
        basis_proj=basis_proj,
        alpha_fc=alpha,
        alpha_proj=alpha,
        pre_gain=pre_gain,
        output_log_gain=output_log_gain,
        pairing=pairing,
    ).to(basis_fc.device)
    return family, diagnostics


def family_from_state(state: dict[str, Tensor], device: str) -> TwoBankAtomwiseMLP:
    return TwoBankAtomwiseMLP(
        basis_fc=state["basis_fc"],
        basis_proj=state["basis_proj"],
        alpha_fc=state["alpha_fc"],
        alpha_proj=state["alpha_proj"],
        pre_gain=state["pre_gain"],
        output_log_gain=state["output_log_gain"],
        pairing=state["pairing"],
    ).to(device)


@torch.no_grad()
def endpoint_error(module: TwoBankAtomwiseMLP, checkpoint: nn.Module) -> float:
    maximum = 0.0
    generator = torch.Generator(device=module.basis_fc.device).manual_seed(20260974)
    for layer in range(module.layers):
        values = torch.randn(
            32,
            module.basis_fc.shape[-1],
            generator=generator,
            device=module.basis_fc.device,
        )
        expected = checkpoint.transformer.h[layer].mlp(values)
        actual = module.forward_layer(layer, values)
        maximum = max(maximum, float((actual - expected).abs().max()))
    return maximum


@torch.no_grad()
def coefficient_diagnostics(module: TwoBankAtomwiseMLP) -> dict[str, Any]:
    result: dict[str, Any] = {}
    endpoint = torch.cat(
        (
            torch.ones(3, module.alpha_fc.shape[1]),
            torch.zeros(4, module.alpha_fc.shape[1]),
        ),
        dim=0,
    ).to(module.alpha_fc.device)
    for name, coefficients in (
        ("c_fc", module.alpha_fc),
        ("c_proj", module.alpha_proj),
    ):
        singular = torch.linalg.svdvals(coefficients.float()).cpu()
        result[name] = {
            "tensor_sha256": tensor_sha256(coefficients),
            "mean_by_layer": coefficients.float().mean(dim=1).cpu().tolist(),
            "standard_deviation_by_layer": coefficients.float().std(dim=1).cpu().tolist(),
            "mean_absolute_endpoint_displacement_by_layer": (
                coefficients.float() - endpoint
            ).abs().mean(dim=1).cpu().tolist(),
            "minimum": float(coefficients.min()),
            "maximum": float(coefficients.max()),
            "singular_values": singular.tolist(),
        }
    return result


def function_measurement(
    module: TwoBankAtomwiseMLP,
    *,
    holdout: dict[str, Tensor],
    teacher: nn.Module,
    measurement: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    output_summary, output_rows = output_metrics(module, holdout)
    jvp_summary, jvp_rows = jvp_metrics(
        module,
        holdout,
        teacher=teacher,
        directions=int(measurement["input_jvp_directions"]),
        seed=int(measurement["input_jvp_seed"]),
        device=device,
    )
    return {
        "summary": {"output": output_summary, "input_jvp": jvp_summary},
        "holdout_output_rows": output_rows,
        "holdout_input_jvp_rows": jvp_rows,
        "coefficient_diagnostics": coefficient_diagnostics(module),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=10)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA or args.device != "cuda":
        raise ValueError("unexpected plan schema or device")
    causal = plan["causal_basis"]
    for key in (
        "seven_trunk_result",
        "layer_axis_result",
        "selected_atom_result",
        "residual_spectrum_result",
    ):
        if sha256_file(Path(causal[key])) != causal[f"{key}_sha256"]:
            raise ValueError(f"causal result identity mismatch: {key}")
    identity = plan["identities"]
    paths = {
        key: Path(identity[key]["path"])
        for key in (
            "dense_teacher_checkpoint",
            "seven_trunk_initialization_checkpoint",
            "state_bank_checkpoint",
        )
    }
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    for key, path in paths.items():
        if sha256_file(path) != identity[key]["sha256"]:
            raise ValueError(f"checkpoint identity mismatch: {key}")
    if sha256_file(data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")
    protocol, measurement = plan["fit_protocol"], plan["measurement"]
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    teacher = load_model(paths["dense_teacher_checkpoint"], args.device)
    initializer = load_model(paths["seven_trunk_initialization_checkpoint"], args.device)
    candidate = load_model(paths["state_bank_checkpoint"], args.device)
    validate_core_configs(initializer, teacher)
    validate_core_configs(candidate, teacher)

    def collect(seed: int, count: int) -> dict[str, dict[int, Tensor]]:
        batches = fixed_validation_batches(
            data_dir,
            int(protocol["token_batch_size"]),
            teacher.config.block_size,
            int(count),
            int(seed),
        )
        return {
            "teacher": collect_stratified_inputs(
                teacher,
                batches,
                sample_cap=int(protocol["sample_cap_per_layer_per_bank"]),
                seed=int(seed),
                device=args.device,
            ),
            "candidate": collect_stratified_inputs(
                candidate,
                batches,
                sample_cap=int(protocol["sample_cap_per_layer_per_bank"]),
                seed=int(seed),
                device=args.device,
            ),
        }

    print("collecting fit and holdout banks", flush=True)
    fit = build_data(
        banks=collect(int(protocol["fit_token_seed"]), int(protocol["fit_batches"])),
        teacher=teacher,
        relative_rms=float(protocol["local_perturbation_relative_rms"]),
        seed=int(protocol["fit_token_seed"]),
        device=args.device,
    )
    holdout = build_data(
        banks=collect(
            int(protocol["holdout_token_seed"]), int(protocol["holdout_batches"])
        ),
        teacher=teacher,
        relative_rms=float(protocol["local_perturbation_relative_rms"]),
        seed=int(protocol["holdout_token_seed"]),
        device=args.device,
    )
    module, pairing_diagnostics = initial_family(initializer, top_k=64)
    initial_endpoint_error = endpoint_error(module, initializer)
    if initial_endpoint_error > float(
        plan["frozen_gates"]["maximum_initial_endpoint_absolute_error"]
    ):
        raise ValueError("aligned atomwise family does not preserve the endpoint")
    if sum(parameter.numel() for parameter in module.parameters()) != int(
        plan["family"]["compact_parameters"]
    ):
        raise ValueError("compact parameter accounting mismatch")

    if args.preflight_only:
        stage = protocol["stages"][-1]
        row, _ = fit_stage(
            module,
            fit=fit,
            holdout=holdout,
            stage=stage,
            rows_per_layer=int(protocol["row_batch_size_per_layer_per_bank"]),
            layers_per_update=int(protocol["layers_per_update"]),
            gradient_clip_norm=float(protocol["gradient_clip_norm"]),
            seed=int(protocol["algorithm_seed"]),
            steps_override=int(args.preflight_steps),
        )
        seconds = row["wall_seconds"] / max(int(args.preflight_steps), 1)
        print(
            json.dumps(
                {
                    "preflight": "complete",
                    "seconds_per_fit_step_including_preflight_evaluations": seconds,
                    "conservative_estimated_fit_seconds": seconds
                    * sum(int(stage["steps"]) for stage in protocol["stages"]),
                    "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "initial_endpoint_maximum_absolute_error": initial_endpoint_error,
                    "pairing": pairing_diagnostics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    stage_results: dict[str, dict[str, Any]] = {}
    stage_states: dict[str, dict[str, Tensor]] = {}
    for stage_index, stage in enumerate(protocol["stages"]):
        row, state = fit_stage(
            module,
            fit=fit,
            holdout=holdout,
            stage=stage,
            rows_per_layer=int(protocol["row_batch_size_per_layer_per_bank"]),
            layers_per_update=int(protocol["layers_per_update"]),
            gradient_clip_norm=float(protocol["gradient_clip_norm"]),
            seed=int(protocol["algorithm_seed"]) + stage_index * 1000,
        )
        row.update(
            function_measurement(
                module,
                holdout=holdout,
                teacher=teacher,
                measurement=measurement,
                device=args.device,
            )
        )
        stage_results[stage["name"]] = row
        stage_states[stage["name"]] = state

    fixed = make_fixed_eval_indices(
        data_dir,
        int(measurement["fixed_eval_batch_size"]),
        int(measurement["fixed_eval_block_size"]),
        int(measurement["fixed_eval_batches"]),
        int(measurement["fixed_eval_seed"]),
    )
    digest = fixed_eval_indices_digest(fixed)
    if digest != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation digest mismatch")
    source = TokenBatchSource(data_dir)
    teacher_ce = evaluate_fixed_ce(
        teacher,
        data_dir=data_dir,
        fixed_indices=fixed,
        split="val",
        eval_iters=int(measurement["fixed_eval_batches"]),
        eval_batch_size=int(measurement["fixed_eval_batch_size"]),
        block_size=int(measurement["fixed_eval_block_size"]),
        device=args.device,
        dtype="bfloat16",
        source=source,
    )
    terminal_name = protocol["stages"][-1]["name"]
    terminal_state = stage_states[terminal_name]
    terminal_family = family_from_state(terminal_state, args.device)
    splice = load_model(paths["dense_teacher_checkpoint"], args.device)
    for layer in range(terminal_family.layers):
        splice.transformer.h[layer].mlp = InstalledTwoBankAtomwiseMLP(
            terminal_family, layer
        )
    candidate_ce = evaluate_fixed_ce(
        splice,
        data_dir=data_dir,
        fixed_indices=fixed,
        split="val",
        eval_iters=int(measurement["fixed_eval_batches"]),
        eval_batch_size=int(measurement["fixed_eval_batch_size"]),
        block_size=int(measurement["fixed_eval_block_size"]),
        device=args.device,
        dtype="bfloat16",
        source=source,
    )
    terminal = stage_results[terminal_name]
    reduction = terminal["objective_reduction_fraction"]
    healthy = bool(
        terminal["finite"]
        and (
            terminal["final_fit_objective"]["total"] <= 0.5
            or (reduction is not None and reduction >= 0.5)
        )
    )
    terminal.update(
        {
            "fixed_validation_cross_entropy": candidate_ce,
            "gap": candidate_ce - teacher_ce,
            "optimization_healthy": healthy,
        }
    )
    terminal["passes"] = passes(
        terminal, candidate_ce - teacher_ce, plan["frozen_gates"]
    )
    if terminal["passes"]:
        classification = "TWO_BANK_ATOMWISE_RECOMBINATION_PASS"
    elif healthy:
        classification = "TWO_BANK_ATOMWISE_RECOMBINATION_FAIL"
    else:
        classification = "OPTIMIZATION_INCONCLUSIVE"
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "fitted_states.pt"
    torch.save(
        {
            "schema_version": "mai_two_bank_atomwise_recombination_state_v1",
            "stages": stage_states,
        },
        state_path,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identity,
        "pairing": pairing_diagnostics,
        "initial_endpoint_maximum_absolute_error": initial_endpoint_error,
        "compact_parameter_count": sum(
            parameter.numel() for parameter in module.parameters()
        ),
        "stage_results": stage_results,
        "teacher_validation_cross_entropy": teacher_ce,
        "fixed_eval_indices_sha256": digest,
        "state_artifact": {"path": str(state_path), "sha256": sha256_file(state_path)},
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.time() - started,
    }
    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "classification": classification,
                "terminal": {
                    "fit": terminal["final_fit_objective"],
                    "holdout": terminal["holdout_objective"],
                    "reduction": reduction,
                    "healthy": healthy,
                    "summary": terminal["summary"],
                    "teacher_ce": teacher_ce,
                    "candidate_ce": candidate_ce,
                    "ce_gap": candidate_ce - teacher_ce,
                    "passes": terminal["passes"],
                },
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
