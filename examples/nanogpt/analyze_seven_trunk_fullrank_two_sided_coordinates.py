#!/usr/bin/env python3
"""Capacity ceiling for full-rank late-layer residual-coordinate maps."""
from __future__ import annotations

import argparse
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
    BOUNDARIES,
    fit_stage,
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


PLAN_SCHEMA = "mai_seven_trunk_fullrank_two_sided_coordinates_plan_v1"
RESULT_SCHEMA = "mai_seven_trunk_fullrank_two_sided_coordinates_result_v1"
ROOTS = (0, 1, 2, 3, 4, 5, 8)
LATE_ASSIGNMENT = (0, 0, 0, 1, 1, 1, 1)


class FullRankTwoSidedMLP(nn.Module):
    """Frozen early singletons and two late trunks with full residual maps."""

    def __init__(
        self,
        *,
        singleton_fc: Tensor,
        singleton_proj: Tensor,
        late_fc: Tensor,
        late_proj: Tensor,
        input_maps: Tensor,
        output_maps: Tensor,
        singleton_pre_gain: Tensor,
        late_pre_gain: Tensor,
        singleton_output_log_gain: Tensor,
        late_output_log_gain: Tensor,
    ) -> None:
        super().__init__()
        self.singleton_fc = nn.Parameter(singleton_fc.detach().float().clone())
        self.singleton_proj = nn.Parameter(singleton_proj.detach().float().clone())
        self.late_fc = nn.Parameter(late_fc.detach().float().clone())
        self.late_proj = nn.Parameter(late_proj.detach().float().clone())
        self.input_maps = nn.Parameter(input_maps.detach().float().clone())
        self.output_maps = nn.Parameter(output_maps.detach().float().clone())
        self.singleton_pre_gain = nn.Parameter(
            singleton_pre_gain.detach().float().clone()
        )
        self.late_pre_gain = nn.Parameter(late_pre_gain.detach().float().clone())
        self.singleton_output_log_gain = nn.Parameter(
            singleton_output_log_gain.detach().float().clone()
        )
        self.late_output_log_gain = nn.Parameter(
            late_output_log_gain.detach().float().clone()
        )
        if self.singleton_fc.shape[0] != 5 or self.late_fc.shape[0] != 2:
            raise ValueError("unexpected singleton or late trunk count")
        hidden, width = self.singleton_fc.shape[1:]
        if self.singleton_proj.shape != (5, width, hidden):
            raise ValueError("singleton matrix pairs do not match")
        if self.late_fc.shape != (2, hidden, width):
            raise ValueError("late c_fc shape mismatch")
        if self.late_proj.shape != (2, width, hidden):
            raise ValueError("late matrix pairs do not match")
        if self.input_maps.shape != (7, width, width):
            raise ValueError("late input-map shape mismatch")
        if self.output_maps.shape != self.input_maps.shape:
            raise ValueError("late output-map shape mismatch")
        if self.singleton_pre_gain.shape != (5, hidden):
            raise ValueError("singleton pre-GELU gain shape mismatch")
        if self.late_pre_gain.shape != (7, hidden):
            raise ValueError("late pre-GELU gain shape mismatch")
        if self.singleton_output_log_gain.shape != (5, width):
            raise ValueError("singleton output gain shape mismatch")
        if self.late_output_log_gain.shape != (7, width):
            raise ValueError("late output gain shape mismatch")

    @property
    def layers(self) -> int:
        return 12

    def set_trainable(self, *, coefficients_only: bool) -> list[nn.Parameter]:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.input_maps.requires_grad_(True)
        self.output_maps.requires_grad_(True)
        if not coefficients_only:
            self.late_fc.requires_grad_(True)
            self.late_proj.requires_grad_(True)
            self.late_pre_gain.requires_grad_(True)
            self.late_output_log_gain.requires_grad_(True)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def weights(self, layer: int) -> tuple[Tensor, Tensor]:
        layer = int(layer)
        if layer < 5:
            zero = (self.input_maps.sum() + self.output_maps.sum()) * 0.0
            return self.singleton_fc[layer] + zero, self.singleton_proj[layer] + zero
        offset = layer - 5
        group = LATE_ASSIGNMENT[offset]
        c_fc = self.late_fc[group] @ self.input_maps[offset]
        c_proj = self.output_maps[offset] @ self.late_proj[group]
        return c_fc, c_proj

    def gains(self, layer: int) -> tuple[Tensor, Tensor]:
        layer = int(layer)
        if layer < 5:
            return (
                self.singleton_pre_gain[layer],
                self.singleton_output_log_gain[layer],
            )
        offset = layer - 5
        return self.late_pre_gain[offset], self.late_output_log_gain[offset]

    def forward_layer(self, layer: int, values: Tensor) -> Tensor:
        c_fc, c_proj = self.weights(layer)
        pre_gain, output_log_gain = self.gains(layer)
        hidden = F.gelu(F.linear(values, c_fc) * pre_gain)
        output = F.linear(hidden, c_proj)
        return output * output_log_gain.exp()

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
        self.late_pre_gain.clamp_(-4.0, 4.0)
        self.late_output_log_gain.clamp_(-4.0, 4.0)


class InstalledFullRankTwoSidedMLP(nn.Module):
    def __init__(self, family: FullRankTwoSidedMLP, layer: int) -> None:
        super().__init__()
        self.family = family
        self.layer = int(layer)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    def forward(self, values: Tensor) -> Tensor:
        return self.family.forward_layer(self.layer, values)


@torch.no_grad()
def initial_family(checkpoint: nn.Module) -> FullRankTwoSidedMLP:
    config = checkpoint.config
    boundaries = tuple(int(value) for value in config.mlp_shared_dense_trunk_boundaries)
    if not config.mlp_shared_dense_trunk:
        raise ValueError("initialization checkpoint is not a shared trunk model")
    if int(config.mlp_shared_dense_trunk_groups) != 7 or boundaries != BOUNDARIES:
        raise ValueError("unexpected seven-trunk depth partition")
    blocks = list(checkpoint.transformer.h)
    singleton_fc = torch.stack([blocks[layer].mlp.c_fc.weight for layer in ROOTS[:5]])
    singleton_proj = torch.stack(
        [blocks[layer].mlp.c_proj.weight for layer in ROOTS[:5]]
    )
    late_fc = torch.stack([blocks[layer].mlp.c_fc.weight for layer in ROOTS[5:]])
    late_proj = torch.stack(
        [blocks[layer].mlp.c_proj.weight for layer in ROOTS[5:]]
    )
    width = singleton_fc.shape[-1]
    identity = torch.eye(width, device=singleton_fc.device, dtype=singleton_fc.dtype)
    input_maps = identity.repeat(7, 1, 1)
    output_maps = identity.repeat(7, 1, 1)
    pre_gain = torch.stack([block.mlp.pregelu_gain for block in blocks])
    output_log_gain = torch.stack(
        [
            block.mlp.residual_output_log_gain
            * block.mlp.residual_output_gain_scale
            for block in blocks
        ]
    )
    return FullRankTwoSidedMLP(
        singleton_fc=singleton_fc,
        singleton_proj=singleton_proj,
        late_fc=late_fc,
        late_proj=late_proj,
        input_maps=input_maps,
        output_maps=output_maps,
        singleton_pre_gain=pre_gain[:5],
        late_pre_gain=pre_gain[5:],
        singleton_output_log_gain=output_log_gain[:5],
        late_output_log_gain=output_log_gain[5:],
    ).to(singleton_fc.device)


def family_from_state(state: dict[str, Tensor], device: str) -> FullRankTwoSidedMLP:
    return FullRankTwoSidedMLP(
        singleton_fc=state["singleton_fc"],
        singleton_proj=state["singleton_proj"],
        late_fc=state["late_fc"],
        late_proj=state["late_proj"],
        input_maps=state["input_maps"],
        output_maps=state["output_maps"],
        singleton_pre_gain=state["singleton_pre_gain"],
        late_pre_gain=state["late_pre_gain"],
        singleton_output_log_gain=state["singleton_output_log_gain"],
        late_output_log_gain=state["late_output_log_gain"],
    ).to(device)


@torch.no_grad()
def endpoint_error(module: FullRankTwoSidedMLP, checkpoint: nn.Module) -> float:
    maximum = 0.0
    generator = torch.Generator(device=module.singleton_fc.device).manual_seed(20260984)
    for layer in range(module.layers):
        values = torch.randn(
            32,
            module.singleton_fc.shape[-1],
            generator=generator,
            device=module.singleton_fc.device,
        )
        expected = checkpoint.transformer.h[layer].mlp(values)
        actual = module.forward_layer(layer, values)
        maximum = max(maximum, float((actual - expected).abs().max()))
    return maximum


@torch.no_grad()
def map_diagnostics(module: FullRankTwoSidedMLP) -> dict[str, Any]:
    ranks = (8, 16, 32, 64, 128, 256)
    identity = torch.eye(
        module.input_maps.shape[-1], device=module.input_maps.device
    )
    result: dict[str, Any] = {}
    for name, maps in (("input", module.input_maps), ("output", module.output_maps)):
        rows = []
        for offset, matrix in enumerate(maps):
            residual = matrix.float() - identity
            singular = torch.linalg.svdvals(residual)
            energy = singular.square()
            total = energy.sum().clamp_min(1e-30)
            probability = energy / total
            entropy_rank = torch.exp(
                -(probability * probability.clamp_min(1e-30).log()).sum()
            )
            rows.append(
                {
                    "layer": offset + 5,
                    "residual_frobenius_norm": float(residual.norm()),
                    "entropy_effective_rank": float(entropy_rank),
                    "rank_energy_recovery": {
                        str(rank): float(energy[:rank].sum() / total) for rank in ranks
                    },
                    "leading_singular_values": singular[:16].cpu().tolist(),
                }
            )
        result[name] = {
            "tensor_sha256": tensor_sha256(maps),
            "layers": rows,
        }
    return result


def function_measurement(
    module: FullRankTwoSidedMLP,
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
        "map_diagnostics": map_diagnostics(module),
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
        "two_bank_atomwise_result",
        "private_residual_spectrum_result",
        "postbranch_result",
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
    module = initial_family(initializer)
    initial_endpoint_error = endpoint_error(module, initializer)
    if initial_endpoint_error > float(
        plan["frozen_gates"]["maximum_initial_endpoint_absolute_error"]
    ):
        raise ValueError("full-rank coordinate family does not preserve the endpoint")
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
        splice.transformer.h[layer].mlp = InstalledFullRankTwoSidedMLP(
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
        classification = "FULLRANK_TWO_SIDED_COORDINATES_PASS"
    elif healthy:
        classification = "FULLRANK_TWO_SIDED_COORDINATES_FAIL"
    else:
        classification = "OPTIMIZATION_INCONCLUSIVE"
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "fitted_states.pt"
    torch.save(
        {
            "schema_version": "mai_seven_trunk_fullrank_two_sided_coordinates_state_v1",
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
