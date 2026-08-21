#!/usr/bin/env python3
"""Teacher-fit a continuous seven-atom layer-axis MLP basis."""
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
from examples.nanogpt.analyze_depth_shared_top1_mlp_teacher_fit import (
    objective_parts,
)
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_shared_mlp_endpoint_function import (
    sha256_file,
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


PLAN_SCHEMA = "mai_layer_axis_basis7_mlp_teacher_fit_plan_v1"
RESULT_SCHEMA = "mai_layer_axis_basis7_mlp_teacher_fit_result_v1"
HARD_EM_RESULT = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_depth_shared_top1_mlp_hard_em_result.json"
)
ROOTS = (0, 1, 2, 3, 4, 5, 8)
ASSIGNMENT = (0, 1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6)
BOUNDARIES = (1, 2, 3, 4, 5, 8, 12)


class LayerAxisBasisMLP(nn.Module):
    """Static per-layer mixtures of full-width learned matrix-pair atoms."""

    def __init__(
        self,
        *,
        basis_fc: Tensor,
        basis_proj: Tensor,
        coefficients_fc: Tensor,
        coefficients_proj: Tensor,
        pre_gain: Tensor,
        output_log_gain: Tensor,
    ) -> None:
        super().__init__()
        self.basis_fc = nn.Parameter(basis_fc.detach().float().clone())
        self.basis_proj = nn.Parameter(basis_proj.detach().float().clone())
        self.coefficients_fc = nn.Parameter(
            coefficients_fc.detach().float().clone()
        )
        self.coefficients_proj = nn.Parameter(
            coefficients_proj.detach().float().clone()
        )
        self.pre_gain = nn.Parameter(pre_gain.detach().float().clone())
        self.output_log_gain = nn.Parameter(
            output_log_gain.detach().float().clone()
        )
        atoms, hidden, width = self.basis_fc.shape
        layers = self.pre_gain.shape[0]
        if self.basis_proj.shape != (atoms, width, hidden):
            raise ValueError("basis matrix pairs have incompatible shapes")
        if self.coefficients_fc.shape != (layers, atoms):
            raise ValueError("c_fc coefficient shape mismatch")
        if self.coefficients_proj.shape != (layers, atoms):
            raise ValueError("c_proj coefficient shape mismatch")
        if self.pre_gain.shape != (layers, hidden):
            raise ValueError("pre-GELU gain shape mismatch")
        if self.output_log_gain.shape != (layers, width):
            raise ValueError("residual-output gain shape mismatch")

    @property
    def layers(self) -> int:
        return int(self.pre_gain.shape[0])

    @property
    def atoms(self) -> int:
        return int(self.basis_fc.shape[0])

    def set_trainable(self, *, coefficients_only: bool) -> list[nn.Parameter]:
        for parameter in self.parameters():
            parameter.requires_grad_(not coefficients_only)
        self.coefficients_fc.requires_grad_(True)
        self.coefficients_proj.requires_grad_(True)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def weights(self, layer: int) -> tuple[Tensor, Tensor]:
        layer = int(layer)
        c_fc = torch.einsum(
            "k,khd->hd", self.coefficients_fc[layer], self.basis_fc
        )
        c_proj = torch.einsum(
            "k,kdh->dh", self.coefficients_proj[layer], self.basis_proj
        )
        return c_fc, c_proj

    def forward_layer(self, layer: int, values: Tensor) -> Tensor:
        layer = int(layer)
        c_fc, c_proj = self.weights(layer)
        hidden = F.linear(values, c_fc)
        hidden = F.gelu(hidden * self.pre_gain[layer])
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
        self.pre_gain.clamp_(-4.0, 4.0)
        self.output_log_gain.clamp_(-4.0, 4.0)


class InstalledLayerAxisBasisMLP(nn.Module):
    """Layer view used only to splice the fitted family into the teacher."""

    def __init__(self, family: LayerAxisBasisMLP, layer: int) -> None:
        super().__init__()
        self.family = family
        self.layer = int(layer)
        self.residual_conditioned_output_slope = None
        self.conditioned_output_gate_source = "residual"

    def forward(self, values: Tensor) -> Tensor:
        return self.family.forward_layer(self.layer, values)


def initial_family(checkpoint: nn.Module) -> LayerAxisBasisMLP:
    config = checkpoint.config
    boundaries = tuple(int(value) for value in config.mlp_shared_dense_trunk_boundaries)
    if not config.mlp_shared_dense_trunk:
        raise ValueError("initialization checkpoint is not a shared trunk model")
    if int(config.mlp_shared_dense_trunk_groups) != 7 or boundaries != BOUNDARIES:
        raise ValueError("unexpected seven-trunk depth partition")
    blocks = list(checkpoint.transformer.h)
    basis_fc = torch.stack([blocks[layer].mlp.c_fc.weight for layer in ROOTS])
    basis_proj = torch.stack(
        [blocks[layer].mlp.c_proj.weight for layer in ROOTS]
    )
    if any(
        blocks[layer].mlp.c_fc.bias is not None
        or blocks[layer].mlp.c_proj.bias is not None
        for layer in ROOTS
    ):
        raise ValueError("registered basis family is bias free")
    coefficients = torch.zeros(
        len(blocks), len(ROOTS), device=basis_fc.device, dtype=basis_fc.dtype
    )
    for layer, atom in enumerate(ASSIGNMENT):
        coefficients[layer, atom] = 1.0
    pre_gain = torch.stack([block.mlp.pregelu_gain for block in blocks])
    output_log_gain = torch.stack(
        [
            block.mlp.residual_output_log_gain
            * block.mlp.residual_output_gain_scale
            for block in blocks
        ]
    )
    return LayerAxisBasisMLP(
        basis_fc=basis_fc,
        basis_proj=basis_proj,
        coefficients_fc=coefficients,
        coefficients_proj=coefficients,
        pre_gain=pre_gain,
        output_log_gain=output_log_gain,
    )


@torch.no_grad()
def full_objective(
    module: LayerAxisBasisMLP,
    data: dict[str, Tensor],
    *,
    chunk: int = 128,
) -> dict[str, float]:
    totals = {"value": 0.0, "direction": 0.0, "total": 0.0}
    rows = 0
    layers = torch.arange(module.layers, device=data["clean"].device)
    for start in range(0, data["variants"].shape[-2], int(chunk)):
        stop = min(start + int(chunk), data["variants"].shape[-2])
        prediction = module.forward_selected(
            data["variants"][..., start:stop, :], layers
        )
        parts = objective_parts(
            prediction, data["targets"][..., start:stop, :]
        )
        for key in totals:
            totals[key] += float(parts[key]) * (stop - start)
        rows += stop - start
    return {key: value / max(rows, 1) for key, value in totals.items()}


def fit_stage(
    module: LayerAxisBasisMLP,
    *,
    fit: dict[str, Tensor],
    holdout: dict[str, Tensor],
    stage: dict[str, Any],
    rows_per_layer: int,
    layers_per_update: int,
    gradient_clip_norm: float,
    seed: int,
    steps_override: int | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    parameters = module.set_trainable(
        coefficients_only=stage["name"] == "coefficients_only"
    )
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
    )
    generator = torch.Generator(device=fit["clean"].device).manual_seed(int(seed))
    steps = int(stage["steps"] if steps_override is None else steps_override)
    initial = full_objective(module, fit)
    best_loss = initial["total"]
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    finite, started = True, time.time()
    last_parts: dict[str, Tensor] | None = None
    for step in range(steps):
        layer_indices = torch.randperm(
            module.layers, generator=generator, device=fit["clean"].device
        )[: int(layers_per_update)]
        row_indices = torch.randint(
            fit["clean"].shape[2],
            (min(int(rows_per_layer), fit["clean"].shape[2]),),
            generator=generator,
            device=fit["clean"].device,
        )
        variants = fit["variants"].index_select(1, layer_indices)
        variants = variants.index_select(-2, row_indices)
        targets = fit["targets"].index_select(1, layer_indices)
        targets = targets.index_select(-2, row_indices)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            last_parts = objective_parts(
                module.forward_selected(variants, layer_indices), targets
            )
        if not torch.isfinite(last_parts["total"]):
            finite = False
            break
        last_parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip_norm))
        optimizer.step()
        module.clamp_charts()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            evaluated = full_objective(module, fit)
            if evaluated["total"] < best_loss:
                best_loss = evaluated["total"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
            print(
                json.dumps(
                    {
                        "stage": stage["name"],
                        "fit_step": step + 1,
                        "fit_steps": steps,
                        "minibatch_value": float(last_parts["value"].detach()),
                        "minibatch_direction": float(
                            last_parts["direction"].detach()
                        ),
                        "evaluated_total": evaluated["total"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    module.load_state_dict(best_state)
    final = (
        full_objective(module, fit)
        if finite
        else {"value": math.inf, "direction": math.inf, "total": math.inf}
    )
    holdout_value = (
        full_objective(module, holdout)
        if finite
        else {"value": math.inf, "direction": math.inf, "total": math.inf}
    )
    state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    return {
        "finite": finite,
        "initial_fit_objective": initial,
        "final_fit_objective": final,
        "objective_reduction_fraction": (
            1.0 - final["total"] / max(initial["total"], 1e-30)
            if finite
            else None
        ),
        "holdout_objective": holdout_value,
        "wall_seconds": time.time() - started,
    }, state


def family_from_state(
    state: dict[str, Tensor], device: str
) -> LayerAxisBasisMLP:
    return LayerAxisBasisMLP(
        basis_fc=state["basis_fc"],
        basis_proj=state["basis_proj"],
        coefficients_fc=state["coefficients_fc"],
        coefficients_proj=state["coefficients_proj"],
        pre_gain=state["pre_gain"],
        output_log_gain=state["output_log_gain"],
    ).to(device)


@torch.no_grad()
def coefficient_spectrum(module: LayerAxisBasisMLP) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, coefficients in (
        ("c_fc", module.coefficients_fc),
        ("c_proj", module.coefficients_proj),
    ):
        singular = torch.linalg.svdvals(coefficients.float()).cpu()
        energy = singular.square()
        probability = energy / energy.sum().clamp_min(1e-30)
        entropy_rank = torch.exp(
            -(probability * probability.clamp_min(1e-30).log()).sum()
        )
        rows[name] = {
            "coefficients": coefficients.detach().float().cpu().tolist(),
            "singular_values": singular.tolist(),
            "explained_energy": probability.tolist(),
            "entropy_effective_rank": float(entropy_rank),
        }
    return rows


def function_measurement(
    module: LayerAxisBasisMLP,
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
        "coefficient_spectrum": coefficient_spectrum(module),
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
    if sha256_file(HARD_EM_RESULT) != plan["causal_basis"]["hard_em_result_sha256"]:
        raise ValueError("hard-EM causal result identity mismatch")
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
    initializer = load_model(
        paths["seven_trunk_initialization_checkpoint"], args.device
    )
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
    module = initial_family(initializer).to(args.device)
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
        splice.transformer.h[layer].mlp = InstalledLayerAxisBasisMLP(
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
        classification = "LAYER_AXIS_BASIS7_PASS"
    elif healthy:
        classification = "LAYER_AXIS_BASIS7_FAMILY_FAIL"
    else:
        classification = "OPTIMIZATION_INCONCLUSIVE"
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "fitted_states.pt"
    torch.save(
        {
            "schema_version": "mai_layer_axis_basis7_mlp_state_v1",
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
