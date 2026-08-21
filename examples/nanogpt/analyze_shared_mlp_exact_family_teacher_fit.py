#!/usr/bin/env python3
"""Teacher-fit the exact seven-trunk MLP family and measure its CE ceiling.

This is an offline representability oracle.  It never updates language-model
parameters with cross entropy.  Only the two genuinely shared MLP groups are
fit to dense-teacher functions on frozen residual-stream samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    prepare_inference_cache,
)
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.analyze_shared_mlp_endpoint_function import (
    module_jvp,
    pair_metrics,
    rademacher_tangent,
    sha256_file,
    summarize,
    tensor_sha256,
    validate_core_configs,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_shared_mlp_exact_family_teacher_fit_plan_v1"
RESULT_SCHEMA = "mai_shared_mlp_exact_family_teacher_fit_result_v1"
GROUPS = ((5, 6, 7), (8, 9, 10, 11))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


class StratifiedMLPInputCollector:
    """Sample the same number of MLP-input rows from every token batch."""

    def __init__(
        self,
        model: nn.Module,
        *,
        sample_cap: int,
        calls: int,
        seed: int,
    ) -> None:
        self.sample_cap = int(sample_cap)
        self.calls = int(calls)
        self.call_index = {layer: 0 for layer in range(model.config.n_layer)}
        self.values: dict[int, list[Tensor]] = {
            layer: [] for layer in range(model.config.n_layer)
        }
        self.counts = {layer: 0 for layer in range(model.config.n_layer)}
        self.generators: dict[int, torch.Generator] = {}
        for layer in range(model.config.n_layer):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + layer * 1009)
            self.generators[layer] = generator
        self.handles = [
            block.mlp.register_forward_pre_hook(self._hook(layer))
            for layer, block in enumerate(model.transformer.h)
        ]

    def _hook(self, layer: int):
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            call = self.call_index[layer]
            self.call_index[layer] = call + 1
            remaining = self.sample_cap - self.counts[layer]
            calls_left = max(self.calls - call, 1)
            take = min(math.ceil(remaining / calls_left), remaining)
            if take <= 0:
                return
            rows = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if rows.shape[0] < take:
                raise RuntimeError("token batch has too few MLP rows")
            indices = torch.randperm(
                rows.shape[0], generator=self.generators[layer]
            )[:take].to(rows.device)
            self.values[layer].append(
                rows.index_select(0, indices).float().cpu()
            )
            self.counts[layer] += int(take)

        return hook

    def tensors(self) -> dict[int, Tensor]:
        if any(count != self.sample_cap for count in self.counts.values()):
            raise RuntimeError(f"incomplete stratified collection: {self.counts}")
        return {
            layer: torch.cat(parts, dim=0)
            for layer, parts in self.values.items()
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_stratified_inputs(
    model: nn.Module,
    batches: list[Tensor],
    *,
    sample_cap: int,
    seed: int,
    device: str,
) -> dict[int, Tensor]:
    collector = StratifiedMLPInputCollector(
        model, sample_cap=sample_cap, calls=len(batches), seed=seed
    )
    prepare_inference_cache(model)
    try:
        with torch.no_grad():
            for batch in batches:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(batch.to(device), None)
        return collector.tensors()
    finally:
        collector.close()
        model.flush_block_fht_cache()


class SharedGroupMLP(nn.Module):
    """Exact production sharing family for one non-singleton depth group."""

    def __init__(
        self,
        *,
        layers: tuple[int, ...],
        c_fc_weight: Tensor,
        c_proj_weight: Tensor,
        pre_gain: Tensor,
        output_log_gain: Tensor,
    ) -> None:
        super().__init__()
        self.layers = tuple(int(layer) for layer in layers)
        self.c_fc_weight = nn.Parameter(c_fc_weight.detach().float().clone())
        self.c_proj_weight = nn.Parameter(
            c_proj_weight.detach().float().clone()
        )
        self.pre_gain = nn.Parameter(pre_gain.detach().float().clone())
        self.output_log_gain = nn.Parameter(
            output_log_gain.detach().float().clone()
        )
        if self.pre_gain.shape != (len(self.layers), self.c_fc_weight.shape[0]):
            raise ValueError("pre_gain shape does not match shared c_fc")
        if self.output_log_gain.shape != (
            len(self.layers),
            self.c_proj_weight.shape[0],
        ):
            raise ValueError("output_log_gain shape does not match shared c_proj")

    def forward(self, values: Tensor) -> Tensor:
        """Values have shape [variant, layer, bank, row, input]."""

        if values.ndim != 5 or values.shape[1] != len(self.layers):
            raise ValueError("unexpected grouped input shape")
        leading = values.shape[:-1]
        hidden = F.linear(values.reshape(-1, values.shape[-1]), self.c_fc_weight)
        hidden = hidden.reshape(*leading, -1)
        hidden = hidden * self.pre_gain[None, :, None, None, :]
        hidden = F.gelu(hidden)
        output = F.linear(hidden.reshape(-1, hidden.shape[-1]), self.c_proj_weight)
        output = output.reshape(*leading, -1)
        return output * self.output_log_gain.exp()[None, :, None, None, :]

    def forward_layer(self, layer_offset: int, values: Tensor) -> Tensor:
        hidden = F.linear(values, self.c_fc_weight)
        hidden = F.gelu(hidden * self.pre_gain[int(layer_offset)])
        output = F.linear(hidden, self.c_proj_weight)
        return output * self.output_log_gain[int(layer_offset)].exp()

    @torch.no_grad()
    def clamp_gains(self) -> None:
        self.pre_gain.clamp_(-4.0, 4.0)
        self.output_log_gain.clamp_(-4.0, 4.0)

    @torch.no_grad()
    def effective_weights(self, layer_offset: int) -> tuple[Tensor, Tensor]:
        c_fc = self.c_fc_weight * self.pre_gain[int(layer_offset)][:, None]
        c_proj = self.c_proj_weight * self.output_log_gain[
            int(layer_offset)
        ].exp()[:, None]
        return c_fc, c_proj


def initialize_group(
    *,
    layers: tuple[int, ...],
    candidate: nn.Module,
    teacher: nn.Module,
    restart: str,
    device: str,
) -> SharedGroupMLP:
    if restart == "compact_endpoint":
        root = candidate.transformer.h[layers[0]].mlp
        pre_gain = torch.stack(
            [candidate.transformer.h[layer].mlp.pregelu_gain for layer in layers]
        )
        output_log_gain = torch.stack(
            [
                candidate.transformer.h[layer].mlp.residual_output_log_gain
                * candidate.transformer.h[layer].mlp.residual_output_gain_scale
                for layer in layers
            ]
        )
        c_fc = root.c_fc.weight
        c_proj = root.c_proj.weight
    elif restart.startswith("dense_layer_"):
        layer = int(restart.rsplit("_", 1)[-1])
        if layer not in layers:
            raise ValueError("dense restart layer is outside the group")
        mlp = teacher.transformer.h[layer].mlp
        c_fc = mlp.c_fc.weight
        c_proj = mlp.c_proj.weight
        pre_gain = torch.ones(
            len(layers), c_fc.shape[0], device=c_fc.device
        )
        output_log_gain = torch.zeros(
            len(layers), c_proj.shape[0], device=c_proj.device
        )
    else:
        raise ValueError(f"unknown restart {restart!r}")
    return SharedGroupMLP(
        layers=layers,
        c_fc_weight=c_fc,
        c_proj_weight=c_proj,
        pre_gain=pre_gain,
        output_log_gain=output_log_gain,
    ).to(device)


def build_group_data(
    *,
    layers: tuple[int, ...],
    banks: dict[str, dict[int, Tensor]],
    teacher: nn.Module,
    relative_rms: float,
    seed: int,
    device: str,
) -> dict[str, Tensor]:
    clean = torch.stack(
        [
            torch.stack(
                [banks[bank][layer] for bank in ("teacher", "candidate")]
            )
            for layer in layers
        ]
    ).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + sum(layers) * 1009)
    signs = torch.randint(
        0,
        2,
        clean.shape,
        generator=generator,
        device=device,
        dtype=torch.int64,
    ).float().mul_(2.0).sub_(1.0)
    scale = clean.square().mean(dim=-1, keepdim=True).sqrt()
    delta = signs * (float(relative_rms) * scale)
    variants = torch.stack((clean, clean + delta, clean - delta))
    targets = torch.empty(
        *variants.shape[:-1],
        teacher.config.n_embd,
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for offset, layer in enumerate(layers):
            values = variants[:, offset].reshape(-1, variants.shape[-1])
            prediction = teacher.transformer.h[layer].mlp(values)
            targets[:, offset].copy_(
                prediction.reshape(*variants[:, offset].shape[:-1], -1)
            )
    return {"clean": clean, "delta": delta, "variants": variants, "targets": targets}


def normalized_objective(prediction: Tensor, target: Tensor) -> Tensor:
    residual = (prediction.float() - target.float()).square().mean(dim=(-2, -1))
    scale = target.float().square().mean(dim=(-2, -1)).clamp_min(1e-12)
    return (residual / scale).mean()


@torch.no_grad()
def full_objective(
    module: SharedGroupMLP,
    data: dict[str, Tensor],
    *,
    chunk_size: int = 256,
) -> float:
    weighted = 0.0
    rows = 0
    total = data["variants"].shape[-2]
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        prediction = module(data["variants"][..., start:stop, :])
        loss = normalized_objective(
            prediction, data["targets"][..., start:stop, :]
        )
        weighted += float(loss) * (stop - start)
        rows += stop - start
    return weighted / max(rows, 1)


@torch.no_grad()
def output_metrics(
    module: SharedGroupMLP,
    data: dict[str, Tensor],
    *,
    chunk_size: int = 256,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records: list[dict[str, float]] = []
    rows: list[dict[str, Any]] = []
    for offset, layer in enumerate(module.layers):
        for bank_index, bank in enumerate(("teacher", "candidate")):
            target_parts: list[Tensor] = []
            prediction_parts: list[Tensor] = []
            clean = data["clean"][offset, bank_index]
            target = data["targets"][0, offset, bank_index]
            for start in range(0, clean.shape[0], chunk_size):
                stop = min(start + chunk_size, clean.shape[0])
                prediction_parts.append(
                    module.forward_layer(offset, clean[start:stop]).cpu()
                )
                target_parts.append(target[start:stop].cpu())
            metric = pair_metrics(
                torch.cat(target_parts), torch.cat(prediction_parts)
            )
            records.append(metric)
            rows.append({"layer": layer, "bank": bank, **metric})
    return summarize(records), rows


def jvp_metrics(
    module: SharedGroupMLP,
    data: dict[str, Tensor],
    *,
    directions: int,
    seed: int,
    teacher: nn.Module,
    device: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records: list[dict[str, float]] = []
    rows: list[dict[str, Any]] = []
    for offset, layer in enumerate(module.layers):
        teacher_mlp = teacher.transformer.h[layer].mlp
        for bank_index, bank in enumerate(("teacher", "candidate")):
            values = data["clean"][offset, bank_index]
            teacher_parts: list[Tensor] = []
            candidate_parts: list[Tensor] = []
            for direction in range(int(directions)):
                tangent = rademacher_tangent(
                    values.shape,
                    device=device,
                    seed=seed + layer * 1000 + bank_index * 100_000 + direction,
                )
                teacher_parts.append(module_jvp(teacher_mlp, values, tangent).cpu())
                candidate_parts.append(
                    module_jvp(
                        lambda x, offset=offset: module.forward_layer(offset, x),
                        values,
                        tangent,
                    ).cpu()
                )
            metric = pair_metrics(
                torch.stack(teacher_parts), torch.stack(candidate_parts)
            )
            records.append(metric)
            rows.append({"layer": layer, "bank": bank, **metric})
    return summarize(records), rows


def fit_restart(
    module: SharedGroupMLP,
    *,
    fit: dict[str, Tensor],
    holdout: dict[str, Tensor],
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    preflight: bool,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    optimizer = torch.optim.Adam(module.parameters(), lr=float(learning_rate))
    generator = torch.Generator(device=fit["clean"].device)
    generator.manual_seed(int(seed))
    initial = full_objective(module, fit)
    started = time.time()
    finite = math.isfinite(initial)
    for step in range(int(steps)):
        indices = torch.randint(
            fit["clean"].shape[2],
            (min(int(batch_size), fit["clean"].shape[2]),),
            generator=generator,
            device=fit["clean"].device,
        )
        optimizer.zero_grad(set_to_none=True)
        variants = fit["variants"].index_select(-2, indices)
        targets = fit["targets"].index_select(-2, indices)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = module(variants)
            loss = normalized_objective(prediction, targets)
        if not torch.isfinite(loss):
            finite = False
            break
        loss.backward()
        optimizer.step()
        module.clamp_gains()
        if preflight or step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            print(
                json.dumps(
                    {"fit_step": step + 1, "fit_steps": steps, "loss": float(loss)}
                ),
                flush=True,
            )
    final = full_objective(module, fit) if finite else float("inf")
    holdout_objective = full_objective(module, holdout) if finite else float("inf")
    output_summary, output_rows = output_metrics(module, holdout)
    state = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }
    return (
        {
            "finite": bool(finite),
            "initial_fit_objective": initial,
            "final_fit_objective": final,
            "objective_reduction_fraction": (
                1.0 - final / max(initial, 1e-30) if finite else None
            ),
            "holdout_objective": holdout_objective,
            "holdout_output_summary": output_summary,
            "holdout_output_rows": output_rows,
            "wall_seconds": time.time() - started,
        },
        state,
    )


@torch.no_grad()
def install_group(
    model: nn.Module, module: SharedGroupMLP
) -> None:
    for offset, layer in enumerate(module.layers):
        c_fc, c_proj = module.effective_weights(offset)
        mlp = model.transformer.h[layer].mlp
        mlp.c_fc.weight.copy_(c_fc.to(mlp.c_fc.weight))
        mlp.c_proj.weight.copy_(c_proj.to(mlp.c_proj.weight))


def classify(
    *,
    output: dict[str, float],
    jvp: dict[str, float],
    ce_gap: float,
    healthy: bool,
    gates: dict[str, Any],
) -> str:
    if not healthy:
        return "OPTIMIZATION_INCONCLUSIVE"
    passes = (
        output["mean_explained_target_energy"]
        >= gates["minimum_mean_output_recovery"]
        and output["minimum_explained_target_energy"]
        >= gates["minimum_worst_layer_bank_output_recovery"]
        and jvp["mean_explained_target_energy"]
        >= gates["minimum_mean_input_jvp_recovery"]
        and jvp["minimum_explained_target_energy"]
        >= gates["minimum_worst_layer_bank_input_jvp_recovery"]
        and ce_gap <= gates["maximum_fixed_validation_cross_entropy_gap"]
    )
    return "REPRESENTATIONAL_PASS" if passes else "REPRESENTATIONAL_FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=10)
    args = parser.parse_args()
    if args.device != "cuda":
        raise ValueError("the preregistered oracle requires CUDA")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected plan schema")
    identity = plan["identities"]
    candidate_path = Path(identity["candidate_checkpoint"]["path"])
    teacher_path = Path(identity["dense_teacher_checkpoint"]["path"])
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    if sha256_file(candidate_path) != identity["candidate_checkpoint"]["sha256"]:
        raise ValueError("candidate checkpoint identity mismatch")
    if sha256_file(teacher_path) != identity["dense_teacher_checkpoint"]["sha256"]:
        raise ValueError("teacher checkpoint identity mismatch")
    if sha256_file(data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")

    protocol = plan["fit_protocol"]
    measurement = plan["holdout_measurement"]
    torch.manual_seed(int(protocol["fit_token_seed"]))
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    candidate = load_model(candidate_path, args.device)
    teacher = load_model(teacher_path, args.device)
    validate_core_configs(candidate, teacher)
    if tuple(candidate.config.mlp_shared_dense_trunk_boundaries) != (
        1, 2, 3, 4, 5, 8, 12
    ):
        raise ValueError("candidate does not implement the frozen partition")
    if not candidate.config.block_fht_ffn_pregelu_gain:
        raise ValueError("candidate lacks the signed pre-GELU gains")
    if not candidate.config.block_fht_mlp_residual_output_gain:
        raise ValueError("candidate lacks residual-output log gains")

    def collect_split(seed: int, batches_count: int) -> dict[str, dict[int, Tensor]]:
        batches = fixed_validation_batches(
            data_dir,
            batch_size=int(protocol["token_batch_size"]),
            block_size=candidate.config.block_size,
            batches=int(batches_count),
            seed=int(seed),
        )
        return {
            "candidate": collect_stratified_inputs(
                candidate,
                batches,
                sample_cap=int(protocol["sample_cap_per_layer_per_bank"]),
                seed=int(seed),
                device=args.device,
            ),
            "teacher": collect_stratified_inputs(
                teacher,
                batches,
                sample_cap=int(protocol["sample_cap_per_layer_per_bank"]),
                seed=int(seed),
                device=args.device,
            ),
        }

    print("collecting fit and holdout state banks", flush=True)
    fit_banks = collect_split(
        int(protocol["fit_token_seed"]), int(protocol["fit_batches"])
    )
    holdout_banks = collect_split(
        int(protocol["holdout_token_seed"]), int(protocol["holdout_batches"])
    )
    data_by_group: dict[tuple[int, ...], tuple[dict[str, Tensor], dict[str, Tensor]]] = {}
    for group_index, group in enumerate(GROUPS):
        data_by_group[group] = (
            build_group_data(
                layers=group,
                banks=fit_banks,
                teacher=teacher,
                relative_rms=float(protocol["local_perturbation_relative_rms"]),
                seed=int(protocol["fit_token_seed"]) + group_index * 10_000,
                device=args.device,
            ),
            build_group_data(
                layers=group,
                banks=holdout_banks,
                teacher=teacher,
                relative_rms=float(protocol["local_perturbation_relative_rms"]),
                seed=int(protocol["holdout_token_seed"]) + group_index * 10_000,
                device=args.device,
            ),
        )

    if args.preflight_only:
        group = GROUPS[0]
        module = initialize_group(
            layers=group,
            candidate=candidate,
            teacher=teacher,
            restart="compact_endpoint",
            device=args.device,
        )
        fit, holdout = data_by_group[group]
        row, _ = fit_restart(
            module,
            fit=fit,
            holdout=holdout,
            steps=int(args.preflight_steps),
            batch_size=int(protocol["row_batch_size_per_layer_per_bank"]),
            learning_rate=float(protocol["learning_rate"]),
            seed=int(protocol["fit_token_seed"]),
            preflight=True,
        )
        estimate = row["wall_seconds"] / max(args.preflight_steps, 1)
        print(
            json.dumps(
                {
                    "preflight": "complete",
                    "seconds_per_fit_step": estimate,
                    "estimated_fit_seconds": estimate
                    * int(protocol["steps_per_restart"])
                    * sum(len(group) + 1 for group in GROUPS),
                    "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                },
                sort_keys=True,
            )
        )
        return

    selected_modules: dict[tuple[int, ...], SharedGroupMLP] = {}
    group_results: list[dict[str, Any]] = []
    all_finite = True
    all_selected_reductions: list[float] = []
    combined_output_records: list[dict[str, float]] = []
    combined_jvp_records: list[dict[str, float]] = []
    for group_index, group in enumerate(GROUPS):
        fit, holdout = data_by_group[group]
        restart_names = ["compact_endpoint", *[f"dense_layer_{layer}" for layer in group]]
        restart_rows: list[dict[str, Any]] = []
        restart_states: dict[str, dict[str, Tensor]] = {}
        for restart_index, restart in enumerate(restart_names):
            print(json.dumps({"group": group, "restart": restart}), flush=True)
            module = initialize_group(
                layers=group,
                candidate=candidate,
                teacher=teacher,
                restart=restart,
                device=args.device,
            )
            row, state = fit_restart(
                module,
                fit=fit,
                holdout=holdout,
                steps=int(protocol["steps_per_restart"]),
                batch_size=int(protocol["row_batch_size_per_layer_per_bank"]),
                learning_rate=float(protocol["learning_rate"]),
                seed=int(protocol["fit_token_seed"])
                + group_index * 100_000
                + restart_index * 10_000,
                preflight=False,
            )
            row["restart"] = restart
            restart_rows.append(row)
            restart_states[restart] = state
            all_finite = all_finite and bool(row["finite"])
        finite_rows = [row for row in restart_rows if row["finite"]]
        if not finite_rows:
            selected_row = restart_rows[0]
        else:
            best_output = max(
                row["holdout_output_summary"][
                    "mean_explained_target_energy"
                ]
                for row in finite_rows
            )
            tied_rows = [
                row
                for row in finite_rows
                if abs(
                    row["holdout_output_summary"][
                        "mean_explained_target_energy"
                    ]
                    - best_output
                )
                <= 1e-12
            ]
            if len(tied_rows) == 1:
                selected_row = tied_rows[0]
            else:
                for tied_row in tied_rows:
                    tied_name = str(tied_row["restart"])
                    tied_module = initialize_group(
                        layers=group,
                        candidate=candidate,
                        teacher=teacher,
                        restart=tied_name,
                        device=args.device,
                    )
                    tied_module.load_state_dict(restart_states[tied_name])
                    tied_jvp, _ = jvp_metrics(
                        tied_module,
                        holdout,
                        directions=int(measurement["input_jvp_directions"]),
                        seed=int(measurement["input_jvp_seed"]),
                        teacher=teacher,
                        device=args.device,
                    )
                    tied_row["tie_break_jvp_summary"] = tied_jvp
                selected_row = max(
                    tied_rows,
                    key=lambda row: row["tie_break_jvp_summary"][
                        "mean_explained_target_energy"
                    ],
                )
        selected_name = str(selected_row["restart"])
        selected = initialize_group(
            layers=group,
            candidate=candidate,
            teacher=teacher,
            restart=selected_name,
            device=args.device,
        )
        selected.load_state_dict(restart_states[selected_name])
        jvp_summary, jvp_rows = jvp_metrics(
            selected,
            holdout,
            directions=int(measurement["input_jvp_directions"]),
            seed=int(measurement["input_jvp_seed"]),
            teacher=teacher,
            device=args.device,
        )
        selected_row["holdout_jvp_summary"] = jvp_summary
        selected_row["holdout_jvp_rows"] = jvp_rows
        combined_output_records.extend(
            selected_row["holdout_output_rows"]
        )
        combined_jvp_records.extend(jvp_rows)
        all_selected_reductions.append(
            float(selected_row["objective_reduction_fraction"])
        )
        selected_modules[group] = selected
        group_results.append(
            {
                "layers": list(group),
                "restarts": restart_rows,
                "selected_restart": selected_name,
            }
        )

    output_summary = summarize(combined_output_records)
    jvp_summary = summarize(combined_jvp_records)
    fixed = make_fixed_eval_indices(
        data_dir,
        int(measurement["fixed_eval_batch_size"]),
        int(measurement["fixed_eval_block_size"]),
        int(measurement["fixed_eval_batches"]),
        int(measurement["fixed_eval_seed"]),
    )
    fixed_digest = fixed_eval_indices_digest(fixed)
    if fixed_digest != identity["fixed_eval_indices_sha256"]:
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
    splice = load_model(teacher_path, args.device)
    for module in selected_modules.values():
        install_group(splice, module)
    splice_ce = evaluate_fixed_ce(
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
    ce_gap = splice_ce - teacher_ce
    healthy = bool(
        all_finite
        and all(
            reduction >= 0.5 for reduction in all_selected_reductions
        )
    )
    classification = classify(
        output=output_summary,
        jvp=jvp_summary,
        ce_gap=ce_gap,
        healthy=healthy,
        gates=plan["frozen_gates"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "selected_group_states.pt"
    torch.save(
        {
            "schema_version": "mai_shared_mlp_exact_family_teacher_fit_state_v1",
            "groups": {
                ",".join(str(layer) for layer in group): {
                    key: value.detach().cpu()
                    for key, value in module.state_dict().items()
                }
                for group, module in selected_modules.items()
            },
        },
        state_path,
    )
    result_path = args.output / "result.json"
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identity,
        "state_artifact": {
            "path": str(state_path),
            "sha256": sha256_file(state_path),
        },
        "fit_and_holdout_state_sha256": {
            "fit_teacher": tensor_sha256(torch.stack(list(fit_banks["teacher"].values()))),
            "fit_candidate": tensor_sha256(torch.stack(list(fit_banks["candidate"].values()))),
            "holdout_teacher": tensor_sha256(torch.stack(list(holdout_banks["teacher"].values()))),
            "holdout_candidate": tensor_sha256(torch.stack(list(holdout_banks["candidate"].values()))),
        },
        "groups": group_results,
        "summaries": {
            "holdout_output": output_summary,
            "holdout_input_jvp": jvp_summary,
        },
        "fixed_evaluation": {
            "indices_sha256": fixed_digest,
            "teacher_validation_cross_entropy": teacher_ce,
            "splice_validation_cross_entropy": splice_ce,
            "gap": ce_gap,
        },
        "optimization_healthy": healthy,
        "frozen_gates": plan["frozen_gates"],
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.time() - started,
    }
    atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "classification": classification,
                "summaries": result["summaries"],
                "fixed_evaluation": result["fixed_evaluation"],
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
