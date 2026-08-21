#!/usr/bin/env python3
"""Localize seven-trunk failure with a zero-update c_fc/c_proj factorial."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.analyze_shared_mlp_endpoint_function import (
    module_jvp,
    pair_metrics,
    rademacher_tangent,
    sha256_file,
    summarize,
    validate_core_configs,
)
from examples.nanogpt.analyze_shared_mlp_exact_family_teacher_fit import (
    GROUPS,
    SharedGroupMLP,
    collect_stratified_inputs,
    git_head,
    initialize_group,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_shared_mlp_exact_family_factorial_plan_v1"
RESULT_SCHEMA = "mai_shared_mlp_exact_family_factorial_result_v1"
VARIANTS = (
    "dense_teacher",
    "shared_c_fc_only",
    "shared_c_proj_only",
    "shared_both",
)


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


class WeightMLP(nn.Module):
    def __init__(self, c_fc: Tensor, c_proj: Tensor) -> None:
        super().__init__()
        self.register_buffer("c_fc", c_fc.detach().float().clone())
        self.register_buffer("c_proj", c_proj.detach().float().clone())

    def forward(self, values: Tensor) -> Tensor:
        return F.linear(F.gelu(F.linear(values, self.c_fc)), self.c_proj)


def variant_weights(
    *,
    teacher: nn.Module,
    fitted: dict[int, tuple[Tensor, Tensor]],
    layer: int,
    variant: str,
) -> tuple[Tensor, Tensor]:
    dense = teacher.transformer.h[layer].mlp
    dense_c_fc = dense.c_fc.weight.detach().float()
    dense_c_proj = dense.c_proj.weight.detach().float()
    fitted_c_fc, fitted_c_proj = fitted[layer]
    use_c_fc = variant in {"shared_c_fc_only", "shared_both"}
    use_c_proj = variant in {"shared_c_proj_only", "shared_both"}
    return (
        fitted_c_fc if use_c_fc else dense_c_fc,
        fitted_c_proj if use_c_proj else dense_c_proj,
    )


def function_metrics(
    *,
    teacher: nn.Module,
    fitted: dict[int, tuple[Tensor, Tensor]],
    banks: dict[str, dict[int, Tensor]],
    variant: str,
    directions: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    output_records: list[dict[str, float]] = []
    jvp_records: list[dict[str, float]] = []
    rows: list[dict[str, Any]] = []
    for layer in sorted(fitted):
        c_fc, c_proj = variant_weights(
            teacher=teacher, fitted=fitted, layer=layer, variant=variant
        )
        candidate = WeightMLP(c_fc, c_proj).to(device)
        teacher_mlp = teacher.transformer.h[layer].mlp
        for bank_index, bank in enumerate(("teacher", "candidate")):
            values = banks[bank][layer].to(device)
            with torch.no_grad():
                target = teacher_mlp(values)
                prediction = candidate(values)
            output = pair_metrics(target, prediction)
            target_jvps: list[Tensor] = []
            prediction_jvps: list[Tensor] = []
            for direction in range(int(directions)):
                tangent = rademacher_tangent(
                    values.shape,
                    device=device,
                    seed=seed + layer * 1000 + bank_index * 100_000 + direction,
                )
                target_jvps.append(module_jvp(teacher_mlp, values, tangent).cpu())
                prediction_jvps.append(module_jvp(candidate, values, tangent).cpu())
            jvp = pair_metrics(
                torch.stack(target_jvps), torch.stack(prediction_jvps)
            )
            output_records.append(output)
            jvp_records.append(jvp)
            rows.append(
                {"layer": layer, "bank": bank, "output": output, "input_jvp": jvp}
            )
    return {
        "output": summarize(output_records),
        "input_jvp": summarize(jvp_records),
        "rows": rows,
    }


@torch.no_grad()
def install_variant(
    model: nn.Module,
    *,
    dense: dict[int, tuple[Tensor, Tensor]],
    fitted: dict[int, tuple[Tensor, Tensor]],
    variant: str,
) -> None:
    for layer in sorted(fitted):
        dense_c_fc, dense_c_proj = dense[layer]
        fitted_c_fc, fitted_c_proj = fitted[layer]
        use_c_fc = variant in {"shared_c_fc_only", "shared_both"}
        use_c_proj = variant in {"shared_c_proj_only", "shared_both"}
        mlp = model.transformer.h[layer].mlp
        mlp.c_fc.weight.copy_(
            (fitted_c_fc if use_c_fc else dense_c_fc).to(mlp.c_fc.weight)
        )
        mlp.c_proj.weight.copy_(
            (fitted_c_proj if use_c_proj else dense_c_proj).to(mlp.c_proj.weight)
        )


def variant_passes(
    metrics: dict[str, Any], ce_gap: float, gates: dict[str, Any]
) -> bool:
    output = metrics["output"]
    jvp = metrics["input_jvp"]
    return bool(
        output["mean_explained_target_energy"]
        >= gates["minimum_mean_output_recovery"]
        and output["minimum_explained_target_energy"]
        >= gates["minimum_worst_output_recovery"]
        and jvp["mean_explained_target_energy"]
        >= gates["minimum_mean_input_jvp_recovery"]
        and jvp["minimum_explained_target_energy"]
        >= gates["minimum_worst_input_jvp_recovery"]
        and ce_gap <= gates["maximum_fixed_validation_cross_entropy_gap"]
    )


def classify_factorial(
    c_fc_passes: bool, c_proj_passes: bool, both_passes: bool
) -> str:
    if not c_fc_passes and c_proj_passes:
        return "C_FC_RESTRICTION"
    if c_fc_passes and not c_proj_passes:
        return "C_PROJ_RESTRICTION"
    if not c_fc_passes and not c_proj_passes:
        return "BILATERAL_RESTRICTION"
    if not both_passes:
        return "NONLINEAR_INTERACTION_RESTRICTION"
    return "NO_LOCALIZED_RESTRICTION"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda":
        raise ValueError("factorial requires CUDA")
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected plan schema")
    causal = plan["causal_basis"]
    fit_result = Path(causal["teacher_fit_result"])
    if sha256_file(fit_result) != causal["teacher_fit_result_sha256"]:
        raise ValueError("teacher-fit result identity mismatch")
    identity = plan["identities"]
    teacher_path = Path(identity["dense_teacher_checkpoint"]["path"])
    candidate_path = Path(identity["compact_candidate_checkpoint"]["path"])
    state_path = Path(identity["selected_fit_state"]["path"])
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    for path, expected, label in (
        (teacher_path, identity["dense_teacher_checkpoint"]["sha256"], "teacher"),
        (candidate_path, identity["compact_candidate_checkpoint"]["sha256"], "candidate"),
        (state_path, identity["selected_fit_state"]["sha256"], "fit state"),
        (data_dir / "manifest.json", identity["dataset_manifest_sha256"], "manifest"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} identity mismatch")

    measurement = plan["measurement"]
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    teacher = load_model(teacher_path, args.device)
    candidate = load_model(candidate_path, args.device)
    validate_core_configs(candidate, teacher)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != "mai_shared_mlp_exact_family_teacher_fit_state_v1":
        raise ValueError("unexpected selected-state schema")
    fitted: dict[int, tuple[Tensor, Tensor]] = {}
    for group in GROUPS:
        key = ",".join(str(layer) for layer in group)
        module = initialize_group(
            layers=group,
            candidate=candidate,
            teacher=teacher,
            restart="compact_endpoint",
            device=args.device,
        )
        module.load_state_dict(state["groups"][key])
        for offset, layer in enumerate(group):
            fitted[layer] = tuple(
                value.detach().float()
                for value in module.effective_weights(offset)
            )
    dense = {
        layer: (
            teacher.transformer.h[layer].mlp.c_fc.weight.detach().float().clone(),
            teacher.transformer.h[layer].mlp.c_proj.weight.detach().float().clone(),
        )
        for layer in fitted
    }

    batches = fixed_validation_batches(
        data_dir,
        batch_size=int(measurement["token_batch_size"]),
        block_size=teacher.config.block_size,
        batches=int(measurement["holdout_batches"]),
        seed=int(measurement["holdout_token_seed"]),
    )
    banks = {
        "teacher": collect_stratified_inputs(
            teacher,
            batches,
            sample_cap=int(measurement["samples_per_layer_per_bank"]),
            seed=int(measurement["holdout_token_seed"]),
            device=args.device,
        ),
        "candidate": collect_stratified_inputs(
            candidate,
            batches,
            sample_cap=int(measurement["samples_per_layer_per_bank"]),
            seed=int(measurement["holdout_token_seed"]),
            device=args.device,
        ),
    }
    function = {
        variant: function_metrics(
            teacher=teacher,
            fitted=fitted,
            banks=banks,
            variant=variant,
            directions=int(measurement["input_jvp_directions"]),
            seed=int(measurement["input_jvp_seed"]),
            device=args.device,
        )
        for variant in VARIANTS
    }

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
    ce: dict[str, float] = {}
    for variant in VARIANTS:
        print(json.dumps({"evaluating": variant}), flush=True)
        install_variant(
            teacher, dense=dense, fitted=fitted, variant=variant
        )
        ce[variant] = evaluate_fixed_ce(
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
    baseline = ce["dense_teacher"]
    gaps = {variant: value - baseline for variant, value in ce.items()}
    passes = {
        variant: variant_passes(function[variant], gaps[variant], plan["frozen_gates"])
        for variant in VARIANTS
    }
    classification = classify_factorial(
        passes["shared_c_fc_only"],
        passes["shared_c_proj_only"],
        passes["shared_both"],
    )
    interaction = (
        gaps["shared_both"]
        - gaps["shared_c_fc_only"]
        - gaps["shared_c_proj_only"]
    )
    payload = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identity,
        "function": function,
        "fixed_evaluation": {
            "indices_sha256": fixed_digest,
            "validation_cross_entropy": ce,
            "gap_to_dense_teacher": gaps,
            "superadditive_interaction_gap": interaction,
        },
        "passes": passes,
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.time() - started,
    }
    result_path = args.output / "result.json"
    atomic_json(result_path, payload)
    print(
        json.dumps(
            {
                "classification": classification,
                "ce": ce,
                "gaps": gaps,
                "passes": passes,
                "interaction": interaction,
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
