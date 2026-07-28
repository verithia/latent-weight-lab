"""Screen residual-compatibility objectives before another MLP training run.

The bilateral ``c_proj`` chart has enough held-out capacity to repair much of
the generated MLP output, but causal CE selects an almost orthogonal chart
direction.  This deterministic, no-update diagnostic asks whether residual
branch statistics provide a useful direction signal without loading a dense
teacher during training.

The target values are frozen per-layer moments measured once from the matched
attention-only control.  The candidate losses use only the source residual
stream and its generated MLP update:

* log update/residual RMS ratio;
* mean residual/update cosine;
* residual-parallel energy (mean squared cosine);
* their equal-weight sum.

Each candidate gradient is compared with the dense-teacher MLP-output MSE
gradient in exactly the same production bilateral-chart coordinates on fixed
fit and held-out validation tokens.  No parameter update is applied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    collect_model,
    tensor_sha256,
)
from examples.nanogpt.analyze_mlp_chart_gradient_alignment import (
    alignment_rows,
    chart_parameters,
    load_chart_model,
    teacher_mse_gradients,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)


OBJECTIVES = ("log_rms_ratio", "cosine", "parallel_energy", "joint")


@dataclass(frozen=True)
class ResidualSplitData:
    name: str
    batches: list[torch.Tensor]
    token_sha256: str
    pre_gelu: dict[int, torch.Tensor]
    source_residual_in: dict[int, torch.Tensor]
    teacher_residual_in: dict[int, torch.Tensor]
    teacher_mlp_out: dict[int, torch.Tensor]
    cproj_weight: dict[int, torch.Tensor]
    cproj_bias: dict[int, torch.Tensor | None]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def residual_moments(
    residual: torch.Tensor,
    update: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return differentiable residual-branch scale and direction moments."""

    residual = residual.float()
    update = update.float()
    if residual.shape != update.shape or residual.ndim != 2:
        raise ValueError(
            "residual and update must be aligned rank-two tensors, got "
            f"{tuple(residual.shape)} and {tuple(update.shape)}"
        )
    epsilon = torch.finfo(residual.dtype).eps
    residual_rms = residual.square().mean(dim=-1).sqrt()
    update_rms = update.square().mean(dim=-1).sqrt()
    cosine = (residual * update).sum(dim=-1) / (
        residual.norm(dim=-1) * update.norm(dim=-1)
    ).clamp_min(epsilon)
    return {
        "log_rms_ratio": (
            update_rms.mean().clamp_min(epsilon).log()
            - residual_rms.mean().clamp_min(epsilon).log()
        ),
        "cosine": cosine.mean(),
        "parallel_energy": cosine.square().mean(),
    }


def residual_moment_losses(
    observed: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return physical-unit squared errors and their equal-weight sum."""

    if set(observed) != set(target) or set(observed) != set(OBJECTIVES[:-1]):
        raise ValueError("residual moment maps have incompatible keys")
    losses = {
        key: (observed[key] - target[key].detach()).square()
        for key in OBJECTIVES[:-1]
    }
    losses["joint"] = sum(losses.values())
    return losses


def collect_split(
    *,
    name: str,
    seed: int,
    attention_only: Path,
    plain_cproj: Path,
    data_dir: Path,
    layers: list[int],
    batch_size: int,
    block_size: int,
    batches_count: int,
    sample_cap: int,
    device: str,
) -> ResidualSplitData:
    batches = fixed_validation_batches(
        data_dir,
        batch_size,
        block_size,
        batches_count,
        seed,
    )
    digest = tensor_sha256(torch.cat(batches))
    print(
        f"collecting split={name} seed={seed} token_sha256={digest}",
        flush=True,
    )
    teacher, _, _ = collect_model(
        attention_only,
        batches,
        layers,
        sample_cap,
        device,
        collect_pre_gelu=False,
        collect_residual_in=True,
    )
    source, weights, biases = collect_model(
        plain_cproj,
        batches,
        layers,
        sample_cap,
        device,
        collect_pre_gelu=True,
        collect_residual_in=True,
    )
    return ResidualSplitData(
        name=name,
        batches=batches,
        token_sha256=digest,
        pre_gelu={
            layer: source[(layer, "pre_gelu")] for layer in layers
        },
        source_residual_in={
            layer: source[(layer, "residual_in")] for layer in layers
        },
        teacher_residual_in={
            layer: teacher[(layer, "residual_in")] for layer in layers
        },
        teacher_mlp_out={
            layer: teacher[(layer, "mlp_out")] for layer in layers
        },
        cproj_weight=weights,
        cproj_bias=biases,
    )


def candidate_gradients(
    model: torch.nn.Module,
    parameters: dict[str, torch.nn.Parameter],
    split: ResidualSplitData,
    layers: list[int],
    device: str,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[int, dict[str, dict[str, float]]],
]:
    output = {objective: {} for objective in OBJECTIVES}
    diagnostics: dict[int, dict[str, dict[str, float]]] = {}
    for layer in layers:
        mlp = model.transformer.h[layer].mlp
        selected = {
            key: parameter
            for key, parameter in parameters.items()
            if key.startswith(f"layer.{layer}.")
        }
        keys = sorted(selected)
        values = [selected[key] for key in keys]
        pre_gelu = split.pre_gelu[layer].to(device)
        source_residual = split.source_residual_in[layer].to(device)
        teacher_residual = split.teacher_residual_in[layer].to(device)
        teacher_update = split.teacher_mlp_out[layer].to(device)
        weight = split.cproj_weight[layer].to(device)
        bias = split.cproj_bias[layer]
        bias = bias.to(device) if bias is not None else None

        charted_weight = mlp._materialize_charted_cproj_weight(weight)
        prediction = F.linear(F.gelu(pre_gelu), charted_weight, bias)
        observed = residual_moments(source_residual, prediction)
        target = residual_moments(teacher_residual, teacher_update)
        losses = residual_moment_losses(observed, target)
        diagnostics[layer] = {
            "observed": {
                key: float(value.detach()) for key, value in observed.items()
            },
            "target": {
                key: float(value.detach()) for key, value in target.items()
            },
            "loss": {
                key: float(value.detach()) for key, value in losses.items()
            },
        }
        for index, objective in enumerate(OBJECTIVES):
            gradients = torch.autograd.grad(
                losses[objective],
                values,
                retain_graph=index + 1 < len(OBJECTIVES),
            )
            for key, gradient in zip(keys, gradients, strict=True):
                output[objective][key] = gradient.detach().float().cpu()
    return output, diagnostics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", required=True, type=Path)
    parser.add_argument("--plain-cproj", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--sample-cap", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=20260716)
    parser.add_argument("--holdout-sample-seed", type=int, default=20260717)
    parser.add_argument(
        "--initial-output-log-gain", type=float, default=0.0
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",") if part]
    if not layers:
        raise ValueError("at least one layer is required")
    if args.sample_cap > args.batch_size * args.block_size * args.batches:
        raise ValueError("sample cap exceeds the available activation rows")
    if not math.isfinite(args.initial_output_log_gain):
        raise ValueError("initial-output-log-gain must be finite")

    fit = collect_split(
        name="fit",
        seed=args.sample_seed,
        attention_only=args.attention_only,
        plain_cproj=args.plain_cproj,
        data_dir=args.data_dir,
        layers=layers,
        batch_size=args.batch_size,
        block_size=args.block_size,
        batches_count=args.batches,
        sample_cap=args.sample_cap,
        device=args.device,
    )
    holdout = collect_split(
        name="holdout",
        seed=args.holdout_sample_seed,
        attention_only=args.attention_only,
        plain_cproj=args.plain_cproj,
        data_dir=args.data_dir,
        layers=layers,
        batch_size=args.batch_size,
        block_size=args.block_size,
        batches_count=args.batches,
        sample_cap=args.sample_cap,
        device=args.device,
    )

    model = load_chart_model(
        args.plain_cproj,
        args.device,
        layers,
        args.initial_output_log_gain,
    )
    parameters = chart_parameters(model, layers)
    rows: list[dict[str, object]] = []
    gradients_by_split: dict[
        str, dict[str, dict[str, torch.Tensor]]
    ] = {}
    diagnostics: dict[str, object] = {}
    for split in (fit, holdout):
        teacher_gradient, teacher_losses, _ = teacher_mse_gradients(
            model,
            parameters,
            split,
            layers,
            args.device,
        )
        candidates, split_diagnostics = candidate_gradients(
            model,
            parameters,
            split,
            layers,
            args.device,
        )
        gradients_by_split[split.name] = candidates
        diagnostics[split.name] = {
            "teacher_mse_by_layer": {
                str(layer): teacher_losses[layer] for layer in layers
            },
            "residual_moments_by_layer": {
                str(layer): split_diagnostics[layer] for layer in layers
            },
        }
        for objective in OBJECTIVES:
            objective_rows = alignment_rows(
                candidates[objective],
                teacher_gradient,
                comparison=f"residual_{objective}_vs_teacher_mse",
                split=split.name,
                initialization=args.initial_output_log_gain,
            )
            rows.extend(objective_rows)
            print(
                f"split={split.name} objective={objective} "
                f"global_cosine={objective_rows[0]['cosine']:.6f}",
                flush=True,
            )

    for objective in OBJECTIVES:
        rows.extend(
            alignment_rows(
                gradients_by_split["fit"][objective],
                gradients_by_split["holdout"][objective],
                comparison=f"residual_{objective}_fit_vs_holdout",
                split="cross_split",
                initialization=args.initial_output_log_gain,
            )
        )

    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "mlp_residual_objective_alignment.csv"
    write_csv(csv_path, rows)
    global_rows = [row for row in rows if row["scope"] == "global"]
    metadata = {
        "schema_version": "mlp_residual_objective_alignment_v1",
        "scientific_scope": (
            "deterministic no-update gradient diagnostic; targets are frozen "
            "attention-control moments, not dense target outputs"
        ),
        "attention_only": {
            "path": str(args.attention_only),
            "sha256": sha256(args.attention_only),
        },
        "plain_cproj": {
            "path": str(args.plain_cproj),
            "sha256": sha256(args.plain_cproj),
        },
        "data_dir": str(args.data_dir),
        "layers": layers,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "batches": args.batches,
        "sample_cap": args.sample_cap,
        "sample_seed": args.sample_seed,
        "holdout_sample_seed": args.holdout_sample_seed,
        "fit_token_sha256": fit.token_sha256,
        "holdout_token_sha256": holdout.token_sha256,
        "initial_output_log_gain": args.initial_output_log_gain,
        "objectives": {
            "moments": list(OBJECTIVES[:-1]),
            "joint_weighting": "equal physical-unit squared errors",
            "runtime_teacher_required": False,
            "targets_teacher_calibrated": True,
        },
        "diagnostics": diagnostics,
        "global_alignment": global_rows,
        "source": {
            "path": str(Path(__file__).relative_to(root)),
            "sha256": sha256(Path(__file__)),
            "git_commit": git_head(root),
        },
        "csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
    }
    metadata_path = (
        args.output / "mlp_residual_objective_alignment_summary.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "csv_sha256": sha256(csv_path),
                "metadata": str(metadata_path),
                "metadata_sha256": sha256(metadata_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
