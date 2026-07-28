"""Measure whether task CE selects the useful fixed MLP chart direction.

The held-out bilateral oracle shows that a fixed, layer-specific chart has
enough local capacity to repair much of a generated ``c_proj`` error, while
end-to-end CE training fails to realize that capacity.  This diagnostic
separates representation from direction selection:

* ``task_ce`` is the gradient of causal validation CE through the complete
  source model with the production bilateral chart attached.
* ``teacher_mse`` is the gradient, in the exact same chart coordinates, of
  the fixed attention-teacher MLP-output error used by the structural oracle.

Both gradients are measured on deterministic fit and held-out token windows.
No parameter update is applied and no learned dense basis is introduced.
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

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    collect_model,
    tensor_sha256,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.model import GPT, GPTConfig


CHART_GROUPS = (
    "hidden_rotation",
    "hidden_gain",
    "output_rotation",
    "output_gain",
)


@dataclass(frozen=True)
class SplitData:
    name: str
    batches: list[torch.Tensor]
    token_sha256: str
    pre_gelu: dict[int, torch.Tensor]
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


def parse_float_list(value: str) -> list[float]:
    parsed = [float(part) for part in value.split(",") if part.strip()]
    if not parsed or any(not math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError(
            "initializations must be a comma-separated list of finite values"
        )
    return parsed


def chart_key(layer: int, group: str) -> str:
    return f"layer.{int(layer)}.{group}"


def split_chart_key(key: str) -> tuple[int, str]:
    prefix, layer, group = key.split(".", 2)
    if prefix != "layer" or group not in CHART_GROUPS:
        raise ValueError(f"invalid chart-gradient key {key!r}")
    return int(layer), group


def vector_alignment(
    left: torch.Tensor, right: torch.Tensor
) -> dict[str, float]:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    if left.shape != right.shape:
        raise ValueError(
            f"gradient shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    left_norm = left.norm()
    right_norm = right.norm()
    denominator = left_norm * right_norm
    cosine = (
        float(torch.dot(left, right) / denominator)
        if denominator > 0
        else float("nan")
    )
    active = (left != 0) & (right != 0)
    sign_agreement = (
        float((left[active].sign() == right[active].sign()).float().mean())
        if torch.any(active)
        else float("nan")
    )
    right_energy = right.square().sum()
    projection = (
        float(torch.dot(left, right) / right_energy)
        if right_energy > 0
        else float("nan")
    )
    return {
        "coordinates": float(left.numel()),
        "left_norm": float(left_norm),
        "right_norm": float(right_norm),
        "left_rms": float(left.square().mean().sqrt()),
        "right_rms": float(right.square().mean().sqrt()),
        "dot": float(torch.dot(left, right)),
        "cosine": cosine,
        "sign_agreement": sign_agreement,
        "left_projection_on_right": projection,
    }


def flatten_gradients(
    gradients: dict[str, torch.Tensor], selected: list[str]
) -> torch.Tensor:
    if not selected:
        raise ValueError("cannot flatten an empty gradient selection")
    return torch.cat(
        [gradients[key].detach().float().reshape(-1).cpu() for key in selected]
    )


def alignment_rows(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    comparison: str,
    split: str,
    initialization: float,
) -> list[dict[str, object]]:
    if set(left) != set(right):
        raise ValueError("gradient maps do not have identical keys")
    keys = sorted(left)
    layers = sorted({split_chart_key(key)[0] for key in keys})
    rows: list[dict[str, object]] = []

    def append(scope: str, layer: int | None, group: str | None) -> None:
        selected = [
            key
            for key in keys
            if (layer is None or split_chart_key(key)[0] == layer)
            and (group is None or split_chart_key(key)[1] == group)
        ]
        metrics = vector_alignment(
            flatten_gradients(left, selected),
            flatten_gradients(right, selected),
        )
        rows.append(
            {
                "comparison": comparison,
                "split": split,
                "initial_effective_output_log_gain": initialization,
                "scope": scope,
                "layer": "" if layer is None else layer,
                "group": "" if group is None else group,
                **metrics,
            }
        )

    append("global", None, None)
    for group in CHART_GROUPS:
        append("group", None, group)
    for layer in layers:
        append("layer", layer, None)
        for group in CHART_GROUPS:
            append("layer_group", layer, group)
    return rows


def expected_chart_state_key(key: str) -> bool:
    return any(
        token in key
        for token in (
            ".hidden_block_rotation.",
            ".hidden_log_gain",
            ".output_block_rotation.",
            ".residual_output_log_gain",
        )
    )


def chart_config(
    source: dict[str, object], initial_output_log_gain: float
) -> GPTConfig:
    values = dict(source)
    values.update(
        {
            "block_fht_cache_weights": True,
            "block_fht_mlp_activation_chart": False,
            "block_fht_mlp_hidden_block_rotation_stages": 2,
            "block_fht_mlp_hidden_block_rotation_size": 32,
            "block_fht_mlp_hidden_block_rotation_basis_size": 256,
            "block_fht_mlp_hidden_block_rotation_coordinate_scale": 4.0,
            "block_fht_mlp_hidden_block_rotation_seed": 314159,
            "block_fht_mlp_hidden_gain": True,
            "block_fht_mlp_hidden_gain_scale": 4.0,
            "block_fht_mlp_hidden_log_gain_init": 0.0,
            "block_fht_mlp_output_rotation_stages": 0,
            "block_fht_mlp_output_block_rotation_stages": 4,
            "block_fht_mlp_output_block_rotation_size": 32,
            "block_fht_mlp_output_block_rotation_basis_size": 256,
            "block_fht_mlp_output_block_rotation_coordinate_scale": 4.0,
            "block_fht_mlp_output_rotation_seed": 271828,
            "block_fht_mlp_residual_output_gain": True,
            "block_fht_mlp_residual_output_gain_scale": 4.0,
            "block_fht_mlp_residual_output_log_gain_init": (
                float(initial_output_log_gain)
            ),
        }
    )
    return GPTConfig(**values)


def load_chart_model(
    checkpoint_path: Path,
    device: str,
    layers: list[int],
    initial_output_log_gain: float,
) -> GPT:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = chart_config(
        checkpoint["model_config"], initial_output_log_gain
    )
    with torch.device(device):
        model = GPT(config)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not expected_chart_state_key(key)
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            "source checkpoint is incompatible with the chart model: "
            f"unexpected={unexpected} invalid_missing={invalid_missing}"
        )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = set(layers)
    for layer, block in enumerate(model.transformer.h):
        mlp = block.mlp
        enabled = layer in selected
        for parameter in (
            (
                mlp.hidden_block_rotation.coordinates
                if mlp.hidden_block_rotation is not None
                else None
            ),
            mlp.hidden_log_gain,
            (
                mlp.output_block_rotation.coordinates
                if mlp.output_block_rotation is not None
                else None
            ),
            mlp.residual_output_log_gain,
        ):
            if parameter is not None:
                parameter.requires_grad_(enabled)
    return model


def chart_parameters(
    model: GPT, layers: list[int]
) -> dict[str, torch.nn.Parameter]:
    output: dict[str, torch.nn.Parameter] = {}
    for layer in layers:
        mlp = model.transformer.h[layer].mlp
        values = {
            "hidden_rotation": (
                mlp.hidden_block_rotation.coordinates
                if mlp.hidden_block_rotation is not None
                else None
            ),
            "hidden_gain": mlp.hidden_log_gain,
            "output_rotation": (
                mlp.output_block_rotation.coordinates
                if mlp.output_block_rotation is not None
                else None
            ),
            "output_gain": mlp.residual_output_log_gain,
        }
        for group, parameter in values.items():
            if parameter is None:
                raise RuntimeError(
                    f"layer {layer} is missing chart group {group}"
                )
            if not parameter.requires_grad:
                raise RuntimeError(
                    f"layer {layer} chart group {group} is frozen"
                )
            output[chart_key(layer, group)] = parameter
    return output


def clone_parameter_gradients(
    parameters: dict[str, torch.nn.Parameter]
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, parameter in parameters.items():
        if parameter.grad is None:
            raise RuntimeError(f"chart parameter {key} has no gradient")
        output[key] = parameter.grad.detach().float().cpu().clone()
    return output


def task_ce_gradients(
    model: GPT,
    parameters: dict[str, torch.nn.Parameter],
    batches: list[torch.Tensor],
    device: str,
) -> tuple[dict[str, torch.Tensor], float]:
    model.zero_grad(set_to_none=True)
    cache_dtype = (
        torch.bfloat16 if device.startswith("cuda") else torch.float32
    )
    model.prepare_block_fht_cache(dtype=cache_dtype)
    losses: list[float] = []
    for tokens in batches:
        tokens = tokens.to(device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with context:
            _, loss = model(inputs, targets)
        assert loss is not None
        losses.append(float(loss.detach()))
        (loss / len(batches)).backward()
    model.flush_block_fht_cache()
    return clone_parameter_gradients(parameters), float(np.mean(losses))


def teacher_mse_gradients(
    model: GPT,
    parameters: dict[str, torch.nn.Parameter],
    split: SplitData,
    layers: list[int],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[int, float]]:
    output: dict[str, torch.Tensor] = {}
    losses: dict[int, float] = {}
    for layer in layers:
        mlp = model.transformer.h[layer].mlp
        selected = {
            key: parameter
            for key, parameter in parameters.items()
            if split_chart_key(key)[0] == layer
        }
        keys = sorted(selected)
        values = [selected[key] for key in keys]
        pre_gelu = split.pre_gelu[layer].to(device)
        target = split.teacher_mlp_out[layer].to(device)
        weight = split.cproj_weight[layer].to(device)
        bias = split.cproj_bias[layer]
        bias = bias.to(device) if bias is not None else None
        activated = F.gelu(pre_gelu)
        charted_weight = mlp._materialize_charted_cproj_weight(weight)
        prediction = F.linear(activated, charted_weight, bias)
        loss = F.mse_loss(prediction, target)
        gradients = torch.autograd.grad(loss, values)
        losses[layer] = float(loss.detach())
        for key, gradient in zip(keys, gradients):
            output[key] = gradient.detach().float().cpu()
    return output, losses


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
) -> SplitData:
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
    )
    source, weights, biases = collect_model(
        plain_cproj,
        batches,
        layers,
        sample_cap,
        device,
        collect_pre_gelu=True,
    )
    return SplitData(
        name=name,
        batches=batches,
        token_sha256=digest,
        pre_gelu={
            layer: source[(layer, "pre_gelu")] for layer in layers
        },
        teacher_mlp_out={
            layer: teacher[(layer, "mlp_out")] for layer in layers
        },
        cproj_weight=weights,
        cproj_bias=biases,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--sample-cap", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=20260716)
    parser.add_argument("--holdout-sample-seed", type=int, default=20260717)
    parser.add_argument(
        "--initializations", type=parse_float_list, default=[0.0, 0.125]
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",") if part]
    if not layers:
        raise ValueError("at least one layer is required")
    if args.sample_cap > args.batch_size * args.block_size * args.batches:
        raise ValueError("sample cap exceeds the available activation rows")

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

    rows: list[dict[str, object]] = []
    objective: dict[str, object] = {}
    for initialization in args.initializations:
        print(
            "measuring initial_effective_output_log_gain="
            f"{initialization}",
            flush=True,
        )
        model = load_chart_model(
            args.plain_cproj,
            args.device,
            layers,
            initialization,
        )
        parameters = chart_parameters(model, layers)
        ce_by_split: dict[str, dict[str, torch.Tensor]] = {}
        teacher_by_split: dict[str, dict[str, torch.Tensor]] = {}
        init_key = f"{initialization:.8g}"
        objective[init_key] = {}
        for split in (fit, holdout):
            ce_gradient, ce_loss = task_ce_gradients(
                model, parameters, split.batches, args.device
            )
            teacher_gradient, teacher_losses = teacher_mse_gradients(
                model, parameters, split, layers, args.device
            )
            ce_by_split[split.name] = ce_gradient
            teacher_by_split[split.name] = teacher_gradient
            objective[init_key][split.name] = {
                "task_ce": ce_loss,
                "teacher_mse_by_layer": {
                    str(layer): teacher_losses[layer] for layer in layers
                },
            }
            rows.extend(
                alignment_rows(
                    ce_gradient,
                    teacher_gradient,
                    comparison="task_ce_vs_teacher_mse",
                    split=split.name,
                    initialization=initialization,
                )
            )
            print(
                f"initialization={initialization} split={split.name} "
                f"task_ce={ce_loss:.6f} global_cosine="
                f"{rows[-(1 + len(CHART_GROUPS) + len(layers) * (1 + len(CHART_GROUPS)))]['cosine']:.6f}",
                flush=True,
            )
        rows.extend(
            alignment_rows(
                ce_by_split["fit"],
                ce_by_split["holdout"],
                comparison="task_ce_fit_vs_holdout",
                split="cross_split",
                initialization=initialization,
            )
        )
        rows.extend(
            alignment_rows(
                teacher_by_split["fit"],
                teacher_by_split["holdout"],
                comparison="teacher_mse_fit_vs_holdout",
                split="cross_split",
                initialization=initialization,
            )
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "mlp_chart_gradient_alignment.csv"
    write_csv(csv_path, rows)
    global_rows = [
        row
        for row in rows
        if row["scope"] == "global"
    ]
    metadata = {
        "schema_version": "mlp_chart_gradient_alignment_v1",
        "scientific_scope": (
            "deterministic no-update gradient diagnostic; not a training result"
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
        "initial_effective_output_log_gains": args.initializations,
        "chart": {
            "hidden_rotation_stages": 2,
            "hidden_rotation_coordinate_scale": 4.0,
            "hidden_gain_scale": 4.0,
            "output_rotation_stages": 4,
            "output_rotation_coordinate_scale": 4.0,
            "output_gain_scale": 4.0,
            "learned_dense_basis": False,
            "lora_adapter": False,
        },
        "objectives": objective,
        "global_alignment": global_rows,
        "source": {
            "path": str(Path(__file__).relative_to(root)),
            "sha256": sha256(Path(__file__)),
            "git_commit": git_head(root),
        },
        "csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
    }
    metadata_path = args.output / "mlp_chart_gradient_alignment_summary.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "csv_sha256": metadata["csv"]["sha256"],
                "global_alignment": global_rows,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
