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
from torch.func import functional_call, jvp, vjp

from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    collect_model,
    tensor_sha256,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon import zeropower_via_newtonschulz5


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


def effective_polar_chart_gradients(
    model: GPT,
    parameters: dict[str, torch.nn.Parameter],
    layers: list[int],
    ns_steps: int,
) -> dict[str, torch.Tensor]:
    """Pull a dense effective-weight polar direction into chart coordinates.

    CE first produces a dense gradient on the cached effective ``c_proj``
    weight.  Dense Muon acts in that matrix space, whereas the current chart
    sends the raw VJP to AdamW-owned coordinates.  This no-update diagnostic
    applies Muon's Newton-Schulz polar transform to the effective gradient,
    then computes the exact VJP through the fixed bilateral chart.  It does
    not change the generated base, create a learned basis, or retain dense
    optimizer state.
    """

    if ns_steps <= 0:
        raise ValueError("ns_steps must be positive")
    output: dict[str, torch.Tensor] = {}
    for layer in layers:
        mlp = model.transformer.h[layer].mlp
        cached = mlp._cached_charted_cproj_weight
        if cached is None or cached.grad is None:
            raise RuntimeError(
                f"layer {layer} has no cached effective c_proj gradient"
            )
        base_weight = getattr(mlp.c_proj, "_cached_weight", None)
        if base_weight is None:
            raise RuntimeError(
                f"layer {layer} has no cached generated c_proj base"
            )
        selected = {
            key: parameter
            for key, parameter in parameters.items()
            if split_chart_key(key)[0] == layer
        }
        keys = sorted(selected)
        values = [selected[key] for key in keys]
        dense_polar = zeropower_via_newtonschulz5(
            cached.grad.detach().float(),
            steps=ns_steps,
        )
        with torch.enable_grad():
            # The base is fixed for this chart-only direction diagnostic.
            # FP32 avoids conflating the metric test with BF16 VJP rounding.
            charted = mlp._materialize_charted_cproj_weight(
                base_weight.detach().float()
            )
            gradients = torch.autograd.grad(
                charted,
                values,
                grad_outputs=dense_polar.to(dtype=charted.dtype),
            )
        for key, gradient in zip(keys, gradients):
            output[key] = gradient.detach().float().cpu()
    return output


class ChartedCProjWeightView(torch.nn.Module):
    """Expose only the materialized effective c_proj weight as a function."""

    def __init__(self, mlp: torch.nn.Module, base_weight: torch.Tensor) -> None:
        super().__init__()
        self.mlp = mlp
        reference = next(mlp.parameters())
        self.register_buffer(
            "base_weight",
            base_weight.detach().to(
                device=reference.device,
                dtype=torch.float32,
            ),
        )

    def forward(self) -> torch.Tensor:
        return self.mlp._materialize_charted_cproj_weight(self.base_weight)


def _tuple_dot(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    return sum(
        (left_value * right_value).sum()
        for left_value, right_value in zip(left, right, strict=True)
    )


def natural_chart_directions(
    model: GPT,
    parameters: dict[str, torch.nn.Parameter],
    layers: list[int],
    dense_gradients: dict[int, torch.Tensor],
    base_weights: dict[int, torch.Tensor],
    *,
    damping_ratio: float,
    cg_steps: int,
    trace_samples: int,
    trace_seed: int,
    metric_activations: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[int, dict[str, float]]]:
    """Solve a matrix-free damped inverse-pullback system per layer.

    For effective weight Jacobian ``J`` and dense objective covector ``u``,
    this returns ``delta = (J^T J + damping I)^-1 J^T u``.  Damping is a
    dimensionless fraction of a deterministic Hutchinson estimate of the
    mean eigenvalue of ``J^T J``.  When ``metric_activations`` is supplied,
    the system instead uses the functional metric induced by the actual MLP
    outputs ``A @ delta_W^T``.  No update is applied.
    """

    if not math.isfinite(damping_ratio) or damping_ratio <= 0.0:
        raise ValueError("damping_ratio must be positive and finite")
    if cg_steps <= 0:
        raise ValueError("cg_steps must be positive")
    if trace_samples <= 0:
        raise ValueError("trace_samples must be positive")
    output: dict[str, torch.Tensor] = {}
    diagnostics: dict[int, dict[str, float]] = {}
    group_to_name = {
        "hidden_rotation": "mlp.hidden_block_rotation.coordinates",
        "hidden_gain": "mlp.hidden_log_gain",
        "output_rotation": "mlp.output_block_rotation.coordinates",
        "output_gain": "mlp.residual_output_log_gain",
    }
    for layer in layers:
        if layer not in dense_gradients or layer not in base_weights:
            raise ValueError(f"missing dense gradient/base for layer {layer}")
        mlp = model.transformer.h[layer].mlp
        view = ChartedCProjWeightView(mlp, base_weights[layer])
        keys = [chart_key(layer, group) for group in CHART_GROUPS]
        values = [parameters[key] for key in keys]
        names = [group_to_name[group] for group in CHART_GROUPS]
        primals = tuple(value.detach() for value in values)

        def materialize(*coordinates: torch.Tensor) -> torch.Tensor:
            replacements = {
                name: coordinate
                for name, coordinate in zip(
                    names, coordinates, strict=True
                )
            }
            return functional_call(
                view,
                replacements,
                (),
                strict=False,
            )

        _, pullback = vjp(materialize, *primals)
        dense_gradient = dense_gradients[layer].to(
            device=primals[0].device,
            dtype=torch.float32,
        )
        rhs = tuple(value.detach() for value in pullback(dense_gradient))
        activations = None
        if metric_activations is not None:
            if layer not in metric_activations:
                raise ValueError(
                    f"missing metric activations for layer {layer}"
                )
            activations = metric_activations[layer].detach().to(
                device=primals[0].device,
                dtype=torch.float32,
            )
            if activations.ndim != 2:
                raise ValueError(
                    "metric activations must be rank two, got "
                    f"{tuple(activations.shape)} for layer {layer}"
                )
            if activations.shape[1] != base_weights[layer].shape[1]:
                raise ValueError(
                    "metric activation width does not match c_proj input "
                    f"for layer {layer}: {activations.shape[1]} vs "
                    f"{base_weights[layer].shape[1]}"
                )

        generator = torch.Generator(device=primals[0].device)
        generator.manual_seed(int(trace_seed) + int(layer) * 1009)
        trace_estimates: list[torch.Tensor] = []
        coordinate_count = sum(value.numel() for value in primals)
        for _ in range(trace_samples):
            probe = tuple(
                (
                    torch.randint(
                        0,
                        2,
                        value.shape,
                        device=value.device,
                        generator=generator,
                    ).to(dtype=value.dtype)
                    * 2.0
                    - 1.0
                )
                for value in primals
            )
            _, weight_image = jvp(materialize, primals, probe)
            if activations is None:
                trace_image = weight_image
            else:
                trace_image = F.linear(
                    activations, weight_image
                ) / math.sqrt(float(activations.shape[0]))
            trace_estimates.append(trace_image.float().square().sum())
        mean_eigenvalue = torch.stack(trace_estimates).mean() / float(
            coordinate_count
        )
        damping = (
            float(damping_ratio)
            * mean_eigenvalue.clamp_min(1e-12)
        )

        def system(
            direction: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            _, weight_image = jvp(materialize, primals, direction)
            if activations is None:
                metric_image = weight_image
            else:
                output_image = F.linear(activations, weight_image)
                metric_image = (
                    output_image.transpose(0, 1) @ activations
                ) / float(activations.shape[0])
            pulled = pullback(metric_image)
            return tuple(
                value.detach() + damping * direction_value
                for value, direction_value in zip(
                    pulled, direction, strict=True
                )
            )

        solution = tuple(torch.zeros_like(value) for value in rhs)
        residual = tuple(value.clone() for value in rhs)
        conjugate = tuple(value.clone() for value in residual)
        initial_norm_squared = _tuple_dot(residual, residual).clamp_min(
            1e-30
        )
        norm_squared = initial_norm_squared
        completed_steps = 0
        for step in range(cg_steps):
            image = system(conjugate)
            denominator = _tuple_dot(conjugate, image)
            if not torch.isfinite(denominator) or denominator <= 0:
                break
            alpha = norm_squared / denominator
            solution = tuple(
                value + alpha * direction
                for value, direction in zip(
                    solution, conjugate, strict=True
                )
            )
            residual = tuple(
                value - alpha * image_value
                for value, image_value in zip(
                    residual, image, strict=True
                )
            )
            next_norm_squared = _tuple_dot(residual, residual)
            completed_steps = step + 1
            if next_norm_squared <= initial_norm_squared * 1e-12:
                norm_squared = next_norm_squared
                break
            beta = next_norm_squared / norm_squared.clamp_min(1e-30)
            conjugate = tuple(
                residual_value + beta * direction
                for residual_value, direction in zip(
                    residual, conjugate, strict=True
                )
            )
            norm_squared = next_norm_squared

        for key, value in zip(keys, solution, strict=True):
            output[key] = value.detach().float().cpu()
        diagnostics[layer] = {
            "coordinate_count": float(coordinate_count),
            "mean_eigenvalue": float(mean_eigenvalue),
            "damping": float(damping),
            "rhs_norm": float(initial_norm_squared.sqrt()),
            "relative_residual": float(
                (
                    norm_squared.clamp_min(0).sqrt()
                    / initial_norm_squared.sqrt()
                )
            ),
            "cg_steps": float(completed_steps),
            "metric": (
                "activation_output_mse"
                if activations is not None
                else "effective_weight_frobenius"
            ),
            "activation_samples": (
                float(activations.shape[0])
                if activations is not None
                else 0.0
            ),
        }
    return output, diagnostics


def task_ce_gradients(
    model: GPT,
    parameters: dict[str, torch.nn.Parameter],
    batches: list[torch.Tensor],
    device: str,
    layers: list[int],
    ns_steps: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    float,
]:
    model.zero_grad(set_to_none=True)
    cache_dtype = (
        torch.bfloat16 if device.startswith("cuda") else torch.float32
    )
    model.prepare_block_fht_cache(dtype=cache_dtype)
    losses: list[float] = []
    for tokens in batches:
        tokens = tokens.to(device)
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()
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
    dense_gradients: dict[int, torch.Tensor] = {}
    base_weights: dict[int, torch.Tensor] = {}
    for layer in layers:
        mlp = model.transformer.h[layer].mlp
        cached = mlp._cached_charted_cproj_weight
        base = getattr(mlp.c_proj, "_cached_weight", None)
        if cached is None or cached.grad is None or base is None:
            raise RuntimeError(
                f"layer {layer} is missing cached CE weight state"
            )
        dense_gradients[layer] = cached.grad.detach().float().clone()
        base_weights[layer] = base.detach().float().clone()
    polar_gradients = effective_polar_chart_gradients(
        model,
        parameters,
        layers,
        ns_steps,
    )
    model.flush_block_fht_cache()
    return (
        clone_parameter_gradients(parameters),
        polar_gradients,
        dense_gradients,
        base_weights,
        float(np.mean(losses)),
    )


def teacher_mse_gradients(
    model: GPT,
    parameters: dict[str, torch.nn.Parameter],
    split: SplitData,
    layers: list[int],
    device: str,
) -> tuple[
    dict[str, torch.Tensor],
    dict[int, float],
    dict[int, torch.Tensor],
]:
    output: dict[str, torch.Tensor] = {}
    losses: dict[int, float] = {}
    dense_gradients: dict[int, torch.Tensor] = {}
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
        gradients = torch.autograd.grad(loss, [charted_weight, *values])
        dense_gradients[layer] = gradients[0].detach().float()
        losses[layer] = float(loss.detach())
        for key, gradient in zip(keys, gradients[1:]):
            output[key] = gradient.detach().float().cpu()
    return output, losses, dense_gradients


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
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--ce-batches", type=int, default=1)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--natural-damping-ratio", type=float, default=0.1)
    parser.add_argument("--natural-cg-steps", type=int, default=8)
    parser.add_argument("--natural-trace-samples", type=int, default=1)
    parser.add_argument("--natural-trace-seed", type=int, default=260219134)
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
    if args.ce_batches <= 0 or args.ce_batches > args.batches:
        raise ValueError("ce-batches must be in [1, batches]")
    if args.muon_ns_steps <= 0:
        raise ValueError("muon-ns-steps must be positive")
    if (
        not math.isfinite(args.natural_damping_ratio)
        or args.natural_damping_ratio <= 0.0
    ):
        raise ValueError("natural-damping-ratio must be positive and finite")
    if args.natural_cg_steps <= 0:
        raise ValueError("natural-cg-steps must be positive")
    if args.natural_trace_samples <= 0:
        raise ValueError("natural-trace-samples must be positive")

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
        polar_by_split: dict[str, dict[str, torch.Tensor]] = {}
        teacher_by_split: dict[str, dict[str, torch.Tensor]] = {}
        natural_ce_by_split: dict[str, dict[str, torch.Tensor]] = {}
        natural_teacher_by_split: dict[str, dict[str, torch.Tensor]] = {}
        functional_natural_ce_by_split: dict[
            str, dict[str, torch.Tensor]
        ] = {}
        functional_natural_teacher_by_split: dict[
            str, dict[str, torch.Tensor]
        ] = {}
        init_key = f"{initialization:.8g}"
        objective[init_key] = {}
        for split in (fit, holdout):
            ce_batches = split.batches[: args.ce_batches]
            (
                ce_gradient,
                polar_gradient,
                ce_dense_gradient,
                ce_base_weights,
                ce_loss,
            ) = task_ce_gradients(
                model,
                parameters,
                ce_batches,
                args.device,
                layers,
                args.muon_ns_steps,
            )
            (
                teacher_gradient,
                teacher_losses,
                teacher_dense_gradient,
            ) = teacher_mse_gradients(
                model, parameters, split, layers, args.device
            )
            natural_ce, natural_ce_diagnostics = natural_chart_directions(
                model,
                parameters,
                layers,
                ce_dense_gradient,
                ce_base_weights,
                damping_ratio=args.natural_damping_ratio,
                cg_steps=args.natural_cg_steps,
                trace_samples=args.natural_trace_samples,
                trace_seed=args.natural_trace_seed,
            )
            natural_teacher, natural_teacher_diagnostics = (
                natural_chart_directions(
                    model,
                    parameters,
                    layers,
                    teacher_dense_gradient,
                    split.cproj_weight,
                    damping_ratio=args.natural_damping_ratio,
                    cg_steps=args.natural_cg_steps,
                    trace_samples=args.natural_trace_samples,
                    trace_seed=args.natural_trace_seed,
                )
            )
            metric_activations = {
                layer: F.gelu(split.pre_gelu[layer])
                for layer in layers
            }
            (
                functional_natural_ce,
                functional_natural_ce_diagnostics,
            ) = natural_chart_directions(
                model,
                parameters,
                layers,
                ce_dense_gradient,
                ce_base_weights,
                damping_ratio=args.natural_damping_ratio,
                cg_steps=args.natural_cg_steps,
                trace_samples=args.natural_trace_samples,
                trace_seed=args.natural_trace_seed,
                metric_activations=metric_activations,
            )
            (
                functional_natural_teacher,
                functional_natural_teacher_diagnostics,
            ) = natural_chart_directions(
                model,
                parameters,
                layers,
                teacher_dense_gradient,
                split.cproj_weight,
                damping_ratio=args.natural_damping_ratio,
                cg_steps=args.natural_cg_steps,
                trace_samples=args.natural_trace_samples,
                trace_seed=args.natural_trace_seed,
                metric_activations=metric_activations,
            )
            ce_by_split[split.name] = ce_gradient
            polar_by_split[split.name] = polar_gradient
            teacher_by_split[split.name] = teacher_gradient
            natural_ce_by_split[split.name] = natural_ce
            natural_teacher_by_split[split.name] = natural_teacher
            functional_natural_ce_by_split[split.name] = (
                functional_natural_ce
            )
            functional_natural_teacher_by_split[split.name] = (
                functional_natural_teacher
            )
            objective[init_key][split.name] = {
                "task_ce": ce_loss,
                "teacher_mse_by_layer": {
                    str(layer): teacher_losses[layer] for layer in layers
                },
                "natural_ce_diagnostics": {
                    str(layer): natural_ce_diagnostics[layer]
                    for layer in layers
                },
                "natural_teacher_diagnostics": {
                    str(layer): natural_teacher_diagnostics[layer]
                    for layer in layers
                },
                "functional_natural_ce_diagnostics": {
                    str(layer): functional_natural_ce_diagnostics[layer]
                    for layer in layers
                },
                "functional_natural_teacher_diagnostics": {
                    str(layer): (
                        functional_natural_teacher_diagnostics[layer]
                    )
                    for layer in layers
                },
            }
            split_rows = alignment_rows(
                ce_gradient,
                teacher_gradient,
                comparison="task_ce_vs_teacher_mse",
                split=split.name,
                initialization=initialization,
            )
            rows.extend(split_rows)
            polar_rows = alignment_rows(
                polar_gradient,
                teacher_gradient,
                comparison="task_ce_effective_polar_vjp_vs_teacher_mse",
                split=split.name,
                initialization=initialization,
            )
            rows.extend(polar_rows)
            natural_rows = alignment_rows(
                natural_ce,
                natural_teacher,
                comparison=(
                    "task_ce_natural_vs_teacher_mse_natural"
                ),
                split=split.name,
                initialization=initialization,
            )
            rows.extend(natural_rows)
            functional_natural_rows = alignment_rows(
                functional_natural_ce,
                functional_natural_teacher,
                comparison=(
                    "task_ce_functional_natural_vs_"
                    "teacher_mse_functional_natural"
                ),
                split=split.name,
                initialization=initialization,
            )
            rows.extend(functional_natural_rows)
            print(
                f"initialization={initialization} split={split.name} "
                f"task_ce={ce_loss:.6f} global_cosine="
                f"{split_rows[0]['cosine']:.6f} polar_vjp_cosine="
                f"{polar_rows[0]['cosine']:.6f} natural_cosine="
                f"{natural_rows[0]['cosine']:.6f} "
                "functional_natural_cosine="
                f"{functional_natural_rows[0]['cosine']:.6f}",
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
                polar_by_split["fit"],
                polar_by_split["holdout"],
                comparison="task_ce_effective_polar_vjp_fit_vs_holdout",
                split="cross_split",
                initialization=initialization,
            )
        )
        rows.extend(
            alignment_rows(
                natural_ce_by_split["fit"],
                natural_ce_by_split["holdout"],
                comparison="task_ce_natural_fit_vs_holdout",
                split="cross_split",
                initialization=initialization,
            )
        )
        rows.extend(
            alignment_rows(
                natural_teacher_by_split["fit"],
                natural_teacher_by_split["holdout"],
                comparison="teacher_mse_natural_fit_vs_holdout",
                split="cross_split",
                initialization=initialization,
            )
        )
        rows.extend(
            alignment_rows(
                functional_natural_ce_by_split["fit"],
                functional_natural_ce_by_split["holdout"],
                comparison="task_ce_functional_natural_fit_vs_holdout",
                split="cross_split",
                initialization=initialization,
            )
        )
        rows.extend(
            alignment_rows(
                functional_natural_teacher_by_split["fit"],
                functional_natural_teacher_by_split["holdout"],
                comparison=(
                    "teacher_mse_functional_natural_fit_vs_holdout"
                ),
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
        "schema_version": "mlp_chart_gradient_alignment_v3",
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
        "ce_batches": args.ce_batches,
        "sample_cap": args.sample_cap,
        "sample_seed": args.sample_seed,
        "holdout_sample_seed": args.holdout_sample_seed,
        "fit_token_sha256": fit.token_sha256,
        "holdout_token_sha256": holdout.token_sha256,
        "fit_ce_token_sha256": tensor_sha256(
            torch.cat(fit.batches[: args.ce_batches])
        ),
        "holdout_ce_token_sha256": tensor_sha256(
            torch.cat(holdout.batches[: args.ce_batches])
        ),
        "initial_effective_output_log_gains": args.initializations,
        "effective_polar_vjp": {
            "muon_ns_steps": args.muon_ns_steps,
            "base_gradient_unchanged": True,
            "chart_only": True,
            "dense_optimizer_state_retained": False,
        },
        "natural_inverse_pullback": {
            "metric": "effective_weight_frobenius_JtJ",
            "damping_ratio_to_estimated_mean_eigenvalue": (
                args.natural_damping_ratio
            ),
            "cg_steps": args.natural_cg_steps,
            "trace_samples": args.natural_trace_samples,
            "trace_seed": args.natural_trace_seed,
            "no_parameter_update": True,
        },
        "functional_natural_inverse_pullback": {
            "metric": "mean_post_gelu_output_mse_JtJ",
            "activations": "source_model_fixed_validation_post_gelu",
            "damping_ratio_to_estimated_mean_eigenvalue": (
                args.natural_damping_ratio
            ),
            "cg_steps": args.natural_cg_steps,
            "trace_samples": args.natural_trace_samples,
            "trace_seed": args.natural_trace_seed,
            "no_parameter_update": True,
        },
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
