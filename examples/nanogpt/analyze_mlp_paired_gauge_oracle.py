"""Fit held-out fixed-basis gauge charts across the complete MLP pair.

The prior bilateral oracle rotates only post-GELU activations before a frozen
generated ``c_proj``. That is an endpoint correction, but it is not the hidden
coordinate chart of the nonlinear MLP itself. The paired chart tested here
uses one fixed-basis orthogonal operator on both sides of GELU:

    pre = R^-1 (c_fc(x))
    output = c_proj(R GELU(pre))

In the linear-activation limit the two applications cancel exactly. GELU is
the only place where this identity-initialized gauge chart changes the
function, so the diagnostic directly tests the observed activation-spectrum
failure. Only Cayley coordinates in fixed signed/permuted FHT bases and
optional channel gains are learned; no dense basis or LoRA factor is fitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    prediction_metrics,
    prepare_inference_cache,
    tensor_sha256,
)
from examples.nanogpt.analyze_cproj_manifold import (
    base_c_proj_weight,
    load_model,
    spectral_residual_weight,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.model import LearnedFHTBlockOrthogonalOutputMix


@dataclass(frozen=True)
class ChartSpec:
    mode: str
    hidden_stages: int
    output_stages: int

    @property
    def name(self) -> str:
        if self.mode == "identity":
            return "identity"
        return f"{self.mode}{self.hidden_stages}_out{self.output_stages}"


SPEC_PATTERN = re.compile(r"^(post|paired)(\d+)_out(\d+)$")


def parse_spec(value: str) -> ChartSpec:
    value = value.strip()
    if value == "identity":
        return ChartSpec("identity", 0, 0)
    match = SPEC_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "chart spec must be identity, postN_outN, or pairedN_outN"
        )
    hidden_stages = int(match.group(2))
    output_stages = int(match.group(3))
    if hidden_stages <= 0:
        raise argparse.ArgumentTypeError(
            "post/paired charts require at least one hidden stage"
        )
    return ChartSpec(match.group(1), hidden_stages, output_stages)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MLPInputCollector:
    """Collect aligned source MLP inputs and complete MLP outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        layers: list[int],
        sample_cap: int,
        collect_inputs: bool,
    ) -> None:
        self.layers = set(layers)
        self.sample_cap = int(sample_cap)
        self.collect_inputs = bool(collect_inputs)
        self.values: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(
            list
        )
        self.counts: dict[tuple[int, str], int] = defaultdict(int)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for index, block in enumerate(model.transformer.h):
            if index not in self.layers:
                continue
            if self.collect_inputs:
                self.handles.append(
                    block.mlp.register_forward_pre_hook(
                        self._input_hook(index)
                    )
                )
            self.handles.append(
                block.mlp.register_forward_hook(self._output_hook(index))
            )

    @property
    def points(self) -> tuple[str, ...]:
        return ("mlp_input", "mlp_out") if self.collect_inputs else ("mlp_out",)

    def _append(self, layer: int, point: str, values: torch.Tensor) -> None:
        key = (layer, point)
        remaining = self.sample_cap - self.counts[key]
        if remaining <= 0:
            return
        rows = values.detach().float().reshape(-1, values.shape[-1])
        rows = rows[:remaining].cpu()
        self.values[key].append(rows)
        self.counts[key] += int(rows.shape[0])

    def _input_hook(self, layer: int):
        def hook(_module, inputs):
            self._append(layer, "mlp_input", inputs[0])

        return hook

    def _output_hook(self, layer: int):
        def hook(_module, _inputs, output):
            self._append(layer, "mlp_out", output)

        return hook

    def complete(self) -> bool:
        return all(
            self.counts[(layer, point)] >= self.sample_cap
            for layer in self.layers
            for point in self.points
        )

    def tensors(self) -> dict[tuple[int, str], torch.Tensor]:
        return {
            (layer, point): torch.cat(
                self.values[(layer, point)], dim=0
            )[: self.sample_cap]
            for layer in self.layers
            for point in self.points
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_model(
    checkpoint: Path,
    batches: list[torch.Tensor],
    layers: list[int],
    sample_cap: int,
    device: str,
    collect_inputs_and_weights: bool,
) -> tuple[
    dict[tuple[int, str], torch.Tensor],
    dict[int, dict[str, torch.Tensor | None]],
]:
    model = load_model(checkpoint, device)
    prepare_inference_cache(model)
    collector = MLPInputCollector(
        model,
        layers,
        sample_cap,
        collect_inputs_and_weights,
    )
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch.to(device), None)
                if collector.complete():
                    break
        weights: dict[int, dict[str, torch.Tensor | None]] = {}
        if collect_inputs_and_weights:
            for layer in layers:
                mlp = model.transformer.h[layer].mlp
                cproj_weight = base_c_proj_weight(model, layer)
                spectral = spectral_residual_weight(model, layer)
                if spectral is not None:
                    cproj_weight = cproj_weight + spectral
                if mlp.has_charted_cproj():
                    cproj_weight = mlp._materialize_charted_cproj_weight(
                        cproj_weight
                    )
                if mlp.output_rotation is not None:
                    raise ValueError(
                        "post-c_proj output_rotation is unsupported"
                    )
                weights[layer] = {
                    "c_fc_weight": mlp.c_fc.weight.detach().float().cpu(),
                    "c_fc_bias": (
                        mlp.c_fc.bias.detach().float().cpu()
                        if mlp.c_fc.bias is not None
                        else None
                    ),
                    "c_proj_weight": cproj_weight.detach().float().cpu(),
                    "c_proj_bias": (
                        mlp.c_proj.bias.detach().float().cpu()
                        if mlp.c_proj.bias is not None
                        else None
                    ),
                }
        return collector.tensors(), weights
    finally:
        collector.close()
        del model
        if "cuda" in device:
            torch.cuda.empty_cache()


class PairedGaugeChart(torch.nn.Module):
    """Fixed-basis post-only or GELU-paired hidden orientation chart."""

    def __init__(
        self,
        hidden_features: int,
        output_features: int,
        spec: ChartSpec,
        rotation_block_size: int,
        basis_block_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.hidden_mixer = (
            LearnedFHTBlockOrthogonalOutputMix(
                hidden_features,
                spec.hidden_stages,
                rotation_block_size,
                basis_block_size,
                seed,
            )
            if spec.hidden_stages
            else None
        )
        self.hidden_log_gain = (
            torch.nn.Parameter(torch.zeros(hidden_features))
            if spec.hidden_stages
            else None
        )
        self.output_mixer = (
            LearnedFHTBlockOrthogonalOutputMix(
                output_features,
                spec.output_stages,
                rotation_block_size,
                basis_block_size,
                seed + 1_000_003,
            )
            if spec.output_stages
            else None
        )
        self.output_log_gain = (
            torch.nn.Parameter(torch.zeros(output_features))
            if spec.output_stages
            else None
        )

    @staticmethod
    def _gain(values: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        return values * parameter.clamp(-3.0, 3.0).exp()

    def forward(
        self,
        mlp_input: torch.Tensor,
        c_fc_weight: torch.Tensor,
        c_fc_bias: torch.Tensor | None,
        c_proj_weight: torch.Tensor,
        c_proj_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        pre_gelu = F.linear(mlp_input, c_fc_weight, c_fc_bias)
        if self.spec.mode == "paired":
            assert self.hidden_mixer is not None
            pre_gelu = self.hidden_mixer.inverse(pre_gelu)
        hidden = F.gelu(pre_gelu)
        if self.hidden_mixer is not None:
            hidden = self.hidden_mixer(hidden)
            assert self.hidden_log_gain is not None
            hidden = self._gain(hidden, self.hidden_log_gain)
        output = F.linear(hidden, c_proj_weight, c_proj_bias)
        if self.output_mixer is not None:
            assert self.output_log_gain is not None
            output = self.output_mixer(
                self._gain(output, self.output_log_gain)
            )
        return output

    def diagnostics(self) -> dict[str, float]:
        output = {
            "parameter_count": float(
                sum(parameter.numel() for parameter in self.parameters())
            )
        }
        if self.hidden_mixer is not None:
            output["hidden_coordinate_rms"] = float(
                self.hidden_mixer.coordinates.detach().square().mean().sqrt()
            )
        if self.hidden_log_gain is not None:
            output["hidden_log_gain_rms"] = float(
                self.hidden_log_gain.detach().square().mean().sqrt()
            )
        if self.output_mixer is not None:
            output["output_coordinate_rms"] = float(
                self.output_mixer.coordinates.detach().square().mean().sqrt()
            )
        if self.output_log_gain is not None:
            output["output_log_gain_rms"] = float(
                self.output_log_gain.detach().square().mean().sqrt()
            )
        return output


def source_prediction(
    mlp_input: torch.Tensor,
    c_fc_weight: torch.Tensor,
    c_fc_bias: torch.Tensor | None,
    c_proj_weight: torch.Tensor,
    c_proj_bias: torch.Tensor | None,
) -> torch.Tensor:
    return F.linear(
        F.gelu(F.linear(mlp_input, c_fc_weight, c_fc_bias)),
        c_proj_weight,
        c_proj_bias,
    )


def fit_spec(
    spec: ChartSpec,
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    holdout_input: torch.Tensor,
    holdout_target: torch.Tensor,
    weights: dict[str, torch.Tensor | None],
    rotation_block_size: int,
    basis_block_size: int,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, float | str]:
    c_fc_weight = weights["c_fc_weight"]
    c_fc_bias = weights["c_fc_bias"]
    c_proj_weight = weights["c_proj_weight"]
    c_proj_bias = weights["c_proj_bias"]
    assert c_fc_weight is not None and c_proj_weight is not None
    with torch.no_grad():
        train_identity = source_prediction(
            train_input,
            c_fc_weight,
            c_fc_bias,
            c_proj_weight,
            c_proj_bias,
        )
        holdout_identity = source_prediction(
            holdout_input,
            c_fc_weight,
            c_fc_bias,
            c_proj_weight,
            c_proj_bias,
        )
    if spec.mode == "identity":
        return {
            "family": spec.name,
            "parameter_count": 0.0,
            **{
                f"train_{key}": value
                for key, value in prediction_metrics(
                    train_target,
                    train_identity,
                    train_identity,
                ).items()
            },
            **{
                f"holdout_{key}": value
                for key, value in prediction_metrics(
                    holdout_target,
                    holdout_identity,
                    holdout_identity,
                ).items()
            },
        }

    chart = PairedGaugeChart(
        hidden_features=c_fc_weight.shape[0],
        output_features=c_proj_weight.shape[0],
        spec=spec,
        rotation_block_size=rotation_block_size,
        basis_block_size=basis_block_size,
        seed=seed,
    ).to(device=train_input.device, dtype=torch.float32)
    optimizer = torch.optim.Adam(chart.parameters(), lr=learning_rate)
    generator = torch.Generator(device=train_input.device)
    generator.manual_seed(seed + 2_000_003)
    for _ in range(steps):
        indices = torch.randint(
            train_input.shape[0],
            (min(batch_size, train_input.shape[0]),),
            generator=generator,
            device=train_input.device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = chart(
            train_input.index_select(0, indices),
            c_fc_weight,
            c_fc_bias,
            c_proj_weight,
            c_proj_bias,
        )
        loss = (
            prediction - train_target.index_select(0, indices)
        ).square().mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_prediction = chart(
            train_input,
            c_fc_weight,
            c_fc_bias,
            c_proj_weight,
            c_proj_bias,
        )
        holdout_prediction = chart(
            holdout_input,
            c_fc_weight,
            c_fc_bias,
            c_proj_weight,
            c_proj_bias,
        )
        row: dict[str, float | str] = {
            "family": spec.name,
            **chart.diagnostics(),
        }
        row.update(
            {
                f"train_{key}": value
                for key, value in prediction_metrics(
                    train_target,
                    train_prediction,
                    train_identity,
                ).items()
            }
        )
        row.update(
            {
                f"holdout_{key}": value
                for key, value in prediction_metrics(
                    holdout_target,
                    holdout_prediction,
                    holdout_identity,
                ).items()
            }
        )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        keys = {
            key
            for row in selected
            for key, value in row.items()
            if key not in {"layer", "family"}
            and isinstance(value, (int, float))
        }
        output[family] = {}
        for key in sorted(keys):
            values = [
                float(row[key])
                for row in selected
                if key in row and np.isfinite(float(row[key]))
            ]
            if values:
                output[family][key] = float(np.mean(values))
    return output


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
        "--specs",
        default=(
            "identity,post2_out0,paired2_out0,"
            "post1_out4,post2_out4,"
            "paired1_out4,paired2_out4,paired4_out4"
        ),
    )
    parser.add_argument("--rotation-block-size", type=int, default=32)
    parser.add_argument("--basis-block-size", type=int, default=256)
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--fit-batch-size", type=int, default=256)
    parser.add_argument("--fit-learning-rate", type=float, default=0.02)
    parser.add_argument("--fit-seed", type=int, default=314159)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",") if part]
    specs = [parse_spec(part) for part in args.specs.split(",") if part]
    fit_batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        args.batches,
        args.sample_seed,
    )
    holdout_batches = fixed_validation_batches(
        args.data_dir,
        args.batch_size,
        args.block_size,
        args.batches,
        args.holdout_sample_seed,
    )
    fit_token_digest = tensor_sha256(torch.cat(fit_batches))
    holdout_token_digest = tensor_sha256(torch.cat(holdout_batches))

    print("collecting target MLP outputs", flush=True)
    target_train, _ = collect_model(
        args.attention_only,
        fit_batches,
        layers,
        args.sample_cap,
        args.device,
        False,
    )
    target_holdout, _ = collect_model(
        args.attention_only,
        holdout_batches,
        layers,
        args.sample_cap,
        args.device,
        False,
    )
    print("collecting source MLP inputs and effective weights", flush=True)
    source_train, weights = collect_model(
        args.plain_cproj,
        fit_batches,
        layers,
        args.sample_cap,
        args.device,
        True,
    )
    source_holdout, _ = collect_model(
        args.plain_cproj,
        holdout_batches,
        layers,
        args.sample_cap,
        args.device,
        True,
    )

    rows: list[dict[str, object]] = []
    for layer in layers:
        train_input = source_train[(layer, "mlp_input")].to(args.device)
        holdout_input = source_holdout[(layer, "mlp_input")].to(args.device)
        train_target = target_train[(layer, "mlp_out")].to(args.device)
        holdout_target = target_holdout[(layer, "mlp_out")].to(args.device)
        layer_weights = {
            key: value.to(args.device) if value is not None else None
            for key, value in weights[layer].items()
        }
        with torch.no_grad():
            reconstructed = source_prediction(
                train_input,
                layer_weights["c_fc_weight"],
                layer_weights["c_fc_bias"],
                layer_weights["c_proj_weight"],
                layer_weights["c_proj_bias"],
            )
            source_output = source_train[(layer, "mlp_out")].to(args.device)
            reconstruction_rms = float(
                (reconstructed - source_output).square().mean().sqrt()
            )
        print(
            f"layer={layer} source_reconstruction_rms="
            f"{reconstruction_rms:.3e}",
            flush=True,
        )
        for spec in specs:
            print(f"fitting layer={layer} family={spec.name}", flush=True)
            row = fit_spec(
                spec,
                train_input,
                train_target,
                holdout_input,
                holdout_target,
                layer_weights,
                args.rotation_block_size,
                args.basis_block_size,
                args.fit_seed + layer * 100,
                args.fit_steps,
                args.fit_batch_size,
                args.fit_learning_rate,
            )
            rows.append(
                {
                    "layer": layer,
                    "source_reconstruction_rms": reconstruction_rms,
                    **row,
                }
            )
        del train_input, holdout_input, train_target, holdout_target
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "mlp_paired_gauge_oracle.csv"
    write_csv(csv_path, rows)
    metadata = {
        "schema_version": "mlp_paired_gauge_oracle_v1",
        "scientific_scope": (
            "fixed-fit/fixed-heldout structural oracle; not a training result"
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
        "fit_token_sha256": fit_token_digest,
        "holdout_token_sha256": holdout_token_digest,
        "layers": layers,
        "sample_cap": args.sample_cap,
        "sample_seed": args.sample_seed,
        "holdout_sample_seed": args.holdout_sample_seed,
        "specs": [spec.name for spec in specs],
        "rotation_block_size": args.rotation_block_size,
        "basis_block_size": args.basis_block_size,
        "fit_steps": args.fit_steps,
        "fit_batch_size": args.fit_batch_size,
        "fit_learning_rate": args.fit_learning_rate,
        "fit_seed": args.fit_seed,
        "source_sha256": sha256(Path(__file__)),
        "csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        "summary": summarize(rows),
    }
    metadata_path = args.output / "mlp_paired_gauge_oracle_summary.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
