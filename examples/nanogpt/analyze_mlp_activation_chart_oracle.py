"""Fit activation-aware MLP manifold charts on fixed held-out tokens.

The dense-Muon trajectory probe finds two different kinds of paired
``c_fc``-row / ``c_proj``-column motion:

* a layerwise mean gauge motion (``c_fc`` shrinks while ``c_proj`` grows);
* centered, same-sign hidden-channel motion shared by the two matrices.

This diagnostic asks whether those coordinates repair a generated ``c_proj``
functionally.  It keeps the source model's pre-GELU activations and effective
``c_proj`` weight fixed, fits compact log-scale charts on one deterministic
set of validation windows, and evaluates them on a disjoint deterministic
set.  It is an oracle/structure-selection probe, not a training result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_manifold import (
    base_c_proj_weight,
    load_model,
    spectral_residual_weight,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)


FAMILIES = (
    "identity",
    "global_common",
    "global_common_gauge",
    "centered_common",
    "common_gauge_centered",
    "independent_channels",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    array = values.detach().cpu().contiguous().numpy()
    digest.update(memoryview(array))
    return digest.hexdigest()


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be NAME=CHECKPOINT")
    name, checkpoint = value.split("=", 1)
    if not name or not checkpoint:
        raise argparse.ArgumentTypeError("source must be NAME=CHECKPOINT")
    return name, Path(checkpoint)


class ActivationCollector:
    """Collect aligned pre-GELU and complete MLP output rows."""

    def __init__(
        self,
        model: torch.nn.Module,
        layers: list[int],
        sample_cap: int,
        collect_pre_gelu: bool,
    ) -> None:
        self.layers = set(layers)
        self.sample_cap = int(sample_cap)
        self.collect_pre_gelu = bool(collect_pre_gelu)
        self.values: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(
            list
        )
        self.counts: dict[tuple[int, str], int] = defaultdict(int)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for index, block in enumerate(model.transformer.h):
            if index not in self.layers:
                continue
            if self.collect_pre_gelu:
                self.handles.append(
                    block.mlp.c_fc.register_forward_hook(
                        self._hook(index, "pre_gelu")
                    )
                )
            self.handles.append(
                block.mlp.register_forward_hook(
                    self._hook(index, "mlp_out")
                )
            )

    @property
    def points(self) -> tuple[str, ...]:
        if self.collect_pre_gelu:
            return ("pre_gelu", "mlp_out")
        return ("mlp_out",)

    def _hook(self, layer: int, point: str):
        def hook(_module, _inputs, output):
            key = (layer, point)
            remaining = self.sample_cap - self.counts[key]
            if remaining <= 0:
                return
            rows = output.detach().float().reshape(-1, output.shape[-1])
            rows = rows[:remaining].cpu()
            self.values[key].append(rows)
            self.counts[key] += int(rows.shape[0])

        return hook

    def complete(self) -> bool:
        return all(
            self.counts[(layer, point)] >= self.sample_cap
            for layer in self.layers
            for point in self.points
        )

    def tensor(self, layer: int, point: str) -> torch.Tensor:
        return torch.cat(self.values[(layer, point)], dim=0)[
            : self.sample_cap
        ]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def prepare_inference_cache(model: torch.nn.Module) -> None:
    if model.config.block_fht:
        model.prepare_block_fht_cache(dtype=next(model.parameters()).dtype)
        model.prepare_charted_cproj_cache()


def collect_model(
    checkpoint: Path,
    batches: list[torch.Tensor],
    layers: list[int],
    sample_cap: int,
    device: str,
    collect_pre_gelu: bool,
) -> tuple[
    dict[tuple[int, str], torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor | None],
]:
    model = load_model(checkpoint, device)
    prepare_inference_cache(model)
    collector = ActivationCollector(
        model, layers, sample_cap, collect_pre_gelu
    )
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch.to(device), None)
                if collector.complete():
                    break
        values = {
            (layer, point): collector.tensor(layer, point)
            for layer in layers
            for point in collector.points
        }
        weights: dict[int, torch.Tensor] = {}
        biases: dict[int, torch.Tensor | None] = {}
        if collect_pre_gelu:
            for layer in layers:
                mlp = model.transformer.h[layer].mlp
                weight = base_c_proj_weight(model, layer)
                spectral = spectral_residual_weight(model, layer)
                if spectral is not None:
                    weight = weight + spectral
                if (
                    mlp.output_block_rotation is not None
                    or mlp.residual_output_log_gain is not None
                ):
                    weight = mlp._materialize_charted_cproj_weight(weight)
                if mlp.output_rotation is not None:
                    raise ValueError(
                        "post-c_proj output_rotation is unsupported by this "
                        "activation-chart oracle"
                    )
                bias = getattr(mlp.c_proj, "bias", None)
                if bias is not None and (
                    mlp.output_block_rotation is not None
                    or mlp.residual_output_log_gain is not None
                ):
                    raise ValueError(
                        "biased charted c_proj is unsupported by this oracle"
                    )
                weights[layer] = weight.detach().float().cpu()
                biases[layer] = (
                    bias.detach().float().cpu() if bias is not None else None
                )
        return values, weights, biases
    finally:
        collector.close()
        del model
        if "cuda" in device:
            torch.cuda.empty_cache()


class ActivationChart(torch.nn.Module):
    """Log-scale chart for paired hidden-channel coordinates."""

    def __init__(self, features: int, family: str) -> None:
        super().__init__()
        if family not in FAMILIES or family == "identity":
            raise ValueError(f"family {family!r} is not trainable")
        self.features = int(features)
        self.family = family
        if family in {"global_common", "global_common_gauge"}:
            self.common = torch.nn.Parameter(torch.zeros(()))
        elif family == "common_gauge_centered":
            self.common = torch.nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("common", None)
        if family in {"global_common_gauge", "common_gauge_centered"}:
            self.gauge = torch.nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("gauge", None)
        if family in {"centered_common", "common_gauge_centered"}:
            self.centered = torch.nn.Parameter(torch.zeros(self.features))
        else:
            self.register_parameter("centered", None)
        if family == "independent_channels":
            self.pre_log_scale = torch.nn.Parameter(
                torch.zeros(self.features)
            )
            self.post_log_scale = torch.nn.Parameter(
                torch.zeros(self.features)
            )
        else:
            self.register_parameter("pre_log_scale", None)
            self.register_parameter("post_log_scale", None)

    def log_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.family == "independent_channels":
            assert self.pre_log_scale is not None
            assert self.post_log_scale is not None
            return self.pre_log_scale, self.post_log_scale
        reference = next(self.parameters())
        common = self.common if self.common is not None else reference.new_zeros(())
        gauge = self.gauge if self.gauge is not None else reference.new_zeros(())
        centered = (
            self.centered - self.centered.mean()
            if self.centered is not None
            else reference.new_zeros(self.features)
        )
        return common - gauge + centered, common + gauge + centered

    def forward(
        self,
        pre_gelu: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        pre_log, post_log = self.log_scales()
        hidden = pre_gelu * pre_log.clamp(-3.0, 3.0).exp()
        activated = F.gelu(hidden)
        activated = activated * post_log.clamp(-3.0, 3.0).exp()
        return F.linear(activated, weight, bias)


def prediction_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    identity_prediction: torch.Tensor,
) -> dict[str, float]:
    target = target.float()
    prediction = prediction.float()
    identity_prediction = identity_prediction.float()
    residual = target - prediction
    identity_residual = target - identity_prediction
    target_energy = target.square().sum().clamp_min(1e-30)
    identity_error = identity_residual.square().sum().clamp_min(1e-30)
    denominator = target.norm() * prediction.norm()
    return {
        "explained_target_energy": float(
            1.0 - residual.square().sum() / target_energy
        ),
        "identity_error_recovery": float(
            1.0 - residual.square().sum() / identity_error
        ),
        "cosine": float(
            (target * prediction).sum() / denominator.clamp_min(1e-30)
        ),
        "target_rms": float(target.square().mean().sqrt()),
        "prediction_rms": float(prediction.square().mean().sqrt()),
        "residual_rms": float(residual.square().mean().sqrt()),
        "identity_residual_rms": float(
            identity_residual.square().mean().sqrt()
        ),
    }


def fit_family(
    family: str,
    train_pre: torch.Tensor,
    train_target: torch.Tensor,
    holdout_pre: torch.Tensor,
    holdout_target: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float | str]:
    with torch.no_grad():
        train_identity = F.linear(F.gelu(train_pre), weight, bias)
        holdout_identity = F.linear(F.gelu(holdout_pre), weight, bias)
    if family == "identity":
        return {
            "family": family,
            "parameter_count": 0.0,
            **{
                f"train_{key}": value
                for key, value in prediction_metrics(
                    train_target, train_identity, train_identity
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

    chart = ActivationChart(train_pre.shape[-1], family).to(
        device=train_pre.device, dtype=torch.float32
    )
    optimizer = torch.optim.Adam(chart.parameters(), lr=float(learning_rate))
    generator = torch.Generator(device=train_pre.device)
    generator.manual_seed(int(seed))
    for _ in range(int(steps)):
        indices = torch.randint(
            train_pre.shape[0],
            (min(int(batch_size), train_pre.shape[0]),),
            generator=generator,
            device=train_pre.device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = chart(
            train_pre.index_select(0, indices),
            weight,
            bias,
        )
        target = train_target.index_select(0, indices)
        loss = (prediction - target).square().mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_prediction = chart(train_pre, weight, bias)
        holdout_prediction = chart(holdout_pre, weight, bias)
        pre_log, post_log = chart.log_scales()
        row: dict[str, float | str] = {
            "family": family,
            "parameter_count": float(
                sum(parameter.numel() for parameter in chart.parameters())
            ),
            "pre_log_mean": float(pre_log.mean()),
            "pre_log_std": float(pre_log.std()),
            "post_log_mean": float(post_log.mean()),
            "post_log_std": float(post_log.std()),
            "pre_post_log_correlation": float(
                torch.corrcoef(
                    torch.stack((pre_log.reshape(-1), post_log.reshape(-1)))
                )[0, 1]
            )
            if pre_log.numel() > 1
            else float("nan"),
        }
        row.update(
            {
                f"train_{key}": value
                for key, value in prediction_metrics(
                    train_target, train_prediction, train_identity
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


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for source in sorted({str(row["source"]) for row in rows}):
        for family in FAMILIES:
            selected = [
                row
                for row in rows
                if row["source"] == source and row["family"] == family
            ]
            if not selected:
                continue
            summary[f"{source}/{family}"] = {
                key: float(
                    np.nanmean([float(row[key]) for row in selected])
                )
                for key, value in selected[0].items()
                if key not in {"source", "family", "layer"}
                and isinstance(value, (int, float))
            }
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", required=True, type=Path)
    parser.add_argument(
        "--source", action="append", required=True, type=parse_source
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--sample-cap", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=20260716)
    parser.add_argument("--holdout-sample-seed", type=int, default=20260717)
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--fit-batch-size", type=int, default=256)
    parser.add_argument("--fit-learning-rate", type=float, default=0.03)
    parser.add_argument("--fit-seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",") if part]
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
    fit_token_digest = tensor_sha256(torch.stack(fit_batches))
    holdout_token_digest = tensor_sha256(torch.stack(holdout_batches))

    print("collecting attention-only target outputs", flush=True)
    target_fit, _, _ = collect_model(
        args.attention_only,
        fit_batches,
        layers,
        args.sample_cap,
        args.device,
        collect_pre_gelu=False,
    )
    target_holdout, _, _ = collect_model(
        args.attention_only,
        holdout_batches,
        layers,
        args.sample_cap,
        args.device,
        collect_pre_gelu=False,
    )

    rows: list[dict[str, object]] = []
    source_metadata: dict[str, dict[str, str]] = {}
    for source_index, (source_name, source_checkpoint) in enumerate(
        args.source
    ):
        print(f"collecting source={source_name} activations", flush=True)
        source_fit, weights, biases = collect_model(
            source_checkpoint,
            fit_batches,
            layers,
            args.sample_cap,
            args.device,
            collect_pre_gelu=True,
        )
        source_holdout, _, _ = collect_model(
            source_checkpoint,
            holdout_batches,
            layers,
            args.sample_cap,
            args.device,
            collect_pre_gelu=True,
        )
        source_metadata[source_name] = {
            "path": str(source_checkpoint),
            "sha256": sha256(source_checkpoint),
        }
        for layer in layers:
            train_pre = source_fit[(layer, "pre_gelu")].to(args.device)
            holdout_pre = source_holdout[(layer, "pre_gelu")].to(args.device)
            train_target = target_fit[(layer, "mlp_out")].to(args.device)
            holdout_target = target_holdout[
                (layer, "mlp_out")
            ].to(args.device)
            weight = weights[layer].to(args.device)
            bias = (
                biases[layer].to(args.device)
                if biases[layer] is not None
                else None
            )
            source_output = source_fit[(layer, "mlp_out")].to(args.device)
            with torch.no_grad():
                reconstructed = F.linear(F.gelu(train_pre), weight, bias)
                reconstruction_rms = float(
                    (reconstructed - source_output)
                    .square()
                    .mean()
                    .sqrt()
                )
            print(
                f"source={source_name} layer={layer} "
                f"reconstruction_rms={reconstruction_rms:.3e}",
                flush=True,
            )
            for family_index, family in enumerate(FAMILIES):
                print(
                    f"fitting source={source_name} layer={layer} "
                    f"family={family}",
                    flush=True,
                )
                row = fit_family(
                    family,
                    train_pre,
                    train_target,
                    holdout_pre,
                    holdout_target,
                    weight,
                    bias,
                    args.fit_steps,
                    args.fit_batch_size,
                    args.fit_learning_rate,
                    (
                        args.fit_seed
                        + source_index * 10_000
                        + layer * 100
                        + family_index
                    ),
                )
                rows.append(
                    {
                        "source": source_name,
                        "layer": layer,
                        "source_reconstruction_rms": reconstruction_rms,
                        **row,
                    }
                )
            del train_pre, holdout_pre, train_target, holdout_target, weight
            if "cuda" in args.device:
                torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "mlp_activation_chart_oracle.csv"
    write_csv(csv_path, rows)
    metadata = {
        "schema_version": "mlp_activation_chart_oracle_v1",
        "scientific_scope": (
            "fixed-fit/fixed-heldout structural oracle; not a training result"
        ),
        "attention_only": {
            "path": str(args.attention_only),
            "sha256": sha256(args.attention_only),
        },
        "sources": source_metadata,
        "data_dir": str(args.data_dir),
        "fit_token_sha256": fit_token_digest,
        "holdout_token_sha256": holdout_token_digest,
        "layers": layers,
        "sample_cap": args.sample_cap,
        "sample_seed": args.sample_seed,
        "holdout_sample_seed": args.holdout_sample_seed,
        "fit_steps": args.fit_steps,
        "fit_batch_size": args.fit_batch_size,
        "fit_learning_rate": args.fit_learning_rate,
        "fit_seed": args.fit_seed,
        "source_sha256": sha256(Path(__file__)),
        "csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        "summary": summarize(rows),
    }
    metadata_path = args.output / "mlp_activation_chart_oracle_summary.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
