"""Fit fixed-basis bilateral charts around a generated MLP ``c_proj``.

The paper-style trajectory analysis says that an optimizer path can be
low-dimensional even when every displacement along that path is a high-rank
matrix.  A hidden-channel diagonal chart captures radial motion, but it cannot
change the orientation of the generated ``c_proj`` in either its 3,072-wide
post-GELU input space or its 768-wide residual output space.

This diagnostic freezes a trained generated ``c_proj`` and fits compact,
identity-initialized charts on deterministic validation rows:

    pre-GELU activation chart -> hidden-side fixed-basis rotation
      -> frozen generated c_proj -> residual-side fixed-basis rotation

The learned bases are only Cayley coordinates inside fixed signed/permuted FHT
bases.  No dense basis, LoRA factor, or target weight is learned.  Fits use one
fixed token set and are scored on a disjoint fixed token set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    ActivationChart,
    collect_model,
    prediction_metrics,
    tensor_sha256,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.model import LearnedFHTBlockOrthogonalOutputMix


@dataclass(frozen=True)
class ChartSpec:
    input_stages: int
    output_stages: int
    activation_chart: bool

    @property
    def name(self) -> str:
        activation = "act" if self.activation_chart else "noact"
        return (
            f"in{self.input_stages}_out{self.output_stages}_{activation}"
        )


SPEC_PATTERN = re.compile(r"^in(\d+)_out(\d+)_(act|noact)$")


def parse_spec(value: str) -> ChartSpec:
    match = SPEC_PATTERN.fullmatch(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "chart spec must be inN_outN_act or inN_outN_noact"
        )
    input_stages = int(match.group(1))
    output_stages = int(match.group(2))
    if input_stages < 0 or output_stages < 0:
        raise argparse.ArgumentTypeError("stage counts cannot be negative")
    return ChartSpec(
        input_stages=input_stages,
        output_stages=output_stages,
        activation_chart=match.group(3) == "act",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BilateralCProjChart(torch.nn.Module):
    """Identity-initialized radial and bilateral orientation coordinates."""

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
        self.hidden_features = int(hidden_features)
        self.output_features = int(output_features)
        self.spec = spec
        self.activation_chart = (
            ActivationChart(
                self.hidden_features, "common_gauge_centered"
            )
            if spec.activation_chart
            else None
        )
        self.input_mixer = (
            LearnedFHTBlockOrthogonalOutputMix(
                features=self.hidden_features,
                stages=spec.input_stages,
                rotation_block_size=rotation_block_size,
                basis_block_size=basis_block_size,
                seed=int(seed),
            )
            if spec.input_stages
            else None
        )
        # A diagonal before the hidden rotation supplies the radial part of a
        # right-side matrix chart.  The activation chart already supplies the
        # corresponding post-GELU channel diagonal, so do not duplicate it.
        self.hidden_log_gain = (
            torch.nn.Parameter(torch.zeros(self.hidden_features))
            if spec.input_stages and not spec.activation_chart
            else None
        )
        self.output_mixer = (
            LearnedFHTBlockOrthogonalOutputMix(
                features=self.output_features,
                stages=spec.output_stages,
                rotation_block_size=rotation_block_size,
                basis_block_size=basis_block_size,
                seed=int(seed) + 1_000_003,
            )
            if spec.output_stages
            else None
        )
        self.output_log_gain = (
            torch.nn.Parameter(torch.zeros(self.output_features))
            if spec.output_stages
            else None
        )

    @staticmethod
    def _gain(values: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        return values * parameter.clamp(-3.0, 3.0).exp()

    def forward(
        self,
        pre_gelu: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.activation_chart is None:
            hidden = F.gelu(pre_gelu)
        else:
            pre_log, post_log = self.activation_chart.log_scales()
            hidden = F.gelu(self._gain(pre_gelu, pre_log))
            hidden = self._gain(hidden, post_log)
        if self.input_mixer is not None:
            if self.hidden_log_gain is not None:
                hidden = self._gain(hidden, self.hidden_log_gain)
            hidden = self.input_mixer(hidden)
        output = F.linear(hidden, weight, bias)
        if self.output_mixer is not None:
            assert self.output_log_gain is not None
            output = self.output_mixer(
                self._gain(output, self.output_log_gain)
            )
        return output

    def diagnostics(self) -> dict[str, float]:
        output: dict[str, float] = {
            "parameter_count": float(
                sum(parameter.numel() for parameter in self.parameters())
            )
        }
        if self.input_mixer is not None:
            output["input_coordinate_rms"] = float(
                self.input_mixer.coordinates.detach().square().mean().sqrt()
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
        if self.activation_chart is not None:
            pre_log, post_log = self.activation_chart.log_scales()
            output.update(
                {
                    "activation_pre_log_rms": float(
                        pre_log.detach().square().mean().sqrt()
                    ),
                    "activation_post_log_rms": float(
                        post_log.detach().square().mean().sqrt()
                    ),
                    "activation_common": float(
                        self.activation_chart.common.detach()
                    ),
                    "activation_gauge": float(
                        self.activation_chart.gauge.detach()
                    ),
                }
            )
        return output


def fit_spec(
    spec: ChartSpec,
    train_pre: torch.Tensor,
    train_target: torch.Tensor,
    holdout_pre: torch.Tensor,
    holdout_target: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    rotation_block_size: int,
    basis_block_size: int,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, float | str]:
    with torch.no_grad():
        train_identity = F.linear(F.gelu(train_pre), weight, bias)
        holdout_identity = F.linear(F.gelu(holdout_pre), weight, bias)
    if (
        spec.input_stages == 0
        and spec.output_stages == 0
        and not spec.activation_chart
    ):
        return {
            "family": spec.name,
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
                    holdout_target, holdout_identity, holdout_identity
                ).items()
            },
        }

    chart = BilateralCProjChart(
        hidden_features=train_pre.shape[-1],
        output_features=weight.shape[0],
        spec=spec,
        rotation_block_size=rotation_block_size,
        basis_block_size=basis_block_size,
        seed=seed,
    ).to(device=train_pre.device, dtype=torch.float32)
    optimizer = torch.optim.Adam(chart.parameters(), lr=float(learning_rate))
    generator = torch.Generator(device=train_pre.device)
    generator.manual_seed(int(seed) + 2_000_003)
    for _ in range(int(steps)):
        indices = torch.randint(
            train_pre.shape[0],
            (min(int(batch_size), train_pre.shape[0]),),
            generator=generator,
            device=train_pre.device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = chart(
            train_pre.index_select(0, indices), weight, bias
        )
        target = train_target.index_select(0, indices)
        loss = (prediction - target).square().mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_prediction = chart(train_pre, weight, bias)
        holdout_prediction = chart(holdout_pre, weight, bias)
        row: dict[str, float | str] = {
            "family": spec.name,
            **chart.diagnostics(),
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
                    holdout_target, holdout_prediction, holdout_identity
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
    rows: list[dict[str, object]]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        numeric_keys = {
            key
            for row in selected
            for key, value in row.items()
            if key not in {"layer", "family"}
            and isinstance(value, (int, float))
        }
        output[family] = {}
        for key in sorted(numeric_keys):
            finite = [
                float(row[key])
                for row in selected
                if key in row and np.isfinite(float(row[key]))
            ]
            if finite:
                output[family][key] = float(np.mean(finite))
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
            "in0_out0_noact,in0_out4_noact,"
            "in1_out0_noact,in2_out0_noact,in4_out0_noact,"
            "in1_out4_noact,in2_out4_noact,in4_out4_noact,"
            "in2_out4_act"
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
    target_train, _, _ = collect_model(
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
    print("collecting generated source activations and weights", flush=True)
    source_train, weights, biases = collect_model(
        args.plain_cproj,
        fit_batches,
        layers,
        args.sample_cap,
        args.device,
        collect_pre_gelu=True,
    )
    source_holdout, _, _ = collect_model(
        args.plain_cproj,
        holdout_batches,
        layers,
        args.sample_cap,
        args.device,
        collect_pre_gelu=True,
    )

    rows: list[dict[str, object]] = []
    for layer in layers:
        train_pre = source_train[(layer, "pre_gelu")].to(args.device)
        holdout_pre = source_holdout[(layer, "pre_gelu")].to(args.device)
        train_target = target_train[(layer, "mlp_out")].to(args.device)
        holdout_target = target_holdout[(layer, "mlp_out")].to(args.device)
        weight = weights[layer].to(args.device)
        bias = (
            biases[layer].to(args.device)
            if biases[layer] is not None
            else None
        )
        with torch.no_grad():
            reconstructed = F.linear(F.gelu(train_pre), weight, bias)
            source_output = source_train[(layer, "mlp_out")].to(args.device)
            reconstruction_rms = float(
                (reconstructed - source_output).square().mean().sqrt()
            )
        print(
            f"layer={layer} source_reconstruction_rms="
            f"{reconstruction_rms:.3e}",
            flush=True,
        )
        for spec_index, spec in enumerate(specs):
            print(f"fitting layer={layer} family={spec.name}", flush=True)
            row = fit_spec(
                spec,
                train_pre,
                train_target,
                holdout_pre,
                holdout_target,
                weight,
                bias,
                args.rotation_block_size,
                args.basis_block_size,
                args.fit_seed + layer * 100 + spec_index,
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
        del train_pre, holdout_pre, train_target, holdout_target, weight
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "cproj_bilateral_oracle.csv"
    write_csv(csv_path, rows)
    metadata = {
        "schema_version": "cproj_bilateral_oracle_v1",
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
    metadata_path = args.output / "cproj_bilateral_oracle_summary.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
