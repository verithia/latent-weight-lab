"""Fit independent fixed-basis MLP charts before and after GELU.

The inverse-tied gauge chart is deliberately restrictive: it assumes hidden
orientation should cancel in the linear limit. Dense trajectory measurements
instead show mostly same-sign/common ``c_fc`` row and ``c_proj`` column
motion. This held-out oracle therefore gives the pre-GELU and post-GELU sides
separate Cayley coordinates while retaining fixed signed/permuted FHT bases.
No dense learned basis, target weight, or LoRA factor is fitted.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    prediction_metrics,
    tensor_sha256,
)
from examples.nanogpt.analyze_mlp_paired_gauge_oracle import (
    collect_model,
    sha256,
    source_prediction,
    summarize,
    write_csv,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.model import LearnedFHTBlockOrthogonalOutputMix


@dataclass(frozen=True)
class ChartSpec:
    pre_stages: int
    post_stages: int
    output_stages: int

    @property
    def name(self) -> str:
        if not (self.pre_stages or self.post_stages or self.output_stages):
            return "identity"
        return (
            f"pre{self.pre_stages}_post{self.post_stages}_"
            f"out{self.output_stages}"
        )


SPEC_PATTERN = re.compile(r"^pre(\d+)_post(\d+)_out(\d+)$")


def parse_spec(value: str) -> ChartSpec:
    value = value.strip()
    if value == "identity":
        return ChartSpec(0, 0, 0)
    match = SPEC_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "chart spec must be identity or preN_postN_outN"
        )
    spec = ChartSpec(*(int(match.group(index)) for index in (1, 2, 3)))
    if not (spec.pre_stages or spec.post_stages or spec.output_stages):
        raise argparse.ArgumentTypeError("use identity for the zero-stage chart")
    return spec


class IndependentHiddenChart(torch.nn.Module):
    """Independent identity-initialized charts around GELU and c_proj."""

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
        self.pre_mixer = (
            LearnedFHTBlockOrthogonalOutputMix(
                hidden_features,
                spec.pre_stages,
                rotation_block_size,
                basis_block_size,
                seed + 500_009,
            )
            if spec.pre_stages
            else None
        )
        self.pre_log_gain = (
            torch.nn.Parameter(torch.zeros(hidden_features))
            if spec.pre_stages
            else None
        )
        # Keep this seed identical to the post-only bilateral oracle so
        # pre0_postN_outN is a direct reference rather than a new random basis.
        self.post_mixer = (
            LearnedFHTBlockOrthogonalOutputMix(
                hidden_features,
                spec.post_stages,
                rotation_block_size,
                basis_block_size,
                seed,
            )
            if spec.post_stages
            else None
        )
        self.post_log_gain = (
            torch.nn.Parameter(torch.zeros(hidden_features))
            if spec.post_stages
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
        if self.pre_mixer is not None:
            assert self.pre_log_gain is not None
            pre_gelu = self.pre_mixer(
                self._gain(pre_gelu, self.pre_log_gain)
            )
        hidden = F.gelu(pre_gelu)
        if self.post_mixer is not None:
            assert self.post_log_gain is not None
            hidden = self.post_mixer(
                self._gain(hidden, self.post_log_gain)
            )
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
        for prefix, mixer, gain in (
            ("pre", self.pre_mixer, self.pre_log_gain),
            ("post", self.post_mixer, self.post_log_gain),
            ("output", self.output_mixer, self.output_log_gain),
        ):
            if mixer is not None:
                output[f"{prefix}_coordinate_rms"] = float(
                    mixer.coordinates.detach().square().mean().sqrt()
                )
            if gain is not None:
                output[f"{prefix}_log_gain_rms"] = float(
                    gain.detach().square().mean().sqrt()
                )
        return output


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
    if spec.name == "identity":
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

    chart = IndependentHiddenChart(
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
            "identity,pre0_post2_out4,"
            "pre1_post0_out4,pre2_post0_out4,"
            "pre1_post1_out4,pre2_post1_out4,"
            "pre1_post2_out4,pre2_post2_out4"
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
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "mlp_independent_hidden_oracle.csv"
    write_csv(csv_path, rows)
    metadata = {
        "schema_version": "mlp_independent_hidden_oracle_v1",
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
    metadata_path = args.output / "mlp_independent_hidden_oracle_summary.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
