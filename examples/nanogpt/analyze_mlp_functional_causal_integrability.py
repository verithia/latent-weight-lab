#!/usr/bin/env python3
"""Audit whether low-rank MLP weight paths are compact in function space.

The causal displacement audit showed that a preceding-gradient row space does
not represent the accumulated dense Muon weight displacement.  This script
tests the only remaining loophole: the discarded Frobenius energy may be
functionally irrelevant on realized MLP inputs.  It evaluates paired c_fc and
c_proj reconstructions on two fixed activation banks and on deterministic
input-Jacobian directions, without updating any model parameter.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    ActivationCollector,
)
from examples.nanogpt.analyze_mlp_causal_displacement_integrability import (
    load_probe_weights,
)
from examples.nanogpt.analyze_mlp_gradient_factor_field import fit_union_basis
from examples.nanogpt.analyze_mlp_highcadence_basis import file_sha256
from examples.nanogpt.analyze_mlp_optimizer_probe_span import (
    load_probe_inventory,
)
from examples.nanogpt.analyze_mlp_product_fht_tangent_anchor import git_commit
from examples.nanogpt.analyze_mlp_raw_gradient_factor_transport import (
    exact_singular_factors,
)
from examples.nanogpt.analyze_mlp_raw_gradient_rolling_prediction import (
    phase_for_step,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)


def direction_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, float]:
    """Return scale-sensitive and scale-free recovery of one tensor change."""
    target = target.double().reshape(-1)
    prediction = prediction.double().reshape(-1)
    target_energy = target.square().sum().clamp_min(1e-30)
    prediction_energy = prediction.square().sum()
    dot = (target * prediction).sum()
    cosine = dot / (target_energy * prediction_energy).sqrt().clamp_min(1e-30)
    positive_scale = torch.clamp(dot / prediction_energy.clamp_min(1e-30), min=0.0)
    positive_residual = target - positive_scale * prediction
    return {
        "cosine": float(cosine),
        "positive_line_recovery": float(
            1.0 - positive_residual.square().sum() / target_energy
        ),
        "fixed_scale_recovery": float(
            1.0 - (target - prediction).square().sum() / target_energy
        ),
        "target_energy": float(target_energy),
        "prediction_energy": float(prediction_energy),
        "positive_scale": float(positive_scale),
    }


def right_project(matrix: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project a matrix onto a supplied right/input subspace."""
    return (matrix @ basis) @ basis.transpose(0, 1)


def truncated_svd_factors(
    matrix: torch.Tensor,
    maximum_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact leading SVD factors once for all requested ranks."""
    left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
    rank = min(int(maximum_rank), singular.numel())
    return left[:, :rank], singular[:rank], right_h[:rank]


def truncated_svd_reconstruct(
    left: torch.Tensor,
    singular: torch.Tensor,
    right_h: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    selected = min(int(rank), singular.numel())
    return (
        left[:, :selected] * singular[:selected].unsqueeze(0)
    ) @ right_h[:selected]


def mlp_output(
    inputs: torch.Tensor,
    c_fc_weight: torch.Tensor,
    c_proj_weight: torch.Tensor,
    c_fc_bias: torch.Tensor | None,
    c_proj_bias: torch.Tensor | None,
) -> torch.Tensor:
    pre = F.linear(inputs, c_fc_weight, c_fc_bias)
    return F.linear(F.gelu(pre), c_proj_weight, c_proj_bias)


def gelu_derivative(values: torch.Tensor) -> torch.Tensor:
    """Derivative of torch.nn.GELU's default exact formulation."""
    return (
        0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))
        + values * torch.exp(-0.5 * values.square()) / math.sqrt(2.0 * math.pi)
    )


def mlp_input_jvp(
    inputs: torch.Tensor,
    directions: torch.Tensor,
    c_fc_weight: torch.Tensor,
    c_proj_weight: torch.Tensor,
    c_fc_bias: torch.Tensor | None,
) -> torch.Tensor:
    pre = F.linear(inputs, c_fc_weight, c_fc_bias)
    pre_jvp = F.linear(directions, c_fc_weight, None)
    return F.linear(gelu_derivative(pre) * pre_jvp, c_proj_weight, None)


def collect_activation_banks(
    checkpoint: Path,
    data_dir: Path,
    *,
    layer: int,
    sample_cap: int,
    seeds: list[int],
    device: str,
) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    model = load_model(checkpoint, device)
    banks: list[torch.Tensor] = []
    try:
        for seed in seeds:
            batches = fixed_validation_batches(
                data_dir,
                batch_size=1,
                block_size=int(model.config.block_size),
                batches=1,
                seed=seed,
            )
            collector = ActivationCollector(
                model,
                [layer],
                sample_cap,
                collect_pre_gelu=False,
                collect_mlp_input=True,
            )
            try:
                with torch.no_grad():
                    for batch in batches:
                        model(batch.to(device), None)
                        if collector.complete():
                            break
                if not collector.complete():
                    raise ValueError(
                        f"activation bank seed {seed} did not reach sample cap"
                    )
                banks.append(collector.tensor(layer, "mlp_input").contiguous())
            finally:
                collector.close()
        mlp = model.transformer.h[layer].mlp
        c_fc_bias = getattr(mlp.c_fc, "bias", None)
        c_proj_bias = getattr(mlp.c_proj, "bias", None)
        metadata = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "data_dir": str(data_dir),
            "block_size": int(model.config.block_size),
            "sample_cap": sample_cap,
            "seeds": seeds,
        }
        return (
            banks,
            c_fc_bias.detach().float().cpu() if c_fc_bias is not None else None,
            c_proj_bias.detach().float().cpu() if c_proj_bias is not None else None,
            metadata,
        )
    finally:
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (row["chart_kind"], row["split"], row["bank"], row["union_rank"])
            for row in rows
        }
    )
    result: list[dict[str, Any]] = []
    metric_fields = tuple(
        f"{kind}_{metric}"
        for kind in ("output", "jvp")
        for metric in (
            "cosine",
            "positive_line_recovery",
            "fixed_scale_recovery",
        )
    )
    for chart_kind, split, bank, rank in keys:
        members = [
            row
            for row in rows
            if (
                row["chart_kind"],
                row["split"],
                row["bank"],
                row["union_rank"],
            )
            == (chart_kind, split, bank, rank)
        ]
        record: dict[str, Any] = {
            "chart_kind": chart_kind,
            "split": split,
            "bank": bank,
            "union_rank": rank,
            "sample_count": len(members),
        }
        for field in metric_fields:
            values = torch.tensor(
                [float(row[field]) for row in members], dtype=torch.float64
            )
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_minimum"] = float(values.min())
            record[f"{field}_p10"] = float(torch.quantile(values, 0.10))
        result.append(record)
    return result


def gate_outcome(
    summary: list[dict[str, Any]],
    *,
    rank: int,
    output_mean_gate: float,
    output_minimum_gate: float,
    jvp_mean_gate: float,
) -> dict[str, Any]:
    selected = [
        row
        for row in summary
        if row["chart_kind"] == "causal"
        and row["split"] == "test"
        and row["union_rank"] == rank
    ]
    if not selected:
        raise ValueError("functional gate has no causal test rows")
    banks = []
    for row in selected:
        banks.append(
            {
                "bank": row["bank"],
                "output_fixed_scale_recovery_mean": row[
                    "output_fixed_scale_recovery_mean"
                ],
                "output_fixed_scale_recovery_minimum": row[
                    "output_fixed_scale_recovery_minimum"
                ],
                "jvp_fixed_scale_recovery_mean": row[
                    "jvp_fixed_scale_recovery_mean"
                ],
                "passed": (
                    row["output_fixed_scale_recovery_mean"] >= output_mean_gate
                    and row["output_fixed_scale_recovery_minimum"]
                    >= output_minimum_gate
                    and row["jvp_fixed_scale_recovery_mean"] >= jvp_mean_gate
                ),
            }
        )
    return {
        "rank": rank,
        "thresholds": {
            "output_fixed_scale_recovery_mean": output_mean_gate,
            "output_fixed_scale_recovery_minimum": output_minimum_gate,
            "jvp_fixed_scale_recovery_mean": jvp_mean_gate,
        },
        "banks": banks,
        "passed": all(row["passed"] for row in banks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--factor-rank", type=int, default=6)
    parser.add_argument("--union-ranks", default="1,3,6,12,24,48")
    parser.add_argument("--history-probes", type=int, default=10)
    parser.add_argument("--discovery-stop", type=int, default=119)
    parser.add_argument("--validation-stop", type=int, default=179)
    parser.add_argument("--sample-cap", type=int, default=256)
    parser.add_argument("--activation-seeds", default="202608261,202608262")
    parser.add_argument("--jvp-seed", type=int, default=202608263)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    ranks = [int(value) for value in args.union_ranks.split(",")]
    activation_seeds = [int(value) for value in args.activation_seeds.split(",")]
    if len(activation_seeds) < 2:
        raise ValueError("at least two disjoint activation seeds are required")

    paths = sorted(args.probe_dir.glob("step_*.pt"))
    steps, inventory, probe_metadata = load_probe_inventory(
        paths,
        layers={args.layer},
        targets={"mlp.c_fc", "mlp.c_proj"},
    )
    names = {
        target: next(name for name in inventory if f".{target}.weight" in name)
        for target in ("mlp.c_fc", "mlp.c_proj")
    }
    weights = load_probe_weights(
        paths,
        parameters=set(inventory),
        expected_steps=steps,
        expected_identity=str(probe_metadata["run_identity_sha256"]),
    )
    right_fields: dict[str, list[torch.Tensor]] = {}
    singular_fields: dict[str, list[torch.Tensor]] = {}
    for target, name in names.items():
        gradients = torch.stack(inventory[name]["raw_gradient_descent"]).to(
            args.device, torch.float32
        )
        _left, singular, right = exact_singular_factors(
            gradients, args.factor_rank
        )
        right_fields[target] = right
        singular_fields[target] = singular
        del gradients, _left

    activation_banks, c_fc_bias, c_proj_bias, activation_metadata = (
        collect_activation_banks(
            args.checkpoint,
            args.data_dir,
            layer=args.layer,
            sample_cap=args.sample_cap,
            seeds=activation_seeds,
            device=args.device,
        )
    )
    activation_banks = [bank.to(args.device, torch.float32) for bank in activation_banks]
    c_fc_bias = c_fc_bias.to(args.device) if c_fc_bias is not None else None
    c_proj_bias = c_proj_bias.to(args.device) if c_proj_bias is not None else None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.jvp_seed)
    jvp_directions = [
        (
            torch.randint(
                0,
                2,
                bank.shape,
                generator=generator,
                dtype=torch.float32,
            )
            * 2.0
            - 1.0
        ).to(args.device)
        / math.sqrt(bank.shape[1])
        for bank in activation_banks
    ]

    c_fc_weights = torch.stack(weights[names["mlp.c_fc"]]).to(
        args.device, torch.float32
    )
    c_proj_weights = torch.stack(weights[names["mlp.c_proj"]]).to(
        args.device, torch.float32
    )
    initial_c_fc = c_fc_weights[0]
    initial_c_proj = c_proj_weights[0]
    base_outputs = [
        mlp_output(bank, initial_c_fc, initial_c_proj, c_fc_bias, c_proj_bias)
        for bank in activation_banks
    ]
    base_jvps = [
        mlp_input_jvp(
            bank,
            direction,
            initial_c_fc,
            initial_c_proj,
            c_fc_bias,
        )
        for bank, direction in zip(activation_banks, jvp_directions, strict=True)
    ]

    rows: list[dict[str, Any]] = []
    maximum_rank = max(ranks)
    for index in range(args.history_probes, len(steps)):
        c_fc_delta = c_fc_weights[index] - initial_c_fc
        c_proj_delta = c_proj_weights[index] - initial_c_proj
        causal_bases = {
            target: fit_union_basis(
                right_fields[target],
                singular_fields[target],
                range(index - args.history_probes, index),
                min(maximum_rank, args.history_probes * args.factor_rank),
            )
            for target in ("mlp.c_fc", "mlp.c_proj")
        }
        oracle_factors = {
            "mlp.c_fc": truncated_svd_factors(c_fc_delta, maximum_rank),
            "mlp.c_proj": truncated_svd_factors(c_proj_delta, maximum_rank),
        }
        teacher_outputs = [
            mlp_output(
                bank,
                c_fc_weights[index],
                c_proj_weights[index],
                c_fc_bias,
                c_proj_bias,
            )
            for bank in activation_banks
        ]
        teacher_jvps = [
            mlp_input_jvp(
                bank,
                direction,
                c_fc_weights[index],
                c_proj_weights[index],
                c_fc_bias,
            )
            for bank, direction in zip(
                activation_banks, jvp_directions, strict=True
            )
        ]
        for rank in ranks:
            candidates = {
                "causal": (
                    initial_c_fc
                    + right_project(c_fc_delta, causal_bases["mlp.c_fc"][:, :rank]),
                    initial_c_proj
                    + right_project(
                        c_proj_delta, causal_bases["mlp.c_proj"][:, :rank]
                    ),
                ),
                "oracle_svd": (
                    initial_c_fc
                    + truncated_svd_reconstruct(
                        *oracle_factors["mlp.c_fc"], rank
                    ),
                    initial_c_proj
                    + truncated_svd_reconstruct(
                        *oracle_factors["mlp.c_proj"], rank
                    ),
                ),
            }
            for chart_kind, (candidate_c_fc, candidate_c_proj) in candidates.items():
                for bank_index, (bank, direction) in enumerate(
                    zip(activation_banks, jvp_directions, strict=True)
                ):
                    candidate_output = mlp_output(
                        bank,
                        candidate_c_fc,
                        candidate_c_proj,
                        c_fc_bias,
                        c_proj_bias,
                    )
                    candidate_jvp = mlp_input_jvp(
                        bank,
                        direction,
                        candidate_c_fc,
                        candidate_c_proj,
                        c_fc_bias,
                    )
                    output_metrics = direction_metrics(
                        teacher_outputs[bank_index] - base_outputs[bank_index],
                        candidate_output - base_outputs[bank_index],
                    )
                    jvp_metrics = direction_metrics(
                        teacher_jvps[bank_index] - base_jvps[bank_index],
                        candidate_jvp - base_jvps[bank_index],
                    )
                    row: dict[str, Any] = {
                        "probe_index": index,
                        "step": steps[index],
                        "split": phase_for_step(
                            steps[index], args.discovery_stop, args.validation_stop
                        ),
                        "bank": bank_index,
                        "activation_seed": activation_seeds[bank_index],
                        "chart_kind": chart_kind,
                        "union_rank": rank,
                    }
                    row.update(
                        {f"output_{key}": value for key, value in output_metrics.items()}
                    )
                    row.update(
                        {f"jvp_{key}": value for key, value in jvp_metrics.items()}
                    )
                    rows.append(row)

    summary = summarize(rows)
    gate = gate_outcome(
        summary,
        rank=6,
        output_mean_gate=0.80,
        output_minimum_gate=0.70,
        jvp_mean_gate=0.60,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    detail_path = args.output / "functional_causal_integrability.csv"
    summary_path = args.output / "functional_causal_integrability_summary.csv"
    write_csv(detail_path, rows)
    write_csv(summary_path, summary)
    gate_path = args.output / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    metadata = {
        "schema_version": "nanogpt_mlp_functional_causal_integrability_v1",
        "source_commit": git_commit(Path(__file__).resolve().parents[2]),
        "probe_input": probe_metadata,
        "activation_input": activation_metadata,
        "steps": steps,
        "layer": args.layer,
        "factor_rank": args.factor_rank,
        "union_ranks": ranks,
        "history_probes": args.history_probes,
        "discovery_stop": args.discovery_stop,
        "validation_stop": args.validation_stop,
        "jvp_seed": args.jvp_seed,
        "bias_policy": "terminal checkpoint biases held identical for initial, teacher, and reconstruction",
        "gate": gate,
        "limitations": [
            "One layer, seed, schedule, horizon, and dense-Muon trajectory.",
            "Activation banks are terminal-model validation activations, not historical per-probe activations.",
            "The audit isolates matrix-weight path effects and does not reconstruct historical biases.",
            "Oracle SVD charts are noncausal descriptive ceilings and are not compact implementations.",
        ],
        "runtime_seconds": time.time() - started,
        "detail_sha256": file_sha256(detail_path),
        "summary_sha256": file_sha256(summary_path),
        "gate_sha256": file_sha256(gate_path),
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
