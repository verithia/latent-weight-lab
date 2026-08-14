"""Fit a dense linear map immediately before each MLP residual addition.

This is a zero-update structural ceiling. It fits no model checkpoint and
never uses heldout targets to estimate the maps. Each discovery bank fits one
matrix per layer to match both the dense-teacher branch value and its input
JVP on the compact candidate's own evolving residual states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
    load_model,
)
from examples.nanogpt.model import GPT, MLP
from examples.nanogpt.optimize_mlp_bilateral_endpoint_ce import (
    autocast_context,
    prepare_frozen_base_cache,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mlp_branch(block: torch.nn.Module, values: torch.Tensor) -> torch.Tensor:
    output = block.mlp(values)
    if not isinstance(block.mlp, MLP):
        return output
    if (
        block.mlp.residual_conditioned_output_slope is None
        or block.mlp.conditioned_output_gate_source == "postgelu"
    ):
        return output
    return block.mlp.apply_residual_conditioned_output_gate(values, output)


def apply_fc(values: torch.Tensor, matrix: torch.Tensor | None) -> torch.Tensor:
    if matrix is None:
        return values
    return F.linear(values.float(), matrix).to(dtype=values.dtype)


def input_jvp(
    block: torch.nn.Module,
    values: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    _output, jvp = torch.autograd.functional.jvp(
        lambda current: mlp_branch(block, current),
        values,
        tangent,
        create_graph=False,
        strict=False,
    )
    return jvp


def normalized_recovery(
    prediction: torch.Tensor, target: torch.Tensor
) -> float:
    numerator = (prediction.float() - target.float()).square().sum()
    denominator = target.float().square().sum().clamp_min(1e-30)
    return float(1.0 - numerator / denominator)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    return float(
        torch.dot(left, right)
        / (left.norm() * right.norm()).clamp_min(1e-30)
    )


def fit_fc(
    features: torch.Tensor,
    targets: torch.Tensor,
    feature_jvps: torch.Tensor,
    target_jvps: torch.Tensor,
    ridge_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit target ~= feature @ matrix.T with an identity ridge prior."""
    features = features.float()
    targets = targets.float()
    feature_jvps = feature_jvps.float()
    target_jvps = target_jvps.float()
    value_energy = targets.square().mean().clamp_min(1e-30)
    jvp_energy = target_jvps.square().mean().clamp_min(1e-30)
    jvp_scale = torch.sqrt(value_energy / jvp_energy)
    design = torch.cat((features, feature_jvps * jvp_scale), dim=0)
    response = torch.cat((targets, target_jvps * jvp_scale), dim=0)
    gram = design.transpose(0, 1).matmul(design)
    rhs = design.transpose(0, 1).matmul(response)
    width = int(gram.shape[0])
    ridge = float(ridge_ratio) * float(torch.trace(gram)) / max(width, 1)
    identity = torch.eye(width, device=gram.device, dtype=gram.dtype)
    # W maps row features to row responses. F.linear expects C = W.T.
    weight = torch.linalg.solve(
        gram + ridge * identity,
        rhs + ridge * identity,
    )
    matrix = weight.transpose(0, 1).contiguous()
    return matrix, {
        "ridge": ridge,
        "jvp_scale": float(jvp_scale),
        "condition_number": float(
            torch.linalg.cond(gram + ridge * identity)
        ),
    }


def initial_states(
    model: GPT,
    batches: list[torch.Tensor],
    device: str,
) -> list[torch.Tensor]:
    states = []
    for tokens in batches:
        indices = tokens[:, :-1].to(device)
        positions = torch.arange(
            indices.shape[1], device=device, dtype=torch.long
        )
        states.append(
            model.transformer.drop(
                model.transformer.wte(indices)
                + model.transformer.wpe(positions)
            )
        )
    return states


def select_rows(
    values: torch.Tensor,
    count: int,
    seed: int,
) -> torch.Tensor:
    flat = values.reshape(-1, values.shape[-1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(flat.shape[0], generator=generator)[
        : min(int(count), flat.shape[0])
    ].to(flat.device)
    return flat.index_select(0, indices)


def fit_bank(
    candidate: GPT,
    teacher: GPT,
    batches: list[torch.Tensor],
    device: str,
    seed: int,
    sample_cap: int,
    jvp_sample_cap: int,
    ridge_ratio: float,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    states = initial_states(candidate, batches, device)
    matrices: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for layer, (candidate_block, teacher_block) in enumerate(
        zip(candidate.transformer.h, teacher.transformer.h, strict=True)
    ):
        layer_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        values: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        value_jvps: list[torch.Tensor] = []
        target_jvps: list[torch.Tensor] = []
        for batch_index, state in enumerate(states):
            with torch.no_grad(), autocast_context(device):
                residual = state + candidate_block.attn(
                    candidate_block.ln_1(state)
                )
                mlp_input = candidate_block.ln_2(residual)
                candidate_output = mlp_branch(candidate_block, mlp_input)
                teacher_output = mlp_branch(teacher_block, mlp_input)
            sample_seed = seed + 1009 * layer + 37 * batch_index
            values.append(
                select_rows(candidate_output, sample_cap, sample_seed)
            )
            targets.append(
                select_rows(teacher_output, sample_cap, sample_seed)
            )
            jvp_input = select_rows(
                mlp_input, jvp_sample_cap, sample_seed + 1
            )
            tangent_generator = torch.Generator(device=device)
            tangent_generator.manual_seed(sample_seed + 2)
            tangent = torch.randn(
                jvp_input.shape,
                generator=tangent_generator,
                device=device,
                dtype=torch.float32,
            )
            tangent = (
                tangent
                / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            ).to(dtype=jvp_input.dtype)
            with torch.enable_grad(), autocast_context(device):
                value_jvps.append(
                    input_jvp(candidate_block, jvp_input, tangent).detach()
                )
                target_jvps.append(
                    input_jvp(teacher_block, jvp_input, tangent).detach()
                )
            layer_states.append((residual, candidate_output))
        all_values = torch.cat(values)
        all_targets = torch.cat(targets)
        all_value_jvps = torch.cat(value_jvps)
        all_target_jvps = torch.cat(target_jvps)
        matrix, fit = fit_fc(
            all_values,
            all_targets,
            all_value_jvps,
            all_target_jvps,
            ridge_ratio,
        )
        matrices.append(matrix.detach())
        states = [
            residual + apply_fc(output, matrix)
            for residual, output in layer_states
        ]
        rows.append(
            {
                "layer": layer,
                **fit,
                "discovery_output_recovery": normalized_recovery(
                    apply_fc(all_values, matrix), all_targets
                ),
                "discovery_jvp_recovery": normalized_recovery(
                    apply_fc(all_value_jvps, matrix), all_target_jvps
                ),
                "matrix_identity_relative_frobenius": float(
                    (
                        matrix
                        - torch.eye(
                            matrix.shape[0],
                            device=matrix.device,
                            dtype=matrix.dtype,
                        )
                    ).norm()
                    / math.sqrt(matrix.shape[0])
                ),
            }
        )
        print(
            f"fit bank={seed} layer={layer} "
            f"out={rows[-1]['discovery_output_recovery']:.4f} "
            f"jvp={rows[-1]['discovery_jvp_recovery']:.4f}",
            flush=True,
        )
    return matrices, rows


@torch.no_grad()
def validation_ce(
    model: GPT,
    batches: list[torch.Tensor],
    matrices: list[torch.Tensor] | None,
    device: str,
) -> float:
    losses: list[float] = []
    for batch_index, tokens in enumerate(batches):
        indices = tokens[:, :-1].to(device)
        targets = tokens[:, 1:].to(device)
        positions = torch.arange(
            indices.shape[1], device=device, dtype=torch.long
        )
        with autocast_context(device):
            values = model.transformer.drop(
                model.transformer.wte(indices)
                + model.transformer.wpe(positions)
            )
            for layer, block in enumerate(model.transformer.h):
                values = values + block.attn(block.ln_1(values))
                mlp_input = block.ln_2(values)
                output = mlp_branch(block, mlp_input)
                matrix = None if matrices is None else matrices[layer]
                values = values + apply_fc(output, matrix)
            values = model.transformer.ln_f(values)
            logits = model.lm_head(values)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        losses.append(float(loss))
        if (batch_index + 1) % 50 == 0:
            print(
                f"eval {batch_index + 1}/{len(batches)} "
                f"mean_ce={sum(losses)/len(losses):.6f}",
                flush=True,
            )
    return float(sum(losses) / len(losses))


def heldout_metrics(
    candidate: GPT,
    teacher: GPT,
    batches: list[torch.Tensor],
    matrices: list[torch.Tensor],
    other_matrices: list[torch.Tensor],
    device: str,
    seed: int,
    sample_cap: int,
    jvp_sample_cap: int,
) -> dict[str, Any]:
    states = initial_states(candidate, batches, device)
    output_recoveries: list[float] = []
    jvp_recoveries: list[float] = []
    action_cosines: list[float] = []
    layer_rows: list[dict[str, float | int]] = []
    for layer, (candidate_block, teacher_block) in enumerate(
        zip(candidate.transformer.h, teacher.transformer.h, strict=True)
    ):
        next_states: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        candidate_jvps: list[torch.Tensor] = []
        teacher_jvps: list[torch.Tensor] = []
        for batch_index, state in enumerate(states):
            with torch.no_grad(), autocast_context(device):
                residual = state + candidate_block.attn(
                    candidate_block.ln_1(state)
                )
                mlp_input = candidate_block.ln_2(residual)
                candidate_output = mlp_branch(candidate_block, mlp_input)
                teacher_output = mlp_branch(teacher_block, mlp_input)
            sample_seed = seed + 1009 * layer + 37 * batch_index
            outputs.append(
                select_rows(candidate_output, sample_cap, sample_seed)
            )
            targets.append(
                select_rows(teacher_output, sample_cap, sample_seed)
            )
            jvp_input = select_rows(
                mlp_input, jvp_sample_cap, sample_seed + 1
            )
            tangent_generator = torch.Generator(device=device)
            tangent_generator.manual_seed(sample_seed + 2)
            tangent = torch.randn(
                jvp_input.shape,
                generator=tangent_generator,
                device=device,
                dtype=torch.float32,
            )
            tangent = (
                tangent
                / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            ).to(dtype=jvp_input.dtype)
            with torch.enable_grad(), autocast_context(device):
                candidate_jvps.append(
                    input_jvp(candidate_block, jvp_input, tangent).detach()
                )
                teacher_jvps.append(
                    input_jvp(teacher_block, jvp_input, tangent).detach()
                )
            next_states.append(
                residual + apply_fc(candidate_output, matrices[layer])
            )
        output = torch.cat(outputs)
        target = torch.cat(targets)
        candidate_jvp = torch.cat(candidate_jvps)
        teacher_jvp = torch.cat(teacher_jvps)
        transformed = apply_fc(output, matrices[layer])
        transformed_other = apply_fc(output, other_matrices[layer])
        transformed_jvp = apply_fc(candidate_jvp, matrices[layer])
        output_recovery = normalized_recovery(transformed, target)
        jvp_recovery = normalized_recovery(
            transformed_jvp, teacher_jvp
        )
        agreement = cosine(transformed, transformed_other)
        output_recoveries.append(output_recovery)
        jvp_recoveries.append(jvp_recovery)
        action_cosines.append(agreement)
        layer_rows.append(
            {
                "layer": layer,
                "output_recovery": output_recovery,
                "jvp_recovery": jvp_recovery,
                "cross_bank_action_cosine": agreement,
            }
        )
        states = next_states
    return {
        "mean_output_recovery": sum(output_recoveries)
        / len(output_recoveries),
        "minimum_output_recovery": min(output_recoveries),
        "mean_jvp_recovery": sum(jvp_recoveries)
        / len(jvp_recoveries),
        "minimum_jvp_recovery": min(jvp_recoveries),
        "mean_cross_bank_action_cosine": sum(action_cosines)
        / len(action_cosines),
        "minimum_cross_bank_action_cosine": min(action_cosines),
        "layers": layer_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--discovery-batch-size", type=int, default=4)
    parser.add_argument("--discovery-batches", type=int, default=2)
    parser.add_argument("--metric-batch-size", type=int, default=4)
    parser.add_argument("--metric-batches", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=400)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--sample-cap", type=int, default=4096)
    parser.add_argument("--jvp-sample-cap", type=int, default=256)
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument(
        "--discovery-seeds", default="2026081401,2026081402"
    )
    parser.add_argument("--metric-seed", type=int, default=2026081403)
    parser.add_argument("--eval-seed", type=int, default=20260715)
    args = parser.parse_args()

    started = time.time()
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("schema_version")
        != "nanogpt_124m_postbranch_fc_oracle_plan_v1"
    ):
        raise ValueError("unexpected plan schema")
    seeds = [int(value) for value in args.discovery_seeds.split(",")]
    if len(seeds) != 2 or seeds[0] == seeds[1]:
        raise ValueError("exactly two distinct discovery seeds are required")
    if args.ridge_ratio <= 0.0 or not math.isfinite(args.ridge_ratio):
        raise ValueError("ridge ratio must be positive and finite")

    candidate = load_model(args.candidate_checkpoint, args.device)
    teacher = load_model(args.teacher_checkpoint, args.device)
    if candidate.config.n_layer != teacher.config.n_layer:
        raise ValueError("candidate/teacher layer mismatch")
    if candidate.config.n_embd != teacher.config.n_embd:
        raise ValueError("candidate/teacher width mismatch")
    candidate_cache_count = prepare_frozen_base_cache(
        candidate, torch.bfloat16
    )

    discovery = {
        seed: fixed_validation_batches(
            args.data_dir,
            args.discovery_batch_size,
            args.block_size + 1,
            args.discovery_batches,
            seed,
        )
        for seed in seeds
    }
    metric_batches = fixed_validation_batches(
        args.data_dir,
        args.metric_batch_size,
        args.block_size + 1,
        args.metric_batches,
        args.metric_seed,
    )
    eval_batches = fixed_validation_batches(
        args.data_dir,
        args.eval_batch_size,
        args.block_size + 1,
        args.eval_batches,
        args.eval_seed,
    )

    fitted: dict[int, list[torch.Tensor]] = {}
    fit_rows: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        fitted[seed], fit_rows[seed] = fit_bank(
            candidate,
            teacher,
            discovery[seed],
            args.device,
            seed,
            args.sample_cap,
            args.jvp_sample_cap,
            args.ridge_ratio,
        )

    metrics = {
        seed: heldout_metrics(
            candidate,
            teacher,
            metric_batches,
            fitted[seed],
            fitted[seeds[1] if seed == seeds[0] else seeds[0]],
            args.device,
            args.metric_seed,
            args.sample_cap,
            args.jvp_sample_cap,
        )
        for seed in seeds
    }
    identity_ce = validation_ce(candidate, eval_batches, None, args.device)
    fitted_ce = {
        seed: validation_ce(
            candidate, eval_batches, fitted[seed], args.device
        )
        for seed in seeds
    }

    gates = plan["frozen_gates"]
    bank_passes = {
        seed: (
            fitted_ce[seed] <= float(gates["layer_private_ce_maximum"])
            and metrics[seed]["mean_output_recovery"]
            >= float(gates["heldout_output_recovery_minimum"])
            and metrics[seed]["mean_jvp_recovery"]
            >= float(gates["heldout_jvp_recovery_minimum"])
            and metrics[seed]["mean_cross_bank_action_cosine"]
            >= float(gates["cross_bank_action_cosine_minimum"])
        )
        for seed in seeds
    }
    passed = all(bank_passes.values())
    result = {
        "schema_version": "nanogpt_124m_postbranch_fc_oracle_result_v1",
        "classification": (
            "LAYER_PRIVATE_POSTBRANCH_FC_CEILING_PASSED"
            if passed
            else "LAYER_PRIVATE_POSTBRANCH_FC_CEILING_REJECTED"
        ),
        "identity": {
            "plan": str(args.plan),
            "plan_sha256": sha256(args.plan),
            "entrypoint_sha256": sha256(Path(__file__)),
            "candidate_checkpoint": str(args.candidate_checkpoint),
            "candidate_checkpoint_sha256": sha256(
                args.candidate_checkpoint
            ),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "teacher_checkpoint_sha256": sha256(
                args.teacher_checkpoint
            ),
        },
        "protocol": {
            "discovery_seeds": seeds,
            "metric_seed": args.metric_seed,
            "eval_seed": args.eval_seed,
            "discovery_batch_size": args.discovery_batch_size,
            "discovery_batches": args.discovery_batches,
            "metric_batch_size": args.metric_batch_size,
            "metric_batches": args.metric_batches,
            "eval_batch_size": args.eval_batch_size,
            "eval_batches": args.eval_batches,
            "block_size": args.block_size,
            "sample_cap": args.sample_cap,
            "jvp_sample_cap": args.jvp_sample_cap,
            "ridge_ratio": args.ridge_ratio,
            "candidate_cache_count": candidate_cache_count,
            "checkpoint_updates": 0,
        },
        "fit": {str(seed): fit_rows[seed] for seed in seeds},
        "heldout": {str(seed): metrics[seed] for seed in seeds},
        "validation_ce": {
            "identity": identity_ce,
            "fitted": {str(seed): fitted_ce[seed] for seed in seeds},
            "gain": {
                str(seed): identity_ce - fitted_ce[seed] for seed in seeds
            },
        },
        "decision": {
            "bank_passes": {
                str(seed): bank_passes[seed] for seed in seeds
            },
            "scientific_gate_passed": passed,
            "shared_fc_oracle_authorized": passed,
            "causal_training_authorized": False,
            "larger_rung_authorized": False,
        },
        "elapsed_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["validation_ce"], sort_keys=True), flush=True)
    print(result["classification"], flush=True)
    print(f"result={output} sha256={sha256(output)}", flush=True)


if __name__ == "__main__":
    main()
