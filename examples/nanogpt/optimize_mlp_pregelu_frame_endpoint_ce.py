"""Optimize a folded pre-GELU MLP frame at a frozen 124M endpoint.

The accepted bilateral endpoint state obtains its small task-CE gain from
post-GELU rotations.  This causal diagnostic freezes those accepted
rotations, resets both scalar-gain groups to identity, and updates only a new
independent orthogonal frame on each ``c_fc`` output before GELU.

Short preflight and scientific runs are foreground-only.  Scientific mode
requires an immutable certificate from the exact code, inputs, protocol, and
hardware path proving at least 20 percent measured MFU and a separate-seed CE
safety check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    tensor_sha256,
)
from examples.nanogpt.analyze_mlp_bilateral_endpoint_ce_oracle import (
    restore_chart_state,
)
from examples.nanogpt.analyze_mlp_bilateral_task_ce_attribution import (
    combine_states,
    load_state,
)
from examples.nanogpt.analyze_mlp_chart_gradient_alignment import (
    load_chart_model,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.mfu_preflight import (
    empirical_bf16_gemm_peak_tflops,
    estimate_active_params,
)
from examples.nanogpt.model import (
    GPT,
    LearnedFHTBlockOrthogonalOutputMix,
)
from examples.nanogpt.optimize_mlp_bilateral_endpoint_ce import (
    ALL_LAYERS,
    clear_frozen_base_cache,
    evaluate_ce,
    git_head,
    indices_digest,
    make_optimizer,
    make_train_indices,
    prepare_frozen_base_cache,
    run_update,
    sha256,
    stable_json_sha256,
    write_csv,
)
from examples.nanogpt.train import TokenBatchSource


PLAN_NAME = "124m_mlp_pregelu_frame_task_ce_endpoint_plan.json"
CERTIFICATE_SCHEMA = "mai_124m_mlp_pregelu_frame_mfu_v1"
IDENTITY_SCHEMA = "mai_124m_mlp_pregelu_frame_identity_v1"
STATE_SCHEMA = "mai_124m_mlp_pregelu_frame_state_v1"


def install_pregelu_frames(
    model: GPT,
    *,
    stages: int,
    rotation_block_size: int,
    basis_block_size: int,
    basis_seed: int,
    per_layer_seed_offset: int,
    coordinate_scale: float,
) -> dict[str, torch.nn.Parameter]:
    parameters: dict[str, torch.nn.Parameter] = {}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer, block in enumerate(model.transformer.h):
        frame = LearnedFHTBlockOrthogonalOutputMix(
            features=4 * model.config.n_embd,
            stages=stages,
            rotation_block_size=rotation_block_size,
            basis_block_size=basis_block_size,
            seed=basis_seed + layer * per_layer_seed_offset,
            coordinate_scale=coordinate_scale,
        ).to(next(model.parameters()).device)
        block.mlp.pregelu_block_rotation = frame
        frame.coordinates.requires_grad_(True)
        parameters[f"layer.{layer}.pregelu_rotation"] = frame.coordinates
    return parameters


def capture_pregelu_state(
    model: GPT,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for layer, block in enumerate(model.transformer.h):
        frame = block.mlp.pregelu_block_rotation
        if frame is None:
            raise RuntimeError(f"layer {layer} has no pre-GELU frame")
        state[f"layer.{layer}.pregelu_rotation"] = (
            frame.coordinates.detach().float().cpu().clone()
        )
    return state


def select_decision(
    rows: list[dict[str, object]],
    minimum_gain: float,
) -> dict[str, object]:
    splits = ("primary", "confirmation", "audit")
    by_split = {
        split: {
            int(row["update"]): row
            for row in rows
            if row["split"] == split
        }
        for split in splits
    }
    updates = set(by_split["primary"])
    if not updates or any(set(by_split[split]) != updates for split in splits):
        raise ValueError("validation rows are incomplete across splits")
    if 0 not in updates:
        raise ValueError("validation rows are missing step 0")
    selected = min(
        by_split["primary"].values(),
        key=lambda row: float(row["ce"]),
    )
    selected_update = int(selected["update"])
    gains = {
        split: (
            float(by_split[split][0]["ce"])
            - float(by_split[split][selected_update]["ce"])
        )
        for split in splits
    }
    positive = (
        selected_update > 0
        and all(gain >= minimum_gain for gain in gains.values())
    )
    return {
        "selected_update": selected_update,
        "minimum_marginal_gain": minimum_gain,
        "baseline_ce": {
            split: float(by_split[split][0]["ce"])
            for split in splits
        },
        "selected_ce": {
            split: float(by_split[split][selected_update]["ce"])
            for split in splits
        },
        "marginal_gain": gains,
        "decision": (
            "POSITIVE_PRE_GELU_FRAME_CAPACITY"
            if positive
            else "REJECT_PRE_GELU_FRAME_CAPACITY"
        ),
    }


def protocol_identity(
    args: argparse.Namespace,
    *,
    root: Path,
    input_hashes: dict[str, str],
    device_name: str,
) -> dict[str, Any]:
    source = Path(__file__).resolve()
    plan = (
        root
        / "examples/nanogpt/configs/selection_artifacts"
        / PLAN_NAME
    )
    value: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA,
        "repository_commit": git_head(root),
        "source_sha256": sha256(source),
        "plan_sha256": sha256(plan),
        "inputs": input_hashes,
        "hardware_device": device_name,
        "reference_chart": {
            "accepted_groups": [
                "hidden_rotation",
                "output_rotation",
            ],
            "identity_groups": ["hidden_gain", "output_gain"],
            "layers": ALL_LAYERS,
            "frozen": True,
        },
        "pregelu_frame": {
            "features": 4 * 768,
            "stages": args.frame_stages,
            "rotation_block_size": args.frame_rotation_block_size,
            "basis_block_size": args.frame_basis_block_size,
            "basis_seed": args.frame_basis_seed,
            "per_layer_seed_offset": args.frame_seed_offset,
            "coordinate_scale": args.frame_coordinate_scale,
            "weight_folding": "apply_R_to_W_transpose",
            "layers": ALL_LAYERS,
        },
        "update": {
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
            "block_size": args.block_size,
            "tokens_per_update": (
                args.batch_size
                * args.gradient_accumulation_steps
                * args.block_size
            ),
            "optimizer": "fused_adamw",
            "learning_rate": args.learning_rate,
            "betas": [args.beta1, args.beta2],
            "weight_decay": args.weight_decay,
            "dtype": "bfloat16",
            "all_non_pregelu_parameters_frozen": True,
        },
        "evaluation": {
            "batch_size": args.eval_batch_size,
            "token_window_length": args.eval_block_size,
            "batches": args.eval_batches,
            "primary_seed": args.primary_eval_seed,
            "confirmation_seed": args.confirmation_eval_seed,
            "audit_seed": args.audit_eval_seed,
            "minimum_marginal_gain": args.minimum_ce_gain,
        },
        "preflight_safety": {
            "seed": args.preflight_safety_eval_seed,
            "batches": args.preflight_safety_eval_batches,
            "maximum_ce_increase": args.preflight_max_ce_increase,
        },
    }
    value["identity_sha256"] = stable_json_sha256(value)
    return value


def validate_mfu_certificate(
    certificate: dict[str, Any],
    identity: dict[str, Any],
    minimum_fraction: float,
) -> None:
    if certificate.get("schema_version") != CERTIFICATE_SCHEMA:
        raise ValueError("MFU certificate schema is incompatible")
    if certificate.get("identity") != identity:
        raise ValueError("MFU certificate does not match this exact run")
    measured = float(certificate.get("measurement", {}).get("mfu_fraction"))
    if not math.isfinite(measured) or measured < minimum_fraction:
        raise ValueError(
            f"MFU certificate failed: {measured:.2%} < "
            f"{minimum_fraction:.2%}"
        )
    stability = certificate.get("stability", {})
    increase = float(stability.get("ce_increase"))
    maximum = float(stability.get("maximum_ce_increase"))
    if not math.isfinite(increase) or increase > maximum:
        raise ValueError(
            "MFU certificate failed its endpoint CE safety gate"
        )
    if certificate.get("passed") is not True:
        raise ValueError("MFU certificate is not marked passed")


def load_runtime(
    args: argparse.Namespace,
) -> tuple[
    GPT,
    torch.optim.Optimizer,
    TokenBatchSource,
    dict[str, Any],
    dict[str, Any],
]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[2]
    manifest = args.data_dir / "manifest.json"
    expected = {
        "checkpoint": (args.checkpoint, args.checkpoint_sha256),
        "identity_state": (
            args.identity_state,
            args.identity_state_sha256,
        ),
        "accepted_state": (
            args.accepted_state,
            args.accepted_state_sha256,
        ),
        "dataset_manifest": (manifest, args.manifest_sha256),
    }
    input_hashes: dict[str, str] = {}
    for name, (path, expected_hash) in expected.items():
        actual = sha256(path)
        if actual != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {path}: "
                f"{actual} != {expected_hash}"
            )
        input_hashes[name] = actual

    identity_state = load_state(args.identity_state)
    accepted_state = load_state(args.accepted_state)
    reference_state = combine_states(
        identity_state,
        accepted_state,
        lambda layer, group: group.endswith("rotation"),
    )
    model = load_chart_model(
        args.checkpoint,
        args.device,
        ALL_LAYERS,
        initial_output_log_gain=0.0,
    )
    restore_chart_state(model, ALL_LAYERS, reference_state)
    parameters = install_pregelu_frames(
        model,
        stages=args.frame_stages,
        rotation_block_size=args.frame_rotation_block_size,
        basis_block_size=args.frame_basis_block_size,
        basis_seed=args.frame_basis_seed,
        per_layer_seed_offset=args.frame_seed_offset,
        coordinate_scale=args.frame_coordinate_scale,
    )
    optimizer = make_optimizer(list(parameters.values()), args)
    cached_modules = prepare_frozen_base_cache(
        model, torch.bfloat16
    )
    device_name = (
        torch.cuda.get_device_name(0)
        if args.device.startswith("cuda")
        else "cpu"
    )
    identity = protocol_identity(
        args,
        root=root,
        input_hashes=input_hashes,
        device_name=device_name,
    )
    runtime = {
        "root": root,
        "manifest": manifest,
        "input_hashes": input_hashes,
        "cached_block_fht_modules": cached_modules,
        "pregelu_parameter_count": sum(
            parameter.numel() for parameter in parameters.values()
        ),
        "model_config": asdict(model.config),
    }
    return model, optimizer, TokenBatchSource(args.data_dir), identity, runtime


def run_preflight(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise ValueError("preflight certificate path already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    model, optimizer, source, identity, runtime = load_runtime(args)
    updates = args.preflight_warmup_updates + args.preflight_timed_updates
    indices = make_train_indices(
        args.data_dir,
        updates=updates,
        accumulation=args.gradient_accumulation_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        seed=args.train_token_seed + 1000003,
    )
    safety_batches = fixed_validation_batches(
        args.data_dir,
        args.eval_batch_size,
        args.eval_block_size,
        args.preflight_safety_eval_batches,
        args.preflight_safety_eval_seed,
    )
    certificate: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA,
        "identity": identity,
        "command": sys.argv,
        "policy": {
            "minimum_fraction": args.minimum_mfu,
            "numerator": "conventional_6N_decoder_model_flops",
            "denominator": "empirical_bf16_tensorcore_gemm_peak",
        },
        "preflight": {
            "warmup_updates": args.preflight_warmup_updates,
            "timed_updates": args.preflight_timed_updates,
            "train_indices_sha256": indices_digest(indices),
        },
        "stability": {
            "seed": args.preflight_safety_eval_seed,
            "token_sha256": tensor_sha256(
                torch.cat(safety_batches)
            ),
            "maximum_ce_increase": args.preflight_max_ce_increase,
        },
        "runtime": {
            "cached_block_fht_modules": runtime[
                "cached_block_fht_modules"
            ],
            "pregelu_parameter_count": runtime[
                "pregelu_parameter_count"
            ],
        },
        "passed": False,
    }
    error: Exception | None = None
    try:
        initial_safety_ce = evaluate_ce(
            model, safety_batches, args.device
        )
        peak_tflops = empirical_bf16_gemm_peak_tflops(
            args.gemm_size,
            args.gemm_warmups,
            args.gemm_trials,
        )
        durations: list[float] = []
        losses: list[float] = []
        torch.cuda.reset_peak_memory_stats()
        for update in range(updates):
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss = run_update(
                model, optimizer, source, indices[update], args
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            print(
                f"preflight update={update + 1}/{updates} "
                f"loss={loss:.6f} iter_ms={elapsed * 1000.0:.3f}",
                flush=True,
            )
            if update >= args.preflight_warmup_updates:
                durations.append(elapsed)
                losses.append(loss)
        tokens_per_update = (
            args.batch_size
            * args.gradient_accumulation_steps
            * args.block_size
        )
        tokens_per_second = (
            tokens_per_update * len(durations) / sum(durations)
        )
        active_params = estimate_active_params(runtime["model_config"])
        model_tflops = 6.0 * active_params * tokens_per_second / 1e12
        mfu = model_tflops / peak_tflops
        final_safety_ce = evaluate_ce(
            model, safety_batches, args.device
        )
        safety_increase = final_safety_ce - initial_safety_ce
        certificate["calibration"] = {
            "bf16_gemm_size": args.gemm_size,
            "bf16_gemm_warmups": args.gemm_warmups,
            "bf16_gemm_trials": args.gemm_trials,
            "empirical_bf16_gemm_peak_tflops": peak_tflops,
        }
        certificate["measurement"] = {
            "active_params_6n_estimate": active_params,
            "tokens_per_update": tokens_per_update,
            "tokens_per_second": tokens_per_second,
            "mean_iter_ms": 1000.0 * sum(durations) / len(durations),
            "model_tflops": model_tflops,
            "mfu_fraction": mfu,
            "mean_loss": float(np.mean(losses)),
            "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
        }
        certificate["stability"].update(
            {
                "initial_ce": initial_safety_ce,
                "final_ce": final_safety_ce,
                "ce_increase": safety_increase,
                "passed": bool(
                    safety_increase <= args.preflight_max_ce_increase
                ),
            }
        )
        certificate["passed"] = bool(
            mfu >= args.minimum_mfu
            and safety_increase <= args.preflight_max_ce_increase
        )
        if not certificate["passed"]:
            if safety_increase > args.preflight_max_ce_increase:
                raise RuntimeError(
                    "endpoint CE safety gate rejected: "
                    f"increase {safety_increase:+.6f} > "
                    f"{args.preflight_max_ce_increase:+.6f}"
                )
            raise RuntimeError(
                f"MFU gate rejected: {mfu:.2%} < "
                f"{args.minimum_mfu:.2%}"
            )
    except Exception as caught:
        certificate["error"] = str(caught)
        error = caught
    finally:
        clear_frozen_base_cache(model)
        output.write_text(
            json.dumps(certificate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(certificate, sort_keys=True), flush=True)
    if error is not None:
        raise error


def run_scientific(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("scientific output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    model, optimizer, source, identity, runtime = load_runtime(args)
    if args.mfu_certificate is None:
        raise ValueError("--mfu-certificate is required in run mode")
    certificate = json.loads(args.mfu_certificate.read_text())
    validate_mfu_certificate(
        certificate, identity, args.minimum_mfu
    )
    train_indices = make_train_indices(
        args.data_dir,
        updates=args.updates,
        accumulation=args.gradient_accumulation_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        seed=args.train_token_seed,
    )
    split_seeds = {
        "primary": args.primary_eval_seed,
        "confirmation": args.confirmation_eval_seed,
        "audit": args.audit_eval_seed,
    }
    validation = {
        split: fixed_validation_batches(
            args.data_dir,
            args.eval_batch_size,
            args.eval_block_size,
            args.eval_batches,
            seed,
        )
        for split, seed in split_seeds.items()
    }
    validation_digests = {
        split: tensor_sha256(torch.cat(batches))
        for split, batches in validation.items()
    }
    evaluation_updates = set(args.evaluation_updates)
    if 0 not in evaluation_updates or args.updates not in evaluation_updates:
        raise ValueError("evaluation updates must include 0 and final update")
    train_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    state_artifacts: dict[str, dict[str, str]] = {}

    def evaluate(update: int) -> None:
        state_path = output / f"pregelu_state_step_{update:04d}.pt"
        torch.save(
            {
                "schema_version": STATE_SCHEMA,
                "update": update,
                "state": capture_pregelu_state(model),
            },
            state_path,
        )
        state_artifacts[str(update)] = {
            "path": str(state_path),
            "sha256": sha256(state_path),
        }
        for split, batches in validation.items():
            ce = evaluate_ce(model, batches, args.device)
            validation_rows.append(
                {
                    "update": update,
                    "split": split,
                    "seed": split_seeds[split],
                    "token_sha256": validation_digests[split],
                    "ce": ce,
                }
            )
            print(
                f"eval update={update} split={split} ce={ce:.6f}",
                flush=True,
            )

    try:
        evaluate(0)
        for update in range(1, args.updates + 1):
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss = run_update(
                model,
                optimizer,
                source,
                train_indices[update - 1],
                args,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            train_rows.append(
                {
                    "update": update,
                    "train_ce": loss,
                    "iter_ms": elapsed * 1000.0,
                    "tokens_per_second": (
                        args.batch_size
                        * args.gradient_accumulation_steps
                        * args.block_size
                        / elapsed
                    ),
                }
            )
            if update % args.log_interval == 0 or update in evaluation_updates:
                print(
                    f"train update={update}/{args.updates} "
                    f"ce={loss:.6f} iter_ms={elapsed * 1000.0:.3f}",
                    flush=True,
                )
            if update in evaluation_updates:
                evaluate(update)
    finally:
        clear_frozen_base_cache(model)

    selection = select_decision(
        validation_rows, args.minimum_ce_gain
    )
    train_csv = output / "optimization.csv"
    validation_csv = output / "validation.csv"
    write_csv(train_csv, train_rows)
    write_csv(validation_csv, validation_rows)
    root: Path = runtime["root"]
    plan = (
        root
        / "examples/nanogpt/configs/selection_artifacts"
        / PLAN_NAME
    )
    summary = {
        "schema_version": (
            "mai_124m_mlp_pregelu_frame_endpoint_result_v1"
        ),
        "repository_commit": git_head(root),
        "command": sys.argv,
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "plan": {"path": str(plan), "sha256": sha256(plan)},
        "inputs": {
            name: {
                "path": str(path),
                "sha256": runtime["input_hashes"][name],
            }
            for name, path in {
                "checkpoint": args.checkpoint,
                "identity_state": args.identity_state,
                "accepted_state": args.accepted_state,
                "dataset_manifest": runtime["manifest"],
            }.items()
        },
        "mfu_certificate": {
            "path": str(args.mfu_certificate),
            "sha256": sha256(args.mfu_certificate),
            "mfu_fraction": certificate["measurement"][
                "mfu_fraction"
            ],
        },
        "protocol_identity": identity,
        "protocol": {
            "updates": args.updates,
            "evaluation_updates": sorted(evaluation_updates),
            "train_token_seed": args.train_token_seed,
            "train_indices_sha256": indices_digest(train_indices),
            "validation_seeds": split_seeds,
            "validation_token_sha256": validation_digests,
            "eval_batch_size": args.eval_batch_size,
            "eval_token_window_length": args.eval_block_size,
            "eval_predicted_tokens_per_window": (
                args.eval_block_size - 1
            ),
            "eval_batches": args.eval_batches,
            "cached_block_fht_modules": runtime[
                "cached_block_fht_modules"
            ],
            "pregelu_parameter_count": runtime[
                "pregelu_parameter_count"
            ],
        },
        "artifacts": {
            "optimization_csv": {
                "path": str(train_csv),
                "sha256": sha256(train_csv),
            },
            "validation_csv": {
                "path": str(validation_csv),
                "sha256": sha256(validation_csv),
            },
            "pregelu_states": state_artifacts,
        },
        "selection": selection,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)
    print(f"summary={summary_path}", flush=True)


def parse_updates(value: str) -> list[int]:
    parsed = [int(part) for part in value.split(",") if part.strip()]
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(
            "evaluation updates must be unique integers"
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-sha256",
        default=(
            "7e450e2f78fa33a049ea990386e0c7b8f9b139ddb174e4d1b"
            "fc76dd0ff0ebcdc"
        ),
    )
    parser.add_argument("--identity-state", required=True, type=Path)
    parser.add_argument(
        "--identity-state-sha256",
        default=(
            "2d18703bf53d62ab52a217730a9e63594c331c5a07118b3f7"
            "9069a18ff91aa01"
        ),
    )
    parser.add_argument("--accepted-state", required=True, type=Path)
    parser.add_argument(
        "--accepted-state-sha256",
        default=(
            "2abeee50138189f055a5462679412965a6c60d524a04c60afd"
            "a1cdc3877a6b75"
        ),
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest-sha256",
        default=(
            "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33"
            "f2392d9492ddf8b"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mfu-certificate", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument(
        "--evaluation-updates",
        type=parse_updates,
        default=parse_updates("0,30,60,120"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=8
    )
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.000072)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-token-seed", type=int, default=20260731)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-block-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--primary-eval-seed", type=int, default=20260717)
    parser.add_argument(
        "--confirmation-eval-seed", type=int, default=20260718
    )
    parser.add_argument("--audit-eval-seed", type=int, default=20260719)
    parser.add_argument("--minimum-ce-gain", type=float, default=0.005)
    parser.add_argument("--frame-stages", type=int, default=2)
    parser.add_argument(
        "--frame-rotation-block-size", type=int, default=32
    )
    parser.add_argument(
        "--frame-basis-block-size", type=int, default=256
    )
    parser.add_argument("--frame-basis-seed", type=int, default=161803)
    parser.add_argument("--frame-seed-offset", type=int, default=64)
    parser.add_argument(
        "--frame-coordinate-scale", type=float, default=4.0
    )
    parser.add_argument("--minimum-mfu", type=float, default=0.2)
    parser.add_argument(
        "--preflight-warmup-updates", type=int, default=2
    )
    parser.add_argument(
        "--preflight-timed-updates", type=int, default=3
    )
    parser.add_argument(
        "--preflight-safety-eval-seed", type=int, default=20260801
    )
    parser.add_argument(
        "--preflight-safety-eval-batches", type=int, default=2
    )
    parser.add_argument(
        "--preflight-max-ce-increase", type=float, default=0.1
    )
    parser.add_argument("--gemm-size", type=int, default=8192)
    parser.add_argument("--gemm-warmups", type=int, default=4)
    parser.add_argument("--gemm-trials", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=5)
    args = parser.parse_args()

    if args.minimum_mfu < 0.2:
        parser.error("--minimum-mfu must be at least 0.2")
    if args.updates <= 0 or min(args.evaluation_updates) < 0:
        parser.error("updates must be positive")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("batch sizes and accumulation must be positive")
    if args.block_size <= 1 or args.eval_block_size <= 1:
        parser.error("block sizes must exceed one")
    if args.learning_rate <= 0.0 or args.weight_decay != 0.0:
        parser.error("registered endpoint optimizer requires LR>0 and WD=0")
    if not (0.0 < args.beta1 < 1.0 and 0.0 < args.beta2 < 1.0):
        parser.error("AdamW betas must be in (0,1)")
    if args.frame_stages != 2:
        parser.error("registered endpoint protocol requires two stages")
    if args.frame_rotation_block_size != 32:
        parser.error(
            "registered endpoint protocol requires rotation block size 32"
        )
    if args.frame_basis_block_size != 256:
        parser.error(
            "registered endpoint protocol requires basis block size 256"
        )
    if args.mode == "preflight":
        run_preflight(args)
    else:
        run_scientific(args)


if __name__ == "__main__":
    main()
