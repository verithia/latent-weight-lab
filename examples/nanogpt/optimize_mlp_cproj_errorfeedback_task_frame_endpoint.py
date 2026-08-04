"""Optimize a coupled MLP activation frame at the accepted c_proj endpoint.

This is a frozen-endpoint capacity diagnostic.  It loads the accepted 124M
Muon-matched c_proj error-feedback checkpoint, freezes every learned tensor,
and updates only three fixed-basis orthogonal coordinate groups: one before
GELU and two folded around c_proj.  The extra optimization tokens are not a
same-budget pretraining result.

Short preflight and scientific runs are foreground-only.  Scientific mode
requires an immutable certificate from the exact code, checkpoint, data,
protocol, hardware, and cached path proving at least 20 percent measured MFU,
finite execution, endpoint safety, and bitwise preservation of frozen state.
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
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.mfu_preflight import (
    empirical_bf16_gemm_peak_tflops,
    estimate_active_params,
)
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon_matched_givens import MuonMatchedGivensLinear
from examples.nanogpt.optimize_mlp_bilateral_endpoint_ce import (
    ALL_LAYERS,
    autocast_context,
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


PLAN_NAME = "124m_mlp_cproj_errorfeedback_task_frame_endpoint_plan.json"
IDENTITY_AMENDMENT_NAME = (
    "124m_mlp_cproj_errorfeedback_task_frame_endpoint_identity_amendment.json"
)
PARENT_RESULT_NAME = "124m_mlp_cproj_error_feedback_0p5tpp_result.json"
INTERVENTION_RESULT_NAME = (
    "124m_mlp_cproj_hybrid_endpoint_interpolation_result.json"
)
CERTIFICATE_SCHEMA = "mai_124m_mlp_cproj_task_frame_mfu_v1"
IDENTITY_SCHEMA = "mai_124m_mlp_cproj_task_frame_identity_v1"
STATE_SCHEMA = "mai_124m_mlp_cproj_task_frame_state_v1"
RESULT_SCHEMA = "mai_124m_mlp_cproj_task_frame_endpoint_result_v1"
VARIANTS = ("full_coupled", "post_only", "pre_only", "identity")


def expected_frame_state_key(key: str) -> bool:
    return any(
        token in key
        for token in (
            ".pregelu_block_rotation.",
            ".hidden_block_rotation.",
            ".output_block_rotation.",
        )
    )


def frame_config(source: dict[str, object]) -> GPTConfig:
    values = dict(source)
    values.update(
        {
            "block_fht_mlp_activation_chart": False,
            "block_fht_mlp_pregelu_block_rotation_stages": 2,
            "block_fht_mlp_pregelu_block_rotation_size": 32,
            "block_fht_mlp_pregelu_block_rotation_basis_size": 256,
            "block_fht_mlp_pregelu_block_rotation_coordinate_scale": 4.0,
            "block_fht_mlp_pregelu_block_rotation_seed": 161803,
            "block_fht_mlp_pregelu_cache_retain_graph": False,
            "block_fht_mlp_hidden_block_rotation_stages": 2,
            "block_fht_mlp_hidden_block_rotation_size": 32,
            "block_fht_mlp_hidden_block_rotation_basis_size": 256,
            "block_fht_mlp_hidden_block_rotation_coordinate_scale": 4.0,
            "block_fht_mlp_hidden_block_rotation_seed": 314159,
            "block_fht_mlp_hidden_gain": False,
            "block_fht_mlp_output_rotation_stages": 0,
            "block_fht_mlp_output_block_rotation_stages": 4,
            "block_fht_mlp_output_block_rotation_size": 32,
            "block_fht_mlp_output_block_rotation_basis_size": 256,
            "block_fht_mlp_output_block_rotation_coordinate_scale": 4.0,
            "block_fht_mlp_output_rotation_seed": 271828,
            "block_fht_mlp_residual_output_gain": False,
        }
    )
    return GPTConfig(**values)


def frame_parameters(model: GPT) -> dict[str, torch.nn.Parameter]:
    output: dict[str, torch.nn.Parameter] = {}
    for layer, block in enumerate(model.transformer.h):
        mlp = block.mlp
        if (
            mlp.pregelu_block_rotation is None
            or mlp.hidden_block_rotation is None
            or mlp.output_block_rotation is None
        ):
            raise RuntimeError(f"layer {layer} is missing a frame group")
        if not isinstance(mlp.c_proj, MuonMatchedGivensLinear):
            raise RuntimeError(
                f"layer {layer} c_proj is not MuonMatchedGivensLinear"
            )
        output[f"layer.{layer}.pregelu_rotation"] = (
            mlp.pregelu_block_rotation.coordinates
        )
        output[f"layer.{layer}.hidden_rotation"] = (
            mlp.hidden_block_rotation.coordinates
        )
        output[f"layer.{layer}.output_rotation"] = (
            mlp.output_block_rotation.coordinates
        )
    return output


def capture_frame_state(model: GPT) -> dict[str, torch.Tensor]:
    return {
        key: parameter.detach().float().cpu().clone()
        for key, parameter in frame_parameters(model).items()
    }


@torch.no_grad()
def restore_frame_state(
    model: GPT, state: dict[str, torch.Tensor]
) -> None:
    parameters = frame_parameters(model)
    if set(parameters) != set(state):
        raise ValueError("frame state keys do not match the model")
    for key, parameter in parameters.items():
        value = state[key]
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(f"frame state shape mismatch for {key}")
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


@torch.no_grad()
def set_variant(model: GPT, state: dict[str, torch.Tensor], variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown frame variant {variant!r}")
    restore_frame_state(model, state)
    for key, parameter in frame_parameters(model).items():
        group = key.rsplit(".", 1)[-1]
        keep = (
            variant == "full_coupled"
            or (variant == "post_only" and group != "pregelu_rotation")
            or (variant == "pre_only" and group == "pregelu_rotation")
        )
        if not keep:
            parameter.zero_()


def load_frame_model(checkpoint_path: Path, device: str) -> GPT:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = frame_config(checkpoint["model_config"])
    with torch.device(device):
        model = GPT(config)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not expected_frame_state_key(key)
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            "checkpoint is incompatible with the coupled frame model: "
            f"unexpected={unexpected} invalid_missing={invalid_missing}"
        )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = frame_parameters(model)
    for parameter in parameters.values():
        parameter.requires_grad_(True)
        if torch.count_nonzero(parameter.detach()).item() != 0:
            raise RuntimeError("new frame coordinates are not identity")
    for layer, block in enumerate(model.transformer.h):
        mlp = block.mlp
        cfc_base = mlp._cfc_base_weight()
        cproj_base = mlp._cproj_base_weight()
        cfc_identity_delta = (
            mlp._materialize_charted_cfc_weight(cfc_base) - cfc_base
        ).float()
        cfc_relative = float(
            cfc_identity_delta.norm() / cfc_base.float().norm()
        )
        if (
            float(cfc_identity_delta.abs().max()) > 3e-8
            or cfc_relative > 2e-7
        ):
            raise RuntimeError(
                f"layer {layer} pre-GELU identity exceeds the registered "
                "roundoff bound"
            )
        if not torch.equal(
            mlp._materialize_charted_cproj_weight(cproj_base), cproj_base
        ):
            raise RuntimeError(
                f"layer {layer} post-GELU identity is not bitwise exact"
            )
    return model


def input_paths(root: Path, args: argparse.Namespace) -> dict[str, Path]:
    artifact_root = root / "examples/nanogpt/configs/selection_artifacts"
    return {
        "checkpoint": args.checkpoint,
        "identity_amendment": artifact_root / IDENTITY_AMENDMENT_NAME,
        "parent_result": artifact_root / PARENT_RESULT_NAME,
        "endpoint_intervention_result": artifact_root / INTERVENTION_RESULT_NAME,
        "dataset_manifest": args.data_dir / "manifest.json",
    }


def validate_input_hashes(
    root: Path, args: argparse.Namespace
) -> tuple[dict[str, Path], dict[str, str]]:
    paths = input_paths(root, args)
    expected = {
        "checkpoint": args.checkpoint_sha256,
        "identity_amendment": args.identity_amendment_sha256,
        "parent_result": args.parent_result_sha256,
        "endpoint_intervention_result": args.intervention_result_sha256,
        "dataset_manifest": args.manifest_sha256,
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    for name, digest in actual.items():
        if digest != expected[name]:
            raise ValueError(
                f"SHA-256 mismatch for {paths[name]}: "
                f"{digest} != {expected[name]}"
            )
    return paths, actual


def protocol_identity(
    args: argparse.Namespace,
    *,
    root: Path,
    input_hashes: dict[str, str],
    device_name: str,
) -> dict[str, Any]:
    plan = root / "examples/nanogpt/configs/selection_artifacts" / PLAN_NAME
    value: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA,
        "repository_commit": git_head(root),
        "source_sha256": sha256(Path(__file__).resolve()),
        "model_source_sha256": sha256(root / "examples/nanogpt/model.py"),
        "plan_sha256": sha256(plan),
        "inputs": input_hashes,
        "hardware_device": device_name,
        "structure": {
            "layers": ALL_LAYERS,
            "pregelu": [2, 32, 256, 4.0, 161803],
            "postgelu_hidden": [2, 32, 256, 4.0, 314159],
            "residual_output": [4, 32, 256, 4.0, 271828],
            "scalar_gains": False,
            "base_parameters_frozen": True,
        },
        "update": {
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
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
        },
        "evaluation": {
            "batch_size": args.eval_batch_size,
            "block_size": args.eval_block_size,
            "batches": args.eval_batches,
            "primary_seed": args.primary_eval_seed,
            "confirmation_seed": args.confirmation_eval_seed,
            "variants": list(VARIANTS),
            "minimum_full_gain": args.minimum_full_gain,
            "minimum_post_gain": args.minimum_post_gain,
            "minimum_pregelu_marginal_gain": args.minimum_pregelu_marginal_gain,
        },
        "preflight_safety": {
            "seed": args.preflight_safety_eval_seed,
            "batches": args.preflight_safety_eval_batches,
            "maximum_ce_increase": args.preflight_max_ce_increase,
        },
    }
    value["identity_sha256"] = stable_json_sha256(value)
    return value


def frozen_snapshot(
    model: GPT, chart_parameter_ids: set[int]
) -> tuple[dict[str, torch.Tensor], int]:
    chart_names = {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in chart_parameter_ids
    }
    snapshot: dict[str, torch.Tensor] = {}
    total_bytes = 0
    for name, tensor in model.state_dict().items():
        if name in chart_names:
            continue
        value = tensor.detach().cpu().clone()
        snapshot[name] = value
        total_bytes += value.numel() * value.element_size()
    return snapshot, total_bytes


def verify_frozen_snapshot(
    model: GPT,
    snapshot: dict[str, torch.Tensor],
    chart_parameter_ids: set[int],
) -> None:
    chart_names = {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in chart_parameter_ids
    }
    current = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name not in chart_names
    }
    if set(current) != set(snapshot):
        raise RuntimeError("frozen state topology changed")
    changed = [
        name
        for name, before in snapshot.items()
        if not torch.equal(current[name].detach().cpu(), before)
    ]
    if changed:
        raise RuntimeError(
            f"frozen model state changed: {changed[:8]}"
        )


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
    paths, input_hashes = validate_input_hashes(root, args)
    model = load_frame_model(args.checkpoint, args.device)
    parameters = frame_parameters(model)
    optimizer = make_optimizer(list(parameters.values()), args)
    cached_modules = prepare_frozen_base_cache(model, torch.bfloat16)
    device_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
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
        "input_paths": paths,
        "input_hashes": input_hashes,
        "cached_block_fht_modules": cached_modules,
        "frame_parameter_count": sum(
            parameter.numel() for parameter in parameters.values()
        ),
        "frame_parameter_count_by_group": {
            group: sum(
                parameter.numel()
                for key, parameter in parameters.items()
                if key.endswith(group)
            )
            for group in (
                "pregelu_rotation",
                "hidden_rotation",
                "output_rotation",
            )
        },
        "chart_parameter_ids": {id(parameter) for parameter in parameters.values()},
        "model_config": asdict(model.config),
    }
    return model, optimizer, TokenBatchSource(args.data_dir), identity, runtime


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
        raise ValueError("MFU certificate failed its performance floor")
    stability = certificate.get("stability", {})
    if stability.get("finite") is not True:
        raise ValueError("MFU certificate has non-finite execution")
    if stability.get("frozen_state_bitwise_unchanged") is not True:
        raise ValueError("MFU certificate changed frozen model state")
    if float(stability.get("ce_increase")) > float(
        stability.get("maximum_ce_increase")
    ):
        raise ValueError("MFU certificate failed endpoint CE safety")
    if certificate.get("passed") is not True:
        raise ValueError("MFU certificate is not marked passed")


def evaluate_without_frames(
    model: GPT, batches: list[torch.Tensor], device: str
) -> float:
    suspended: list[tuple[object, object, object, object]] = []
    for block in model.transformer.h:
        mlp = block.mlp
        suspended.append(
            (
                mlp,
                mlp.pregelu_block_rotation,
                mlp.hidden_block_rotation,
                mlp.output_block_rotation,
            )
        )
        mlp.pregelu_block_rotation = None
        mlp.hidden_block_rotation = None
        mlp.output_block_rotation = None
    try:
        losses: list[float] = []
        with torch.no_grad():
            for tokens in batches:
                tokens = tokens.to(device)
                inputs = tokens[:, :-1].contiguous()
                targets = tokens[:, 1:].contiguous()
                with autocast_context(device):
                    _, loss = model(inputs, targets)
                if loss is None:
                    raise RuntimeError("model returned no CE loss")
                losses.append(float(loss))
        return float(np.mean(losses))
    finally:
        for mlp, pregelu, hidden, output in suspended:
            mlp.pregelu_block_rotation = pregelu
            mlp.hidden_block_rotation = hidden
            mlp.output_block_rotation = output


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
    snapshot, snapshot_bytes = frozen_snapshot(
        model, runtime["chart_parameter_ids"]
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
            "token_sha256": tensor_sha256(torch.cat(safety_batches)),
            "maximum_ce_increase": args.preflight_max_ce_increase,
            "frozen_snapshot_bytes": snapshot_bytes,
        },
        "runtime": {
            "cached_block_fht_modules": runtime["cached_block_fht_modules"],
            "frame_parameter_count": runtime["frame_parameter_count"],
            "frame_parameter_count_by_group": runtime[
                "frame_parameter_count_by_group"
            ],
        },
        "passed": False,
    }
    error: Exception | None = None
    try:
        parent_reference_ce = evaluate_without_frames(
            model, safety_batches, args.device
        )
        initial_ce = evaluate_ce(model, safety_batches, args.device)
        identity_ce_difference = initial_ce - parent_reference_ce
        if abs(identity_ce_difference) > 1e-6:
            raise RuntimeError(
                "zero-coordinate complete-model CE identity failed: "
                f"{identity_ce_difference:+.9f}"
            )
        peak_tflops = empirical_bf16_gemm_peak_tflops(
            args.gemm_size, args.gemm_warmups, args.gemm_trials
        )
        durations: list[float] = []
        losses: list[float] = []
        torch.cuda.reset_peak_memory_stats()
        for update in range(updates):
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss = run_update(model, optimizer, source, indices[update], args)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not math.isfinite(loss):
                raise RuntimeError("non-finite preflight loss")
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
        tokens_per_second = tokens_per_update * len(durations) / sum(durations)
        active_params = estimate_active_params(runtime["model_config"])
        model_tflops = 6.0 * active_params * tokens_per_second / 1e12
        mfu = model_tflops / peak_tflops
        final_ce = evaluate_ce(model, safety_batches, args.device)
        ce_increase = final_ce - initial_ce
        finite = all(
            math.isfinite(value)
            for value in [initial_ce, final_ce, ce_increase, mfu, *losses]
        ) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in frame_parameters(model).values()
        )
        verify_frozen_snapshot(
            model, snapshot, runtime["chart_parameter_ids"]
        )
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
                "initial_ce": initial_ce,
                "parent_reference_ce": parent_reference_ce,
                "identity_ce_difference": identity_ce_difference,
                "identity_ce_absolute_tolerance": 1e-6,
                "final_ce": final_ce,
                "ce_increase": ce_increase,
                "finite": bool(finite),
                "frozen_state_bitwise_unchanged": True,
            }
        )
        certificate["passed"] = bool(
            mfu >= args.minimum_mfu
            and finite
            and ce_increase <= args.preflight_max_ce_increase
        )
        if not certificate["passed"]:
            raise RuntimeError(
                "preflight rejected: "
                f"mfu={mfu:.2%}, finite={finite}, "
                f"ce_increase={ce_increase:+.6f}"
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


def select_decision(
    rows: list[dict[str, object]],
    minimum_full_gain: float,
    minimum_post_gain: float,
    minimum_pregelu_marginal_gain: float,
) -> dict[str, object]:
    table = {
        (str(row["split"]), str(row["variant"]), int(row["update"])): float(
            row["ce"]
        )
        for row in rows
    }
    splits = ("primary", "confirmation")
    updates = sorted(
        {
            update
            for split, variant, update in table
            if split == "primary" and variant == "full_coupled"
        }
    )
    if not updates or 0 not in updates:
        raise ValueError("validation rows are missing identity update")
    for split in splits:
        for variant in VARIANTS:
            if any((split, variant, update) not in table for update in updates):
                raise ValueError("validation rows are incomplete")
    selected_update = min(
        updates, key=lambda update: table[("primary", "full_coupled", update)]
    )
    identity_ce = {
        split: table[(split, "identity", selected_update)] for split in splits
    }
    full_ce = {
        split: table[(split, "full_coupled", selected_update)] for split in splits
    }
    post_ce = {
        split: table[(split, "post_only", selected_update)] for split in splits
    }
    pre_ce = {
        split: table[(split, "pre_only", selected_update)] for split in splits
    }
    full_gain = {
        split: identity_ce[split] - full_ce[split] for split in splits
    }
    post_gain = {
        split: identity_ce[split] - post_ce[split] for split in splits
    }
    pre_gain = {
        split: identity_ce[split] - pre_ce[split] for split in splits
    }
    pregelu_marginal = {
        split: post_ce[split] - full_ce[split] for split in splits
    }
    coupled = (
        selected_update > 0
        and all(value >= minimum_full_gain for value in full_gain.values())
        and all(value >= minimum_post_gain for value in post_gain.values())
        and all(
            value >= minimum_pregelu_marginal_gain
            for value in pregelu_marginal.values()
        )
    )
    post_only = (
        selected_update > 0
        and all(value >= minimum_post_gain for value in post_gain.values())
    )
    decision = (
        "COUPLED_ACTIVATION_FRAME_CAPACITY"
        if coupled
        else (
            "POST_CPROJ_FRAME_ONLY_CAPACITY"
            if post_only
            else "LOCAL_FRAME_CAPACITY_INSUFFICIENT"
        )
    )
    return {
        "selected_update": selected_update,
        "identity_ce": identity_ce,
        "full_coupled_ce": full_ce,
        "post_only_ce": post_ce,
        "pre_only_ce": pre_ce,
        "full_coupled_gain": full_gain,
        "post_only_gain": post_gain,
        "pre_only_gain": pre_gain,
        "pregelu_marginal_over_post_only": pregelu_marginal,
        "thresholds": {
            "minimum_full_gain": minimum_full_gain,
            "minimum_post_gain": minimum_post_gain,
            "minimum_pregelu_marginal_gain": minimum_pregelu_marginal_gain,
        },
        "decision": decision,
        "automatic_training_authorized": False,
    }


def run_scientific(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("scientific output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    model, optimizer, source, identity, runtime = load_runtime(args)
    if args.mfu_certificate is None:
        raise ValueError("--mfu-certificate is required in run mode")
    certificate = json.loads(args.mfu_certificate.read_text())
    validate_mfu_certificate(certificate, identity, args.minimum_mfu)
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
        state = capture_frame_state(model)
        state_path = output / f"frame_state_step_{update:04d}.pt"
        torch.save(
            {"schema_version": STATE_SCHEMA, "update": update, "state": state},
            state_path,
        )
        state_artifacts[str(update)] = {
            "path": str(state_path),
            "sha256": sha256(state_path),
        }
        try:
            for variant in VARIANTS:
                set_variant(model, state, variant)
                for split, batches in validation.items():
                    ce = evaluate_ce(model, batches, args.device)
                    validation_rows.append(
                        {
                            "update": update,
                            "split": split,
                            "variant": variant,
                            "seed": split_seeds[split],
                            "token_sha256": validation_digests[split],
                            "ce": ce,
                        }
                    )
                    print(
                        f"eval update={update} split={split} "
                        f"variant={variant} ce={ce:.6f}",
                        flush=True,
                    )
        finally:
            restore_frame_state(model, state)

    try:
        evaluate(0)
        for update in range(1, args.updates + 1):
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss = run_update(
                model, optimizer, source, train_indices[update - 1], args
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not math.isfinite(loss):
                raise RuntimeError("non-finite scientific loss")
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
        validation_rows,
        args.minimum_full_gain,
        args.minimum_post_gain,
        args.minimum_pregelu_marginal_gain,
    )
    train_csv = output / "optimization.csv"
    validation_csv = output / "validation.csv"
    write_csv(train_csv, train_rows)
    write_csv(validation_csv, validation_rows)
    root: Path = runtime["root"]
    plan = root / "examples/nanogpt/configs/selection_artifacts" / PLAN_NAME
    summary = {
        "schema_version": RESULT_SCHEMA,
        "repository_commit": git_head(root),
        "command": sys.argv,
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "model_source": {
            "path": str(root / "examples/nanogpt/model.py"),
            "sha256": sha256(root / "examples/nanogpt/model.py"),
        },
        "plan": {"path": str(plan), "sha256": sha256(plan)},
        "inputs": {
            name: {"path": str(runtime["input_paths"][name]), "sha256": digest}
            for name, digest in runtime["input_hashes"].items()
        },
        "mfu_certificate": {
            "path": str(args.mfu_certificate),
            "sha256": sha256(args.mfu_certificate),
            "mfu_fraction": certificate["measurement"]["mfu_fraction"],
        },
        "protocol_identity": identity,
        "protocol": {
            "updates": args.updates,
            "diagnostic_optimization_tokens": (
                args.updates
                * args.batch_size
                * args.gradient_accumulation_steps
                * args.block_size
            ),
            "evaluation_updates": sorted(evaluation_updates),
            "train_token_seed": args.train_token_seed,
            "train_indices_sha256": indices_digest(train_indices),
            "validation_seeds": split_seeds,
            "validation_token_sha256": validation_digests,
            "eval_batch_size": args.eval_batch_size,
            "eval_block_size": args.eval_block_size,
            "eval_batches": args.eval_batches,
            "cached_block_fht_modules": runtime["cached_block_fht_modules"],
            "frame_parameter_count": runtime["frame_parameter_count"],
            "frame_parameter_count_by_group": runtime[
                "frame_parameter_count_by_group"
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
            "frame_states": state_artifacts,
        },
        "selection": selection,
        "interpretation_limit": (
            "extra endpoint optimization tokens; not a same-budget "
            "pretraining result and no automatic causal run is authorized"
        ),
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
            "7e614ddfdb6fd53d95c8cb790a70deb334f055aab42f6c8c9"
            "a17312071755063"
        ),
    )
    parser.add_argument(
        "--identity-amendment-sha256",
        default=(
            "67abd0ecdfdee12c149c8fb2c1c85ec05689eef248c90ad44"
            "4e6726bf7bcffb6"
        ),
    )
    parser.add_argument(
        "--parent-result-sha256",
        default=(
            "272f8709ddc805175542b2f163398d9823141aca0e1f6026f"
            "699a51bc5af87df"
        ),
    )
    parser.add_argument(
        "--intervention-result-sha256",
        default=(
            "b3da66aa6c041d177a78613192b12eab3fe38a3b3d08b0d5"
            "8ad9672ff526e628"
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
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.000072)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-token-seed", type=int, default=20260882)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-block-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--primary-eval-seed", type=int, default=20260880)
    parser.add_argument("--confirmation-eval-seed", type=int, default=20260881)
    parser.add_argument("--minimum-full-gain", type=float, default=0.0075)
    parser.add_argument("--minimum-post-gain", type=float, default=0.005)
    parser.add_argument(
        "--minimum-pregelu-marginal-gain", type=float, default=0.0025
    )
    parser.add_argument("--minimum-mfu", type=float, default=0.2)
    parser.add_argument("--preflight-warmup-updates", type=int, default=2)
    parser.add_argument("--preflight-timed-updates", type=int, default=3)
    parser.add_argument(
        "--preflight-safety-eval-seed", type=int, default=20260883
    )
    parser.add_argument(
        "--preflight-safety-eval-batches", type=int, default=2
    )
    parser.add_argument(
        "--preflight-max-ce-increase", type=float, default=0.05
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
    if max(args.evaluation_updates) > args.updates:
        parser.error("evaluation update exceeds total updates")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("batch and accumulation must be positive")
    if args.eval_batches <= 0 or args.preflight_safety_eval_batches <= 0:
        parser.error("evaluation batches must be positive")
    if not args.device.startswith("cuda"):
        parser.error("the registered diagnostic requires CUDA")

    if args.mode == "preflight":
        run_preflight(args)
    else:
        run_scientific(args)


if __name__ == "__main__":
    main()
