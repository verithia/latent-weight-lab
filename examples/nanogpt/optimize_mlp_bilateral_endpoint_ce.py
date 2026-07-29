"""Optimize only the exact bilateral MLP chart at a frozen LM endpoint.

This is a causal endpoint diagnostic.  It loads the terminal plain-c_proj
checkpoint, freezes every checkpoint parameter and generated BlockFHT base,
then updates only the production bilateral chart against next-token CE.

Short preflight and scientific runs are intentionally foreground processes.
The preflight measures the exact update path and writes an immutable >=20%
MFU certificate; the scientific mode refuses to run without a matching
certificate from the same code commit, checkpoint, data, GPU, and protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.nanogpt.analyze_mlp_activation_chart_oracle import tensor_sha256
from examples.nanogpt.analyze_mlp_bilateral_endpoint_ce_oracle import (
    capture_chart_state,
)
from examples.nanogpt.analyze_mlp_chart_gradient_alignment import (
    chart_parameters,
    load_chart_model,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.mfu_preflight import (
    empirical_bf16_gemm_peak_tflops,
    estimate_active_params,
)
from examples.nanogpt.model import GPT
from examples.nanogpt.train import TokenBatchSource, move_batch_to_device
from latent_weight_lab.block_fht import (
    BlockFHTLinear,
    prepare_block_fht_weight_cache,
)


ALL_LAYERS = list(range(12))
PLAN_NAME = "124m_mlp_bilateral_task_ce_endpoint_plan.json"


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


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def indices_digest(indices: torch.Tensor) -> str:
    values = (
        indices.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
        .numpy()
    )
    digest = hashlib.sha256()
    digest.update(b"mai_endpoint_train_indices_v1\0")
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(values, dtype="<i8").tobytes())
    return digest.hexdigest()


def make_train_indices(
    data_dir: Path,
    *,
    updates: int,
    accumulation: int,
    batch_size: int,
    block_size: int,
    seed: int,
) -> torch.Tensor:
    data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randint(
        len(data) - int(block_size),
        (int(updates), int(accumulation), int(batch_size)),
        generator=generator,
    )


def protocol_identity(
    args: argparse.Namespace,
    *,
    root: Path,
    checkpoint_sha256: str,
    manifest_sha256: str,
    device_name: str,
) -> dict[str, Any]:
    script = Path(__file__).resolve()
    plan = (
        root
        / "examples/nanogpt/configs/selection_artifacts"
        / PLAN_NAME
    )
    value: dict[str, Any] = {
        "schema_version": "mai_124m_mlp_bilateral_task_ce_identity_v1",
        "repository_commit": git_head(root),
        "source_sha256": sha256(script),
        "plan_sha256": sha256(plan),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "hardware_device": device_name,
        "chart": {
            "layers": ALL_LAYERS,
            "initial_output_log_gain": args.initial_output_log_gain,
            "implementation": "production_cached_bilateral_cproj",
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
            "base_parameters_frozen": True,
        },
        "evaluation": {
            "batch_size": args.eval_batch_size,
            "token_window_length": args.eval_block_size,
            "batches": args.eval_batches,
            "primary_seed": args.primary_eval_seed,
            "confirmation_seed": args.confirmation_eval_seed,
            "minimum_ce_gain": args.minimum_ce_gain,
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
    if certificate.get("schema_version") != (
        "mai_124m_mlp_bilateral_task_ce_mfu_v1"
    ):
        raise ValueError("MFU certificate schema is incompatible")
    if certificate.get("identity") != identity:
        raise ValueError("MFU certificate does not match this exact run")
    measured = float(certificate.get("measurement", {}).get("mfu_fraction"))
    if not math.isfinite(measured) or measured < minimum_fraction:
        raise ValueError(
            f"MFU certificate failed: {measured:.2%} < "
            f"{minimum_fraction:.2%}"
        )
    if certificate.get("passed") is not True:
        raise ValueError("MFU certificate is not marked passed")
    stability = certificate.get("stability", {})
    increase = float(stability.get("ce_increase"))
    maximum = float(stability.get("maximum_ce_increase"))
    if not math.isfinite(increase) or increase > maximum:
        raise ValueError(
            "MFU certificate failed its endpoint CE safety gate"
        )


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


def prepare_frozen_base_cache(
    model: GPT, dtype: torch.dtype
) -> int:
    prepare_block_fht_weight_cache(model, dtype=dtype)
    cached = 0
    for module in model.modules():
        if not isinstance(module, BlockFHTLinear):
            continue
        weight = module._cached_weight
        if weight is None:
            raise RuntimeError(
                "endpoint diagnostic requires every BlockFHT weight cached"
            )
        weight.requires_grad_(False)
        weight.grad = None
        cached += 1
    return cached


def clear_frozen_base_cache(model: GPT) -> None:
    for module in model.modules():
        if isinstance(module, BlockFHTLinear):
            module._cached_weight = None


def prepare_chart_caches(model: GPT) -> None:
    for block in model.transformer.h:
        block.mlp.prepare_charted_cproj_cache()
        if block.mlp._cached_charted_cproj_weight is None:
            raise RuntimeError("failed to prepare a charted c_proj cache")


def flush_chart_caches(model: GPT) -> None:
    for block in model.transformer.h:
        block.mlp.flush_charted_cproj_cache(
            project_base_gradient=False
        )


def discard_chart_caches(model: GPT) -> None:
    for block in model.transformer.h:
        block.mlp._cached_charted_cproj_weight = None


def make_optimizer(
    parameters: list[torch.nn.Parameter],
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    fused_available = "fused" in inspect.signature(
        torch.optim.AdamW
    ).parameters
    if args.device.startswith("cuda") and not fused_available:
        raise RuntimeError("CUDA endpoint diagnostic requires fused AdamW")
    return torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        betas=(float(args.beta1), float(args.beta2)),
        weight_decay=float(args.weight_decay),
        fused=bool(args.device.startswith("cuda")),
    )


def autocast_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def run_update(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    source: TokenBatchSource,
    indices: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    prepare_chart_caches(model)
    losses: list[float] = []
    completed = False
    try:
        for microbatch in range(args.gradient_accumulation_steps):
            x_cpu, y_cpu = source.get_batch_cpu(
                "train",
                args.batch_size,
                args.block_size,
                indices=indices[microbatch],
            )
            x, y = move_batch_to_device(
                x_cpu, y_cpu, args.device
            )
            with autocast_context(args.device):
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("model returned no CE loss")
                scaled = loss / args.gradient_accumulation_steps
            scaled.backward()
            losses.append(float(loss.detach()))
        flush_chart_caches(model)
        optimizer.step()
        completed = True
    finally:
        if not completed:
            discard_chart_caches(model)
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_ce(
    model: GPT,
    batches: list[torch.Tensor],
    device: str,
) -> float:
    prepare_chart_caches(model)
    losses: list[float] = []
    try:
        for tokens in batches:
            tokens = tokens.to(device)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            with autocast_context(device):
                _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("model returned no CE loss")
            losses.append(float(loss))
    finally:
        discard_chart_caches(model)
    return float(np.mean(losses))


def select_decision(
    validation_rows: list[dict[str, object]],
    minimum_gain: float,
) -> dict[str, object]:
    primary = [
        row for row in validation_rows if row["split"] == "primary"
    ]
    confirmation = {
        int(row["update"]): row
        for row in validation_rows
        if row["split"] == "confirmation"
    }
    identity_primary = next(
        float(row["ce"]) for row in primary if int(row["update"]) == 0
    )
    identity_confirmation = float(confirmation[0]["ce"])
    selected = min(primary, key=lambda row: float(row["ce"]))
    selected_update = int(selected["update"])
    selected_confirmation = confirmation[selected_update]
    primary_gain = identity_primary - float(selected["ce"])
    confirmation_gain = (
        identity_confirmation - float(selected_confirmation["ce"])
    )
    positive = (
        selected_update > 0
        and primary_gain >= minimum_gain
        and confirmation_gain >= minimum_gain
    )
    return {
        "selected_update": selected_update,
        "identity_primary_ce": identity_primary,
        "selected_primary_ce": float(selected["ce"]),
        "primary_gain": primary_gain,
        "identity_confirmation_ce": identity_confirmation,
        "selected_confirmation_ce": float(
            selected_confirmation["ce"]
        ),
        "confirmation_gain": confirmation_gain,
        "minimum_gain": minimum_gain,
        "decision": (
            "POSITIVE_TASK_CONDITIONED_CHART_CAPACITY"
            if positive
            else "REJECT_BILATERAL_CHART_TASK_CAPACITY"
        ),
    }


def load_runtime(
    args: argparse.Namespace,
) -> tuple[
    GPT,
    torch.optim.Optimizer,
    TokenBatchSource,
    dict[str, Any],
    dict[str, Any],
]:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required")
    root = Path(__file__).resolve().parents[2]
    checkpoint_sha256 = sha256(args.checkpoint)
    manifest = args.data_dir / "manifest.json"
    manifest_sha256 = sha256(manifest)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("plain-cproj checkpoint SHA-256 mismatch")
    if manifest_sha256 != args.manifest_sha256:
        raise ValueError("dataset manifest SHA-256 mismatch")
    model = load_chart_model(
        args.checkpoint,
        args.device,
        ALL_LAYERS,
        args.initial_output_log_gain,
    )
    model.eval()
    parameters = list(chart_parameters(model, ALL_LAYERS).values())
    optimizer = make_optimizer(parameters, args)
    cached_modules = prepare_frozen_base_cache(
        model, torch.bfloat16
    )
    identity = protocol_identity(
        args,
        root=root,
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha256,
        device_name=torch.cuda.get_device_name(0),
    )
    runtime = {
        "root": root,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_path": manifest,
        "manifest_sha256": manifest_sha256,
        "cached_block_fht_modules": cached_modules,
        "chart_parameter_count": sum(
            parameter.numel() for parameter in parameters
        ),
        "model_config": asdict(model.config),
    }
    return model, optimizer, TokenBatchSource(args.data_dir), identity, runtime


def run_preflight(args: argparse.Namespace) -> None:
    output = args.output.resolve()
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
    safety_digest = tensor_sha256(torch.cat(safety_batches))
    certificate: dict[str, Any] = {
        "schema_version": "mai_124m_mlp_bilateral_task_ce_mfu_v1",
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
            "token_sha256": safety_digest,
            "maximum_ce_increase": args.preflight_max_ce_increase,
        },
        "runtime": {
            "cached_block_fht_modules": runtime[
                "cached_block_fht_modules"
            ],
            "chart_parameter_count": runtime["chart_parameter_count"],
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
        discard_chart_caches(model)
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
    validation = {
        "primary": fixed_validation_batches(
            args.data_dir,
            args.eval_batch_size,
            args.eval_block_size,
            args.eval_batches,
            args.primary_eval_seed,
        ),
        "confirmation": fixed_validation_batches(
            args.data_dir,
            args.eval_batch_size,
            args.eval_block_size,
            args.eval_batches,
            args.confirmation_eval_seed,
        ),
    }
    validation_digests = {
        name: tensor_sha256(torch.cat(batches))
        for name, batches in validation.items()
    }
    evaluation_updates = set(args.evaluation_updates)
    if 0 not in evaluation_updates or args.updates not in evaluation_updates:
        raise ValueError("evaluation updates must include 0 and final update")
    train_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    state_artifacts: dict[str, dict[str, str]] = {}

    def evaluate(update: int) -> None:
        state_path = output / f"chart_state_step_{update:04d}.pt"
        torch.save(
            {
                "schema_version": (
                    "mai_124m_mlp_bilateral_task_ce_chart_state_v1"
                ),
                "update": update,
                "state": capture_chart_state(model, ALL_LAYERS),
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
                    "seed": (
                        args.primary_eval_seed
                        if split == "primary"
                        else args.confirmation_eval_seed
                    ),
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
        discard_chart_caches(model)
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
            "mai_124m_mlp_bilateral_task_ce_endpoint_result_v1"
        ),
        "repository_commit": git_head(root),
        "command": sys.argv,
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "plan": {"path": str(plan), "sha256": sha256(plan)},
        "plain_cproj": {
            "path": str(args.checkpoint),
            "sha256": runtime["checkpoint_sha256"],
        },
        "dataset_manifest": {
            "path": str(runtime["manifest_path"]),
            "sha256": runtime["manifest_sha256"],
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
            "primary_eval_seed": args.primary_eval_seed,
            "confirmation_eval_seed": args.confirmation_eval_seed,
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
            "chart_parameter_count": runtime["chart_parameter_count"],
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
            "chart_states": state_artifacts,
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
    parser.add_argument("--train-token-seed", type=int, default=20260729)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-block-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--primary-eval-seed", type=int, default=20260717)
    parser.add_argument(
        "--confirmation-eval-seed", type=int, default=20260718
    )
    parser.add_argument("--minimum-ce-gain", type=float, default=0.005)
    parser.add_argument(
        "--initial-output-log-gain", type=float, default=0.0
    )
    parser.add_argument("--minimum-mfu", type=float, default=0.2)
    parser.add_argument(
        "--preflight-warmup-updates", type=int, default=2
    )
    parser.add_argument(
        "--preflight-timed-updates", type=int, default=3
    )
    parser.add_argument(
        "--preflight-safety-eval-seed", type=int, default=20260730
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
    if args.mode == "preflight":
        run_preflight(args)
    else:
        run_scientific(args)


if __name__ == "__main__":
    main()
