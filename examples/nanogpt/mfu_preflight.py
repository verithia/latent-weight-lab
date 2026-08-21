#!/usr/bin/env python3
"""Measure a short real training run and enforce a minimum model-FLOP utilization.

The preflight deliberately runs the configured model, generator, optimizer and
regularizers against the configured dataset.  It is therefore a launch gate,
not a synthetic kernel benchmark or a post-hoc estimate.  The denominator is
an empirical BF16 tensor-core GEMM peak measured on the selected, otherwise
idle GPU during the same preflight.  The certificate records both the model
throughput and calibration so it remains auditable when hardware changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.mai_selection_artifacts import (
    POLICY_VERSION as MAI_SELECTION_POLICY_VERSION,
    validate_v2_launch_config,
)


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    return value


def execution_provenance(config_path: Path, source: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    paths = (
        root / "examples/nanogpt/mfu_preflight.py",
        root / "examples/nanogpt/train.py",
        root / "examples/nanogpt/model.py",
        root / "examples/nanogpt/muon.py",
        root / "examples/nanogpt/muon_int8_lattice.py",
    )
    data_manifest = Path(source["data_dir"]) / "manifest.json"
    return {
        "git_commit": commit,
        "entrypoint": "examples.nanogpt.mfu_preflight",
        "literal_command": [sys.executable, *sys.argv],
        "config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "dataset_manifest": {
            "path": str(data_manifest),
            "sha256": hashlib.sha256(data_manifest.read_bytes()).hexdigest(),
        },
    }


def estimate_active_params(config: dict[str, Any]) -> int:
    """GPT parameter count used by the conventional 6N model-FLOP estimate."""
    n_layer = int(config["n_layer"])
    n_embd = int(config["n_embd"])
    vocab_size = int(config.get("vocab_size", 50304))
    block_size = int(config["block_size"])
    num_experts = int(config.get("moe_num_experts", 0))
    if num_experts > 0:
        top_k = int(config["moe_top_k"])
        hidden_multiplier = int(config["moe_expert_hidden_multiplier"])
        shared = (
            vocab_size * n_embd
            + block_size * n_embd
            + n_layer * (4 * n_embd * n_embd + 2 * n_embd)
            + n_embd
        )
        single_expert = 2 * hidden_multiplier * n_embd * n_embd
        router = n_layer * num_experts * n_embd
        return int(shared + router + n_layer * top_k * single_expert)
    # GPT decoder block: QKV, projection, two FFN linears and two layer norms.
    per_block = 12 * n_embd * n_embd + 13 * n_embd
    embeddings = vocab_size * n_embd + block_size * n_embd
    final_ln = 2 * n_embd
    return int(embeddings + n_layer * per_block + final_ln)


def parse_perf_rows(text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        if not line.startswith("perf "):
            continue
        values = {key: float(value) for key, value in re.findall(r"(\w+)=([0-9.]+)", line)}
        if {"iter", "tokens_per_s", "iter_ms"} <= values.keys():
            rows.append(values)
    return rows


def parse_snapshot_elapsed_seconds(text: str) -> list[float]:
    return [
        float(value)
        for value in re.findall(
            r"parameter trajectory snapshot .* elapsed_s=([0-9.]+)",
            text,
        )
    ]


def parse_optimizer_probe_steps(text: str) -> list[int]:
    """Return optimizer probes whose serialization completed in the timed loop."""
    return [
        int(value)
        for value in re.findall(r"optimizer probe step=(\d+) path=", text)
    ]


def parse_training_loss_values(text: str) -> list[float]:
    """Read only the explicit iteration/evaluation loss fields."""
    token = r"(?:nan|[-+]?inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
    values: list[float] = []
    for line in text.splitlines():
        iteration = re.match(
            rf"^iter\s+\d+:\s+loss\s+({token})(?:,|\s|$)",
            line,
            flags=re.IGNORECASE,
        )
        if iteration is not None:
            values.append(float(iteration.group(1)))
            continue
        evaluation = re.match(
            rf"^step\s+\d+:\s+train loss\s+({token}),\s+"
            rf"val loss\s+({token})(?:\s|$)",
            line,
            flags=re.IGNORECASE,
        )
        if evaluation is not None:
            values.extend(
                (float(evaluation.group(1)), float(evaluation.group(2)))
            )
    return values


def parse_feedback_cap_events(text: str) -> list[dict[str, Any]]:
    """Parse the once-per-update aggregate emitted when the cap is active."""
    rows: list[dict[str, Any]] = []
    prefix = "muon_matched_givens_feedback_cap "
    for line in text.splitlines():
        if line.startswith(prefix):
            value = json.loads(line[len(prefix) :])
            if not isinstance(value, dict):
                raise ValueError("feedback-cap event must be a JSON object")
            rows.append(value)
    return rows


def parse_stochastic_retraction_events(text: str) -> list[dict[str, Any]]:
    """Parse one request-energy-weighted aggregate per optimizer update."""
    rows: list[dict[str, Any]] = []
    prefix = "pair_vq_stochastic_retraction "
    for line in text.splitlines():
        if line.startswith(prefix):
            value = json.loads(line[len(prefix) :])
            if not isinstance(value, dict):
                raise ValueError("stochastic-retraction event must be a JSON object")
            rows.append(value)
    return rows


def parse_pair_vq_persistent_training_bytes(text: str) -> list[int]:
    return [
        int(value.replace(",", ""))
        for value in re.findall(
            r"^mlp_pair_vq: .* persistent_training_bytes=([0-9,]+) ",
            text,
            flags=re.MULTILINE,
        )
    ]


def empirical_bf16_gemm_peak_tflops(size: int, warmups: int, trials: int) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("MFU preflight requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected GPU does not support BF16; cannot issue BF16 MFU certificate")
    dtype = torch.bfloat16
    left = torch.randn((size, size), device="cuda", dtype=dtype)
    right = torch.randn((size, size), device="cuda", dtype=dtype)
    for _ in range(warmups):
        torch.mm(left, right)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(trials):
        torch.mm(left, right)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if elapsed <= 0.0:
        raise RuntimeError("invalid BF16 GEMM calibration time")
    # One dense GEMM costs 2mnk FLOPs.
    return (2.0 * size * size * size * trials) / elapsed / 1e12


def verify_native_block_fht_extension(
    config: dict[str, Any],
    loader: Any | None = None,
) -> dict[str, Any]:
    """Fail before timing if a registered CUDA BlockFHT run would fall back."""
    required = (
        config.get("method") == "block_fht"
        and config.get("device", "cuda") == "cuda"
    )
    if not required:
        return {"required": False, "loaded": None, "module": None}
    if loader is None:
        from latent_weight_lab.block_fht import _load_block_fht_ext

        loader = _load_block_fht_ext
    extension = loader()
    if extension is None:
        raise RuntimeError(
            "native BlockFHT CUDA extension did not load; refusing the MFU "
            "gate instead of measuring the eager fallback"
        )
    return {
        "required": True,
        "loaded": True,
        "module": getattr(extension, "__name__", type(extension).__name__),
    }


def make_preflight_config(
    source: dict[str, Any],
    temporary_out: Path,
    warmups: int,
    timed: int,
    *,
    include_diagnostic_io: bool = False,
) -> dict[str, Any]:
    config = dict(source)
    atlas_start_steps = tuple(
        int(step)
        for step in source.get(
            "block_fht_attn_cayley_atlas_start_steps", ()
        )
    )
    task_frame_start_iter = int(
        source.get("block_fht_mlp_task_frame_start_iter", 0)
    )
    # The scientific boundaries are far outside a short scratch horizon. Use
    # one compact warmup step per registered chart, then time only the final
    # (worst-cost) cumulative atlas. The scientific config itself is immutable.
    effective_warmups = max(warmups, len(atlas_start_steps))
    # A preflight is never a scientific result.  Keep the real train path but
    # avoid checkpoint/evaluation overhead and any deterministic-run policy
    # intended for registered results.
    config.pop("prelaunch_provenance_requirements", None)
    # The immutable source config is validated before this transformation.
    # The short scratch run deliberately changes checkpoint/evaluation policy,
    # so it must not masquerade as a registered scientific rung.
    config.pop("mai_ladder_policy_version", None)
    config["registered_resume_determinism_required"] = False
    # A registered candidate can remain launch-blocked until this exact gate
    # passes.  The scratch copy must still be executable so the gate can make
    # that decision; never mutate the immutable scientific source mapping.
    config["launch_ready"] = True
    config["out_dir"] = str(temporary_out)
    config["init_from"] = "scratch"
    config["max_iters"] = effective_warmups + timed
    config["lr_decay_iters"] = max(
        effective_warmups + timed,
        int(config.get("lr_decay_iters", 1)),
    )
    config["eval_interval"] = effective_warmups + timed + 100
    config["eval_iters"] = 1
    config["fixed_eval_indices"] = False
    config["eval_seed"] = None
    config["save_checkpoint"] = False
    config["checkpoint_history"] = False
    if config.get("pair_vq_dense_shadow_replay"):
        config["pair_vq_dense_shadow_result"] = str(
            temporary_out / "pair_vq_dense_shadow_preflight.json"
        )
        if config.get("pair_vq_dense_shadow_functional_result"):
            config["pair_vq_dense_shadow_functional_result"] = str(
                temporary_out / "pair_vq_functional_oracle_preflight.json"
            )
    if atlas_start_steps:
        config["block_fht_attn_cayley_atlas_start_steps"] = list(
            range(len(atlas_start_steps))
        )
    if task_frame_start_iter > 0:
        # Time the active scientific path, not the cheaper held-identity
        # prefix. Update 0 remains the warmup; every timed update executes all
        # three frame VJPs and AdamW coordinate updates.
        config["block_fht_mlp_task_frame_start_iter"] = effective_warmups
    scratch_feedback_cap = source.get(
        "mfu_preflight_error_feedback_max_nominal_steps"
    )
    if scratch_feedback_cap is not None:
        # The scientific scalar remains immutable in ``source``.  A compact
        # performance-only horizon cannot naturally build hundreds of nominal
        # steps of residual state, so use the registered scratch threshold to
        # execute and time the same active norm/rescale kernel.
        config[
            "block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps"
        ] = float(scratch_feedback_cap)
    if not include_diagnostic_io:
        # Parameter-trajectory I/O is normally a scientific sampling side
        # effect, not part of the steady-state compute gate.  A diagnostic can
        # explicitly request the stricter path below when snapshot/probe I/O
        # is itself frequent enough to affect end-to-end throughput.
        config["trajectory_snapshot_interval"] = 0
        config["optimizer_probe_steps"] = None
    # One-shot pullback diagnostics write scientific calibration artifacts.
    # Keep the repaired optimizer active, but suppress that side effect during
    # the performance-only scratch run.
    config["block_fht_cproj_product_fht_pullback_probe"] = False
    config["block_fht_cproj_product_fht_pullback_probe_output"] = None
    config["perf_profile"] = True
    # Strict diagnostic-I/O accounting needs every update row, including the
    # update-0 optimizer probe. Evaluation time is emitted separately and
    # subtracted by the certificate calculation below.
    config["perf_warmup_iters"] = (
        0 if include_diagnostic_io else effective_warmups
    )
    config["perf_log_interval"] = 1
    config["log_interval"] = 1
    return config


def task_frame_preflight_metadata(
    source: dict[str, Any],
    effective_warmup_updates: int,
) -> dict[str, Any]:
    """Describe the immutable scientific and transformed scratch boundaries."""
    task_frame_start_iter = int(
        source.get("block_fht_mlp_task_frame_start_iter", 0)
    )
    return {
        "scientific_task_frame_start_iter": task_frame_start_iter,
        "scratch_task_frame_start_iter": (
            effective_warmup_updates if task_frame_start_iter > 0 else 0
        ),
        "timed_task_frame_active": bool(task_frame_start_iter > 0),
    }


def feedback_cap_preflight_metadata(source: dict[str, Any]) -> dict[str, Any]:
    scientific = source.get(
        "block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps"
    )
    scratch = source.get(
        "mfu_preflight_error_feedback_max_nominal_steps",
        scientific,
    )
    return {
        "scientific_feedback_cap_nominal_steps": scientific,
        "scratch_feedback_cap_nominal_steps": scratch,
        "feedback_cap_activity_required": bool(
            source.get("mfu_preflight_require_feedback_cap_active", False)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-fraction", type=float, required=True)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--timed-updates", type=int, default=3)
    parser.add_argument("--gemm-size", type=int, default=8192)
    parser.add_argument("--gemm-warmups", type=int, default=4)
    parser.add_argument("--gemm-trials", type=int, default=8)
    parser.add_argument(
        "--include-diagnostic-io",
        action="store_true",
        help=(
            "retain registered trajectory snapshots and optimizer probes, "
            "then charge snapshot serialization to effective iteration time"
        ),
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        help=(
            "optional durable copy of the complete scratch-training log; "
            "use for directly polled stability diagnostics"
        ),
    )
    args = parser.parse_args()
    if args.min_fraction < 0.20:
        parser.error("--min-fraction must be at least 0.20")
    if args.warmup_updates < 1 or args.timed_updates < 2:
        parser.error("need at least one warmup and two timed updates")
    if args.gemm_size < 1024 or args.gemm_size % 256:
        parser.error("--gemm-size must be a multiple of 256 and at least 1024")
    if args.gemm_warmups < 1 or args.gemm_trials < 2:
        parser.error("GEMM calibration needs warmups and at least two trials")

    config_path = args.config.resolve()
    source = load_json_object(config_path)
    if source.get("mai_ladder_policy_version") == MAI_SELECTION_POLICY_VERSION:
        validate_v2_launch_config(source)
    required = source.get("mfu_preflight_required")
    if required is not True:
        raise ValueError("config must set mfu_preflight_required=true")
    configured_min = float(source.get("mfu_min_fraction", 0.0))
    if configured_min < 0.20:
        raise ValueError("config must set mfu_min_fraction >= 0.20")
    if abs(configured_min - args.min_fraction) > 1e-12:
        raise ValueError("launcher minimum and config mfu_min_fraction disagree")
    if not torch.cuda.is_available():
        raise RuntimeError("MFU preflight requires CUDA")

    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    atlas_start_steps = tuple(
        int(step)
        for step in source.get(
            "block_fht_attn_cayley_atlas_start_steps", ()
        )
    )
    effective_warmup_updates = max(
        args.warmup_updates,
        len(atlas_start_steps),
    )
    device_name = torch.cuda.get_device_name(0)
    temporary_root = Path(tempfile.mkdtemp(prefix="mfu-preflight-"))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = temporary_root / "train.log"
    start = time.time()
    certificate: dict[str, Any] = {
        "schema_version": "nanogpt_mfu_preflight_v1",
        "config": {"path": str(config_path), "sha256": config_sha256},
        "policy": {
            "mfu_preflight_required": True,
            "minimum_fraction": args.min_fraction,
            "denominator": "empirical_bf16_tensorcore_gemm_peak",
        },
        "hardware": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": device_name},
        "preflight": {
            "requested_warmup_updates": args.warmup_updates,
            "warmup_updates": effective_warmup_updates,
            "timed_updates": args.timed_updates,
            "include_diagnostic_io": args.include_diagnostic_io,
            "scientific_atlas_start_steps": list(atlas_start_steps),
            "scratch_atlas_start_steps": list(range(len(atlas_start_steps))),
            "timed_atlas_stage": (
                len(atlas_start_steps) - 1 if atlas_start_steps else None
            ),
            **task_frame_preflight_metadata(
                source,
                effective_warmup_updates,
            ),
            **feedback_cap_preflight_metadata(source),
        },
        "passed": False,
        "provenance": execution_provenance(config_path, source),
    }
    try:
        certificate["native_block_fht_extension"] = (
            verify_native_block_fht_extension(source)
        )
        gemm_peak = empirical_bf16_gemm_peak_tflops(args.gemm_size, args.gemm_warmups, args.gemm_trials)
        certificate["calibration"] = {
            "bf16_gemm_size": args.gemm_size,
            "bf16_gemm_warmups": args.gemm_warmups,
            "bf16_gemm_trials": args.gemm_trials,
            "empirical_bf16_gemm_peak_tflops": gemm_peak,
        }
        preflight_config = make_preflight_config(
            source,
            temporary_root / "run",
            args.warmup_updates,
            args.timed_updates,
            include_diagnostic_io=args.include_diagnostic_io,
        )
        preflight_config_path = temporary_root / "config.json"
        preflight_config_path.write_text(json.dumps(preflight_config, sort_keys=True) + "\n")
        command = [sys.executable, "-u", "-m", "examples.nanogpt.train", "--config", str(preflight_config_path)]
        process = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(process.stdout)
        certificate["train_command"] = command
        certificate["train_exit_code"] = process.returncode
        if process.returncode != 0:
            raise RuntimeError(f"real-training preflight failed (exit={process.returncode}); log={log_path}")
        loss_values = parse_training_loss_values(process.stdout)
        finite_loss_values = sum(math.isfinite(value) for value in loss_values)
        certificate["stability"] = {
            "logged_loss_values": len(loss_values),
            "finite_loss_values": finite_loss_values,
            "all_logged_losses_finite": (
                bool(loss_values) and finite_loss_values == len(loss_values)
            ),
        }
        if not loss_values:
            raise RuntimeError("real-training preflight emitted no loss values")
        if finite_loss_values != len(loss_values):
            raise RuntimeError("real-training preflight emitted nonfinite loss")
        feedback_cap_events = parse_feedback_cap_events(process.stdout)
        active_cap_layer_updates = sum(
            int(row.get("active_layers", 0)) for row in feedback_cap_events
        )
        timed_cap_events = [
            row
            for row in feedback_cap_events
            if int(row.get("step", -1)) >= effective_warmup_updates
        ]
        certificate["feedback_cap"] = {
            "event_count": len(feedback_cap_events),
            "timed_event_count": len(timed_cap_events),
            "active_layer_updates": active_cap_layer_updates,
            "maximum_pre_cap_nominal_steps": max(
                (
                    float(row.get("max_pre_cap_nominal_steps", 0.0))
                    for row in feedback_cap_events
                ),
                default=0.0,
            ),
        }
        if source.get("mfu_preflight_require_feedback_cap_active", False):
            if not feedback_cap_events or active_cap_layer_updates < 1:
                raise RuntimeError(
                    "registered preflight did not exercise an active feedback cap"
                )
            if not timed_cap_events:
                raise RuntimeError(
                    "registered preflight did not exercise the cap during timed updates"
                )
        stochastic_events = parse_stochastic_retraction_events(process.stdout)
        timed_stochastic_events = [
            row
            for row in stochastic_events
            if int(row.get("step", -1)) >= effective_warmup_updates
        ]
        persistent_training_bytes = parse_pair_vq_persistent_training_bytes(
            process.stdout
        )
        stochastic_required = source.get(
            "mfu_preflight_stochastic_weighted_variance_ratio_max"
        )
        certificate["stochastic_retraction"] = {
            "event_count": len(stochastic_events),
            "timed_event_count": len(timed_stochastic_events),
            "maximum_timed_weighted_sampling_variance_ratio": max(
                (
                    float(row["weighted_sampling_variance_ratio"])
                    for row in timed_stochastic_events
                ),
                default=None,
            ),
            "minimum_timed_expected_bias_recovery": min(
                (
                    float(row["minimum_expected_bias_recovery"])
                    for row in timed_stochastic_events
                ),
                default=None,
            ),
            "timed_boundary_clipped_values": sum(
                int(row["boundary_clipped_values"])
                for row in timed_stochastic_events
            ),
            "persistent_training_bytes": persistent_training_bytes,
        }
        if stochastic_required is not None:
            if len(timed_stochastic_events) < args.timed_updates:
                raise RuntimeError(
                    "stochastic-retraction gate did not observe every timed update"
                )
            maximum_variance = max(
                float(row["weighted_sampling_variance_ratio"])
                for row in timed_stochastic_events
            )
            if maximum_variance > float(stochastic_required):
                raise RuntimeError(
                    "stochastic-retraction variance gate rejected launch: "
                    f"{maximum_variance:.6f} > {float(stochastic_required):.6f}"
                )
            minimum_bias = min(
                float(row["minimum_expected_bias_recovery"])
                for row in timed_stochastic_events
            )
            bias_required = float(
                source["mfu_preflight_stochastic_expected_bias_recovery_min"]
            )
            if minimum_bias < bias_required:
                raise RuntimeError(
                    "stochastic-retraction bias gate rejected launch: "
                    f"{minimum_bias:.12f} < {bias_required:.12f}"
                )
            clipping_max = int(
                source["mfu_preflight_stochastic_boundary_clipped_values_max"]
            )
            clipped = sum(
                int(row["boundary_clipped_values"])
                for row in timed_stochastic_events
            )
            if clipped > clipping_max:
                raise RuntimeError(
                    "stochastic-retraction clipping gate rejected launch: "
                    f"{clipped} > {clipping_max}"
                )
            expected_bytes = int(
                source["mfu_preflight_pair_vq_persistent_training_bytes_exact"]
            )
            if persistent_training_bytes != [expected_bytes]:
                raise RuntimeError(
                    "pair-VQ persistent-byte gate rejected launch: "
                    f"observed={persistent_training_bytes} expected={[expected_bytes]}"
                )
        rows = parse_perf_rows(process.stdout)
        if len(rows) < args.timed_updates:
            raise RuntimeError(f"preflight emitted only {len(rows)} timed perf rows; expected {args.timed_updates}")
        steady = rows[-args.timed_updates:]
        base_iter_ms = sum(row["iter_ms"] for row in steady) / len(steady)
        snapshot_seconds = parse_snapshot_elapsed_seconds(process.stdout)
        optimizer_probe_steps = parse_optimizer_probe_steps(process.stdout)
        if args.include_diagnostic_io:
            expected_rows = effective_warmup_updates + args.timed_updates
            if len(rows) < expected_rows:
                raise RuntimeError(
                    "I/O-inclusive preflight emitted only "
                    f"{len(rows)} perf rows; expected {expected_rows}"
                )
            trajectory_io_requested = int(
                source.get("trajectory_snapshot_interval", 0) or 0
            ) > 0
            optimizer_probe_io_requested = bool(
                source.get("optimizer_probe_steps")
            )
            if trajectory_io_requested and not snapshot_seconds:
                raise RuntimeError(
                    "I/O-inclusive preflight emitted no parameter-snapshot "
                    "timings"
                )
            if optimizer_probe_io_requested and not optimizer_probe_steps:
                raise RuntimeError(
                    "I/O-inclusive preflight emitted no optimizer-probe "
                    "completion"
                )
            if not trajectory_io_requested and not optimizer_probe_io_requested:
                raise RuntimeError(
                    "I/O-inclusive preflight was requested for a config with "
                    "no registered diagnostic I/O"
                )
            measured = rows[-expected_rows:]
            # Evaluation is not training work and is excluded. Optimizer-probe
            # serialization already occurs inside iter_ms; parameter snapshots
            # occur after the perf row and are charged explicitly here.
            charged_training_ms = sum(
                max(row["iter_ms"] - row.get("eval_ms", 0.0), 0.0)
                for row in measured
            )
            charged_snapshot_ms = 1000.0 * sum(snapshot_seconds)
            iter_ms = (
                charged_training_ms + charged_snapshot_ms
            ) / len(measured)
            tokens_per_iter = (
                float(source["batch_size"])
                * float(source["gradient_accumulation_steps"])
                * float(source["block_size"])
            )
            tokens_per_second = tokens_per_iter / (iter_ms / 1000.0)
        else:
            iter_ms = base_iter_ms
            tokens_per_second = (
                sum(row["tokens_per_s"] for row in steady) / len(steady)
            )
        active_params = estimate_active_params(source)
        model_tflops = 6.0 * active_params * tokens_per_second / 1e12
        mfu_fraction = model_tflops / gemm_peak
        timing_keys = (
            "prepare_ms", "fwbw_ms", "flush_ms", "grad_ms", "opt_ms",
            "data_ms", "other_ms", "eval_ms",
        )
        certificate.update(
            {
                "measurement": {
                    "active_params_6n_estimate": active_params,
                    "tokens_per_second": tokens_per_second,
                    "iter_ms": iter_ms,
                    "base_timed_iter_ms": base_iter_ms,
                    "model_tflops": model_tflops,
                    "mfu_fraction": mfu_fraction,
                    "peak_mib": max(row.get("peak_mib", 0.0) for row in steady),
                    "diagnostic_io": {
                        "included": args.include_diagnostic_io,
                        "snapshot_count": len(snapshot_seconds),
                        "snapshot_seconds": sum(snapshot_seconds),
                        "optimizer_probe_io_included_in_iter_ms": (
                            args.include_diagnostic_io
                        ),
                        "optimizer_probe_count": len(optimizer_probe_steps),
                        "optimizer_probe_steps": optimizer_probe_steps,
                    },
                    "timing_breakdown_ms": {
                        key: sum(row.get(key, 0.0) for row in steady) / len(steady)
                        for key in timing_keys
                    },
                },
                "passed": mfu_fraction >= args.min_fraction,
            }
        )
        if not certificate["passed"]:
            raise RuntimeError(
                f"MFU gate rejected launch: measured {mfu_fraction:.2%} < required {args.min_fraction:.2%}"
            )
    except Exception as error:
        certificate["error"] = str(error)
        raise
    finally:
        certificate["finished_at_unix"] = time.time()
        certificate["elapsed_seconds"] = certificate["finished_at_unix"] - start
        temporary_log = log_path.read_text(errors="replace") if log_path.exists() else ""
        certificate["preflight_log_sha256"] = hashlib.sha256(temporary_log.encode()).hexdigest()
        if args.log_output is not None:
            durable_log = args.log_output.resolve()
            durable_log.parent.mkdir(parents=True, exist_ok=True)
            temporary_durable_log = durable_log.with_suffix(
                durable_log.suffix + ".part"
            )
            temporary_durable_log.write_text(temporary_log)
            os.replace(temporary_durable_log, durable_log)
            certificate["preflight_log_path"] = str(durable_log)
        # Failed qualification must be diagnosable after its temporary working
        # tree is removed. Keep only a bounded tail in the durable certificate.
        certificate["preflight_log_tail"] = temporary_log[-12000:]
        temporary_certificate = output.with_suffix(output.suffix + ".part")
        temporary_certificate.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_certificate, output)
        shutil.rmtree(temporary_root, ignore_errors=True)
        print(json.dumps(certificate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
