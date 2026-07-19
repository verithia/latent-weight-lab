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


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    return value


def estimate_active_params(config: dict[str, Any]) -> int:
    """GPT parameter count used by the conventional 6N model-FLOP estimate."""
    n_layer = int(config["n_layer"])
    n_embd = int(config["n_embd"])
    vocab_size = int(config.get("vocab_size", 50304))
    block_size = int(config["block_size"])
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


def make_preflight_config(source: dict[str, Any], temporary_out: Path, warmups: int, timed: int) -> dict[str, Any]:
    config = dict(source)
    # A preflight is never a scientific result.  Keep the real train path but
    # avoid checkpoint/evaluation overhead and any deterministic-run policy
    # intended for registered results.
    config.pop("prelaunch_provenance_requirements", None)
    config["registered_resume_determinism_required"] = False
    config["out_dir"] = str(temporary_out)
    config["init_from"] = "scratch"
    config["max_iters"] = warmups + timed
    config["lr_decay_iters"] = max(warmups + timed, int(config.get("lr_decay_iters", 1)))
    config["eval_interval"] = warmups + timed + 100
    config["eval_iters"] = 1
    config["fixed_eval_indices"] = False
    config["eval_seed"] = None
    config["save_checkpoint"] = False
    config["checkpoint_history"] = False
    config["perf_profile"] = True
    config["perf_warmup_iters"] = warmups
    config["perf_log_interval"] = 1
    config["log_interval"] = 1
    return config


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
        "preflight": {"warmup_updates": args.warmup_updates, "timed_updates": args.timed_updates},
        "passed": False,
    }
    try:
        gemm_peak = empirical_bf16_gemm_peak_tflops(args.gemm_size, args.gemm_warmups, args.gemm_trials)
        certificate["calibration"] = {
            "bf16_gemm_size": args.gemm_size,
            "bf16_gemm_warmups": args.gemm_warmups,
            "bf16_gemm_trials": args.gemm_trials,
            "empirical_bf16_gemm_peak_tflops": gemm_peak,
        }
        preflight_config = make_preflight_config(
            source, temporary_root / "run", args.warmup_updates, args.timed_updates
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
        rows = parse_perf_rows(process.stdout)
        if len(rows) < args.timed_updates:
            raise RuntimeError(f"preflight emitted only {len(rows)} timed perf rows; expected {args.timed_updates}")
        steady = rows[-args.timed_updates:]
        tokens_per_second = sum(row["tokens_per_s"] for row in steady) / len(steady)
        iter_ms = sum(row["iter_ms"] for row in steady) / len(steady)
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
                    "model_tflops": model_tflops,
                    "mfu_fraction": mfu_fraction,
                    "peak_mib": max(row.get("peak_mib", 0.0) for row in steady),
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
