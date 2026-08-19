#!/usr/bin/env python3
"""Gate a hybrid-precision codec for the accepted full-MLP temporal state.

The preregistered joint int8 audit localized failure to Muon's polar transform:
raw int8 momentum reconstruction was accurate, but its induced polar direction
was not.  Compression residuals did not show that amplification.  This
zero-update follow-up therefore keeps momentum in a 16-bit floating format and
uses blockwise int8 only for compression residuals.  An FP16-all-state arm is
included as a non-compressed-direction control.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_full_mlp_temporal_state_quantization import (
    OPTIMIZER_ROLES,
    aggregate_metrics,
    blockwise_symmetric_quantize,
    file_sha256,
    flatten_metric,
    git_commit,
    tensor_metrics,
)
from examples.nanogpt.muon import zeropower_via_newtonschulz5


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "mai_124m_full_mlp_temporal_state_hybrid_precision_plan_v1"
RESULT_SCHEMA = "mai_124m_full_mlp_temporal_state_hybrid_precision_result_v1"
CANDIDATES = (
    ("fp16_momentum_int8_feedback", "float16", "int8_block4096"),
    ("bf16_momentum_int8_feedback", "bfloat16", "int8_block4096"),
    ("fp16_momentum_fp16_feedback_control", "float16", "float16"),
)


def floating_reconstruct(
    source: torch.Tensor, dtype_name: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    if dtype_name not in {"float16", "bfloat16"}:
        raise ValueError("floating codec must be float16 or bfloat16")
    dtype = getattr(torch, dtype_name)
    reconstructed = source.detach().to(dtype).float()
    element_size = torch.tensor([], dtype=dtype).element_size()
    return reconstructed, {
        "persistent_bytes": source.numel() * element_size,
        "dense_fp32_bytes": source.numel() * 4,
        "codec": dtype_name,
    }


def classify(summaries: list[dict[str, Any]], rule: dict[str, Any]) -> dict[str, Any]:
    passing = []
    for summary in summaries:
        raw = summary["raw"]
        polar = summary["momentum_state_only_polar_proxy"]
        passed = (
            summary["persistent_storage_ratio_to_dense_fp32"]
            <= float(rule["maximum_persistent_storage_ratio"])
            and raw["momentum_buffer"]["energy_recovery"]
            >= float(rule["minimum_global_raw_energy_recovery"])
            and raw["compression_residual"]["energy_recovery"]
            >= float(rule["minimum_global_raw_energy_recovery"])
            and raw["momentum_buffer"]["minimum_layer_cosine"]
            >= float(rule["minimum_layer_raw_cosine"])
            and raw["compression_residual"]["minimum_layer_cosine"]
            >= float(rule["minimum_layer_raw_cosine"])
            and polar["energy_recovery"]
            >= float(rule["minimum_global_polar_energy_recovery"])
            and polar["minimum_layer_cosine"]
            >= float(rule["minimum_layer_polar_cosine"])
        )
        summary["gate_passed"] = passed
        if passed:
            passing.append(summary)
    selected = min(
        passing,
        key=lambda row: (
            int(row["persistent_storage_bytes"]),
            -float(row["momentum_state_only_polar_proxy"]["energy_recovery"]),
        ),
        default=None,
    )
    return {
        "classification": (
            "HYBRID_PRECISION_AMBIENT_STATE_PLAUSIBLE"
            if selected is not None
            else "HYBRID_PRECISION_AMBIENT_STATE_REJECTED"
        ),
        "selected_candidate": None if selected is None else selected["name"],
        "passing_candidate_count": len(passing),
        "thresholds": rule,
        "parameter_updates_to_checkpoint": 0,
        "interpretation_boundary": (
            "Terminal state-only oracle. A pass authorizes codec implementation, "
            "resume tests, and exact-config MFU/VRAM measurement only."
        ),
    }


def validate(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected hybrid-precision plan schema")
    observed = {
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "dataset_manifest_sha256": file_sha256(args.data_manifest),
        "parent_oracle_result_sha256": file_sha256(args.parent_oracle_result),
    }
    if observed != plan.get("identity"):
        raise ValueError(f"hybrid-precision identity mismatch: {observed}")
    parent = json.loads(args.parent_oracle_result.read_text())
    if parent["decision"]["classification"] != "BLOCKWISE_AMBIENT_STATE_QUANTIZATION_REJECTED":
        raise ValueError("parent oracle does not authorize hybrid precision")
    expected = {
        "parameter_updates": 0,
        "checkpoint_next_iter": 238,
        "optimizer_roles": [list(value) for value in OPTIMIZER_ROLES],
        "candidates": [list(value) for value in CANDIDATES],
        "feedback_int8_block_size": 4096,
        "feedback_int8_scale_dtype": "float16",
        "muon_state_only_polar_proxy_steps": 5,
    }
    if plan.get("protocol") != expected:
        raise ValueError("hybrid-precision protocol changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--parent-oracle-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text())
    validate(args, plan)
    started = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["next_iter"]) != int(plan["protocol"]["checkpoint_next_iter"]):
        raise ValueError("checkpoint next_iter changed")
    optimizers = checkpoint["optimizer"]["optimizers"]
    rows: list[dict[str, Any]] = []
    for optimizer_index, role in OPTIMIZER_ROLES:
        state_dict = optimizers[optimizer_index]["state"]
        for layer, parameter_id in enumerate(sorted(state_dict, key=int)):
            parameter_state = state_dict[parameter_id]
            momentum = parameter_state["momentum_buffer"].detach().float().to(args.device)
            feedback = parameter_state["compression_residual"].detach().float().to(args.device)
            original_polar = zeropower_via_newtonschulz5(momentum, steps=5).float()
            for name, momentum_codec, feedback_codec in CANDIDATES:
                momentum_hat, momentum_storage = floating_reconstruct(
                    momentum, momentum_codec
                )
                if feedback_codec == "int8_block4096":
                    feedback_hat, feedback_storage = blockwise_symmetric_quantize(
                        feedback, bits=8, block_size=4096
                    )
                else:
                    feedback_hat, feedback_storage = floating_reconstruct(
                        feedback, feedback_codec
                    )
                polar_hat = zeropower_via_newtonschulz5(momentum_hat, steps=5).float()
                momentum_metric = tensor_metrics(momentum, momentum_hat)
                feedback_metric = tensor_metrics(feedback, feedback_hat)
                polar_metric = tensor_metrics(original_polar, polar_hat)
                rows.append(
                    {
                        "optimizer_index": optimizer_index,
                        "role": role,
                        "layer": layer,
                        "name": name,
                        "momentum_codec": momentum_codec,
                        "feedback_codec": feedback_codec,
                        "momentum_persistent_bytes": int(momentum_storage["persistent_bytes"]),
                        "feedback_persistent_bytes": int(feedback_storage["persistent_bytes"]),
                        "dense_fp32_bytes": int(momentum_storage["dense_fp32_bytes"])
                        + int(feedback_storage["dense_fp32_bytes"]),
                        **flatten_metric("momentum", momentum_metric),
                        **flatten_metric("feedback", feedback_metric),
                        **flatten_metric("polar", polar_metric),
                    }
                )
                print(
                    f"role={role} layer={layer} candidate={name} "
                    f"momentum={momentum_metric['energy_recovery']:.9f} "
                    f"feedback={feedback_metric['energy_recovery']:.9f} "
                    f"polar={polar_metric['energy_recovery']:.9f}",
                    flush=True,
                )
    summaries: list[dict[str, Any]] = []
    for name, momentum_codec, feedback_codec in CANDIDATES:
        selected = [row for row in rows if row["name"] == name]
        persistent_bytes = sum(
            int(row["momentum_persistent_bytes"])
            + int(row["feedback_persistent_bytes"])
            for row in selected
        )
        dense_bytes = sum(int(row["dense_fp32_bytes"]) for row in selected)
        summaries.append(
            {
                "name": name,
                "momentum_codec": momentum_codec,
                "feedback_codec": feedback_codec,
                "raw": {
                    "momentum_buffer": aggregate_metrics(selected, "momentum"),
                    "compression_residual": aggregate_metrics(selected, "feedback"),
                },
                "momentum_state_only_polar_proxy": aggregate_metrics(selected, "polar"),
                "persistent_storage_bytes": persistent_bytes,
                "dense_fp32_bytes": dense_bytes,
                "persistent_storage_ratio_to_dense_fp32": persistent_bytes / dense_bytes,
                "storage_reduction_factor": dense_bytes / persistent_bytes,
            }
        )
    decision = classify(summaries, plan["decision_rule"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "summaries": summaries,
        "rows": rows,
        "identity": {
            **plan["identity"],
            "plan_sha256": file_sha256(args.plan),
            "checkpoint_next_iter": int(checkpoint["next_iter"]),
        },
        "execution": {
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "command": sys.argv,
            "device": args.device,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
        },
        "authorization": {
            "exact_resume_codec_implementation": decision["selected_candidate"] is not None,
            "focused_tests": decision["selected_candidate"] is not None,
            "exact_config_mfu_vram_gate": False,
            "language_model_training": False,
            "larger_rung": False,
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"decision": decision, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
