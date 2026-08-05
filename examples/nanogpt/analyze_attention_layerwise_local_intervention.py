#!/usr/bin/env python3
"""Probe dense attention component directions locally, one layer at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_attention_endpoint_attribution import (
    clear_frozen_base_cache,
    evaluate_ce,
    hybrid_attention,
    load_endpoint_model,
    prepare_frozen_base_cache,
    sha256,
    tensor_sha256,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches


MASKS = {
    "score": (1, 0, 0),
    "value": (0, 1, 0),
    "projection": (0, 0, 1),
    "value_projection": (0, 1, 1),
    "all": (1, 1, 1),
}


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


class LocalBlendHook:
    def __init__(
        self,
        dense_attention: torch.nn.Module,
        candidate_attention: torch.nn.Module,
        mask: tuple[int, int, int],
        ratio: float,
    ) -> None:
        def hook(_module, inputs, output):
            target = hybrid_attention(
                dense_attention, candidate_attention, inputs[0], mask
            )
            return output + float(ratio) * (target - output)

        self.handle = candidate_attention.register_forward_hook(hook)

    def close(self) -> None:
        self.handle.remove()


def local_blend_ce(
    dense: torch.nn.Module,
    candidate: torch.nn.Module,
    batches: list[torch.Tensor],
    device: str,
    layer: int,
    mask: tuple[int, int, int],
    ratio: float,
) -> float:
    hook = LocalBlendHook(
        dense.transformer.h[layer].attn,
        candidate.transformer.h[layer].attn,
        mask,
        ratio,
    )
    try:
        return evaluate_ce(candidate, batches, device)
    finally:
        hook.close()


def summarize(
    rows: list[dict[str, Any]], baseline: dict[str, float]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for window in baseline:
        summary[window] = {}
        for name in MASKS:
            summary[window][name] = {}
            for ratio in sorted(
                {float(row["ratio"]) for row in rows if row["window"] == window}
            ):
                selected = [
                    row
                    for row in rows
                    if row["window"] == window
                    and row["component"] == name
                    and float(row["ratio"]) == ratio
                ]
                deltas = [float(row["ce_delta"]) for row in selected]
                summary[window][name][str(ratio)] = {
                    "layers": len(selected),
                    "sum_single_layer_ce_delta": sum(deltas),
                    "mean_single_layer_ce_delta": sum(deltas) / len(deltas),
                    "improving_layer_fraction": sum(delta < 0 for delta in deltas) / len(deltas),
                    "minimum_layer_ce_delta": min(deltas),
                    "maximum_layer_ce_delta": max(deltas),
                }
    return summary


def decide(summary: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    selection_ratio = str(float(protocol["selection_ratio"]))
    minimum = float(protocol["minimum_replicated_sum_improvement_ce"])
    minimum_fraction = float(protocol["minimum_improving_layer_fraction"])
    tops = {}
    for window in ("primary", "confirmation"):
        tops[window] = min(
            MASKS,
            key=lambda name: summary[window][name][selection_ratio][
                "sum_single_layer_ce_delta"
            ],
        )
    selected = tops["primary"] if tops["primary"] == tops["confirmation"] else None
    admitted = bool(
        selected is not None
        and all(
            summary[window][selected][selection_ratio][
                "sum_single_layer_ce_delta"
            ]
            <= -minimum
            and summary[window][selected][selection_ratio][
                "improving_layer_fraction"
            ]
            >= minimum_fraction
            for window in ("primary", "confirmation")
        )
    )
    return {
        "classification": (
            "REPLICATED_LOCAL_DENSE_DIRECTION"
            if admitted
            else "NO_REPLICATED_LOCAL_DENSE_DIRECTION"
        ),
        "selected_component": selected if admitted else None,
        "primary_top_component": tops["primary"],
        "confirmation_top_component": tops["confirmation"],
        "selection_ratio": float(selection_ratio),
        "minimum_replicated_sum_improvement_ce": minimum,
        "minimum_improving_layer_fraction": minimum_fraction,
        "automatic_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--dense-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != "mai_124m_attention_layerwise_local_intervention_plan_v1":
        raise ValueError("unexpected local intervention plan schema")
    protocol = plan["protocol"]
    manifest_sha256 = sha256(args.data_dir / "manifest.json")
    identities = {
        "dataset_manifest_sha256": manifest_sha256,
        "dense_checkpoint_sha256": sha256(args.dense_checkpoint),
        "candidate_checkpoint_sha256": sha256(args.candidate_checkpoint),
    }
    for key, actual in identities.items():
        if actual != protocol[f"required_{key}"]:
            raise ValueError(f"{key} mismatch")
    dense = load_endpoint_model(args.dense_checkpoint, args.device)
    candidate = load_endpoint_model(args.candidate_checkpoint, args.device)
    cached = prepare_frozen_base_cache(candidate, torch.bfloat16)
    windows = {
        name: fixed_validation_batches(
            args.data_dir,
            int(spec["batch_size"]),
            int(spec["block_size"]) + 1,
            int(spec["batches"]),
            int(spec["seed"]),
        )
        for name, spec in protocol["validation_windows"].items()
    }
    try:
        baseline = {
            name: evaluate_ce(candidate, batches, args.device)
            for name, batches in windows.items()
        }
        rows: list[dict[str, Any]] = []
        for window, batches in windows.items():
            for layer in protocol["layers"]:
                for component, mask in MASKS.items():
                    for ratio in protocol["ratios"]:
                        ce = local_blend_ce(
                            dense,
                            candidate,
                            batches,
                            args.device,
                            int(layer),
                            mask,
                            float(ratio),
                        )
                        row = {
                            "window": window,
                            "layer": int(layer),
                            "component": component,
                            "mask": "".join(str(value) for value in mask),
                            "ratio": float(ratio),
                            "ce": ce,
                            "ce_delta": ce - baseline[window],
                            "finite_difference_slope": (
                                ce - baseline[window]
                            ) / float(ratio),
                        }
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        clear_frozen_base_cache(candidate)
    summary = summarize(rows, baseline)
    decision = decide(summary, protocol)
    root = Path(__file__).resolve().parents[2]
    output = {
        "schema_version": "mai_124m_attention_layerwise_local_intervention_v1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": {
            "git_commit": git_commit(root),
            "entrypoint": "examples.nanogpt.analyze_attention_layerwise_local_intervention",
            "command": sys.argv,
            "plan": str(args.plan),
            "plan_sha256": sha256(args.plan),
            **identities,
            "cached_block_fht_modules": cached,
            "device": args.device,
            "device_name": torch.cuda.get_device_name(0),
            "validation_token_sha256": {
                name: tensor_sha256(torch.cat(batches))
                for name, batches in windows.items()
            },
        },
        "protocol": protocol,
        "baseline_ce": baseline,
        "summary": summary,
        "decision": decision,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
