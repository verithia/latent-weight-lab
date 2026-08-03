#!/usr/bin/env python3
"""Localize terminal MLP co-adaptation by cross-checkpoint endpoint splicing.

This diagnostic performs zero parameter updates.  At both 124M and 350M it
transplants materialized ``c_fc``/``c_proj`` endpoints, their paired branch,
and the pre-MLP LayerNorm between matched parent and candidate checkpoints.
The same fixed validation windows distinguish intra-MLP co-adaptation from
normalization or wider residual-block context.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cfc_checkpoint_splice import (
    evaluate,
    git_head,
    sha256,
    tensor_sha256,
    write_csv,
)
from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.model import GPT, GPTConfig
from latent_weight_lab.block_fht import prepare_block_fht_weight_cache


SCHEMA_VERSION = "nanogpt_mlp_pair_checkpoint_splice_v1"


def depth_bands(n_layer: int) -> dict[str, list[int]]:
    if n_layer % 3:
        raise ValueError("registered depth bands require a layer count divisible by three")
    width = n_layer // 3
    return {
        "early": list(range(0, width)),
        "middle": list(range(width, 2 * width)),
        "late": list(range(2 * width, n_layer)),
    }


def variant_specs(n_layer: int) -> list[dict[str, object]]:
    all_layers = list(range(n_layer))
    bands = depth_bands(n_layer)
    return [
        {"name": "parent", "base": "parent", "source": None, "components": [], "layers": []},
        {"name": "candidate", "base": "candidate", "source": None, "components": [], "layers": []},
        {
            "name": "candidate_parent_cfc_all",
            "base": "candidate",
            "source": "parent",
            "components": ["c_fc"],
            "layers": all_layers,
        },
        {
            "name": "candidate_parent_cproj_all",
            "base": "candidate",
            "source": "parent",
            "components": ["c_proj"],
            "layers": all_layers,
        },
        {
            "name": "candidate_parent_mlp_pair_all",
            "base": "candidate",
            "source": "parent",
            "components": ["c_fc", "c_proj"],
            "layers": all_layers,
        },
        *[
            {
                "name": f"candidate_parent_mlp_pair_{band}",
                "base": "candidate",
                "source": "parent",
                "components": ["c_fc", "c_proj"],
                "layers": layers,
            }
            for band, layers in bands.items()
        ],
        {
            "name": "candidate_parent_ln2_all",
            "base": "candidate",
            "source": "parent",
            "components": ["ln_2"],
            "layers": all_layers,
        },
        {
            "name": "candidate_parent_mlp_pair_ln2_all",
            "base": "candidate",
            "source": "parent",
            "components": ["c_fc", "c_proj", "ln_2"],
            "layers": all_layers,
        },
        {
            "name": "parent_candidate_mlp_pair_all",
            "base": "parent",
            "source": "candidate",
            "components": ["c_fc", "c_proj"],
            "layers": all_layers,
        },
    ]


def component_keys(state: dict[str, torch.Tensor], layer: int, component: str) -> list[str]:
    root = f"transformer.h.{layer}"
    if component == "c_fc":
        candidates = [f"{root}.mlp.c_fc.weight"]
    elif component == "c_proj":
        candidates = [f"{root}.mlp.c_proj.weight"]
    elif component == "ln_2":
        candidates = [f"{root}.ln_2.weight", f"{root}.ln_2.bias"]
    else:
        raise ValueError(f"unknown component: {component}")
    keys = [key for key in candidates if key in state]
    if not keys:
        raise KeyError(f"checkpoint has no {component} endpoint for layer {layer}")
    return keys


def load_checkpoint(
    path: Path, *, expected_next_iter: int, expected_n_layer: int
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("next_iter", -1)) != expected_next_iter:
        raise ValueError(f"checkpoint iteration mismatch: {path}")
    config = checkpoint.get("model_config")
    state = checkpoint.get("model")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state: {path}")
    if int(config.get("n_layer", -1)) != expected_n_layer:
        raise ValueError(f"checkpoint layer count mismatch: {path}")
    return config, state


def build_model(
    spec: dict[str, object],
    configs: dict[str, dict[str, Any]],
    states: dict[str, dict[str, torch.Tensor]],
    device: str,
) -> GPT:
    base = str(spec["base"])
    model = GPT(GPTConfig(**configs[base]))
    model.load_state_dict(states[base])
    source_name = spec.get("source")
    if source_name is not None:
        source = states[str(source_name)]
        target = model.state_dict()
        with torch.no_grad():
            for layer in spec["layers"]:
                for component in spec["components"]:
                    for key in component_keys(source, int(layer), str(component)):
                        if key not in target or target[key].shape != source[key].shape:
                            raise ValueError(f"incompatible transplant endpoint: {key}")
                        target[key].copy_(source[key])
    model.to(device)
    model.eval()
    prepare_block_fht_weight_cache(model, dtype=torch.bfloat16)
    return model


def recovery_decision(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped[str(row["scale"])][str(row["window"])][str(row["variant"])] = float(row["ce"])
    classifications: dict[str, str] = {}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    required = {
        "parent",
        "candidate",
        "candidate_parent_cfc_all",
        "candidate_parent_cproj_all",
        "candidate_parent_mlp_pair_all",
        "candidate_parent_mlp_pair_ln2_all",
    }
    for scale, windows in sorted(grouped.items()):
        scale_metrics: dict[str, dict[str, float]] = {}
        for window, values in sorted(windows.items()):
            if not required.issubset(values):
                raise ValueError(f"missing registered variants for {scale}/{window}")
            gap = values["candidate"] - values["parent"]
            if not math.isfinite(gap) or gap <= 0:
                classifications[scale] = "INCONCLUSIVE_NONPOSITIVE_BASE_GAP"
                continue
            recovery = {
                name: (values["candidate"] - values[name]) / gap
                for name in required - {"parent", "candidate"}
            }
            recovery["pair_synergy_over_best_single"] = recovery[
                "candidate_parent_mlp_pair_all"
            ] - max(
                recovery["candidate_parent_cfc_all"],
                recovery["candidate_parent_cproj_all"],
            )
            recovery["ln2_increment_over_pair"] = recovery[
                "candidate_parent_mlp_pair_ln2_all"
            ] - recovery["candidate_parent_mlp_pair_all"]
            scale_metrics[window] = recovery
        metrics[scale] = scale_metrics
        if scale in classifications:
            continue
        window_metrics = list(scale_metrics.values())
        pair = [item["candidate_parent_mlp_pair_all"] for item in window_metrics]
        pair_synergy = [item["pair_synergy_over_best_single"] for item in window_metrics]
        pair_ln = [item["candidate_parent_mlp_pair_ln2_all"] for item in window_metrics]
        ln_increment = [item["ln2_increment_over_pair"] for item in window_metrics]
        if min(pair) >= 0.50 and min(pair_synergy) >= 0.25:
            classification = "INTRA_MLP_PAIR_COADAPTATION_DOMINATES"
        elif min(pair_ln) >= 0.50 and min(ln_increment) >= 0.25:
            classification = "PRE_MLP_NORMALIZATION_COADAPTATION_DOMINATES"
        elif max(pair_ln) <= 0.25:
            classification = "WIDER_RESIDUAL_BLOCK_CONTEXT_DOMINATES"
        else:
            classification = "MIXED_MLP_NORMALIZATION_AND_BLOCK_COADAPTATION"
        classifications[scale] = classification
    unique = set(classifications.values())
    overall = next(iter(unique)) if len(unique) == 1 else "SCALE_DEPENDENT_COADAPTATION"
    return {
        "classification": overall,
        "classification_by_scale": classifications,
        "recovery_by_scale_and_window": metrics,
        "registered_boundaries": {
            "minimum_pair_or_pair_ln_recovery_each_window": 0.50,
            "minimum_pair_synergy_or_ln_increment_each_window": 0.25,
            "maximum_pair_ln_recovery_for_wider_context_each_window": 0.25,
        },
    }


def validate_plan(plan: dict[str, Any], root: Path) -> None:
    if plan["schema_version"] != "mlp_pair_checkpoint_splice_plan_v1":
        raise ValueError("unsupported plan schema")
    for path, digest in plan["source_hashes"].items():
        if sha256(root / path) != digest:
            raise ValueError(f"source SHA-256 mismatch: {path}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("refusing diagnostic from a dirty worktree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, root)
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing splice result")
    manifest = args.data_dir / "manifest.json"
    if sha256(manifest) != plan["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")
    args.output.mkdir(parents=True)

    protocol = plan["protocol"]
    windows = {
        f"seed{seed}": fixed_validation_batches(
            args.data_dir,
            batch_size=int(protocol["batch_size"]),
            block_size=int(protocol["block_size"]),
            batches=int(protocol["batches_per_window"]),
            seed=int(seed),
        )
        for seed in protocol["validation_seeds"]
    }
    token_hashes = {name: tensor_sha256(values) for name, values in windows.items()}
    evaluations: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    for experiment in plan["experiments"]:
        scale = str(experiment["scale"])
        n_layer = int(experiment["n_layer"])
        expected_next_iter = int(experiment["next_iter"])
        configs: dict[str, dict[str, Any]] = {}
        states: dict[str, dict[str, torch.Tensor]] = {}
        for name in ("parent", "candidate"):
            checkpoint = Path(experiment[f"{name}_checkpoint"])
            if sha256(checkpoint) != experiment[f"{name}_checkpoint_sha256"]:
                raise ValueError(f"{scale} {name} checkpoint SHA-256 mismatch")
            configs[name], states[name] = load_checkpoint(
                checkpoint,
                expected_next_iter=expected_next_iter,
                expected_n_layer=n_layer,
            )
        registered_variants = plan["variants_by_layer_count"][str(n_layer)]
        if variant_specs(n_layer) != registered_variants:
            raise ValueError(f"{scale} registered variants differ from implementation")
        for spec in registered_variants:
            model = build_model(spec, configs, states, args.device)
            for window, batches in windows.items():
                ce, stats = evaluate(
                    model,
                    batches,
                    args.device,
                    [int(layer) for layer in experiment["activation_layers"]],
                )
                row = {"scale": scale, "variant": spec["name"], "window": window, "ce": ce}
                evaluations.append(row)
                activation_rows.extend(
                    {"scale": scale, "variant": spec["name"], "window": window, **item}
                    for item in stats
                )
                print(
                    f"scale={scale} variant={spec['name']} window={window} ce={ce:.8f}",
                    flush=True,
                )
            del model
            torch.cuda.empty_cache()

    write_csv(args.output / "evaluation.csv", evaluations)
    write_csv(args.output / "activation_residual.csv", activation_rows)
    decision = recovery_decision(evaluations)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "parameter_updates": 0,
        "repository_commit": git_head(root),
        "command": sys.argv,
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "plan": {"path": str(args.plan), "sha256": sha256(args.plan)},
        "dataset_manifest": {"path": str(manifest), "sha256": sha256(manifest)},
        "protocol": {**protocol, "token_sha256": token_hashes},
        "evaluations": evaluations,
        "elapsed_seconds": time.time() - started,
        "artifacts": {
            name: {"path": str(args.output / name), "sha256": sha256(args.output / name)}
            for name in ("evaluation.csv", "activation_residual.csv")
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
