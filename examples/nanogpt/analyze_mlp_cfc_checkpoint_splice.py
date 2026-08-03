#!/usr/bin/env python3
"""Causally splice terminal dense c_fc weights into a matched MLP candidate.

This is a zero-update diagnostic.  Parent and candidate are evaluated on the
same deterministic validation windows.  Candidate variants receive terminal
parent ``mlp.c_fc.weight`` tensors globally or by depth band; the converse
splice places candidate c_fc weights in the parent.  CE recovery, activation
statistics, residual compatibility, and c_fc spectra distinguish missing
expansion direction from downstream co-adaptation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.nanogpt.analyze_residual_compatibility import fixed_validation_batches
from examples.nanogpt.model import GPT, GPTConfig
from latent_weight_lab.block_fht import prepare_block_fht_weight_cache


SCHEMA_VERSION = "nanogpt_mlp_cfc_checkpoint_splice_v1"


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


def tensor_sha256(values: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def variant_specs(n_layer: int) -> list[dict[str, object]]:
    if n_layer != 24:
        raise ValueError("the registered splice bands require 24 layers")
    return [
        {"name": "parent", "base": "parent", "parent_cfc_layers": []},
        {"name": "candidate", "base": "candidate", "parent_cfc_layers": []},
        {
            "name": "candidate_parent_cfc_all",
            "base": "candidate",
            "parent_cfc_layers": list(range(24)),
        },
        {
            "name": "candidate_parent_cfc_early",
            "base": "candidate",
            "parent_cfc_layers": list(range(0, 8)),
        },
        {
            "name": "candidate_parent_cfc_middle",
            "base": "candidate",
            "parent_cfc_layers": list(range(8, 16)),
        },
        {
            "name": "candidate_parent_cfc_late",
            "base": "candidate",
            "parent_cfc_layers": list(range(16, 24)),
        },
        {
            "name": "parent_candidate_cfc_all",
            "base": "parent",
            "parent_cfc_layers": [],
            "candidate_cfc_layers": list(range(24)),
        },
    ]


def splice_decision(rows: list[dict[str, object]]) -> dict[str, object]:
    by_window: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_window[str(row["window"])][str(row["variant"])] = float(row["ce"])
    recoveries: dict[str, float] = {}
    for window, values in sorted(by_window.items()):
        required = {"parent", "candidate", "candidate_parent_cfc_all"}
        if not required.issubset(values):
            raise ValueError(f"missing registered variants for {window}")
        gap = values["candidate"] - values["parent"]
        if not math.isfinite(gap) or gap <= 0:
            return {
                "classification": "INCONCLUSIVE_NONPOSITIVE_BASE_GAP",
                "recovery_by_window": recoveries,
            }
        recoveries[window] = (
            values["candidate"] - values["candidate_parent_cfc_all"]
        ) / gap
    values = list(recoveries.values())
    if values and min(values) >= 0.50:
        classification = "MISSING_CFC_ENDPOINT_DIRECTION_DOMINATES"
    elif values and max(values) <= 0.25:
        classification = "DOWNSTREAM_COADAPTATION_DOMINATES"
    else:
        classification = "MIXED_CFC_DIRECTION_AND_COADAPTATION"
    return {
        "classification": classification,
        "recovery_by_window": recoveries,
        "registered_boundaries": {
            "direction_dominates_minimum_each_window": 0.50,
            "coadaptation_dominates_maximum_each_window": 0.25,
        },
    }


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("next_iter", -1)) != 677:
        raise ValueError(f"checkpoint is not terminal: {path}")
    model_config = checkpoint.get("model_config")
    state = checkpoint.get("model")
    if not isinstance(model_config, dict) or not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state: {path}")
    return model_config, state


def cfc_key(layer: int) -> str:
    return f"transformer.h.{layer}.mlp.c_fc.weight"


def build_model(
    spec: dict[str, object],
    configs: dict[str, dict[str, Any]],
    states: dict[str, dict[str, torch.Tensor]],
    device: str,
) -> GPT:
    base = str(spec["base"])
    model = GPT(GPTConfig(**configs[base]))
    model.load_state_dict(states[base])
    with torch.no_grad():
        for layer in spec.get("parent_cfc_layers", []):
            model.transformer.h[int(layer)].mlp.c_fc.weight.copy_(
                states["parent"][cfc_key(int(layer))]
            )
        for layer in spec.get("candidate_cfc_layers", []):
            model.transformer.h[int(layer)].mlp.c_fc.weight.copy_(
                states["candidate"][cfc_key(int(layer))]
            )
    model.to(device)
    model.eval()
    prepare_block_fht_weight_cache(model, dtype=torch.bfloat16)
    return model


class ActivationResidualStats:
    def __init__(self, model: GPT, layers: list[int]) -> None:
        self.layers = set(layers)
        self.values: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        self.residual: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer, block in enumerate(model.transformer.h):
            if layer not in self.layers:
                continue
            self.handles.append(block.ln_2.register_forward_pre_hook(self._residual(layer)))
            self.handles.append(block.mlp.c_fc.register_forward_hook(self._activation(layer, "pre")))
            self.handles.append(block.mlp.gelu.register_forward_hook(self._activation(layer, "post")))
            self.handles.append(block.mlp.register_forward_hook(self._mlp(layer)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _add(self, layer: int, key: str, value: torch.Tensor) -> None:
        scalar = value.detach().float()
        current = self.values[layer].get(key)
        self.values[layer][key] = scalar if current is None else current + scalar

    def _residual(self, layer: int):
        def hook(_module, inputs):
            self.residual[layer] = inputs[0].detach()
        return hook

    def _activation(self, layer: int, point: str):
        def hook(_module, _inputs, output):
            values = output.detach().float()
            self._add(layer, f"{point}_count", torch.tensor(values.numel(), device=values.device))
            self._add(layer, f"{point}_sum", values.sum())
            self._add(layer, f"{point}_sq", values.square().sum())
            if point == "pre":
                self._add(layer, "pre_positive", (values > 0).sum())
            else:
                self._add(layer, "post_nearzero", (values.abs() < 0.1).sum())
        return hook

    def _mlp(self, layer: int):
        def hook(_module, _inputs, output):
            residual = self.residual.pop(layer).detach().float()
            update = output.detach().float()
            self._add(layer, "residual_count", torch.tensor(residual.numel(), device=residual.device))
            self._add(layer, "residual_sq", residual.square().sum())
            self._add(layer, "update_sq", update.square().sum())
            flat_residual = residual.reshape(-1, residual.shape[-1])
            flat_update = update.reshape(-1, update.shape[-1])
            cosine = (flat_residual * flat_update).sum(dim=1) / (
                flat_residual.norm(dim=1) * flat_update.norm(dim=1)
            ).clamp_min(1e-30)
            self._add(layer, "cosine_sum", cosine.sum())
            self._add(layer, "cosine_count", torch.tensor(cosine.numel(), device=cosine.device))
        return hook

    def rows(self) -> list[dict[str, float | int]]:
        output = []
        for layer in sorted(self.values):
            values = {key: float(value.item()) for key, value in self.values[layer].items()}
            pre_count = max(values["pre_count"], 1.0)
            post_count = max(values["post_count"], 1.0)
            residual_count = max(values["residual_count"], 1.0)
            pre_rms = math.sqrt(values["pre_sq"] / pre_count)
            post_rms = math.sqrt(values["post_sq"] / post_count)
            residual_rms = math.sqrt(values["residual_sq"] / residual_count)
            update_rms = math.sqrt(values["update_sq"] / residual_count)
            output.append(
                {
                    "layer": layer,
                    "pre_mean": values["pre_sum"] / pre_count,
                    "pre_rms": pre_rms,
                    "pre_positive_frac": values["pre_positive"] / pre_count,
                    "post_mean": values["post_sum"] / post_count,
                    "post_rms": post_rms,
                    "post_nearzero_frac": values["post_nearzero"] / post_count,
                    "post_pre_rms_ratio": post_rms / max(pre_rms, 1e-30),
                    "residual_rms": residual_rms,
                    "update_rms": update_rms,
                    "update_to_residual_rms": update_rms / max(residual_rms, 1e-30),
                    "residual_update_cosine": values["cosine_sum"] / max(values["cosine_count"], 1.0),
                }
            )
        return output


@torch.no_grad()
def evaluate(
    model: GPT,
    batches: list[torch.Tensor],
    device: str,
    layers: list[int],
) -> tuple[float, list[dict[str, float | int]]]:
    collector = ActivationResidualStats(model, layers)
    losses = []
    try:
        for tokens in batches:
            tokens = tokens.to(device)
            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(inputs, targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("nonfinite splice evaluation loss")
            losses.append(float(loss))
    finally:
        collector.close()
    return float(np.mean(losses)), collector.rows()


def spectrum_metrics(weight: torch.Tensor, device: str) -> dict[str, float]:
    values = weight.to(device=device, dtype=torch.float32)
    eigen = torch.linalg.eigvalsh(values.T @ values).clamp_min(0).flip(0)
    total = eigen.sum().clamp_min(1e-30)
    probability = eigen / total
    return {
        "soft_rank": float(torch.exp(-(probability * torch.log(probability.clamp_min(1e-30))).sum())),
        "hard_rank": float(1.0 / probability.square().sum()),
        "stable_rank": float(total / eigen[0].clamp_min(1e-30)),
        "top1_energy": float(probability[0]),
        "top10_energy": float(probability[:10].sum()),
        "fro_norm": float(torch.sqrt(total)),
    }


def weight_rows(
    states: dict[str, dict[str, torch.Tensor]], device: str
) -> list[dict[str, object]]:
    rows = []
    for layer in range(24):
        parent = states["parent"][cfc_key(layer)].float()
        candidate = states["candidate"][cfc_key(layer)].float()
        parent_flat = parent.reshape(-1)
        candidate_flat = candidate.reshape(-1)
        row: dict[str, object] = {
            "layer": layer,
            "weight_cosine": float(
                torch.dot(parent_flat, candidate_flat)
                / (parent_flat.norm() * candidate_flat.norm()).clamp_min(1e-30)
            ),
            "fro_distance": float(torch.linalg.matrix_norm(parent - candidate)),
            "fro_distance_to_parent": float(
                torch.linalg.matrix_norm(parent - candidate)
                / torch.linalg.matrix_norm(parent).clamp_min(1e-30)
            ),
        }
        for name, weight in (("parent", parent), ("candidate", candidate)):
            row_norm = weight.norm(dim=1)
            col_norm = weight.norm(dim=0)
            row.update(
                {
                    f"{name}_row_norm_cv": float(row_norm.std() / row_norm.mean()),
                    f"{name}_col_norm_cv": float(col_norm.std() / col_norm.mean()),
                    **{
                        f"{name}_{key}": value
                        for key, value in spectrum_metrics(weight, device).items()
                    },
                }
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_plan(args: argparse.Namespace, plan: dict[str, Any], root: Path) -> None:
    expected = plan["inputs"]
    checks = {
        "parent_checkpoint": (args.parent_checkpoint, expected["parent_checkpoint_sha256"]),
        "candidate_checkpoint": (args.candidate_checkpoint, expected["candidate_checkpoint_sha256"]),
        "dataset_manifest": (args.data_dir / "manifest.json", expected["dataset_manifest_sha256"]),
        "source": (Path(__file__).resolve(), plan["source_sha256"]),
    }
    for name, (path, digest) in checks.items():
        if sha256(path) != digest:
            raise ValueError(f"{name} SHA-256 mismatch")
    protocol = plan["protocol"]
    if (
        args.batch_size != protocol["batch_size"]
        or args.block_size != protocol["block_size"]
        or args.batches != protocol["batches_per_window"]
        or args.seeds != protocol["validation_seeds"]
        or args.activation_layers != protocol["activation_layers"]
    ):
        raise ValueError("runtime protocol differs from preregistration")
    if variant_specs(24) != plan["variants"]:
        raise ValueError("registered variants differ from implementation")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("refusing diagnostic from a dirty worktree")


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("20260831,20260832"))
    parser.add_argument("--activation-layers", type=parse_ints, default=parse_ints("0,3,7,11,15,19,23"))
    args = parser.parse_args()
    started = time.time()
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(args, plan, root)
    if (args.output / "summary.json").exists():
        raise ValueError("refusing to overwrite an existing splice result")
    args.output.mkdir(parents=True, exist_ok=True)

    configs = {}
    states = {}
    for name, path in (
        ("parent", args.parent_checkpoint),
        ("candidate", args.candidate_checkpoint),
    ):
        configs[name], states[name] = load_checkpoint(path)
    if int(configs["parent"]["n_layer"]) != 24 or int(configs["candidate"]["n_layer"]) != 24:
        raise ValueError("registered diagnostic requires two 24-layer checkpoints")

    windows = {
        f"seed{seed}": fixed_validation_batches(
            args.data_dir,
            batch_size=args.batch_size,
            block_size=args.block_size,
            batches=args.batches,
            seed=seed,
        )
        for seed in args.seeds
    }
    token_hashes = {name: tensor_sha256(values) for name, values in windows.items()}
    evaluations: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    for spec in variant_specs(24):
        model = build_model(spec, configs, states, args.device)
        for window, batches in windows.items():
            ce, stats = evaluate(model, batches, args.device, args.activation_layers)
            evaluations.append({"variant": spec["name"], "window": window, "ce": ce})
            activation_rows.extend(
                {"variant": spec["name"], "window": window, **row}
                for row in stats
            )
            print(f"variant={spec['name']} window={window} ce={ce:.6f}", flush=True)
        del model
        torch.cuda.empty_cache()
    weights = weight_rows(states, args.device)
    decision = splice_decision(evaluations)

    write_csv(args.output / "evaluation.csv", evaluations)
    write_csv(args.output / "activation_residual.csv", activation_rows)
    write_csv(args.output / "cfc_weight_metrics.csv", weights)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "parameter_updates": 0,
        "repository_commit": git_head(root),
        "command": sys.argv,
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "plan": {"path": str(args.plan), "sha256": sha256(args.plan)},
        "inputs": {
            "parent_checkpoint": {"path": str(args.parent_checkpoint), "sha256": sha256(args.parent_checkpoint)},
            "candidate_checkpoint": {"path": str(args.candidate_checkpoint), "sha256": sha256(args.candidate_checkpoint)},
            "dataset_manifest": {"path": str(args.data_dir / "manifest.json"), "sha256": sha256(args.data_dir / "manifest.json")},
        },
        "protocol": {
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "batches_per_window": args.batches,
            "validation_seeds": args.seeds,
            "activation_layers": args.activation_layers,
            "token_sha256": token_hashes,
            "variants": variant_specs(24),
            "dtype": "bfloat16",
        },
        "evaluations": evaluations,
        "elapsed_seconds": time.time() - started,
        "artifacts": {
            name: {"path": str(args.output / name), "sha256": sha256(args.output / name)}
            for name in ("evaluation.csv", "activation_residual.csv", "cfc_weight_metrics.csv")
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
