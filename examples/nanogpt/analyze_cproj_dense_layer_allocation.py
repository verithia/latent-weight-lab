#!/usr/bin/env python3
"""Evaluate LWT-style dense c_proj exception allocation at fixed checkpoints."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_mlp_cproj_activation_weighted_output_selector import (
    all_finite,
    file_sha256,
    git_commit,
)
from examples.nanogpt.analyze_parameter_trajectory import write_csv
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
    load_model,
)


SCHEMA_VERSION = "mai_124m_repaired_attention_cproj_dense_layer_allocation_result_v1"
EXPECTED_PLAN_SCHEMA = "mai_124m_repaired_attention_cproj_dense_layer_allocation_plan_v1"
WINDOWS = ("primary", "confirmation")


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    expected = {
        "schema_version": EXPECTED_PLAN_SCHEMA,
        "parameter_updates": 0,
        "layers": list(range(12)),
        "primary_seed": 20260905,
        "confirmation_seed": 20260906,
        "batch_size": 16,
        "block_size": 1024,
        "batches": 32,
        "cumulative_sizes": [1, 2, 3, 4, 6, 8, 10, 12],
        "maximum_selected_dense_layers": 4,
        "training_authorized": False,
    }
    observed = {
        "schema_version": plan.get("schema_version"),
        "parameter_updates": analysis.get("parameter_updates"),
        "layers": analysis.get("layers"),
        "primary_seed": analysis.get("primary_window", {}).get("seed"),
        "confirmation_seed": analysis.get("confirmation_window", {}).get(
            "seed"
        ),
        "batch_size": analysis.get("primary_window", {}).get("batch_size"),
        "block_size": analysis.get("primary_window", {}).get("block_size"),
        "batches": analysis.get("primary_window", {}).get("batches"),
        "cumulative_sizes": analysis.get("cumulative_sizes"),
        "maximum_selected_dense_layers": analysis.get(
            "maximum_selected_dense_layers"
        ),
        "training_authorized": plan.get("authorization", {}).get(
            "run_language_model_training"
        ),
    }
    if observed != expected:
        raise ValueError(
            f"dense-layer allocation plan drifted: observed={observed!r} "
            f"expected={expected!r}"
        )
    confirmation = analysis.get("confirmation_window", {})
    primary = analysis.get("primary_window", {})
    for key in ("batch_size", "block_size", "batches"):
        if confirmation.get(key) != primary.get(key):
            raise ValueError("primary and confirmation eval shapes differ")


def parameter_name(layer: int) -> str:
    return f"transformer.h.{layer}.mlp.c_proj.weight"


def checkpoint_state(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"checkpoint has no model state: {path}")
    if not isinstance(payload.get("model_config"), dict):
        raise ValueError(f"checkpoint has no model config: {path}")
    return payload


def evaluate_loss(
    model: torch.nn.Module, batches: list[torch.Tensor], device: str
) -> float:
    losses = []
    with torch.no_grad():
        for tokens in batches:
            tokens = tokens.to(device)
            _logits, loss = model(
                tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            )
            if loss is None:
                raise RuntimeError("model did not return a loss")
            losses.append(float(loss))
    return sum(losses) / len(losses)


def evaluate_restore(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    donor: dict[int, torch.Tensor],
    layers: tuple[int, ...],
    device: str,
) -> tuple[float, bool]:
    parameters = {
        layer: model.transformer.h[layer].mlp.c_proj.weight for layer in layers
    }
    originals = {
        layer: parameter.detach().clone()
        for layer, parameter in parameters.items()
    }
    try:
        with torch.no_grad():
            for layer, parameter in parameters.items():
                parameter.copy_(
                    donor[layer].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                )
        loss = evaluate_loss(model, batches, device)
    finally:
        with torch.no_grad():
            for layer, parameter in parameters.items():
                parameter.copy_(originals[layer])
    restored = all(
        torch.equal(parameter.detach(), originals[layer])
        for layer, parameter in parameters.items()
    )
    return loss, restored


def aggregate_results(
    rows: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any]:
    indexed = {
        (str(row["window"]), str(row["variant"])): float(row["loss"])
        for row in rows
    }
    baselines = {window: indexed[(window, "joint_control")] for window in WINDOWS}
    all12 = {window: indexed[(window, "all12_restore")] for window in WINDOWS}
    all12_gains = {
        window: baselines[window] - all12[window] for window in WINDOWS
    }
    single_primary = []
    for layer in range(12):
        gain = baselines["primary"] - indexed[
            ("primary", f"single_layer_{layer}")
        ]
        single_primary.append((gain, layer))
    ranking = [layer for _gain, layer in sorted(single_primary, key=lambda x: (-x[0], x[1]))]

    restore_requirements = plan["decision_rule"]["restore_control_requirements"]
    candidate_requirements = plan["decision_rule"]["candidate_requirements"]
    restore_gate = {
        "all12_primary_ce_gain": all12_gains["primary"]
        >= float(restore_requirements["all12_primary_ce_gain_minimum"]),
        "all12_confirmation_ce_gain": all12_gains["confirmation"]
        >= float(restore_requirements["all12_confirmation_ce_gain_minimum"]),
    }
    candidates = {}
    selected_k = None
    for k in plan["analysis"]["cumulative_sizes"]:
        variant = f"cumulative_top_{k}"
        losses = {window: indexed[(window, variant)] for window in WINDOWS}
        gains = {window: baselines[window] - losses[window] for window in WINDOWS}
        fractions = {
            window: (
                gains[window] / all12_gains[window]
                if all12_gains[window] > 0.0
                else None
            )
            for window in WINDOWS
        }
        rows_for_variant = [row for row in rows if row["variant"] == variant]
        gate = {
            "maximum_dense_layers": k
            <= int(candidate_requirements["maximum_dense_layers"]),
            "primary_gain_fraction": fractions["primary"] is not None
            and fractions["primary"]
            >= float(
                candidate_requirements[
                    "primary_gain_fraction_of_all12_minimum"
                ]
            ),
            "confirmation_gain_fraction": fractions["confirmation"] is not None
            and fractions["confirmation"]
            >= float(
                candidate_requirements[
                    "confirmation_gain_fraction_of_all12_minimum"
                ]
            ),
            "primary_ce_gain": gains["primary"]
            >= float(candidate_requirements["primary_ce_gain_minimum"]),
            "confirmation_ce_gain": gains["confirmation"]
            >= float(candidate_requirements["confirmation_ce_gain_minimum"]),
            "confirmation_near_all12": losses["confirmation"]
            - all12["confirmation"]
            <= float(
                candidate_requirements[
                    "confirmation_ce_above_all12_maximum"
                ]
            ),
            "finite": all_finite(
                {"losses": losses, "gains": gains, "fractions": fractions}
            ),
            "exact_restore": all(
                bool(row["exact_restore_after_eval"])
                for row in rows_for_variant
            ),
        }
        passed = all(restore_gate.values()) and all(gate.values())
        candidates[str(k)] = {
            "layers": ranking[:k],
            "losses": losses,
            "gains": gains,
            "gain_fraction_of_all12": fractions,
            "gate": gate,
            "passed": passed,
        }
        if selected_k is None and passed:
            selected_k = int(k)
    return {
        "joint_control_loss": baselines,
        "all12_restore_loss": all12,
        "all12_restore_gain": all12_gains,
        "restore_control_gate": restore_gate,
        "primary_single_layer_ranking": ranking,
        "primary_single_layer_gains": {
            str(layer): gain for gain, layer in single_primary
        },
        "candidates": candidates,
        "selected_k": selected_k,
        "selected_layers": ranking[:selected_k] if selected_k is not None else None,
        "passed": selected_k is not None,
        "classification": (
            "PASS_CPROJ_DENSE_LAYER_EXCEPTION_ALLOCATION"
            if selected_k is not None
            else "REJECT_CPROJ_DENSE_LAYER_EXCEPTION_ALLOCATION"
        ),
        "authorization": {
            "mixed_layer_implementation_authorized": selected_k is not None,
            "exact_config_mfu_preflight_authorized": selected_k is not None,
            "language_model_training_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--joint-checkpoint", required=True, type=Path)
    parser.add_argument("--dense-cproj-parent-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    identity = plan["identity"]
    if file_sha256(args.joint_checkpoint) != identity["joint_checkpoint_sha256"]:
        raise ValueError("joint checkpoint SHA-256 mismatch")
    if file_sha256(args.dense_cproj_parent_checkpoint) != identity[
        "dense_cproj_parent_checkpoint_sha256"
    ]:
        raise ValueError("dense c_proj parent checkpoint SHA-256 mismatch")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")

    parent = checkpoint_state(args.dense_cproj_parent_checkpoint)
    layers = [int(value) for value in plan["analysis"]["layers"]]
    donor = {
        layer: parent["model"][parameter_name(layer)].float().clone()
        for layer in layers
    }
    if not all(torch.isfinite(value).all() for value in donor.values()):
        raise ValueError("nonfinite donor c_proj tensor")
    model = load_model(args.joint_checkpoint, args.device)
    model.eval()
    windows = {
        name: fixed_validation_batches(
            args.data_dir,
            int(plan["analysis"][f"{name}_window"]["batch_size"]),
            int(plan["analysis"][f"{name}_window"]["block_size"]) + 1,
            int(plan["analysis"][f"{name}_window"]["batches"]),
            int(plan["analysis"][f"{name}_window"]["seed"]),
        )
        for name in WINDOWS
    }
    rows = []
    baselines = {}
    for window in WINDOWS:
        baselines[window] = evaluate_loss(model, windows[window], args.device)
        rows.append(
            {
                "window": window,
                "variant": "joint_control",
                "layers": "",
                "layer_count": 0,
                "loss": baselines[window],
                "gain_vs_joint": 0.0,
                "exact_restore_after_eval": True,
            }
        )
        loss, restored = evaluate_restore(
            model, windows[window], donor, tuple(layers), args.device
        )
        rows.append(
            {
                "window": window,
                "variant": "all12_restore",
                "layers": ",".join(str(layer) for layer in layers),
                "layer_count": 12,
                "loss": loss,
                "gain_vs_joint": baselines[window] - loss,
                "exact_restore_after_eval": restored,
            }
        )
    for layer in layers:
        for window in WINDOWS:
            loss, restored = evaluate_restore(
                model, windows[window], donor, (layer,), args.device
            )
            rows.append(
                {
                    "window": window,
                    "variant": f"single_layer_{layer}",
                    "layers": str(layer),
                    "layer_count": 1,
                    "loss": loss,
                    "gain_vs_joint": baselines[window] - loss,
                    "exact_restore_after_eval": restored,
                }
            )

    primary_single = {
        layer: next(
            float(row["gain_vs_joint"])
            for row in rows
            if row["window"] == "primary"
            and row["variant"] == f"single_layer_{layer}"
        )
        for layer in layers
    }
    ranking = sorted(layers, key=lambda layer: (-primary_single[layer], layer))
    for k in plan["analysis"]["cumulative_sizes"]:
        selected = tuple(ranking[: int(k)])
        for window in WINDOWS:
            loss, restored = evaluate_restore(
                model, windows[window], donor, selected, args.device
            )
            rows.append(
                {
                    "window": window,
                    "variant": f"cumulative_top_{k}",
                    "layers": ",".join(str(layer) for layer in selected),
                    "layer_count": int(k),
                    "loss": loss,
                    "gain_vs_joint": baselines[window] - loss,
                    "exact_restore_after_eval": restored,
                }
            )

    aggregate = aggregate_results(rows, plan)
    args.output.mkdir(parents=True)
    rows_path = args.output / "cproj_dense_layer_allocation_rows.csv"
    result_path = args.output / "cproj_dense_layer_allocation_result.json"
    write_csv(rows_path, rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": aggregate["classification"],
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_cproj_dense_layer_allocation",
            "parameter_updates": 0,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
            "started_at": started_at,
            "elapsed_seconds": time.time() - started,
        },
        "identity": {
            "plan_path": str(args.plan),
            "plan_sha256": file_sha256(args.plan),
            "joint_checkpoint_sha256": file_sha256(args.joint_checkpoint),
            "dense_cproj_parent_checkpoint_sha256": file_sha256(
                args.dense_cproj_parent_checkpoint
            ),
            "dataset_manifest_sha256": file_sha256(manifest),
        },
        "aggregate": aggregate,
    }
    result["artifacts"] = {"rows_sha256": file_sha256(rows_path)}
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": aggregate["classification"],
                "selected_k": aggregate["selected_k"],
                "selected_layers": aggregate["selected_layers"],
                "output": str(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
