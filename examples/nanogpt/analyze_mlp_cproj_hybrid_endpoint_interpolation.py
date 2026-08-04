#!/usr/bin/env python3
"""Diagnose c_proj endpoint direction versus wider MLP co-adaptation.

This is a zero-update terminal-checkpoint diagnostic.  It evaluates symmetric
affine c_proj interpolations in each endpoint's native non-c_proj context,
depth-band transplants, and two wider-context controls on two independent
fixed validation windows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    get_batch,
    make_fixed_eval_indices,
)


SCHEMA_VERSION = "mai_124m_mlp_cproj_hybrid_endpoint_interpolation_plan_v1"
CONTEXTS = ("hybrid", "parent")
INTERIOR_ALPHAS = (0.25, 0.50, 0.75)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def interpolation_variant(context: str, alpha: float) -> str:
    return f"{context}_context_all_cproj_alpha{alpha_tag(alpha)}"


def build_variant_specs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for context in CONTEXTS:
        for alpha in plan["interpolation"]["alphas"]:
            specs.append(
                {
                    "variant": interpolation_variant(context, float(alpha)),
                    "context": context,
                    "kind": "all_cproj_interpolation",
                    "alpha": float(alpha),
                    "layers": list(plan["interpolation"]["layers"]),
                }
            )
    for name, layers in plan["interpolation"][
        "depth_band_variants_in_hybrid_context"
    ].items():
        specs.append(
            {
                "variant": name,
                "context": "hybrid",
                "kind": "cproj_depth_band_transplant",
                "alpha": 1.0,
                "layers": list(layers),
            }
        )
    for name in plan["interpolation"][
        "secondary_wider_context_variants_in_hybrid_context"
    ]:
        specs.append(
            {
                "variant": name,
                "context": "hybrid",
                "kind": "wider_context_transplant",
                "alpha": 1.0,
                "layers": list(plan["interpolation"]["layers"]),
            }
        )
    return specs


def family_tensor_names(
    state: dict[str, torch.Tensor], family: str, layers: list[int]
) -> list[str]:
    names: list[str] = []
    for layer in layers:
        if family == "cproj":
            expected = f"transformer.h.{layer}.mlp.c_proj.weight"
            if expected not in state:
                raise ValueError(f"missing required c_proj tensor: {expected}")
            names.append(expected)
            continue
        if family == "cfc":
            prefix = f"transformer.h.{layer}.mlp.c_fc."
        elif family == "ln2":
            prefix = f"transformer.h.{layer}.ln_2."
        else:
            raise ValueError(f"unknown tensor family: {family}")
        matched = sorted(name for name in state if name.startswith(prefix))
        if not matched:
            raise ValueError(f"no tensors found for {family} layer {layer}")
        names.extend(matched)
    return names


def mutable_tensor_names(
    state: dict[str, torch.Tensor], layers: list[int]
) -> list[str]:
    names: list[str] = []
    for family in ("cproj", "cfc", "ln2"):
        names.extend(family_tensor_names(state, family, layers))
    return sorted(set(names))


@torch.no_grad()
def install_variant(
    model: GPT,
    *,
    base_state: dict[str, torch.Tensor],
    other_state: dict[str, torch.Tensor],
    spec: dict[str, Any],
    all_layers: list[int],
) -> None:
    """Restore the native context, then apply exactly one registered transplant."""
    live = model.state_dict(keep_vars=True)
    for name in mutable_tensor_names(base_state, all_layers):
        target = live[name]
        target.copy_(base_state[name].to(device=target.device, dtype=target.dtype))

    kind = str(spec["kind"])
    layers = [int(layer) for layer in spec["layers"]]
    if kind in {"all_cproj_interpolation", "cproj_depth_band_transplant"}:
        alpha = float(spec["alpha"])
        for name in family_tensor_names(base_state, "cproj", layers):
            value = torch.lerp(base_state[name].float(), other_state[name].float(), alpha)
            target = live[name]
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    elif kind == "wider_context_transplant":
        families = ["cproj", "cfc"]
        if str(spec["variant"]).endswith("plus_all_ln2"):
            families.append("ln2")
        for family in families:
            for name in family_tensor_names(base_state, family, layers):
                target = live[name]
                target.copy_(other_state[name].to(device=target.device, dtype=target.dtype))
    else:
        raise ValueError(f"unknown variant kind: {kind}")


def validate_state_topology(
    parent: dict[str, torch.Tensor], hybrid: dict[str, torch.Tensor]
) -> None:
    if parent.keys() != hybrid.keys():
        missing_parent = sorted(hybrid.keys() - parent.keys())
        missing_hybrid = sorted(parent.keys() - hybrid.keys())
        raise ValueError(
            f"model state topology differs: parent_missing={missing_parent[:3]} "
            f"hybrid_missing={missing_hybrid[:3]}"
        )
    for name in parent:
        if parent[name].shape != hybrid[name].shape:
            raise ValueError(
                f"model tensor shape differs for {name}: "
                f"{tuple(parent[name].shape)} != {tuple(hybrid[name].shape)}"
            )


def resolve_metadata_sidecar(checkpoint: Path, expected_sha256: str) -> Path:
    candidates = [
        checkpoint.with_name(checkpoint.name + ".meta.json"),
        checkpoint.with_suffix(".meta.json"),
        checkpoint.parent / "checkpoint_meta.json",
    ]
    candidates.extend(sorted(checkpoint.parent.glob("*.json")))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if file_sha256(candidate) == expected_sha256:
            return candidate
    raise ValueError(
        f"no checkpoint metadata sidecar matches {expected_sha256} beside {checkpoint}"
    )


def verify_identity(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    parent_checkpoint: Path,
    hybrid_checkpoint: Path,
    data_dir: Path,
) -> dict[str, Any]:
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected plan schema")
    inputs = plan["inputs"]
    observed = {
        "plan_sha256": file_sha256(plan_path),
        "parent_checkpoint_sha256": file_sha256(parent_checkpoint),
        "hybrid_checkpoint_sha256": file_sha256(hybrid_checkpoint),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
    }
    expected = {
        "parent_checkpoint_sha256": inputs["parent_checkpoint_sha256"],
        "hybrid_checkpoint_sha256": inputs["hybrid_checkpoint_sha256"],
        "dataset_manifest_sha256": inputs["dataset_manifest_sha256"],
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError(f"{key} mismatch: {observed[key]} != {value}")

    for endpoint, checkpoint in (
        ("parent", parent_checkpoint),
        ("hybrid", hybrid_checkpoint),
    ):
        result_path = REPO_ROOT / inputs[f"{endpoint}_result"]
        result_hash = file_sha256(result_path)
        if result_hash != inputs[f"{endpoint}_result_sha256"]:
            raise ValueError(f"{endpoint} result artifact hash mismatch")
        sidecar = resolve_metadata_sidecar(
            checkpoint, inputs[f"{endpoint}_checkpoint_meta_sha256"]
        )
        observed[f"{endpoint}_result_sha256"] = result_hash
        observed[f"{endpoint}_checkpoint_meta"] = str(sidecar)
        observed[f"{endpoint}_checkpoint_meta_sha256"] = file_sha256(sidecar)
    return observed


def endpoint_geometry(
    parent_state: dict[str, torch.Tensor],
    hybrid_state: dict[str, torch.Tensor],
    layers: list[int],
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        name = f"transformer.h.{layer}.mlp.c_proj.weight"
        parent = parent_state[name].to(device=device, dtype=torch.float32)
        hybrid = hybrid_state[name].to(device=device, dtype=torch.float32)
        delta = hybrid - parent
        parent_flat = parent.reshape(-1)
        hybrid_flat = hybrid.reshape(-1)
        parent_fro = parent.norm()
        hybrid_fro = hybrid.norm()
        delta_fro = delta.norm()
        singular = torch.linalg.svdvals(delta)
        singular_energy = singular.square()
        total_singular_energy = singular_energy.sum().clamp_min(1e-30)
        parent_row_gram = parent @ parent.transpose(0, 1)
        hybrid_row_gram = hybrid @ hybrid.transpose(0, 1)
        parent_col_gram = parent.transpose(0, 1) @ parent
        hybrid_col_gram = hybrid.transpose(0, 1) @ hybrid
        rows.append(
            {
                "layer": layer,
                "flattened_weight_cosine": float(
                    torch.dot(parent_flat, hybrid_flat)
                    / (parent_fro * hybrid_fro).clamp_min(1e-30)
                ),
                "hybrid_minus_parent_frobenius_norm": float(delta_fro),
                "distance_normalized_by_parent_frobenius_norm": float(
                    delta_fro / parent_fro.clamp_min(1e-30)
                ),
                "hybrid_to_parent_frobenius_norm_ratio": float(
                    hybrid_fro / parent_fro.clamp_min(1e-30)
                ),
                "delta_stable_rank": float(
                    total_singular_energy / singular_energy.max().clamp_min(1e-30)
                ),
                "delta_top1_singular_energy": float(
                    singular_energy.max() / total_singular_energy
                ),
                "normalized_row_gram_difference": float(
                    (hybrid_row_gram - parent_row_gram).norm()
                    / parent_row_gram.norm().clamp_min(1e-30)
                ),
                "normalized_column_gram_difference": float(
                    (hybrid_col_gram - parent_col_gram).norm()
                    / parent_col_gram.norm().clamp_min(1e-30)
                ),
            }
        )
        del parent, hybrid, delta, singular
        del parent_row_gram, hybrid_row_gram, parent_col_gram, hybrid_col_gram
    return rows


@torch.no_grad()
def evaluate_losses(
    model: GPT,
    *,
    data_dir: Path,
    indices: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
    dtype: str,
    source: TokenBatchSource,
    progress: dict[str, Any],
) -> list[float]:
    if dtype == "bfloat16":
        autocast_dtype = torch.bfloat16
    elif dtype == "float16":
        autocast_dtype = torch.float16
    elif dtype == "float32":
        autocast_dtype = torch.float32
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    losses: list[float] = []
    model.eval()
    for batch_index in range(indices.shape[0]):
        x, y = get_batch(
            data_dir,
            "val",
            batch_size,
            block_size,
            device,
            indices=indices[batch_index],
            source=source,
        )
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if device.startswith("cuda") and dtype != "float32"
            else nullcontext()
        )
        with context:
            _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss for {progress} at batch {batch_index}"
            )
        losses.append(float(loss.detach().float().cpu()))
        if batch_index == 0 or batch_index + 1 == indices.shape[0]:
            print(
                json.dumps(
                    {
                        **progress,
                        "eval_batch": batch_index + 1,
                        "eval_batches": int(indices.shape[0]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return losses


def summarize_losses(losses: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(losses, dtype=torch.float64)
    mean = float(tensor.mean())
    if tensor.numel() <= 1:
        return mean, float("nan")
    return mean, float(tensor.std(unbiased=True) / math.sqrt(tensor.numel()))


def rows_by_window_variant(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(row["window_seed"]), str(row["variant"])): row for row in rows
    }


def classify_endpoint_result(
    rows: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any]:
    by_key = rows_by_window_variant(rows)
    seeds = [int(seed) for seed in plan["evaluation"]["independent_window_seeds"]]
    per_window: list[dict[str, Any]] = []
    for seed in seeds:
        h0 = float(by_key[(seed, interpolation_variant("hybrid", 0.0))]["val_ce"])
        h1 = float(by_key[(seed, interpolation_variant("hybrid", 1.0))]["val_ce"])
        p0 = float(by_key[(seed, interpolation_variant("parent", 0.0))]["val_ce"])
        native_gap = h0 - p0
        recovery = (h0 - h1) / native_gap if native_gap > 0 else float("nan")
        improvements = {
            alpha_tag(alpha): h0
            - float(
                by_key[(seed, interpolation_variant("hybrid", alpha))]["val_ce"]
            )
            for alpha in (*INTERIOR_ALPHAS, 1.0)
        }
        per_window.append(
            {
                "window_seed": seed,
                "hybrid_native_ce": h0,
                "parent_native_ce": p0,
                "native_gap": native_gap,
                "hybrid_context_parent_cproj_ce": h1,
                "cproj_recovery_fraction": recovery,
                "hybrid_context_improvement_by_alpha": improvements,
            }
        )

    finite_positive_gaps = all(
        math.isfinite(float(row["native_gap"])) and float(row["native_gap"]) > 0
        for row in per_window
    )
    recoveries = [float(row["cproj_recovery_fraction"]) for row in per_window]
    any_common_improving_alpha = any(
        all(
            float(row["hybrid_context_improvement_by_alpha"][alpha_tag(alpha)]) > 0
            for row in per_window
        )
        for alpha in (*INTERIOR_ALPHAS, 1.0)
    )
    any_common_interior_0p01 = any(
        all(
            float(row["hybrid_context_improvement_by_alpha"][alpha_tag(alpha)])
            >= 0.01
            for row in per_window
        )
        for alpha in INTERIOR_ALPHAS
    )
    if (
        finite_positive_gaps
        and all(value >= 0.50 for value in recoveries)
        and any_common_improving_alpha
    ):
        primary = "HYBRID_CPROJ_ENDPOINT_DIRECTION_DOMINATES"
    elif (
        finite_positive_gaps
        and all(value <= 0.25 for value in recoveries)
        and not any_common_interior_0p01
    ):
        primary = "WIDER_BLOCK_COADAPTATION_DOMINATES"
    else:
        primary = "MIXED_ENDPOINT_DIRECTION_AND_COADAPTATION"

    barrier_rows: list[dict[str, Any]] = []
    barrier = False
    for context in CONTEXTS:
        for alpha in INTERIOR_ALPHAS:
            common = []
            for seed in seeds:
                ce0 = float(
                    by_key[(seed, interpolation_variant(context, 0.0))]["val_ce"]
                )
                ce1 = float(
                    by_key[(seed, interpolation_variant(context, 1.0))]["val_ce"]
                )
                observed = float(
                    by_key[(seed, interpolation_variant(context, alpha))]["val_ce"]
                )
                linear = (1.0 - alpha) * ce0 + alpha * ce1
                excess = observed - linear
                common.append(excess >= 0.01)
                barrier_rows.append(
                    {
                        "context": context,
                        "alpha": alpha,
                        "window_seed": seed,
                        "observed_ce": observed,
                        "linear_endpoint_ce": linear,
                        "excess_ce": excess,
                    }
                )
            if all(common):
                barrier = True
    return {
        "primary_classification": primary,
        "functional_barrier_classification": (
            "INTERIOR_FUNCTIONAL_BARRIER"
            if barrier
            else "NO_LARGE_INTERIOR_FUNCTIONAL_BARRIER"
        ),
        "per_window": per_window,
        "barrier_rows": barrier_rows,
        "finite_positive_native_gaps": finite_positive_gaps,
        "any_common_improving_alpha": any_common_improving_alpha,
        "any_common_interior_improvement_at_least_0p01": any_common_interior_0p01,
        "language_model_training_authorized": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_geometry(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        key: float(sum(float(row[key]) for row in rows) / len(rows))
        for key in rows[0]
        if key != "layer"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--hybrid-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.time()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    identity = verify_identity(
        plan=plan,
        plan_path=args.plan,
        parent_checkpoint=args.parent_checkpoint,
        hybrid_checkpoint=args.hybrid_checkpoint,
        data_dir=args.data_dir,
    )
    print(json.dumps({"identity_verified": identity}, sort_keys=True), flush=True)

    parent_checkpoint = torch.load(
        args.parent_checkpoint, map_location="cpu", weights_only=False
    )
    hybrid_checkpoint = torch.load(
        args.hybrid_checkpoint, map_location="cpu", weights_only=False
    )
    if parent_checkpoint["model_config"] != hybrid_checkpoint["model_config"]:
        raise ValueError("endpoint model configurations differ")
    parent_state = parent_checkpoint["model"]
    hybrid_state = hybrid_checkpoint["model"]
    validate_state_topology(parent_state, hybrid_state)
    layers = [int(layer) for layer in plan["interpolation"]["layers"]]

    print(json.dumps({"phase": "parameter_geometry"}), flush=True)
    geometry_rows = endpoint_geometry(parent_state, hybrid_state, layers, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    evaluation = plan["evaluation"]
    dtype = str(evaluation["dtype"])
    batch_size = int(evaluation["batch_size"])
    block_size = int(evaluation["block_size"])
    eval_iters = int(evaluation["eval_batches_per_window"])
    seeds = [int(seed) for seed in evaluation["independent_window_seeds"]]
    fixed_indices: dict[int, dict[str, torch.Tensor]] = {}
    fixed_digests: dict[str, str] = {}
    for seed in seeds:
        indices = make_fixed_eval_indices(
            args.data_dir, batch_size, block_size, eval_iters, seed
        )
        fixed_indices[seed] = indices
        fixed_digests[str(seed)] = fixed_eval_indices_digest(indices)

    model = GPT(GPTConfig(**parent_checkpoint["model_config"]))
    model.load_state_dict(hybrid_state, strict=True)
    model.to(args.device)
    model.eval()
    source = TokenBatchSource(args.data_dir)
    specs = build_variant_specs(plan)
    rows: list[dict[str, Any]] = []
    for context_name in CONTEXTS:
        base_state = hybrid_state if context_name == "hybrid" else parent_state
        other_state = parent_state if context_name == "hybrid" else hybrid_state
        model.load_state_dict(base_state, strict=True)
        context_specs = [spec for spec in specs if spec["context"] == context_name]
        for spec in context_specs:
            install_variant(
                model,
                base_state=base_state,
                other_state=other_state,
                spec=spec,
                all_layers=layers,
            )
            for seed in seeds:
                print(
                    json.dumps(
                        {
                            "phase": "fixed_validation",
                            "context": context_name,
                            "variant": spec["variant"],
                            "window_seed": seed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                losses = evaluate_losses(
                    model,
                    data_dir=args.data_dir,
                    indices=fixed_indices[seed]["val"],
                    batch_size=batch_size,
                    block_size=block_size,
                    device=args.device,
                    dtype=dtype,
                    source=source,
                    progress={
                        "variant": spec["variant"],
                        "window_seed": seed,
                    },
                )
                mean, standard_error = summarize_losses(losses)
                rows.append(
                    {
                        **spec,
                        "layers": ",".join(str(layer) for layer in spec["layers"]),
                        "window_seed": seed,
                        "fixed_eval_indices_sha256": fixed_digests[str(seed)],
                        "val_ce": mean,
                        "val_ce_standard_error": standard_error,
                        "eval_batches": eval_iters,
                        "batch_size": batch_size,
                        "block_size": block_size,
                    }
                )

    decision = classify_endpoint_result(rows, plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "variant_validation_ce.csv"
    geometry_path = args.output_dir / "cproj_endpoint_geometry.csv"
    result_path = args.output_dir / "result.json"
    metadata_path = args.output_dir / "metadata.json"
    write_csv(rows_path, rows)
    write_csv(geometry_path, geometry_rows)
    result = {
        "schema_version": "mai_124m_mlp_cproj_hybrid_endpoint_interpolation_result_v1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plan": {
            "path": str(args.plan),
            "sha256": identity["plan_sha256"],
        },
        "identity": identity,
        "fixed_eval_indices_sha256_by_seed": fixed_digests,
        "parameter_geometry_mean_across_layers": mean_geometry(geometry_rows),
        "parameter_geometry_by_layer": geometry_rows,
        "variant_results": rows,
        "decision": decision,
        "limitations": [
            "This is a zero-update endpoint intervention, not a training run.",
            "The affine endpoint segment cannot establish the intrinsic dimension or curvature of the full training trajectory.",
            "The secondary wider-context controls are interpretive and do not alter the frozen primary classification."
        ],
        "execution": {
            "host": socket.gethostname(),
            "device": args.device,
            "git_commit": git_commit(),
            "entrypoint": "examples/nanogpt/analyze_mlp_cproj_hybrid_endpoint_interpolation.py",
            "command": sys.argv,
            "parameter_updates": 0,
            "direct_foreground_polling": True,
            "watchdog": False,
            "callback": False,
            "elapsed_seconds": time.time() - started,
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    metadata = {
        "schema_version": "mai_zero_update_diagnostic_metadata_v1",
        "result_sha256": file_sha256(result_path),
        "variant_validation_ce_csv_sha256": file_sha256(rows_path),
        "cproj_endpoint_geometry_csv_sha256": file_sha256(geometry_path),
        "source_sha256": file_sha256(Path(__file__)),
        "plan_sha256": identity["plan_sha256"],
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "git_commit": git_commit(),
        "entrypoint": "examples/nanogpt/analyze_mlp_cproj_hybrid_endpoint_interpolation.py",
        "command": sys.argv,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "result": str(result_path),
                "metadata": str(metadata_path),
                "decision": decision,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
