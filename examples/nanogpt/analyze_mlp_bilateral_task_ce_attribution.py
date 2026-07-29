"""Attribute a task-conditioned bilateral chart's full-model CE gain.

The accepted endpoint state improves two fixed validation splits, but all
chart groups and layers moved.  This no-update diagnostic restores only
selected coordinates from that accepted state and evaluates the complete
language model.  It distinguishes hidden-frame, output-frame, depth-local,
and genuinely distributed chart effects without another training run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch

from examples.nanogpt.analyze_mlp_activation_chart_oracle import tensor_sha256
from examples.nanogpt.analyze_mlp_bilateral_endpoint_ce_oracle import (
    restore_chart_state,
)
from examples.nanogpt.analyze_mlp_chart_gradient_alignment import (
    load_chart_model,
    split_chart_key,
)
from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.optimize_mlp_bilateral_endpoint_ce import (
    ALL_LAYERS,
    clear_frozen_base_cache,
    evaluate_ce,
    prepare_frozen_base_cache,
)


GROUPS = (
    "hidden_rotation",
    "hidden_gain",
    "output_rotation",
    "output_gain",
)
PLAN_NAME = "124m_mlp_bilateral_task_ce_attribution_plan.json"


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


def load_state(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema_version") != (
        "mai_124m_mlp_bilateral_task_ce_chart_state_v1"
    ):
        raise ValueError(f"incompatible chart state: {path}")
    state = value.get("state")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"chart state is empty: {path}")
    return state


def combine_states(
    identity: dict[str, torch.Tensor],
    accepted: dict[str, torch.Tensor],
    selected: Callable[[int, str], bool],
) -> dict[str, torch.Tensor]:
    if set(identity) != set(accepted):
        raise ValueError("identity and accepted chart states differ")
    output: dict[str, torch.Tensor] = {}
    for key in identity:
        layer, group = split_chart_key(key)
        output[key] = (
            accepted[key].clone()
            if selected(layer, group)
            else identity[key].clone()
        )
    return output


def variant_specs() -> list[
    tuple[str, Callable[[int, str], bool]]
]:
    variants: list[tuple[str, Callable[[int, str], bool]]] = [
        ("identity", lambda layer, group: False),
        ("accepted_all", lambda layer, group: True),
    ]
    for selected_group in GROUPS:
        variants.append(
            (
                f"only_{selected_group}",
                lambda layer, group, selected_group=selected_group: (
                    group == selected_group
                ),
            )
        )
    variants.extend(
        [
            (
                "rotations_only",
                lambda layer, group: group.endswith("rotation"),
            ),
            (
                "gains_only",
                lambda layer, group: group.endswith("gain"),
            ),
        ]
    )
    for omitted_group in GROUPS:
        variants.append(
            (
                f"without_{omitted_group}",
                lambda layer, group, omitted_group=omitted_group: (
                    group != omitted_group
                ),
            )
        )
    bands = {
        "early": set(range(0, 4)),
        "middle": set(range(4, 8)),
        "late": set(range(8, 12)),
    }
    for name, layers in bands.items():
        variants.append(
            (
                f"only_depth_{name}",
                lambda layer, group, layers=layers: layer in layers,
            )
        )
    for selected_layer in ALL_LAYERS:
        variants.append(
            (
                f"only_layer_{selected_layer}",
                lambda layer, group, selected_layer=selected_layer: (
                    layer == selected_layer
                ),
            )
        )
    return variants


def passes(
    row: dict[str, object], minimum_gain: float
) -> bool:
    return (
        float(row["primary_gain"]) >= minimum_gain
        and float(row["confirmation_gain"]) >= minimum_gain
    )


def attribution_decision(
    rows: list[dict[str, object]], minimum_gain: float
) -> dict[str, object]:
    by_name = {str(row["variant"]): row for row in rows}
    accepted = by_name["accepted_all"]
    accepted_primary = float(accepted["primary_gain"])
    accepted_confirmation = float(accepted["confirmation_gain"])

    def group_supported(group: str) -> bool:
        alone = by_name[f"only_{group}"]
        without = by_name[f"without_{group}"]
        alone_passes = passes(alone, minimum_gain)
        destroys_half = (
            float(without["primary_gain"]) < 0.5 * accepted_primary
            and float(without["confirmation_gain"])
            < 0.5 * accepted_confirmation
        )
        return alone_passes or destroys_half

    output_supported = any(
        group_supported(group)
        for group in ("output_rotation", "output_gain")
    )
    hidden_supported = any(
        group_supported(group)
        for group in ("hidden_rotation", "hidden_gain")
    )
    only_groups = [by_name[f"only_{group}"] for group in GROUPS]
    bands = [
        by_name["only_depth_early"],
        by_name["only_depth_middle"],
        by_name["only_depth_late"],
    ]
    distributed = (
        passes(accepted, minimum_gain)
        and not any(passes(row, minimum_gain) for row in only_groups + bands)
    )
    best_group = max(
        only_groups,
        key=lambda row: min(
            float(row["primary_gain"]),
            float(row["confirmation_gain"]),
        ),
    )
    best_band = max(
        bands,
        key=lambda row: min(
            float(row["primary_gain"]),
            float(row["confirmation_gain"]),
        ),
    )
    if output_supported:
        next_structure = "TEST_NARROW_RESIDUAL_MLP_OUTPUT_FRAME"
    elif hidden_supported:
        next_structure = "TEST_HIDDEN_ACTIVATION_FRAME"
    else:
        next_structure = "RETAIN_DISTRIBUTED_LATE_CHART_CONTROL"
    return {
        "minimum_gain": minimum_gain,
        "accepted_primary_gain": accepted_primary,
        "accepted_confirmation_gain": accepted_confirmation,
        "output_frame_supported": output_supported,
        "hidden_frame_supported": hidden_supported,
        "distributed_chart": distributed,
        "best_group": best_group["variant"],
        "best_group_primary_gain": best_group["primary_gain"],
        "best_group_confirmation_gain": best_group[
            "confirmation_gain"
        ],
        "best_depth_band": best_band["variant"],
        "best_depth_primary_gain": best_band["primary_gain"],
        "best_depth_confirmation_gain": best_band[
            "confirmation_gain"
        ],
        "next_structure": next_structure,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-sha256",
        default=(
            "7e450e2f78fa33a049ea990386e0c7b8f9b139ddb174e4d1b"
            "fc76dd0ff0ebcdc"
        ),
    )
    parser.add_argument("--identity-state", required=True, type=Path)
    parser.add_argument(
        "--identity-state-sha256",
        default=(
            "2d18703bf53d62ab52a217730a9e63594c331c5a07118b3f7"
            "9069a18ff91aa01"
        ),
    )
    parser.add_argument("--accepted-state", required=True, type=Path)
    parser.add_argument(
        "--accepted-state-sha256",
        default=(
            "2abeee50138189f055a5462679412965a6c60d524a04c60afd"
            "a1cdc3877a6b75"
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-window-length", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--primary-seed", type=int, default=20260717)
    parser.add_argument("--confirmation-seed", type=int, default=20260718)
    parser.add_argument("--minimum-gain", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = args.data_dir / "manifest.json"
    expected = {
        args.checkpoint: args.checkpoint_sha256,
        args.identity_state: args.identity_state_sha256,
        args.accepted_state: args.accepted_state_sha256,
        manifest: args.manifest_sha256,
    }
    for path, expected_hash in expected.items():
        actual = sha256(path)
        if actual != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {path}: {actual} != {expected_hash}"
            )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    identity = load_state(args.identity_state)
    accepted = load_state(args.accepted_state)
    model = load_chart_model(
        args.checkpoint, args.device, ALL_LAYERS, 0.0
    )
    cached_modules = prepare_frozen_base_cache(
        model, torch.bfloat16
    )
    batches = {
        "primary": fixed_validation_batches(
            args.data_dir,
            args.batch_size,
            args.token_window_length,
            args.batches,
            args.primary_seed,
        ),
        "confirmation": fixed_validation_batches(
            args.data_dir,
            args.batch_size,
            args.token_window_length,
            args.batches,
            args.confirmation_seed,
        ),
    }
    digests = {
        name: tensor_sha256(torch.cat(values))
        for name, values in batches.items()
    }
    rows: list[dict[str, object]] = []
    try:
        for name, selected in variant_specs():
            state = combine_states(identity, accepted, selected)
            restore_chart_state(model, ALL_LAYERS, state)
            primary_ce = evaluate_ce(
                model, batches["primary"], args.device
            )
            confirmation_ce = evaluate_ce(
                model, batches["confirmation"], args.device
            )
            row: dict[str, object] = {
                "variant": name,
                "primary_ce": primary_ce,
                "confirmation_ce": confirmation_ce,
            }
            rows.append(row)
            print(
                f"variant={name} primary_ce={primary_ce:.6f} "
                f"confirmation_ce={confirmation_ce:.6f}",
                flush=True,
            )
    finally:
        clear_frozen_base_cache(model)
    identity_row = next(
        row for row in rows if row["variant"] == "identity"
    )
    for row in rows:
        row["primary_gain"] = (
            float(identity_row["primary_ce"]) - float(row["primary_ce"])
        )
        row["confirmation_gain"] = (
            float(identity_row["confirmation_ce"])
            - float(row["confirmation_ce"])
        )
        row["passes_both"] = passes(row, args.minimum_gain)
    decision = attribution_decision(rows, args.minimum_gain)

    csv_path = output / "task_ce_attribution.csv"
    write_csv(csv_path, rows)
    root = Path(__file__).resolve().parents[2]
    plan = (
        root
        / "examples/nanogpt/configs/selection_artifacts"
        / PLAN_NAME
    )
    summary = {
        "schema_version": (
            "mai_124m_mlp_bilateral_task_ce_attribution_result_v1"
        ),
        "repository_commit": git_head(root),
        "command": sys.argv,
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "plan": {"path": str(plan), "sha256": sha256(plan)},
        "inputs": {
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": args.checkpoint_sha256,
            },
            "identity_state": {
                "path": str(args.identity_state),
                "sha256": args.identity_state_sha256,
            },
            "accepted_state": {
                "path": str(args.accepted_state),
                "sha256": args.accepted_state_sha256,
            },
            "dataset_manifest": {
                "path": str(manifest),
                "sha256": args.manifest_sha256,
            },
        },
        "protocol": {
            "primary_seed": args.primary_seed,
            "confirmation_seed": args.confirmation_seed,
            "token_sha256": digests,
            "batch_size": args.batch_size,
            "token_window_length": args.token_window_length,
            "predicted_tokens_per_window": (
                args.token_window_length - 1
            ),
            "batches": args.batches,
            "minimum_gain": args.minimum_gain,
            "cached_block_fht_modules": cached_modules,
            "language_model_parameter_updates": 0,
        },
        "rows": rows,
        "decision": decision,
        "artifacts": {
            "attribution_csv": {
                "path": str(csv_path),
                "sha256": sha256(csv_path),
            }
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
