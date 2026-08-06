#!/usr/bin/env python3
"""Evaluate whole-donor-MLP functional layer allocation at fixed checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nanogpt.analyze_cproj_dense_layer_allocation import evaluate_loss
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


SCHEMA_VERSION = (
    "mai_124m_repaired_attention_whole_mlp_functional_layer_allocation_result_v1"
)
EXPECTED_PLAN_SCHEMA = (
    "mai_124m_repaired_attention_whole_mlp_functional_layer_allocation_plan_v1"
)
WINDOWS = ("primary", "confirmation")


def validate_plan(plan: dict[str, Any]) -> None:
    analysis = plan.get("analysis", {})
    expected = {
        "schema_version": EXPECTED_PLAN_SCHEMA,
        "parameter_updates": 0,
        "layers": list(range(12)),
        "primary_seed": 20260907,
        "confirmation_seed": 20260908,
        "batch_size": 16,
        "block_size": 1024,
        "batches": 32,
        "cumulative_sizes": [1, 2, 3, 4, 6, 8, 10, 12],
        "maximum_selected_donor_layers": 4,
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
        "maximum_selected_donor_layers": analysis.get(
            "maximum_selected_donor_layers"
        ),
        "training_authorized": plan.get("authorization", {}).get(
            "run_language_model_training"
        ),
    }
    if observed != expected:
        raise ValueError(
            f"whole-MLP functional allocation plan drifted: "
            f"observed={observed!r} expected={expected!r}"
        )
    primary = analysis.get("primary_window", {})
    confirmation = analysis.get("confirmation_window", {})
    for key in ("batch_size", "block_size", "batches"):
        if confirmation.get(key) != primary.get(key):
            raise ValueError("primary and confirmation eval shapes differ")


def mlp_tree_identity(model: torch.nn.Module) -> tuple[Any, ...]:
    rows = []
    for layer, block in enumerate(model.transformer.h):
        module = block.mlp
        rows.append(
            (
                layer,
                id(module),
                tuple((name, id(value)) for name, value in module.named_parameters()),
                tuple((name, id(value)) for name, value in module.named_buffers()),
            )
        )
    return tuple(rows)


def mlp_state_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for layer, block in enumerate(model.transformer.h):
        for name, value in sorted(block.mlp.state_dict().items()):
            tensor = value.detach().cpu().contiguous().reshape(-1)
            digest.update(f"{layer}:{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def evaluate_patch(
    joint: torch.nn.Module,
    donor: torch.nn.Module,
    batches: list[torch.Tensor],
    layers: tuple[int, ...],
    device: str,
) -> tuple[float, bool]:
    originals = {layer: joint.transformer.h[layer].mlp for layer in layers}
    original_parameter_ids = {
        layer: tuple(id(value) for value in module.parameters())
        for layer, module in originals.items()
    }
    try:
        for layer in layers:
            joint.transformer.h[layer].mlp = donor.transformer.h[layer].mlp
        loss = evaluate_loss(joint, batches, device)
    finally:
        for layer, module in originals.items():
            joint.transformer.h[layer].mlp = module
    restored = all(
        joint.transformer.h[layer].mlp is originals[layer]
        and tuple(
            id(value) for value in joint.transformer.h[layer].mlp.parameters()
        )
        == original_parameter_ids[layer]
        for layer in layers
    )
    return loss, restored


def aggregate_results(
    rows: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any]:
    indexed = {
        (str(row["window"]), str(row["variant"])): float(row["loss"])
        for row in rows
    }
    joint = {window: indexed[(window, "joint_control")] for window in WINDOWS}
    donor = {window: indexed[(window, "donor_control")] for window in WINDOWS}
    all12 = {
        window: indexed[(window, "all12_functional_patch")]
        for window in WINDOWS
    }
    donor_gains = {window: joint[window] - donor[window] for window in WINDOWS}
    all12_gains = {window: joint[window] - all12[window] for window in WINDOWS}
    single_primary = []
    for layer in range(12):
        gain = joint["primary"] - indexed[
            ("primary", f"single_layer_{layer}")
        ]
        single_primary.append((gain, layer))
    ranking = [
        layer
        for _gain, layer in sorted(single_primary, key=lambda value: (-value[0], value[1]))
    ]

    control_requirements = plan["decision_rule"]["control_requirements"]
    candidate_requirements = plan["decision_rule"]["candidate_requirements"]
    control_gate = {
        "donor_primary_gain": donor_gains["primary"]
        >= float(control_requirements["donor_primary_ce_gain_over_joint_minimum"]),
        "donor_confirmation_gain": donor_gains["confirmation"]
        >= float(
            control_requirements[
                "donor_confirmation_ce_gain_over_joint_minimum"
            ]
        ),
        "all12_primary_gain": all12_gains["primary"]
        >= float(control_requirements["all12_primary_ce_gain_over_joint_minimum"]),
        "all12_confirmation_gain": all12_gains["confirmation"]
        >= float(
            control_requirements[
                "all12_confirmation_ce_gain_over_joint_minimum"
            ]
        ),
    }
    candidates = {}
    selected_k = None
    for k in plan["analysis"]["cumulative_sizes"]:
        variant = f"cumulative_top_{k}"
        losses = {window: indexed[(window, variant)] for window in WINDOWS}
        gains = {window: joint[window] - losses[window] for window in WINDOWS}
        fractions = {
            window: (
                gains[window] / all12_gains[window]
                if all12_gains[window] > 0.0
                else None
            )
            for window in WINDOWS
        }
        arm_rows = [row for row in rows if row["variant"] == variant]
        gate = {
            "maximum_donor_layers": k
            <= int(candidate_requirements["maximum_donor_layers"]),
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
            >= float(
                candidate_requirements[
                    "primary_ce_gain_over_joint_minimum"
                ]
            ),
            "confirmation_ce_gain": gains["confirmation"]
            >= float(
                candidate_requirements[
                    "confirmation_ce_gain_over_joint_minimum"
                ]
            ),
            "confirmation_near_all12": losses["confirmation"]
            - all12["confirmation"]
            <= float(
                candidate_requirements["confirmation_ce_above_all12_maximum"]
            ),
            "finite": all_finite(
                {"losses": losses, "gains": gains, "fractions": fractions}
            ),
            "exact_restore": all(
                bool(row["exact_module_restore_after_eval"])
                for row in arm_rows
            ),
        }
        passed = all(control_gate.values()) and all(gate.values())
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
        "joint_control_loss": joint,
        "donor_control_loss": donor,
        "donor_control_gain": donor_gains,
        "all12_functional_patch_loss": all12,
        "all12_functional_patch_gain": all12_gains,
        "control_gate": control_gate,
        "primary_single_layer_ranking": ranking,
        "primary_single_layer_gains": {
            str(layer): gain for gain, layer in single_primary
        },
        "candidates": candidates,
        "selected_k": selected_k,
        "selected_layers": ranking[:selected_k] if selected_k is not None else None,
        "passed": selected_k is not None,
        "classification": (
            "PASS_WHOLE_MLP_FUNCTIONAL_LAYER_ALLOCATION"
            if selected_k is not None
            else "REJECT_WHOLE_MLP_FUNCTIONAL_LAYER_ALLOCATION"
        ),
        "authorization": {
            "mixed_layer_training_implementation_authorized": False,
            "exact_config_mfu_preflight_authorized": False,
            "language_model_training_authorized": False,
        },
    }


def append_row(
    rows: list[dict[str, Any]],
    window: str,
    variant: str,
    layers: tuple[int, ...],
    loss: float,
    joint_loss: float,
    restored: bool,
) -> None:
    rows.append(
        {
            "window": window,
            "variant": variant,
            "layers": ",".join(str(layer) for layer in layers),
            "layer_count": len(layers),
            "loss": loss,
            "gain_vs_joint": joint_loss - loss,
            "exact_module_restore_after_eval": restored,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--joint-checkpoint", required=True, type=Path)
    parser.add_argument("--donor-checkpoint", required=True, type=Path)
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
    if file_sha256(args.donor_checkpoint) != identity["donor_checkpoint_sha256"]:
        raise ValueError("donor checkpoint SHA-256 mismatch")
    manifest = args.data_dir / "manifest.json"
    if file_sha256(manifest) != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest SHA-256 mismatch")

    joint = load_model(args.joint_checkpoint, args.device)
    donor = load_model(args.donor_checkpoint, args.device)
    joint.eval()
    donor.eval()
    joint_identity_before = mlp_tree_identity(joint)
    donor_identity_before = mlp_tree_identity(donor)
    joint_fingerprint_before = mlp_state_fingerprint(joint)
    donor_fingerprint_before = mlp_state_fingerprint(donor)
    layers = [int(value) for value in plan["analysis"]["layers"]]
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
    rows: list[dict[str, Any]] = []
    joint_losses = {}
    for window in WINDOWS:
        joint_losses[window] = evaluate_loss(joint, windows[window], args.device)
        append_row(
            rows,
            window,
            "joint_control",
            (),
            joint_losses[window],
            joint_losses[window],
            True,
        )
        donor_loss = evaluate_loss(donor, windows[window], args.device)
        append_row(
            rows,
            window,
            "donor_control",
            tuple(layers),
            donor_loss,
            joint_losses[window],
            True,
        )
        loss, restored = evaluate_patch(
            joint, donor, windows[window], tuple(layers), args.device
        )
        append_row(
            rows,
            window,
            "all12_functional_patch",
            tuple(layers),
            loss,
            joint_losses[window],
            restored,
        )

    for layer in layers:
        for window in WINDOWS:
            loss, restored = evaluate_patch(
                joint, donor, windows[window], (layer,), args.device
            )
            append_row(
                rows,
                window,
                f"single_layer_{layer}",
                (layer,),
                loss,
                joint_losses[window],
                restored,
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
            loss, restored = evaluate_patch(
                joint, donor, windows[window], selected, args.device
            )
            append_row(
                rows,
                window,
                f"cumulative_top_{k}",
                selected,
                loss,
                joint_losses[window],
                restored,
            )

    joint_identity_after = mlp_tree_identity(joint)
    donor_identity_after = mlp_tree_identity(donor)
    joint_fingerprint_after = mlp_state_fingerprint(joint)
    donor_fingerprint_after = mlp_state_fingerprint(donor)
    integrity = {
        "joint_module_and_parameter_identity_unchanged": joint_identity_after
        == joint_identity_before,
        "donor_module_and_parameter_identity_unchanged": donor_identity_after
        == donor_identity_before,
        "joint_mlp_state_fingerprint_before": joint_fingerprint_before,
        "joint_mlp_state_fingerprint_after": joint_fingerprint_after,
        "joint_mlp_state_unchanged": joint_fingerprint_after
        == joint_fingerprint_before,
        "donor_mlp_state_fingerprint_before": donor_fingerprint_before,
        "donor_mlp_state_fingerprint_after": donor_fingerprint_after,
        "donor_mlp_state_unchanged": donor_fingerprint_after
        == donor_fingerprint_before,
    }
    if not all(
        value
        for key, value in integrity.items()
        if key.endswith("_unchanged")
    ):
        raise RuntimeError(f"module integrity failed: {integrity}")

    aggregate = aggregate_results(rows, plan)
    args.output.mkdir(parents=True)
    rows_path = args.output / "whole_mlp_functional_layer_rows.csv"
    result_path = args.output / "whole_mlp_functional_layer_result.json"
    write_csv(rows_path, rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": aggregate["classification"],
        "execution": {
            "host": "PRO6",
            "device": args.device,
            "git_commit": git_commit(REPO_ROOT),
            "entrypoint": "examples.nanogpt.analyze_whole_mlp_functional_layer_allocation",
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
            "donor_checkpoint_sha256": file_sha256(args.donor_checkpoint),
            "dataset_manifest_sha256": file_sha256(manifest),
        },
        "integrity": integrity,
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
