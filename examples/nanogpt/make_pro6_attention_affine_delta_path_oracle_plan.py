#!/usr/bin/env python3
"""Write the immutable attention affine-delta path-oracle plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKTREE = Path(
    "/mnt/ssd-data/orj/MappingNetworks/latent-weight-lab-attention-affine-path"
)
SCIENTIFIC = Path(
    "/mnt/ssd-data/orj/MappingNetworks/outputs/"
    "pro6_mai_v3_attention_dense_5tpp_replay/scientific"
)
OUTPUT = Path(
    "/mnt/ssd-data/orj/MappingNetworks/outputs/"
    "pro6_mai_v3_attention_affine_delta_path_oracle_v1"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan() -> dict[str, Any]:
    analyzer = REPO_ROOT / "examples/nanogpt/analyze_attention_affine_delta_path_oracle.py"
    activation = REPO_ROOT / (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_attention_paper_activation_oracle_result.json"
    )
    mlp_affine = REPO_ROOT / (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_mlp_cproj_affine_delta_result.json"
    )
    product = REPO_ROOT / (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_attention_product_fht_residual_gate_result.json"
    )
    trajectory_steps = list(range(0, 2341, 60)) + [2373]
    plan_path = WORKTREE / (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_attention_affine_delta_path_oracle_plan.json"
    )
    command = [
        "/home/pro6000-9980x/MappingNetworks/.venv/bin/python",
        "-u",
        "-m",
        "examples.nanogpt.analyze_attention_affine_delta_path_oracle",
        "--plan",
        str(plan_path),
        "--trajectory-dir",
        str(SCIENTIFIC / "parameter_trajectory"),
        "--probe-dir",
        str(SCIENTIFIC / "optimizer_probe"),
        "--terminal-checkpoint",
        str(SCIENTIFIC / "ckpt.pt"),
        "--data-dir",
        "/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b",
        "--output-dir",
        str(OUTPUT),
        "--device",
        "cuda",
    ]
    return {
        "schema_version": "mai_124m_attention_affine_delta_path_oracle_plan_v1",
        "recorded_at": "2026-08-09",
        "status": "registered_before_zero_update_analysis",
        "scientific_question": (
            "Does an exact frozen GPT base plus the production-seeded fixed 1% "
            "BlockFHT delta contain the dense V and attention-cproj path, local "
            "chords, discovery affine span, and exact Muon action in an "
            "out-of-sample functional metric?"
        ),
        "theory": {
            "decoder": "W(z)=W0_gpt+A_blockfht z, z0=0",
            "paper_link": (
                "The paper asserts smooth low-dimensional layer manifolds and "
                "uses fixed orthogonal maps. An affine chart is the first-order "
                "model of such a manifold, but it must match both image and tangent."
            ),
            "why_not_initialization_only": (
                "The MLP cproj affine-delta control passed exact initialization "
                "yet remained +0.2376 CE behind attention-only because its tangent "
                "was direction-limited. This gate therefore requires dense path, "
                "local chord, early affine-span, and exact-Muon recovery together."
            ),
            "why_disjoint_metrics": (
                "Coordinates are fitted on one frozen terminal-dense batch set and "
                "evaluated unchanged on another, preventing metric-batch overfit "
                "from masquerading as functional manifold coverage."
            ),
            "storage_caveat": (
                "The frozen dense base preserves trainable-coordinate compression "
                "but not fixed storage or materialized inference compression."
            ),
        },
        "identity": {
            "dense_config": (
                "examples/nanogpt/configs/"
                "pro6_mai_v3_124m_muon_5tpp_attention_trajectory_replay_lr24e4.json"
            ),
            "entrypoint": "examples.nanogpt.analyze_attention_affine_delta_path_oracle",
            "entrypoint_sha256": file_sha256(analyzer),
            "trajectory_directory": str(SCIENTIFIC / "parameter_trajectory"),
            "trajectory_file_count": 41,
            "trajectory_total_bytes": 4644344659,
            "trajectory_inventory_sha256": (
                "03c28add07ad1445d7f049eff56a7e178ae99eb47c4ca32594ddc787110d12e9"
            ),
            "probe_directory": str(SCIENTIFIC / "optimizer_probe"),
            "probe_run_identity_sha256": (
                "aea2a472e6e5b6013d6254ff2339211796bcedb441db955706ce01b978a961ac"
            ),
            "terminal_checkpoint": str(SCIENTIFIC / "ckpt.pt"),
            "terminal_checkpoint_sha256": (
                "522fe8333f2e445066cfdbca4bbe4491d2ffebd71b314464ecf8bdacd3be4b5b"
            ),
            "dataset_manifest_sha256": (
                "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
            ),
            "parent_activation_result": str(activation.relative_to(REPO_ROOT)),
            "parent_activation_result_sha256": file_sha256(activation),
            "mlp_affine_delta_control": str(mlp_affine.relative_to(REPO_ROOT)),
            "mlp_affine_delta_control_sha256": file_sha256(mlp_affine),
            "product_fht_residual_control": str(product.relative_to(REPO_ROOT)),
            "product_fht_residual_control_sha256": file_sha256(product),
            "output_directory_must_be_absent": str(OUTPUT),
        },
        "protocol": {
            "parameter_updates": 0,
            "decoder": "exact_base_plus_fixed_linear_delta",
            "latent_ratio": 0.01,
            "block_fht_layers": 2,
            "block_fht_seed": 1000,
            "cgls_iterations": 32,
            "span_relative_cutoff": 1e-7,
            "layers": list(range(12)),
            "trajectory_steps": trajectory_steps,
            "discovery_max_step": 1140,
            "heldout_min_step": 1200,
            "probe_steps": [0, 594, 1188, 1782, 2372],
            "heldout_probe_steps": [1782, 2372],
            "fit_metric_seed": 20260809,
            "eval_metric_seed": 20260810,
            "metric_batch_size": 2,
            "metric_batches": 2,
            "metric_block_size": 256,
            "metric_policy": (
                "Freeze terminal-dense attention sources separately on fitting "
                "and disjoint evaluation batches. V is measured after attention "
                "mixing and dense O; cproj is measured as H Delta O."
            ),
            "targets": {
                "v": {
                    "parameter": "attn.c_attn.weight",
                    "slice": "final n_embd rows",
                    "seed_stride": 8,
                    "seed_offset": 2,
                    "target_std": 0.02,
                },
                "cproj": {
                    "parameter": "attn.c_proj.weight",
                    "slice": "full matrix",
                    "seed_stride": 4,
                    "seed_offset": 1,
                    "target_std": 0.004082482904638631,
                },
            },
            "near_affine_diagnostics": [
                "discovery-coordinate PCA ranks at 90/95/99 percent energy",
                "linear extrapolation in normalized cumulative scheduled LR",
            ],
        },
        "decision_rule": {
            "thresholds": {
                "aggregate_recovery_minimum": 0.8,
                "minimum_layer_recovery_minimum": 0.6,
            },
            "pass": (
                "Both V and cproj must pass aggregate and every-layer held-out "
                "state, local-chord, discovery-affine-span, and exact-Muon gates. "
                "A pass authorizes implementation and one exact-config MFU "
                "preflight, not training."
            ),
            "fail": (
                "Keep the failed target dense and close the exact-base affine "
                "BlockFHT decoder at this budget; do not run MFU or training."
            ),
            "no_posthoc_threshold_changes": True,
        },
        "authorization": {
            "model_implementation": False,
            "mfu_preflight": False,
            "language_model_training": False,
            "larger_rung": False,
        },
        "execution": {
            "host": "PRO6",
            "device": "cuda:0",
            "command": command,
            "expected_duration": "under five minutes; zero-update GPU analysis",
            "direct_foreground_polling": True,
            "watchdog": False,
            "callbacks": False,
        },
    }


def main() -> None:
    output = REPO_ROOT / (
        "examples/nanogpt/configs/selection_artifacts/"
        "124m_attention_affine_delta_path_oracle_plan.json"
    )
    output.write_text(json.dumps(build_plan(), indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
