#!/usr/bin/env python3
"""Preregister the exact-base residual-cubic attention oracle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "examples/nanogpt/analyze_attention_residual_cubic_oracle.py"
DENSE_CONFIG = Path("examples/nanogpt/configs/pro6_mai_v3_124m_muon_5tpp_attention_trajectory_replay_lr24e4.json")
PARENT_RESULT = Path("examples/nanogpt/configs/selection_artifacts/124m_attention_mapping_loss_closure_result.json")
PROBE_IDENTITY_PLAN = Path("examples/nanogpt/configs/selection_artifacts/124m_attention_paper_activation_oracle_plan.json")
OUTPUT = REPO_ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_residual_cubic_oracle_plan.json"
REMOTE_ROOT = Path("/mnt/ssd-data/orj/MappingNetworks")
REMOTE_REPO = REMOTE_ROOT / "latent-weight-lab-attention-residual-cubic"
PROBE_DIR = REMOTE_ROOT / "outputs/pro6_mai_v3_attention_dense_5tpp_replay/scientific/optimizer_probe"
CHECKPOINT = REMOTE_ROOT / "outputs/pro6_mai_v3_attention_dense_5tpp_replay/scientific/ckpt.pt"
DATA_DIR = REMOTE_ROOT / "data/finewebedu_20b"
OUTPUT_DIR = REMOTE_ROOT / "outputs/pro6_mai_v3_attention_residual_cubic_oracle_v1"
CHECKPOINT_SHA256 = "522fe8333f2e445066cfdbca4bbe4491d2ffebd71b314464ecf8bdacd3be4b5b"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan() -> dict[str, Any]:
    config = json.loads((REPO_ROOT / DENSE_CONFIG).read_text())
    parent = json.loads((REPO_ROOT / PARENT_RESULT).read_text())
    probe_plan = json.loads((REPO_ROOT / PROBE_IDENTITY_PLAN).read_text())
    if parent["classification"] != "CLOSE_LITERAL_PAPER_MAPPING_LOSS_FOR_FULL_ATTENTION":
        raise ValueError("literal Mapping Loss branch is not sealed")
    n_layer = int(config["n_layer"])
    return {
        "schema_version": "mai_124m_attention_residual_cubic_oracle_plan_v1",
        "recorded_at": "2026-08-09",
        "status": "registered_before_zero_update_analysis",
        "scientific_question": "Can the minimal exact-base unbounded residual cubic give the fixed 1% BlockFHT chart enough state-dependent orientation to represent held-out dense V or attention-cproj states and exact Muon action on disjoint batches?",
        "theory": {
            "decoder": "W(z)=W0+h_s(Az), h_s(x)=x+x^3/(3s^2)",
            "jacobian": "diag(1+(Az/s)^2) A",
            "why_this_family": "It is the lowest-order odd smooth warp with exact base, identity initialization tangent, unbounded image, and a state-dependent diagonal tangent. It isolates curvature without tanh saturation, learned bases, dense residuals, or extra trainable coordinates.",
            "scale_derivation": "For each target/layer, freeze s to max(abs(W_t-W0)) over discovery steps 0/594/1188. Because inverse h_s has |x|<=|delta|, the discovery Jacobian diagonal is at most 2; the preregistered all-state ceiling remains 10.",
            "disjoint_metric_requirement": "Tangent coordinates are fitted on seed 20260809 and scored unchanged on seed 20260810, preventing the same-batch tangent overfit exposed by the affine audit.",
            "prior_boundary": "The prior MLP quadratic-FHT arms parameterized coordinate products and the paper tanh acted on the full weight. Neither tested this exact-base unbounded output warp for attention V/O."
        },
        "identity": {
            "entrypoint": "examples.nanogpt.analyze_attention_residual_cubic_oracle",
            "entrypoint_sha256": file_sha256(ENTRYPOINT),
            "dense_config": str(DENSE_CONFIG),
            "dense_config_sha256": file_sha256(REPO_ROOT / DENSE_CONFIG),
            "parent_result": str(PARENT_RESULT),
            "parent_result_sha256": file_sha256(REPO_ROOT / PARENT_RESULT),
            "probe_directory": str(PROBE_DIR),
            "probe_run_identity_sha256": probe_plan["identity"]["probe_run_identity_sha256"],
            "probe_sha256": probe_plan["identity"]["probe_sha256"],
            "terminal_checkpoint": str(CHECKPOINT),
            "terminal_checkpoint_sha256": CHECKPOINT_SHA256,
            "dataset_manifest_sha256": config["data_manifest_sha256"],
            "output_directory_must_be_absent": str(OUTPUT_DIR)
        },
        "protocol": {
            "parameter_updates": 0,
            "layers": list(range(n_layer)),
            "steps": [0, 594, 1188, 1782, 2372],
            "discovery_steps": [0, 594, 1188],
            "heldout_steps": [1782, 2372],
            "direction": "exact dense Muon applied_direction_per_lr",
            "weight": "same-probe weight_before_step",
            "latent_ratio": 0.01,
            "block_fht_layers": 2,
            "block_fht_seed": 1000,
            "decoder": "exact_base_residual_cubic",
            "scale_calibration": "per_target_layer_max_abs_dense_delta_over_discovery_steps",
            "minimum_scale": 1e-8,
            "inverse_iterations": 16,
            "cgls_iterations": 32,
            "metric_batch_size": 2,
            "metric_block_size": 256,
            "metric_batches": 2,
            "fit_metric_seed": 20260809,
            "eval_metric_seed": 20260810,
            "metric_policy": "Freeze terminal-dense attention sources on disjoint fit/eval batches. V is measured after attention mixing and dense O; cproj is measured as H Delta O.",
            "targets": {
                "v": {"parameter": "attn.c_attn.weight", "slice": "final n_embd rows", "seed_stride": 8, "seed_offset": 2, "target_std": 0.02},
                "cproj": {"parameter": "attn.c_proj.weight", "slice": "full matrix", "seed_stride": 4, "seed_offset": 1, "target_std": 0.02 / math.sqrt(2 * n_layer)}
            },
            "controls": ["same 1% affine identity BlockFHT tangent", "fit-metric versus disjoint-eval tangent recovery", "all-state Jacobian diagonal ceiling"]
        },
        "decision_rule": {
            "thresholds": {
                "maximum_jacobian_diagonal": 10.0,
                "eval_functional_image_recovery_minimum": 0.80,
                "eval_cubic_tangent_recovery_minimum": 0.80,
                "eval_cubic_gain_over_identity_minimum": 0.05
            },
            "pass": "A target must pass condition, held-out disjoint image, held-out disjoint tangent, and gain-over-identical-affine gates. A pass authorizes only a separate cross-step coordinate-transport oracle.",
            "fail": "Close residual-cubic output warping for the failed target at 1% and launch no implementation, MFU, training, or larger rung.",
            "threshold_changed_after_measurement": False
        },
        "execution": {
            "host": "PRO6",
            "device": "cuda:0",
            "direct_foreground_polling": True,
            "watchdog": False,
            "callbacks": False,
            "expected_duration": "under five minutes; zero-update GPU analysis",
            "command": [
                str(REMOTE_ROOT / ".venv/bin/python"), "-u", "-m", "examples.nanogpt.analyze_attention_residual_cubic_oracle",
                "--plan", str(REMOTE_REPO / OUTPUT.relative_to(REPO_ROOT)),
                "--probe-dir", str(PROBE_DIR),
                "--terminal-checkpoint", str(CHECKPOINT),
                "--data-dir", str(DATA_DIR),
                "--output-dir", str(OUTPUT_DIR),
                "--device", "cuda"
            ]
        },
        "authorization": {"model_implementation": False, "mfu_preflight": False, "language_model_training": False, "larger_rung": False}
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(build_plan(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
