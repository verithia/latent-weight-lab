#!/usr/bin/env python3
"""Preregister the attention paper-activation exact-Muon capacity oracle."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from examples.nanogpt.analyze_attention_paper_activation_oracle import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "examples/nanogpt/analyze_attention_paper_activation_oracle.py"
DENSE_CONFIG = Path(
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_muon_5tpp_attention_trajectory_replay_lr24e4.json"
)
PARENT_RESULT = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_muon_layer_attribution_result.json"
)
PARENT_PLAN = Path(
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_muon_layer_attribution_plan.json"
)
OUTPUT = REPO_ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_attention_paper_activation_oracle_plan.json"
)
REMOTE_ROOT = Path("/mnt/ssd-data/orj/MappingNetworks")
REMOTE_REPO = REMOTE_ROOT / "latent-weight-lab-attention-paper-activation"
PROBE_DIR = REMOTE_ROOT / (
    "outputs/pro6_mai_v3_attention_dense_5tpp_replay/scientific/optimizer_probe"
)
CHECKPOINT = REMOTE_ROOT / (
    "outputs/pro6_mai_v3_attention_dense_5tpp_replay/scientific/ckpt.pt"
)
DATA_DIR = REMOTE_ROOT / "data/finewebedu_20b"
OUTPUT_DIR = REMOTE_ROOT / "outputs/pro6_mai_v3_attention_paper_activation_oracle_v1"
CHECKPOINT_SHA256 = "522fe8333f2e445066cfdbca4bbe4491d2ffebd71b314464ecf8bdacd3be4b5b"


def build_plan() -> dict[str, Any]:
    parent_plan = json.loads((REPO_ROOT / PARENT_PLAN).read_text())
    parent_result = json.loads((REPO_ROOT / PARENT_RESULT).read_text())
    config = json.loads((REPO_ROOT / DENSE_CONFIG).read_text())
    if parent_result["classification"] != "REJECT_ATTENTION_LAYERWISE_ORBIT_ATTRIBUTION":
        raise ValueError("parent attention attribution is not sealed as rejected")
    probes = parent_plan["identity"]["probe_sha256"]
    n_layer = int(config["n_layer"])
    plan = {
        "schema_version": "mai_124m_attention_paper_activation_oracle_plan_v1",
        "recorded_at": "2026-08-09",
        "status": "registered_before_zero_update_analysis",
        "scientific_question": (
            "Can the paper's state-dependent elementwise activation rotate a fixed 1% "
            "BlockFHT chart into the dense V or attention-c_proj manifold strongly enough "
            "to recover held-out exact Muon action in attention-output geometry?"
        ),
        "theory": {
            "decoder": "g(z)=s*tanh((b+A z)/s), b=s*atanh(W0/s)",
            "jacobian": "J_g(z)=diag(1-(g(z)/s)^2) A",
            "scale_derivation": (
                "s=sqrt(10/9)*max(abs(W0)) is uniquely fixed by a step-zero "
                "derivative floor 0.1, hence Jacobian condition ceiling 10"
            ),
            "why_this_is_new": (
                "The rejected Cayley/LWT audit used a state-independent orbit tangent. "
                "This is the paper's algebraically distinct D_sigma(W) mechanism, whose "
                "diagonal gate changes with the represented state."
            ),
            "why_an_oracle_first": (
                "Oracle state coordinates and oracle tangent coefficients make this an "
                "optimistic representability upper bound. Failure excludes optimizer, "
                "Mapping-Loss, and learning-rate explanations."
            ),
            "prior_boundary": (
                "The same activation family failed for MLP c_proj, but V/O has a distinct "
                "softmax-conditioned functional metric and must be tested rather than inferred."
            ),
        },
        "identity": {
            "entrypoint": "examples.nanogpt.analyze_attention_paper_activation_oracle",
            "entrypoint_sha256": file_sha256(ENTRYPOINT),
            "dense_config": str(DENSE_CONFIG),
            "dense_config_sha256": file_sha256(REPO_ROOT / DENSE_CONFIG),
            "parent_result": str(PARENT_RESULT),
            "parent_result_sha256": file_sha256(REPO_ROOT / PARENT_RESULT),
            "parent_plan": str(PARENT_PLAN),
            "parent_plan_sha256": file_sha256(REPO_ROOT / PARENT_PLAN),
            "probe_directory": str(PROBE_DIR),
            "probe_run_identity_sha256": parent_plan["identity"][
                "probe_run_identity_sha256"
            ],
            "probe_sha256": probes,
            "terminal_checkpoint": str(CHECKPOINT),
            "terminal_checkpoint_sha256": CHECKPOINT_SHA256,
            "dataset_manifest_sha256": config["data_manifest_sha256"],
            "output_directory_must_be_absent": str(OUTPUT_DIR),
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
            "activation": "signed_condition_bounded_tanh",
            "activation_scale_multiplier": math.sqrt(10.0 / 9.0),
            "minimum_step0_activation_derivative": 0.1,
            "maximum_step0_jacobian_condition": 10.0,
            "coordinate_fit": "oracle_cgls_in_inverse_activation_preactivation",
            "tangent_fit": "oracle_cgls_in_frozen_terminal_attention_output_metric",
            "cgls_iterations": 32,
            "metric_batch_size": 2,
            "metric_block_size": 256,
            "metric_batches": 2,
            "metric_seed": 20260809,
            "metric_policy": (
                "Freeze terminal-dense ln_1 inputs, causal attention probabilities, "
                "concatenated head states, and dense attention c_proj. For V measure "
                "Delta residual = O * concat_h(A_h X Delta V_h); for c_proj measure "
                "Delta residual = H Delta O."
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
                    "target_std": 0.02 / math.sqrt(2 * n_layer),
                },
            },
            "controls": [
                "identical-budget identity BlockFHT tangent",
                "range validity against the frozen step-zero activation scale",
            ],
        },
        "decision_rule": {
            "thresholds": {
                "functional_image_recovery_minimum": 0.80,
                "activated_tangent_recovery_minimum": 0.80,
                "activation_gain_over_identity_minimum": 0.05,
            },
            "pass": (
                "A target passes only if every state stays inside the frozen range and "
                "all three held-out thresholds pass. A pass authorizes one separate "
                "causal-coordinate transport analysis, not implementation or training."
            ),
            "fail": (
                "Close the paper D_sigma activation family for that attention target at "
                "the tested 1% budget; keep it dense and launch no MFU or training arm."
            ),
            "threshold_changed_after_measurement": False,
        },
        "execution": {
            "host": "PRO6",
            "device": "cuda:0",
            "direct_foreground_polling": True,
            "watchdog": False,
            "callbacks": False,
            "expected_duration": "under five minutes; zero-update GPU analysis",
            "command": [
                str(REMOTE_ROOT / ".venv/bin/python"),
                "-u",
                "-m",
                "examples.nanogpt.analyze_attention_paper_activation_oracle",
                "--plan",
                str(REMOTE_REPO / OUTPUT.relative_to(REPO_ROOT)),
                "--probe-dir",
                str(PROBE_DIR),
                "--terminal-checkpoint",
                str(CHECKPOINT),
                "--data-dir",
                str(DATA_DIR),
                "--output-dir",
                str(OUTPUT_DIR),
                "--device",
                "cuda",
            ],
        },
        "authorization": {
            "model_implementation": False,
            "mfu_preflight": False,
            "language_model_training": False,
            "larger_rung": False,
        },
    }
    return plan


def main() -> None:
    plan = build_plan()
    OUTPUT.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
