#!/usr/bin/env python3
"""Build the preregistered 124M/5TPP bounded c_proj metric config."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONTROL = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_cprojdecay0p5_5tpp_lr24e4.json"
PLAN = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_cproj_activation_energy_metric_5tpp_plan.json"
CPROJ_RESULT = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_cproj_only_5tpp_result.json"
OUTPUT = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_cprojdecay0p5_activationenergymetric_5tpp_lr24e4.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(CONTROL.read_text())
    implementation_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    config.update(
        {
            "candidate_scope": (
                "single-factor bounded diagonal post-GELU activation-energy "
                "metric for the existing c_proj hidden64+24 fresh Givens "
                "projection; repaired attention, dense c_fc, error-feedback "
                "decay 0.5, optimizer, schedule, data, and seeds remain fixed"
            ),
            "hpo_stage": "repaired_attention_cproj_activation_energy_metric_124m_5tpp",
            "ladder_interpretation": (
                "one preregistered same-model same-token causal metric test; "
                "not a fitted scaling ladder and not a hyperparameter sweep"
            ),
            "ladder_role": "mlp_cproj_activation_energy_metric_repaired_attention_5tpp",
            "ladder_slot": "124m_5tpp_cproj_activation_energy_metric_v1",
            "out_dir": (
                "/home/pro6000-9980x/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_repairedattn_cproj_activationenergymetric_5tpp/scientific"
            ),
            "practical_equivalence_policy": (
                "promote only if terminal fixed-window validation CE <=3.6478 "
                "with finite bounded metric diagnostics and no material "
                "performance regression; 3.6378 is a nonbinding near-parent "
                "diagnostic; thresholds never change after observation"
            ),
            "selection_endpoint": (
                "terminal step-2373 held-out CE on the shared fixed windows, "
                "retaining steps 594/1188/1782/2373 for direct c_proj-only, "
                "repaired-attention, and dense comparisons"
            ),
            "block_fht_mlp_cproj_activation_energy_metric": True,
            "block_fht_mlp_cproj_activation_energy_metric_decay": 0.95,
            "block_fht_mlp_cproj_activation_energy_metric_minimum": 0.25,
            "block_fht_mlp_cproj_activation_energy_metric_maximum": 4.0,
            "block_fht_mlp_cproj_activation_energy_metric_epsilon": 1e-6,
            "mfu_preflight_certificate": (
                "/home/pro6000-9980x/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_repairedattn_cproj_activationenergymetric_5tpp/"
                "performance_preflight.json"
            ),
            "mfu_measurement_protocol": (
                "foreground exact scientific config, one warmup plus eight "
                "timed real training updates on PRO6 GPU0; includes online "
                "post-GELU energy accumulation, bounded EMA metric, native "
                "weighted fresh hidden64+24 matching/refit, error feedback, "
                "repaired attention, dense c_fc, and materialization"
            ),
            "monitoring_policy": (
                "MFU preflight is foreground-polled; the admitted 2373-update "
                "run uses one idempotent watchdog with 20/50/100 callbacks to "
                "send-opencode-test mentioning @Codex, a 90-minute heartbeat "
                "reset by progress, terminal ownership persistence, and no "
                "duplicate terminal callback"
            ),
            "registered_plan": str(PLAN.relative_to(REPO)),
            "registered_plan_sha256": sha256(PLAN),
            "cproj_control_result": str(CPROJ_RESULT.relative_to(REPO)),
            "cproj_control_result_sha256": sha256(CPROJ_RESULT),
            "component_attribution_controls": {
                "repaired_attention_terminal_validation_ce": 3.6278,
                "cproj_only_terminal_validation_ce_exact": 3.6562013626098633,
                "dense_terminal_validation_ce": 3.5401,
                "promote_terminal_validation_ce_maximum": 3.6478,
                "nonbinding_near_parent_validation_ce_maximum": 3.6378,
            },
            "additional_trainable_parameters_vs_cproj_control": 0,
            "additional_inference_parameters_vs_cproj_control": 0,
            "additional_inference_flops_vs_cproj_control": 0,
            "additional_optimizer_state_bytes_vs_cproj_control": 12 * 3072 * 4 + 12 * 8,
            "implementation_commit": implementation_commit,
        }
    )
    source_paths = (
        "examples/nanogpt/model.py",
        "examples/nanogpt/muon.py",
        "examples/nanogpt/muon_matched_givens.py",
        "examples/nanogpt/train.py",
        "examples/nanogpt/mfu_preflight.py",
        "latent_weight_lab/block_fht.py",
        "examples/nanogpt/test_muon_matched_givens.py",
        "examples/nanogpt/test_124m_repaired_attention_cproj_activation_energy_metric_5tpp_plan.py",
    )
    config["implementation_source_hashes"] = {
        path: sha256(REPO / path) for path in source_paths
    }
    OUTPUT.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    print(sha256(OUTPUT))


if __name__ == "__main__":
    main()
