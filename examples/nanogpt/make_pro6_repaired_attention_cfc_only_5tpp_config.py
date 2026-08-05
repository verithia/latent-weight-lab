#!/usr/bin/env python3
"""Build the preregistered repaired-attention plus c_fc-only 5TPP config."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
JOINT = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_cfcdecay1_cprojdecay0p5_5tpp_lr24e4_v2.json"
PLAN = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_cfc_only_5tpp_plan.json"
ACTIVATION_RESULT = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_cproj_activation_energy_metric_5tpp_result.json"
CPROJ_RESULT = REPO / "examples/nanogpt/configs/selection_artifacts/124m_repaired_attention_cproj_only_5tpp_result.json"
OUTPUT = REPO / "examples/nanogpt/configs/pro6_mai_v3_124m_repairedfullattn_plus_cfconly_decay1_5tpp_lr24e4.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(JOINT.read_text())
    for key in tuple(config):
        if key.startswith("block_fht_mlp_cproj_"):
            del config[key]
    config["block_fht_targets"] = [
        target for target in config["block_fht_targets"]
        if target != "mlp.c_proj"
    ]
    implementation_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    config.update(
        {
            "candidate_scope": (
                "missing 124M/5TPP factorial arm: preserve repaired attention "
                "and the accepted c_fc six-cell directed-product decay-1 "
                "error-feedback chart exactly while restoring c_proj to dense "
                "Muon; no c_proj mapping, metric, gain, or output transform"
            ),
            "hpo_stage": "repaired_attention_cfc_only_component_attribution_124m_5tpp",
            "ladder_interpretation": (
                "one preregistered same-model same-token c_fc-only factorial "
                "attribution; not a fitted scaling ladder or hyperparameter sweep"
            ),
            "ladder_role": "mlp_cfc_only_repaired_attention_component_attribution_5tpp",
            "ladder_slot": "124m_5tpp_cfc_decay1_component_attribution_v1",
            "out_dir": (
                "/home/pro6000-9980x/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_repairedattn_cfconly_decay1_5tpp/scientific"
            ),
            "mfu_preflight_certificate": (
                "/home/pro6000-9980x/MappingNetworks/outputs/"
                "pro6_mai_v3_124m_repairedattn_cfconly_decay1_5tpp/"
                "performance_preflight.json"
            ),
            "mfu_measurement_protocol": (
                "foreground exact scientific config, one warmup plus eight "
                "timed real updates on PRO6 GPU0; includes native BlockFHT, "
                "repaired attention Cayley, c_fc directed-product selection, "
                "refit and decay-1 error feedback, dense Muon c_proj, and materialization"
            ),
            "monitoring_policy": (
                "MFU preflight is foreground-polled; the admitted 2373-update "
                "run uses one idempotent watcher with 20/50/100 callbacks to "
                "send-opencode-test mentioning @Codex and a 90-minute "
                "progress-reset heartbeat with terminal delivery once"
            ),
            "practical_equivalence_policy": (
                "terminal fixed-window validation CE <=3.6478 passes the "
                "frozen +0.0200 attention-relative gate; 3.6378 is a "
                "nonbinding near-parent diagnostic; thresholds never change"
            ),
            "selection_endpoint": (
                "terminal step-2373 held-out CE on the shared fixed windows, "
                "retaining steps 594/1188/1782/2373 for factorial comparison"
            ),
            "registered_plan": str(PLAN.relative_to(REPO)),
            "registered_plan_sha256": sha256(PLAN),
            "activation_metric_terminal_result": str(
                ACTIVATION_RESULT.relative_to(REPO)
            ),
            "activation_metric_terminal_result_sha256": sha256(ACTIVATION_RESULT),
            "cproj_component_result": str(CPROJ_RESULT.relative_to(REPO)),
            "cproj_component_result_sha256": sha256(CPROJ_RESULT),
            "additional_dense_optimizer_state_bytes": 12 * 768 * 3072 * 4,
            "additional_trainable_parameters_vs_attention_parent": 0,
            "additional_inference_parameters_vs_materialized_attention_parent": 0,
            "additional_inference_flops_vs_materialized_attention_parent": 0,
            "control_only_runtime_guard": (
                "directed-product c_fc is admitted with c_proj absent from "
                "block_fht_targets; the existing guard still rejects every "
                "unqualified generated c_proj"
            ),
            "implementation_commit": implementation_commit,
        }
    )
    for key in (
        "supersedes_config",
        "supersedes_config_sha256",
        "pretraining_validation_repair",
    ):
        config.pop(key, None)
    source_paths = (
        "examples/nanogpt/model.py",
        "examples/nanogpt/muon.py",
        "examples/nanogpt/muon_matched_givens.py",
        "examples/nanogpt/train.py",
        "examples/nanogpt/mfu_preflight.py",
        "latent_weight_lab/block_fht.py",
        "examples/nanogpt/test_muon_directed_product.py",
    )
    config["implementation_source_hashes"] = {
        path: sha256(REPO / path) for path in source_paths
    }
    OUTPUT.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    print(sha256(OUTPUT))


if __name__ == "__main__":
    main()
