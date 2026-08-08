#!/usr/bin/env python3
"""Preregister the exact-base quadratic attention image oracle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "examples/nanogpt/analyze_attention_quadratic_image_oracle.py"
CONFIG = Path("examples/nanogpt/configs/pro6_mai_v3_124m_muon_5tpp_attention_trajectory_replay_lr24e4.json")
PARENT = Path("examples/nanogpt/configs/selection_artifacts/124m_attention_residual_cubic_oracle_result.json")
PROBE_PLAN = Path("examples/nanogpt/configs/selection_artifacts/124m_attention_paper_activation_oracle_plan.json")
OUTPUT = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_attention_quadratic_image_oracle_plan.json"
REMOTE = Path("/mnt/ssd-data/orj/MappingNetworks")
WORKTREE = REMOTE / "latent-weight-lab-attention-quadratic-image"
INITIAL_PROBE = REMOTE / "outputs/pro6_mai_v3_attention_dense_5tpp_replay/scientific/optimizer_probe/step_000000.pt"
TERMINAL_PROBE = REMOTE / "outputs/pro6_mai_v3_attention_dense_5tpp_replay/scientific/optimizer_probe/step_002372.pt"
CHECKPOINT = REMOTE / "outputs/pro6_mai_v3_attention_dense_5tpp_replay/scientific/ckpt.pt"
DATA = REMOTE / "data/finewebedu_20b"
OUT = REMOTE / "outputs/pro6_mai_v3_attention_quadratic_image_oracle_v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict:
    config = json.loads((ROOT / CONFIG).read_text())
    parent = json.loads((ROOT / PARENT).read_text())
    probe_plan = json.loads((ROOT / PROBE_PLAN).read_text())
    if parent["classification"] != "ATTENTION_RESIDUAL_CUBIC_ORACLE_REJECT_ALL":
        raise ValueError("residual cubic is not sealed rejected")
    n_layer = int(config["n_layer"])
    return {
        "schema_version": "mai_124m_attention_quadratic_image_oracle_plan_v1",
        "recorded_at": "2026-08-09",
        "status": "registered_before_zero_update_analysis",
        "scientific_question": "Can the exact-base three-map quadratic BlockFHT manifold represent the terminal dense V or attention-cproj functional state on disjoint batches before any tangent or training work?",
        "theory": {
            "decoder": "W(z)=W0+scale_init*N*[Az+center((Bz)*(Cz))/sigma_z]",
            "why_now": "Affine and all scalar row-wise warps fail state-image transport. The bilinear term is the smallest already-implemented fixed cross-coordinate map whose tangent rotates with z without a learned basis or extra trainable coordinates.",
            "why_terminal_first": "Terminal state membership is a necessary condition. Failure on an optimistic Gauss-Newton oracle closes the family without paying for tangent transport, implementation, MFU, or training.",
            "scale_choice": "q=1 is fixed by the implementation's analytic initialization-variance normalization; no q sweep is allowed. The earlier MLP q sweep is negative but not substituted for the distinct attention metric."
        },
        "identity": {
            "entrypoint_sha256": sha(ENTRY),
            "dense_config": str(CONFIG),
            "dense_config_sha256": sha(ROOT / CONFIG),
            "parent_result": str(PARENT),
            "parent_result_sha256": sha(ROOT / PARENT),
            "initial_probe": str(INITIAL_PROBE),
            "initial_probe_sha256": probe_plan["identity"]["probe_sha256"]["step_000000.pt"],
            "terminal_probe": str(TERMINAL_PROBE),
            "terminal_probe_sha256": probe_plan["identity"]["probe_sha256"]["step_002372.pt"],
            "probe_run_identity_sha256": probe_plan["identity"]["probe_run_identity_sha256"],
            "terminal_checkpoint_sha256": "522fe8333f2e445066cfdbca4bbe4491d2ffebd71b314464ecf8bdacd3be4b5b",
            "dataset_manifest_sha256": config["data_manifest_sha256"],
            "output_directory_must_be_absent": str(OUT)
        },
        "protocol": {
            "parameter_updates": 0,
            "terminal_step": 2372,
            "layers": list(range(n_layer)),
            "latent_ratio": 0.01,
            "block_fht_layers": 2,
            "block_fht_seed": 1000,
            "quadratic_scale": 1.0,
            "quadratic_seed_offset": 104729,
            "outer_gauss_newton_iterations": 8,
            "inner_cgls_iterations": 12,
            "line_search_multipliers": [1.0, 0.5, 0.25, 0.125, 0.0625],
            "metric_batch_size": 2,
            "metric_block_size": 256,
            "metric_batches": 2,
            "fit_metric_seed": 20260809,
            "eval_metric_seed": 20260810,
            "targets": {
                "v": {"parameter": "attn.c_attn.weight", "slice": "final n_embd rows", "seed_stride": 8, "seed_offset": 2, "target_std": 0.02},
                "cproj": {"parameter": "attn.c_proj.weight", "slice": "full matrix", "seed_stride": 4, "seed_offset": 1, "target_std": 0.02 / math.sqrt(2 * n_layer)}
            }
        },
        "decision_rule": {
            "thresholds": {"aggregate_eval_image_recovery_minimum": 0.80, "minimum_layer_eval_image_recovery": 0.60},
            "pass": "A target passes only if disjoint terminal image recovery is at least 0.8 aggregate and 0.6 in every layer. A pass authorizes a separate tangent/path oracle only.",
            "fail": "Close the exact-base three-map quadratic family for that target; no scale sweep, tangent gate, implementation, MFU, training, or larger rung.",
            "threshold_changed_after_measurement": False
        },
        "execution": {
            "host": "PRO6", "device": "cuda:0", "direct_foreground_polling": True, "watchdog": False, "callbacks": False,
            "expected_duration": "under five minutes; zero-update GPU analysis",
            "command": [str(REMOTE / ".venv/bin/python"), "-u", "-m", "examples.nanogpt.analyze_attention_quadratic_image_oracle", "--plan", str(WORKTREE / OUTPUT.relative_to(ROOT)), "--initial-probe", str(INITIAL_PROBE), "--terminal-probe", str(TERMINAL_PROBE), "--terminal-checkpoint", str(CHECKPOINT), "--data-dir", str(DATA), "--output-dir", str(OUT), "--device", "cuda"]
        },
        "authorization": {"tangent_oracle": False, "model_implementation": False, "mfu_preflight": False, "language_model_training": False, "larger_rung": False}
    }


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
