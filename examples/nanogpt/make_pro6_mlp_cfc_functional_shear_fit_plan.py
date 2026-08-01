#!/usr/bin/env python3
"""Generate the preregistered PRO6 c_fc functional-shear fit plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
ENTRYPOINT = ROOT / "examples/nanogpt/analyze_mlp_cfc_functional_shear_fit.py"
CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_functional_shear_fit_pro6_plan.json"
CHECKPOINT = WORKSPACE / "outputs/pro6_mai_v3_mlp_hidden88_replay/pro6_mai_v3_124m_twopassfresh88_replay1/ckpt.pt"
OUTPUT = WORKSPACE / "outputs/pro6_mai_v3_mlp_cfc_functional_shear_fit1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    remote = WORKSPACE / "latent-weight-lab-hidden88-replay"
    python = WORKSPACE / ".venv/bin/python"
    parent = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_task_shear_ce_pro6_result.json"
    payload = {
        "schema_version": "mai_124m_mlp_cfc_functional_shear_fit_pro6_plan_v1",
        "recorded_at": "2026-08-02",
        "question": "Can the same equal-coordinate shear24 residual become materially more task-effective when pair topology and coordinates are fitted in the observed post-GELU/c_proj function metric rather than weight Frobenius geometry?",
        "authorization": {
            "parameter_updates": 0,
            "single_directly_polled_functional_diagnostic": True,
            "finite_ce": False,
            "production_implementation": False,
            "mfu_preflight": False,
            "scientific_training": False,
            "larger_rung": False,
        },
        "identity": {
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": "0586c08fae79854d35ba765b822ae56c25efdd534df25b52797be0e8517fb075",
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": "55340bb5c035300fba9fb23b11ddf345b6bd879e3f6996c6e4e993952e01cf59",
            "dataset_manifest": str(WORKSPACE / "data/finewebedu_20b/manifest.json"),
            "dataset_manifest_sha256": "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b",
            "entrypoint": str(ENTRYPOINT.relative_to(ROOT)),
            "entrypoint_sha256": sha256_file(ENTRYPOINT),
            "parent_task_shear_ce_result_sha256": sha256_file(parent),
        },
        "fixed_protocol": {
            "layers": list(range(12)),
            "batch_size": 4,
            "block_size": 1024,
            "batches_per_window": 4,
            "fit_train_seed": 20260820,
            "holdout_train_seed": 20261202,
            "functional_sample_cap": 2048,
            "functional_sample_seed": 20261220,
            "matching_seed": 20260820,
            "matching_neighbors": 64,
            "parent_rotational_stages": 64,
            "residual_shear_stages": 24,
            "equal_coordinates_per_layer": 135168,
            "attribution_grid": [
                "weight topology + weight fit",
                "functional topology + weight fit",
                "weight topology + functional fit",
                "functional topology + functional fit"
            ],
            "functional_metric": "exact first-order MLP-output displacement under observed input activations, exact GELU slopes, and fixed c_proj; fit and holdout use disjoint train-token windows",
            "finite_map": "exact determinant-one exp(s*[[0,1],[1,0]]) per selected pair",
            "evaluation_dtype": "float32",
        },
        "decision_rule": {
            "minimum_functional_ratio": 1.10,
            "minimum_ce_descent_ratio": 1.00,
            "minimum_weight_ratio": 0.90,
            "maximum_determinant_error": 0.000002,
            "maximum_condition_number": 1.01,
            "selection": "Functional topology plus functional coordinate fitting must beat both weight-shear24 and fresh88 MLP-output recovery by at least 1.10x on fit and independent holdout train windows, not reduce predicted CE descent relative to weight-shear24 on either window, and retain at least 0.90x of weight recovery.",
            "threshold_change_after_observation": False,
        },
        "execution": {
            "host": "PRO6",
            "gpu": 0,
            "foreground_direct_polling": True,
            "watchdog": False,
            "callback": False,
            "queue_worker": False,
            "heartbeat": False,
            "command": [
                str(python),
                "-u",
                "-m",
                "examples.nanogpt.analyze_mlp_cfc_functional_shear_fit",
                "--checkpoint",
                str(CHECKPOINT),
                "--config",
                str(remote / CONFIG.relative_to(ROOT)),
                "--data-dir",
                str(WORKSPACE / "data/finewebedu_20b"),
                "--plan",
                str(remote / PLAN.relative_to(ROOT)),
                "--output",
                str(OUTPUT),
                "--device",
                "cuda",
                "--native-cache",
                str(WORKSPACE / "native_cache"),
            ],
        },
        "limitations": [
            "This is a zero-update linearized function-space fit/holdout diagnostic, not finite CE or language-model training.",
            "The functional metric uses observed train activations only; no validation CE window from the preceding gate is reused for selection or fitting.",
            "The earlier c_proj activation-weighted inverse-metric and activation-correlation rejections remain valid. This test differs by fitting the newly identified c_fc symmetric-shear residual directly in MLP-output space rather than reweighting an existing c_proj chart or using raw activation correlation.",
            "Passing authorizes only a separately preregistered held-out finite-CE diagnostic, not production integration, MFU testing, training, or a larger rung.",
        ],
    }
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode() + b"\n"
    PLAN.write_bytes(encoded)
    print(
        json.dumps(
            {
                "plan": str(PLAN.relative_to(ROOT)),
                "plan_sha256": hashlib.sha256(encoded).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
