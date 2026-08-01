#!/usr/bin/env python3
"""Generate the preregistered PRO6 equal-coordinate c_fc task-shear plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
ENTRYPOINT = ROOT / "examples/nanogpt/analyze_mlp_cfc_task_shear_fit.py"
CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_task_shear_fit_pro6_plan.json"
CHECKPOINT = WORKSPACE / "outputs/pro6_mai_v3_mlp_hidden88_replay/pro6_mai_v3_124m_twopassfresh88_replay1/ckpt.pt"
OUTPUT = WORKSPACE / "outputs/pro6_mai_v3_mlp_cfc_task_shear_fit1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    replay = WORKSPACE / "latent-weight-lab-hidden88-replay"
    python = WORKSPACE / ".venv/bin/python"
    parent = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_conjugated_general_pro6_result.json"
    payload = {
        "schema_version": "mai_124m_mlp_cfc_task_shear_fit_pro6_plan_v1",
        "recorded_at": "2026-08-01",
        "question": "At the exact fresh88 coordinate budget, can task-selected c_fc output pairs use symmetric shear more efficiently than additional rotation stages?",
        "authorization": {
            "parameter_updates": 0,
            "single_directly_polled_fit_diagnostic": True,
            "heldout_ce": False,
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
            "parent_conjugated_result_sha256": sha256_file(parent),
            "scientific_source_base_commit": "c2cb932a29b2aafca58315d487937b36b8296e15",
        },
        "fixed_protocol": {
            "layers": list(range(12)),
            "batch_size": 4,
            "block_size": 1024,
            "fit_batches": 4,
            "fit_train_seed": 20260820,
            "matching_seed": 20260820,
            "matching_neighbors": 64,
            "equal_coordinates_per_layer": 135168,
            "control": "fresh64 rotational stages + fresh residual24 rotational stages",
            "candidate_fresh64_shear24": "64*1536 skew coordinates + 24*1536 symmetric-shear coordinates = 135168",
            "candidate_fresh64_skew_shear12": "64*1536 skew coordinates + 12*1536*2 skew-plus-shear coordinates = 135168",
            "candidate_fresh48_skew_shear20": "48*1536 skew coordinates + 20*1536*2 skew-plus-shear coordinates = 135168",
            "pair_map": "exact exp(s*[[0,1],[1,0]] + a*[[0,-1],[1,0]]) with determinant one",
            "fit": "causal local least squares with fresh task-matched pair selection at each parent/residual chart; decoupled weight decay applied exactly once after the fitted map",
            "evaluation_dtype": "float32",
        },
        "decision_rule": {
            "minimum_layer_delta": 0.0,
            "minimum_aggregate_ratio": 1.05,
            "maximum_determinant_error": 0.000002,
            "maximum_condition_number": 1.01,
            "selection": "A candidate advances to a separately preregistered held-out CE gate only if it is no worse than fresh88 in every layer and improves aggregate exact-current dense-update recovery by at least 5%. Choose by the best minimum layer delta, then aggregate recovery.",
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
                "examples.nanogpt.analyze_mlp_cfc_task_shear_fit",
                "--checkpoint",
                str(CHECKPOINT),
                "--config",
                str(replay / CONFIG.relative_to(ROOT)),
                "--data-dir",
                str(WORKSPACE / "data/finewebedu_20b"),
                "--plan",
                str(replay / PLAN.relative_to(ROOT)),
                "--output",
                str(OUTPUT),
                "--device",
                "cuda",
                "--native-cache",
                str(WORKSPACE / "native_cache"),
            ],
        },
        "limitations": [
            "This is a zero-update fit-only local capacity screen, not language-model training.",
            "The coordinates are oracle-fitted to one exact-current Muon update; no held-out CE is measured in this gate.",
            "The earlier c_proj SL2 rejection remains valid. This c_fc test is justified only because the new one-sided c_fc attribution shows a different 99% skew-plus-shear residual mechanism and negligible radial value.",
            "Passing authorizes only a separately preregistered held-out CE diagnostic, not production integration, MFU testing, training, or a larger rung.",
            "No learned basis, LoRA, dense residual adapter, HyperConnection, optimizer update, or model update is introduced.",
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
