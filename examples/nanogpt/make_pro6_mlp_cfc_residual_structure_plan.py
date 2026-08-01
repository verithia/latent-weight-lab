#!/usr/bin/env python3
"""Generate the preregistered PRO6 c_fc residual-attribution plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
ENTRYPOINT = ROOT / "examples/nanogpt/analyze_mlp_cfc_residual_structure.py"
CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_residual_structure_pro6_plan.json"
CHECKPOINT = WORKSPACE / "outputs/pro6_mai_v3_mlp_hidden88_replay/pro6_mai_v3_124m_twopassfresh88_replay1/ckpt.pt"
OUTPUT = WORKSPACE / "outputs/pro6_mai_v3_mlp_cfc_residual_structure1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    replay = WORKSPACE / "latent-weight-lab-hidden88-replay"
    python = WORKSPACE / ".venv/bin/python"
    payload = {
        "schema_version": "mai_124m_mlp_cfc_residual_structure_pro6_plan_v1",
        "recorded_at": "2026-08-01",
        "status": "registered_before_zero_update_diagnostic",
        "question": "Which fixed structural family captures the held-out finite-CE value in dense_exact minus fresh88: input-channel diagonal, expansion-channel diagonal, bilateral diagonal, or low-rank spectral residual?",
        "authorization": {
            "parameter_updates": 0,
            "single_directly_polled_diagnostic": True,
            "production_implementation": False,
            "mfu_preflight": False,
            "scientific_training": False,
            "larger_rung": False
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
            "parent_trust_result_sha256": "299098e6c1f08d07bb7c40a2a84099720b25192399420f7576f2f11e439fefa2",
            "scientific_source_base_commit": "c2cb932a29b2aafca58315d487937b36b8296e15"
        },
        "fixed_protocol": {
            "layers": list(range(12)),
            "batch_size": 4,
            "block_size": 1024,
            "fit_batches": 4,
            "fit_train_seed": 20260820,
            "matching_seed": 20260820,
            "matching_neighbors": 64,
            "validation_seeds": [20260821, 20260822, 20260823, 20260824],
            "validation_batches_per_window": 8,
            "evaluation_repeats": 3,
            "bilateral_fit_iterations": 32,
            "low_rank_bracket": [1, 4, 16, 64],
            "evaluation_dtype": "float32"
        },
        "decision_rule": {
            "maximum_replicate_range": 0.0000002,
            "minimum_gap_recovery_every_window": 0.5,
            "minimum_median_gap_recovery": 0.8,
            "positive_control": "dense_exact must beat fresh88 on all four windows; otherwise residual attribution is rejected.",
            "qualification": "With the dense positive control satisfied, candidate must beat fresh88 and baseline on all four windows, recover at least 50% of the dense-over-fresh CE gap on every window, and recover at least 80% at the median window.",
            "selection": "Select the qualifying candidate with fewest coordinates per layer; ties use fixed family priority input, expansion, bilateral, spectral.",
            "on_none": "Report the highest worst-window recovery but authorize no structure.",
            "threshold_change_after_observation": False
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
                str(python), "-u", "-m", "examples.nanogpt.analyze_mlp_cfc_residual_structure",
                "--checkpoint", str(CHECKPOINT),
                "--config", str(replay / CONFIG.relative_to(ROOT)),
                "--data-dir", str(WORKSPACE / "data/finewebedu_20b"),
                "--plan", str(replay / PLAN.relative_to(ROOT)),
                "--output", str(OUTPUT),
                "--device", "cuda",
                "--native-cache", str(WORKSPACE / "native_cache")
            ]
        },
        "limitations": [
            "This is a no-update local residual attribution, not a trained architecture result.",
            "All residual structures are fit only to the exact-current fit-window direction and evaluated on new validation windows.",
            "Diagonal candidates are additive tangent projections; low-rank candidates use the best truncated SVD of the fit residual.",
            "No result directly authorizes training; the selected family requires a separate production design and MFU gate."
        ]
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    PLAN.write_bytes(encoded)
    print(json.dumps({"plan": str(PLAN.relative_to(ROOT)), "plan_sha256": hashlib.sha256(encoded).hexdigest()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
