#!/usr/bin/env python3
"""Generate the preregistered PRO6 c_fc layer-local attribution plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
ENTRYPOINT = ROOT / "examples/nanogpt/analyze_mlp_cfc_layer_local_residual.py"
CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_layer_local_residual_pro6_plan.json"
CHECKPOINT = WORKSPACE / "outputs/pro6_mai_v3_mlp_hidden88_replay/pro6_mai_v3_124m_twopassfresh88_replay1/ckpt.pt"
OUTPUT = WORKSPACE / "outputs/pro6_mai_v3_mlp_cfc_layer_local_residual1"


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
        "schema_version": "mai_124m_mlp_cfc_layer_local_residual_pro6_plan_v1",
        "recorded_at": "2026-08-01",
        "status": "registered_before_zero_update_diagnostic",
        "question": "Is dense-over-fresh c_fc value concentrated in a few layers, and can current-weight left, right, or joint rank-64 subspaces recover it?",
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
            "parent_residual_structure_result_sha256": "2e641c8d6724b1fc16fdada2a97960ee2b986348ac937759c8d83fe36d3ad016",
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
            "validation_seeds": [20260831, 20260832, 20260833, 20260834],
            "validation_batches_per_window": 8,
            "evaluation_repeats": 3,
            "existing_weight_subspace_rank": 64,
            "evaluation_dtype": "float32",
            "layer_ordering": "Rank exact single-layer dense rescues by mean fit-window CE improvement over fresh88; ties use ascending layer index; freeze before validation.",
            "registered_subsets": [1, 3, 6]
        },
        "decision_rule": {
            "maximum_replicate_range": 0.0000002,
            "minimum_subspace_recovery_every_window": 0.5,
            "minimum_subspace_median_recovery": 0.65,
            "top3_minimum_recovery_for_concentration": 0.5,
            "top6_minimum_recovery_for_concentration": 0.75,
            "positive_control": "dense_exact must beat fresh88 on every held-out validation window.",
            "subspace_selection": "A top-3 or all-layer left/right/joint candidate qualifies only if it beats fresh88 and recovers at least 50% of dense-over-fresh CE on every window and 65% at the median. Select fewest added coordinates; ties prefer joint, left, right.",
            "concentration": "If no subspace qualifies, call the deficit top-3 concentrated only if exact top-3 recovers at least 50% on every window; otherwise top-6 concentrated only if exact top-6 recovers at least 75% on every window; otherwise call it distributed.",
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
                str(python), "-u", "-m", "examples.nanogpt.analyze_mlp_cfc_layer_local_residual",
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
            "This is a zero-update local tangent attribution, not a trained architecture result.",
            "The current dense c_fc weight singular frame is fixed; no learned basis, LoRA, or dense adapter is introduced.",
            "Layer order is selected only on the fit window and frozen before four new validation windows.",
            "No result directly authorizes training; a selected topology still requires production implementation and a host-local MFU gate."
        ]
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    PLAN.write_bytes(encoded)
    print(json.dumps({"plan": str(PLAN.relative_to(ROOT)), "plan_sha256": hashlib.sha256(encoded).hexdigest()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
