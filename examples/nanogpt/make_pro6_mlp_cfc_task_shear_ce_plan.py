#!/usr/bin/env python3
"""Generate the preregistered PRO6 held-out c_fc task-shear CE plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
ENTRYPOINT = ROOT / "examples/nanogpt/analyze_mlp_cfc_task_shear_ce.py"
CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_task_shear_ce_pro6_plan.json"
CHECKPOINT = WORKSPACE / "outputs/pro6_mai_v3_mlp_hidden88_replay/pro6_mai_v3_124m_twopassfresh88_replay1/ckpt.pt"
OUTPUT = WORKSPACE / "outputs/pro6_mai_v3_mlp_cfc_task_shear_ce1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    replay = WORKSPACE / "latent-weight-lab-hidden88-replay"
    python = WORKSPACE / ".venv/bin/python"
    parent = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_task_shear_fit_pro6_result.json"
    payload = {
        "schema_version": "mai_124m_mlp_cfc_task_shear_ce_pro6_plan_v1",
        "recorded_at": "2026-08-01",
        "question": "Does the selected equal-coordinate fresh64-plus-shear24 c_fc chart improve finite CE over fresh88 on untouched validation windows?",
        "authorization": {
            "parameter_updates": 0,
            "single_directly_polled_ce_diagnostic": True,
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
            "parent_task_shear_fit_result_sha256": sha256_file(parent),
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
            "selected_candidate": "fresh64_shear24",
            "control": "fresh88",
            "dense_reference": "dense_exact",
            "equal_coordinates_per_layer": 135168,
            "validation_seeds": [20261101, 20261102, 20261103, 20261104],
            "validation_batches_per_window": 8,
            "evaluation_repeats": 3,
            "evaluation_dtype": "float32",
        },
        "decision_rule": {
            "maximum_replicate_range": 0.0000002,
            "minimum_recovery": 0.05,
            "median_recovery": 0.10,
            "selection": "The selected chart must beat fresh88 on every window and recover at least 5% of the dense-over-fresh CE gap in the worst window and 10% at the median. Dense exact must beat fresh88 on every window.",
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
                "examples.nanogpt.analyze_mlp_cfc_task_shear_ce",
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
            "This is a zero-update held-out finite-CE diagnostic, not language-model training.",
            "The chart coordinates are fitted to the fixed train window; the four validation windows are scoring-only and were not used by the preceding fit gate.",
            "Passing establishes local held-out task value and authorizes only production implementation plus an >=20% host-local measured MFU preflight.",
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
