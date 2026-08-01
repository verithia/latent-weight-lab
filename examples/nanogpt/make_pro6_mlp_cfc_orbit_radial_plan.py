#!/usr/bin/env python3
"""Generate the preregistered PRO6 c_fc orbit/radial plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
ENTRYPOINT = ROOT / "examples/nanogpt/analyze_mlp_cfc_orbit_radial.py"
CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_orbit_radial_pro6_plan.json"
CHECKPOINT = WORKSPACE / "outputs/pro6_mai_v3_mlp_hidden88_replay/pro6_mai_v3_124m_twopassfresh88_replay1/ckpt.pt"
OUTPUT = WORKSPACE / "outputs/pro6_mai_v3_mlp_cfc_orbit_radial1"


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
        "schema_version": "mai_124m_mlp_cfc_orbit_radial_pro6_plan_v1",
        "recorded_at": "2026-08-01",
        "status": "registered_before_zero_update_diagnostic",
        "question": "Is the distributed c_fc deficit left-orbit capacity, missing input-side rotation, radial singular-value motion, or their interaction?",
        "authorization": {"parameter_updates": 0, "single_directly_polled_diagnostic": True, "production_implementation": False, "mfu_preflight": False, "scientific_training": False, "larger_rung": False},
        "identity": {
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": "0586c08fae79854d35ba765b822ae56c25efdd534df25b52797be0e8517fb075",
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": "55340bb5c035300fba9fb23b11ddf345b6bd879e3f6996c6e4e993952e01cf59",
            "dataset_manifest": str(WORKSPACE / "data/finewebedu_20b/manifest.json"),
            "dataset_manifest_sha256": "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b",
            "entrypoint": str(ENTRYPOINT.relative_to(ROOT)),
            "entrypoint_sha256": sha256_file(ENTRYPOINT),
            "parent_layer_local_result_sha256": "9be75105e6e41c2c6c2cb28a2ea41dbe99dfb2b75e05af4b59705b9c84d2431b",
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
            "validation_seeds": [20260901, 20260902, 20260903, 20260904],
            "validation_batches_per_window": 8,
            "evaluation_repeats": 3,
            "evaluation_dtype": "float32",
            "decomposition": "At each layer, decompose dense_exact-fresh88 around the current dense c_fc SVD into the exact fixed-singular-value bilateral orbit and its orthogonal radial singular-value component; also compute least-squares left-only and right-only orbit projections."
        },
        "decision_rule": {
            "maximum_replicate_range": 0.0000002,
            "sufficient_minimum_recovery": 0.75,
            "sufficient_median_recovery": 0.85,
            "radial_minimum_recovery": 0.5,
            "radial_median_recovery": 0.65,
            "positive_control": "dense_exact must beat fresh88 on every held-out validation window.",
            "priority": ["left orbit sufficient -> solver/capacity deficit", "bilateral orbit sufficient -> add input-side rotation", "left orbit plus radial sufficient -> add radial spectrum", "right orbit plus radial sufficient -> composite chart", "radial alone material -> radial deficit", "otherwise orbit/radial interaction"],
            "threshold_change_after_observation": False
        },
        "execution": {
            "host": "PRO6", "gpu": 0, "foreground_direct_polling": True,
            "watchdog": False, "callback": False, "queue_worker": False, "heartbeat": False,
            "command": [
                str(python), "-u", "-m", "examples.nanogpt.analyze_mlp_cfc_orbit_radial",
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
            "This is an exact local upper-bound decomposition with zero parameter updates, not a trained parameterization.",
            "Full orbit projections use dense SVD algebra only for diagnosis; they are not proposed as runtime implementations.",
            "No learned basis, LoRA, or dense residual adapter is introduced.",
            "Any selected family still requires a structured implementation, coordinate budget, and host-local MFU gate before training."
        ]
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    PLAN.write_bytes(encoded)
    print(json.dumps({"plan": str(PLAN.relative_to(ROOT)), "plan_sha256": hashlib.sha256(encoded).hexdigest()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
