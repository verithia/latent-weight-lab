#!/usr/bin/env python3
"""Generate the preregistered PRO6 conjugated-general c_fc plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
ENTRYPOINT = ROOT / "examples/nanogpt/analyze_mlp_cfc_conjugated_general.py"
CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_conjugated_general_pro6_plan.json"
CHECKPOINT = WORKSPACE / "outputs/pro6_mai_v3_mlp_hidden88_replay/pro6_mai_v3_124m_twopassfresh88_replay1/ckpt.pt"
OUTPUT = WORKSPACE / "outputs/pro6_mai_v3_mlp_cfc_conjugated_general1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    replay = WORKSPACE / "latent-weight-lab-hidden88-replay"
    python = WORKSPACE / ".venv/bin/python"
    parent = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_output_general_pro6_result.json"
    payload = {
        "schema_version": "mai_124m_mlp_cfc_conjugated_general_pro6_plan_v1",
        "recorded_at": "2026-08-01",
        "question": "At the exact fresh88 coordinate budget, can fixed global FHT-conjugated 2x2 output blocks recover the missing c_fc direction, and is general shear required beyond dense orthogonal connectivity?",
        "authorization": {
            "parameter_updates": 0,
            "single_directly_polled_diagnostic": True,
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
            "parent_output_general_result_sha256": sha256_file(parent),
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
            "control_parent_stages": 64,
            "control_residual_stages": 24,
            "equal_coordinates_per_layer": 135168,
            "basis_block_size": 256,
            "orthogonal_stages": 88,
            "orthogonal_seed": 20261001,
            "general_stages": 22,
            "general_seeds": [20261011, 20261012],
            "general_damping": 0.000001,
            "coordinate_identity": "orthogonal: 88*1536*1=135168; each general arm: 22*1536*4=135168",
            "fit": "causal block-coordinate least squares in fixed random permutation/sign/FHT bases; orthogonal uses one skew coordinate per pair, general uses all four 2x2 coordinates; decoupled weight decay is applied once after the fitted transform",
            "validation_seeds": [20260901, 20260902, 20260903, 20260904],
            "validation_batches_per_window": 8,
            "evaluation_repeats": 3,
            "evaluation_dtype": "float32",
        },
        "decision_rule": {
            "maximum_replicate_range": 0.0000002,
            "minimum_recovery": 0.50,
            "median_recovery": 0.65,
            "orthogonal_selection": "select the simpler dense orthogonal family if it passes the CE gate",
            "general_selection": "otherwise both independent general bases must pass the CE gate and beat the orthogonal arm on every window",
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
                "examples.nanogpt.analyze_mlp_cfc_conjugated_general",
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
            "This is a zero-update local capacity screen, not language-model training.",
            "The coordinates are oracle-fitted to one exact-current Muon update. Passing proves fixed-basis span and held-out finite-CE value, not that task-gradient training will find the same coordinates.",
            "Causal block-coordinate fitting is a lower bound on the fixed basis span; no post-observation solver tuning is allowed.",
            "A selected arm still requires production integration and >=20% host-local measured MFU before the smallest training rung.",
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
