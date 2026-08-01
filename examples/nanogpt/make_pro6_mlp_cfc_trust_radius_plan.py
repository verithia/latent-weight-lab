#!/usr/bin/env python3
"""Generate the immutable PRO6 c_fc trust-radius diagnostic plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
ENTRYPOINT = ROOT / "examples/nanogpt/analyze_mlp_cfc_trust_radius.py"
CONFIG = ROOT / (
    "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
)
REPLAY_RESULT = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_two_pass_fresh_hidden88_pro6_replay_result.json"
)
PLAN = ROOT / (
    "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cfc_trust_radius_pro6_plan.json"
)
CHECKPOINT = WORKSPACE / (
    "outputs/pro6_mai_v3_mlp_hidden88_replay/"
    "pro6_mai_v3_124m_twopassfresh88_replay1/ckpt.pt"
)
OUTPUT = WORKSPACE / "outputs/pro6_mai_v3_mlp_cfc_trust_radius1"
EXPECTED = {
    "config": "55340bb5c035300fba9fb23b11ddf345b6bd879e3f6996c6e4e993952e01cf59",
    "replay_result": "cc0c50e4876a5aa60d9c87a3465495dc56caa6d9a7225094e0e383bc7dd46439",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )


def make_plan() -> dict[str, Any]:
    if sha256_file(CONFIG) != EXPECTED["config"]:
        raise RuntimeError("PRO6 replay config drifted")
    if sha256_file(REPLAY_RESULT) != EXPECTED["replay_result"]:
        raise RuntimeError("PRO6 replay result drifted")
    remote_repo = WORKSPACE / "latent-weight-lab-hidden88-replay"
    python = WORKSPACE / ".venv/bin/python"
    return {
        "schema_version": "mai_124m_mlp_cfc_trust_radius_pro6_plan_v1",
        "recorded_at": "2026-08-01",
        "status": "registered_before_zero_update_diagnostic",
        "hypothesis": (
            "The fresh 64+24 expansion-side chart already retains useful "
            "task-aligned direction, but the persisted full Muon step lies "
            "outside its transferable local trust region. Selecting only a "
            "scalar radius on the registered fit window may convert that "
            "direction into reproducible finite-CE improvement."
        ),
        "authorization": {
            "single_directly_polled_diagnostic": True,
            "parameter_updates": 0,
            "topology_change": False,
            "coordinate_change": False,
            "production_implementation": False,
            "mfu_preflight": False,
            "scientific_training": False,
            "larger_rung": False,
        },
        "identity": {
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": (
                "0586c08fae79854d35ba765b822ae56c25efdd534df25b52797be0e8517fb075"
            ),
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": EXPECTED["config"],
            "dataset_manifest": str(
                WORKSPACE / "data/finewebedu_20b/manifest.json"
            ),
            "dataset_manifest_sha256": (
                "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
            ),
            "fixed_eval_indices_sha256": (
                "5ca31b59768e43de808ad5e206ed152a4a0a3515ad68d29a0b2338c4db140747"
            ),
            "replay_result": str(REPLAY_RESULT.relative_to(ROOT)),
            "replay_result_sha256": EXPECTED["replay_result"],
            "entrypoint": str(ENTRYPOINT.relative_to(ROOT)),
            "entrypoint_sha256": sha256_file(ENTRYPOINT),
            "scientific_source_base_commit": (
                "c2cb932a29b2aafca58315d487937b36b8296e15"
            ),
        },
        "fixed_protocol": {
            "layers": list(range(12)),
            "batch_size": 4,
            "block_size": 1024,
            "fit_batches": 4,
            "fit_train_seed": 20260806,
            "matching_seed": 20260806,
            "matching_neighbors": 64,
            "parent_stages": 64,
            "residual_stages": 24,
            "trust_scales": [
                0.015625,
                0.03125,
                0.0625,
                0.125,
                0.25,
                0.5,
                1.0,
            ],
            "selection_rule": (
                "minimize fresh88 finite CE on the fit window; within "
                "1e-8 CE choose the smallest positive scale; freeze before "
                "reading any validation loss"
            ),
            "validation_seeds": [
                20260811,
                20260812,
                20260813,
                20260814,
            ],
            "validation_batches_per_window": 8,
            "evaluation_repeats": 3,
            "gradient_dtype": "bfloat16",
            "evaluation_dtype": "float32",
        },
        "decision_rule": {
            "minimum_fit_ce_improvement": 0.000001,
            "fit_tie_tolerance": 0.00000001,
            "maximum_replicate_range": 0.0000002,
            "minimum_validation_ce_margin": 0.0000002,
            "pass": (
                "The frozen positive radius improves fit CE by at least "
                "1e-6; every repeated float32 evaluation is stable within "
                "2e-7; and fresh88 has at least 2e-7 lower CE than baseline, "
                "dense exact, and random88 in the worst replicate on every "
                "one of four new validation windows."
            ),
            "on_pass": (
                "SELECT_TRUST_SCALED_FRESH88_CFC_FOR_PRODUCTION_MFU_GATE"
            ),
            "on_fail": "REJECT_TRUST_SCALED_FRESH88_CFC",
            "threshold_change_after_observation": False,
        },
        "execution": {
            "host": "PRO6",
            "gpu": 0,
            "entrypoint": str(ENTRYPOINT.relative_to(ROOT)),
            "foreground_direct_polling": True,
            "watchdog": False,
            "callback": False,
            "queue_worker": False,
            "heartbeat": False,
            "command": [
                str(python),
                "-u",
                "-m",
                "examples.nanogpt.analyze_mlp_cfc_trust_radius",
                "--checkpoint",
                str(CHECKPOINT),
                "--config",
                str(remote_repo / CONFIG.relative_to(ROOT)),
                "--data-dir",
                str(WORKSPACE / "data/finewebedu_20b"),
                "--plan",
                str(remote_repo / PLAN.relative_to(ROOT)),
                "--output",
                str(OUTPUT),
                "--device",
                "cuda",
                "--native-cache",
                str(WORKSPACE / "native_cache"),
            ],
        },
        "limitations": [
            "This is a terminal local line-search diagnostic, not global manifold evidence.",
            "The fit window supplies both the exact-current chart and scalar radius; only the four validation windows are held out.",
            "The scalar multiplies the complete finite update, including its decoupled-weight-decay component.",
            "The hidden88 c_proj, attention replacement, residual stream, and every non-c_fc parameter remain fixed.",
            "Passing authorizes only production implementation followed by a separate measured MFU gate; it does not authorize training.",
            "No learned dense basis, LoRA, dense residual adapter, topology change, coordinate increase, or static-frame retry is permitted.",
        ],
    }


def main() -> None:
    payload = json_bytes(make_plan())
    PLAN.write_bytes(payload)
    print(
        json.dumps(
            {
                "plan": str(PLAN.relative_to(ROOT)),
                "plan_sha256": hashlib.sha256(payload).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
