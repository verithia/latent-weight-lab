#!/usr/bin/env python3
"""Generate the preregistered PRO6 structured bilateral ``c_fc`` plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
ENTRYPOINT = ROOT / "examples/nanogpt/analyze_mlp_cfc_structured_bilateral.py"
CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_structured_bilateral_pro6_plan.json"
CHECKPOINT = WORKSPACE / "outputs/pro6_mai_v3_mlp_hidden88_replay/pro6_mai_v3_124m_twopassfresh88_replay1/ckpt.pt"
OUTPUT = WORKSPACE / "outputs/pro6_mai_v3_mlp_cfc_structured_bilateral1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    replay = WORKSPACE / "latent-weight-lab-hidden88-replay"
    python = WORKSPACE / ".venv/bin/python"
    allocations = [
        {"candidate": "output88_input0", "output_stages": 88, "input_stages": 0},
        {"candidate": "output80_input32", "output_stages": 80, "input_stages": 32},
        {"candidate": "output72_input64", "output_stages": 72, "input_stages": 64},
        {"candidate": "output64_input96", "output_stages": 64, "input_stages": 96},
        {"candidate": "output56_input128", "output_stages": 56, "input_stages": 128},
    ]
    payload = {
        "schema_version": "mai_124m_mlp_cfc_structured_bilateral_pro6_plan_v1",
        "recorded_at": "2026-08-01",
        "status": "registered_before_zero_update_diagnostic",
        "question": "Can a deployable two-sided sparse-Givens c_fc chart recover the useful dense Muon direction at the exact continuous-coordinate cost of output-only hidden88?",
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
            "parent_orbit_radial_result_sha256": "9913ed28768aee993f511f064f867857abd461846d17f7a4eb673db848886724",
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
            "validation_seeds": [20260901, 20260902, 20260903, 20260904],
            "validation_batches_per_window": 8,
            "evaluation_repeats": 3,
            "evaluation_dtype": "float32",
            "composition_order": "all output-axis causal passes, then all input-axis causal residual passes",
            "stage_allocations": allocations,
            "coordinate_formula": "output_stages*(3072/2)+input_stages*(768/2)=135168 continuous coordinates per layer for every non-dense candidate",
            "matching": "first output pass selects connectivity from the exact polar Muon direction; every later output/input pass selects and fits only the remaining exact-current update residual",
            "weight_decay": "apply the registered decoupled Muon weight decay exactly once after composing rotations",
        },
        "decision_rule": {
            "maximum_replicate_range": 0.0000002,
            "minimum_recovery": 0.50,
            "median_recovery": 0.65,
            "positive_control": "dense_exact must beat output88_input0 on every held-out window",
            "candidate_gate": "candidate must beat output88_input0 on every held-out window and recover at least 50% of its dense CE gap in the worst window and 65% at the median",
            "selection": "among qualifying equal-coordinate candidates, maximize minimum held-out recovery, then median recovery",
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
                "examples.nanogpt.analyze_mlp_cfc_structured_bilateral",
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
            "This is an exact-current, zero-update tangent/finite-CE screen, not language-model training.",
            "The fixed output-then-input order is one causal block Gauss-Seidel sweep; rejection would motivate an alternating-order screen, not refute the exact bilateral orbit result.",
            "No learned basis, LoRA, dense residual adapter, per-channel gain, or radial singular-value parameter is introduced.",
            "A selected allocation still requires production integration, correctness tests, and host-local measured MFU >=20% before training.",
        ],
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )
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
