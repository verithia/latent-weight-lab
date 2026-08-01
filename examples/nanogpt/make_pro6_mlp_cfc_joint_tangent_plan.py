#!/usr/bin/env python3
"""Generate the preregistered PRO6 joint ``c_fc`` tangent plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/pro6000-9980x/MappingNetworks")
ENTRYPOINT = ROOT / "examples/nanogpt/analyze_mlp_cfc_joint_tangent.py"
CONFIG = ROOT / "examples/nanogpt/configs/pro6_mai_v3_124m_fullattn_plus_mlp_cproj_twopassfresh88_0p5tpp_replay1.json"
PLAN = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_mlp_cfc_joint_tangent_pro6_plan.json"
CHECKPOINT = WORKSPACE / "outputs/pro6_mai_v3_mlp_hidden88_replay/pro6_mai_v3_124m_twopassfresh88_replay1/ckpt.pt"
OUTPUT = WORKSPACE / "outputs/pro6_mai_v3_mlp_cfc_joint_tangent1"


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
        "schema_version": "mai_124m_mlp_cfc_joint_tangent_pro6_plan_v1",
        "recorded_at": "2026-08-01",
        "status": "registered_before_zero_update_diagnostic",
        "question": "Does joint solution of fixed sparse output/input tangent coordinates recover the bilateral value that one diagonal Gauss-Seidel sweep missed?",
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
            "parent_structured_result_sha256": "28c1efea5b9a903b15a8a9ba4a6e3651e924fa0d861de097f261509cb7b6dc96",
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
            "cg_iterations": 8,
            "cg_damping": 0.0001,
            "connectivity": "select a shared output parent64 and output residual24 exactly as the diagonal output88 control; select one input residual64 after that control; joint candidates use fixed prefixes",
            "candidates": {
                "diagonal_output88": "registered one-sweep control",
                "joint_output88": "88 output stages solved jointly",
                "joint_output80_input32": "80 output plus 32 input stages solved jointly",
                "joint_output72_input64": "72 output plus 64 input stages solved jointly"
            },
            "coordinate_formula": "every non-dense candidate has 135168 continuous coordinates per layer",
            "solver": "eight iterations of diagonal-preconditioned conjugate gradients on (J^T J + 1e-4 diag(J^T J)) x = J^T target, using exact dense-GEMM matrix-free JVP/VJP",
            "finite_update": "apply the solved linear tangent and then decoupled weight decay once; angles are approximately 1e-5 so finite-rotation disagreement is second order",
        },
        "decision_rule": {
            "maximum_replicate_range": 0.0000002,
            "minimum_recovery": 0.50,
            "median_recovery": 0.65,
            "positive_control": "dense_exact must beat diagonal_output88 on every held-out window",
            "capacity_gate": "candidate must beat diagonal_output88 on every held-out window and recover at least 50% of its dense CE gap in the worst window and 65% at the median",
            "topology_gate": "a bilateral candidate is selected only if it also beats joint_output88 on every held-out window",
            "selection": "among qualifying bilateral candidates, maximize minimum recovery then median recovery; otherwise attribute sufficiency to joint output-only solver if it alone passes",
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
                "examples.nanogpt.analyze_mlp_cfc_joint_tangent",
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
            "This is a zero-update local tangent-capacity diagnostic, not language-model training.",
            "The linear tangent is the first-order form of Q_out W Q_in; finite-rotation error is measured only indirectly through the small solved coordinate scale.",
            "Passing proves fixed sparse connectivity capacity but does not prove an eight-CG-step production optimizer will meet the MFU gate.",
            "No learned basis, LoRA, dense residual adapter, channel gain, or radial singular-value parameter is introduced.",
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
