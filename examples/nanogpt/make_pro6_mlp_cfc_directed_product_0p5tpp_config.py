#!/usr/bin/env python3
"""Preregister the smallest-rung directed-product c_fc training run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "examples/nanogpt/configs"
ARTIFACTS = CONFIGS / "selection_artifacts"
PARENT = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct30_29_29_mfu_retry1.json"
)
MFU_RESULT = (
    ARTIFACTS / "124m_mlp_cfc_directed_product_mfu_retry1_result.json"
)
OUTPUT_CONFIG = (
    CONFIGS
    / "pro6_mai_v3_124m_fullattn_plus_mlp_cfc_"
    "directedproduct30_29_29_0p5tpp.json"
)
OUTPUT_PLAN = (
    ARTIFACTS / "124m_mlp_cfc_directed_product_0p5tpp_plan.json"
)

PARENT_SHA256 = (
    "cbdb67234128a8f99b8735ea0964d15159e821fc27f3c82ff44bda4355aae6f2"
)
MFU_RESULT_SHA256 = (
    "0da2c69612eb61852b58c951a2c6e2755da5166de0ddbc3e9ee998eb5f4a70c4"
)
DATASET_MANIFEST_SHA256 = (
    "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b"
)
FIXED_EVAL_SPEC_SHA256 = (
    "7cb7f4a2e14eca6229bcca0bde184ab21acaab01cc041cf39d0c87837b90cb52"
)
IMPLEMENTATION_COMMIT = "29d5e90ff419cde1b13fc5541aff79b12ec49f27"
SOURCE_HASHES = {
    "examples/nanogpt/model.py": (
        "07602c5045077a848ab1e0a176431dde6b15c07be08b271a56276e91ad13ceae"
    ),
    "examples/nanogpt/muon_matched_givens.py": (
        "b2973183268673859272837c80a13a8ddeec6b2bd5a43cef1703bcc9a039c641"
    ),
    "examples/nanogpt/train.py": (
        "bc43b09497dd396025f1335c40889698ba1e7f4d5ad7ca76809e6e8d388cda44"
    ),
    "examples/nanogpt/muon.py": (
        "532e172d91306d12284507c96aa3176792b33eb657f568512ce278bb5a9874ff"
    ),
    "latent_weight_lab/block_fht.py": (
        "864ba9a79664cba2f830c06b11214538b7817685e1ba990f6e103feefb49b561"
    ),
}

REMOTE_WORKTREE = (
    "/home/pro6000-9980x/MappingNetworks/"
    "latent-weight-lab-cfc-midpoint-replay"
)
PYTHON = "/mnt/ssd-data/orj/MappingNetworks/.venv/bin/python"
OUTPUT_ROOT = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_0p5tpp"
)
RUN_DIR = f"{OUTPUT_ROOT}/pro6_mai_v3_124m_cfc_directed_product_0p5tpp"
LOG = f"{OUTPUT_ROOT}/train.log"
RUN_METADATA = f"{OUTPUT_ROOT}/prelaunch_run_metadata.json"
CERTIFICATE = (
    "/home/pro6000-9980x/MappingNetworks/outputs/"
    "pro6_mai_v3_mlp_cfc_directed_product_mfu_retry1/"
    "performance_preflight.json"
)
CERTIFICATE_SHA256 = (
    "a14389834573bb5195ace071018886ee642146143e568541c57b9f1d45a04432"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def validate_inputs() -> None:
    if sha256_file(PARENT) != PARENT_SHA256:
        raise RuntimeError("qualified MFU config hash drifted")
    if sha256_file(MFU_RESULT) != MFU_RESULT_SHA256:
        raise RuntimeError("qualified MFU result hash drifted")
    result = json.loads(MFU_RESULT.read_text())
    if not result.get("passed") or result["measurement"]["mfu_fraction"] < 0.2:
        raise RuntimeError("MFU result does not authorize smallest-rung training")
    for relative, expected in SOURCE_HASHES.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"runtime source hash drifted: {relative}")


def make_config() -> dict[str, Any]:
    config = json.loads(PARENT.read_text())
    config.update(
        {
            "candidate_scope": (
                "held-out-selected full-attention replacement plus qualified "
                "two-pass c_proj and task-selected three-stage 30+29+29 "
                "directed-product c_fc; first scientific 124M/0.5TPP test"
            ),
            "checkpoint_history": False,
            "checkpoint_wall_clock_seconds": 7200,
            "eval_protocol_id": "mai_ladder_fixed_eval_indices_v2",
            "eval_seed": 20260715,
            "fixed_eval_index_spec_sha256": FIXED_EVAL_SPEC_SHA256,
            "fixed_eval_indices": True,
            "fixed_eval_indices_protocol": (
                "split_local_cpu_generators_eval_seed_plus_split_offset_v2_shared_b16"
            ),
            "hpo_stage": "directed_product_cfc_smallest_rung_validation",
            "instability_policy": (
                "hard reject NaN, Inf, divergence, failed terminal evaluation, "
                "or incomplete exact-resume state"
            ),
            "ladder_role": "mlp_full_replacement_directed_product_smallest_rung",
            "mfu_preflight_certificate": CERTIFICATE,
            "mfu_preflight_certificate_sha256": CERTIFICATE_SHA256,
            "mfu_preflight_result": str(MFU_RESULT.relative_to(ROOT)),
            "mfu_preflight_result_sha256": MFU_RESULT_SHA256,
            "monitoring_policy": (
                "short 238-update run is directly foreground-polled; no "
                "watchdog, callback, queue worker, or heartbeat"
            ),
            "out_dir": RUN_DIR,
            "planned_tokens": 62186880,
            "prelaunch_provenance_requirements": (
                "record execution commit, entrypoint, literal command, config "
                "and source hashes, dataset manifest, fixed-eval protocol, "
                "MFU certificate, log, and exact-resume checkpoint"
            ),
            "preregistered_decision_rule": {
                "primary_metric": (
                    "terminal fixed-window validation cross entropy at update 238"
                ),
                "attention_only_validation_ce": 5.4918,
                "accepted_attention_gap": 0.1,
                "success_ce_maximum": 5.5918,
                "qualified_cproj_only_validation_ce": 5.592058181762695,
                "success": (
                    "stable terminal validation CE <= 5.5918, closing the "
                    "full-replacement gap to at most +0.10 versus attention-only"
                ),
                "directional_only": (
                    "stable terminal validation CE > 5.5918 but lower than "
                    "the qualified cproj-only control 5.592058181762695"
                ),
                "reject": (
                    "terminal validation CE >= 5.592058181762695, instability, "
                    "incomplete run, or provenance failure"
                ),
            },
            "registered_execution_stack": (
                "eager PyTorch/CUDA BF16 with required native BlockFHT backend; "
                "torch.compile=false"
            ),
            "registered_resume_determinism_required": True,
            "registered_resume_protocol": (
                "atomic latest checkpoint with full RNG state, folded c_fc and "
                "c_proj weights, optimizer steps, and exact Muon momentum buffers"
            ),
            "run_metadata_path": RUN_METADATA,
            "save_checkpoint": True,
            "schedule_rounding": "ceil(planned_tokens / tokens_per_iter)",
            "scheduled_tokens": 62390272,
            "scheduled_tpp": 0.5016353288667963,
            "screen_only": True,
            "screen_only_resolution": (
                "only this preregistered 124M/0.5TPP run is authorized; no "
                "larger rung until its result is frozen"
            ),
            "selection_endpoint": (
                "terminal fixed-window validation CE versus attention-only and "
                "the qualified cproj-only dense-c_fc control"
            ),
            "terminal_eval_protocol": (
                "evaluate at max_iters even when it is off periodic cadence"
            ),
            "terminal_eval_required": True,
            "tokens_per_iter": 262144,
            "training_sampling_protocol": (
                "dedicated_cpu_generator_train_data_seed_v1"
            ),
        }
    )
    return config


def make_plan(config_sha256: str) -> dict[str, Any]:
    remote_config = (
        f"{REMOTE_WORKTREE}/examples/nanogpt/configs/{OUTPUT_CONFIG.name}"
    )
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "CUDA_HOME=/mnt/ssd-data/orj/MappingNetworks/.cuda-12.8",
        "PYTHONPATH=.",
        PYTHON,
        "-u",
        "-m",
        "examples.nanogpt.train",
        "--config",
        remote_config,
    ]
    return {
        "schema_version": "mai_124m_mlp_cfc_directed_product_0p5tpp_plan_v1",
        "created_at": "2026-08-03",
        "status": "registered_after_geometry_and_native_mfu_pass_before_training",
        "question": (
            "Does the moving 30+29+29 directed-product c_fc chart preserve the "
            "qualified cproj-only trajectory and close full MLP replacement to "
            "within +0.10 CE of attention-only at 124M/0.5TPP?"
        ),
        "candidate": {
            "config": str(OUTPUT_CONFIG.relative_to(ROOT)),
            "config_sha256": config_sha256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "model_tier": "124m",
            "planned_tpp": 0.5,
            "max_iters": 238,
            "incoming_schedule": [30, 29, 29],
            "coordinates_per_cfc_layer": 270336,
            "family_radius_ratio": 0.6589686140591383,
        },
        "qualification": {
            "mfu_result": str(MFU_RESULT.relative_to(ROOT)),
            "mfu_result_sha256": MFU_RESULT_SHA256,
            "certificate": CERTIFICATE,
            "certificate_sha256": CERTIFICATE_SHA256,
            "mfu_fraction": 0.2890836323811324,
            "minimum_mfu_fraction": 0.2,
        },
        "identity": {
            "entrypoint": "examples/nanogpt/train.py",
            "implementation_source_hashes": SOURCE_HASHES,
            "dataset_manifest": (
                "/home/pro6000-9980x/MappingNetworks/data/"
                "finewebedu_20b/manifest.json"
            ),
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "fixed_eval_spec_sha256": FIXED_EVAL_SPEC_SHA256,
        },
        "controls": {
            "attention_only_validation_ce": 5.4918,
            "qualified_cproj_only_validation_ce": 5.592058181762695,
            "note": (
                "The cproj-only control leaves c_fc dense, so matching it is the "
                "strict causal test of the new c_fc replacement."
            ),
        },
        "decision_rule": {
            "primary_metric": "terminal fixed-window validation CE at update 238",
            "success": "finite complete run with terminal validation CE <= 5.5918",
            "directional_only": (
                "terminal validation CE > 5.5918 and < 5.592058181762695"
            ),
            "reject": (
                "terminal validation CE >= 5.592058181762695, nonfinite path, "
                "incomplete terminal evaluation, or identity mismatch"
            ),
            "threshold_changes_after_measurement": False,
        },
        "protocol": {
            "host": "PRO6",
            "gpu": 0,
            "python": PYTHON,
            "command": command,
            "working_directory": REMOTE_WORKTREE,
            "log": LOG,
            "run_directory": RUN_DIR,
            "prelaunch_run_metadata": RUN_METADATA,
            "execution": "direct foreground polling through terminal exit",
            "watchdog": False,
            "callback": False,
            "queue_worker": False,
            "heartbeat": False,
        },
        "authorization": {
            "scope": "exactly one 124M/0.5TPP scientific training run",
            "larger_rung_authorized": False,
            "additional_structure_authorized": False,
            "automatic_rerun_authorized": False,
        },
    }


def main() -> None:
    validate_inputs()
    config = make_config()
    config_payload = json_bytes(config)
    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    plan = make_plan(config_sha256)
    OUTPUT_CONFIG.write_bytes(config_payload)
    OUTPUT_PLAN.write_bytes(json_bytes(plan))
    print(
        json.dumps(
            {
                "config": str(OUTPUT_CONFIG),
                "config_sha256": config_sha256,
                "plan": str(OUTPUT_PLAN),
                "plan_sha256": sha256_file(OUTPUT_PLAN),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
