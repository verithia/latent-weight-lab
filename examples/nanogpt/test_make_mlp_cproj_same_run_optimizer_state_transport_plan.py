from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.nanogpt.analyze_mlp_cproj_same_run_optimizer_state_transport import (
    validate_plan,
)
from examples.nanogpt.make_mlp_cproj_same_run_optimizer_state_transport_plan import (
    PROBE_STEPS,
    SNAPSHOT_STEPS,
    build_plan,
)


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> dict[str, Path]:
    data_dir = tmp_path / "data"
    manifest = data_dir / "manifest.json"
    manifest_sha = write_json(manifest, {"dataset": "fixed"})
    config = tmp_path / "config.json"
    config_sha = write_json(config, {"data_dir": str(data_dir)})
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    acquisition_plan = tmp_path / "acquisition_plan.json"
    plan_sha = write_json(
        acquisition_plan,
        {
            "identity": {
                "dataset_manifest_sha256": manifest_sha,
                "fixed_eval_indices_sha256": "f" * 64,
            }
        },
    )
    acquisition_result = tmp_path / "acquisition_result.json"
    write_json(
        acquisition_result,
        {
            "classification": (
                "ACCEPTED_SAME_RUN_CPROJ_PARAMETER_OPTIMIZER_TRAJECTORY"
            ),
            "passed": True,
            "authorization": {"zero_update_state_transport_analysis": True},
            "identity": {
                "plan_sha256": plan_sha,
                "config_sha256": config_sha,
                "checkpoint_sha256": checkpoint_sha,
                "run_identity_sha256": "a" * 64,
            },
            "inventory": {
                "snapshot_count": len(SNAPSHOT_STEPS),
                "probe_count": len(PROBE_STEPS),
                "snapshot_sha256_by_step": {
                    str(step): "b" * 64 for step in SNAPSHOT_STEPS
                },
                "probe_sha256_by_step": {
                    str(step): "c" * 64 for step in PROBE_STEPS
                },
            },
        },
    )
    return {
        "acquisition_result": acquisition_result,
        "acquisition_plan": acquisition_plan,
        "config": config,
        "checkpoint": checkpoint,
    }


def make(tmp_path: Path) -> dict[str, object]:
    paths = fixture(tmp_path)
    return build_plan(
        acquisition_result_path=paths["acquisition_result"],
        acquisition_plan_path=paths["acquisition_plan"],
        config_path=paths["config"],
        checkpoint_path=paths["checkpoint"],
        snapshot_dir=tmp_path / "snapshots",
        probe_dir=tmp_path / "probes",
        analysis_output_dir=tmp_path / "analysis",
    )


def test_builder_emits_analyzer_valid_fail_closed_plan(tmp_path: Path) -> None:
    plan = make(tmp_path)
    validate_plan(plan)
    assert plan["analysis"]["probe_steps"][0] == 0
    assert plan["analysis"]["future_phase_target_by_probe"]["2372"] is None
    assert plan["authorization"]["run_language_model_training"] is False


def test_builder_rejects_unaccepted_or_drifted_inputs(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    acquisition = json.loads(paths["acquisition_result"].read_text())
    acquisition["classification"] = "REJECTED"
    write_json(paths["acquisition_result"], acquisition)
    with pytest.raises(ValueError, match="not accepted"):
        build_plan(
            acquisition_result_path=paths["acquisition_result"],
            acquisition_plan_path=paths["acquisition_plan"],
            config_path=paths["config"],
            checkpoint_path=paths["checkpoint"],
            snapshot_dir=tmp_path / "snapshots",
            probe_dir=tmp_path / "probes",
            analysis_output_dir=tmp_path / "analysis",
        )
