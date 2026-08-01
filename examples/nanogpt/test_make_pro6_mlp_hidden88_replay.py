from __future__ import annotations

import json
from pathlib import Path

from examples.nanogpt.make_pro6_mlp_hidden88_replay import (
    CERTIFICATE,
    OUTPUT_CONFIG,
    SOURCE_CONFIG,
    make_config,
    make_plan,
    sha256_bytes,
    json_bytes,
)


def test_replay_changes_only_host_fields_and_adds_provenance() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    replay = make_config(source)
    changed = {
        key
        for key in set(source) | set(replay)
        if source.get(key) != replay.get(key)
    }
    assert changed == {
        "data_dir",
        "out_dir",
        "mfu_preflight_certificate",
        "monitoring_policy",
        "replay_provenance",
    }
    assert replay["mfu_preflight_required"] is True
    assert replay["mfu_min_fraction"] == 0.2
    assert replay["data_manifest_sha256"] == source["data_manifest_sha256"]
    assert replay["implementation_source_hashes"] == source["implementation_source_hashes"]


def test_plan_binds_config_and_preregisters_recovery_thresholds() -> None:
    source = json.loads(SOURCE_CONFIG.read_text())
    config_sha256 = sha256_bytes(json_bytes(make_config(source)))
    plan = make_plan(config_sha256)
    assert plan["identity"]["replay_config_sha256"] == config_sha256
    assert plan["mfu_gate"]["minimum_fraction"] == 0.2
    assert plan["mfu_gate"]["certificate"] == str(CERTIFICATE)
    assert "5.65" in plan["decision_rule"]["usable_recovery_reference"]
    assert plan["decision_rule"]["threshold_changes_after_result"] is False
