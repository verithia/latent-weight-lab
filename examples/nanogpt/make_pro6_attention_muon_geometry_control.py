#!/usr/bin/env python3
"""Build the dense packed-QKV blockwise-Muon attribution control."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_muon_5tpp_attention_trajectory_replay_lr24e4.json"
)
OUTPUT = (
    ROOT
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_dense_blockwise_qkv_muon_5tpp_lr24e4.json"
)


def build(parent: dict[str, object]) -> dict[str, object]:
    config = copy.deepcopy(parent)
    for key in (
        "diagnostic_caveat",
        "diagnostic_protocol",
        "optimizer_probe_dtype",
        "optimizer_probe_layers",
        "optimizer_probe_steps",
        "optimizer_probe_targets",
        "trajectory_snapshot_dtype",
        "trajectory_snapshot_interval",
        "trajectory_snapshot_targets",
    ):
        config.pop(key, None)
    config.update(
        {
            "candidate_scope": (
                "Dense packed-QKV optimizer-geometry control. Keep the exact "
                "dense model, initialization, schedule, data, and parameter "
                "layout, but apply Muon's polar map independently to the "
                "contiguous QK and V row blocks of each packed c_attn weight."
            ),
            "hpo_stage": "attention_partial_replacement_optimizer_attribution",
            "muon_split_attention_qkv_rows": True,
            "operator_override": (
                "attribute the QK-only partial replacement's unexpectedly "
                "strong result before claiming a mapping advantage"
            ),
            "out_dir": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_attention_partial_controls/"
                "pro6_mai_v3_124m_dense_blockwise_qkv_muon_5tpp_lr24e4"
            ),
            "mfu_preflight_certificate": (
                "/mnt/ssd-data/orj/MappingNetworks/outputs/"
                "pro6_mai_v3_attention_partial_controls/"
                "dense_blockwise_qkv_muon_5tpp_preflight.json"
            ),
            "monitoring_policy": (
                "short 124M run: direct foreground polling; no watchdog, "
                "callback, or heartbeat"
            ),
            "prelaunch_provenance_requirements": (
                "record commit, entrypoint, literal command, archived config "
                "SHA256, source hashes, dataset manifest SHA256, runtime fixed "
                "evaluation digest, exact-config MFU certificate, log/status, "
                "and checkpoint hashes"
            ),
        }
    )
    return config


def main() -> None:
    parent = json.loads(PARENT.read_text())
    OUTPUT.write_text(
        json.dumps(build(parent), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
