#!/usr/bin/env python3
"""Derive path-only PRO6 variants of registered Y400 attention controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
PRO6_ROOT = Path("/home/pro6000-9980x/MappingNetworks")
SOURCES = {
    "y400_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_qk64_5tpp_lr24e4.json": (
        "pro6_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_qk64_5tpp_lr24e4.json"
    ),
    "y400_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_outputgain_5tpp_lr24e4.json": (
        "pro6_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_outputgain_5tpp_lr24e4.json"
    ),
    "y400_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_qk64_outputgain_5tpp_lr24e4.json": (
        "pro6_mai_v3_124m_fullattn_targeted_bilateral_fullcayleylr_qk64_outputgain_5tpp_lr24e4.json"
    ),
    "y400_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_cproj8_targeted_bilateral_fullcayleylr_5tpp_lr24e4.json": (
        "pro6_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_cproj8_targeted_bilateral_fullcayleylr_5tpp_lr24e4.json"
    ),
    "y400_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_cproj8_one_sided_fullcayleylr_0p5tpp_lr24e4.json": (
        "pro6_mai_v3_124m_fullattn_cayley_horizon_capacity_qk32_v16_cproj8_one_sided_fullcayleylr_0p5tpp_lr24e4.json"
    ),
    "y400_mai_v3_124m_fullattn_globalfht_0p5tpp_lr24e4.json": (
        "pro6_mai_v3_124m_fullattn_globalfht_0p5tpp_lr24e4.json"
    ),
}


def derive_config(source_name: str, destination_name: str, source: dict[str, Any]) -> dict[str, Any]:
    config = dict(source)
    config["data_dir"] = str(PRO6_ROOT / "data/finewebedu_20b")
    config["out_dir"] = str(
        PRO6_ROOT / "outputs/pro6_mai_v3_attention_recovery" / Path(destination_name).stem
    )
    config["execution_host"] = "PRO6"
    config["host_transfer_source_config"] = (
        f"examples/nanogpt/configs/{source_name}"
    )
    config["host_transfer_policy"] = (
        "scientific settings are unchanged; only data_dir, out_dir, and "
        "explicit host-transfer metadata differ"
    )
    return config


def main() -> None:
    for source_name, destination_name in SOURCES.items():
        source = json.loads((CONFIG_DIR / source_name).read_text())
        destination = derive_config(source_name, destination_name, source)
        path = CONFIG_DIR / destination_name
        path.write_text(json.dumps(destination, indent=2, sort_keys=True) + "\n")
        print(path)


if __name__ == "__main__":
    main()
