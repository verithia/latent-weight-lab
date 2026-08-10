#!/usr/bin/env python3
"""Materialize the preregistered 124M-active sparse-MoE LR screen."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples/nanogpt/configs"
BASE = CONFIG_DIR / "pro6_mai_v3_124m_dense_moe8_top2_0p5tpp_lr24e4.json"
VARIANTS = {
    "16e4": 1.6e-3,
    "20e4": 2.0e-3,
}


def make_variant(source: dict[str, Any], label: str, learning_rate: float) -> dict[str, Any]:
    config = dict(source)
    config["learning_rate"] = learning_rate
    config["min_lr"] = learning_rate / 10.0
    config["out_dir"] = (
        "/mnt/ssd-data/orj/MappingNetworks/outputs/"
        f"pro6_mai_v3_124m_dense_moe_ladder/lr{label}_0p5tpp"
    )
    config["mfu_preflight_certificate"] = (
        "/mnt/ssd-data/orj/MappingNetworks/outputs/"
        f"pro6_mai_v3_124m_dense_moe8_top2_mfu_lr{label}/performance_preflight.json"
    )
    return config


def main() -> None:
    source = json.loads(BASE.read_text())
    for label, learning_rate in VARIANTS.items():
        output = CONFIG_DIR / (
            f"pro6_mai_v3_124m_dense_moe8_top2_0p5tpp_lr{label}.json"
        )
        output.write_text(
            json.dumps(
                make_variant(source, label, learning_rate),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(output)


if __name__ == "__main__":
    main()
