from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
PRO6_ROOT = "/home/pro6000-9980x/MappingNetworks"

SOURCES = (
    "y400_mai_v3_690m_muon_20tpp_selected_b16ga16_prefetch_restart",
    "y400_mai_v3_985m_muon_0p5tpp_lr16e4_prefetch",
    "y400_mai_v3_985m_muon_0p5tpp_lr20e4_prefetch",
    "y400_mai_v3_985m_muon_0p5tpp_lr24e4_prefetch",
)


def pro6_name(source_name: str) -> str:
    return source_name.replace("y400_", "pro6_", 1) + "_fresh1"


def transform(source_name: str, source: dict[str, Any]) -> dict[str, Any]:
    name = pro6_name(source_name)
    config = dict(source)
    config["data_dir"] = f"{PRO6_ROOT}/data/finewebedu_20b"
    config["out_dir"] = f"{PRO6_ROOT}/outputs/mai_v3_dense/{name}"
    config["ladder_interpretation"] = (
        "fresh PRO6 host lineage for the same registered dense scientific recipe; "
        "never pool optimizer trajectory with a Y400 exact-resume lineage"
    )
    config["scheduler_host"] = "PRO6"
    config["scheduler_lineage"] = "fresh_host_relocation_v1"
    config["registered_execution_stack"] = (
        "PRO6 RTX PRO 6000 Blackwell eager PyTorch/CUDA BF16 with persistent "
        "vectorized CPU data prefetch; b16 x ga16 = 262144 tokens/update"
    )
    config["prelaunch_provenance_requirements"] = (
        "record clean Git commit, literal command, config SHA256, source hashes, "
        "data manifest SHA256, fixed-evaluation digest, and host-local >=20% MFU certificate"
    )
    config.pop("init_from", None)
    return config


def main() -> None:
    for source_name in SOURCES:
        source_path = CONFIG_DIR / f"{source_name}.json"
        config = transform(source_name, json.loads(source_path.read_text()))
        destination = CONFIG_DIR / f"{pro6_name(source_name)}.json"
        destination.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        print(destination)


if __name__ == "__main__":
    main()
