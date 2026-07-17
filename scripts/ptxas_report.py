from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
    try:
        import ninja

        os.environ["PATH"] = f"{ninja.BIN_DIR}:" + os.environ.get("PATH", "")
    except Exception:
        pass
    load(
        name="latent_weight_lab_block_fht_ext_ptxas_report",
        sources=[
            str(root / "csrc" / "block_fht_ext.cpp"),
            str(root / "csrc" / "block_fht_ext_cuda.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
        extra_cflags=["-O3"],
        verbose=True,
    )
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
