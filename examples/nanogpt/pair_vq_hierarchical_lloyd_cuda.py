"""Block-private hierarchical CUDA statistics for scalar Lloyd fits."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch


_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None


def _discover_cuda_home(root: Path) -> str | None:
    explicit = os.environ.get("CUDA_HOME")
    if explicit:
        return explicit
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parents[1])
    candidates = [Path("/usr/local/cuda"), Path("/opt/cuda")]
    for parent in (root, root.parent):
        candidates.extend(sorted(parent.glob(".cuda-*"), reverse=True))
    for variable in (
        "MAPPING_NETWORKS_WORKSPACE",
        "MAPPING_NETWORKS_NATIVE_CACHE",
        "TORCH_EXTENSIONS_DIR",
    ):
        value = os.environ.get(variable)
        if not value:
            continue
        anchor = Path(value).expanduser().resolve()
        for parent in (anchor, *list(anchor.parents)[:4]):
            candidates.extend(sorted(parent.glob(".cuda-*"), reverse=True))
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").is_file():
            return str(candidate.resolve())
    return None


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None or _EXTENSION_ERROR is not None:
        return _EXTENSION
    try:
        root = Path(__file__).resolve().parents[2]
        cuda_home = _discover_cuda_home(root)
        if cuda_home:
            os.environ["CUDA_HOME"] = cuda_home
            os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")
        from torch.utils import cpp_extension

        if cpp_extension.CUDA_HOME is None and cuda_home:
            cpp_extension.CUDA_HOME = cuda_home
        try:
            import ninja

            os.environ["PATH"] = f"{ninja.BIN_DIR}:" + os.environ.get("PATH", "")
        except Exception:
            pass
        _EXTENSION = cpp_extension.load(
            name="latent_weight_lab_pair_vq_hierarchical_lloyd_ext_v1",
            sources=[
                str(root / "csrc" / "pair_vq_hierarchical_lloyd_ext.cpp"),
                str(root / "csrc" / "pair_vq_hierarchical_lloyd_ext_cuda.cu"),
            ],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        _EXTENSION_ERROR = exc
        _EXTENSION = None
    return _EXTENSION


def hierarchical_lloyd_stats(
    values: torch.Tensor,
    midpoints: torch.Tensor,
    *,
    level_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or midpoints.ndim != 2:
        raise ValueError("hierarchical Lloyd inputs must be matrices")
    if not values.is_contiguous() or not midpoints.is_contiguous():
        raise ValueError("hierarchical Lloyd inputs must be contiguous")
    if values.dtype != torch.float32 or midpoints.dtype != torch.float32:
        raise ValueError("hierarchical Lloyd inputs must be float32")
    if values.shape[0] != midpoints.shape[0]:
        raise ValueError("hierarchical Lloyd row counts must match")
    if midpoints.shape[1] != int(level_count) - 1:
        raise ValueError("hierarchical Lloyd midpoint count does not match levels")
    if not values.is_cuda or not midpoints.is_cuda:
        raise ValueError("hierarchical Lloyd oracle requires CUDA")
    extension = _load_extension()
    if extension is None:
        raise RuntimeError("hierarchical Lloyd extension failed to load") from _EXTENSION_ERROR
    sums, counts = extension.stats(values, midpoints, int(level_count))
    return sums, counts
