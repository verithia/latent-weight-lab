"""Native CUDA statistics for exact scalar Pair-VQ Lloyd iterations."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch


_PAIR_VQ_LLOYD_EXT = None
_PAIR_VQ_LLOYD_EXT_ERROR: Exception | None = None


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


def _load_pair_vq_lloyd_ext():
    global _PAIR_VQ_LLOYD_EXT, _PAIR_VQ_LLOYD_EXT_ERROR
    if _PAIR_VQ_LLOYD_EXT is not None or _PAIR_VQ_LLOYD_EXT_ERROR is not None:
        return _PAIR_VQ_LLOYD_EXT
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
        _PAIR_VQ_LLOYD_EXT = cpp_extension.load(
            name="latent_weight_lab_pair_vq_lloyd_ext_v1",
            sources=[
                str(root / "csrc" / "pair_vq_lloyd_ext.cpp"),
                str(root / "csrc" / "pair_vq_lloyd_ext_cuda.cu"),
            ],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        _PAIR_VQ_LLOYD_EXT_ERROR = exc
        _PAIR_VQ_LLOYD_EXT = None
    return _PAIR_VQ_LLOYD_EXT


def pair_vq_lloyd_stats(
    values: torch.Tensor,
    midpoints: torch.Tensor,
    *,
    level_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return assignment sums/counts without materializing intermediate codes."""
    if values.ndim != 1 or not values.is_contiguous():
        raise ValueError("Pair-VQ Lloyd values must be a contiguous vector")
    if midpoints.ndim != 1 or not midpoints.is_contiguous():
        raise ValueError("Pair-VQ Lloyd midpoints must be a contiguous vector")
    if values.dtype != torch.float32 or midpoints.dtype != torch.float32:
        raise ValueError("Pair-VQ Lloyd statistics require float32 tensors")
    if midpoints.numel() != int(level_count) - 1:
        raise ValueError("Pair-VQ Lloyd midpoint count does not match levels")
    if values.is_cuda:
        extension = _load_pair_vq_lloyd_ext()
        if extension is None:
            raise RuntimeError(
                "native Pair-VQ Lloyd CUDA extension did not load; refusing "
                "the exact compact path instead of silently falling back"
            ) from _PAIR_VQ_LLOYD_EXT_ERROR
        sums, counts = extension.stats(values, midpoints, int(level_count))
        return sums, counts
    codes = torch.bucketize(values, midpoints)
    sums = torch.zeros(level_count, device=values.device, dtype=torch.float32)
    sums.index_add_(0, codes, values)
    counts = torch.bincount(codes, minlength=level_count)
    return sums, counts
