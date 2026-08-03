"""Compiled single-pass edge coloring for fresh task-matched Givens charts."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


SOURCE = Path(__file__).with_name("csrc") / "task_edge_coloring.cpp"
_LIBRARIES: dict[Path, ctypes.CDLL] = {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_task_edge_coloring(
    cache_dir: Path | None = None,
) -> tuple[ctypes.CDLL, Path]:
    """Build/load the tiny standard-C++ matcher without PyTorch headers."""
    if cache_dir is None:
        configured = os.environ.get("MAPPING_NETWORKS_NATIVE_CACHE")
        cache_dir = (
            Path(configured)
            if configured
            else Path(tempfile.gettempdir())
            / "mapping_networks_native"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_digest = file_sha256(SOURCE)
    library_path = (
        cache_dir / f"task_edge_coloring_{source_digest[:16]}.so"
    )
    if library_path not in _LIBRARIES:
        if not library_path.exists():
            temporary = library_path.with_suffix(".so.tmp")
            subprocess.run(
                [
                    os.environ.get("CXX", "c++"),
                    "-std=c++17",
                    "-O3",
                    "-DNDEBUG",
                    "-shared",
                    "-fPIC",
                    str(SOURCE),
                    "-o",
                    str(temporary),
                ],
                check=True,
            )
            temporary.replace(library_path)
        library = ctypes.CDLL(str(library_path))
        function = library.task_edge_color
        function.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        function.restype = ctypes.c_int
        _LIBRARIES[library_path] = library
    return _LIBRARIES[library_path], library_path


def color_sorted_edges(
    sorted_edges: torch.Tensor,
    *,
    width: int,
    stages: int,
    seed: int,
    cache_dir: Path | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Color score-sorted CPU edges into edge-disjoint perfect matchings."""
    if (
        sorted_edges.ndim != 2
        or sorted_edges.shape[1] != 2
        or sorted_edges.device.type != "cpu"
    ):
        raise ValueError("sorted_edges must be a CPU tensor shaped [E,2]")
    if width <= 0 or width % 2 or stages <= 0 or stages >= width:
        raise ValueError("require even width and 0 < stages < width")
    edges = np.ascontiguousarray(
        sorted_edges.to(dtype=torch.int32).numpy()
    )
    output = np.empty((stages, width), dtype=np.int32)
    candidate_counts = np.empty(stages, dtype=np.int32)
    library, library_path = build_task_edge_coloring(cache_dir)
    started = time.perf_counter()
    return_code = library.task_edge_color(
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        edges.shape[0],
        width,
        stages,
        seed,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        candidate_counts.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32)
        ),
    )
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(
            f"compiled task edge coloring failed with code {return_code}"
        )
    permutations = torch.from_numpy(output.astype(np.int64))
    return permutations, {
        "native_library": str(library_path),
        "native_library_sha256": file_sha256(library_path),
        "source_sha256": file_sha256(SOURCE),
        "native_output_validated": True,
        "native_seconds": elapsed,
        "candidate_edge_fraction": float(
            candidate_counts.sum()
            / (stages * (width // 2))
        ),
        "minimum_stage_candidate_edge_fraction": float(
            candidate_counts.min() / (width // 2)
        ),
    }


def fast_muon_matched_permutations(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    stages: int,
    neighbors: int,
    seed: int,
    cache_dir: Path | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select fresh task matchings with one compiled edge-coloring pass."""
    if (
        weight.ndim != 2
        or weight.shape != direction.shape
        or weight.shape[1] <= 0
        or weight.shape[1] % 2
    ):
        raise ValueError(
            "weight and direction must be same-shaped matrices with even width"
        )
    width = int(weight.shape[1])
    if stages <= 0 or neighbors < stages:
        raise ValueError("require 0 < stages and neighbors >= stages")
    if neighbors >= width:
        raise ValueError("neighbors must be smaller than width")

    if weight.is_cuda:
        torch.cuda.synchronize(weight.device)
    started = time.perf_counter()
    source = weight.float()
    requested = direction.float()
    cross = source.T @ requested
    scores = (cross - cross.T).abs()
    scores.fill_diagonal_(-1.0)
    top_scores, top_indices = torch.topk(
        scores, k=neighbors, dim=1
    )
    flattened_scores = top_scores.reshape(-1)
    order = torch.argsort(flattened_scores, descending=True)
    left = (
        torch.arange(width, device=weight.device)
        .repeat_interleave(neighbors)
        .index_select(0, order)
    )
    right = top_indices.reshape(-1).index_select(0, order)
    edges = torch.stack(
        (torch.minimum(left, right), torch.maximum(left, right)),
        dim=1,
    ).to(device="cpu", dtype=torch.int32)
    if weight.is_cuda:
        torch.cuda.synchronize(weight.device)
    prepared_seconds = time.perf_counter() - started
    permutations, diagnostics = color_sorted_edges(
        edges,
        width=width,
        stages=stages,
        seed=seed,
        cache_dir=cache_dir,
    )
    diagnostics.update(
        {
            "prepared_seconds": prepared_seconds,
            "total_seconds": (
                prepared_seconds + diagnostics["native_seconds"]
            ),
            "candidate_edges": int(edges.shape[0]),
        }
    )
    return permutations, diagnostics
