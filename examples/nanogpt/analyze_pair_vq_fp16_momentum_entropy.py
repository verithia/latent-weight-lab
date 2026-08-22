#!/usr/bin/env python3
"""Audit exact lossless codability of Pair-VQ FP16 ambient Muon history."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import time
import zlib
from pathlib import Path
from typing import Callable

import numpy as np
import torch


SCHEMA = "mai_pair_vq_fp16_momentum_lossless_entropy_result_v1"
CODEC_METADATA_BYTES = 64 + 24 * 16


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total == 0.0:
        return 0.0
    p = counts[counts > 0.0] / total
    return float(-(p * np.log2(p)).sum())


def word_counts(words: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(words, dtype=np.uint16).reshape(-1), minlength=65536)


def prefix_entropy(counts: np.ndarray, prefix_mantissa_bits: int) -> float:
    if not 0 <= prefix_mantissa_bits <= 10:
        raise ValueError("prefix_mantissa_bits must be in [0, 10]")
    words = np.arange(65536, dtype=np.uint16)
    lower = 10 - prefix_mantissa_bits
    contexts = words >> lower
    context_counts = np.bincount(
        contexts.astype(np.int64), weights=np.asarray(counts, dtype=np.float64)
    )
    return entropy_from_counts(context_counts)


def conditional_lower_mantissa_entropy(
    counts: np.ndarray, prefix_mantissa_bits: int
) -> float:
    return entropy_from_counts(counts) - prefix_entropy(counts, prefix_mantissa_bits)


def bit_entropies(counts: np.ndarray) -> list[float]:
    words = np.arange(65536, dtype=np.uint16)
    total = float(np.asarray(counts).sum())
    result = []
    for bit in range(16):
        ones = float(np.asarray(counts)[((words >> bit) & 1) == 1].sum())
        result.append(entropy_from_counts(np.array([total - ones, ones])))
    return result


def _byte_shuffle(words: np.ndarray) -> bytes:
    pairs = np.asarray(words, dtype="<u2").reshape(-1).view(np.uint8).reshape(-1, 2)
    return pairs[:, 0].tobytes() + pairs[:, 1].tobytes()


def _byte_unshuffle(payload: bytes, count: int) -> np.ndarray:
    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw.size != count * 2:
        raise ValueError("byte-shuffled payload has the wrong size")
    pairs = np.empty((count, 2), dtype=np.uint8)
    pairs[:, 0] = raw[:count]
    pairs[:, 1] = raw[count:]
    return pairs.reshape(-1).view("<u2").copy()


def encode_raw_zlib1(words: np.ndarray, shape: tuple[int, ...]) -> bytes:
    del shape
    return zlib.compress(np.asarray(words, dtype="<u2").reshape(-1).tobytes(), 1)


def decode_raw_zlib1(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    count = math.prod(shape)
    return np.frombuffer(zlib.decompress(payload), dtype="<u2", count=count).copy()


def encode_byte_shuffle_zlib1(words: np.ndarray, shape: tuple[int, ...]) -> bytes:
    del shape
    return zlib.compress(_byte_shuffle(words), 1)


def decode_byte_shuffle_zlib1(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return _byte_unshuffle(zlib.decompress(payload), math.prod(shape))


def _row_xor(words: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError("row XOR requires a matrix")
    matrix = np.asarray(words, dtype=np.uint16).reshape(shape)
    delta = np.empty_like(matrix)
    delta[:, 0] = matrix[:, 0]
    delta[:, 1:] = np.bitwise_xor(matrix[:, 1:], matrix[:, :-1])
    return delta.reshape(-1)


def encode_xor_byte_shuffle_zlib1(words: np.ndarray, shape: tuple[int, ...]) -> bytes:
    return zlib.compress(_byte_shuffle(_row_xor(words, shape)), 1)


def decode_xor_byte_shuffle_zlib1(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    delta = _byte_unshuffle(zlib.decompress(payload), math.prod(shape)).reshape(shape)
    return np.bitwise_xor.accumulate(delta, axis=1).reshape(-1)


def _bitplane_pack(words: np.ndarray, block_words: int = 4096) -> bytes:
    flat = np.asarray(words, dtype=np.uint16).reshape(-1)
    chunks: list[bytes] = []
    for start in range(0, flat.size, block_words):
        block = flat[start : start + block_words]
        for bit in range(16):
            plane = ((block >> bit) & 1).astype(np.uint8, copy=False)
            chunks.append(np.packbits(plane, bitorder="little").tobytes())
    return b"".join(chunks)


def _bitplane_unpack(payload: bytes, count: int, block_words: int = 4096) -> np.ndarray:
    out = np.zeros(count, dtype=np.uint16)
    offset = 0
    for start in range(0, count, block_words):
        length = min(block_words, count - start)
        plane_bytes = (length + 7) // 8
        for bit in range(16):
            packed = np.frombuffer(payload, dtype=np.uint8, count=plane_bytes, offset=offset)
            plane = np.unpackbits(packed, bitorder="little", count=length)
            out[start : start + length] |= plane.astype(np.uint16) << bit
            offset += plane_bytes
    if offset != len(payload):
        raise ValueError("bit-plane payload has trailing bytes")
    return out


def encode_bitplane_zlib1(words: np.ndarray, shape: tuple[int, ...]) -> bytes:
    del shape
    return zlib.compress(_bitplane_pack(words), 1)


def decode_bitplane_zlib1(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return _bitplane_unpack(zlib.decompress(payload), math.prod(shape))


CODECS: dict[str, tuple[Callable, Callable]] = {
    "raw_zlib1": (encode_raw_zlib1, decode_raw_zlib1),
    "byte_shuffle_zlib1": (encode_byte_shuffle_zlib1, decode_byte_shuffle_zlib1),
    "xor_byte_shuffle_zlib1": (
        encode_xor_byte_shuffle_zlib1,
        decode_xor_byte_shuffle_zlib1,
    ),
    "bitplane_zlib1": (encode_bitplane_zlib1, decode_bitplane_zlib1),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_ambient(checkpoint: dict) -> list[dict]:
    optimizer = checkpoint.get("optimizer", {})
    optimizers = optimizer.get("optimizers", [])
    owners = []
    for owner_index, owner in enumerate(optimizers):
        states = owner.get("state", {}) if isinstance(owner, dict) else {}
        tensors = [
            (state_index, state["ambient_momentum"])
            for state_index, state in states.items()
            if isinstance(state, dict) and "ambient_momentum" in state
        ]
        if tensors:
            owners.append((owner_index, tensors))
    if len(owners) != 1:
        raise RuntimeError(f"expected one ambient-momentum owner, found {len(owners)}")
    owner_index, tensors = owners[0]
    result = []
    for ordinal, (state_index, tensor) in enumerate(tensors):
        if tensor.dtype != torch.float16 or tensor.ndim != 2:
            raise RuntimeError("ambient momentum is not an FP16 matrix")
        shape = tuple(int(x) for x in tensor.shape)
        if shape == (3072, 768):
            scope = "c_fc"
        elif shape == (768, 3072):
            scope = "c_proj"
        else:
            raise RuntimeError(f"unexpected ambient shape {shape}")
        result.append(
            {
                "name": f"optimizer{owner_index}.state{state_index}.{scope}",
                "scope": scope,
                "layer": ordinal // 2,
                "shape": shape,
                "tensor": tensor.contiguous(),
            }
        )
    return result


def _tensor_words(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy().view(np.uint16).reshape(-1)


def _block_entropies(words: np.ndarray, block_words: int = 4096) -> dict:
    values = []
    for start in range(0, words.size, block_words):
        _, counts = np.unique(words[start : start + block_words], return_counts=True)
        values.append(entropy_from_counts(counts))
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "blocks": len(values),
    }


def _histogram_report(counts: np.ndarray) -> dict:
    word_h = entropy_from_counts(counts)
    bits = bit_entropies(counts)
    return {
        "word_entropy_bits": word_h,
        "ideal_zero_order_bytes": int(math.ceil(float(counts.sum()) * word_h / 8.0)),
        "sign_entropy_bits": bits[15],
        "exponent_bit_entropies": bits[10:15],
        "mantissa_bit_entropies_low_to_high": bits[:10],
        "sum_marginal_bit_entropies": float(sum(bits)),
        "conditional_lower_mantissa_entropy": {
            str(prefix): conditional_lower_mantissa_entropy(counts, prefix)
            for prefix in (2, 4, 6, 8)
        },
    }


def run_audit(checkpoint_path: Path, checkpoint_sha256: str, plan_path: Path) -> dict:
    observed_sha = _sha256(checkpoint_path)
    if observed_sha != checkpoint_sha256:
        raise RuntimeError(f"checkpoint SHA mismatch: {observed_sha}")
    plan_sha = _sha256(plan_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tensors = _extract_ambient(checkpoint)
    if len(tensors) != 24:
        raise RuntimeError(f"expected 24 tensors, found {len(tensors)}")
    elements = sum(x["tensor"].numel() for x in tensors)
    raw_bytes = elements * 2
    if elements != 56623104 or raw_bytes != 113246208:
        raise RuntimeError("ambient state size does not match the registered source")

    global_counts = np.zeros(65536, dtype=np.int64)
    scope_counts = {"c_fc": np.zeros(65536, dtype=np.int64), "c_proj": np.zeros(65536, dtype=np.int64)}
    flat_xor_counts = np.zeros(65536, dtype=np.int64)
    row_xor_counts = np.zeros(65536, dtype=np.int64)
    matrix_rows = []
    source_digest = hashlib.sha256()
    for item in tensors:
        words = _tensor_words(item["tensor"])
        source_digest.update(words.astype("<u2", copy=False).tobytes())
        counts = word_counts(words)
        global_counts += counts
        scope_counts[item["scope"]] += counts
        flat_xor_counts += word_counts(np.bitwise_xor(words[1:], words[:-1]))
        row_xor_counts += word_counts(_row_xor(words, item["shape"]))
        matrix_rows.append(
            {
                "name": item["name"],
                "scope": item["scope"],
                "layer": item["layer"],
                "shape": list(item["shape"]),
                **_histogram_report(counts),
                "block_4096_entropy": _block_entropies(words),
            }
        )

    codec_results = {}
    for name, (encode, decode) in CODECS.items():
        payload_bytes = 0
        decoded_digest = hashlib.sha256()
        encode_seconds = 0.0
        decode_seconds = 0.0
        per_matrix = []
        for item in tensors:
            words = _tensor_words(item["tensor"])
            start = time.perf_counter()
            payload = encode(words, item["shape"])
            encode_seconds += time.perf_counter() - start
            start = time.perf_counter()
            recovered = decode(payload, item["shape"])
            decode_seconds += time.perf_counter() - start
            if not np.array_equal(words, recovered):
                raise RuntimeError(f"{name} failed exact round trip for {item['name']}")
            decoded_digest.update(recovered.astype("<u2", copy=False).tobytes())
            payload_bytes += len(payload)
            per_matrix.append({"name": item["name"], "payload_bytes": len(payload)})
        total_bytes = payload_bytes + CODEC_METADATA_BYTES
        codec_results[name] = {
            "payload_bytes": payload_bytes,
            "metadata_bytes": CODEC_METADATA_BYTES,
            "total_bytes": total_bytes,
            "bits_per_coordinate": total_bytes * 8.0 / elements,
            "compression_ratio_vs_raw": raw_bytes / total_bytes,
            "encode_seconds": encode_seconds,
            "decode_seconds": decode_seconds,
            "encode_gb_per_second": raw_bytes / encode_seconds / 1e9,
            "decode_gb_per_second": raw_bytes / decode_seconds / 1e9,
            "source_sha256": source_digest.hexdigest(),
            "decoded_sha256": decoded_digest.hexdigest(),
            "exact_roundtrip": decoded_digest.hexdigest() == source_digest.hexdigest(),
            "per_matrix": per_matrix,
        }

    target_bytes = 99090432
    passing = [
        name
        for name, row in codec_results.items()
        if row["exact_roundtrip"]
        and row["total_bytes"] <= target_bytes
        and row["decode_gb_per_second"] >= 1.0
    ]
    best_name = min(codec_results, key=lambda name: codec_results[name]["total_bytes"])
    best = codec_results[best_name]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema_version": SCHEMA,
        "status": "finished",
        "classification": (
            "PASS_BIT_EXACT_FP16_MOMENTUM_SPATIAL_CODEC_STAGE_A"
            if passing
            else "REJECT_BIT_EXACT_FP16_MOMENTUM_SPATIAL_CODEC_STAGE_A"
        ),
        "passed": bool(passing),
        "identity": {
            "source_commit": commit,
            "plan": str(plan_path),
            "plan_sha256": plan_sha,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": observed_sha,
        },
        "source": {
            "tensor_count": len(tensors),
            "elements": elements,
            "raw_bytes": raw_bytes,
            "source_word_stream_sha256": source_digest.hexdigest(),
            "transient_decoded_fp16_bytes": raw_bytes,
        },
        "entropy": {
            "all": _histogram_report(global_counts),
            "c_fc": _histogram_report(scope_counts["c_fc"]),
            "c_proj": _histogram_report(scope_counts["c_proj"]),
            "flat_adjacent_xor_word_entropy_bits": entropy_from_counts(flat_xor_counts),
            "row_xor_word_entropy_bits": entropy_from_counts(row_xor_counts),
            "matrices": matrix_rows,
        },
        "codecs": codec_results,
        "gate": {
            "target_momentum_bytes": target_bytes,
            "maximum_bits_per_coordinate": 14.0,
            "minimum_decode_gb_per_second": 1.0,
            "passing_codecs": passing,
            "best_size_codec": best_name,
            "best_size_bytes": best["total_bytes"],
            "best_size_bits_per_coordinate": best["bits_per_coordinate"],
            "best_decode_gb_per_second": best["decode_gb_per_second"],
        },
        "storage": {
            "common_compact_forward_and_feedback_bytes": 157427136,
            "best_total_training_bytes": 157427136 + best["total_bytes"],
            "best_training_reduction_factor": 452984832 / (157427136 + best["total_bytes"]),
            "working_fp16_training_reduction_factor": 1.6735479944415952,
        },
        "systems": {
            "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "gpu_used": False,
        },
        "decision": {
            "automatic_endpoint": false,
            "automatic_scale_up": false,
            "automatic_horizon_transfer": false,
            "automatic_sweep": false,
            "next_authorized_work": (
                "Implement only a GPU-native or overlapped exact codec and require exact checkpoint/resume plus >=20% MFU."
                if passing
                else "Retain FP16 ambient momentum as the measured state floor and analyze temporal/shared generative structure rather than another coordinate codec."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.checkpoint, args.checkpoint_sha256, args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": result["status"], "classification": result["classification"], "gate": result["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
