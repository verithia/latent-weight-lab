#!/usr/bin/env python3
"""Run the preregistered standalone GPU gate for exact FP16 momentum coding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch

from examples.nanogpt.analyze_pair_vq_fp16_momentum_entropy import (
    _extract_ambient,
    _sha256,
    _tensor_words,
)
from examples.nanogpt.pair_vq_fp16_reserved_escape_cuda import (
    decode_reserved_escape,
    encode_reserved_escape,
)


SCHEMA = "mai_pair_vq_fp16_reserved_escape_cuda_benchmark_v1"
RAW_FP16_BYTES = 113246208
MAX_MOMENTUM_BYTES = 99090432
MAX_CODEC_SECONDS = 0.250
CONTAINER_METADATA_BYTES = 64 + 24 * 16


def _event_seconds(callable_) -> tuple[float, object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0, result


def _scope_sources(items: list[dict]) -> tuple[dict[str, torch.Tensor], dict]:
    sources = {}
    slices = {}
    for scope in ("c_fc", "c_proj"):
        selected = [item for item in items if item["scope"] == scope]
        sources[scope] = torch.cat(
            [item["tensor"].reshape(-1) for item in selected]
        ).cuda()
        cursor = 0
        for item in selected:
            stop = cursor + item["tensor"].numel()
            slices[item["name"]] = (scope, cursor, stop)
            cursor = stop
    return sources, slices


def _digest_reconstruction(
    items: list[dict], decoded: dict[str, torch.Tensor], slices: dict
) -> str:
    digest = hashlib.sha256()
    for item in items:
        scope, start, stop = slices[item["name"]]
        words = _tensor_words(decoded[scope][start:stop].cpu())
        digest.update(words.astype("<u2", copy=False).tobytes())
    return digest.hexdigest()


def benchmark_candidate(
    *,
    name: str,
    granularity: str,
    sources: dict[str, torch.Tensor],
    items: list[dict],
    slices: dict,
    source_digest: str,
) -> dict:
    def encode_all():
        return {
            scope: encode_reserved_escape(
                source,
                scope=scope,
                granularity=granularity,
                block_words=4096,
            )
            for scope, source in sources.items()
        }

    encoded = encode_all()
    decoded = {scope: decode_reserved_escape(state) for scope, state in encoded.items()}
    torch.cuda.synchronize()
    exact_gpu = all(
        torch.equal(decoded[scope].view(torch.uint8), source.view(torch.uint8))
        for scope, source in sources.items()
    )
    decoded_digest = _digest_reconstruction(items, decoded, slices)

    for _ in range(2):
        warm = encode_all()
        _ = {scope: decode_reserved_escape(state) for scope, state in warm.items()}
    torch.cuda.synchronize()

    encode_seconds = []
    latest = encoded
    for _ in range(5):
        seconds, latest = _event_seconds(encode_all)
        encode_seconds.append(seconds)
    decode_seconds = []
    for _ in range(5):
        seconds, _ = _event_seconds(
            lambda: {
                scope: decode_reserved_escape(state)
                for scope, state in latest.items()
            }
        )
        decode_seconds.append(seconds)

    persistent_tensor_bytes = sum(
        state.persistent_tensor_bytes for state in latest.values()
    )
    logical_registered_bytes = persistent_tensor_bytes + CONTAINER_METADATA_BYTES
    median_encode = statistics.median(encode_seconds)
    median_decode = statistics.median(decode_seconds)
    median_total = median_encode + median_decode
    return {
        "name": name,
        "granularity": granularity,
        "exact_gpu_roundtrip": exact_gpu,
        "decoded_word_stream_sha256": decoded_digest,
        "source_word_stream_sha256": source_digest,
        "persistent_tensor_bytes": persistent_tensor_bytes,
        "logical_registered_bytes": logical_registered_bytes,
        "bits_per_coordinate": logical_registered_bytes * 8 / 56623104,
        "exception_bytes": {
            scope: state.exception_high_bytes.numel()
            for scope, state in latest.items()
        },
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "median_encode_seconds": median_encode,
        "median_decode_seconds": median_decode,
        "median_encode_plus_decode_seconds": median_total,
        "raw_throughput_gb_per_second": RAW_FP16_BYTES / median_total / 1e9,
        "size_gate_passed": logical_registered_bytes <= MAX_MOMENTUM_BYTES,
        "speed_gate_passed": median_total <= MAX_CODEC_SECONDS,
        "exact_gate_passed": exact_gpu and decoded_digest == source_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    if _sha256(args.plan) != args.plan_sha256:
        raise RuntimeError("plan SHA-256 mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    items = _extract_ambient(checkpoint)
    if len(items) != 24:
        raise RuntimeError("expected 24 ambient momentum tensors")
    source_digest_object = hashlib.sha256()
    for item in items:
        source_digest_object.update(
            _tensor_words(item["tensor"]).astype("<u2", copy=False).tobytes()
        )
    source_digest = source_digest_object.hexdigest()
    if source_digest != "407c142e7e4098d3c3972d6b3b772fa7f913e0a057eec84181821ab07b1ea3e0":
        raise RuntimeError("frozen word-stream SHA-256 mismatch")
    sources, slices = _scope_sources(items)
    torch.cuda.reset_peak_memory_stats()
    candidates = [
        benchmark_candidate(
            name="scope_kfc16_kproj32_b4096",
            granularity="scope",
            sources=sources,
            items=items,
            slices=slices,
            source_digest=source_digest,
        ),
        benchmark_candidate(
            name="block_kfc16_kproj32_b4096",
            granularity="block",
            sources=sources,
            items=items,
            slices=slices,
            source_digest=source_digest,
        ),
    ]
    passing = [
        row
        for row in candidates
        if row["size_gate_passed"]
        and row["speed_gate_passed"]
        and row["exact_gate_passed"]
    ]
    result = {
        "schema_version": SCHEMA,
        "status": "finished",
        "classification": (
            "PASS_BIT_EXACT_RESERVED_ESCAPE_STANDALONE_GPU_GATE"
            if passing
            else "REJECT_BIT_EXACT_RESERVED_ESCAPE_STANDALONE_GPU_GATE"
        ),
        "passed": bool(passing),
        "identity": {
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "plan": str(args.plan),
            "plan_sha256": _sha256(args.plan),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "source_word_stream_sha256": source_digest,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "gate": {
            "maximum_momentum_bytes": MAX_MOMENTUM_BYTES,
            "maximum_encode_plus_decode_seconds": MAX_CODEC_SECONDS,
            "passing_candidates": [row["name"] for row in passing],
        },
        "candidates": candidates,
        "decision": {
            "next_authorized_work": (
                "Integrate only the fastest passing candidate and require exact one-step, nine-step, resume, and >=20% exact-config MFU gates."
                if passing
                else "Reject this GPU table-code implementation family and retain raw FP16 ambient momentum."
            ),
            "automatic_loss_endpoint": False,
            "automatic_scale_up": False,
            "automatic_horizon_transfer": False,
            "automatic_sweep": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"classification": result["classification"], "gate": result["gate"], "candidates": candidates}, sort_keys=True))


if __name__ == "__main__":
    main()
