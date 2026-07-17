#!/usr/bin/env python3
"""Bounded, dataset-free CUDA training-step throughput benchmark for GPT-350M."""
from __future__ import annotations

import sys
from pathlib import Path

# Run directly from any working directory on the remote host.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import os
import time

import torch

from examples.nanogpt.model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--timed-updates", type=int, default=1)
    parser.add_argument("--max-case-seconds", type=float, default=180.0)
    parser.add_argument("--cases", choices=("both", "baseline", "c_fc_blockfht"), default="both")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.warmup_updates <= 2:
        parser.error("--warmup-updates must be in 0..2")
    if not 1 <= args.timed_updates <= 2:
        parser.error("--timed-updates must be in 1..2")
    if not 1.0 <= args.max_case_seconds <= 300.0:
        parser.error("--max-case-seconds must be in 1..300")
    if not 1 <= args.gradient_accumulation_steps <= 8:
        parser.error("--gradient-accumulation-steps must be in 1..8")
    return args


def run_update(model: GPT, optimizer, tokens: torch.Tensor, targets: torch.Tensor, cache_weights: bool, gradient_accumulation_steps: int) -> float:
    optimizer.zero_grad(set_to_none=True)
    if cache_weights:
        model.prepare_block_fht_cache(dtype=torch.bfloat16)
    losses = []
    for _ in range(gradient_accumulation_steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(tokens, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {None if loss is None else loss.item()}")
        (loss / gradient_accumulation_steps).backward()
        losses.append(float(loss.detach()))
    if cache_weights:
        # Cached generated weights are leaf tensors; explicitly project their
        # gradients back to each BlockFHT latent before the optimizer update.
        model.flush_block_fht_cache()
    optimizer.step()
    return sum(losses) / len(losses)


def benchmark_case(name: str, block_fht: bool, args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(350_006)
    torch.cuda.manual_seed_all(350_006)
    config = GPTConfig(
        block_size=1024, vocab_size=50304, n_layer=24, n_head=16, n_embd=1024,
        dropout=0.0, bias=False, block_fht=block_fht,
        block_fht_targets=("mlp.c_fc",), block_fht_latent_ratio=0.01,
        block_fht_layers=2, block_fht_match_gpt_init=True,
    )
    model = GPT(config).cuda()
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type="cuda",
        optimizer="muon", muon_momentum=0.95, muon_ns_steps=5,
    )
    tokens = torch.randint(0, config.vocab_size, (32, 1024), device="cuda", dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (32, 1024), device="cuda", dtype=torch.long)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup_updates):
        run_update(model, optimizer, tokens, targets, cache_weights=block_fht, gradient_accumulation_steps=args.gradient_accumulation_steps)
    torch.cuda.synchronize()
    started = time.perf_counter()
    losses = []
    for _ in range(args.timed_updates):
        losses.append(run_update(model, optimizer, tokens, targets, cache_weights=block_fht, gradient_accumulation_steps=args.gradient_accumulation_steps))
        torch.cuda.synchronize()
        if time.perf_counter() - started > args.max_case_seconds:
            raise TimeoutError(f"{name} exceeded {args.max_case_seconds}s timed-case limit")
    elapsed = time.perf_counter() - started
    result = {
        "case": name, "tokens_per_second": (32 * 1024 * args.gradient_accumulation_steps * args.timed_updates) / elapsed,
        "elapsed_seconds": elapsed, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "finite_loss": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "losses": losses, "warmup_updates": args.warmup_updates, "timed_updates": args.timed_updates,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    del optimizer, model, tokens, targets
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cases = [("baseline", False), ("c_fc_blockfht", True)]
    if args.cases != "both":
        cases = [case for case in cases if case[0] == args.cases]
    results = [benchmark_case(name, block_fht, args) for name, block_fht in cases]
    print(json.dumps({"benchmark": "pro6_synthetic_gpt350m", "results": results}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
