from __future__ import annotations

import argparse
import time

import torch

from latent_weight_lab import BlockFHT


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-power", type=int, default=5)
    parser.add_argument("--max-power", type=int, default=23)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("power,block_size,forward_ms,forward_backward_ms")
    for power in range(args.min_power, args.max_power + 1):
        block_size = 1 << power
        bfht = BlockFHT(block_size, size=block_size, layers=args.layers, seed=power).to(device)
        # The prototype CUDA extension currently supports float32.
        bfht.latent.data = bfht.latent.data.float()
        bfht.slice(0, min(block_size, 1024)).sum().backward()
        bfht.latent.grad = None
        synchronize(device)

        t0 = time.perf_counter()
        for _ in range(args.iters):
            y = bfht()
            synchronize(device)
        forward_ms = (time.perf_counter() - t0) * 1000 / args.iters

        t0 = time.perf_counter()
        for _ in range(args.iters):
            bfht.latent.grad = None
            y = bfht()
            y.square().mean().backward()
            synchronize(device)
        fb_ms = (time.perf_counter() - t0) * 1000 / args.iters
        print(f"{power},{block_size},{forward_ms:.3f},{fb_ms:.3f}")


if __name__ == "__main__":
    main()
