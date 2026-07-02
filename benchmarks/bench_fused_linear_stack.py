from __future__ import annotations

import argparse
import time

import torch

from latent_weight_lab import BlockFHTLinear


def synchronize() -> None:
    torch.cuda.synchronize()


def bench(layers: list[BlockFHTLinear], x: torch.Tensor, mode: str, iters: int) -> tuple[float, float]:
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        outputs = []
        for layer in layers:
            if mode == "fused":
                outputs.append(layer.forward_fused(x))
            elif mode == "materialized":
                outputs.append(layer(x))
            else:
                raise ValueError(mode)
        y = torch.stack([out.float().mean() for out in outputs]).sum()
    synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / iters
    peak_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    # Keep y live until after synchronization so the compiler/runtime cannot trivially discard work.
    _ = float(y)
    return elapsed_ms, peak_mib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--matrices", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=4096)
    parser.add_argument("--fht-layers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(123)
    x = torch.randn(args.tokens, args.in_features, device="cuda")
    layers = [
        BlockFHTLinear(
            args.in_features,
            args.out_features,
            latent_dim=args.latent_dim,
            layers=args.fht_layers,
            seed=1000 + index,
        ).cuda()
        for index in range(args.matrices)
    ]

    for _ in range(3):
        bench(layers, x, "fused", 1)
        bench(layers, x, "materialized", 1)

    print("mode,matrices,tokens,in_features,out_features,latent_dim,ms,peak_mib")
    for mode in ["fused", "materialized"]:
        ms, peak_mib = bench(layers, x, mode, args.iters)
        print(
            f"{mode},{args.matrices},{args.tokens},{args.in_features},{args.out_features},"
            f"{args.latent_dim},{ms:.3f},{peak_mib:.1f}"
        )


if __name__ == "__main__":
    main()
