from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from latent_weight_lab import block_fht_linear_forward, block_fht_slice


def synchronize() -> None:
    torch.cuda.synchronize()


def bench(
    x: torch.Tensor,
    latent: torch.Tensor,
    resident_weights: list[torch.Tensor] | None,
    out_features: int,
    fht_layers: int,
    mode: str,
    matrices: int,
    iters: int,
) -> tuple[float, float]:
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        outputs = []
        for index in range(len(resident_weights) if resident_weights is not None else matrices):
            seed = 1000 + index
            if mode == "fused_shared_latent":
                outputs.append(block_fht_linear_forward(x, latent, out_features, fht_layers, seed))
            elif mode == "resident_dense":
                assert resident_weights is not None
                weight = resident_weights[index]
                outputs.append(F.linear(x, weight))
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
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--matrices", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=16384)
    parser.add_argument("--fht-layers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--mode", choices=["fused_shared_latent", "resident_dense"], required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(123)
    x = torch.randn(args.tokens, args.in_features, device="cuda")
    latent = torch.randn(args.latent_dim, device="cuda")
    resident_weights = None
    if args.mode == "resident_dense":
        resident_weights = []
        weight_size = args.in_features * args.out_features
        for index in range(args.matrices):
            flat = block_fht_slice(latent, weight_size, args.fht_layers, 1000 + index, 0, weight_size)
            resident_weights.append(flat.view(args.out_features, args.in_features).detach().clone())
    synchronize()

    for _ in range(3):
        bench(x, latent, resident_weights, args.out_features, args.fht_layers, args.mode, args.matrices, 1)

    print("mode,matrices,tokens,in_features,out_features,latent_dim,ms,peak_mib")
    ms, peak_mib = bench(x, latent, resident_weights, args.out_features, args.fht_layers, args.mode, args.matrices, args.iters)
    print(
        f"{args.mode},{args.matrices},{args.tokens},{args.in_features},{args.out_features},"
        f"{args.latent_dim},{ms:.3f},{peak_mib:.1f}"
    )


if __name__ == "__main__":
    main()
