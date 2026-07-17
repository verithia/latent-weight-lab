from __future__ import annotations

import torch

from latent_weight_lab import BlockFHTLinear


def main() -> None:
    torch.manual_seed(0)
    layer = BlockFHTLinear(1024, 4096, latent_dim=16384, layers=2, seed=123).cuda().half()
    x = torch.randn(1, 1024, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    y = layer.forward_fused(x, weight_scale=0.5)
    torch.cuda.synchronize()
    print(float(y.float().mean()))


if __name__ == "__main__":
    main()
