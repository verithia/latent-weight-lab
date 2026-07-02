# latent-weight-lab

Experiments for latent-generated neural network weights, continuing the MappingNetworks work.

Current prototype: `BlockFHT`, an implicit block Fast Hadamard Transform mapping with deterministic on-the-fly sign flips.

```python
import torch
from latent_weight_lab import BlockFHT

latent = torch.nn.Parameter(torch.randn(1024))
bfht = BlockFHT(latent, size=108_618, layers=3, seed=123)

# Computes only the requested generated coordinates.
y = bfht.slice(1000, 2000)
```

You can also let the module create its own latent parameter:

```python
bfht = BlockFHT(1024, size=108_618, layers=3, seed=123)
```

## Sign Generation

Signs are stateless and deterministic. One 32-bit hash word is generated for 32 contiguous positions:

```text
word = pos >> 5
bit = pos & 31
bits = lowbias32(mix(seed, block, layer, word))
sign = ((bits >> bit) & 1) ? +1 : -1
```

This avoids storing sign matrices and avoids one hash per output coordinate.

## Status

- CPU/PyTorch fallback supports autograd.
- CUDA extension prototype supports float32 and `block_size <= 16384`.
- CUDA kernel is correctness-first, not the final optimized path.
- Next performance target is adapting Dao-AILab `fast-hadamard-transform` style kernels for fp16/bf16 and `block_size=32768`.
