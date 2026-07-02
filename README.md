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
- CUDA extension prototype supports float32 and padded `block_size` from `2^5` through `2^23`.
- Small blocks use a shared-memory CTA kernel; large blocks use global-memory butterfly passes.
- The large-block backend is `O(n log n)` with each CUDA butterfly stage implemented as a linear pass.
- CUDA kernels are correctness/scalability-first, not the final optimized path.
- Next performance target is adapting Dao-AILab `fast-hadamard-transform` style kernels for fp16/bf16 and `block_size=32768`.

## Roadmap

The final systems goal is not just fewer trainable parameters. The goal is blockwise execution where a forward pass keeps HBM footprint close to latent size plus activations, instead of materializing full generated weight tensors in HBM.

Near-term verification path:

1. Run a short GPT-2 BlockFHT attempt with the same 2000-step FineWeb-Edu setup as the vanilla baseline.
2. Compare loss curve, wall time, peak VRAM, and stability against the vanilla `262M` token baseline.
3. Keep this as a functional check only; it is not a final scaling-law baseline.

CUDA/kernel optimization path:

1. Add fp16/bf16 native kernels. Current extension is float32-only.
2. Make `block_size=32768` fit in the per-SM working set. On RTX 4080-class Ada, opt-in shared memory is roughly 100 KiB per block and L1/shared capacity is finite; `32768` values are `128 KiB` in fp32 but `64 KiB` in fp16/bf16.
3. Port/adapt Dao-AILab style register/shared-memory FHT for `32768`, instead of relying on the global-memory fallback for GPT-sized per-matrix latents.
4. Benchmark sign generation variants inside the full transform. Current sign generation uses one 32-bit hash word for 32 contiguous signs.
5. Add the trivial fused `BlockFHTLinear` inference path first: generate/apply the BlockFHT transform on the fly, immediately multiply by the corresponding input slice, and accumulate output. This may still reload/regenerate work, but it establishes correctness and profiling baselines.
6. Add the shared-memory interleaved sub-block path for large MLP/linear layers. Split the latent/weight block into hidden-dimension-aligned sub-blocks that fit in shared memory, launch interleaved CTAs to generate each sub-block with FHT in shared memory, compute `x[i] * generated_rows_or_columns`, accumulate partial outputs, then reduce/collect across CTAs. This is the intended path to avoid full generated-weight HBM residency while increasing reuse inside each CTA.
7. Extend fusion to attention/MLP blocks so inference and training can operate block by block.

Memory target:

- Current verification path may still materialize generated weights in PyTorch modules.
- Intermediate CUDA path should hold one generated tile/block at a time.
- Final path should avoid full generated-weight HBM residency; persistent weight state should be latent vectors plus deterministic seeds.

Portability path:

- CUDA first, because FHT performance depends heavily on warp/register/shared-memory details.
- Evaluate TileLang after CUDA semantics are stable, especially for future Metal/MPS support.
- Expect a dedicated Metal or TileLang backend; CUDA kernels will not port directly to MPS.
