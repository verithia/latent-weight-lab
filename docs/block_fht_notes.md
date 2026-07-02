# Block-FHT Notes

This repo continues the MappingNetworks line of work but is intentionally broader than one projection family.

`BlockFHT` is the first implemented implicit weight generator. Other structured projections can live beside it.

## Current CUDA Prototype

- Correctness-first implementation.
- One CTA handles one generated output block.
- Signs are computed on the fly with a 32-bit avalanche hash.
- One hash word supplies signs for 32 contiguous positions.
- Forward and backward regenerate the same signs from `(seed, block, layer, word)`.
- Current native CUDA path supports float32 and padded `block_size` from `2^5` through `2^23`.
- `block_size <= 16384` uses the original single-CTA shared-memory kernel.
- Larger blocks use a global-memory butterfly backend with one kernel launch per FHT stage. This is `O(n log n)` and scales in memory as `O(overlapped_blocks * block_size)`.

For the global backend, each FHT stage is a linear pass over the overlapped blocks:

```text
for each layer:
  for step in 1, 2, 4, ..., block_size / 2:
    launch linear butterfly pass over all pairs
  launch linear normalization pass
  launch linear sign pass
```

Backward applies the same normalized FHT passes in reverse sign order and accumulates into `grad_latent`. The transform cost is therefore `O(overlapped_blocks * block_size * log2(block_size))`; each CUDA kernel is a linear pass.

Selected RTX 4080 prototype timings, `layers=1`, output slice length `min(block_size, 1024)`:

| log2(block_size) | block_size | forward ms | backward ms | peak MiB |
|---:|---:|---:|---:|---:|
| 5 | 32 | 0.117 | 0.506 | 0 |
| 10 | 1,024 | 0.054 | 0.312 | 0 |
| 14 | 16,384 | 0.088 | 0.219 | 0 |
| 15 | 32,768 | 0.206 | 0.399 | 0 |
| 20 | 1,048,576 | 0.364 | 0.522 | 12 |
| 23 | 8,388,608 | 1.039 | 1.298 | 96 |

These are first-pass timings for the scalable prototype, not final optimized-kernel claims.

## Performance Direction

The best known CUDA FHT baseline is `Dao-AILab/fast-hadamard-transform`:

- PyTorch interface.
- fp32/fp16/bf16.
- dimensions up to `32768`.
- near-memcpy speed on A100 according to its README.

The large-block backend here is a scalable correctness/performance baseline. It is not expected to beat a Dao-style register/shared-memory implementation for dimensions like `32768`, but it keeps the algorithm tractable up to `2^23` while we port the optimized kernels.

## TileLang Direction

TileLang is worth evaluating for a portable implementation, especially for Metal/MPS later. It is not the first path for peak CUDA FHT performance because the hand-written CUDA FHT kernels are already close to memory bandwidth limits.
