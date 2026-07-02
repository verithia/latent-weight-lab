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

## Fused Linear Direction

The near-term inference kernel should start with the trivial on-the-fly version:

```text
for each generated BlockFHT weight block:
  load latent / signs
  apply BlockFHT transform
  multiply the generated weights by the matching input slice
  accumulate into output
```

This version is intentionally simple. It may regenerate or reload the same generated block across CTAs, but it gives a correctness baseline and exposes where time is spent before adding scheduling complexity.

The current fused forward prototype has already moved past output atomics to a row-block kernel, but it is still not a fully optimized CUDA GEMM replacement:

- It is not tensor-core/MMA based.
- It is not vectorized with `float4`, `half2`, or `bfloat162` loads.
- It performs scalar reductions inside a CTA.
- It requires `block_size % in_features == 0` to let one generated block cover whole output rows.
- It is useful as a non-materialized correctness/memory baseline, not as a speed target.

The benchmark should compare stacks of matrices, not a single tiny linear, because the system goal is reducing generated-weight residency across many MLP/attention matrices. A representative row-aligned stacked benchmark on RTX 4080 showed:

```text
tokens=1024, in=1024, out=4096, matrices=4, latent_dim=16384, FHT layers=2
fused row-block: 491.4 ms, peak 76.4 MiB
materialized:       6.2 ms, peak 92.4 MiB
```

This confirms the expected state: lower memory footprint, but unacceptable speed until the fused path uses vectorized loads and tensor-core style tiling/reuse.

The more ambitious shared-memory design is to split each latent-generated MLP/linear weight block into small sub-blocks that fit in SMEM and align with the hidden dimension. For an MLP written as:

```text
y[j] = sum_i x[i] * A[i, j]
```

the kernel can launch interleaved CTAs over sub-blocks of `A`:

```text
for each hidden-dim-aligned generated sub-block:
  generate the sub-block from its latent slice in shared memory
  run sign/FHT/sign locally in shared memory
  multiply x[i] values by the generated rows/columns
  accumulate partial y[j]
collect/reduce partial outputs across CTAs
```

The goal is to avoid full generated-weight HBM residency while getting useful reuse from shared memory. The constraint is that a Hadamard output normally depends on the full latent block, so the sub-block partition must be chosen carefully: either the latent generator itself is block-partitioned, or each sub-block corresponds to a complete smaller FHT domain. For now, prefer the trivial on-the-fly implementation first, then introduce this interleaved shared-memory scheme once the baseline fused path is measured.

## TileLang Direction

TileLang is worth evaluating for a portable implementation, especially for Metal/MPS later. It is not the first path for peak CUDA FHT performance because the hand-written CUDA FHT kernels are already close to memory bandwidth limits.
