# Block-FHT Notes

This repo continues the MappingNetworks line of work but is intentionally broader than one projection family.

`BlockFHT` is the first implemented implicit weight generator. Other structured projections can live beside it.

## Current CUDA Prototype

- Correctness-first implementation.
- One CTA handles one generated output block.
- Signs are computed on the fly with a 32-bit avalanche hash.
- One hash word supplies signs for 32 contiguous positions.
- Forward and backward regenerate the same signs from `(seed, block, layer, word)`.
- Current native CUDA path supports float32 and `block_size <= 16384`.

## Performance Direction

The best known CUDA FHT baseline is `Dao-AILab/fast-hadamard-transform`:

- PyTorch interface.
- fp32/fp16/bf16.
- dimensions up to `32768`.
- near-memcpy speed on A100 according to its README.

The prototype here should be replaced with a Dao-style tiled/register/shared-memory implementation before serious benchmarking.

## TileLang Direction

TileLang is worth evaluating for a portable implementation, especially for Metal/MPS later. It is not the first path for peak CUDA FHT performance because the hand-written CUDA FHT kernels are already close to memory bandwidth limits.
