# nanoGPT Example

This example contains a minimal nanoGPT-style baseline and a full transformer-block `BlockFHT` variant.

It expects GPT-2 BPE token memmaps:

```text
train.bin
val.bin
```

The remote FineWeb-Edu 2B dataset path used in current experiments is:

```text
/home/jerson/mapping_networks/nanoGPT/data/finewebedu_2b
```

## Baseline

```bash
PYTHONPATH=. python examples/nanogpt/train.py --config examples/nanogpt/configs/baseline_finewebedu_2b.json
```

Baseline smoke:

```bash
PYTHONPATH=. python examples/nanogpt/train.py --config examples/nanogpt/configs/baseline_finewebedu_smoke.json
```

## Full BlockFHT Smoke

This replaces all transformer block attention/MLP linear weights:

```text
attn.c_attn
attn.c_proj
mlp.c_fc
mlp.c_proj
```

with `BlockFHTLinear` using 1% latent ratio and 2 FHT layers.

```bash
PYTHONPATH=. python examples/nanogpt/train.py --config examples/nanogpt/configs/block_fht_full_finewebedu_smoke.json
```

## Full BlockFHT 2B

```bash
PYTHONPATH=. python examples/nanogpt/train.py --config examples/nanogpt/configs/block_fht_full_finewebedu_2b.json
```

The training path supports materialized per-optimizer-step BlockFHT weight caching via:

```json
"block_fht_cache_weights": true
```

This is the intended training path. The fused non-materialized path is currently inference-focused and exposed through `BlockFHTLinear.forward_fused()`.
