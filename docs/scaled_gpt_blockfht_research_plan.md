# Scaled GPT/BlockFHT Research Plan

Created: 2026-07-10

## Executive decision

The current 124M nanoGPT-style experiments are useful for implementation debugging and early screening, but they are not adoption-grade evidence. They run at very low tokens per active parameter (TPP), use aggressive learning rates, and underutilize H100s. The next research track should move to an open-baseline-driven 0.3B-1B ladder with short 5-10 TPP diagnostics, longer 40 TPP ablations, and rare 100-200 TPP confirmations only after a candidate is already strong.

Primary local source references:

- `.slim/clonedeps/repos/KellerJordan__modded-nanogpt/` — Muon/fused-kernel GPT speedrun source for short diagnostic recipe ideas.
- `.slim/clonedeps/repos/k-luka__GPT/` — cleaner ~280M GPT Muon/fused-projection ablation harness, best source for 40-200 TPP methodology.
- `.slim/clonedeps/repos/kcc-lion__muown/` — optimizer/schedule breadth and longer-horizon Muon-family harness.
- `.slim/clonedeps/repos/huggingface__pytorch-image-models/` — `timm/optim/muon.py`, maintained Muon implementation for routing/fallback ideas.

## Core principle

Report TPP using active/materialized model parameters, not latent trainable parameters. BlockFHT/Mapping style methods reduce trainable parameters, but the forward network still has dense active parameters and dense compute.

## Current status and lessons

### 124M anchor

- Existing 12L/768/12h vanilla has about 124M active params.
- Current 1B-token runs are only about 8-10 TPP for this scale.
- Current 4B-token runs are only about 32-40 TPP.
- Prior results are screening evidence, not final scaling-ladder evidence.

### Speed status

On Y400 H100 after cached-path speed patches:

- vanilla 124M: about `1058-1070 ms/iter` at `131,072 tok/iter`;
- BlockFHT FHT-LoRA before fixes: about `1625-1687 ms/iter`;
- after cached headwise fusion: about `1186-1195 ms/iter`;
- after cached c_proj-lowrank fusion: about `1141-1158 ms/iter`.

BlockFHT overhead is now around 8-10% versus vanilla for the tested 124M config. This is acceptable for larger speed probes. The remaining bottleneck is mostly general PyTorch/model utilization, not solely BlockFHT decode/flush.

### Current true-structure result

`y400_true_struct_1b_01_tied_cfc_skip_vec` finished cleanly with best val `3.5980` at step `7000`, worse than same-step/local attention-only control `3.3973` by `+0.2007` CE. This standalone tied-c_fc skip is rejected unless later combined interactions change the picture.

## Baseline acceptance criteria

A new baseline recipe is accepted only if it is open, stable at 5-10 TPP, fast enough before long runs, scaling-sane from 350M to 690M, and optimizer-validated. Muon must match or beat AdamW on the accepted 350M short diagnostic before becoming default.

## Model ladder

Use head dimension 64, block size 1024, tied embeddings, dropout 0 unless deliberately ablated.

| Tier | Config | Estimated active params | Role |
|---|---:|---:|---|
| legacy | 12L / 768 / 12h | ~124M | implementation and speed sanity only |
| 350m | 24L / 1024 / 16h | ~355M | primary recipe and method search |
| 690m | 32L / 1280 / 20h | ~690M | scaling confirmation |
| 1b | 32L / 1536 / 24h | ~1B | final confirmation only |

Generated config templates:

- `examples/nanogpt/make_y400_scaled_ladder_configs.py`
- `examples/nanogpt/y400_scaled_ladder_queue.tsv`
- `examples/nanogpt/configs/y400_speed_350m_*.json`
- `examples/nanogpt/configs/y400_350m_*tpp.json`
- `examples/nanogpt/prepare_finewebedu.py` and `prepare_y400_finewebedu.sh` for scaled local data preparation.
- `examples/nanogpt/write_data_manifest.py` for hashing/counting prepared `.bin` files.

Important data target: use `/root/userdata/MappingNetworks/data/finewebedu_20b` for real scaled configs. The helper script targets 20B train tokens plus validation by default. A 350M 40TPP run needs about 14B tokens, so the older `/root/userdata/MappingNetworks/data/finewebedu_2b` shard is only suitable for speed probes/smoke tests unless cycling is deliberate.

Current cap: Y400 `/root/userdata/MappingNetworks` working directory cap is now `256G`. The workspace was about `63G` before expanded-data prep, so the 20B-token uint16 target is now permitted, but still check `du -sh /root/userdata/MappingNetworks` before and after preparation.

## Token budgets

| Params | 5 TPP | 10 TPP | 40 TPP | 100 TPP | 200 TPP |
|---:|---:|---:|---:|---:|---:|
| 350M | 1.75B | 3.5B | 14B | 35B | 70B |
| 690M | 3.5B | 6.9B | 28B | 69B | 138B |
| 1B | 5B | 10B | 40B | 100B | 200B |

Policy:

- 5 TPP: speed, optimizer, and LR diagnostics.
- 10 TPP: recipe acceptance.
- 40 TPP: main method ablation budget.
- 100 TPP: finalist confirmation only.
- 200 TPP: avoid unless strategically justified.

## Training recipe starting point

Shared defaults:

```text
block_size = 1024
dropout = 0
weight_decay = 0.1
grad_clip = 1.0
dtype = bfloat16
compile = true for speed probes after compile sanity passes
warmup = 1-2% of total steps
schedule = cosine to min_lr = 0.1 * lr
eval cadence = token-budget based
checkpointing = disabled for speed probes; sparse milestones for long runs
```

Initial LR brackets:

- AdamW 350M: `2e-4`, `3e-4`, `4e-4`
- AdamW 690M: `1.5e-4`, `2e-4`, `3e-4`
- AdamW 1B: `1.2e-4`, `1.8e-4`, `2.5e-4`
- Muon matrix LR: `1.2e-3`, `1.8e-3`, `2.4e-3`
- Muon AdamW fallback LR scale: `0.1`, `0.2`, `0.3`

The code now supports `muon_adamw_lr_scale`; default is `1.0` for backward compatibility, and scaled-ladder templates use `0.2`.

## Optimizer plan

Current local Muon is minimal. Immediate code changes completed:

- Muon matrix params are separated from AdamW fallback params in `GPT.configure_optimizers`.
- AdamW fallback LR can now be scaled by `muon_adamw_lr_scale`.
- Optimizer logs include fallback LR scale and actual fallback LR.

Next optimizer work:

1. Add detailed routing logs: matrix param count, other param count, matrix numel, other numel.
2. Run AdamW vs Muon 350M 5 TPP bracket.
3. Only port more `timm` Muon behavior if profiling shows optimizer overhead or instability.
4. Do not switch Muon implementation in the middle of a method ablation; rerun vanilla after optimizer changes.

## Fused-op and speed plan

Priority order:

1. Cached-path fusion: already headwise and c_proj-lowrank; next inspect remaining split/headwise QKV or grouped paths.
2. `torch.compile`: test vanilla 350M compile before BlockFHT compile.
3. Fused optimizer paths: AdamW already uses `fused=True` when available; monitor Python Muon at 350M/690M.
4. Attention backend: compare SDPA/FlashAttention-compatible path if not already active.
5. Dataloader: adopt pinned/prefetch worker model from `muown` if CPU/data time grows.
6. Triton/FP8 from `modded-nanogpt`: high-risk, only after clean BF16 baseline is accepted.

Defer custom CUDA BlockFHT kernel work until prepare/flush or live fused path is a confirmed bottleneck at 350M+.

## Speed gate protocol

Before any 5+ TPP run:

1. Warm up 100-200 iterations.
2. Time 300-500 steady iterations.
3. Disable checkpointing.
4. Keep eval separated or disabled for headline timing.
5. Run no-profile headline timing and perf-profile attribution separately.

Report ms/iter, tokens/s, estimated MFU, peak VRAM, fwd/bwd ms, optimizer ms, data ms, BlockFHT prepare/flush ms, generated params, and latent params.

Minimum speed probes:

- 350M vanilla AdamW;
- 350M vanilla Muon;
- 350M BlockFHT cached.

Stop if vanilla MFU is unexpectedly low, BlockFHT overhead exceeds 15-20%, optimizer time dominates, or VRAM margin is under 10-15%.

Helper for post-run reporting: `examples/nanogpt/report_tier_perf.py` parses train logs with `perf` or `iter` timing lines and reports tokens/s, estimated train TFLOP/s, and MFU versus an H100 BF16 peak of `989 TFLOP/s`.

Current vanilla performance status:

| Tier | Params | H100 BF16 theoretical max tokens/s (`6N` estimate) | Measured tokens/s | Estimated TFLOP/s | Estimated MFU | Status |
|---|---:|---:|---:|---:|---:|---|
| legacy 12L/768 | 124,475,904 | 1,324,219 | 123,249 | 92.05 | 9.31% | measured from completed 124M vanilla perf log |
| 350m 24L/1024 | 354,871,296 | 464,488 | pending | pending | pending | speed config prepared; GPU busy |
| 690m 32L/1280 | 695,380,480 | 237,040 | pending | pending | pending | speed config prepared; GPU busy |
| 1b 32L/1536 | 985,451,520 | 167,267 | pending | pending | pending | speed config prepared; GPU busy |

The theoretical max is a rough dense-training bound using `6 * active_params * tokens/s`. Real MFU will be lower due to attention, optimizer, data, launch overhead, and nonideal model shapes. The measured legacy value shows the current stack is still far from H100 peak even before scaling.

## Experiment stages

### Stage 0 — finish current true-structure queue

Do not poll logs. Reconcile hook-driven terminal callbacks only. Treat results as 124M/low-TPP screening evidence.

### Stage 1 — 350M speed gate

Use generated speed configs:

- `y400_speed_350m_adamw`
- `y400_speed_350m_muon`
- `y400_speed_350m_blockfht_fullattn_cproj`

### Stage 2 — 350M optimizer/recipe diagnostic at 5 TPP

Run AdamW and Muon LR brackets. Choose baseline optimizer by stability, loss, and wall-clock-to-loss.

### Stage 3 — 350M recipe acceptance at 10 TPP

Run the winning optimizer/recipe. Freeze baseline only if stable and fast enough.

### Stage 4 — 350M BlockFHT diagnostics at 5-10 TPP

Compare only strongest candidates: best full-attn replacement, best c_proj structure if any true-structure run beats attention-only control, and one paper-method Mapping Loss/alignment variant if implemented.

### Stage 5 — 350M 40 TPP method ablation

Run at most 2-3 candidates plus vanilla. This is the main ablation tier.

### Stage 6 — 690M scale confirmation

Run vanilla and best BlockFHT candidate at 5-10 TPP. Proceed only if scaling trend is sane.

### Stage 7 — 1B final candidate

Run 1B 5-10 TPP first. A 1B 40 TPP run is reserved for a candidate with clear 350M/690M evidence.

## Candidate interpretation rules

Reject a method if it loses to same-step vanilla by a large stable gap at 10 TPP, only wins at 1 TPP and degrades later, requires high LR/tiny TPP to look good, worsens overhead beyond 20%, or repeats a tiny dataset while being reported as long-data evidence.

Promote a method if it is stable at 5-10 TPP, narrows or beats the vanilla gap at 40 TPP, has acceptable speed overhead, scales from 350M to 690M, and has clear source/config/dataset traceability.

## Paper-method alignment track

The GPT trials have not fully tested the original paper's Mapping Loss/manifold-alignment recipe. A faithful scaled test should include latent perturbation stability on logits, finite-difference or Hutchinson smoothness, meaningful latent/mapping alignment or norm-control, coefficient scheduling/adaptive weights, and same-model same-TPP baseline comparison. Do this only after the 350M vanilla baseline is accepted.

## Required traceability per run

Each run must record run name, git commit, remote source hashes if remote is not a git checkout, config path and SHA256, dataset manifest, active/trainable/generated/latent params, tokens per iter, planned tokens, planned TPP, optimizer routing and LRs, eval cadence, checkpoint policy, and speed-gate result.

## Immediate next actions

1. Finish/reconcile current Y400 true-structure queue by callbacks only.
2. Sync scaled-ladder configs only after current queue frees GPUs.
3. Run 350M speed probes, not long training.
4. Decide AdamW vs Muon at 350M 5 TPP.
5. Only after an accepted 350M vanilla recipe, schedule BlockFHT diagnostics.
