# nanoGPT Example

This example contains a minimal nanoGPT-style baseline and a full transformer-block `BlockFHT` variant.

## Safe FineWeb-Edu preparation (Y400)

Bootstrap once (never during preparation), then use a small CPU-only smoke run:

```bash
examples/nanogpt/bootstrap_finewebedu_venv.sh
SMOKE=1 TARGET_TOKENS=10000 VAL_TOKENS=1000 OUT_DIR=/root/userdata/MappingNetworks/data/finewebedu_smoke examples/nanogpt/prepare_y400_finewebedu.sh
```

The launcher sets `CUDA_VISIBLE_DEVICES=''`, `NVIDIA_VISIBLE_DEVICES=void`, and isolated
HF/pip/tmp caches at normal CPU/I/O priority for throughput. Set `LOW_PRIORITY=1` only
for conservative shared-host operation. Production runs stage under a unique
sibling directory and print its path; resume only with `STAGING_DIR=<printed path>`.
They refuse an existing output, promote atomically only after exact size/token/hash
validation, and retain immutable fixed-size committed shards plus durable state.
The Y400 launcher defaults `PREP_HARD_EXIT=1`: after all pools are explicitly closed
and final state/manifest writes are durable, prep flushes output and calls
`os._exit(0)` to avoid the observed Y400 `PyGILState_Release` interpreter-finalization
fatal. Use `--no-hard-exit` or `PREP_HARD_EXIT=0` for ordinary local debugging.

Preparation tokenizes ordered streaming documents in bounded batches (default: 128
documents or 2MiB UTF-8, one in-flight batch) through persistent backends: serial,
`ThreadPoolExecutor`, or spawn-based process pool. The default production backend is
`TOKENIZER_BACKEND=processpool` with 8 workers; workers load only GPT-2 tiktoken and
never datasets. It preserves serial `encode_ordinary(text)+[eot]` ordering and only
commits state at shard boundaries. The default benchmark compares serial with
processpool8 only, performs one startup/warmup batch, then measures 8 steady batches;
retain processpool unless a different backend is proven clean and faster. Threadpool
benchmarking is explicitly opt-in (`--benchmark-backends serial,threadpool,processpool`)
because it has shown fatal shutdown behavior. To measure
the next replay window without writing tokens, use the fixed ordered benchmark:

```bash
HF_ENDPOINT=https://hf-mirror.com /root/userdata/MappingNetworks/.venv-finewebedu/bin/python examples/nanogpt/prepare_finewebedu.py \
  --benchmark --staging-dir /path/to/printed.staging --dataset HuggingFaceFW/fineweb-edu \
  --name sample-10BT --streaming
```

The benchmark replays the durable cursor once, holds only one bounded window, and
prints startup seconds, steady docs/s, steady tokens/s, speedup versus serial, and
explicit worker lifecycle status; it does not alter output/state. Each backend is explicitly closed/joined (terminated
and joined on exceptions or SIGTERM); no interpreter-finalization cleanup is relied on.

### Opt-in fast continuation

`FAST_CONTINUATION=1` is for a partially prepared staging directory whose validation
split is complete. It preserves every committed shard, migrates the legacy
`sample-10BT` prefix into manifest segment metadata, and appends fresh deterministic
`sample-100BT` shuffle segments without replaying `documents_seen`. Every restart
closes the previous segment at its durable train-token boundary and starts a new seed;
therefore post-interruption continuation can overlap source content and is explicitly
not byte-identical to a single uninterrupted source stream.

```bash
# No-output mirror/source probe, with a pinned commit for paginated data listing.
HF_ENDPOINT=https://hf-mirror.com /root/userdata/MappingNetworks/.venv-finewebedu/bin/python examples/nanogpt/prepare_finewebedu.py \
  --probe --dataset HuggingFaceFW/fineweb-edu --name sample-100BT --revision <COMMIT> --probe-commit <COMMIT>

# No-write fresh sample-100BT tokenizer benchmark (does not replay staging cursor).
HF_ENDPOINT=https://hf-mirror.com /root/userdata/MappingNetworks/.venv-finewebedu/bin/python examples/nanogpt/prepare_finewebedu.py \
  --fast-benchmark --dataset HuggingFaceFW/fineweb-edu --name sample-100BT --revision <COMMIT> --streaming

# Resume appending to the printed staging directory; launcher forces sample-100BT.
FAST_CONTINUATION=1 STAGING_DIR=/path/to/finewebedu_20b.staging.<id> OUT_DIR=/root/userdata/MappingNetworks/data/finewebedu_20b \
  examples/nanogpt/prepare_y400_finewebedu.sh
```

For a detached run, start the launcher with `setsid`, then pass its PGID, staging
directory, and workspace to `watch_finewebedu_prep.sh`. Set `PREP_CALLBACK_URL` to
the host's documented Feishu callback endpoint; notifications explicitly disable
proxies. The watchdog reports 20/50/80% once, warns after 15 minutes without growth,
stops after 45 minutes except during verification, and stops at 192 GiB workspace
usage or 64 GiB free-space danger.

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
## Mandatory MFU launch gate

`launch_y400_ladder_detached.sh` refuses every training launch unless the exact
config sets `mfu_preflight_required: true` and `mfu_min_fraction: 0.20` (or
higher). Before it writes run provenance or starts a detached worker, it
requires the selected GPU to be exclusive, measures a real short training run
with the exact model/generator/regularizers, calibrates the same GPU with BF16
tensor-core GEMM, and writes a passing MFU certificate. The certificate SHA,
measured MFU, and denominator are recorded in the run provenance.

This is a hard rejection criterion. A synthetic benchmark, `nvidia-smi`
utilization, power draw, or a value copied from another accelerator cannot
substitute for a passing certificate. For a changed model, microbatch shape,
regularizer, compiler setting, or GPU, run the gate again before launch.

## Y400 dense queue worker

`y400_dense_queue_worker.py` is the single admission and callback owner for the
registered MAI-v3 dense queue. It launches only on an exclusive GPU, requires
enough workspace headroom for every active atomic checkpoint temporary plus an
8 GiB reserve, validates the exact resume checkpoint iteration, synchronizes a
clean `origin/main` checkout, and verifies the registered config and training
source hashes before submission. Each exact config still passes the launcher's
real >=20% MFU gate. The worker sends aggregate `@Codex` callbacks at 20%, 50%,
and terminal 100%; one 90-minute heartbeat is reset by any delivered progress
or submission callback.

The active queue contract is
`configs/y400_mai_v3_dense_queue.json`. Its deferred stages deliberately encode
the required order: finish/rank dense baselines, then materialize the 985M 5TPP
and 20TPP selections, then run attention-only full replacements. MLP work is
not submitted by this queue.

`multi_host_dense_queue_worker.py` supersedes the single-host service when both
Y400 and PRO6 are available. Its contract
`configs/mai_v3_dense_multi_host_queue.json` gives each scientific task one
global claim while describing host-specific variants. Y400 variants preserve
their exact path-bound resume checkpoints; PRO6 variants are explicitly fresh
host lineages with matching seeds, data manifest, recipe, token budget, and
fixed evaluation. The scheduler never labels a PRO6 fresh start as a Y400
resume, and never launches the same task on both hosts. One aggregate callback
stream covers submissions, 20%, 50%, terminal states, and the resettable
90-minute heartbeat across both machines. Admission enforces both the configured
workspace cap and live physical-filesystem free space, including active atomic
checkpoint budgets and the reserve, so a larger policy cap cannot mask a nearly
full host volume.
