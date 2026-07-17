#!/usr/bin/env bash
# CPU-only launcher. Bootstrap separately; this script never installs packages.
set -euo pipefail
ROOT_DIR="${ROOT_DIR:-/root/userdata/MappingNetworks}"
REPO_DIR="${REPO_DIR:-$ROOT_DIR/latent-weight-lab}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-finewebedu}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/data/finewebedu_20b}"
TARGET_TOKENS="${TARGET_TOKENS:-20000000000}"; VAL_TOKENS="${VAL_TOKENS:-20000000}"; SHARD_TOKENS="${SHARD_TOKENS:-50000000}"
TOKENIZER_BACKEND="${TOKENIZER_BACKEND:-processpool}"; TOKENIZER_THREADS="${TOKENIZER_THREADS:-8}"; TOKENIZER_BATCH_DOCS="${TOKENIZER_BATCH_DOCS:-128}"; TOKENIZER_BATCH_BYTES="${TOKENIZER_BATCH_BYTES:-2097152}"; LOW_PRIORITY="${LOW_PRIORITY:-0}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"; NAME="${NAME:-sample-10BT}"; SPLIT="${SPLIT:-train}"; REVISION="${REVISION:-main}"; FAST_CONTINUATION="${FAST_CONTINUATION:-0}"
if [[ "$FAST_CONTINUATION" == 1 ]]; then NAME="sample-100BT"; fi
[[ -x "$VENV_DIR/bin/python" ]] || { echo "Missing venv; run examples/nanogpt/bootstrap_finewebedu_venv.sh first" >&2; exit 2; }
mkdir -p "$ROOT_DIR/.cache/finewebedu/hf" "$ROOT_DIR/.cache/finewebedu/pip" "$ROOT_DIR/.tmp/finewebedu"
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void
export HF_HOME="$ROOT_DIR/.cache/finewebedu/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PREP_HARD_EXIT="${PREP_HARD_EXIT:-1}"
export PIP_CACHE_DIR="$ROOT_DIR/.cache/finewebedu/pip" TMPDIR="$ROOT_DIR/.tmp/finewebedu"
cmd=("$VENV_DIR/bin/python" "$REPO_DIR/examples/nanogpt/prepare_finewebedu.py" --output-dir "$OUT_DIR" --target-tokens "$TARGET_TOKENS" --val-tokens "$VAL_TOKENS" --shard-tokens "$SHARD_TOKENS" --dataset "$DATASET" --name "$NAME" --split "$SPLIT" --revision "$REVISION" --tokenizer-backend "$TOKENIZER_BACKEND" --tokenizer-threads "$TOKENIZER_THREADS" --tokenizer-batch-docs "$TOKENIZER_BATCH_DOCS" --tokenizer-batch-bytes "$TOKENIZER_BATCH_BYTES" --streaming)
[[ -n "${STAGING_DIR:-}" ]] && cmd+=(--staging-dir "$STAGING_DIR")
[[ "${SMOKE:-0}" == 1 ]] && cmd+=(--smoke)
[[ "$FAST_CONTINUATION" == 1 ]] && cmd+=(--fast-continuation)
if [[ "$LOW_PRIORITY" == 1 ]]; then
  if command -v ionice >/dev/null; then exec nice -n 19 ionice -c 3 "${cmd[@]}"; else exec nice -n 19 "${cmd[@]}"; fi
fi
exec "${cmd[@]}"
