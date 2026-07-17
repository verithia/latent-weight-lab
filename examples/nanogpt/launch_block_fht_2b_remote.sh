#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/mapping_networks"
REPO="$BASE/latent-weight-lab"
VENV="$BASE/.venv-gpt2"
LOG_DIR="$BASE/logs"
LATEST="$LOG_DIR/latent_weight_lab_nanogpt_block_fht_full_2b_latest"
RUN_DIR="$BASE/runs"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_LOG="$LOG_DIR/latent_weight_lab_nanogpt_block_fht_full_2b_${TS}.log"
SMI_LOG="$LOG_DIR/latent_weight_lab_nanogpt_block_fht_full_2b_${TS}_smi.log"
STATUS="$LOG_DIR/latent_weight_lab_nanogpt_block_fht_full_2b_${TS}.status"

mkdir -p "$LOG_DIR" "$RUN_DIR"
printf "%s\n%s\n%s\n" "$TRAIN_LOG" "$SMI_LOG" "$STATUS" > "$LATEST"
printf "stage=launching\nstarted=%s\ntrain_pid=\nsmi_pid=\ntrain_exit_code=\n" "$TS" > "$STATUS"

(
  set +e
  while true; do
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "$SMI_LOG"
    sleep 60
  done &
  SMIPID=$!

  cd "$REPO" || exit 111
  # shellcheck source=/dev/null
  source "$VENV/bin/activate" || exit 112
  export PYTHONPATH=.
  export TORCH_CUDA_ARCH_LIST=8.9
  export HTTP_PROXY=http://127.0.0.1:7897
  export HTTPS_PROXY=http://127.0.0.1:7897

  python -u examples/nanogpt/train.py --config examples/nanogpt/configs/block_fht_full_finewebedu_2b.json > "$TRAIN_LOG" 2>&1 &
  TRAINPID=$!
  printf "stage=running\nstarted=%s\ntrain_pid=%s\nsmi_pid=%s\ntrain_exit_code=\n" "$TS" "$TRAINPID" "$SMIPID" > "$STATUS"

  wait "$TRAINPID"
  CODE=$?
  kill "$SMIPID" >/dev/null 2>&1 || true
  if [ "$CODE" -eq 0 ]; then
    STAGE=done
  else
    STAGE=failed
  fi
  printf "stage=%s\nstarted=%s\nfinished=%s\ntrain_pid=%s\nsmi_pid=%s\ntrain_exit_code=%s\n" \
    "$STAGE" "$TS" "$(date -u +%Y%m%dT%H%M%SZ)" "$TRAINPID" "$SMIPID" "$CODE" > "$STATUS"
) >/dev/null 2>&1 &

echo "train_log=$TRAIN_LOG"
echo "smi_log=$SMI_LOG"
echo "status=$STATUS"
echo "supervisor_pid=$!"
