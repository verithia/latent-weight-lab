#!/usr/bin/env bash
# Y400-only detached launcher for one ladder config. It does not poll or notify.
# Usage (on Y400):
#   examples/nanogpt/launch_y400_ladder_detached.sh examples/nanogpt/configs/NAME.json 0 NAME [WORKSPACE_ROOT]
# The printed PID is also the setsid process-group leader; use it with the local
# watcher. The worker records its own terminal JSON status and exits with the
# exact training exit code.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

write_status() {
  local path="$1" state="$2" run_name="$3" config="$4" gpu="$5" pid="$6" pgid="$7" started="$8" finished="$9" exit_code="${10}" log="${11}"
  "${PYTHON_BIN}" - "$path" "$state" "$run_name" "$config" "$gpu" "$pid" "$pgid" "$started" "$finished" "$exit_code" "$log" <<'PY'
import json, os, sys
path, state, run_name, config, gpu, pid, pgid, started, finished, code, log = sys.argv[1:]
payload = {
    "state": state, "classification": "clean" if state == "finished" else "failed" if state == "failed" else "running",
    "run_name": run_name, "config": config, "gpu": int(gpu), "pid": int(pid), "pgid": int(pgid),
    "started_at": started, "finished_at": finished or None, "exit_code": None if code == "" else int(code), "log": log,
}

tmp = path + ".part"
with open(tmp, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, path)
PY
}

if [[ "${1:-}" == "--status-self-test" ]]; then
  [[ "$#" -eq 2 ]] || { echo "usage: $0 --status-self-test STATUS_JSON_PATH" >&2; exit 2; }
  mkdir -p "$(dirname "$2")"
  write_status "$2" finished status-self-test /tmp/status-self-test.json 0 1 1 "1970-01-01T00:00:00+00:00" "1970-01-01T00:00:01+00:00" 0 /tmp/status-self-test.log
  exit 0
fi

worker() {
  local config="$1" gpu="$2" run_name="$3" workspace="$4" log="$5" status="$6" pgid
  pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
  local started finished rc
  started="$(date -Is)"
  write_status "$status" running "$run_name" "$config" "$gpu" "$$" "$pgid" "$started" "" "" "$log"
  finish() {
    rc="$1"; finished="$(date -Is)"
    if [[ "$rc" -eq 0 ]]; then
      write_status "$status" finished "$run_name" "$config" "$gpu" "$$" "$pgid" "$started" "$finished" "$rc" "$log"
    else
      write_status "$status" failed "$run_name" "$config" "$gpu" "$$" "$pgid" "$started" "$finished" "$rc" "$log"
    fi
    exit "$rc"
  }
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1
  cd "$REPO_DIR"
  set +e
  "$PYTHON_BIN" -u -m examples.nanogpt.train --config "$config"
  rc=$?
  set -e
  finish "$rc"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  [[ "$#" -eq 6 ]] || { echo "internal worker argument error" >&2; exit 2; }
  worker "$@"
fi

[[ "$#" -ge 3 && "$#" -le 4 ]] || { echo "usage: $0 CONFIG_PATH GPU_INDEX RUN_NAME [WORKSPACE_ROOT]" >&2; exit 2; }
CONFIG_INPUT="$1"; GPU="$2"; RUN_NAME="$3"; WORKSPACE="${4:-/root/userdata/MappingNetworks}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU_INDEX must be a non-negative integer" >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash" >&2; exit 2; }
if [[ "$CONFIG_INPUT" = /* ]]; then CONFIG="$CONFIG_INPUT"; else CONFIG="$REPO_DIR/$CONFIG_INPUT"; fi
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }
RUN_DIR="$WORKSPACE/outputs/y400_ladder_runs"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/status"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOG="$RUN_DIR/logs/${RUN_NAME}_${STAMP}.log"
STATUS="$RUN_DIR/status/${RUN_NAME}_${STAMP}.json"
# Pass every value as a distinct argv element: no generated remote shell string.
setsid "$0" --worker "$CONFIG" "$GPU" "$RUN_NAME" "$WORKSPACE" "$LOG" "$STATUS" </dev/null >"$LOG" 2>&1 &
PID=$!
printf 'launched run=%s pid=%s pgid=%s\nlog=%s\nstatus=%s\n' "$RUN_NAME" "$PID" "$PID" "$LOG" "$STATUS"
