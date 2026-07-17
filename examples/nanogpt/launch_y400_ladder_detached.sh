#!/usr/bin/env bash
# Y400-only detached launcher for one ladder config. It does not poll or notify.
# Usage (on Y400):
#   examples/nanogpt/launch_y400_ladder_detached.sh [--resume] [--foreground] examples/nanogpt/configs/NAME.json 0 NAME [WORKSPACE_ROOT]
# The printed PID is also the setsid process-group leader; use it with the local
# watcher. The worker records its own terminal JSON status and exits with the
# exact training exit code.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROVENANCE_SCHEMA_VERSION="y400_experiment_provenance_v1"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_clean_git_checkout() {
  git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || { echo "refusing launch: source directory is not a Git worktree" >&2; exit 2; }
  [[ -z "$(git -C "$REPO_DIR" status --porcelain)" ]] \
    || { echo "refusing launch: source worktree is dirty; commit or discard changes first" >&2; exit 2; }
  GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
  GIT_ORIGIN="$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || true)"
  [[ -n "$GIT_ORIGIN" ]] \
    || { echo "refusing launch: Git worktree has no origin remote" >&2; exit 2; }
}

write_status() {
  local path="$1" state="$2" run_name="$3" config="$4" gpu="$5" pid="$6" pgid="$7" started="$8" finished="$9" exit_code="${10}" log="${11}" provenance="${12}" provenance_sha256="${13}"
  "${PYTHON_BIN}" - "$path" "$state" "$run_name" "$config" "$gpu" "$pid" "$pgid" "$started" "$finished" "$exit_code" "$log" "$provenance" "$provenance_sha256" <<'PY'
import json, os, sys
path, state, run_name, config, gpu, pid, pgid, started, finished, code, log, provenance, provenance_sha256 = sys.argv[1:]
payload = {
    "state": state, "classification": "clean" if state == "finished" else "failed" if state == "failed" else "running",
    "run_name": run_name, "config": config, "gpu": int(gpu), "pid": int(pid), "pgid": int(pgid),
    "started_at": started, "finished_at": finished or None, "exit_code": None if code == "" else int(code), "log": log,
    "provenance": None if not provenance else {"path": provenance, "sha256": provenance_sha256},
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

write_provenance() {
  local config="$1" config_archive="$2" provenance="$3" run_id="$4" run_name="$5" gpu="$6" workspace="$7" launched_at="$8" git_commit="$9" git_origin="${10}" resume_mode="${11}" foreground_mode="${12}"
  "${PYTHON_BIN}" - "$config" "$config_archive" "$provenance" "$run_id" "$run_name" "$gpu" "$workspace" "$launched_at" "$git_commit" "$git_origin" "$REPO_DIR" "$SCRIPT_DIR" "$PYTHON_BIN" "$PROVENANCE_SCHEMA_VERSION" "$resume_mode" "$foreground_mode" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

(
    config_path, config_archive, provenance_path, run_id, run_name, gpu, workspace,
    launched_at, git_commit, git_origin, repo_dir, launcher_dir, python_bin, schema_version, resume_mode, foreground_mode,
) = sys.argv[1:]

source = Path(config_path).resolve()
archive = Path(config_archive)
provenance = Path(provenance_path)
raw_config = source.read_bytes()
config = json.loads(raw_config)
if not isinstance(config, dict):
    raise SystemExit("refusing launch: config must be a JSON object")
data_dir = config.get("data_dir")
if not isinstance(data_dir, str) or not data_dir:
    raise SystemExit("refusing launch: config has no data_dir for dataset provenance")
manifest = (Path(data_dir) / "manifest.json").resolve()
if not manifest.is_file():
    raise SystemExit(f"refusing launch: dataset manifest is absent: {manifest}")
manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
expected_manifest_sha256 = config.get("data_manifest_sha256")
if not isinstance(expected_manifest_sha256, str) or expected_manifest_sha256 != manifest_sha256:
    raise SystemExit("refusing launch: config data_manifest_sha256 does not match dataset manifest")

def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()

atomic_write(archive, raw_config)
config_sha256 = hashlib.sha256(raw_config).hexdigest()
entrypoint = [python_bin, "-u", "-m", "examples.nanogpt.train"]
command = [*entrypoint, "--config", str(archive.resolve())]
if resume_mode == "1":
    command.extend(["--init-from", "resume"])
payload = {
    "schema_version": schema_version,
    "run_id": run_id,
    "run_name": run_name,
    "launched_at": launched_at,
    "repository": {
        "root": str(Path(repo_dir).resolve()),
        "origin": git_origin,
        "git_commit": git_commit,
        "worktree_dirty": False,
    },
    "entrypoint": entrypoint,
    "command": command,
    "working_directory": str(Path(repo_dir).resolve()),
    "launcher": str((Path(launcher_dir) / "launch_y400_ladder_detached.sh").resolve()),
    "config": {
        "source_path": str(source),
        "archive_path": str(archive.resolve()),
        "sha256": config_sha256,
    },
    "dataset_manifest": {"path": str(manifest), "sha256": manifest_sha256},
    "runtime": {
        "workspace": str(Path(workspace).resolve()),
        "cuda_visible_devices": str(gpu),
        "python": str(Path(python_bin).resolve()) if Path(python_bin).exists() else python_bin,
        "launch_mode": "foreground" if foreground_mode == "1" else "setsid_detached",
    },
}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
atomic_write(provenance, encoded)
print(hashlib.sha256(encoded).hexdigest())
PY
}

if [[ "${1:-}" == "--status-self-test" ]]; then
  [[ "$#" -eq 2 ]] || { echo "usage: $0 --status-self-test STATUS_JSON_PATH" >&2; exit 2; }
  mkdir -p "$(dirname "$2")"
  write_status "$2" finished status-self-test /tmp/status-self-test.json 0 1 1 "1970-01-01T00:00:00+00:00" "1970-01-01T00:00:01+00:00" 0 /tmp/status-self-test.log "" ""
  exit 0
fi

if [[ "${1:-}" == "--provenance-self-test" ]]; then
  [[ "$#" -eq 2 ]] || { echo "usage: $0 --provenance-self-test CONFIG_PATH" >&2; exit 2; }
  CONFIG_INPUT="$2"
  if [[ "$CONFIG_INPUT" = /* ]]; then CONFIG="$CONFIG_INPUT"; else CONFIG="$REPO_DIR/$CONFIG_INPUT"; fi
  [[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }
  require_clean_git_checkout
  SELFTEST_DIR="${TMPDIR:-/tmp}/y400-provenance-selftest-$$"
  CONFIG_ARCHIVE="$SELFTEST_DIR/config.json"
  PROVENANCE="$SELFTEST_DIR/provenance.json"
  LAUNCHED_AT="$(date -Is)"
  PROVENANCE_SHA256="$(write_provenance "$CONFIG" "$CONFIG_ARCHIVE" "$PROVENANCE" provenance-self-test provenance-self-test 0 /tmp "$LAUNCHED_AT" "$GIT_COMMIT" "$GIT_ORIGIN" 0 0)"
  "${PYTHON_BIN}" - "$PROVENANCE" "$PROVENANCE_SHA256" "$GIT_COMMIT" <<'PY'
import hashlib, json, sys
from pathlib import Path
path, expected_sha, expected_commit = sys.argv[1:]
raw = Path(path).read_bytes()
payload = json.loads(raw)
assert hashlib.sha256(raw).hexdigest() == expected_sha
assert payload["repository"]["git_commit"] == expected_commit
assert payload["entrypoint"][-1] == "examples.nanogpt.train"
assert payload["command"][-2] == "--config"
assert Path(payload["config"]["archive_path"]).is_file()
assert len(payload["dataset_manifest"]["sha256"]) == 64
print("provenance-self-test-ok")
PY
  exit 0
fi

worker() {
  local config="$1" gpu="$2" run_name="$3" workspace="$4" log="$5" status="$6" provenance="$7" provenance_sha256="$8" resume_mode="$9" pgid
  pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
  local started finished rc
  started="$(date -Is)"
  write_status "$status" running "$run_name" "$config" "$gpu" "$$" "$pgid" "$started" "" "" "$log" "$provenance" "$provenance_sha256"
  finish() {
    rc="$1"; finished="$(date -Is)"
    if [[ "$rc" -eq 0 ]]; then
      write_status "$status" finished "$run_name" "$config" "$gpu" "$$" "$pgid" "$started" "$finished" "$rc" "$log" "$provenance" "$provenance_sha256"
    else
      write_status "$status" failed "$run_name" "$config" "$gpu" "$$" "$pgid" "$started" "$finished" "$rc" "$log" "$provenance" "$provenance_sha256"
    fi
    exit "$rc"
  }
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1
  export EXPERIMENT_PROVENANCE_PATH="$provenance"
  export EXPERIMENT_PROVENANCE_SHA256="$provenance_sha256"
  cd "$REPO_DIR"
  set +e
  local train_args=(--config "$config")
  if [[ "$resume_mode" == "1" ]]; then
    train_args+=(--init-from resume)
  fi
  "$PYTHON_BIN" -u -m examples.nanogpt.train "${train_args[@]}"
  rc=$?
  set -e
  finish "$rc"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  [[ "$#" -eq 9 ]] || { echo "internal worker argument error" >&2; exit 2; }
  worker "$@"
fi

RESUME_MODE=0
FOREGROUND_MODE=0
while [[ "${1:-}" == "--resume" || "${1:-}" == "--foreground" ]]; do
  if [[ "$1" == "--resume" ]]; then RESUME_MODE=1; fi
  if [[ "$1" == "--foreground" ]]; then FOREGROUND_MODE=1; fi
  shift
done

[[ "$#" -ge 3 && "$#" -le 4 ]] || { echo "usage: $0 [--resume] [--foreground] CONFIG_PATH GPU_INDEX RUN_NAME [WORKSPACE_ROOT]" >&2; exit 2; }
CONFIG_INPUT="$1"; GPU="$2"; RUN_NAME="$3"; WORKSPACE="${4:-/root/userdata/MappingNetworks}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU_INDEX must be a non-negative integer" >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash" >&2; exit 2; }
if [[ "$CONFIG_INPUT" = /* ]]; then CONFIG="$CONFIG_INPUT"; else CONFIG="$REPO_DIR/$CONFIG_INPUT"; fi
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }
require_clean_git_checkout
RUN_DIR="$WORKSPACE/outputs/y400_ladder_runs"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/status" "$RUN_DIR/provenance"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
RUN_ID="${RUN_NAME}_${STAMP}"
LOG="$RUN_DIR/logs/${RUN_ID}.log"
STATUS="$RUN_DIR/status/${RUN_ID}.json"
CONFIG_ARCHIVE="$RUN_DIR/provenance/${RUN_ID}.config.json"
PROVENANCE="$RUN_DIR/provenance/${RUN_ID}.json"
LAUNCHED_AT="$(date -Is)"
PROVENANCE_SHA256="$(write_provenance "$CONFIG" "$CONFIG_ARCHIVE" "$PROVENANCE" "$RUN_ID" "$RUN_NAME" "$GPU" "$WORKSPACE" "$LAUNCHED_AT" "$GIT_COMMIT" "$GIT_ORIGIN" "$RESUME_MODE" "$FOREGROUND_MODE")"
# Pass every value as a distinct argv element: no generated remote shell string.
if [[ "$FOREGROUND_MODE" == "1" ]]; then
  printf 'launched foreground run=%s\nlog=%s\nstatus=%s\nprovenance=%s\n' "$RUN_NAME" "$LOG" "$STATUS" "$PROVENANCE"
  worker "$CONFIG_ARCHIVE" "$GPU" "$RUN_NAME" "$WORKSPACE" "$LOG" "$STATUS" "$PROVENANCE" "$PROVENANCE_SHA256" "$RESUME_MODE" >"$LOG" 2>&1
fi
setsid "$0" --worker "$CONFIG_ARCHIVE" "$GPU" "$RUN_NAME" "$WORKSPACE" "$LOG" "$STATUS" "$PROVENANCE" "$PROVENANCE_SHA256" "$RESUME_MODE" </dev/null >"$LOG" 2>&1 &
PID=$!
printf 'launched run=%s pid=%s pgid=%s\nlog=%s\nstatus=%s\nprovenance=%s\n' "$RUN_NAME" "$PID" "$PID" "$LOG" "$STATUS" "$PROVENANCE"
