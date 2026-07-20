#!/usr/bin/env bash
# Y400-only ladder launcher for one config. It does not poll or notify.
# Usage (on Y400):
#   examples/nanogpt/launch_y400_ladder_detached.sh examples/nanogpt/configs/NAME.json 0 NAME [WORKSPACE_ROOT]
# Detached mode prints a setsid process-group leader.  Foreground mode is for a
# caller-owned persistent supervisor (normally a named tmux session); the
# worker records terminal JSON status and exits with the exact training code.
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
  local config="$1" config_archive="$2" provenance="$3" run_id="$4" run_name="$5" gpu="$6" workspace="$7" launched_at="$8" git_commit="$9" git_origin="${10}" mfu_certificate="${11}"
  "${PYTHON_BIN}" - "$config" "$config_archive" "$provenance" "$run_id" "$run_name" "$gpu" "$workspace" "$launched_at" "$git_commit" "$git_origin" "$mfu_certificate" "$REPO_DIR" "$SCRIPT_DIR" "$PYTHON_BIN" "$PROVENANCE_SCHEMA_VERSION" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

(
    config_path, config_archive, provenance_path, run_id, run_name, gpu, workspace,
    launched_at, git_commit, git_origin, mfu_certificate, repo_dir, launcher_dir, python_bin, schema_version,
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
if config.get("mfu_preflight_required") is not True:
    raise SystemExit("refusing launch: config must set mfu_preflight_required=true")
if float(config.get("mfu_min_fraction", 0.0)) < 0.20:
    raise SystemExit("refusing launch: config mfu_min_fraction must be >= 0.20")
certificate = Path(mfu_certificate)
if not certificate.is_file():
    raise SystemExit("refusing launch: measured MFU certificate is absent")
certificate_raw = certificate.read_bytes()
certificate_payload = json.loads(certificate_raw)
if certificate_payload.get("passed") is not True:
    raise SystemExit("refusing launch: measured MFU certificate did not pass")
if certificate_payload.get("config", {}).get("sha256") != hashlib.sha256(raw_config).hexdigest():
    raise SystemExit("refusing launch: MFU certificate does not match config")

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
    "performance_preflight": {
        "path": str(certificate.resolve()),
        "sha256": hashlib.sha256(certificate_raw).hexdigest(),
        "mfu_fraction": certificate_payload["measurement"]["mfu_fraction"],
        "minimum_fraction": certificate_payload["policy"]["minimum_fraction"],
        "denominator": certificate_payload["policy"]["denominator"],
    },
    "runtime": {
        "workspace": str(Path(workspace).resolve()),
        "cuda_visible_devices": str(gpu),
        "python": str(Path(python_bin).resolve()) if Path(python_bin).exists() else python_bin,
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
  SELFTEST_CONFIG="$SELFTEST_DIR/input.json"
  CONFIG_ARCHIVE="$SELFTEST_DIR/config.json"
  PROVENANCE="$SELFTEST_DIR/provenance.json"
  MFU_CERTIFICATE="$SELFTEST_DIR/mfu.json"
  mkdir -p "$SELFTEST_DIR"
  "${PYTHON_BIN}" - "$CONFIG" "$SELFTEST_CONFIG" "$MFU_CERTIFICATE" <<'PY'
import hashlib, json, sys
from pathlib import Path
source, config_path, certificate_path = map(Path, sys.argv[1:])
config = json.loads(source.read_text())
config["mfu_preflight_required"] = True
config["mfu_min_fraction"] = 0.20
raw = json.dumps(config, sort_keys=True).encode() + b"\n"
config_path.write_bytes(raw)
certificate = {
    "passed": True,
    "config": {"sha256": hashlib.sha256(raw).hexdigest()},
    "policy": {"minimum_fraction": 0.20, "denominator": "self_test"},
    "measurement": {"mfu_fraction": 0.20},
}
certificate_path.write_text(json.dumps(certificate, sort_keys=True) + "\n")
PY
  LAUNCHED_AT="$(date -Is)"
  PROVENANCE_SHA256="$(write_provenance "$SELFTEST_CONFIG" "$CONFIG_ARCHIVE" "$PROVENANCE" provenance-self-test provenance-self-test 0 /tmp "$LAUNCHED_AT" "$GIT_COMMIT" "$GIT_ORIGIN" "$MFU_CERTIFICATE")"
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
assert payload["performance_preflight"]["mfu_fraction"] == 0.20
print("provenance-self-test-ok")
PY
  exit 0
fi

worker() {
  local config="$1" gpu="$2" run_name="$3" workspace="$4" log="$5" status="$6" provenance="$7" provenance_sha256="$8" pgid
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
  "$PYTHON_BIN" -u -m examples.nanogpt.train --config "$config"
  rc=$?
  set -e
  finish "$rc"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  [[ "$#" -eq 8 ]] || { echo "internal worker argument error" >&2; exit 2; }
  worker "$@"
fi

FOREGROUND_MODE=0
if [[ "${1:-}" == "--foreground" ]]; then
  FOREGROUND_MODE=1
  shift
fi

[[ "$#" -ge 3 && "$#" -le 4 ]] || { echo "usage: $0 CONFIG_PATH GPU_INDEX RUN_NAME [WORKSPACE_ROOT]" >&2; exit 2; }
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
MFU_CERTIFICATE="$RUN_DIR/provenance/${RUN_ID}.mfu.json"
LAUNCHED_AT="$(date -Is)"

# A launch is forbidden unless the selected card is idle and the exact config
# clears a real-training MFU measurement.  This is intentionally before both
# provenance publication and detached worker creation.
GPU_PIDS="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^No running processes found$/d;/^$/d' || true)"
[[ -z "$GPU_PIDS" ]] || { echo "refusing launch: GPU $GPU is not exclusive for MFU preflight (PIDs: $GPU_PIDS)" >&2; exit 2; }
MFU_MIN_FRACTION="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
if config.get("mfu_preflight_required") is not True:
    raise SystemExit("refusing launch: config must set mfu_preflight_required=true")
minimum = float(config.get("mfu_min_fraction", 0.0))
if minimum < 0.20:
    raise SystemExit("refusing launch: config mfu_min_fraction must be >= 0.20")
print(minimum)
PY
)"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m examples.nanogpt.mfu_preflight \
  --config "$CONFIG" --output "$MFU_CERTIFICATE" --min-fraction "$MFU_MIN_FRACTION"
PROVENANCE_SHA256="$(write_provenance "$CONFIG" "$CONFIG_ARCHIVE" "$PROVENANCE" "$RUN_ID" "$RUN_NAME" "$GPU" "$WORKSPACE" "$LAUNCHED_AT" "$GIT_COMMIT" "$GIT_ORIGIN" "$MFU_CERTIFICATE")"
if [[ "$FOREGROUND_MODE" -eq 1 ]]; then
  # The caller must keep this process alive (for example with `tmux new-session
  # -d`).  Do not use a second detached process group: otherwise tmux cannot
  # reliably report or terminate the actual training process.
  printf 'launching foreground run=%s\nlog=%s\nstatus=%s\nprovenance=%s\n' "$RUN_NAME" "$LOG" "$STATUS" "$PROVENANCE"
  exec "$0" --worker "$CONFIG_ARCHIVE" "$GPU" "$RUN_NAME" "$WORKSPACE" "$LOG" "$STATUS" "$PROVENANCE" "$PROVENANCE_SHA256" >"$LOG" 2>&1
fi

# Pass every value as a distinct argv element: no generated remote shell string.
setsid "$0" --worker "$CONFIG_ARCHIVE" "$GPU" "$RUN_NAME" "$WORKSPACE" "$LOG" "$STATUS" "$PROVENANCE" "$PROVENANCE_SHA256" </dev/null >"$LOG" 2>&1 &
PID=$!
printf 'launched detached run=%s pid=%s pgid=%s\nlog=%s\nstatus=%s\nprovenance=%s\n' "$RUN_NAME" "$PID" "$PID" "$LOG" "$STATUS" "$PROVENANCE"
