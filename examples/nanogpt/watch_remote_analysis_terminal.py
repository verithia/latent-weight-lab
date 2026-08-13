#!/usr/bin/env python3
"""Monitor a remote zero-update audit and send idempotent callbacks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


CALLBACK_URL = "http://127.0.0.1:8766/send-opencode-test"
TERMINAL_ACTION = (
    "Action required: verify the terminal audit and exact artifacts against the "
    "active project note; seal hashes and gate outcomes, update durable notes, "
    "then continue with only the next causally authorized experiment. Do not "
    "merely acknowledge this callback."
)
ERROR_ACTION = (
    "Action required: inspect the remote status and log, diagnose the failure, "
    "update durable state, and repair or requeue only if scientifically justified."
)
LIVE_ACTION = (
    "Action required: verify the live process, GPU health, status, and active "
    "project note; intervene only if needed, record durable changes, and "
    "continue the causally authorized analysis. Do not merely acknowledge this "
    "callback."
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def send(chat_id: str, text: str) -> bool:
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    request = urllib.request.Request(
        CALLBACK_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=20) as response:
            response.read()
        return True
    except Exception as error:  # noqa: BLE001
        print(f"callback failed: {error!r}", flush=True)
        return False


def probe(
    host: str,
    pgid: int,
    status: str,
    result: str,
    log: str,
    gpu: int,
) -> dict[str, Any]:
    script = r'''
set -u
pgid=$1; status=$2; result=$3; log=$4; gpu=$5
alive=0
kill -0 -- "-$pgid" 2>/dev/null && alive=1
gpu_health=$(
    nvidia-smi -i "$gpu" \
        --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true
)
python3 - "$alive" "$status" "$result" "$log" "$gpu_health" <<'PY'
import hashlib, json, pathlib, sys
alive = bool(int(sys.argv[1]))
status_path, result_path, log_path = map(pathlib.Path, sys.argv[2:5])
gpu_health = sys.argv[5]
def load(path):
    try: return json.loads(path.read_text())
    except Exception: return None
def digest(path):
    if not path.is_file(): return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
print(json.dumps({
    'alive': alive,
    'status': load(status_path),
    'result_exists': result_path.is_file(),
    'result_sha256': digest(result_path),
    'gpu_health': gpu_health,
    'log_tail': '\n'.join(log_path.read_text(errors='replace').splitlines()[-30:]) if log_path.is_file() else '',
}))
PY
'''
    completed = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host,
            "bash", "-s", "--", str(pgid), status, result, log, str(gpu),
        ],
        input=script,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="PRO6")
    parser.add_argument("--pgid", required=True, type=int)
    parser.add_argument("--status", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--heartbeat-minutes",
        type=float,
        default=0.0,
        help="Send live heartbeats at this interval; zero preserves terminal-only mode.",
    )
    args = parser.parse_args()
    state: dict[str, Any] = {
        "schema_version": "remote_analysis_watch_v2",
        "run_label": args.run_label,
        "pgid": args.pgid,
        "state": "running",
        "consecutive_probe_errors": 0,
        "missing_process_samples": 0,
        "started_at_unix": time.time(),
    }
    if args.state.exists():
        previous = json.loads(args.state.read_text())
        if previous.get("pgid") != args.pgid:
            raise ValueError("watch state PGID identity mismatch")
        state.update(previous)
    state["schema_version"] = "remote_analysis_watch_v2"
    atomic_json(args.state, state)

    while True:
        try:
            sample = probe(
                args.host,
                args.pgid,
                args.status,
                args.result,
                args.log,
                args.gpu,
            )
            state["last_sample"] = sample
            state["consecutive_probe_errors"] = 0
            status = sample.get("status") or {}
            terminal = status.get("state") in {"finished", "failed"}
            if terminal:
                exit_code = status.get("exit_code")
                state["terminal_signature"] = [status.get("state"), exit_code, sample.get("result_sha256")]
                if exit_code == 0 and sample.get("result_exists"):
                    message = (
                        f"[bot] @Codex {args.run_label} FINISHED: exit=0 "
                        f"result_sha256={sample['result_sha256']}\n\n{TERMINAL_ACTION}"
                    )
                else:
                    message = (
                        f"[bot] @Codex {args.run_label} ERROR: exit={exit_code}; "
                        f"result_exists={sample.get('result_exists')}\n\n{ERROR_ACTION}"
                    )
                if send(args.chat_id, message):
                    state["state"] = "finished"
                    state["callback_delivered_at_unix"] = time.time()
                    atomic_json(args.state, state)
                    return
            elif not sample.get("alive"):
                state["missing_process_samples"] = int(state.get("missing_process_samples", 0)) + 1
                if state["missing_process_samples"] >= 3:
                    message = f"[bot] @Codex {args.run_label} ERROR: process group missing without terminal status.\n\n{ERROR_ACTION}"
                    if send(args.chat_id, message):
                        state["state"] = "failed"
                        atomic_json(args.state, state)
                        return
            else:
                state["missing_process_samples"] = 0
                now = time.time()
                last_callback = float(
                    state.get("last_callback_at_unix", state["started_at_unix"])
                )
                if (
                    args.heartbeat_minutes > 0
                    and now - last_callback >= args.heartbeat_minutes * 60
                ):
                    remote_started = float(status.get("started_at_unix", now))
                    elapsed_hours = max(0.0, now - remote_started) / 3600.0
                    message = (
                        f"[bot] @Codex {args.run_label} HEARTBEAT: "
                        f"alive=true status={status.get('state', 'unknown')} "
                        f"elapsed={elapsed_hours:.1f}h "
                        f"gpu={sample.get('gpu_health') or 'unavailable'} "
                        f"result_exists={sample.get('result_exists')}\n\n{LIVE_ACTION}"
                    )
                    if send(args.chat_id, message):
                        state["last_callback_at_unix"] = now
                        state["last_callback_kind"] = "heartbeat"
            atomic_json(args.state, state)
        except Exception as error:  # noqa: BLE001
            state["consecutive_probe_errors"] = int(state.get("consecutive_probe_errors", 0)) + 1
            state["last_error"] = repr(error)
            atomic_json(args.state, state)
            if state["consecutive_probe_errors"] >= 3:
                message = f"[bot] @Codex {args.run_label} MONITOR ERROR: {error!r}\n\n{ERROR_ACTION}"
                if send(args.chat_id, message):
                    state["state"] = "monitor_failed"
                    atomic_json(args.state, state)
                    return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
