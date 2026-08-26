#!/usr/bin/env python3
"""Persistent local watcher for the Y400 high-cadence MLP acquisition.

The watcher is deliberately local because the Feishu/OpenCode callback bridge
is local-only.  It observes exactly one remote process group and emits
idempotent, action-bearing callbacks at 20%, 50%, terminal success, failure,
stall, and the resettable 90-minute health heartbeat.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path


CALLBACK_URL = "http://127.0.0.1:8766/send-opencode-test"
CHAT_ID = "oc_fa5c2ec0190c9444cce960125eafff50"

REMOTE = r'''set -eu
pgid="$1"; log="$2"; status="$3"; gpu="$4"; out_dir="$5"
alive=false
kill -0 -- "-$pgid" 2>/dev/null && alive=true || true
python3 - "$alive" "$log" "$status" "$gpu" "$out_dir" <<'PY'
import json, pathlib, re, subprocess, sys
alive, log_path, status_path, gpu, out_dir = sys.argv[1:]
text = ""
try:
    with pathlib.Path(log_path).open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 131072))
        text = handle.read().decode("utf-8", "replace")
except OSError:
    pass
try:
    status = json.loads(pathlib.Path(status_path).read_text())
except (OSError, ValueError):
    status = {}
iters = re.findall(r"(?im)^\s*(?:iter(?:ation)?|step)\s*[=:]?\s*(\d+)\b", text)
evals = re.findall(r"(?im)^.*(?:val(?:idation)?|eval).*$", text)
errors = re.findall(
    r"Traceback|CUDA (?:out of memory|OOM)|AssertionError|\bfatal\b|\bNaN\b|\bInf\b",
    text,
    re.I,
)
try:
    gpu_info = subprocess.check_output(
        [
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
except Exception:
    gpu_info = ""
try:
    output_bytes = int(
        subprocess.check_output(
            ["du", "-sb", out_dir], text=True, timeout=60
        ).split()[0]
    )
except Exception:
    output_bytes = None
print(json.dumps({
    "alive": alive == "true",
    "status": status,
    "last_iter": int(iters[-1]) if iters else None,
    "last_eval": evals[-1].strip() if evals else None,
    "errors": sorted(set(errors)),
    "gpu": gpu_info,
    "output_bytes": output_bytes,
}))
PY'''


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {"sent": {}}
    except (OSError, json.JSONDecodeError):
        if path.exists():
            os.replace(path, path.with_name(path.name + f".corrupt.{int(time.time())}"))
        return {"sent": {}, "recovered_corrupt_state": True}


def callback(text: str) -> None:
    request = urllib.request.Request(
        CALLBACK_URL,
        data=json.dumps({"chat_id": CHAT_ID, "text": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=20
    ).read()


def emit(state: dict, path: Path, key: str, text: str) -> bool:
    if key in state.setdefault("sent", {}):
        return False
    state.setdefault("pending", {})[key] = text
    atomic_json(path, state)
    try:
        callback(text)
    except OSError as error:
        state["last_callback_error"] = str(error)
        atomic_json(path, state)
        return False
    now = time.time()
    state["sent"][key] = now
    state["last_successful_callback_at"] = now
    state["pending"].pop(key, None)
    atomic_json(path, state)
    return True


def retry_pending(state: dict, path: Path) -> None:
    for key, text in list(state.get("pending", {}).items()):
        if key in state.get("sent", {}):
            state["pending"].pop(key, None)
            continue
        try:
            callback(text)
        except OSError:
            continue
        now = time.time()
        state.setdefault("sent", {})[key] = now
        state["last_successful_callback_at"] = now
        state["pending"].pop(key, None)
    atomic_json(path, state)


def probe(args: argparse.Namespace) -> dict:
    command = "bash -s -- " + " ".join(
        shlex.quote(str(value))
        for value in (
            args.pgid,
            args.log_path,
            args.status_path,
            args.gpu,
            args.out_dir,
        )
    )
    result = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            args.host, command,
        ],
        input=REMOTE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout)


def action_prompt(kind: str) -> str:
    if kind == "progress":
        return (
            "Action required: verify live iteration, loss, GPU health, output "
            "inventory, and storage budget against the active high-cadence MLP "
            "note; intervene only if needed and continue the preregistered analysis. "
            "Do not merely acknowledge this callback."
        )
    if kind == "success":
        return (
            "Action required: verify terminal checkpoint, registered scientific "
            "probe inventory and fields, config/provenance hashes, fixed validation "
            "CE, GPU/performance data, and storage accounting; seal the active note, "
            "run the frozen preregistered analysis, then continue only along the "
            "causally authorized compact-MLP branch. Do not merely acknowledge."
        )
    return (
        "Action required: inspect remote status, log, GPU, output inventory, and "
        "active high-cadence MLP note; diagnose and safely recover or requeue only "
        "if scientifically justified, record remediation, and continue the plan. "
        "Do not merely acknowledge this callback."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgid", required=True, type=int)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--state-path", required=True, type=Path)
    parser.add_argument("--host", default="Y400")
    parser.add_argument("--max-iters", type=int, default=238)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stall-minutes", type=int, default=20)
    parser.add_argument("--heartbeat-minutes", type=int, default=90)
    parser.add_argument("--output-budget-gib", type=float, default=15.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert "Action required:" in action_prompt("progress")
        assert "scientific probe inventory" in action_prompt("success")
        print("self-test passed")
        return
    if args.pgid < 1 or args.max_iters < 1 or args.interval < 15:
        parser.error("invalid PGID, max-iters, or interval")

    state = load_state(args.state_path)
    state.setdefault("started_at", time.time())
    state.setdefault("last_progress_at", time.time())
    state.setdefault("last_successful_callback_at", time.time())
    retry_pending(state, args.state_path)

    while True:
        try:
            sample = probe(args)
            now = time.time()
            prior_iter = state.get("last_iter")
            current_iter = sample.get("last_iter")
            if current_iter is not None and (
                prior_iter is None or current_iter > prior_iter
            ):
                state["last_progress_at"] = now
            state.update(last_iter=current_iter, sample=sample, updated_at=now)
            retry_pending(state, args.state_path)

            status = str(
                sample.get("status", {}).get(
                    "state", sample.get("status", {}).get("status", "")
                )
            ).lower()
            if sample.get("errors"):
                emit(
                    state,
                    args.state_path,
                    "error:" + ",".join(sample["errors"]),
                    f"[bot] @Codex {args.run_name} ERROR: {sample['errors']}\n\n"
                    + action_prompt("failure"),
                )
            if sample.get("output_bytes") is not None and (
                sample["output_bytes"] > args.output_budget_gib * (1024**3)
            ):
                emit(
                    state,
                    args.state_path,
                    "output_budget_exceeded",
                    f"[bot] @Codex {args.run_name} STORAGE_RISK: "
                    f"output={sample['output_bytes'] / 1024**3:.2f} GiB exceeds "
                    f"{args.output_budget_gib:.2f} GiB\n\n" + action_prompt("failure"),
                )

            terminal = status in {"finished", "failed"}
            if terminal:
                kind = "success" if status == "finished" else "failure"
                label = "100% FINISHED" if status == "finished" else "FAILED"
                emit(
                    state,
                    args.state_path,
                    "terminal",
                    f"[bot] @Codex {args.run_name} {label}: "
                    f"iter={current_iter}/{args.max_iters} gpu={sample.get('gpu','')}\n\n"
                    + action_prompt(kind),
                )
                retry_pending(state, args.state_path)
                if not state.get("pending"):
                    return
            if not sample.get("alive") and not terminal:
                emit(
                    state,
                    args.state_path,
                    "missing_process_group",
                    f"[bot] @Codex {args.run_name} ERROR: process group missing "
                    f"while status={status or 'unknown'} iter={current_iter}\n\n"
                    + action_prompt("failure"),
                )
                return

            if current_iter is not None:
                for milestone in (20, 50):
                    if current_iter * 100 >= milestone * args.max_iters:
                        emit(
                            state,
                            args.state_path,
                            f"milestone_{milestone}",
                            f"[bot] @Codex {args.run_name} PROGRESS: {milestone}% "
                            f"({current_iter}/{args.max_iters}) gpu={sample.get('gpu','')}\n\n"
                            + action_prompt("progress"),
                        )

            since_progress = now - float(state["last_progress_at"])
            if since_progress >= args.stall_minutes * 60:
                emit(
                    state,
                    args.state_path,
                    f"stall_{int(since_progress // (args.stall_minutes * 60))}",
                    f"[bot] @Codex {args.run_name} STALL: no iteration progress "
                    f"for {int(since_progress)}s at {current_iter}/{args.max_iters}\n\n"
                    + action_prompt("failure"),
                )
            since_callback = now - float(state["last_successful_callback_at"])
            if since_callback >= args.heartbeat_minutes * 60:
                emit(
                    state,
                    args.state_path,
                    f"heartbeat_{int(now // (args.heartbeat_minutes * 60))}",
                    f"[bot] @Codex {args.run_name} HEARTBEAT: "
                    f"iter={current_iter}/{args.max_iters} gpu={sample.get('gpu','')}\n\n"
                    + action_prompt("progress"),
                )
            atomic_json(args.state_path, state)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            emit(
                state,
                args.state_path,
                f"monitor_degraded_{int(time.time() // 900)}",
                f"[bot] @Codex {args.run_name} MONITOR_DEGRADED: {error}\n\n"
                + action_prompt("failure"),
            )
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
