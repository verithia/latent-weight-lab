#!/usr/bin/env python3
"""One aggregate GPU-ladder heartbeat with progress-aware clock resets.

Milestone watchers own 20/50/80/terminal callbacks.  This supervisor sends a
single combined health update only when no successfully delivered milestone has
reset the shared clock during the preceding heartbeat period.  It also reports
per-run stalls or log errors immediately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path


CALLBACK_URL = "http://127.0.0.1:8766/send-opencode-test"
AGENT_MENTION = "@Codex"

REMOTE_PROBE = r'''python3 - "$1" <<'PY'
import json, pathlib, re, subprocess, sys
runs = json.loads(sys.argv[1])
samples = []
for run in runs:
    alive = False
    try:
        subprocess.run(["bash", "-c", "kill -0 -- -$1", "_", str(run["pgid"])], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        alive = True
    except Exception:
        pass
    text = ""
    try:
        path = pathlib.Path(run["log"])
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 65536))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        pass
    try:
        status = json.loads(pathlib.Path(run["status"]).read_text())
    except (OSError, ValueError):
        status = {}
    iters = re.findall(r"(?im)^\s*(?:iter(?:ation)?|step)\s*[=:]?\s*(\d+)\b", text)
    errors = re.findall(r"Traceback|CUDA (?:out of memory|OOM)|\bNaN\b|\bInf\b|AssertionError|\bfatal\b", text, re.I)
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--id=" + str(run["gpu"]), "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    except Exception:
        gpu = ""
    samples.append({"name": run["name"], "alive": alive, "last_iter": int(iters[-1]) if iters else None, "errors": sorted(set(errors)), "status": status, "gpu": gpu})
print(json.dumps(samples))
PY'''


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {"runs": {}, "sent": {}}
    except (OSError, json.JSONDecodeError):
        return {"runs": {}, "sent": {}}


def parse_run(value: str) -> dict[str, object]:
    fields = value.split(",")
    if len(fields) not in (5, 6):
        raise argparse.ArgumentTypeError("run must be NAME,PGID,LOG,STATUS,GPU[,MAX_ITERS]")
    name, pgid, log, status, gpu = fields[:5]
    max_iters = fields[5] if len(fields) == 6 else ""
    if not name or not pgid.isdecimal() or not gpu.isdecimal() or not log or not status:
        raise argparse.ArgumentTypeError("invalid run specification")
    if max_iters and (not max_iters.isdecimal() or int(max_iters) < 1):
        raise argparse.ArgumentTypeError("MAX_ITERS must be a positive integer")
    return {
        "name": name,
        "pgid": int(pgid),
        "log": log,
        "status": status,
        "gpu": int(gpu),
        "max_iters": int(max_iters) if max_iters else None,
    }


def send(chat_id: str, text: str) -> bool:
    request = urllib.request.Request(
        CALLBACK_URL,
        # The bridge invocation endpoint wakes the active coding agent.  The
        # explicit visible mention makes ownership unambiguous in the group.
        data=json.dumps({"chat_id": chat_id, "text": f"{AGENT_MENTION} {text}"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=20).read()
    except OSError:
        return False
    return True


def progress_reset_at(path: Path) -> float:
    try:
        return float(json.loads(path.read_text()).get("last_progress_callback_at", 0.0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0.0


def heartbeat_due(now: float, last_callback_at: float, progress_at: float, period_seconds: int) -> bool:
    return now - max(last_callback_at, progress_at) >= period_seconds


def heartbeat_text(label: str, samples: list[dict]) -> str:
    parts = []
    for sample in samples:
        current = sample.get("last_iter")
        total = sample.get("max_iters")
        progress = f"{current}/{total}" if current is not None and total is not None else "unavailable"
        terminal = str(sample.get("status", {}).get("state", "")).lower() in {"finished", "failed"}
        process_state = "process=missing" if not sample.get("alive") and not terminal else ""
        parts.append(
            f"{sample['name']}: iter={progress} {process_state} gpu={sample['gpu'] or 'unavailable'}".rstrip()
        )
    return f"{label} HEARTBEAT: " + " | ".join(parts)


def milestone_crossings(previous: int | None, current: int | None, maximum: int | None, sent: set[int]) -> list[int]:
    """Return only milestones newly crossed since the previous aggregate probe."""
    if previous is None or current is None or maximum is None or current <= previous:
        return []
    return [percent for percent in (20, 50) if percent not in sent and previous * 100 < percent * maximum <= current * 100]


def event_text(
    label: str,
    progress_events: list[tuple[str, int, int, int]],
    terminal_events: list[tuple[str, str, int | None, int | None, int | None]],
) -> str:
    parts = [f"{name} {percent}% ({iteration}/{maximum})" for name, percent, iteration, maximum in progress_events]
    parts.extend(
        f"{name} 100% ({current}/{maximum}) {state}{'' if exit_code is None else f' exit={exit_code}'}"
        for name, state, exit_code, current, maximum in terminal_events
    )
    return f"{label} PROGRESS: " + " | ".join(parts)


def probe(host: str, runs: list[dict]) -> list[dict]:
    command = "bash -s -- " + shlex.quote(json.dumps(runs, separators=(",", ":")))
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command],
        input=REMOTE_PROBE,
        text=True,
        capture_output=True,
        timeout=90,
        check=True,
    )
    return json.loads(completed.stdout)


def event_key(name: str, kind: str, detail: object) -> str:
    digest = hashlib.sha256(f"{name}:{kind}:{detail}".encode()).hexdigest()[:16]
    return f"{name}:{kind}:{digest}"


def monitor_error_text(exc: Exception) -> str:
    """Return a short actionable probe failure without leaking command details."""
    if isinstance(exc, subprocess.CalledProcessError):
        return f"remote probe failed (ssh exit {exc.returncode}); retrying"
    if isinstance(exc, subprocess.TimeoutExpired):
        return "remote probe timed out; retrying"
    return f"remote probe failed ({type(exc).__name__}); retrying"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="Y400")
    parser.add_argument("--label", default="Y400 dense ladder")
    parser.add_argument("--state-key", default="y400_dense_aggregate")
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument("--heartbeat-minutes", type=int, default=90)
    parser.add_argument("--stall-minutes", type=int, default=15)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.heartbeat_minutes < 1 or args.stall_minutes < 1 or args.interval < 15:
        parser.error("heartbeat/stall minutes must be >=1 and interval >=15 seconds")

    if not args.state_key or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in args.state_key):
        parser.error("--state-key may contain only letters, digits, dot, underscore, and dash")
    state_path = args.state_dir / f"{args.state_key}.json"
    progress_path = args.state_dir / f"{args.state_key}_progress.json"
    state = load(state_path)
    initialized = bool(state.get("initialized"))
    while True:
        now = time.time()
        try:
            samples = probe(args.host, args.run)
        except Exception as exc:  # Monitoring must keep trying after a transport fault.
            key = event_key("aggregate", "monitor_degraded", type(exc).__name__)
            if key not in state.setdefault("sent", {}) and send(
                args.chat_id,
                f"{args.label} MONITOR_DEGRADED: {monitor_error_text(exc)}",
            ):
                state["sent"][key] = now
            atomic(state_path, state)
            time.sleep(min(args.interval * 2, 600))
            continue

        # The aggregate supervisor is the sole owner of progress and terminal
        # callbacks.  Persisted iteration state lets it survive a restart
        # without replaying milestones that another sample already observed.
        for sample, run in zip(samples, args.run):
            sample["max_iters"] = run["max_iters"]
        progress_events: list[tuple[str, int, int, int]] = []
        terminal_events: list[tuple[str, str, int | None, int | None, int | None]] = []
        for sample in samples:
            run_state = state.setdefault("runs", {}).setdefault(sample["name"], {})
            previous_iter = run_state.get("last_iter")
            current_iter = sample.get("last_iter")
            maximum = sample.get("max_iters")
            sent_milestones = set(run_state.get("sent_milestones", []))
            if previous_iter is None and current_iter is not None and maximum is not None:
                # The first sample is a baseline, never a catch-up notification.
                sent_milestones.update(percent for percent in (20, 50) if current_iter * 100 >= percent * maximum)
            else:
                for percent in milestone_crossings(previous_iter, current_iter, maximum, sent_milestones):
                    progress_events.append((sample["name"], percent, current_iter, maximum))
            if current_iter is not None and (previous_iter is None or current_iter > previous_iter):
                run_state["last_progress_at"] = now
            run_state["last_iter"] = current_iter
            run_state["sent_milestones"] = sorted(sent_milestones)
            run_state["sample"] = sample
            terminal = str(sample.get("status", {}).get("state", "")).lower() in {"finished", "failed"}
            if terminal:
                terminal_signature = (
                    str(sample.get("status", {}).get("state", "")),
                    sample.get("status", {}).get("exit_code"),
                    str(sample.get("status", {}).get("finished_at", "")),
                )
                if run_state.get("terminal_signature") != terminal_signature:
                    terminal_events.append((sample["name"], terminal_signature[0], terminal_signature[1], current_iter, maximum))
            elif not sample.get("alive"):
                key = event_key(sample["name"], "process_missing", sample.get("pgid", sample["name"]))
                if key not in state.setdefault("sent", {}) and send(
                    args.chat_id,
                    f"{args.label} {sample['name']} ERROR: process group missing while status=running; "
                    f"last_iter={current_iter}",
                ):
                    state["sent"][key] = now
            for error in sample.get("errors", []):
                key = event_key(sample["name"], "error", error)
                if key not in state.setdefault("sent", {}) and send(args.chat_id, f"{args.label} {sample['name']} ERROR: {error}"):
                    state["sent"][key] = now
            progress_at = float(run_state.get("last_progress_at", now))
            if sample.get("alive") and not terminal and now - progress_at >= args.stall_minutes * 60:
                key = event_key(sample["name"], "stall", int((now - progress_at) // (args.stall_minutes * 60)))
                if key not in state.setdefault("sent", {}) and send(args.chat_id, f"{args.label} {sample['name']} STALL: no progress for {int(now - progress_at)}s"):
                    state["sent"][key] = now

        # One message can describe every milestone/terminal transition found in
        # this probe.  Only a successful delivery advances the heartbeat clock.
        if progress_events or terminal_events:
            if send(args.chat_id, event_text(args.label, progress_events, terminal_events)):
                for name, percent, _iteration, _maximum in progress_events:
                    run_state = state["runs"][name]
                    run_state["sent_milestones"] = sorted(set(run_state.get("sent_milestones", [])) | {percent})
                for name, _terminal_state, _exit_code, _current, _maximum in terminal_events:
                    sample = state["runs"][name]["sample"]
                    status = sample.get("status", {})
                    state["runs"][name]["terminal_signature"] = (
                        str(status.get("state", "")),
                        status.get("exit_code"),
                        str(status.get("finished_at", "")),
                    )
                state["last_callback_at"] = now
                # Persist canonical event ownership immediately after a
                # successful delivery. A service reload between the bridge
                # call and the end-of-probe state write must not replay a
                # terminal or milestone callback.
                atomic(state_path, state)
                atomic(progress_path, {"last_progress_callback_at": now, "event": "aggregate_progress"})

        progress_at = progress_reset_at(progress_path)
        if not initialized:
            state["last_callback_at"] = max(float(state.get("last_callback_at", 0.0)), progress_at, now)
            initialized = True
            state["initialized"] = True
        if heartbeat_due(now, float(state.get("last_callback_at", now)), progress_at, args.heartbeat_minutes * 60):
            if send(args.chat_id, heartbeat_text(args.label, samples)):
                state["last_callback_at"] = now
        state["updated_at"] = now
        atomic(state_path, state)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
