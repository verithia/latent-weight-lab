#!/usr/bin/env python3
"""Submit and monitor the ordered, performance-gated Y400 dense queue.

This is intentionally one persistent local worker.  It owns GPU admission,
workspace/checkpoint headroom, remote Git/config validation, submission, and
aggregate 20/50/100 plus resettable 90-minute callbacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CALLBACK_URL = "http://127.0.0.1:8766/send-opencode-test"
AGENT_MENTION = "@Codex"
REMOTE_REPO_RELATIVE = "latent-weight-lab"
REMOTE_PYTHON_RELATIVE = ".venv-gpt2/bin/python"
REMOTE_LAUNCHER = "examples/nanogpt/launch_y400_ladder_detached.sh"


REMOTE_SNAPSHOT = r'''import hashlib, json, os, pathlib, re, subprocess, sys
root = pathlib.Path(sys.argv[1])
payload = json.loads(sys.argv[2])
repo = root / "latent-weight-lab"

def command(argv, default=""):
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL, timeout=30).strip()
    except Exception:
        return default

def file_sha(path):
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None

used = int(command(["du", "-sb", str(root)], "0").split()[0])
git_commit = command(["git", "-C", str(repo), "rev-parse", "HEAD"])
git_dirty = bool(command(["git", "-C", str(repo), "status", "--porcelain"]))
source_hashes = {path: file_sha(repo / path) for path in payload.get("source_paths", [])}

gpus = []
gpu_rows = command([
    "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
])
for row in gpu_rows.splitlines():
    fields = [field.strip() for field in row.split(",")]
    if len(fields) != 5:
        continue
    index, memory_used, memory_total, utilization, temperature = map(int, fields)
    pids = command([
        "nvidia-smi", "-i", str(index), "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ])
    pid_values = [int(value.strip()) for value in pids.splitlines() if value.strip().isdigit()]
    gpus.append({
        "index": index,
        "memory_used_mib": memory_used,
        "memory_total_mib": memory_total,
        "utilization": utilization,
        "temperature": temperature,
        "compute_pids": pid_values,
    })

entries = {}
status_root = root / "outputs/y400_ladder_runs/status"
for entry in payload.get("entries", []):
    name = entry["name"]
    config = repo / entry["config"]
    result = {
        "config_sha256": file_sha(config),
        "checkpoint_next_iter": None,
        "checkpoint_metadata_path": None,
        "status_path": entry.get("status_path"),
        "status": {},
        "last_iter": None,
        "alive": False,
        "tmux_alive": False,
        "log_tail": "",
    }
    try:
        config_payload = json.loads(config.read_text())
        metadata = pathlib.Path(config_payload["out_dir"]) / "ckpt.meta.json"
        result["checkpoint_metadata_path"] = str(metadata)
        if metadata.is_file():
            result["checkpoint_next_iter"] = json.loads(metadata.read_text()).get("next_iter")
    except (OSError, KeyError, ValueError, TypeError):
        pass

    status_path_value = entry.get("status_path")
    if not status_path_value and status_root.is_dir():
        candidates = sorted(status_root.glob(name + "_*.json"), key=lambda path: path.stat().st_mtime)
        submitted_at = float(entry.get("submitted_at", 0.0))
        candidates = [path for path in candidates if path.stat().st_mtime >= submitted_at - 60]
        if candidates:
            status_path_value = str(candidates[-1])
    if status_path_value:
        status_path = pathlib.Path(status_path_value)
        result["status_path"] = str(status_path)
        try:
            result["status"] = json.loads(status_path.read_text())
        except (OSError, ValueError):
            pass
    status = result["status"]
    pgid = status.get("pgid")
    if isinstance(pgid, int):
        try:
            os.killpg(pgid, 0)
            result["alive"] = True
        except (OSError, ProcessLookupError, PermissionError):
            pass
    session = entry.get("session")
    if session:
        result["tmux_alive"] = subprocess.run(
            ["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    log_path_value = status.get("log") or entry.get("submit_log")
    if log_path_value:
        try:
            log_path = pathlib.Path(log_path_value)
            with log_path.open("rb") as handle:
                handle.seek(0, 2)
                handle.seek(max(0, handle.tell() - 65536))
                result["log_tail"] = handle.read().decode("utf-8", "replace")
        except OSError:
            pass
    iterations = re.findall(
        r"(?im)^\s*(?:iter(?:ation)?|step)\s*[=:]?\s*(\d+)\b", result["log_tail"]
    )
    if iterations:
        result["last_iter"] = int(iterations[-1])
    entries[name] = result

print(json.dumps({
    "workspace_used_bytes": used,
    "git_commit": git_commit,
    "git_dirty": git_dirty,
    "source_hashes": source_hashes,
    "gpus": gpus,
    "entries": entries,
}))'''


REMOTE_LAUNCH = r'''set -euo pipefail
root="$1"; config="$2"; gpu="$3"; name="$4"; session="$5"; submit_log="$6"; resume="$7"
repo="$root/latent-weight-lab"
python_bin="$root/.venv-gpt2/bin/python"
launcher="$repo/examples/nanogpt/launch_y400_ladder_detached.sh"
mkdir -p "$(dirname "$submit_log")"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "refusing duplicate queue session: $session" >&2
  exit 2
fi
printf -v command 'cd %q && export PYTHON_BIN=%q && exec bash %q --foreground' "$repo" "$python_bin" "$launcher"
if [[ "$resume" == "true" ]]; then command+=' --resume'; fi
printf -v command '%s %q %q %q %q >%q 2>&1' "$command" "$config" "$gpu" "$name" "$root" "$submit_log"
tmux new-session -d -s "$session" "$command"
'''


REMOTE_DEPLOY = r'''set -euo pipefail
repo="$1"
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo "remote worktree is dirty" >&2; exit 2; }
git -C "$repo" fetch --quiet origin main
git -C "$repo" checkout --quiet --detach FETCH_HEAD
git -C "$repo" rev-parse HEAD
'''


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, repo: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "y400_dense_queue_v1":
        raise ValueError("unsupported queue schema")
    names = set()
    for entry in payload.get("entries", []):
        name = entry.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError("invalid queue entry name")
        if name in names:
            raise ValueError("duplicate queue entry name")
        names.add(name)
        config = repo / entry["config"]
        if not config.is_file() or file_sha256(config) != entry.get("config_sha256"):
            raise ValueError("queue config hash mismatch: " + str(config))
        config_payload = json.loads(config.read_text())
        if config_payload.get("mfu_preflight_required") is not True:
            raise ValueError("queue config lacks mandatory MFU preflight: " + name)
        if float(config_payload.get("mfu_min_fraction", 0.0)) < 0.20:
            raise ValueError("queue config MFU floor is below 20%: " + name)
        if entry.get("resume") is not True:
            raise ValueError("this queue accepts only exact resume entries: " + name)
        if int(entry.get("checkpoint_bytes", 0)) <= 0:
            raise ValueError("queue entry lacks checkpoint budget: " + name)
    return payload


def load_state(path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        state = {}
    entries = state.setdefault("entries", {})
    for entry in manifest["entries"]:
        entries.setdefault(
            entry["name"],
            {
                "state": "pending",
                "attempts": 0,
                "last_iter": int(entry["expected_checkpoint_next_iter"]),
                "sent_milestones": [],
            },
        )
    state.setdefault("last_callback_at", time.time())
    return state


def send(chat_id: str, message: str) -> bool:
    request = urllib.request.Request(
        CALLBACK_URL,
        data=json.dumps({"chat_id": chat_id, "text": f"{AGENT_MENTION} {message}"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        opener.open(request, timeout=20).read()
        return True
    except OSError:
        return False


def ssh_script(host: str, script: str, arguments: Iterable[str], timeout: int = 90) -> str:
    command = "bash -s -- " + " ".join(shlex.quote(str(value)) for value in arguments)
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command],
        input=script,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return completed.stdout.strip()


def snapshot(host: str, root: str, manifest: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    entries = []
    for entry in manifest["entries"]:
        runtime = state["entries"][entry["name"]]
        entries.append(
            {
                "name": entry["name"],
                "config": entry["config"],
                "session": runtime.get("session"),
                "submit_log": runtime.get("submit_log"),
                "submitted_at": runtime.get("submitted_at", 0.0),
                "status_path": runtime.get("status_path"),
            }
        )
    payload = {
        "source_paths": list(manifest["required_source_hashes"]),
        "entries": entries,
    }
    command = "python3 - " + shlex.quote(root) + " " + shlex.quote(json.dumps(payload, separators=(",", ":")))
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command],
        input=REMOTE_SNAPSHOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    return json.loads(completed.stdout)


def idle_gpu_indices(remote: Dict[str, Any], idle_memory_mib: int) -> List[int]:
    return sorted(
        int(gpu["index"])
        for gpu in remote.get("gpus", [])
        if not gpu.get("compute_pids") and int(gpu.get("memory_used_mib", 10**9)) <= idle_memory_mib
    )


def checkpoint_budget(manifest: Dict[str, Any], state: Dict[str, Any]) -> int:
    definitions = {entry["name"]: entry for entry in manifest["entries"]}
    active_states = {"submitting", "running"}
    return sum(
        int(definitions[name]["checkpoint_bytes"])
        for name, runtime in state["entries"].items()
        if runtime.get("state") in active_states
    )


def can_admit(
    workspace_used: int,
    workspace_cap: int,
    reserve: int,
    active_checkpoint_budget: int,
    candidate_checkpoint_bytes: int,
) -> bool:
    required = reserve + active_checkpoint_budget + candidate_checkpoint_bytes
    return workspace_cap - workspace_used >= required


def remote_identity_valid(remote: Dict[str, Any], manifest: Dict[str, Any]) -> Tuple[bool, str]:
    if remote.get("git_dirty"):
        return False, "remote Git worktree is dirty"
    if remote.get("source_hashes") != manifest.get("required_source_hashes"):
        return False, "remote training source hashes do not match the checkpoint identity"
    by_name = remote.get("entries", {})
    for entry in manifest["entries"]:
        observed = by_name.get(entry["name"], {})
        if observed.get("config_sha256") != entry["config_sha256"]:
            return False, "remote queue config hash mismatch for " + entry["name"]
    return True, ""


def pending_validation(entry: Dict[str, Any], observed: Dict[str, Any]) -> Tuple[bool, str]:
    if observed.get("checkpoint_next_iter") != entry["expected_checkpoint_next_iter"]:
        return (
            False,
            f"checkpoint next_iter changed for {entry['name']}: "
            f"expected {entry['expected_checkpoint_next_iter']}, observed {observed.get('checkpoint_next_iter')}",
        )
    return True, ""


def launch_entry(
    host: str,
    root: str,
    entry: Dict[str, Any],
    gpu: int,
    attempt: int,
) -> Tuple[str, str]:
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", entry["name"])
    session = ("y400q_" + safe_name + f"_a{attempt}")[:120]
    submit_log = f"{root}/outputs/y400_ladder_runs/queue/{safe_name}_a{attempt}.submit.log"
    config = f"{root}/{REMOTE_REPO_RELATIVE}/{entry['config']}"
    ssh_script(
        host,
        REMOTE_LAUNCH,
        [root, config, str(gpu), entry["name"], session, submit_log, "true"],
    )
    return session, submit_log


def deploy_main(host: str, root: str) -> str:
    return ssh_script(host, REMOTE_DEPLOY, [f"{root}/{REMOTE_REPO_RELATIVE}"], timeout=180)


def terminal_signature(status: Dict[str, Any]) -> List[Any]:
    return [status.get("state"), status.get("exit_code"), status.get("finished_at")]


def queue_summary(
    manifest: Dict[str, Any], state: Dict[str, Any], remote: Dict[str, Any], blocker: str
) -> Dict[str, Any]:
    cap = int(manifest["workspace_cap_bytes"])
    used = int(remote.get("workspace_used_bytes", 0))
    return {
        "updated_at": time.time(),
        "host": "Y400",
        "workspace": {
            "used_bytes": used,
            "cap_bytes": cap,
            "headroom_bytes": cap - used,
        },
        "gpus": remote.get("gpus", []),
        "blocker": blocker,
        "entries": [
            {
                "priority": entry["priority"],
                "name": entry["name"],
                "stage": entry["stage"],
                "max_iters": entry["max_iters"],
                "expected_checkpoint_next_iter": entry["expected_checkpoint_next_iter"],
                **state["entries"][entry["name"]],
            }
            for entry in sorted(manifest["entries"], key=lambda item: item["priority"])
        ],
        "deferred_stages": manifest.get("deferred_stages", []),
    }


def progress_text(label: str, progress: List[Tuple[str, int, int, int]], terminals: List[Tuple[str, str, int, int, Any]]) -> str:
    parts = [f"{name} {percent}% ({current}/{maximum})" for name, percent, current, maximum in progress]
    for name, terminal_state, current, maximum, exit_code in terminals:
        if terminal_state == "finished" and exit_code == 0:
            parts.append(f"{name} 100% ({current}/{maximum}) finished exit=0")
        else:
            parts.append(f"{name} FAILED ({current}/{maximum}) exit={exit_code}")
    return label + " PROGRESS: " + " | ".join(parts)


def heartbeat_text(label: str, manifest: Dict[str, Any], state: Dict[str, Any], remote: Dict[str, Any], blocker: str) -> str:
    parts = []
    for entry in sorted(manifest["entries"], key=lambda item: item["priority"]):
        runtime = state["entries"][entry["name"]]
        parts.append(
            f"{entry['name']}: {runtime.get('state')} "
            f"iter={runtime.get('last_iter')}/{entry['max_iters']}"
        )
    used = int(remote.get("workspace_used_bytes", 0)) / (1024**3)
    cap = int(manifest["workspace_cap_bytes"]) / (1024**3)
    suffix = f" | workspace={used:.1f}/{cap:.0f}GiB"
    if blocker:
        suffix += " | blocked=" + blocker
    return label + " HEARTBEAT: " + " | ".join(parts) + suffix


def run_once(args: argparse.Namespace, manifest: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    remote = snapshot(args.host, args.remote_root, manifest, state)
    progress_events: List[Tuple[str, int, int, int]] = []
    terminal_events: List[Tuple[str, str, int, int, Any]] = []
    error_parts: List[str] = []

    for entry in manifest["entries"]:
        runtime = state["entries"][entry["name"]]
        observed = remote["entries"].get(entry["name"], {})
        if observed.get("status_path"):
            runtime["status_path"] = observed["status_path"]
        current = observed.get("last_iter")
        if current is not None:
            previous = int(runtime.get("last_iter", entry["expected_checkpoint_next_iter"]))
            current = int(current)
            sent = set(runtime.get("sent_milestones", []))
            for percent in (20, 50):
                if percent not in sent and previous * 100 < percent * entry["max_iters"] <= current * 100:
                    progress_events.append((entry["name"], percent, current, entry["max_iters"]))
            if current > previous:
                runtime["last_progress_at"] = now
            runtime["last_iter"] = current

        status = observed.get("status", {})
        status_state = str(status.get("state", "")).lower()
        if status_state == "running":
            runtime["state"] = "running"
            runtime["pgid"] = status.get("pgid")
            runtime["gpu"] = status.get("gpu", runtime.get("gpu"))
            if observed.get("alive"):
                runtime.pop("missing_since", None)
            else:
                missing_since = float(runtime.setdefault("missing_since", now))
                if now - missing_since >= args.missing_grace_minutes * 60 and not runtime.get("external_failure_notified"):
                    runtime["state"] = "failed_external"
                    runtime["terminal_pending"] = ["failed_external", None, now]
                    error_parts.append(
                        f"{entry['name']}: process group missing at {runtime.get('last_iter')}/{entry['max_iters']}"
                    )
        elif status_state in {"finished", "failed"}:
            signature = terminal_signature(status)
            runtime["state"] = status_state
            if runtime.get("terminal_signature") != signature:
                runtime["terminal_pending"] = signature
        elif runtime.get("state") == "submitting":
            submitted_at = float(runtime.get("submitted_at", now))
            if not observed.get("tmux_alive") and now - submitted_at >= args.submission_grace_minutes * 60:
                runtime["state"] = "failed_submission"
                runtime["terminal_pending"] = ["failed_submission", None, now]
                error_parts.append(entry["name"] + ": launcher/preflight session disappeared before status publication")

        pending_terminal = runtime.get("terminal_pending")
        if pending_terminal and not runtime.get("terminal_notified"):
            terminal_state = str(pending_terminal[0])
            exit_code = pending_terminal[1]
            terminal_events.append(
                (entry["name"], terminal_state, int(runtime.get("last_iter", 0)), int(entry["max_iters"]), exit_code)
            )

    if error_parts and send(args.chat_id, manifest["label"] + " ERROR: " + " | ".join(error_parts)):
        for entry in manifest["entries"]:
            runtime = state["entries"][entry["name"]]
            if runtime.get("state") in {"failed_external", "failed_submission"}:
                runtime["external_failure_notified"] = True
        state["last_callback_at"] = now

    if progress_events or terminal_events:
        if send(args.chat_id, progress_text(manifest["label"], progress_events, terminal_events)):
            for name, percent, _current, _maximum in progress_events:
                runtime = state["entries"][name]
                runtime["sent_milestones"] = sorted(set(runtime.get("sent_milestones", [])) | {percent})
            for name, _terminal_state, _current, _maximum, _exit_code in terminal_events:
                runtime = state["entries"][name]
                runtime["terminal_notified"] = True
                runtime["terminal_signature"] = runtime.get("terminal_pending")
            state["last_callback_at"] = now

    identity_ok, identity_reason = remote_identity_valid(remote, manifest)
    idle = idle_gpu_indices(remote, int(manifest["gpu_idle_memory_mib"]))
    pending = [
        entry
        for entry in sorted(manifest["entries"], key=lambda item: item["priority"])
        if state["entries"][entry["name"]].get("state") == "pending"
        and int(state["entries"][entry["name"]].get("attempts", 0)) < int(entry["max_attempts"])
    ]
    used = int(remote["workspace_used_bytes"])
    cap = int(manifest["workspace_cap_bytes"])
    reserve = int(manifest["workspace_reserve_bytes"])
    active_budget = checkpoint_budget(manifest, state)
    blocker_parts: List[str] = []
    capacity_ok = not pending or can_admit(
        used, cap, reserve, active_budget, int(pending[0]["checkpoint_bytes"])
    )
    if pending and not capacity_ok:
        required = reserve + active_budget + int(pending[0]["checkpoint_bytes"])
        blocker_parts.append(
            f"checkpoint headroom {(cap-used)/(1024**3):.1f}GiB < required {required/(1024**3):.1f}GiB"
        )
    if pending and not idle:
        blocker_parts.append("no exclusive GPU is free")
    if pending and not identity_ok:
        blocker_parts.append(identity_reason)
    blocker = "; ".join(blocker_parts)

    if pending and idle and capacity_ok and not identity_ok and not args.dry_run:
        try:
            deploy_main(args.host, args.remote_root)
            remote = snapshot(args.host, args.remote_root, manifest, state)
            identity_ok, identity_reason = remote_identity_valid(remote, manifest)
            if not identity_ok:
                blocker = identity_reason
            else:
                idle = idle_gpu_indices(remote, int(manifest["gpu_idle_memory_mib"]))
                blocker = "" if idle else "no exclusive GPU is free"
        except (subprocess.SubprocessError, OSError) as exc:
            blocker = "remote Git deployment failed: " + type(exc).__name__

    launched: List[Tuple[str, int]] = []
    if not args.dry_run and identity_ok and capacity_ok and idle and not blocker:
        for entry, gpu in zip(pending, idle):
            runtime = state["entries"][entry["name"]]
            valid, reason = pending_validation(entry, remote["entries"].get(entry["name"], {}))
            if not valid:
                blocker = reason
                break
            active_budget = checkpoint_budget(manifest, state)
            if not can_admit(used, cap, reserve, active_budget, int(entry["checkpoint_bytes"])):
                blocker = "insufficient checkpoint headroom for another concurrent admission"
                break
            attempt = int(runtime.get("attempts", 0)) + 1
            try:
                session, submit_log = launch_entry(args.host, args.remote_root, entry, gpu, attempt)
            except (subprocess.SubprocessError, OSError) as exc:
                runtime["state"] = "failed_submission"
                runtime["attempts"] = attempt
                runtime["terminal_pending"] = ["failed_submission", None, now]
                blocker = f"submission failed for {entry['name']}: {type(exc).__name__}"
                break
            runtime.update(
                {
                    "state": "submitting",
                    "attempts": attempt,
                    "gpu": gpu,
                    "session": session,
                    "submit_log": submit_log,
                    "submitted_at": now,
                    "last_progress_at": now,
                }
            )
            launched.append((entry["name"], gpu))
        if launched:
            message = manifest["label"] + " SUBMITTED: " + " | ".join(
                f"{name} GPU{gpu}" for name, gpu in launched
            )
            if send(args.chat_id, message):
                state["last_callback_at"] = now

    if not blocker and pending and not launched:
        blocker = "waiting for the next admission poll"
    state["blocker"] = blocker
    state["remote_git_commit"] = remote.get("git_commit")
    state["updated_at"] = now

    if now - float(state.get("last_callback_at", now)) >= args.heartbeat_minutes * 60:
        if send(args.chat_id, heartbeat_text(manifest["label"], manifest, state, remote, blocker)):
            state["last_callback_at"] = now

    summary = queue_summary(manifest, state, remote, blocker)
    atomic_json(args.status_path, summary)
    atomic_json(args.state_path, state)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="Y400")
    parser.add_argument("--remote-root", default="/root/userdata/MappingNetworks")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--heartbeat-minutes", type=int, default=90)
    parser.add_argument("--missing-grace-minutes", type=int, default=2)
    parser.add_argument("--submission-grace-minutes", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.interval < 15 or args.heartbeat_minutes < 1:
        parser.error("interval must be >=15 seconds and heartbeat must be >=1 minute")
    manifest = load_manifest(args.queue.resolve(), args.repo.resolve())
    state = load_state(args.state_path, manifest)
    while True:
        try:
            run_once(args, manifest, state)
        except Exception as exc:
            now = time.time()
            state["last_probe_error"] = {"type": type(exc).__name__, "at": now}
            atomic_json(args.state_path, state)
            if now - float(state.get("last_error_callback_at", 0.0)) >= args.heartbeat_minutes * 60:
                if send(args.chat_id, manifest["label"] + " MONITOR_DEGRADED: remote queue probe failed; retrying"):
                    state["last_error_callback_at"] = now
                    state["last_callback_at"] = now
                    atomic_json(args.state_path, state)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
