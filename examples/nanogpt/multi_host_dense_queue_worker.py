#!/usr/bin/env python3
"""One claim-once dense queue spanning Y400 and PRO6."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from . import y400_dense_queue_worker as base
except ImportError:  # Direct script execution places this directory on sys.path.
    import y400_dense_queue_worker as base


REMOTE_LAUNCH = r'''set -euo pipefail
root="$1"; python_bin="$2"; config="$3"; gpu="$4"; run_name="$5"; session="$6"; submit_log="$7"; resume="$8"; launch_mode="$9"
repo="$root/latent-weight-lab"
launcher="$repo/examples/nanogpt/launch_y400_ladder_detached.sh"
mkdir -p "$(dirname "$submit_log")"
case "$launch_mode" in
  tmux)
    command -v tmux >/dev/null 2>&1 || { echo "tmux launch mode requested but tmux is unavailable" >&2; exit 2; }
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "refusing duplicate queue session: $session" >&2
      exit 2
    fi
    printf -v command_line 'cd %q && export PYTHON_BIN=%q && exec bash %q --foreground' "$repo" "$python_bin" "$launcher"
    if [[ "$resume" == "true" ]]; then command_line+=' --resume'; fi
    printf -v command_line '%s %q %q %q %q >%q 2>&1' "$command_line" "$config" "$gpu" "$run_name" "$root" "$submit_log"
    tmux new-session -d -s "$session" "$command_line"
    ;;
  detached)
    args=()
    if [[ "$resume" == "true" ]]; then args+=(--resume); fi
    cd "$repo"
    PYTHON_BIN="$python_bin" bash "$launcher" "${args[@]}" "$config" "$gpu" "$run_name" "$root" >"$submit_log" 2>&1
    ;;
  *)
    echo "unsupported launch mode: $launch_mode" >&2
    exit 2
    ;;
esac
'''


def load_manifest(path: Path, repo: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "multi_host_dense_queue_v1":
        raise ValueError("unsupported multi-host queue schema")
    names = set()
    for task in payload.get("entries", []):
        name = task.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name in names:
            raise ValueError("invalid or duplicate task name")
        names.add(name)
        if set(task.get("host_preference", [])) - set(payload["hosts"]):
            raise ValueError("task host preference references an unknown host: " + name)
        for host, variant in task.get("variants", {}).items():
            if host not in payload["hosts"]:
                raise ValueError("task variant references an unknown host: " + host)
            config = repo / variant["config"]
            if not config.is_file() or base.file_sha256(config) != variant["config_sha256"]:
                raise ValueError("variant config hash mismatch: " + str(config))
            config_payload = json.loads(config.read_text())
            if config_payload.get("mfu_preflight_required") is not True:
                raise ValueError("variant lacks mandatory MFU preflight: " + name + "/" + host)
            if float(config_payload.get("mfu_min_fraction", 0.0)) < 0.20:
                raise ValueError("variant MFU floor is below 20%: " + name + "/" + host)
            if int(variant.get("checkpoint_budget_bytes", 0)) <= 0:
                raise ValueError("variant lacks checkpoint budget: " + name + "/" + host)
    for relative, expected in payload["required_source_hashes"].items():
        if base.file_sha256(repo / relative) != expected:
            raise ValueError("local registered source hash mismatch: " + relative)
    return payload


def load_state(path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        state = {}
    entries = state.setdefault("entries", {})
    for task in manifest["entries"]:
        y400 = task["variants"].get("Y400", {})
        initial = y400.get("expected_checkpoint_next_iter")
        entries.setdefault(
            task["name"],
            {
                "state": "pending",
                "assigned_host": None,
                "last_iter": 0 if initial is None else int(initial),
                "sent_milestones": [],
                "attempts_by_host": {},
                "rejected_hosts": [],
            },
        )
    state.setdefault("last_callback_at", time.time())
    return state


def task_by_name(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {task["name"]: task for task in manifest["entries"]}


def host_probe_manifest(manifest: Dict[str, Any], host: str) -> Dict[str, Any]:
    entries = []
    for task in manifest["entries"]:
        variant = task.get("variants", {}).get(host)
        if variant:
            entries.append(
                {"name": variant["run_name"], "config": variant["config"], "config_sha256": variant["config_sha256"]}
            )
    return {"required_source_hashes": manifest["required_source_hashes"], "entries": entries}


def host_probe_state(manifest: Dict[str, Any], state: Dict[str, Any], host: str) -> Dict[str, Any]:
    entries: Dict[str, Any] = {}
    for task in manifest["entries"]:
        variant = task.get("variants", {}).get(host)
        if not variant:
            continue
        runtime = state["entries"][task["name"]]
        if runtime.get("assigned_host") == host:
            entries[variant["run_name"]] = runtime
        else:
            entries[variant["run_name"]] = {}
    return {"entries": entries}


def probe_all(manifest: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    snapshots = {}
    for host, definition in manifest["hosts"].items():
        snapshots[host] = base.snapshot(
            host,
            definition["root"],
            host_probe_manifest(manifest, host),
            host_probe_state(manifest, state, host),
        )
    return snapshots


def observed_for(
    task: Dict[str, Any], runtime: Dict[str, Any], snapshots: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    host = runtime.get("assigned_host")
    if not host:
        return {}
    run_name = task["variants"][host]["run_name"]
    return snapshots[host].get("entries", {}).get(run_name, {})


def host_identity_valid(manifest: Dict[str, Any], host: str, remote: Dict[str, Any]) -> Tuple[bool, str]:
    return base.remote_identity_valid(remote, host_probe_manifest(manifest, host))


def active_budget(manifest: Dict[str, Any], state: Dict[str, Any], host: str) -> int:
    tasks = task_by_name(manifest)
    return sum(
        int(tasks[name]["variants"][host]["checkpoint_budget_bytes"])
        for name, runtime in state["entries"].items()
        if runtime.get("assigned_host") == host and runtime.get("state") in {"submitting", "running"}
    )


def validate_pending_variant(variant: Dict[str, Any], observed: Dict[str, Any]) -> Tuple[bool, str]:
    expected = variant.get("expected_checkpoint_next_iter")
    actual = observed.get("checkpoint_next_iter")
    if variant.get("resume") is True and actual != expected:
        return False, f"resume checkpoint next_iter mismatch: expected {expected}, observed {actual}"
    if variant.get("resume") is not True and actual is not None:
        return False, f"fresh lineage output already contains checkpoint next_iter={actual}"
    return True, ""


def launch(
    host: str,
    host_definition: Dict[str, Any],
    task_name: str,
    variant: Dict[str, Any],
    gpu: int,
    attempt: int,
) -> Tuple[str, str]:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", variant["run_name"])
    launch_mode = host_definition.get("launch_mode", "tmux")
    session = ("denseq_" + safe + f"_a{attempt}")[:120] if launch_mode == "tmux" else ""
    root = host_definition["root"]
    submit_log = f"{root}/outputs/y400_ladder_runs/queue/{safe}_a{attempt}.submit.log"
    config = f"{root}/latent-weight-lab/{variant['config']}"
    python_bin = f"{root}/{host_definition['python_relative']}"
    base.ssh_script(
        host,
        REMOTE_LAUNCH,
        [
            root,
            python_bin,
            config,
            str(gpu),
            variant["run_name"],
            session,
            submit_log,
            str(bool(variant.get("resume"))).lower(),
            launch_mode,
        ],
    )
    return session, submit_log


def progress_text(
    label: str,
    progress: List[Tuple[str, str, int, int, int]],
    terminals: List[Tuple[str, str, str, int, int, Any]],
) -> str:
    parts = [f"{name}@{host} {percent}% ({current}/{maximum})" for name, host, percent, current, maximum in progress]
    for name, host, terminal_state, current, maximum, exit_code in terminals:
        if terminal_state == "finished" and exit_code == 0:
            parts.append(f"{name}@{host} 100% ({current}/{maximum}) finished exit=0")
        else:
            parts.append(f"{name}@{host} FAILED ({current}/{maximum}) exit={exit_code}")
    return label + " PROGRESS: " + " | ".join(parts)


def heartbeat_text(
    manifest: Dict[str, Any], state: Dict[str, Any], snapshots: Dict[str, Dict[str, Any]], blockers: Dict[str, str]
) -> str:
    tasks = []
    for task in sorted(manifest["entries"], key=lambda item: item["priority"]):
        runtime = state["entries"][task["name"]]
        host = runtime.get("assigned_host") or "unassigned"
        tasks.append(
            f"{task['name']}@{host}: {runtime.get('state')} iter={runtime.get('last_iter')}/{task['max_iters']}"
        )
    hosts = []
    for host, definition in manifest["hosts"].items():
        used = int(snapshots[host].get("workspace_used_bytes", 0)) / (1024**3)
        cap = int(definition["workspace_cap_bytes"]) / (1024**3)
        hosts.append(f"{host}={used:.1f}/{cap:.0f}GiB blocked={blockers.get(host) or 'none'}")
    return manifest["label"] + " HEARTBEAT: " + " | ".join(tasks + hosts)


def status_summary(
    manifest: Dict[str, Any], state: Dict[str, Any], snapshots: Dict[str, Dict[str, Any]], blockers: Dict[str, str]
) -> Dict[str, Any]:
    return {
        "updated_at": time.time(),
        "label": manifest["label"],
        "hosts": {
            host: {
                "workspace_used_bytes": snapshots[host].get("workspace_used_bytes"),
                "workspace_cap_bytes": definition["workspace_cap_bytes"],
                "gpus": snapshots[host].get("gpus", []),
                "git_commit": snapshots[host].get("git_commit"),
                "blocker": blockers.get(host, ""),
            }
            for host, definition in manifest["hosts"].items()
        },
        "entries": [
            {"priority": task["priority"], "name": task["name"], "max_iters": task["max_iters"], **state["entries"][task["name"]]}
            for task in sorted(manifest["entries"], key=lambda item: item["priority"])
        ],
        "deferred_stages": manifest.get("deferred_stages", []),
    }


def run_once(args: argparse.Namespace, manifest: Dict[str, Any], state: Dict[str, Any]) -> None:
    now = time.time()
    snapshots = probe_all(manifest, state)
    tasks = task_by_name(manifest)
    progress_events: List[Tuple[str, str, int, int, int]] = []
    terminal_events: List[Tuple[str, str, str, int, int, Any]] = []

    for name, runtime in state["entries"].items():
        if runtime.get("state") not in {"submitting", "running"}:
            if runtime.get("terminal_pending") and not runtime.get("terminal_notified"):
                pending = runtime["terminal_pending"]
                terminal_events.append((name, runtime.get("assigned_host") or "unknown", pending[0], int(runtime.get("last_iter", 0)), int(tasks[name]["max_iters"]), pending[1]))
            continue
        task = tasks[name]
        host = runtime["assigned_host"]
        observed = observed_for(task, runtime, snapshots)
        if observed.get("status_path"):
            runtime["status_path"] = observed["status_path"]
        current = observed.get("last_iter")
        if current is not None:
            previous = int(runtime.get("last_iter", 0))
            current = int(current)
            sent = set(runtime.get("sent_milestones", []))
            for percent in (20, 50):
                if percent not in sent and previous * 100 < percent * task["max_iters"] <= current * 100:
                    progress_events.append((name, host, percent, current, task["max_iters"]))
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
                if now - missing_since >= args.missing_grace_minutes * 60:
                    runtime["state"] = "failed_external"
                    runtime["terminal_pending"] = ["failed_external", None, now]
        elif status_state in {"finished", "failed"}:
            signature = base.terminal_signature(status)
            runtime["state"] = status_state
            if runtime.get("terminal_signature") != signature:
                runtime["terminal_pending"] = signature
        elif runtime.get("state") == "submitting":
            submitted_at = float(runtime.get("submitted_at", now))
            if not observed.get("tmux_alive") and now - submitted_at >= args.submission_grace_minutes * 60:
                rejected = set(runtime.get("rejected_hosts", []))
                rejected.add(host)
                runtime.update(
                    {
                        "state": "pending",
                        "assigned_host": None,
                        "rejected_hosts": sorted(rejected),
                        "submission_error_pending": host,
                    }
                )
        if runtime.get("terminal_pending") and not runtime.get("terminal_notified"):
            pending = runtime["terminal_pending"]
            terminal_events.append((name, host, pending[0], int(runtime.get("last_iter", 0)), int(task["max_iters"]), pending[1]))

    submission_errors = [
        (name, runtime.pop("submission_error_pending"))
        for name, runtime in state["entries"].items()
        if runtime.get("submission_error_pending")
    ]
    if submission_errors:
        base.send(
            args.chat_id,
            manifest["label"] + " ERROR: " + " | ".join(f"{name}@{host} preflight/launcher failed; host variant rejected" for name, host in submission_errors),
        )
        state["last_callback_at"] = now

    if progress_events or terminal_events:
        if base.send(args.chat_id, progress_text(manifest["label"], progress_events, terminal_events)):
            for name, _host, percent, _current, _maximum in progress_events:
                runtime = state["entries"][name]
                runtime["sent_milestones"] = sorted(set(runtime.get("sent_milestones", [])) | {percent})
            for name, _host, _terminal_state, _current, _maximum, _exit_code in terminal_events:
                runtime = state["entries"][name]
                runtime["terminal_notified"] = True
                runtime["terminal_signature"] = runtime.get("terminal_pending")
            state["last_callback_at"] = now

    idle_by_host: Dict[str, List[int]] = {}
    identity_by_host: Dict[str, Tuple[bool, str]] = {}
    blockers: Dict[str, str] = {}
    for host, definition in manifest["hosts"].items():
        idle_by_host[host] = base.idle_gpu_indices(snapshots[host], int(definition["gpu_idle_memory_mib"]))
        identity_by_host[host] = host_identity_valid(manifest, host, snapshots[host])

    launched: List[Tuple[str, str, int]] = []
    for task in sorted(manifest["entries"], key=lambda item: item["priority"]):
        runtime = state["entries"][task["name"]]
        if runtime.get("state") != "pending":
            continue
        rejected = set(runtime.get("rejected_hosts", []))
        for host in task["host_preference"]:
            if host in rejected or host not in task["variants"] or not idle_by_host.get(host):
                continue
            definition = manifest["hosts"][host]
            variant = task["variants"][host]
            used = int(snapshots[host]["workspace_used_bytes"])
            if not base.can_admit(
                used,
                int(definition["workspace_cap_bytes"]),
                int(definition["workspace_reserve_bytes"]),
                active_budget(manifest, state, host),
                int(variant["checkpoint_budget_bytes"]),
            ):
                continue
            identity_ok, _reason = identity_by_host[host]
            if not identity_ok:
                try:
                    base.deploy_main(host, definition["root"])
                    snapshots[host] = base.snapshot(
                        host,
                        definition["root"],
                        host_probe_manifest(manifest, host),
                        host_probe_state(manifest, state, host),
                    )
                    identity_by_host[host] = host_identity_valid(manifest, host, snapshots[host])
                    identity_ok = identity_by_host[host][0]
                except (OSError, subprocess.SubprocessError):
                    identity_ok = False
            if not identity_ok:
                continue
            observed = snapshots[host]["entries"].get(variant["run_name"], {})
            valid, reason = validate_pending_variant(variant, observed)
            if not valid:
                blockers[host] = reason
                continue
            attempt = int(runtime.get("attempts_by_host", {}).get(host, 0)) + 1
            try:
                session, submit_log = launch(host, definition, task["name"], variant, idle_by_host[host][0], attempt)
            except (OSError, subprocess.SubprocessError):
                rejected.add(host)
                runtime["rejected_hosts"] = sorted(rejected)
                continue
            runtime.setdefault("attempts_by_host", {})[host] = attempt
            runtime.update(
                {
                    "state": "submitting",
                    "assigned_host": host,
                    "run_name": variant["run_name"],
                    "gpu": idle_by_host[host].pop(0),
                    "session": session,
                    "submit_log": submit_log,
                    "submitted_at": now,
                    "last_progress_at": now,
                }
            )
            launched.append((task["name"], host, runtime["gpu"]))
            break

    for host, definition in manifest["hosts"].items():
        parts = []
        if not idle_by_host.get(host):
            parts.append("no exclusive GPU is free")
        identity_ok, reason = identity_by_host[host]
        if not identity_ok:
            parts.append(reason)
        blockers.setdefault(host, "; ".join(parts))

    if launched and base.send(
        args.chat_id,
        manifest["label"] + " SUBMITTED: " + " | ".join(f"{name}@{host} GPU{gpu}" for name, host, gpu in launched),
    ):
        state["last_callback_at"] = now

    if now - float(state.get("last_callback_at", now)) >= args.heartbeat_minutes * 60:
        if base.send(args.chat_id, heartbeat_text(manifest, state, snapshots, blockers)):
            state["last_callback_at"] = now
    state["updated_at"] = now
    state["host_blockers"] = blockers
    base.atomic_json(args.state_path, state)
    base.atomic_json(args.status_path, status_summary(manifest, state, snapshots, blockers))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--state-path", required=True, type=Path)
    parser.add_argument("--status-path", required=True, type=Path)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--heartbeat-minutes", type=int, default=90)
    parser.add_argument("--missing-grace-minutes", type=int, default=2)
    parser.add_argument("--submission-grace-minutes", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.interval < 15:
        parser.error("interval must be >=15")
    manifest = load_manifest(args.queue.resolve(), args.repo.resolve())
    state = load_state(args.state_path, manifest)
    while True:
        try:
            if args.dry_run:
                snapshots = probe_all(manifest, state)
                blockers = {host: "dry-run" for host in manifest["hosts"]}
                base.atomic_json(args.status_path, status_summary(manifest, state, snapshots, blockers))
            else:
                run_once(args, manifest, state)
        except Exception as exc:
            now = time.time()
            state["last_probe_error"] = {"type": type(exc).__name__, "at": now}
            base.atomic_json(args.state_path, state)
            if now - float(state.get("last_error_callback_at", 0.0)) >= args.heartbeat_minutes * 60:
                if base.send(args.chat_id, manifest["label"] + " MONITOR_DEGRADED: multi-host probe failed; retrying"):
                    state["last_error_callback_at"] = now
                    state["last_callback_at"] = now
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
