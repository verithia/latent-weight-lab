from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.nanogpt.multi_host_dense_queue_worker import (
    active_budget,
    heartbeat_text,
    host_admission_status,
    host_probe_manifest,
    launch,
    load_manifest,
    load_state,
    mark_probe_failure_notified,
    progress_text,
    record_probe_failure,
    record_probe_recovery,
    stalled_run_event,
    stall_text,
    submitted_text,
    validate_pending_variant,
)
from examples.nanogpt.y400_dense_queue_worker import REMOTE_SNAPSHOT
from unittest import mock


class MultiHostDenseQueueWorkerTest(unittest.TestCase):
    GIB = 1024**3

    def test_remote_snapshot_rejects_unavailable_workspace(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            completed = subprocess.run(
                [sys.executable, "-", str(missing), '{"source_paths":[],"entries":[]}'],
                input=REMOTE_SNAPSHOT,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("workspace unavailable", completed.stderr)

    def test_load_state_preserves_operator_host_pause(self) -> None:
        manifest = {"entries": [{"name": "task", "variants": {"Y400": {}}}]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text('{"paused_hosts": ["Y400"]}')
            state = load_state(path, manifest)
        self.assertEqual(state["paused_hosts"], ["Y400"])
        self.assertEqual(state["entries"]["task"]["state"], "pending")

    def test_manifest_remote_identity_need_not_match_local_control_plane(self) -> None:
        with TemporaryDirectory() as directory:
            repo = Path(directory)
            config = repo / "config.json"
            config.write_text(
                json.dumps({"mfu_preflight_required": True, "mfu_min_fraction": 0.20})
            )
            queue = repo / "queue.json"
            queue.write_text(
                json.dumps(
                    {
                        "schema_version": "multi_host_dense_queue_v1",
                        "required_source_hashes": {"train.py": "a" * 64},
                        "hosts": {"Y400": {"launch_mode": "tmux-isolated"}},
                        "entries": [
                            {
                                "name": "task",
                                "host_preference": ["Y400"],
                                "variants": {
                                    "Y400": {
                                        "config": "config.json",
                                        "config_sha256": hashlib.sha256(
                                            config.read_bytes()
                                        ).hexdigest(),
                                        "checkpoint_budget_bytes": 1,
                                    }
                                },
                            }
                        ],
                    }
                )
            )
            manifest = load_manifest(queue, repo)
        self.assertEqual(manifest["required_source_hashes"], {"train.py": "a" * 64})

    def test_load_state_migrates_existing_probe_outage_without_realerting(self) -> None:
        manifest = {"entries": [{"name": "task", "variants": {"Y400": {}}}]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(
                '{"last_probe_error":{"type":"RuntimeError","at":100},'
                '"last_error_callback_at":100}'
            )
            state = load_state(path, manifest)
        self.assertTrue(state["probe_outage_active"])
        self.assertTrue(state["probe_outage_notified"])

    def test_probe_outage_is_edge_triggered_and_recovery_is_one_shot(self) -> None:
        state = {}
        self.assertTrue(record_probe_failure(state, RuntimeError("down"), 100.0))
        mark_probe_failure_notified(state, 100.0)
        self.assertFalse(record_probe_failure(state, RuntimeError("still down"), 200.0))
        self.assertEqual(state["last_probe_error"]["at"], 200.0)
        self.assertTrue(record_probe_recovery(state))
        self.assertNotIn("probe_outage_active", state)
        self.assertNotIn("last_probe_error", state)
        self.assertFalse(record_probe_recovery(state))

    def test_failed_degraded_callback_is_retried(self) -> None:
        state = {}
        self.assertTrue(record_probe_failure(state, RuntimeError("down"), 100.0))
        self.assertTrue(record_probe_failure(state, RuntimeError("still down"), 200.0))

    def test_one_global_assignment_counts_budget_only_on_assigned_host(self) -> None:
        manifest = {
            "entries": [
                {
                    "name": "task",
                    "variants": {
                        "Y400": {"checkpoint_budget_bytes": 10},
                        "PRO6": {"checkpoint_budget_bytes": 20},
                    },
                }
            ]
        }
        state = {
            "entries": {
                "task": {"state": "running", "assigned_host": "PRO6"},
            }
        }
        self.assertEqual(active_budget(manifest, state, "Y400"), 0)
        self.assertEqual(active_budget(manifest, state, "PRO6"), 20)

    def test_active_budget_uses_one_copy_only_after_checkpoint_is_observed(self) -> None:
        manifest = {
            "entries": [
                {
                    "name": "task",
                    "variants": {
                        "Y400": {
                            "run_name": "run",
                            "checkpoint_budget_bytes": 20,
                            "active_checkpoint_budget_bytes": 11,
                        }
                    },
                }
            ]
        }
        running = {"entries": {"task": {"state": "running", "assigned_host": "Y400"}}}
        no_checkpoint = {"entries": {"run": {"checkpoint_next_iter": None}}}
        checkpoint = {"entries": {"run": {"checkpoint_next_iter": 0}}}
        self.assertEqual(active_budget(manifest, running, "Y400", no_checkpoint), 20)
        self.assertEqual(active_budget(manifest, running, "Y400", checkpoint), 11)
        submitting = {"entries": {"task": {"state": "submitting", "assigned_host": "Y400"}}}
        self.assertEqual(active_budget(manifest, submitting, "Y400", checkpoint), 20)

    def test_resume_requires_exact_checkpoint_but_fresh_requires_empty_output(self) -> None:
        resume = {"resume": True, "expected_checkpoint_next_iter": 2196}
        self.assertEqual(validate_pending_variant(resume, {"checkpoint_next_iter": 2196}), (True, ""))
        self.assertFalse(validate_pending_variant(resume, {"checkpoint_next_iter": 0})[0])
        fresh = {"resume": False, "expected_checkpoint_next_iter": None}
        self.assertEqual(validate_pending_variant(fresh, {"checkpoint_next_iter": None}), (True, ""))
        self.assertFalse(validate_pending_variant(fresh, {"checkpoint_next_iter": 0})[0])

    def test_admission_honors_policy_cap_and_physical_free_space(self) -> None:
        definition = {
            "workspace_cap_bytes": 256 * self.GIB,
            "workspace_reserve_bytes": 8 * self.GIB,
        }
        admitted = host_admission_status(
            {"workspace_used_bytes": 100 * self.GIB, "filesystem_available_bytes": 20 * self.GIB},
            definition,
            0,
            6 * self.GIB,
        )
        self.assertEqual(admitted, (True, ""))
        physical = host_admission_status(
            {"workspace_used_bytes": 100 * self.GIB, "filesystem_available_bytes": 10 * self.GIB},
            definition,
            0,
            6 * self.GIB,
        )
        self.assertFalse(physical[0])
        self.assertIn("physical free", physical[1])
        policy = host_admission_status(
            {"workspace_used_bytes": 250 * self.GIB, "filesystem_available_bytes": 100 * self.GIB},
            definition,
            0,
            1 * self.GIB,
        )
        self.assertFalse(policy[0])
        self.assertIn("workspace headroom", policy[1])

    def test_submission_callback_names_attempt_identity(self) -> None:
        self.assertEqual(
            submitted_text("dense queue", [("top1", "Y400", 0, 2), ("top2", "Y400", 1, 3)]),
            "dense queue SUBMITTED: top1@Y400 GPU0 attempt=2 | top2@Y400 GPU1 attempt=3",
        )

    def test_progress_callback_names_attempt_identity(self) -> None:
        self.assertEqual(
            progress_text(
                "dense queue",
                [("top1", "Y400", 3, 20, 400, 2000)],
                [("top2", "Y400", 2, "failed_external", 0, 2000, None)],
            ),
            "dense queue PROGRESS: top1@Y400 attempt=3 20% (400/2000) | "
            "top2@Y400 attempt=2 FAILED (0/2000) exit=None",
        )

    def test_stall_callback_is_alive_only_deduplicated_and_attempt_scoped(self) -> None:
        runtime = {
            "state": "running",
            "last_iter": 400,
            "last_progress_at": 100.0,
        }
        event = stalled_run_event(
            "top1",
            "Y400",
            3,
            runtime,
            {"alive": True},
            2000,
            1000.0,
            15 * 60,
        )
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(
            stall_text("dense queue", [event]),
            "dense queue STALL: top1@Y400 attempt=3 no iteration progress for 900s "
            "(400/2000); process remains alive",
        )
        runtime["stall_notified_marker"] = event[-1]
        self.assertIsNone(
            stalled_run_event("top1", "Y400", 3, runtime, {"alive": True}, 2000, 1100.0, 15 * 60)
        )
        self.assertIsNone(
            stalled_run_event("top1", "Y400", 3, runtime, {"alive": False}, 2000, 1200.0, 15 * 60)
        )
        self.assertIsNotNone(
            stalled_run_event("top1", "Y400", 4, runtime, {"alive": True}, 2000, 1200.0, 15 * 60)
        )

    def test_heartbeat_names_active_attempt_identity(self) -> None:
        manifest = {
            "label": "dense queue",
            "hosts": {"Y400": {"workspace_cap_bytes": self.GIB}},
            "entries": [
                {"priority": 1, "name": "top1", "max_iters": 2000},
                {"priority": 2, "name": "old", "max_iters": 1000},
            ],
        }
        state = {
            "entries": {
                "top1": {
                    "assigned_host": "Y400",
                    "attempts_by_host": {"Y400": 3},
                    "state": "running",
                    "last_iter": 400,
                },
                "old": {
                    "assigned_host": "Y400",
                    "attempts_by_host": {"Y400": 1},
                    "state": "finished",
                    "last_iter": 1000,
                },
            }
        }
        snapshots = {"Y400": {"workspace_used_bytes": 0, "filesystem_available_bytes": 2 * self.GIB}}
        heartbeat = heartbeat_text(manifest, state, snapshots, {"Y400": ""})
        self.assertIn(
            "top1@Y400 attempt=3: running iter=400/2000",
            heartbeat,
        )
        self.assertNotIn("old@Y400", heartbeat)
        self.assertIn("totals=running:1,submitting:0,pending:0,finished:1,failed:0", heartbeat)

    @mock.patch("examples.nanogpt.multi_host_dense_queue_worker.base.ssh_script")
    def test_detached_host_does_not_publish_a_tmux_session(self, ssh_script: mock.Mock) -> None:
        session, _ = launch(
            "PRO6",
            {"root": "/remote", "python_relative": ".venv/bin/python", "launch_mode": "detached"},
            "task",
            {"run_name": "run", "config": "config.json", "resume": False},
            0,
            1,
        )
        self.assertEqual(session, "")
        self.assertEqual(ssh_script.call_args.args[2][-2], "detached")
        self.assertEqual(ssh_script.call_args.args[2][-1], "latent-weight-lab")

    @mock.patch("examples.nanogpt.multi_host_dense_queue_worker.base.ssh_script")
    def test_isolated_tmux_host_publishes_per_attempt_socket(self, ssh_script: mock.Mock) -> None:
        session, _ = launch(
            "Y400",
            {"root": "/remote", "python_relative": ".venv/bin/python", "launch_mode": "tmux-isolated"},
            "task",
            {"run_name": "run", "config": "config.json", "resume": True},
            2,
            4,
        )
        self.assertRegex(session, r"^tmuxl:denseq_[0-9a-f]{16}:run$")
        self.assertEqual(ssh_script.call_args.args[2][5], session)
        self.assertEqual(ssh_script.call_args.args[2][-2], "tmux-isolated")
        self.assertEqual(ssh_script.call_args.args[2][-1], "latent-weight-lab")

    @mock.patch("examples.nanogpt.multi_host_dense_queue_worker.base.ssh_script")
    def test_launch_can_use_an_isolated_registered_worktree(self, ssh_script: mock.Mock) -> None:
        launch(
            "Y400",
            {"root": "/remote", "python_relative": ".venv/bin/python", "launch_mode": "tmux-isolated"},
            "task",
            {
                "run_name": "run",
                "config": "config.json",
                "resume": False,
                "repo_relative": "latent-weight-lab-spectral64",
            },
            3,
            2,
        )
        arguments = ssh_script.call_args.args[2]
        self.assertEqual(arguments[2], "/remote/latent-weight-lab-spectral64/config.json")
        self.assertEqual(arguments[-1], "latent-weight-lab-spectral64")

    def test_host_probe_manifest_preserves_variant_worktree_identity(self) -> None:
        manifest = {
            "required_source_hashes": {"model.py": "old"},
            "entries": [
                {
                    "name": "task",
                    "variants": {
                        "Y400": {
                            "run_name": "run",
                            "config": "config.json",
                            "config_sha256": "abc",
                            "repo_relative": "latent-weight-lab-spectral64",
                            "required_source_hashes": {"model.py": "new"},
                        }
                    },
                }
            ],
        }
        self.assertEqual(
            host_probe_manifest(manifest, "Y400")["entries"][0],
            {
                "name": "run",
                "config": "config.json",
                "config_sha256": "abc",
                "repo_relative": "latent-weight-lab-spectral64",
                "required_source_hashes": {"model.py": "new"},
            },
        )

    def test_host_probe_manifest_ignores_terminal_variant_worktree_identity(self) -> None:
        manifest = {
            "required_source_hashes": {"model.py": "default"},
            "entries": [
                {
                    "name": "finished",
                    "variants": {
                        "Y400": {
                            "run_name": "old-run",
                            "config": "old.json",
                            "config_sha256": "old-config",
                            "repo_relative": "latent-weight-lab-spectral64",
                            "required_source_hashes": {"model.py": "old"},
                        }
                    },
                },
                {
                    "name": "pending",
                    "variants": {
                        "Y400": {
                            "run_name": "new-run",
                            "config": "new.json",
                            "config_sha256": "new-config",
                            "repo_relative": "latent-weight-lab-spectral64",
                            "required_source_hashes": {"model.py": "new"},
                        }
                    },
                },
            ],
        }
        state = {
            "entries": {
                "finished": {"state": "finished"},
                "pending": {"state": "pending"},
            }
        }
        probe = host_probe_manifest(manifest, "Y400", state)
        self.assertEqual([entry["name"] for entry in probe["entries"]], ["new-run"])
        self.assertEqual(probe["entries"][0]["required_source_hashes"], {"model.py": "new"})


if __name__ == "__main__":
    unittest.main()
