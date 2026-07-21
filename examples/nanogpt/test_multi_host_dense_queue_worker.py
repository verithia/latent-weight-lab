from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.nanogpt.multi_host_dense_queue_worker import (
    active_budget,
    host_admission_status,
    launch,
    load_state,
    validate_pending_variant,
)
from unittest import mock


class MultiHostDenseQueueWorkerTest(unittest.TestCase):
    GIB = 1024**3

    def test_load_state_preserves_operator_host_pause(self) -> None:
        manifest = {"entries": [{"name": "task", "variants": {"Y400": {}}}]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text('{"paused_hosts": ["Y400"]}')
            state = load_state(path, manifest)
        self.assertEqual(state["paused_hosts"], ["Y400"])
        self.assertEqual(state["entries"]["task"]["state"], "pending")

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
        self.assertEqual(ssh_script.call_args.args[2][-1], "detached")


if __name__ == "__main__":
    unittest.main()
