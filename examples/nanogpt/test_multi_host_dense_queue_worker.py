from __future__ import annotations

import unittest

from examples.nanogpt.multi_host_dense_queue_worker import (
    active_budget,
    launch,
    validate_pending_variant,
)
from unittest import mock


class MultiHostDenseQueueWorkerTest(unittest.TestCase):
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
