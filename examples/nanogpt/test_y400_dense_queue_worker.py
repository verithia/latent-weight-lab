from __future__ import annotations

import unittest

from examples.nanogpt.y400_dense_queue_worker import (
    can_admit,
    idle_gpu_indices,
    remote_identity_valid,
)


class Y400DenseQueueWorkerTest(unittest.TestCase):
    def test_checkpoint_admission_reserves_active_and_candidate_atomic_files(self) -> None:
        gib = 1024**3
        self.assertTrue(can_admit(200 * gib, 256 * gib, 8 * gib, 10 * gib, 6 * gib))
        self.assertFalse(can_admit(240 * gib, 256 * gib, 8 * gib, 4 * gib, 6 * gib))

    def test_idle_gpu_requires_no_compute_process_and_low_memory(self) -> None:
        remote = {
            "gpus": [
                {"index": 0, "memory_used_mib": 4, "compute_pids": []},
                {"index": 1, "memory_used_mib": 900, "compute_pids": [123]},
                {"index": 2, "memory_used_mib": 2048, "compute_pids": []},
                {"index": 3, "memory_used_mib": 800, "compute_pids": []},
            ]
        }
        self.assertEqual(idle_gpu_indices(remote, 1024), [0, 3])

    def test_remote_identity_binds_source_and_every_config(self) -> None:
        manifest = {
            "required_source_hashes": {"train.py": "a" * 64},
            "entries": [{"name": "run", "config_sha256": "b" * 64}],
        }
        remote = {
            "git_dirty": False,
            "source_hashes": {"train.py": "a" * 64},
            "entries": {"run": {"config_sha256": "b" * 64}},
        }
        self.assertEqual(remote_identity_valid(remote, manifest), (True, ""))
        remote["source_hashes"]["train.py"] = "c" * 64
        valid, reason = remote_identity_valid(remote, manifest)
        self.assertFalse(valid)
        self.assertIn("source hashes", reason)


if __name__ == "__main__":
    unittest.main()
