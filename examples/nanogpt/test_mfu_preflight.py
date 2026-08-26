from __future__ import annotations

import unittest
from pathlib import Path

from examples.nanogpt.mfu_preflight import (
    make_preflight_config,
    parse_snapshot_elapsed_seconds,
)


class MfuPreflightTest(unittest.TestCase):
    def test_registered_selection_config_becomes_non_scientific_scratch_probe(self) -> None:
        source = {
            "mai_ladder_policy_version": "mai_ladder_selection_v2",
            "registered_resume_determinism_required": True,
            "save_checkpoint": True,
            "checkpoint_history": False,
            "lr_decay_iters": 100,
        }
        probe = make_preflight_config(source, Path("/tmp/probe"), 2, 3)
        self.assertNotIn("mai_ladder_policy_version", probe)
        self.assertFalse(probe["registered_resume_determinism_required"])
        self.assertFalse(probe["save_checkpoint"])
        self.assertEqual(probe["trajectory_snapshot_interval"], 0)
        self.assertEqual(source["mai_ladder_policy_version"], "mai_ladder_selection_v2")

    def test_diagnostic_io_can_be_preserved_for_strict_gate(self) -> None:
        source = {
            "registered_resume_determinism_required": True,
            "save_checkpoint": True,
            "checkpoint_history": False,
            "lr_decay_iters": 100,
            "trajectory_snapshot_interval": 15,
            "optimizer_probe_steps": [0, 15, 30],
        }
        probe = make_preflight_config(
            source,
            Path("/tmp/probe"),
            1,
            29,
            include_diagnostic_io=True,
        )
        self.assertEqual(probe["trajectory_snapshot_interval"], 15)
        self.assertEqual(probe["optimizer_probe_steps"], [0, 15, 30])
        self.assertEqual(probe["perf_warmup_iters"], 0)

    def test_optimizer_probe_only_diagnostic_io_needs_no_snapshots(self) -> None:
        source = {
            "registered_resume_determinism_required": True,
            "save_checkpoint": True,
            "checkpoint_history": False,
            "lr_decay_iters": 100,
            "trajectory_snapshot_interval": 0,
            "optimizer_probe_steps": [0, 2, 5],
            "optimizer_probe_fields": [
                "weight_before_step",
                "gradient_after_clip",
            ],
        }
        probe = make_preflight_config(
            source,
            Path("/tmp/probe"),
            1,
            4,
            include_diagnostic_io=True,
        )
        self.assertEqual(probe["trajectory_snapshot_interval"], 0)
        self.assertEqual(probe["optimizer_probe_steps"], [0, 2, 5])
        self.assertEqual(
            probe["optimizer_probe_fields"],
            ["weight_before_step", "gradient_after_clip"],
        )

    def test_snapshot_elapsed_parser(self) -> None:
        text = "\n".join(
            [
                "parameter trajectory snapshot step=0 path=/tmp/a elapsed_s=1.250",
                "perf iter=0 tokens_per_s=1 iter_ms=2",
                "parameter trajectory snapshot step=15 path=/tmp/b elapsed_s=0.750",
            ]
        )
        self.assertEqual(parse_snapshot_elapsed_seconds(text), [1.25, 0.75])


if __name__ == "__main__":
    unittest.main()
