from __future__ import annotations

import unittest
from pathlib import Path

from examples.nanogpt.mfu_preflight import make_preflight_config


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
        self.assertEqual(source["mai_ladder_policy_version"], "mai_ladder_selection_v2")


if __name__ == "__main__":
    unittest.main()
