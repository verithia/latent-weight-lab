from __future__ import annotations

import unittest

from examples.nanogpt.requeue_verified_external_failures import parse_verified, requeue


class RequeueVerifiedExternalFailuresTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "entries": [
                {
                    "name": "task",
                    "variants": {
                        "Y400": {
                            "resume": True,
                            "expected_checkpoint_next_iter": 17,
                        }
                    },
                }
            ]
        }
        self.state = {
            "entries": {
                "task": {
                    "state": "failed_external",
                    "assigned_host": "Y400",
                    "last_iter": 21,
                    "attempts_by_host": {"Y400": 1},
                    "sent_milestones": [20],
                    "terminal_notified": True,
                    "terminal_pending": ["failed_external", None, 1],
                    "terminal_signature": ["failed_external", None, 1],
                    "pgid": 123,
                    "gpu": 2,
                    "rejected_hosts": ["Y400"],
                }
            }
        }

    def test_requeue_preserves_attempts_and_milestones(self) -> None:
        records = requeue(self.state, self.manifest, {"task": 17}, "Y400")
        runtime = self.state["entries"]["task"]
        self.assertEqual(records[0]["next_iter"], 17)
        self.assertEqual(runtime["state"], "pending")
        self.assertIsNone(runtime["assigned_host"])
        self.assertEqual(runtime["last_iter"], 17)
        self.assertEqual(runtime["attempts_by_host"], {"Y400": 1})
        self.assertEqual(runtime["sent_milestones"], [20])
        self.assertEqual(runtime["rejected_hosts"], [])
        self.assertNotIn("terminal_notified", runtime)
        self.assertNotIn("terminal_pending", runtime)
        self.assertNotIn("pgid", runtime)

    def test_rejects_wrong_checkpoint_and_nonfailed_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
            requeue(self.state, self.manifest, {"task": 18}, "Y400")
        self.state["entries"]["task"]["state"] = "finished"
        with self.assertRaisesRegex(ValueError, "not in failed_external"):
            requeue(self.state, self.manifest, {"task": 17}, "Y400")

    def test_parse_verified_rejects_duplicates(self) -> None:
        self.assertEqual(parse_verified(["task=17"]), {"task": 17})
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_verified(["task=17", "task=17"])


if __name__ == "__main__":
    unittest.main()
