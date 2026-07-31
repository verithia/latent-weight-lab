from __future__ import annotations

import json
import unittest

from examples.nanogpt.make_pro6_attention_recovery_configs import (
    CONFIG_DIR,
    PRO6_ROOT,
    SOURCES,
    derive_config,
)


class Pro6AttentionRecoveryConfigsTest(unittest.TestCase):
    def test_derivation_changes_only_paths_and_host_metadata(self) -> None:
        allowed = {
            "data_dir",
            "out_dir",
            "execution_host",
            "host_transfer_source_config",
            "host_transfer_policy",
        }
        for source_name, destination_name in SOURCES.items():
            with self.subTest(source=source_name):
                source = json.loads((CONFIG_DIR / source_name).read_text())
                derived = derive_config(source_name, destination_name, source)
                changed = {
                    key
                    for key in source.keys() | derived.keys()
                    if source.get(key) != derived.get(key)
                }
                self.assertEqual(changed, allowed)
                self.assertEqual(
                    derived["data_dir"],
                    str(PRO6_ROOT / "data/finewebedu_20b"),
                )
                self.assertEqual(derived["execution_host"], "PRO6")
                self.assertTrue(derived["mfu_preflight_required"])
                self.assertGreaterEqual(derived["mfu_min_fraction"], 0.20)

    def test_checked_in_configs_match_derivation(self) -> None:
        for source_name, destination_name in SOURCES.items():
            with self.subTest(destination=destination_name):
                source = json.loads((CONFIG_DIR / source_name).read_text())
                expected = derive_config(source_name, destination_name, source)
                actual = json.loads((CONFIG_DIR / destination_name).read_text())
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
