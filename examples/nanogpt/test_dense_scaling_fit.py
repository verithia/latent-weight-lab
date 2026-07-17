import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

from examples.nanogpt import dense_scaling_fit as fit
from examples.nanogpt import train


class DenseScalingFitTests(unittest.TestCase):
    def _write_records(self, root: Path) -> list[Path]:
        paths = []
        costs = {"124m": 124_373_760, "350m": 354_599_936, "690m": 694_928_640, "985m": 984_909_312}
        for index, tier in enumerate(fit.REQUIRED_TIERS):
            cost = costs[tier]
            record = {
                "schema_version": fit.RESULT_RECORD_SCHEMA_VERSION,
                "acceptance_state": "ACCEPTED",
                "family": "dense",
                "model_tier": tier,
                "terminal_held_out_nll": 1.0 + 20.0 * cost ** -0.2,
                "estimated_active_params": cost,
                "scheduled_tokens": cost * 20,
                "scheduled_tpp": 20.0,
                "identity": {
                    "config_sha256": f"{index + 1:064x}",
                    "source_hashes": {"examples/nanogpt/train.py": f"{index + 11:064x}"},
                    "data_manifest_sha256": "a" * 64,
                    "fixed_eval_indices_sha256": "b" * 64,
                    "eval_protocol_id": "mai_ladder_fixed_eval_indices_v2",
                },
            }
            path = root / f"{tier}.json"
            path.write_text(json.dumps(record, sort_keys=True))
            paths.append(path)
        return paths

    def _resolved_muon(self) -> argparse.Namespace:
        return argparse.Namespace(
            optimizer="muon",
            learning_rate=0.002,
            min_lr=0.0002,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            muon_momentum=0.95,
            muon_ns_steps=5,
            muon_adamw_lr_scale=0.3,
            checkpoint_history=False,
        )

    def test_fit_requires_explicit_acceptance_and_pinned_artifact_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            records = self._write_records(root)
            draft = fit.build_artifact(records, accepted=False)
            self.assertEqual(draft["state"], "DRAFT_NOT_ACCEPTED")
            with self.assertRaisesRegex(ValueError, "not ACCEPTED"):
                fit.validate_dense_fit_artifact(draft)

            artifact = fit.build_artifact(records, accepted=True)
            coefficients = fit.validate_dense_fit_artifact(artifact)
            self.assertGreater(coefficients["A"], 0.0)
            self.assertGreater(coefficients["alpha"], 0.0)
            self.assertTrue(math.isfinite(coefficients["E"]))
            self.assertEqual(len(artifact["terminal_inputs"]), 4)
            self.assertEqual(len(artifact["sensitivity"]["leave_one_out"]), 4)

            artifact_path = root / "accepted-fit.json"
            fit.write_immutable_artifact(artifact_path, artifact)
            with self.assertRaises(FileExistsError):
                fit.write_immutable_artifact(artifact_path, artifact)

            train.validate_launch_config(
                {
                    "launch_ready": True,
                    "dense_fit_gate_required": True,
                    "dense_fit_artifact": str(artifact_path),
                    "dense_fit_artifact_sha256": fit.sha256_file(artifact_path),
                    "dense_fit_coefficients": coefficients,
                },
                self._resolved_muon(),
            )

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                train.validate_launch_config(
                    {
                        "launch_ready": True,
                        "dense_fit_gate_required": True,
                        "dense_fit_artifact": str(artifact_path),
                        "dense_fit_artifact_sha256": "0" * 64,
                        "dense_fit_coefficients": coefficients,
                    },
                    self._resolved_muon(),
                )


if __name__ == "__main__":
    unittest.main()
