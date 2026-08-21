from __future__ import annotations

import math
import types
import unittest
from pathlib import Path

from examples.nanogpt.mfu_preflight import (
    estimate_active_params,
    feedback_cap_preflight_metadata,
    make_preflight_config,
    parse_feedback_cap_events,
    parse_pair_vq_persistent_training_bytes,
    parse_stochastic_retraction_events,
    parse_snapshot_elapsed_seconds,
    parse_optimizer_probe_steps,
    parse_training_loss_values,
    task_frame_preflight_metadata,
    validate_pair_vq_persistent_training_bytes,
    verify_native_block_fht_extension,
)


class MfuPreflightTest(unittest.TestCase):
    def test_sparse_moe_mfu_counts_only_active_complete_experts(self) -> None:
        config = {
            "n_layer": 12,
            "n_embd": 768,
            "vocab_size": 50304,
            "block_size": 1024,
            "moe_num_experts": 8,
            "moe_top_k": 2,
            "moe_expert_hidden_multiplier": 2,
        }
        self.assertEqual(estimate_active_params(config), 124447488)

    def test_registered_selection_config_becomes_non_scientific_scratch_probe(self) -> None:
        source = {
            "mai_ladder_policy_version": "mai_ladder_selection_v2",
            "registered_resume_determinism_required": True,
            "save_checkpoint": True,
            "checkpoint_history": False,
            "lr_decay_iters": 100,
            "launch_ready": False,
        }
        probe = make_preflight_config(source, Path("/tmp/probe"), 2, 3)
        self.assertNotIn("mai_ladder_policy_version", probe)
        self.assertFalse(probe["registered_resume_determinism_required"])
        self.assertFalse(probe["save_checkpoint"])
        self.assertTrue(probe["launch_ready"])
        self.assertEqual(probe["trajectory_snapshot_interval"], 0)
        self.assertEqual(source["mai_ladder_policy_version"], "mai_ladder_selection_v2")
        self.assertFalse(source["launch_ready"])

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

    def test_attention_atlas_is_compacted_and_final_stage_is_timed(self) -> None:
        source = {
            "registered_resume_determinism_required": True,
            "save_checkpoint": True,
            "checkpoint_history": False,
            "lr_decay_iters": 2373,
            "block_fht_attn_cayley_atlas_start_steps": [
                0,
                594,
                1188,
                1782,
            ],
        }
        probe = make_preflight_config(source, Path("/tmp/probe"), 2, 3)
        self.assertEqual(probe["block_fht_attn_cayley_atlas_start_steps"], [0, 1, 2, 3])
        self.assertEqual(probe["perf_warmup_iters"], 4)
        self.assertEqual(probe["max_iters"], 7)
        self.assertEqual(probe["eval_interval"], 107)
        self.assertEqual(
            source["block_fht_attn_cayley_atlas_start_steps"],
            [0, 594, 1188, 1782],
        )

    def test_delayed_task_frame_certificate_reports_active_timed_path(self) -> None:
        source = {"block_fht_mlp_task_frame_start_iter": 120}
        self.assertEqual(
            task_frame_preflight_metadata(source, effective_warmup_updates=1),
            {
                "scientific_task_frame_start_iter": 120,
                "scratch_task_frame_start_iter": 1,
                "timed_task_frame_active": True,
            },
        )

    def test_feedback_cap_is_compacted_only_in_scratch_copy(self) -> None:
        source = {
            "block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps": 192.0,
            "mfu_preflight_error_feedback_max_nominal_steps": 0.5,
            "mfu_preflight_require_feedback_cap_active": True,
        }
        probe = make_preflight_config(
            source,
            Path("/tmp/probe"),
            warmups=1,
            timed=8,
        )
        self.assertEqual(
            probe[
                "block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps"
            ],
            0.5,
        )
        self.assertEqual(
            source[
                "block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps"
            ],
            192.0,
        )
        self.assertEqual(
            feedback_cap_preflight_metadata(source),
            {
                "scientific_feedback_cap_nominal_steps": 192.0,
                "scratch_feedback_cap_nominal_steps": 0.5,
                "feedback_cap_activity_required": True,
            },
        )

    def test_feedback_cap_event_parser(self) -> None:
        text = "\n".join(
            [
                "iter 0: loss 1.0",
                (
                    "muon_matched_givens_feedback_cap "
                    '{"active_layers":3,"max_pre_cap_nominal_steps":0.7,"step":1}'
                ),
            ]
        )
        self.assertEqual(
            parse_feedback_cap_events(text),
            [
                {
                    "active_layers": 3,
                    "max_pre_cap_nominal_steps": 0.7,
                    "step": 1,
                }
            ],
        )
        self.assertEqual(
            task_frame_preflight_metadata({}, effective_warmup_updates=1),
            {
                "scientific_task_frame_start_iter": 0,
                "scratch_task_frame_start_iter": 0,
                "timed_task_frame_active": False,
            },
        )

    def test_stochastic_retraction_and_persistent_byte_parsers(self) -> None:
        text = "\n".join(
            [
                'pair_vq_stochastic_retraction {"step":3,"weighted_sampling_variance_ratio":0.08}',
                "mlp_pair_vq: modules=24 elements=1 persistent_training_bytes=157,500,864 model_compression_vs_dense_bf16=1",
            ]
        )
        self.assertEqual(
            parse_stochastic_retraction_events(text),
            [{"step": 3, "weighted_sampling_variance_ratio": 0.08}],
        )
        self.assertEqual(
            parse_pair_vq_persistent_training_bytes(text), [157500864]
        )

    def test_pair_vq_persistent_byte_gate_is_independent(self) -> None:
        config = {"persistent_training_bytes_exact": 157500864}
        self.assertEqual(
            validate_pair_vq_persistent_training_bytes(
                config, [157500864]
            ),
            {
                "observed": [157500864],
                "expected": 157500864,
                "passed": True,
            },
        )
        with self.assertRaisesRegex(
            RuntimeError, "persistent-byte gate rejected launch"
        ):
            validate_pair_vq_persistent_training_bytes(
                config, [157500865]
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

    def test_optimizer_probe_completion_parser(self) -> None:
        text = "\n".join(
            [
                "optimizer probe step=0 path=/tmp/probe/step_000000.pt",
                "iter 0: loss 7.0, time 2.0ms",
                "optimizer probe step=98 path=/tmp/probe/step_000098.pt",
            ]
        )
        self.assertEqual(parse_optimizer_probe_steps(text), [0, 98])

    def test_training_loss_parser_exposes_nonfinite_values(self) -> None:
        text = "\n".join(
            [
                "step 0: train loss 10.5, val loss 10.4",
                "iter 0: loss 10.3, time 1.0ms",
                "iter 1: loss nan, time 1.0ms",
                "step 2: train loss inf, val loss -inf",
            ]
        )
        values = parse_training_loss_values(text)
        self.assertEqual(len(values), 6)
        self.assertTrue(math.isfinite(values[0]))
        self.assertFalse(math.isfinite(values[-1]))

    def test_native_extension_is_required_for_cuda_block_fht(self) -> None:
        extension = types.SimpleNamespace(__name__="native_test_ext")
        result = verify_native_block_fht_extension(
            {"method": "block_fht", "device": "cuda"},
            loader=lambda: extension,
        )
        self.assertTrue(result["required"])
        self.assertTrue(result["loaded"])
        self.assertEqual(result["module"], "native_test_ext")

        with self.assertRaisesRegex(RuntimeError, "refusing the MFU gate"):
            verify_native_block_fht_extension(
                {"method": "block_fht", "device": "cuda"},
                loader=lambda: None,
            )

    def test_dense_or_cpu_preflight_does_not_require_extension(self) -> None:
        self.assertEqual(
            verify_native_block_fht_extension(
                {"method": "baseline", "device": "cuda"},
                loader=lambda: None,
            ),
            {"required": False, "loaded": None, "module": None},
        )


if __name__ == "__main__":
    unittest.main()
