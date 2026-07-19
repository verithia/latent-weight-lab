import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from examples.nanogpt import train
from examples.nanogpt import dense_scaling_fit as dense_fit
from examples.nanogpt import make_y400_mai_scaling_ladder_configs as mai_configs
from examples.nanogpt import mai_selection_artifacts as mai_artifacts


class FixedEvaluationRngTests(unittest.TestCase):
    def test_chunked_stability_gradient_matches_full_kl(self):
        torch.manual_seed(20260720)
        reference = torch.randn(2, 5, 7)
        perturbed_full = torch.randn(2, 5, 7, requires_grad=True)
        perturbed_chunked = perturbed_full.detach().clone().requires_grad_(True)
        temperature = 1.7

        full = train.logits_kl_stability_loss(reference, perturbed_full, temperature)
        full.backward()

        chunk_value = 0.0
        chunks_remaining = math.ceil((perturbed_chunked.shape[0] * perturbed_chunked.shape[1]) / 3)
        for output_slice, value, gradient in train.iter_logits_kl_stability_backward_chunks(
            reference, perturbed_chunked, temperature, 3
        ):
            chunks_remaining -= 1
            output_slice.backward(gradient=gradient, retain_graph=chunks_remaining > 0)
            chunk_value += float(value.item())

        self.assertAlmostEqual(chunk_value, float(full.item()), places=5)
        self.assertTrue(torch.allclose(perturbed_chunked.grad, perturbed_full.grad, atol=1e-6, rtol=1e-6))

    def test_omitted_seed_keeps_global_generator_behavior(self):
        self.assertIsNone(train.make_cpu_generator(None))

        torch.manual_seed(1234)
        expected = torch.randint(10_000, (16,))
        torch.manual_seed(1234)
        actual = torch.randint(10_000, (16,), generator=train.make_cpu_generator(None))

        self.assertTrue(torch.equal(expected, actual))

    def test_fixed_indices_and_digest_ignore_global_model_rng(self):
        with tempfile.TemporaryDirectory() as raw:
            data_dir = Path(raw)
            np.arange(1024, dtype=np.uint16).tofile(data_dir / "train.bin")
            np.arange(1024, 2048, dtype=np.uint16).tofile(data_dir / "val.bin")

            torch.manual_seed(1337)
            _ = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 8))
            _ = torch.rand(4096)
            first = train.make_fixed_eval_indices(data_dir, 4, 32, 7, 20260715)
            first_digest = train.fixed_eval_indices_digest(first)

            torch.manual_seed(9999)
            _ = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.Linear(64, 32))
            _ = torch.rand(8192)
            second = train.make_fixed_eval_indices(data_dir, 4, 32, 7, 20260715)
            second_digest = train.fixed_eval_indices_digest(second)

            self.assertEqual(first_digest, second_digest)
            for split in ("train", "val"):
                self.assertTrue(torch.equal(first[split], second[split]))
            self.assertNotEqual(first_digest, train.fixed_eval_indices_digest({
                "train": first["val"], "val": first["train"],
            }))

    def test_opt_in_split_generators_are_independent_of_global_rng(self):
        with tempfile.TemporaryDirectory() as raw:
            data_dir = Path(raw)
            np.arange(1024, dtype=np.uint16).tofile(data_dir / "train.bin")
            first_x, first_y = train.get_batch(
                data_dir,
                "train",
                batch_size=8,
                block_size=32,
                device="cpu",
                generator=train.make_cpu_generator(20260715),
            )

            torch.manual_seed(9999)
            _ = torch.rand(8192)
            second_x, second_y = train.get_batch(
                data_dir,
                "train",
                batch_size=8,
                block_size=32,
                device="cpu",
                generator=train.make_cpu_generator(20260715),
            )

            self.assertTrue(torch.equal(first_x, second_x))
            self.assertTrue(torch.equal(first_y, second_y))

    def test_registered_eval_shape_is_common_across_training_rungs(self):
        configs = [
            mai_configs.make_config(f"test_{tier}", tier, 0.5, launch_ready=True)
            for tier in mai_configs.TIERS
        ]
        eval_shapes = {
            (
                config["eval_batch_size"],
                config["eval_iters"],
                config["eval_tokens_per_split"],
                config["eval_total_tokens"],
                config["fixed_eval_index_spec_sha256"],
            )
            for config in configs
        }
        self.assertEqual(len(eval_shapes), 1)
        self.assertEqual(
            next(iter(eval_shapes)),
            (16, 400, 16 * 1024 * 400, 2 * 16 * 1024 * 400, mai_configs.eval_index_spec_sha256(16, 400)),
        )
        self.assertEqual({config["batch_size"] for config in configs}, {16, 32})
        with tempfile.TemporaryDirectory() as raw:
            data_dir = Path(raw)
            np.arange(16_384, dtype=np.uint16).tofile(data_dir / "train.bin")
            np.arange(16_384, 32_768, dtype=np.uint16).tofile(data_dir / "val.bin")
            runtime_digests = {
                train.fixed_eval_indices_digest(
                    train.make_fixed_eval_indices(
                        data_dir,
                        config["eval_batch_size"],
                        config["block_size"],
                        config["eval_iters"],
                        config["eval_seed"],
                    )
                )
                for config in configs
            }
        self.assertEqual(len(runtime_digests), 1)

    def test_mai_materialized_counts_and_tpp_schedules(self):
        for tier, expected_active in mai_configs.EXPECTED_MATERIALIZED_PARAM_COUNTS.items():
            for tpp, expected_schedule in mai_configs.EXPECTED_TPP_SCHEDULES[tier].items():
                config = mai_configs.make_config(f"test_{tier}_{tpp}", tier, tpp, launch_ready=True)
                self.assertEqual(config["estimated_active_params"], expected_active)
                self.assertEqual(
                    (config["planned_tokens"], config["max_iters"], config["scheduled_tokens"]),
                    expected_schedule,
                )
                self.assertEqual(config["scheduled_tpp"], config["scheduled_tokens"] / expected_active)
                self.assertIs(config["bias"], False)
                self.assertEqual(config["dropout"], 0.0)
                self.assertIs(config["tie_word_embeddings"], True)
                self.assertEqual(config["checkpoint_wall_clock_seconds"], 7200)

    def test_checkpointed_train_generator_replays_the_pending_batch(self):
        with tempfile.TemporaryDirectory() as raw:
            data_dir = Path(raw)
            np.arange(8192, dtype=np.uint16).tofile(data_dir / "train.bin")

            original_generator = train.make_cpu_generator(20260714)
            pending_state = train.generator_state(original_generator)
            current_x, current_y = train.get_batch(
                data_dir, "train", batch_size=8, block_size=32, device="cpu", generator=original_generator
            )
            next_x, _ = train.get_batch(
                data_dir, "train", batch_size=8, block_size=32, device="cpu", generator=original_generator
            )

            resumed_generator = train.make_cpu_generator(999)  # The checkpoint state, not this seed, wins.
            train.restore_generator_state(resumed_generator, pending_state)
            resumed_x, resumed_y = train.get_batch(
                data_dir, "train", batch_size=8, block_size=32, device="cpu", generator=resumed_generator
            )

            self.assertTrue(torch.equal(current_x, resumed_x))
            self.assertTrue(torch.equal(current_y, resumed_y))
            self.assertFalse(torch.equal(next_x, resumed_x))

    def test_launch_guards_reject_blocked_or_unresolved_templates_only(self):
        resolved_muon = argparse.Namespace(
            optimizer="muon",
            learning_rate=0.002,
            min_lr=0.0002,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            muon_momentum=0.95,
            muon_ns_steps=5,
            muon_adamw_lr_scale=0.3,
        )
        train.validate_launch_config({}, resolved_muon)  # Legacy configs omit the gate metadata.
        legacy_history = argparse.Namespace(**vars(resolved_muon), checkpoint_history=True)
        train.validate_launch_config({}, legacy_history)
        with self.assertRaisesRegex(ValueError, "checkpoint history"):
            train.validate_launch_config(
                {"registered_resume_determinism_required": True}, legacy_history
            )
        with self.assertRaisesRegex(ValueError, "launch_ready"):
            train.validate_launch_config({"launch_ready": False}, resolved_muon)
        with self.assertRaisesRegex(ValueError, "requires recipe resolution"):
            train.validate_launch_config({"recipe_resolution_required": True}, resolved_muon)

        unresolved = argparse.Namespace(**vars(resolved_muon))
        unresolved.learning_rate = None
        unresolved.muon_adamw_lr_scale = None
        with self.assertRaisesRegex(ValueError, "learning_rate, muon_adamw_lr_scale"):
            train.validate_launch_config({}, unresolved)

    def test_generated_blocked_template_cannot_reach_training_setup(self):
        config = mai_configs.make_dense_confirmation_template("blocked", "124m", "top1")
        with tempfile.TemporaryDirectory() as raw:
            config_path = Path(raw) / "blocked.json"
            config_path.write_text(json.dumps(config))
            with patch.object(sys, "argv", ["train.py", "--config", str(config_path)]):
                with self.assertRaisesRegex(ValueError, "launch_ready"):
                    train.parse_args()

    def test_prospective_mai_v2_ladder_state_machine_and_cleanup(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config_dir = root / "configs"
            config_dir.mkdir()
            resolved = config_dir / "y400_mai_legacy_resolved.json"
            resolved.write_text('{"preserve": true}\n')
            stale_template = config_dir / "y400_mai_v2_stale_template.json"
            stale_template.write_text('{"remove": true}\n')
            queue_path = root / "queue.tsv"

            with (
                patch.object(mai_configs, "CONFIG_DIR", config_dir),
                patch.object(mai_configs, "QUEUE_PATH", queue_path),
            ):
                mai_configs.main()

            self.assertEqual(resolved.read_text(), '{"preserve": true}\n')
            self.assertFalse(stale_template.exists())
            with queue_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 48)
            self.assertTrue(all(row["policy_version"] == mai_configs.POLICY_VERSION for row in rows))
            self.assertFalse(any("scale" in row["phase"] for row in rows))

            configs = {
                row["name"]: json.loads((config_dir / f"{row['name']}.json").read_text())
                for row in rows
            }
            self.assertTrue(all(config["muon_adamw_lr_scale"] == 0.3 for config in configs.values()))
            self.assertTrue(all("fallback_scale_screen" not in config["hpo_stage"] for config in configs.values()))

            for tier in mai_configs.TIERS:
                for method in ("baseline", "block_fht"):
                    tier_rows = [
                        row for row in rows if row["tier"] == tier and row["method"] == method
                    ]
                    self.assertEqual(len(tier_rows), 6)
                    screens = [row for row in tier_rows if row["ladder_role"] == "screen_only"]
                    confirmations = [row for row in tier_rows if row["ladder_role"] == "confirmation"]
                    selected = [row for row in tier_rows if row["ladder_role"] == "selected_recipe"]
                    self.assertEqual(len(screens), 3)
                    self.assertTrue(all(row["planned_tpp"] == "0.500000" for row in screens))
                    self.assertTrue(all(configs[row["name"]]["screen_only"] for row in screens))
                    self.assertEqual(len(confirmations), 2)
                    self.assertEqual({row["ladder_slot"] for row in confirmations}, {"top1", "top2"})
                    self.assertTrue(all(row["launch_ready"] == "false" for row in confirmations))
                    self.assertTrue(
                        all(configs[row["name"]]["zero_point_five_tpp_ranking_artifact_required"] for row in confirmations)
                    )
                    self.assertTrue(
                        all(configs[row["name"]]["zero_point_five_tpp_ranking_artifact"] is None for row in confirmations)
                    )
                    self.assertEqual(len(selected), 1)
                    selected_config = configs[selected[0]["name"]]
                    self.assertTrue(selected_config["five_tpp_comparison_artifact_required"])
                    self.assertIsNone(selected_config["five_tpp_comparison_artifact"])
                    self.assertIsNone(selected_config["five_tpp_comparison_artifact_sha256"])
                    self.assertEqual(
                        selected_config["ladder_slot"], "selected_from_5tpp_comparison"
                    )

                    if method == "baseline":
                        self.assertEqual(
                            sorted(configs[row["name"]]["learning_rate"] for row in screens),
                            list(mai_configs.BASELINE_SCREEN_LRS),
                        )
                    else:
                        self.assertTrue(
                            all(row["status"] == "BLOCKED_ON_DENSE_SCALING_FIT" for row in tier_rows)
                        )
                        self.assertTrue(
                            all(configs[row["name"]]["dense_fit_gate_required"] for row in tier_rows)
                        )
                        self.assertEqual(
                            sorted(configs[row["name"]]["candidate_main_lr_multiplier"] for row in screens),
                            list(mai_configs.CANDIDATE_MAIN_LR_MULTIPLIERS),
                        )
                        self.assertTrue(all(
                            {
                                field: configs[row["name"]][field]
                                for field in mai_artifacts.REGISTERED_V2_BLOCK_FHT_METHOD_SPEC
                            }
                            == mai_artifacts.REGISTERED_V2_BLOCK_FHT_METHOD_SPEC
                            for row in tier_rows
                        ))

    def _mai_run_contract(self, tier="124m", method="baseline", *, legacy=False):
        config = mai_configs.make_config("test-run-contract", tier, 0.5, launch_ready=True)
        contract = {
            "schema_version": (
                mai_artifacts.LEGACY_RUN_CONTRACT_SCHEMA_VERSION
                if legacy
                else mai_artifacts.RUN_CONTRACT_SCHEMA_VERSION
            ),
            **{field: config[field] for field in mai_artifacts.RUN_CONTRACT_FIELDS},
        }
        if legacy:
            return contract
        method_structure = {"method": method}
        if method == "block_fht":
            block_config = mai_configs.make_fullattn_template(
                "test-block-contract",
                tier,
                0.5,
                main_lr_multiplier=0.5,
                stage="full_attention_blockfht_screen_0p5tpp",
                role="screen_only",
                slot="mult0p50",
                screen_only=True,
            )
            method_structure["block_fht"] = {
                field: block_config[field] for field in mai_artifacts.BLOCK_FHT_STRUCTURE_FIELDS
            }
        contract["method_structure"] = method_structure
        return contract

    def _mai_identity(self, tier="124m", method="baseline", *, legacy=False):
        return {
            "config_sha256": "0" * 63 + "1",
            "source_hashes": train.source_hashes(),
            "data_manifest_sha256": "b" * 64,
            "fixed_eval_indices_sha256": "c" * 64,
            "eval_protocol_id": mai_configs.REGISTERED_EVAL_PROTOCOL,
            "run_contract": self._mai_run_contract(tier, method, legacy=legacy),
        }

    def _mai_record(
        self,
        *,
        tier="124m",
        method="baseline",
        stage="dense_screen_0p5tpp",
        role="screen_only",
        slot="lr16e4",
        tpp=0.5,
        candidate_value=0.0016,
        nll=3.0,
        config_index=1,
        identity=None,
        selection_provenance=None,
    ):
        candidate_field = "learning_rate" if method == "baseline" else "candidate_main_lr_multiplier"
        recipe = {
            "learning_rate": candidate_value if method == "baseline" else 0.002 * candidate_value,
            "min_lr": (candidate_value if method == "baseline" else 0.002 * candidate_value) * 0.1,
            "muon_adamw_lr_scale": 0.3,
        }
        if method == "block_fht":
            recipe[candidate_field] = candidate_value
        record_identity = self._mai_identity(tier, method) if identity is None else identity
        record_identity = json.loads(json.dumps(record_identity))
        record_identity["config_sha256"] = f"{config_index:064x}"
        result = {
            "schema_version": mai_artifacts.TERMINAL_RESULT_SCHEMA_VERSION,
            "acceptance_state": "ACCEPTED",
            "mai_ladder_policy_version": mai_artifacts.POLICY_VERSION,
            "model_tier": tier,
            "method": method,
            "hpo_stage": stage,
            "ladder_role": role,
            "ladder_slot": slot,
            "planned_tpp": tpp,
            "candidate": {"field": candidate_field, "value": candidate_value},
            "selection_recipe": recipe,
            "terminal_held_out_nll": nll,
            "identity": record_identity,
        }
        if role == "confirmation":
            result["selection_provenance"] = selection_provenance
        return result

    def _write_mai_record(self, root, name, record):
        path = root / name
        path.write_text(json.dumps(record, sort_keys=True))
        return path

    def _resolved_muon(self, recipe, method="baseline"):
        contract = self._mai_run_contract(method=method)
        resolved = {
            **{field: contract[field] for field in mai_artifacts.RUN_CONTRACT_FIELDS},
            "method": method,
            "learning_rate": recipe["learning_rate"],
            "min_lr": recipe["min_lr"],
            "muon_adamw_lr_scale": recipe["muon_adamw_lr_scale"],
            "candidate_main_lr_multiplier": recipe.get("candidate_main_lr_multiplier"),
            "checkpoint_history": False,
            "save_checkpoint": True,
            "checkpoint_wall_clock_seconds": 7200,
            "registered_resume_determinism_required": True,
        }
        if method == "block_fht":
            resolved.update(contract["method_structure"]["block_fht"])
        return argparse.Namespace(
            **resolved,
        )

    def _write_accepted_dense_fit(self, root: Path, name="accepted-dense-fit.json"):
        costs = {
            "124m": 124_373_760,
            "350m": 354_599_936,
            "690m": 694_928_640,
            "985m": 984_909_312,
        }
        records = []
        for index, tier in enumerate(dense_fit.REQUIRED_TIERS, start=1):
            cost = costs[tier]
            path = root / f"dense-fit-{name}-{tier}.json"
            path.write_text(json.dumps({
                "schema_version": dense_fit.RESULT_RECORD_SCHEMA_VERSION,
                "acceptance_state": "ACCEPTED",
                "family": "dense",
                "model_tier": tier,
                "terminal_held_out_nll": 1.0 + 20.0 * cost ** -0.2,
                "estimated_active_params": cost,
                "scheduled_tokens": cost * 20,
                "scheduled_tpp": 20.0,
                "identity": {
                    "config_sha256": f"{index:064x}",
                    "source_hashes": {"examples/nanogpt/train.py": f"{index + 10:064x}"},
                    "data_manifest_sha256": "b" * 64,
                    "fixed_eval_indices_sha256": "c" * 64,
                    "eval_protocol_id": mai_configs.REGISTERED_EVAL_PROTOCOL,
                },
            }, sort_keys=True))
            records.append(path)
        artifact = dense_fit.build_artifact(records, accepted=True)
        artifact_path = root / name
        dense_fit.write_immutable_artifact(artifact_path, artifact)
        return artifact_path, dense_fit.validate_dense_fit_artifact(artifact)

    def _make_mai_artifacts(self, root, method="baseline"):
        if method == "baseline":
            screen_stage = "dense_screen_0p5tpp"
            confirmation_stage = "dense_recipe_confirmation_5tpp"
            screen_candidates = ((0.0016, 3.0), (0.0020, 3.1), (0.0024, 3.2))
            confirmation_candidates = (("top1", 0.0016, 2.95), ("top2", 0.0020, 3.10))
        else:
            screen_stage = "full_attention_blockfht_screen_0p5tpp"
            confirmation_stage = "full_attention_blockfht_confirmation_5tpp"
            screen_candidates = ((0.5, 3.0), (0.75, 3.1), (1.0, 3.2))
            confirmation_candidates = (("top1", 0.5, 2.95), ("top2", 0.75, 3.10))
        screen_paths = []
        for index, (candidate, nll) in enumerate(screen_candidates, start=1):
            screen_paths.append(self._write_mai_record(
                root,
                f"screen-{index}.json",
                self._mai_record(
                    method=method,
                    stage=screen_stage,
                    slot=f"candidate-{candidate}",
                    candidate_value=candidate,
                    nll=nll,
                    config_index=index,
                ),
            ))
        ranking = mai_artifacts.build_ranking_artifact(screen_paths)
        ranking_path = root / "ranking.json"
        mai_artifacts.write_immutable_artifact(ranking_path, ranking)

        confirmation_paths = []
        ranking_provenance = {
            "path": str(ranking_path.resolve()),
            "sha256": mai_artifacts.sha256_file(ranking_path),
            "schema_version": mai_artifacts.RANKING_ARTIFACT_SCHEMA_VERSION,
        }
        for index, (slot, candidate, nll) in enumerate(confirmation_candidates, start=11):
            confirmation_paths.append(self._write_mai_record(
                root,
                f"confirmation-{slot}.json",
                self._mai_record(
                    method=method,
                    stage=confirmation_stage,
                    role="confirmation",
                    slot=slot,
                    tpp=5.0,
                    candidate_value=candidate,
                    nll=nll,
                    config_index=index,
                    selection_provenance={**ranking_provenance, "slot": slot},
                ),
            ))
        comparison = mai_artifacts.build_comparison_artifact(ranking_path, confirmation_paths)
        comparison_path = root / "comparison.json"
        mai_artifacts.write_immutable_artifact(comparison_path, comparison)
        return ranking, ranking_path, comparison, comparison_path, screen_paths, confirmation_paths

    def test_mai_selection_artifacts_reject_stale_inputs_and_pin_launch_recipes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ranking, ranking_path, comparison, comparison_path, screens, confirmations = self._make_mai_artifacts(root)
            self.assertEqual(set(ranking["ranked_slots"]), {"top1", "top2"})
            self.assertEqual(comparison["selected_slot"], "top1")
            with self.assertRaises(FileExistsError):
                mai_artifacts.write_immutable_artifact(ranking_path, ranking)

            missing_hash_record = self._mai_record()
            del missing_hash_record["identity"]["config_sha256"]
            with self.assertRaisesRegex(ValueError, "config_sha256"):
                mai_artifacts.validate_terminal_result(missing_hash_record)
            bad_source_record = self._mai_record()
            bad_source_record["identity"]["source_hashes"] = {"examples/nanogpt/train.py": "bad"}
            with self.assertRaisesRegex(ValueError, "source_hashes"):
                mai_artifacts.validate_terminal_result(bad_source_record)

            arbitrary_candidate = self._mai_record(candidate_value=0.003)
            with self.assertRaisesRegex(ValueError, "registered screen set"):
                mai_artifacts.validate_terminal_result(arbitrary_candidate)
            block_screen_paths = [
                self._write_mai_record(
                    root,
                    f"block-screen-{index}.json",
                    self._mai_record(
                        method="block_fht",
                        stage="full_attention_blockfht_screen_0p5tpp",
                        slot=f"mult{candidate}",
                        candidate_value=candidate,
                        config_index=30 + index,
                    ),
                )
                for index, candidate in enumerate((0.5, 0.75, 1.0), start=1)
            ]
            self.assertEqual(
                mai_artifacts.build_ranking_artifact(block_screen_paths)["method"], "block_fht"
            )
            wrong_min_lr = self._mai_record()
            wrong_min_lr["selection_recipe"]["min_lr"] = 0.00017
            with self.assertRaisesRegex(ValueError, "min_lr must equal 0.1"):
                mai_artifacts.validate_terminal_result(wrong_min_lr)
            replaced_screen = self._write_mai_record(
                root,
                "replaced-screen.json",
                self._mai_record(candidate_value=0.0016, nll=3.3, config_index=4),
            )
            with self.assertRaisesRegex(ValueError, "exactly the registered screen candidate set"):
                mai_artifacts.build_ranking_artifact([screens[0], screens[1], replaced_screen])

            incompatible = self._mai_record(candidate_value=0.0024, nll=3.3, config_index=5)
            incompatible["identity"]["fixed_eval_indices_sha256"] = "d" * 64
            incompatible_path = self._write_mai_record(root, "incompatible.json", incompatible)
            with self.assertRaisesRegex(ValueError, "incompatible shared identity"):
                mai_artifacts.build_ranking_artifact([*screens[:2], incompatible_path])
            incompatible_contract = self._mai_record(candidate_value=0.0024, nll=3.3, config_index=6)
            incompatible_contract["identity"]["run_contract"]["batch_size"] += 1
            incompatible_contract_path = self._write_mai_record(
                root, "incompatible-contract.json", incompatible_contract
            )
            with self.assertRaisesRegex(ValueError, "incompatible shared identity"):
                mai_artifacts.build_ranking_artifact([*screens[:2], incompatible_contract_path])

            stale = self._mai_record(
                stage="dense_recipe_confirmation_5tpp",
                role="confirmation",
                slot="top2",
                tpp=5.0,
                candidate_value=0.0024,
                nll=2.9,
                config_index=22,
                selection_provenance={
                    "path": str(ranking_path.resolve()),
                    "sha256": mai_artifacts.sha256_file(ranking_path),
                    "schema_version": mai_artifacts.RANKING_ARTIFACT_SCHEMA_VERSION,
                    "slot": "top2",
                },
            )
            stale_path = self._write_mai_record(root, "stale.json", stale)
            with self.assertRaisesRegex(ValueError, "stale or not one of the ranked top-two"):
                mai_artifacts.build_comparison_artifact(ranking_path, [confirmations[0], stale_path])

            mutable_threshold = json.loads(json.dumps(comparison))
            mutable_threshold["practical_equivalence_nll"] = 0.03
            with self.assertRaisesRegex(ValueError, "exactly 0.02"):
                mai_artifacts.validate_comparison_artifact(mutable_threshold)

            ranked = ranking["ranked_slots"]["top1"]
            config = mai_configs.make_dense_confirmation_template("resolved-top1", "124m", "top1")
            config.update(
                launch_ready=True,
                recipe_resolution_required=False,
                registered_resume_determinism_required=True,
                zero_point_five_tpp_ranking_artifact=str(ranking_path),
                zero_point_five_tpp_ranking_artifact_sha256=mai_artifacts.sha256_file(ranking_path),
                mai_selection_candidate=ranked["candidate"],
                mai_selection_recipe=ranked["selection_recipe"],
                **ranked["selection_recipe"],
            )
            resolved_muon = self._resolved_muon(ranked["selection_recipe"])
            train.validate_launch_config(config, resolved_muon)

            wrong_run_contract = dict(config)
            wrong_run_contract["batch_size"] = config["batch_size"] + 1
            with self.assertRaisesRegex(ValueError, "run-contract field batch_size"):
                train.validate_launch_config(wrong_run_contract, resolved_muon)
            with patch.object(train, "source_hashes", return_value={"examples/nanogpt/train.py": "0" * 64}):
                with self.assertRaisesRegex(ValueError, "source identity"):
                    train.validate_launch_config(config, resolved_muon)

            bad_hash = dict(config)
            bad_hash["zero_point_five_tpp_ranking_artifact_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                train.validate_launch_config(bad_hash, resolved_muon)
            missing_hash = dict(config)
            missing_hash["zero_point_five_tpp_ranking_artifact_sha256"] = None
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                train.validate_launch_config(missing_hash, resolved_muon)
            wrong_tier = dict(config)
            wrong_tier["model_tier"] = "350m"
            with self.assertRaisesRegex(ValueError, "tier/method/stage/slot"):
                train.validate_launch_config(wrong_tier, resolved_muon)
            wrong_method = dict(config)
            wrong_method["method"] = "block_fht"
            with self.assertRaisesRegex(ValueError, "tier/method/stage/slot"):
                train.validate_launch_config(wrong_method, resolved_muon)
            wrong_slot = dict(config)
            wrong_slot["ladder_slot"] = "top2"
            with self.assertRaisesRegex(ValueError, "tier/method/stage/slot"):
                train.validate_launch_config(wrong_slot, resolved_muon)
            wrong_candidate = dict(config)
            wrong_candidate["mai_selection_candidate"] = {"field": "learning_rate", "value": 0.0024}
            with self.assertRaisesRegex(ValueError, "selected candidate"):
                train.validate_launch_config(wrong_candidate, resolved_muon)
            mutable_config_threshold = dict(config)
            mutable_config_threshold["practical_equivalence_nll"] = 0.03
            with self.assertRaisesRegex(ValueError, "exactly 0.02"):
                train.validate_launch_config(mutable_config_threshold, resolved_muon)

            selected_recipe = comparison["selected_selection_recipe"]
            config20 = mai_configs.make_dense_20tpp_template("resolved-20", "124m")
            config20.update(
                launch_ready=True,
                recipe_resolution_required=False,
                registered_resume_determinism_required=True,
                five_tpp_comparison_artifact=str(comparison_path),
                five_tpp_comparison_artifact_sha256=mai_artifacts.sha256_file(comparison_path),
                mai_selection_candidate=comparison["selected_candidate"],
                mai_selection_recipe=selected_recipe,
                **selected_recipe,
            )
            resolved20 = self._resolved_muon(selected_recipe)
            train.validate_launch_config(config20, resolved20)
            wrong_recipe = dict(config20)
            wrong_recipe["learning_rate"] = 0.0020
            with self.assertRaisesRegex(ValueError, "min_lr must equal|recipe field learning_rate"):
                train.validate_launch_config(wrong_recipe, resolved20)
            wrong_fallback = dict(config20)
            wrong_fallback["muon_adamw_lr_scale"] = 0.2
            with self.assertRaisesRegex(ValueError, "muon_adamw_lr_scale=0.3|recipe field muon_adamw_lr_scale"):
                train.validate_launch_config(wrong_fallback, resolved20)

            historic_baseline = self._mai_record(
                identity=self._mai_identity(legacy=True),
            )
            self.assertEqual(
                mai_artifacts.validate_terminal_result(historic_baseline)["identity"]["run_contract"]["schema_version"],
                mai_artifacts.LEGACY_RUN_CONTRACT_SCHEMA_VERSION,
            )

            screens[0].write_text(json.dumps(self._mai_record(candidate_value=0.0016, nll=1.0, config_index=1)))
            with self.assertRaisesRegex(ValueError, "source result record hash mismatch"):
                mai_artifacts.validate_ranking_artifact(ranking)

    def test_block_fht_selection_contract_binds_structure_at_5tpp_and_20tpp(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ranking, ranking_path, comparison, comparison_path, screens, confirmations = self._make_mai_artifacts(
                root, method="block_fht"
            )
            dense_fit_path, dense_fit_coefficients = self._write_accepted_dense_fit(root)
            ranked = ranking["ranked_slots"]["top1"]
            recipe5 = ranked["selection_recipe"]
            config5 = mai_configs.make_fullattn_template(
                "resolved-block-5", "124m", 5.0,
                main_lr_multiplier=None,
                stage="full_attention_blockfht_confirmation_5tpp",
                role="confirmation",
                slot="top1",
                screen_only=False,
            )
            config5.update(
                launch_ready=True,
                recipe_resolution_required=False,
                registered_resume_determinism_required=True,
                zero_point_five_tpp_ranking_artifact=str(ranking_path),
                zero_point_five_tpp_ranking_artifact_sha256=mai_artifacts.sha256_file(ranking_path),
                dense_fit_gate_required=True,
                dense_fit_artifact=str(dense_fit_path),
                dense_fit_artifact_sha256=mai_artifacts.sha256_file(dense_fit_path),
                dense_fit_coefficients=dense_fit_coefficients,
                mai_selection_candidate=ranked["candidate"],
                mai_selection_recipe=recipe5,
                **recipe5,
            )
            resolved5 = self._resolved_muon(recipe5, method="block_fht")
            train.validate_launch_config(config5, resolved5)

            changed_controls = {
                "block_fht_targets": ["attn.c_proj"],
                "block_fht_layers": 3,
                "block_fht_latent_ratio": 0.02,
                "block_fht_latent_ratios": {"attn.c_proj": 0.02},
                "block_fht_match_gpt_init": False,
                "block_fht_latent_init_std": 0.03,
                "block_fht_seed": 1001,
                "block_fht_modulation_alpha": 1e-4,
                "block_fht_modulation_centered": True,
                "block_fht_weight_scale": 0.5,
                "block_fht_residual_base_scale": 0.1,
                "block_fht_output_gain_targets": ["attn.c_proj"],
                "block_fht_input_gain_targets": ["attn.c_proj"],
                "block_fht_ffn_pregelu_gain": True,
                "block_fht_ffn_pregelu_bias": True,
                "block_fht_ffn_pregelu_bias_init": -0.5,
                "block_fht_ffn_lowrank_rank": 1,
                "block_fht_ffn_lowrank_scale": 0.5,
                "block_fht_ffn_lowrank_init_std": 0.03,
                "block_fht_ffn_spectral_rank": 1,
                "block_fht_ffn_spectral_out_groups": 2,
                "block_fht_ffn_spectral_in_groups": 2,
                "block_fht_cproj_lowrank_rank": 1,
                "block_fht_cproj_lowrank_scale": 0.5,
                "block_fht_cproj_lowrank_init_std": 0.03,
                "block_fht_cproj_lowrank_mode": "block_fht",
                "block_fht_cproj_lowrank_latent_ratio": 0.02,
                "block_fht_cproj_lowrank_b_zero_init": False,
                "block_fht_cproj_lowrank_bias": True,
                "block_fht_cproj_tied_cfc_skip": True,
                "block_fht_cproj_tied_cfc_scale_init": 0.1,
                "block_fht_cproj_tied_cfc_vector": False,
                "block_fht_cproj_quarter_diag": True,
                "block_fht_cproj_quarter_diag_scale_init": 0.1,
                "block_fht_cproj_quarter_diag_init_std": 0.03,
                "block_fht_cproj_spectral_resid_rank": 1,
                "block_fht_cproj_spectral_resid_scale_init": 0.1,
                "block_fht_cproj_spectral_resid_seed": 1,
                "block_fht_ffn_postgelu_std_target": 0.1,
                "block_fht_ffn_postgelu_std_lambda": 0.1,
                "block_fht_cache_weights": False,
                "freeze_non_block_fht": True,
                "train_embeddings_when_frozen": True,
                "block_fht_latent_grad_normalize": True,
                "block_fht_latent_grad_target_rms": 0.02,
                "mapping_stability_lambda": 0.1,
                "mapping_stability_sigma": 0.002,
                "mapping_stability_temperature": 2.0,
                "mapping_norm_lambda": 0.1,
                "mapping_norm_target_rms": 0.04,
                "grad_clip": 0.5,
            }
            self.assertEqual(set(changed_controls), set(mai_artifacts.BLOCK_FHT_STRUCTURE_FIELDS))
            for field, changed_value in changed_controls.items():
                changed_config = dict(config5)
                changed_config[field] = changed_value
                with self.assertRaisesRegex(ValueError, f"run-contract field {field}"):
                    train.validate_launch_config(changed_config, resolved5)
                changed_args = argparse.Namespace(**vars(resolved5))
                setattr(changed_args, field, changed_value)
                with self.assertRaisesRegex(ValueError, f"run-contract field {field}"):
                    train.validate_launch_config(config5, changed_args)
            omitted_none_control = dict(config5)
            del omitted_none_control["block_fht_latent_ratios"]
            with self.assertRaisesRegex(ValueError, "run-contract field block_fht_latent_ratios"):
                train.validate_launch_config(omitted_none_control, resolved5)

            recipe20 = comparison["selected_selection_recipe"]
            config20 = mai_configs.make_fullattn_template(
                "resolved-block-20", "124m", 20.0,
                main_lr_multiplier=None,
                stage="full_attention_blockfht_selected_recipe_20tpp",
                role="selected_recipe",
                slot="selected_from_5tpp_comparison",
                screen_only=False,
            )
            config20.update(
                launch_ready=True,
                recipe_resolution_required=False,
                registered_resume_determinism_required=True,
                five_tpp_comparison_artifact=str(comparison_path),
                five_tpp_comparison_artifact_sha256=mai_artifacts.sha256_file(comparison_path),
                dense_fit_gate_required=True,
                dense_fit_artifact=str(dense_fit_path),
                dense_fit_artifact_sha256=mai_artifacts.sha256_file(dense_fit_path),
                dense_fit_coefficients=dense_fit_coefficients,
                mai_selection_candidate=comparison["selected_candidate"],
                mai_selection_recipe=recipe20,
                **recipe20,
            )
            resolved20 = self._resolved_muon(recipe20, method="block_fht")
            train.validate_launch_config(config20, resolved20)

            changed_ratio_20 = argparse.Namespace(**vars(resolved20))
            changed_ratio_20.block_fht_latent_ratio = 0.02
            with self.assertRaisesRegex(ValueError, "run-contract field block_fht_latent_ratio"):
                train.validate_launch_config(config20, changed_ratio_20)
            changed_targets_20 = dict(config20)
            changed_targets_20["block_fht_targets"] = ["attn.c_proj"]
            with self.assertRaisesRegex(ValueError, "run-contract field block_fht_targets"):
                train.validate_launch_config(changed_targets_20, resolved20)

            incompatible_structure = self._mai_record(
                method="block_fht",
                stage="full_attention_blockfht_screen_0p5tpp",
                slot="candidate-1.0",
                candidate_value=1.0,
                nll=3.2,
                config_index=99,
            )
            incompatible_structure["identity"]["run_contract"]["method_structure"]["block_fht"][
                "block_fht_modulation_alpha"
            ] = 1e-4
            incompatible_path = self._write_mai_record(root, "incompatible-block-structure.json", incompatible_structure)
            with self.assertRaisesRegex(ValueError, "incompatible shared identity"):
                mai_artifacts.build_ranking_artifact([screens[0], screens[1], incompatible_path])

            incompatible_confirmation = self._mai_record(
                method="block_fht",
                stage="full_attention_blockfht_confirmation_5tpp",
                role="confirmation",
                slot="top2",
                tpp=5.0,
                candidate_value=0.75,
                nll=3.10,
                config_index=100,
                selection_provenance={
                    "path": str(ranking_path.resolve()),
                    "sha256": mai_artifacts.sha256_file(ranking_path),
                    "schema_version": mai_artifacts.RANKING_ARTIFACT_SCHEMA_VERSION,
                    "slot": "top2",
                },
            )
            incompatible_confirmation["identity"]["run_contract"]["method_structure"]["block_fht"][
                "mapping_stability_lambda"
            ] = 0.1
            incompatible_confirmation_path = self._write_mai_record(
                root, "incompatible-block-confirmation.json", incompatible_confirmation
            )
            with self.assertRaisesRegex(ValueError, "incompatible shared identity"):
                mai_artifacts.build_comparison_artifact(
                    ranking_path, [confirmations[0], incompatible_confirmation_path]
                )

            missing_control = self._mai_record(
                method="block_fht",
                stage="full_attention_blockfht_screen_0p5tpp",
                slot="candidate-1.0",
                candidate_value=1.0,
            )
            del missing_control["identity"]["run_contract"]["method_structure"]["block_fht"][
                "block_fht_latent_ratios"
            ]
            with self.assertRaisesRegex(ValueError, "BlockFHT structure has incompatible fields"):
                mai_artifacts.validate_terminal_result(missing_control)

    def test_mai_selection_artifact_cli_publishes_rank_then_comparison(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screens = [
                self._write_mai_record(
                    root,
                    f"screen-{index}.json",
                    self._mai_record(candidate_value=candidate, nll=nll, config_index=index),
                )
                for index, (candidate, nll) in enumerate(((0.0016, 3.0), (0.0020, 3.1), (0.0024, 3.2)), start=1)
            ]
            ranking_path = root / "rank.json"
            with patch.object(sys, "argv", [
                "mai_selection_artifacts.py", "rank",
                "--record", str(screens[0]), "--record", str(screens[1]), "--record", str(screens[2]),
                "--output", str(ranking_path),
            ]):
                mai_artifacts.main()
            self.assertTrue(ranking_path.is_file())
            ranking_provenance = {
                "path": str(ranking_path.resolve()),
                "sha256": mai_artifacts.sha256_file(ranking_path),
                "schema_version": mai_artifacts.RANKING_ARTIFACT_SCHEMA_VERSION,
            }

            confirmations = [
                self._write_mai_record(
                    root,
                    f"confirmation-{slot}.json",
                    self._mai_record(
                        stage="dense_recipe_confirmation_5tpp",
                        role="confirmation",
                        slot=slot,
                        tpp=5.0,
                        candidate_value=candidate,
                        nll=nll,
                        config_index=index,
                        selection_provenance={**ranking_provenance, "slot": slot},
                    ),
                )
                for index, (slot, candidate, nll) in enumerate((("top1", 0.0016, 3.0), ("top2", 0.0020, 3.01)), start=10)
            ]
            comparison_path = root / "comparison.json"
            with patch.object(sys, "argv", [
                "mai_selection_artifacts.py", "compare", "--ranking", str(ranking_path),
                "--record", str(confirmations[0]), "--record", str(confirmations[1]),
                "--output", str(comparison_path),
            ]):
                mai_artifacts.main()
            comparison = json.loads(comparison_path.read_text())
            self.assertEqual(comparison["selected_slot"], "top1")

    def _write_terminal_bridge_inputs(self, root, config):
        config = json.loads(json.dumps(config))
        config["data_manifest_sha256"] = "b" * 64
        config_path = root / "launch.json"
        config_path.write_text(json.dumps(config, sort_keys=True))
        log_path = root / "train.log"
        log_path.write_text(
            "step 0: train loss 9.0000, val loss 9.1000\n"
            f"step {config['max_iters']}: train loss 2.1250, val loss 2.2500\n"
        )
        resolved = json.loads(json.dumps(config))
        resolved["fixed_eval_indices_sha256"] = "c" * 64
        identity = {
            "resolved_config": resolved,
            "config_sha256": hashlib.sha256(
                mai_artifacts.canonical_json_bytes(resolved)
            ).hexdigest(),
            "source_hashes": train.source_hashes(),
            "data_manifest": {"path": "/registered/data/manifest.json", "sha256": "b" * 64},
            "evaluation": {
                "protocol": config["eval_protocol_id"],
                "fixed_eval_indices": True,
                "fixed_eval_indices_sha256": "c" * 64,
                "fixed_eval_index_spec_sha256": config["fixed_eval_index_spec_sha256"],
                "fixed_eval_indices_protocol": config["fixed_eval_indices_protocol"],
                "eval_seed": config["eval_seed"],
                "eval_batch_size": config["eval_batch_size"],
                "eval_iters": config["eval_iters"],
                "block_size": config["block_size"],
            },
        }
        metadata_path = root / "ckpt.meta.json"
        metadata_path.write_text(json.dumps({
            "schema_version": "nanogpt_checkpoint_metadata_v2",
            "checkpoint_schema_version": train.CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_file": "ckpt.pt",
            "next_iter": config["max_iters"],
            "saved_at_unix": 123.0,
            "config_sha256": identity["config_sha256"],
            "run_identity": identity,
        }, sort_keys=True))
        status_path = root / "status.json"
        status_path.write_text(json.dumps({
            "state": "finished",
            "classification": "clean",
            "exit_code": 0,
            "alive": False,
            "failed": False,
            "killed": False,
            "max_iters": config["max_iters"],
            "config": str(config_path.resolve()),
            "log": str(log_path.resolve()),
        }, sort_keys=True))
        return {
            "config_path": config_path,
            "status_path": status_path,
            "log_path": log_path,
            "metadata_path": metadata_path,
        }

    def _build_terminal_bridge_result(self, inputs, *, accept=True):
        return mai_artifacts.build_terminal_result(
            config_path=inputs["config_path"],
            status_path=inputs["status_path"],
            log_path=inputs["log_path"],
            checkpoint_metadata_path=inputs["metadata_path"],
            accept=accept,
        )

    def test_terminal_result_builder_accepts_screen_and_confirmation_runs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_inputs = self._write_terminal_bridge_inputs(
                root, mai_configs.make_dense_screen("bridge-screen", "124m", 0.0016)
            )
            screen = self._build_terminal_bridge_result(screen_inputs)
            self.assertEqual(screen["ladder_role"], "screen_only")
            self.assertEqual(screen["terminal_held_out_nll"], 2.25)
            self.assertEqual(screen["completion"]["terminal_train_loss"], 2.125)
            self.assertEqual(screen["completion"]["terminal_val_loss"], 2.25)
            self.assertEqual(mai_artifacts.validate_terminal_result(screen), screen)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ranking, ranking_path, _, _, _, _ = self._make_mai_artifacts(root)
            ranked = ranking["ranked_slots"]["top1"]
            config = mai_configs.make_dense_confirmation_template("bridge-confirmation", "124m", "top1")
            config.update(
                launch_ready=True,
                recipe_resolution_required=False,
                zero_point_five_tpp_ranking_artifact=str(ranking_path),
                zero_point_five_tpp_ranking_artifact_sha256=mai_artifacts.sha256_file(ranking_path),
                mai_selection_candidate=ranked["candidate"],
                mai_selection_recipe=ranked["selection_recipe"],
                **ranked["selection_recipe"],
            )
            confirmation = self._build_terminal_bridge_result(
                self._write_terminal_bridge_inputs(root, config)
            )
            self.assertEqual(confirmation["ladder_role"], "confirmation")
            self.assertEqual(confirmation["selection_provenance"]["slot"], "top1")
            self.assertEqual(confirmation["selection_provenance"]["sha256"], mai_artifacts.sha256_file(ranking_path))

    def test_terminal_result_builder_rejects_status_log_and_metadata_failures(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs = self._write_terminal_bridge_inputs(
                root, mai_configs.make_dense_screen("bridge-reject", "124m", 0.0016)
            )
            status = json.loads(inputs["status_path"].read_text())
            status["config"] = str(root / "other-config.json")
            inputs["status_path"].write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "status config path"):
                self._build_terminal_bridge_result(inputs)

            status["config"] = str(inputs["config_path"].resolve())
            status["log"] = str(root / "other.log")
            inputs["status_path"].write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "status log path"):
                self._build_terminal_bridge_result(inputs)

            status["log"] = str(inputs["log_path"].resolve())
            status["state"] = "failed"
            inputs["status_path"].write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "not clean/finished"):
                self._build_terminal_bridge_result(inputs)

            status["state"] = "finished"
            inputs["status_path"].write_text(json.dumps(status))
            inputs["log_path"].write_text("")
            with self.assertRaisesRegex(ValueError, "terminal evaluation is missing"):
                self._build_terminal_bridge_result(inputs)

            inputs["log_path"].write_text("step 0: train loss 3.0, val loss 3.1\n")
            with self.assertRaisesRegex(ValueError, "terminal evaluation iteration mismatch"):
                self._build_terminal_bridge_result(inputs)

            max_iters = json.loads(inputs["config_path"].read_text())["max_iters"]
            inputs["log_path"].write_text(f"step {max_iters}: train loss nan, val loss 2.1\n")
            with self.assertRaisesRegex(ValueError, "losses must be finite"):
                self._build_terminal_bridge_result(inputs)

            inputs["log_path"].write_text(
                f"step {max_iters}: train loss 2.0, val loss 2.1\n"
                f"step {max_iters}: train loss 2.2, val loss 2.3\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate terminal evaluation"):
                self._build_terminal_bridge_result(inputs)

            inputs["log_path"].write_text(f"step {max_iters}: train loss 2.0, val loss 2.1\n")
            metadata = json.loads(inputs["metadata_path"].read_text())
            metadata["run_identity"]["evaluation"]["protocol"] = "unexpected_protocol"
            inputs["metadata_path"].write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "evaluation field protocol disagrees"):
                self._build_terminal_bridge_result(inputs)

            metadata["run_identity"]["evaluation"]["protocol"] = json.loads(
                inputs["config_path"].read_text()
            )["eval_protocol_id"]
            del metadata["run_identity"]
            inputs["metadata_path"].write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "run_identity is required"):
                self._build_terminal_bridge_result(inputs)

    def test_terminal_result_builder_rejects_metadata_contract_mismatch_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs = self._write_terminal_bridge_inputs(
                root, mai_configs.make_dense_screen("bridge-contract", "124m", 0.0016)
            )
            metadata = json.loads(inputs["metadata_path"].read_text())
            metadata["run_identity"]["resolved_config"]["batch_size"] += 1
            metadata["run_identity"]["config_sha256"] = hashlib.sha256(
                mai_artifacts.canonical_json_bytes(metadata["run_identity"]["resolved_config"])
            ).hexdigest()
            metadata["config_sha256"] = metadata["run_identity"]["config_sha256"]
            inputs["metadata_path"].write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "launch config field batch_size"):
                self._build_terminal_bridge_result(inputs)

            for field, changed_value in {
                "data_dir": "/other/registered/data",
                "fixed_eval_indices": False,
                "fixed_eval_indices_protocol": "different_fixed_index_construction",
                "eval_seed": 20260716,
                "eval_batch_size": 17,
                "eval_iters": 401,
                "fixed_eval_index_spec_sha256": "d" * 64,
            }.items():
                with self.subTest(field=field):
                    inputs = self._write_terminal_bridge_inputs(
                        root, mai_configs.make_dense_screen(f"bridge-{field}", "124m", 0.0016)
                    )
                    metadata = json.loads(inputs["metadata_path"].read_text())
                    resolved = metadata["run_identity"]["resolved_config"]
                    resolved[field] = changed_value
                    evaluation_field = {
                        "fixed_eval_indices": "fixed_eval_indices",
                        "fixed_eval_indices_protocol": "fixed_eval_indices_protocol",
                        "eval_seed": "eval_seed",
                        "eval_batch_size": "eval_batch_size",
                        "eval_iters": "eval_iters",
                        "fixed_eval_index_spec_sha256": "fixed_eval_index_spec_sha256",
                    }.get(field)
                    if evaluation_field is not None:
                        metadata["run_identity"]["evaluation"][evaluation_field] = changed_value
                    config_sha256 = hashlib.sha256(
                        mai_artifacts.canonical_json_bytes(resolved)
                    ).hexdigest()
                    metadata["run_identity"]["config_sha256"] = config_sha256
                    metadata["config_sha256"] = config_sha256
                    inputs["metadata_path"].write_text(json.dumps(metadata))
                    error = "lacks fixed evaluation indices" if field == "fixed_eval_indices" else f"launch config field {field}"
                    with self.assertRaisesRegex(ValueError, error):
                        self._build_terminal_bridge_result(inputs)

            inputs = self._write_terminal_bridge_inputs(
                root, mai_configs.make_dense_screen("bridge-output", "124m", 0.0016)
            )
            output = root / "terminal-result.json"
            with patch.object(sys, "argv", [
                "mai_selection_artifacts.py", "terminal-result",
                "--config", str(inputs["config_path"]),
                "--status", str(inputs["status_path"]),
                "--log", str(inputs["log_path"]),
                "--checkpoint-metadata", str(inputs["metadata_path"]),
                "--output", str(output), "--accept",
            ]):
                mai_artifacts.main()
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                mai_artifacts.write_immutable_artifact(output, json.loads(output.read_text()))
            with self.assertRaisesRegex(ValueError, "explicit acceptance"):
                self._build_terminal_bridge_result(inputs, accept=False)

    def test_terminal_result_builder_requires_a_launchable_v2_config(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)

            def build(config):
                return self._build_terminal_bridge_result(
                    self._write_terminal_bridge_inputs(root, config)
                )

            blocked = mai_configs.make_dense_screen("bridge-blocked", "124m", 0.0016)
            blocked["launch_ready"] = False
            with self.assertRaisesRegex(ValueError, "launch_ready=true"):
                build(blocked)

            unresolved = mai_configs.make_dense_screen("bridge-unresolved", "124m", 0.0016)
            unresolved["recipe_resolution_required"] = True
            with self.assertRaisesRegex(ValueError, "requires recipe resolution"):
                build(unresolved)

            missing_optimizer = mai_configs.make_dense_screen("bridge-missing-optimizer", "124m", 0.0016)
            missing_optimizer["muon_ns_steps"] = None
            with self.assertRaisesRegex(ValueError, "missing optimizer field muon_ns_steps"):
                build(missing_optimizer)

            incoherent_optimizer = mai_configs.make_dense_screen("bridge-incoherent-optimizer", "124m", 0.0016)
            incoherent_optimizer["min_lr"] = 0.0002
            with self.assertRaisesRegex(ValueError, "min_lr must equal 0.1"):
                build(incoherent_optimizer)

            block_templates = (
                mai_configs.make_fullattn_template(
                    "bridge-block-0p5", "124m", 0.5,
                    main_lr_multiplier=0.5,
                    stage="full_attention_blockfht_screen_0p5tpp",
                    role="screen_only",
                    slot="mult0p50",
                    screen_only=True,
                ),
                mai_configs.make_fullattn_template(
                    "bridge-block-5", "124m", 5.0,
                    main_lr_multiplier=None,
                    stage="full_attention_blockfht_confirmation_5tpp",
                    role="confirmation",
                    slot="top1",
                    screen_only=False,
                ),
                mai_configs.make_fullattn_template(
                    "bridge-block-20", "124m", 20.0,
                    main_lr_multiplier=None,
                    stage="full_attention_blockfht_selected_recipe_20tpp",
                    role="selected_recipe",
                    slot="selected_from_5tpp_comparison",
                    screen_only=False,
                ),
            )
            for template in block_templates:
                with self.subTest(tpp=template["planned_tpp"]):
                    with self.assertRaisesRegex(ValueError, "launch_ready=true"):
                        build(template)

            unpinned_block = mai_configs.make_fullattn_template(
                "bridge-unpinned-block", "124m", 0.5,
                main_lr_multiplier=0.5,
                stage="full_attention_blockfht_screen_0p5tpp",
                role="screen_only",
                slot="mult0p50",
                screen_only=True,
            )
            unpinned_block.update(
                launch_ready=True,
                recipe_resolution_required=False,
                learning_rate=0.001,
                min_lr=0.0001,
            )
            with self.assertRaisesRegex(ValueError, "missing immutable dense-fit artifact"):
                build(unpinned_block)

            draft_fit = root / "draft-dense-fit.json"
            draft_fit.write_text(json.dumps({
                "schema_version": "dense_scaling_fit_v1",
                "artifact_kind": "registered_dense_scaling_fit",
                "state": "DRAFT_NOT_ACCEPTED",
            }))
            unpinned_block.update(
                dense_fit_artifact=str(draft_fit),
                dense_fit_artifact_sha256=mai_artifacts.sha256_file(draft_fit),
                dense_fit_coefficients={"A": 1.0, "alpha": 1.0, "E": 0.0},
            )
            with self.assertRaisesRegex(ValueError, "not ACCEPTED"):
                build(unpinned_block)

    def test_terminal_builder_rejects_fixed_eval_flag_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs = self._write_terminal_bridge_inputs(
                root, mai_configs.make_dense_screen("bridge-fixed-eval-flag", "124m", 0.0016)
            )
            metadata = json.loads(inputs["metadata_path"].read_text())
            resolved = metadata["run_identity"]["resolved_config"]
            resolved["fixed_eval_indices"] = False
            config_sha256 = hashlib.sha256(
                mai_artifacts.canonical_json_bytes(resolved)
            ).hexdigest()
            metadata["run_identity"]["config_sha256"] = config_sha256
            metadata["config_sha256"] = config_sha256
            inputs["metadata_path"].write_text(json.dumps(metadata, sort_keys=True))
            with self.assertRaisesRegex(
                ValueError, "evaluation field fixed_eval_indices disagrees"
            ):
                self._build_terminal_bridge_result(inputs)

    def test_terminal_builder_cross_binds_every_fixed_eval_identity_field(self):
        bindings = {
            "protocol": ("eval_protocol_id", "unexpected_fixed_protocol"),
            "eval_seed": ("eval_seed", 20260716),
            "eval_batch_size": ("eval_batch_size", 17),
            "eval_iters": ("eval_iters", 401),
            "block_size": ("block_size", 512),
            "fixed_eval_index_spec_sha256": ("fixed_eval_index_spec_sha256", "d" * 64),
            "fixed_eval_indices_protocol": (
                "fixed_eval_indices_protocol", "different_fixed_index_construction"
            ),
            "fixed_eval_indices_sha256": ("fixed_eval_indices_sha256", "d" * 64),
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for evaluation_field, (resolved_field, changed_value) in bindings.items():
                for mutate_resolved in (False, True):
                    with self.subTest(field=evaluation_field, mutate_resolved=mutate_resolved):
                        inputs = self._write_terminal_bridge_inputs(
                            root,
                            mai_configs.make_dense_screen(
                                f"bridge-cross-bind-{evaluation_field}-{mutate_resolved}", "124m", 0.0016
                            ),
                        )
                        metadata = json.loads(inputs["metadata_path"].read_text())
                        if mutate_resolved:
                            resolved = metadata["run_identity"]["resolved_config"]
                            resolved[resolved_field] = changed_value
                            config_sha256 = hashlib.sha256(
                                mai_artifacts.canonical_json_bytes(resolved)
                            ).hexdigest()
                            metadata["run_identity"]["config_sha256"] = config_sha256
                            metadata["config_sha256"] = config_sha256
                        else:
                            metadata["run_identity"]["evaluation"][evaluation_field] = changed_value
                        inputs["metadata_path"].write_text(json.dumps(metadata, sort_keys=True))
                        with self.assertRaisesRegex(
                            ValueError, f"evaluation field {evaluation_field} disagrees"
                        ):
                            self._build_terminal_bridge_result(inputs)

    def test_v2_launch_policy_parity_between_train_and_terminal_builder(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)

            def rejects_both(name, config, resolved):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        train.validate_launch_config(config, resolved)
                    with self.assertRaises(ValueError):
                        self._build_terminal_bridge_result(
                            self._write_terminal_bridge_inputs(root, config)
                        )

            screen = mai_configs.make_dense_screen("bridge-parity-screen", "124m", 0.0016)
            screen_resolved = self._resolved_muon({
                "learning_rate": screen["learning_rate"],
                "min_lr": screen["min_lr"],
                "muon_adamw_lr_scale": screen["muon_adamw_lr_scale"],
            })
            train.validate_launch_config(screen, screen_resolved)
            self._build_terminal_bridge_result(self._write_terminal_bridge_inputs(root, screen))
            for field, changed_value in {
                "launch_ready": False,
                "recipe_resolution_required": True,
                "checkpoint_history": True,
                "save_checkpoint": False,
                "checkpoint_wall_clock_seconds": 3600,
                "beta1": 0.8,
                "practical_equivalence_nll": 0.03,
            }.items():
                changed = dict(screen)
                changed[field] = changed_value
                rejects_both(field, changed, screen_resolved)

            ranking, ranking_path, comparison, comparison_path, _, _ = self._make_mai_artifacts(root)
            ranked = ranking["ranked_slots"]["top1"]
            confirmation = mai_configs.make_dense_confirmation_template(
                "bridge-parity-5", "124m", "top1"
            )
            confirmation.update(
                launch_ready=True,
                recipe_resolution_required=False,
                zero_point_five_tpp_ranking_artifact=str(ranking_path),
                zero_point_five_tpp_ranking_artifact_sha256=mai_artifacts.sha256_file(ranking_path),
                mai_selection_candidate=ranked["candidate"],
                mai_selection_recipe=ranked["selection_recipe"],
                **ranked["selection_recipe"],
            )
            confirmation_resolved = self._resolved_muon(ranked["selection_recipe"])
            train.validate_launch_config(confirmation, confirmation_resolved)
            self._build_terminal_bridge_result(
                self._write_terminal_bridge_inputs(root, confirmation)
            )
            for field, changed_value in {
                "zero_point_five_tpp_ranking_artifact_required": False,
                "zero_point_five_tpp_ranking_artifact_schema": "wrong_schema",
                "zero_point_five_tpp_ranking_artifact_sha256": "0" * 64,
            }.items():
                changed = dict(confirmation)
                changed[field] = changed_value
                rejects_both(f"5tpp-{field}", changed, confirmation_resolved)

            selected = mai_configs.make_dense_20tpp_template("bridge-parity-20", "124m")
            selected_recipe = comparison["selected_selection_recipe"]
            selected.update(
                launch_ready=True,
                recipe_resolution_required=False,
                five_tpp_comparison_artifact=str(comparison_path),
                five_tpp_comparison_artifact_sha256=mai_artifacts.sha256_file(comparison_path),
                mai_selection_candidate=comparison["selected_candidate"],
                mai_selection_recipe=selected_recipe,
                **selected_recipe,
            )
            selected_resolved = self._resolved_muon(selected_recipe)
            train.validate_launch_config(selected, selected_resolved)
            self._build_terminal_bridge_result(self._write_terminal_bridge_inputs(root, selected))
            for field, changed_value in {
                "five_tpp_comparison_artifact_required": False,
                "five_tpp_comparison_artifact_schema": "wrong_schema",
                "five_tpp_comparison_artifact_sha256": "0" * 64,
            }.items():
                changed = dict(selected)
                changed[field] = changed_value
                rejects_both(f"20tpp-{field}", changed, selected_resolved)

            dense_fit_path, dense_fit_coefficients = self._write_accepted_dense_fit(
                root, "parity-dense-fit.json"
            )
            block = mai_configs.make_fullattn_template(
                "bridge-parity-block", "124m", 0.5,
                main_lr_multiplier=0.5,
                stage="full_attention_blockfht_screen_0p5tpp",
                role="screen_only",
                slot="mult0p50",
                screen_only=True,
            )
            block.update(
                launch_ready=True,
                recipe_resolution_required=False,
                learning_rate=0.001,
                min_lr=0.0001,
                dense_fit_gate_required=True,
                dense_fit_artifact=str(dense_fit_path),
                dense_fit_artifact_sha256=mai_artifacts.sha256_file(dense_fit_path),
                dense_fit_coefficients=dense_fit_coefficients,
            )
            block_recipe = {
                "learning_rate": block["learning_rate"],
                "min_lr": block["min_lr"],
                "muon_adamw_lr_scale": block["muon_adamw_lr_scale"],
                "candidate_main_lr_multiplier": block["candidate_main_lr_multiplier"],
            }
            block_resolved = self._resolved_muon(block_recipe, method="block_fht")
            train.validate_launch_config(block, block_resolved)
            self._build_terminal_bridge_result(self._write_terminal_bridge_inputs(root, block))
            changed_block = dict(block)
            changed_block["dense_fit_gate_required"] = False
            rejects_both("blockfht-dense-fit", changed_block, block_resolved)

    def test_atomic_checkpoint_replaces_latest_then_publishes_metadata(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "ckpt.pt"
            checkpoint = {
                "schema_version": train.CHECKPOINT_SCHEMA_VERSION,
                "next_iter": 7,
                "saved_at_unix": 123.0,
                "run_identity": {"config_sha256": "a" * 64},
                "payload": torch.tensor([1, 2, 3]),
            }
            train.atomic_save_checkpoint(path, checkpoint)
            restored = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(restored["next_iter"], 7)
            metadata = json.loads((root / "ckpt.meta.json").read_text())
            self.assertEqual(metadata["schema_version"], "nanogpt_checkpoint_metadata_v2")
            self.assertEqual(metadata["next_iter"], 7)
            self.assertEqual(metadata["checkpoint_file"], "ckpt.pt")
            self.assertEqual(metadata["run_identity"], checkpoint["run_identity"])
            self.assertFalse(list(root.glob("ckpt_iter*.pt")))
            self.assertFalse(list(root.glob("*.tmp")))

    def test_resume_identity_mismatch_is_rejected_before_model_setup(self):
        gpt_config = train.GPTConfig(block_size=32, vocab_size=64, n_layer=1, n_head=1, n_embd=16)
        identity = {
            "resolved_config": {"model_seed": 1337},
            "config_sha256": "a" * 64,
            "source_hashes": {"examples/nanogpt/train.py": "b" * 64},
            "data_manifest": {"path": "/data/manifest.json", "sha256": "c" * 64},
            "evaluation": {"protocol": "fixed", "fixed_eval_indices_sha256": "d" * 64},
        }
        checkpoint = {
            "schema_version": train.CHECKPOINT_SCHEMA_VERSION,
            "model": {},
            "optimizer": {},
            "grad_scaler": {},
            "model_config": train.asdict(gpt_config),
            "next_iter": 3,
            "best_val_loss": 1.0,
            "train_data_generator_state": torch.Generator(device="cpu").get_state(),
            "run_identity": identity,
            "saved_at_unix": 1.0,
            "block_fht_cache_state": "flushed_not_serialized",
            **train.capture_rng_state(),
        }
        changed = dict(identity)
        changed["source_hashes"] = {"examples/nanogpt/train.py": "e" * 64}
        with self.assertRaisesRegex(ValueError, "source identity mismatch"):
            train.validate_resume_checkpoint(
                checkpoint,
                run_identity=changed,
                expected_model_config=gpt_config,
                registered_resume_required=True,
            )

    def test_phase_gate_rejects_candidate_without_pinned_dense_fit(self):
        resolved_muon = argparse.Namespace(
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
        with self.assertRaisesRegex(ValueError, "missing immutable dense-fit artifact"):
            train.validate_launch_config(
                {
                    "launch_ready": True,
                    "dense_fit_gate_required": True,
                    "dense_fit_artifact": None,
                    "dense_fit_artifact_sha256": None,
                    "dense_fit_coefficients": None,
                },
                resolved_muon,
            )


if __name__ == "__main__":
    unittest.main()
