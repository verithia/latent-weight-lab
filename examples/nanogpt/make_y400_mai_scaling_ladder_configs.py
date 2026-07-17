from __future__ import annotations

"""Generate prospective v2 registered, baseline-first MAI Y400 ladder controls.

This is deliberately a config/queue generator only.  It never launches a run.
Recipe-dependent templates remain launch-disabled until their fixed-window
terminal-NLL selection dependency has been resolved and recorded.
"""

import hashlib
import json
import math
from pathlib import Path

try:  # Support both ``python -m`` and this repository's direct generator command.
    from examples.nanogpt.mai_selection_artifacts import (
        COMPARISON_ARTIFACT_SCHEMA_VERSION,
        POLICY_VERSION,
        PRACTICAL_EQUIVALENCE_NLL,
        REGISTERED_V2_BLOCK_FHT_METHOD_SPEC,
        RANKING_ARTIFACT_SCHEMA_VERSION,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct invocation.
    from mai_selection_artifacts import (
        COMPARISON_ARTIFACT_SCHEMA_VERSION,
        POLICY_VERSION,
        PRACTICAL_EQUIVALENCE_NLL,
        REGISTERED_V2_BLOCK_FHT_METHOD_SPEC,
        RANKING_ARTIFACT_SCHEMA_VERSION,
    )


DATA_DIR = "/root/userdata/MappingNetworks/data/finewebedu_20b"
OUT_DIR = "/root/userdata/MappingNetworks/outputs/y400_mai_scaling_ladder_v2"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
QUEUE_PATH = Path(__file__).resolve().parent / "y400_mai_scaling_ladder_queue.tsv"
REMOTE_CONFIG_DIR = "/root/userdata/MappingNetworks/latent-weight-lab/examples/nanogpt/configs"

TIERS = {
    "124m": {"n_layer": 12, "n_embd": 768, "n_head": 12},
    "350m": {"n_layer": 24, "n_embd": 1024, "n_head": 16},
    "690m": {"n_layer": 32, "n_embd": 1280, "n_head": 20},
    "985m": {"n_layer": 32, "n_embd": 1536, "n_head": 24},
}
BATCH_SHAPES = {
    "124m": (32, 8),
    "350m": (32, 8),
    "690m": (16, 16),
    "985m": (16, 16),
}
EXPECTED_MATERIALIZED_PARAM_COUNTS = {
    "124m": 124_373_760,
    "350m": 354_599_936,
    "690m": 694_928_640,
    "985m": 984_909_312,
}
EXPECTED_TPP_SCHEDULES = {
    "124m": {
        0.5: (62_186_880, 238, 62_390_272),
        5.0: (621_868_800, 2_373, 622_067_712),
        20.0: (2_487_475_200, 9_489, 2_487_484_416),
    },
    "350m": {
        0.5: (177_299_968, 677, 177_471_488),
        5.0: (1_772_999_680, 6_764, 1_773_142_016),
        20.0: (7_091_998_720, 27_054, 7_092_043_776),
    },
    "690m": {
        0.5: (347_464_320, 1_326, 347_602_944),
        5.0: (3_474_643_200, 13_255, 3_474_718_720),
        20.0: (13_898_572_800, 53_019, 13_898_612_736),
    },
    "985m": {
        0.5: (492_454_656, 1_879, 492_568_576),
        5.0: (4_924_546_560, 18_786, 4_924_637_184),
        20.0: (19_698_186_240, 75_143, 19_698_286_592),
    },
}
BASELINE_SCREEN_LRS = (0.0016, 0.0020, 0.0024)
CANDIDATE_MAIN_LR_MULTIPLIERS = (0.5, 0.75, 1.0)
FIXED_ADAMW_LR_SCALE = 0.3
FULL_ATTENTION_TARGETS = ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj"]
# Evaluation is deliberately decoupled from every rung's training microbatch.
# This shape is conservative enough for the 985M rung and gives every rung the
# same ordered train/validation windows and evaluated-token budget.
REGISTERED_EVAL_PROTOCOL = "mai_ladder_fixed_eval_indices_v2"
REGISTERED_EVAL_BATCH_SIZE = 16
REGISTERED_EVAL_ITERS = 400

PROVENANCE = {
    "protocol": "FineWeb-Edu 20B mixed potentially-overlapping continuation",
    "data_manifest_sha256": "1e1de075c504906a93637bd79450d30da2243797d2e1d3e33f2392d9492ddf8b",
    "train_tokens": 20_000_000_000,
    "validation_tokens": 20_000_000,
    "immutable_prefix": {
        "source": "sample-10BT",
        "train_token_range": [0, 8_150_000_000],
    },
    "continuation": {
        "source": "sample-100BT",
        "train_token_range": [8_150_000_000, 20_000_000_000],
        "potentially_overlapping": True,
    },
}
SELECTION_NOTE = (
    "0.5TPP is SCREEN_ONLY: rank stable candidates by terminal held-out NLL and "
    "populate ordered top1/top2 5TPP slots. At 5TPP, select the lower terminal NLL "
    "when the gap exceeds 0.02; otherwise retain the 0.5TPP leader as a practical tie."
)
REPORTING_NOTE = (
    "EGTime is primary. EG_FLOPs must include BlockFHT decode/materialization, cache "
    "prepare/flush, and generated-weight gradient mapping, or be labelled nominal."
)


def estimate_active_params(
    n_layer: int,
    n_embd: int,
    vocab_size: int = 50304,
    block_size: int = 1024,
) -> int:
    """Count materialized GPT parameters with no biases and tied embeddings."""
    return int(vocab_size * n_embd + block_size * n_embd + n_layer * (12 * n_embd * n_embd + 2 * n_embd) + n_embd)


def lr_slug(lr: float) -> str:
    return f"lr{int(round(lr * 1e4)):02d}e4"


def multiplier_slug(multiplier: float) -> str:
    return f"mult{multiplier:.2f}".replace(".", "p")


def eval_index_spec_sha256(eval_batch_size: int, eval_iters: int) -> str:
    """Hash the immutable inputs that define the fixed evaluation windows.

    ``train.py`` logs the runtime digest of the materialized indices. This
    companion hash is available at config-generation time and makes the data
    lengths, seed, and window shape auditable before a run is launched.
    """
    spec = {
        "eval_batch_size": eval_batch_size,
        "block_size": 1024,
        "eval_iters": eval_iters,
        "eval_tokens_per_split": eval_batch_size * 1024 * eval_iters,
        "eval_total_tokens": 2 * eval_batch_size * 1024 * eval_iters,
        "eval_seed": 20260715,
        "index_generator": "torch_cpu_split_local_generators_v2",
        "protocol": REGISTERED_EVAL_PROTOCOL,
        "train_tokens": PROVENANCE["train_tokens"],
        "validation_tokens": PROVENANCE["validation_tokens"],
    }
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recipe_template_fields(*, stage: str, dependency: str) -> dict:
    """Mark blocked recipe templates so they cannot masquerade as selected recipes."""
    return {
        "learning_rate": None,
        "min_lr": None,
        "recipe_resolution_required": True,
        "recipe_resolution_stage": stage,
        "recipe_resolution_dependency": dependency,
        "selection_endpoint": "terminal held-out NLL on fixed eval windows",
        "instability_policy": "hard reject NaN, Inf, divergence, or failed terminal evaluation",
        "practical_equivalence_policy": (
            "at 5TPP select the lower stable terminal NLL only when its gap exceeds "
            f"{PRACTICAL_EQUIVALENCE_NLL:.2f}; otherwise retain the 0.5TPP leader and "
            "record PRACTICAL_TIE"
        ),
    }


def make_config(name: str, tier: str, tpp: float, *, launch_ready: bool) -> dict:
    model = TIERS[tier]
    batch_size, grad_accum = BATCH_SHAPES[tier]
    if model["n_embd"] // model["n_head"] != 64:
        raise ValueError(f"{tier} violates the registered head dimension of 64")
    active = estimate_active_params(model["n_layer"], model["n_embd"])
    planned_tokens = int(active * tpp)
    tokens_per_iter = batch_size * 1024 * grad_accum
    max_iters = max(1, math.ceil(planned_tokens / tokens_per_iter))
    scheduled_tokens = max_iters * tokens_per_iter
    return {
        "data_dir": DATA_DIR,
        "out_dir": f"{OUT_DIR}/{name}",
        "method": "baseline",
        "model_tier": tier,
        "n_layer": model["n_layer"],
        "n_embd": model["n_embd"],
        "n_head": model["n_head"],
        "vocab_size": 50304,
        "block_size": 1024,
        "bias": False,
        "dropout": 0.0,
        "tie_word_embeddings": True,
        "estimated_active_params": active,
        "active_parameter_count_definition": "materialized GPT parameters; never substitute latent trainable count",
        "planned_tokens": planned_tokens,
        "planned_tpp": tpp,
        "tokens_per_iter": tokens_per_iter,
        "max_iters": max_iters,
        "lr_decay_iters": max_iters,
        "scheduled_tokens": scheduled_tokens,
        "scheduled_tpp": scheduled_tokens / active,
        "schedule_rounding": "ceil(planned_tokens / tokens_per_iter); lr decay and terminal eval occur at max_iters",
        "warmup_iters": max(10, max_iters // 100),
        # The registered fixed window evaluates 6.55M tokens per split.  Use
        # quarterly trajectory points plus the required terminal point rather
        # than spending the short HPO budget repeatedly re-evaluating it.
        "eval_interval": max(1, math.ceil(max_iters / 4)),
        "eval_batch_size": REGISTERED_EVAL_BATCH_SIZE,
        "eval_iters": REGISTERED_EVAL_ITERS,
        "eval_tokens_per_split": REGISTERED_EVAL_BATCH_SIZE * 1024 * REGISTERED_EVAL_ITERS,
        "eval_total_tokens": 2 * REGISTERED_EVAL_BATCH_SIZE * 1024 * REGISTERED_EVAL_ITERS,
        "eval_batch_shape_policy": (
            "shared b16 evaluation independent of rung training microbatch; ordered "
            "fixed train/val windows and token budget must match across all rungs"
        ),
        "terminal_eval_required": True,
        "terminal_eval_protocol": "evaluate at max_iters even when it is off periodic cadence",
        "eval_protocol_id": REGISTERED_EVAL_PROTOCOL,
        "fixed_eval_indices": True,
        "model_seed": 1337,
        "train_data_seed": 20260714,
        "eval_seed": 20260715,
        "training_sampling_protocol": "dedicated_cpu_generator_train_data_seed_v1",
        "fixed_eval_indices_protocol": "split_local_cpu_generators_eval_seed_plus_split_offset_v2_shared_b16",
        "fixed_eval_index_spec_sha256": eval_index_spec_sha256(
            REGISTERED_EVAL_BATCH_SIZE, REGISTERED_EVAL_ITERS
        ),
        "fixed_eval_index_runtime_digest": (
            "record train.py rng_eval_metadata.fixed_eval_indices_sha256 before launch; "
            "all four rungs and their baseline/candidate pairs must match"
        ),
        "mixed_continuation_provenance": PROVENANCE,
        "data_manifest_sha256": PROVENANCE["data_manifest_sha256"],
        "optimizer": "muon",
        "muon_adamw_lr_scale": FIXED_ADAMW_LR_SCALE,
        "adamw_fallback_scale_policy": "fixed_0.3_for_registered_main_queue",
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "dtype": "bfloat16",
        "device": "cuda",
        "compile": False,
        "registered_execution_stack": "eager PyTorch/CUDA BF16; torch.compile=false",
        "save_checkpoint": True,
        "checkpoint_history": False,
        "checkpoint_wall_clock_seconds": 7200,
        "registered_resume_determinism_required": True,
        "registered_resume_protocol": "atomic_latest_checkpoint_v2_with_pre_current_batch_and_full_rng_state",
        "launch_ready": launch_ready,
        "recipe_resolution_required": False,
        "mai_ladder_policy_version": POLICY_VERSION,
        "practical_equivalence_nll": PRACTICAL_EQUIVALENCE_NLL,
        "ladder_interpretation": "initial descriptive four-rung, one-20TPP matched-size ladder only",
        "primary_efficiency_metric": "EGTime",
        "eg_flops_accounting": REPORTING_NOTE,
        "prelaunch_provenance_requirements": (
            "record source hashes, resolved config SHA256, data manifest SHA256, and "
            "runtime fixed-evaluation index digest"
        ),
    }


def make_dense_screen(name: str, tier: str, lr: float) -> dict:
    config = make_config(name, tier, 0.5, launch_ready=True)
    config.update(
        learning_rate=lr,
        min_lr=lr * 0.1,
        hpo_stage="dense_screen_0p5tpp",
        candidate_scope="baseline_only_no_block_fht",
        ladder_role="screen_only",
        ladder_slot=lr_slug(lr),
        screen_only=True,
        screen_only_resolution="rank stable candidates to populate ordered top1/top2 5TPP slots",
        selection_endpoint="terminal held-out NLL on fixed eval windows",
        instability_policy="hard reject NaN, Inf, divergence, or failed terminal evaluation",
        practical_equivalence_policy=recipe_template_fields(stage="unused", dependency="unused")[
            "practical_equivalence_policy"
        ],
    )
    return config


def make_dense_confirmation_template(name: str, tier: str, slot: str) -> dict:
    config = make_config(name, tier, 5.0, launch_ready=False)
    config.update(recipe_template_fields(
        stage="dense_recipe_confirmation_5tpp",
        dependency=(
            f"immutable {tier} 0.5TPP stable-candidate ranking artifact that assigns {slot} "
            "from the three SCREEN_ONLY rows"
        ),
    ))
    config.update(
        hpo_stage="dense_recipe_confirmation_5tpp",
        candidate_scope="baseline_only_no_block_fht",
        ladder_role="confirmation",
        ladder_slot=slot,
        confirmation_slot=slot,
        confirmation_source=f"ordered stable 0.5TPP {slot} candidate",
        launch_block_reason="BLOCKED_ON_0P5TPP_TOP2_RESOLUTION",
    )
    config.update(zero_point_five_tpp_ranking_fields(tier, "baseline", slot))
    return config


def zero_point_five_tpp_ranking_fields(tier: str, method: str, slot: str) -> dict:
    """Declare the immutable ranking pin required for a resolved 5TPP slot."""
    return {
        "zero_point_five_tpp_ranking_artifact_required": True,
        "zero_point_five_tpp_ranking_artifact": None,
        "zero_point_five_tpp_ranking_artifact_sha256": None,
        "zero_point_five_tpp_ranking_artifact_schema": RANKING_ARTIFACT_SCHEMA_VERSION,
        "zero_point_five_tpp_ranking_tier": tier,
        "zero_point_five_tpp_ranking_method": method,
        "zero_point_five_tpp_ranking_hpo_stage": (
            "dense_screen_0p5tpp" if method == "baseline" else "full_attention_blockfht_screen_0p5tpp"
        ),
        "zero_point_five_tpp_ranking_slot": slot,
        "mai_selection_candidate": None,
        "mai_selection_recipe": None,
    }


def five_tpp_comparison_fields(tier: str, method: str) -> dict:
    """Declare the immutable comparison pin required for a resolved 20TPP recipe."""
    return {
        "five_tpp_comparison_artifact_required": True,
        "five_tpp_comparison_artifact": None,
        "five_tpp_comparison_artifact_sha256": None,
        "five_tpp_comparison_artifact_schema": COMPARISON_ARTIFACT_SCHEMA_VERSION,
        "five_tpp_comparison_tier": tier,
        "five_tpp_comparison_method": method,
        "five_tpp_comparison_hpo_stage": (
            "dense_recipe_confirmation_5tpp"
            if method == "baseline"
            else "full_attention_blockfht_confirmation_5tpp"
        ),
        "mai_selection_candidate": None,
        "mai_selection_recipe": None,
        "five_tpp_selection_rule": (
            f"if abs(top1_terminal_nll - top2_terminal_nll) > {PRACTICAL_EQUIVALENCE_NLL:.2f}, "
            "select the lower 5TPP terminal NLL; otherwise retain the 0.5TPP leader"
        ),
    }


def make_dense_20tpp_template(name: str, tier: str) -> dict:
    config = make_config(name, tier, 20.0, launch_ready=False)
    config.update(recipe_template_fields(
        stage="dense_selected_recipe_20tpp",
        dependency=f"immutable resolved {tier} dense top1/top2 5TPP comparison artifact",
    ))
    config.update(
        hpo_stage="dense_selected_recipe_20tpp",
        candidate_scope="baseline_only_no_block_fht",
        ladder_role="selected_recipe",
        ladder_slot="selected_from_5tpp_comparison",
        launch_block_reason="BLOCKED_ON_5TPP_COMPARISON_ARTIFACT",
    )
    config.update(five_tpp_comparison_fields(tier, "baseline"))
    return config


def make_fullattn_template(
    name: str,
    tier: str,
    tpp: float,
    *,
    main_lr_multiplier: float | None,
    stage: str,
    role: str,
    slot: str,
    screen_only: bool,
) -> dict:
    config = make_config(name, tier, tpp, launch_ready=False)
    dense_name = f"y400_mai_v2_{tier}_muon_20tpp_selected"
    config.update(recipe_template_fields(
        stage=stage,
        dependency=f"accepted {dense_name} recipe and immutable candidate selection artifact",
    ))
    config.update(
        method="block_fht",
        matched_dense_baseline=dense_name,
        inherit_dense_main_learning_rate_from=dense_name,
        inherit_dense_fallback_scale_from=dense_name,
        candidate_main_lr_multiplier=main_lr_multiplier,
        candidate_learning_rate_resolution=(
            "candidate learning_rate = candidate_main_lr_multiplier * accepted dense main learning rate"
            if main_lr_multiplier is not None
            else "fill with the selected stable full-attention 0.5TPP/5TPP recipe"
        ),
        # Always materialize the whole method specification.  Registered v2
        # candidates may not inherit model/training behavior from CLI defaults.
        **json.loads(json.dumps(REGISTERED_V2_BLOCK_FHT_METHOD_SPEC)),
        candidate_scope="full_attention_only_qk_headwise_v_cproj_no_mlp_targets",
        hpo_stage=stage,
        ladder_role=role,
        ladder_slot=slot,
        screen_only=screen_only,
        screen_only_resolution=(
            "rank stable candidates to populate ordered top1/top2 5TPP slots" if screen_only else None
        ),
        dense_fit_gate_required=True,
        dense_fit_gate="BLOCKED_ON_DENSE_SCALING_FIT",
        dense_fit_artifact=None,
        dense_fit_artifact_sha256=None,
        dense_fit_coefficients=None,
        launch_block_reason="BLOCKED_ON_DENSE_SCALING_FIT",
    )
    if tpp == 5.0:
        config.update(zero_point_five_tpp_ranking_fields(tier, "block_fht", slot))
    if tpp == 20.0:
        config.update(five_tpp_comparison_fields(tier, "block_fht"))
    return config


def write_config(name: str, config: dict) -> None:
    (CONFIG_DIR / f"{name}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def clean_previous_generated_configs() -> None:
    """Refresh only v2 templates; never touch legacy/current resolved configs."""
    for config_path in CONFIG_DIR.glob("y400_mai_v2_*.json"):
        if "_resolved" in config_path.name:
            continue
        config_path.unlink()


def validate_generated_artifacts(rows: list[dict[str, object]]) -> None:
    """Fail generation if the prospective v2 ladder or hard ordering is malformed."""
    if len(rows) != 48:
        raise AssertionError(f"expected 48 queue rows, got {len(rows)}")

    first_candidate = next(index for index, row in enumerate(rows) if row["method"] == "block_fht")
    if any(row["method"] != "baseline" for row in rows[:first_candidate]):
        raise AssertionError("only dense baseline stages may precede full-attention candidates")

    expected_eval_spec = eval_index_spec_sha256(REGISTERED_EVAL_BATCH_SIZE, REGISTERED_EVAL_ITERS)
    expected_eval_tokens_per_split = REGISTERED_EVAL_BATCH_SIZE * 1024 * REGISTERED_EVAL_ITERS
    configs: dict[str, dict] = {}
    for row in rows:
        name = str(row["name"])
        tier = str(row["tier"])
        method = str(row["method"])
        config = json.loads((CONFIG_DIR / f"{name}.json").read_text())
        configs[name] = config
        if config["model_tier"] != tier or config["method"] != method:
            raise AssertionError(f"{name} queue/config identity mismatch")
        if config["launch_ready"] != (row["launch_ready"] == "true"):
            raise AssertionError(f"{name} queue/config launch gate mismatch")
        if (
            config["mai_ladder_policy_version"] != POLICY_VERSION
            or config["ladder_role"] != row["ladder_role"]
            or config["ladder_slot"] != row["ladder_slot"]
            or config["practical_equivalence_nll"] != PRACTICAL_EQUIVALENCE_NLL
            or config["muon_adamw_lr_scale"] != FIXED_ADAMW_LR_SCALE
            or "fallback_scale_screen" in config["hpo_stage"]
            or config["compile"] is not False
            or config["eval_batch_size"] != REGISTERED_EVAL_BATCH_SIZE
            or config["eval_iters"] != REGISTERED_EVAL_ITERS
            or config["eval_tokens_per_split"] != expected_eval_tokens_per_split
            or config["eval_total_tokens"] != 2 * expected_eval_tokens_per_split
            or config["fixed_eval_index_spec_sha256"] != expected_eval_spec
            or config["eval_protocol_id"] != REGISTERED_EVAL_PROTOCOL
            or config["checkpoint_wall_clock_seconds"] != 7200
            or config["data_manifest_sha256"] != PROVENANCE["data_manifest_sha256"]
        ):
            raise AssertionError(f"{name} misses the registered v2 settings")
        if not config["registered_resume_determinism_required"]:
            raise AssertionError(f"{name} must require deterministic registered resume")
        if config["n_embd"] // config["n_head"] != 64:
            raise AssertionError(f"{name} violates head dimension 64")
        expected_active = EXPECTED_MATERIALIZED_PARAM_COUNTS[tier]
        if config["estimated_active_params"] != expected_active:
            raise AssertionError(f"{name} has the wrong materialized-parameter count")
        if (
            estimate_active_params(
                config["n_layer"], config["n_embd"], config["vocab_size"], config["block_size"]
            )
            != expected_active
        ):
            raise AssertionError(f"{name} uses the wrong materialized-parameter formula")
        if (
            config.get("bias") is not False
            or config.get("dropout") != 0.0
            or config.get("tie_word_embeddings") is not True
        ):
            raise AssertionError(f"{name} must explicitly use no biases, zero dropout, and tied embeddings")
        expected_schedule = EXPECTED_TPP_SCHEDULES[tier][config["planned_tpp"]]
        if (config["planned_tokens"], config["max_iters"], config["scheduled_tokens"]) != expected_schedule:
            raise AssertionError(f"{name} has an incorrect regenerated TPP schedule")
        if config["scheduled_tpp"] != config["scheduled_tokens"] / expected_active:
            raise AssertionError(f"{name} has an incorrect scheduled TPP")
        if row["launch_ready"] == "true":
            if config.get("recipe_resolution_required") is True:
                raise AssertionError(f"{name} unexpectedly requires recipe resolution")
            if config.get("learning_rate") is None or config.get("min_lr") is None:
                raise AssertionError(f"{name} has unresolved optimizer fields")
        elif not config.get("recipe_resolution_required", False):
            raise AssertionError(f"{name} blocked template must require recipe resolution")
        if config["ladder_role"] == "screen_only":
            if config.get("screen_only") is not True or config["planned_tpp"] != 0.5:
                raise AssertionError(f"{name} is not a valid SCREEN_ONLY 0.5TPP row")
        if config["planned_tpp"] == 5.0 and config["ladder_role"] == "confirmation":
            if (
                config.get("zero_point_five_tpp_ranking_artifact_required") is not True
                or config.get("zero_point_five_tpp_ranking_artifact") is not None
                or config.get("zero_point_five_tpp_ranking_artifact_sha256") is not None
                or config.get("zero_point_five_tpp_ranking_artifact_schema")
                != RANKING_ARTIFACT_SCHEMA_VERSION
                or config.get("zero_point_five_tpp_ranking_tier") != tier
                or config.get("zero_point_five_tpp_ranking_method") != method
                or not config.get("zero_point_five_tpp_ranking_hpo_stage")
                or config.get("zero_point_five_tpp_ranking_slot") not in {"top1", "top2"}
                or config.get("mai_selection_candidate") is not None
                or config.get("mai_selection_recipe") is not None
            ):
                raise AssertionError(f"{name} must wait for a pinned 0.5TPP ranking artifact")
        if config["planned_tpp"] == 20.0:
            if (
                config.get("five_tpp_comparison_artifact_required") is not True
                or config.get("five_tpp_comparison_artifact") is not None
                or config.get("five_tpp_comparison_artifact_sha256") is not None
                or config.get("five_tpp_comparison_artifact_schema")
                != COMPARISON_ARTIFACT_SCHEMA_VERSION
                or config.get("five_tpp_comparison_tier") != tier
                or config.get("five_tpp_comparison_method") != method
                or not config.get("five_tpp_comparison_hpo_stage")
                or config.get("mai_selection_candidate") is not None
                or config.get("mai_selection_recipe") is not None
            ):
                raise AssertionError(f"{name} must wait for a pinned 5TPP comparison artifact")
        if method == "block_fht":
            if row["status"] != "BLOCKED_ON_DENSE_SCALING_FIT" or row["launch_ready"] != "false":
                raise AssertionError(f"{name} must remain blocked on the accepted dense scaling fit")
            if row["dependency"] != "accepted immutable four-rung dense scaling-fit artifact pinned by SHA-256":
                raise AssertionError(f"{name} has an incomplete dense-fit dependency")
            if (
                config.get("dense_fit_gate_required") is not True
                or config.get("dense_fit_gate") != "BLOCKED_ON_DENSE_SCALING_FIT"
                or config.get("dense_fit_artifact") is not None
                or config.get("dense_fit_artifact_sha256") is not None
                or config.get("dense_fit_coefficients") is not None
                or {
                    field: config.get(field)
                    for field in REGISTERED_V2_BLOCK_FHT_METHOD_SPEC
                }
                != REGISTERED_V2_BLOCK_FHT_METHOD_SPEC
            ):
                raise AssertionError(f"{name} has an incomplete executable dense-fit or method gate")

    for tier in TIERS:
        for method in ("baseline", "block_fht"):
            tier_configs = [
                config for name, config in configs.items()
                if config["model_tier"] == tier and config["method"] == method
            ]
            if len(tier_configs) != 6:
                raise AssertionError(f"{tier} {method} must have exactly six ladder rows")
            screens = [config for config in tier_configs if config["ladder_role"] == "screen_only"]
            confirmations = [config for config in tier_configs if config["ladder_role"] == "confirmation"]
            selected = [config for config in tier_configs if config["ladder_role"] == "selected_recipe"]
            if len(screens) != 3 or len(confirmations) != 2 or len(selected) != 1:
                raise AssertionError(f"{tier} {method} has malformed stage cardinalities")
            if sorted(config["ladder_slot"] for config in confirmations) != ["top1", "top2"]:
                raise AssertionError(f"{tier} {method} confirmation slots must be top1/top2")
            if selected[0]["ladder_slot"] != "selected_from_5tpp_comparison":
                raise AssertionError(f"{tier} {method} has no comparison-selected 20TPP row")
            if method == "baseline":
                if sorted(config["learning_rate"] for config in screens) != list(BASELINE_SCREEN_LRS):
                    raise AssertionError(f"{tier} dense screen LRs are wrong")
            elif sorted(config["candidate_main_lr_multiplier"] for config in screens) != list(
                CANDIDATE_MAIN_LR_MULTIPLIERS
            ):
                raise AssertionError(f"{tier} BlockFHT screen multipliers are wrong")


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    clean_previous_generated_configs()
    rows: list[dict[str, object]] = []
    queue_columns = (
        "name",
        "policy_version",
        "phase",
        "status",
        "tier",
        "method",
        "ladder_role",
        "ladder_slot",
        "planned_tokens",
        "planned_tpp",
        "launch_ready",
        "dependency",
        "selection_notes",
        "config",
        "reporting_note",
    )

    def add(name: str, phase: str, status: str, config: dict, dependency: str) -> None:
        write_config(name, config)
        rows.append({
            "name": name,
            "policy_version": POLICY_VERSION,
            "phase": phase,
            "status": status,
            "tier": config["model_tier"],
            "method": config["method"],
            "ladder_role": config["ladder_role"],
            "ladder_slot": config["ladder_slot"],
            "planned_tokens": config["planned_tokens"],
            "planned_tpp": f"{config['planned_tpp']:.6f}",
            "launch_ready": str(config["launch_ready"]).lower(),
            "dependency": dependency,
            "selection_notes": SELECTION_NOTE,
            "config": f"{REMOTE_CONFIG_DIR}/{name}.json",
            "reporting_note": REPORTING_NOTE,
        })

    # 0.5TPP is screen-only; its fixed AdamW fallback scale is 0.3 for every row.
    for tier in TIERS:
        for lr in BASELINE_SCREEN_LRS:
            name = f"y400_mai_v2_{tier}_muon_0p5tpp_{lr_slug(lr)}"
            add(
                name,
                "dense_screen_0p5tpp",
                "READY",
                make_dense_screen(name, tier, lr),
                "none; SCREEN_ONLY dense fixed-window terminal-NLL screen",
            )

    # Exactly two ordered confirmation slots are dynamically materialized from each
    # 0.5TPP screen's stable top-two ranking; they do not predeclare a recipe.
    for tier in TIERS:
        for slot in ("top1", "top2"):
            name = f"y400_mai_v2_{tier}_muon_5tpp_{slot}"
            add(
                name,
                "dense_recipe_confirmation_5tpp",
                "BLOCKED_ON_0P5TPP_TOP2_RESOLUTION",
                make_dense_confirmation_template(name, tier, slot),
                f"immutable {tier} 0.5TPP ranking artifact assigns this ordered {slot} slot",
            )

    for tier in TIERS:
        name = f"y400_mai_v2_{tier}_muon_20tpp_selected"
        add(
            name,
            "dense_selected_recipe_20tpp",
            "BLOCKED_ON_5TPP_COMPARISON_ARTIFACT",
            make_dense_20tpp_template(name, tier),
            f"immutable resolved {tier} 5TPP top1/top2 comparison artifact",
        )

    # Full-attention BlockFHT rows use the same screen/confirmation/selection
    # state machine, but every one remains blocked until the dense fit is pinned.
    all_dense_dependency = "accepted immutable four-rung dense scaling-fit artifact pinned by SHA-256"
    for tier in TIERS:
        for multiplier in CANDIDATE_MAIN_LR_MULTIPLIERS:
            name = f"y400_mai_v2_{tier}_fullattn_blockfht_0p5tpp_{multiplier_slug(multiplier)}"
            add(
                name,
                "full_attention_blockfht_screen_0p5tpp",
                "BLOCKED_ON_DENSE_SCALING_FIT",
                make_fullattn_template(
                    name,
                    tier,
                    0.5,
                    main_lr_multiplier=multiplier,
                    stage="full_attention_blockfht_screen_0p5tpp",
                    role="screen_only",
                    slot=multiplier_slug(multiplier),
                    screen_only=True,
                ),
                all_dense_dependency,
            )
        for slot in ("top1", "top2"):
            name = f"y400_mai_v2_{tier}_fullattn_blockfht_5tpp_{slot}"
            config = make_fullattn_template(
                name,
                tier,
                5.0,
                main_lr_multiplier=None,
                stage="full_attention_blockfht_confirmation_5tpp",
                role="confirmation",
                slot=slot,
                screen_only=False,
            )
            config.update(
                confirmation_slot=slot,
                confirmation_source=f"ordered stable 0.5TPP {slot} BlockFHT candidate",
            )
            add(
                name,
                "full_attention_blockfht_confirmation_5tpp",
                "BLOCKED_ON_DENSE_SCALING_FIT",
                config,
                all_dense_dependency,
            )
        name = f"y400_mai_v2_{tier}_fullattn_blockfht_20tpp_selected"
        add(
            name,
            "full_attention_blockfht_selected_recipe_20tpp",
            "BLOCKED_ON_DENSE_SCALING_FIT",
            make_fullattn_template(
                name,
                tier,
                20.0,
                main_lr_multiplier=None,
                stage="full_attention_blockfht_selected_recipe_20tpp",
                role="selected_recipe",
                slot="selected_from_5tpp_comparison",
                screen_only=False,
            ),
            all_dense_dependency,
        )

    QUEUE_PATH.write_text(
        "\t".join(queue_columns)
        + "\n"
        + "".join("\t".join(map(str, (row[column] for column in queue_columns))) + "\n" for row in rows)
    )
    validate_generated_artifacts(rows)
    print(f"wrote {len(rows)} MAI scaling-ladder configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
