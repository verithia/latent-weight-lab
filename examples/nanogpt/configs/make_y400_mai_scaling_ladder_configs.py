from __future__ import annotations

"""Generate the registered, baseline-first MAI Y400 scaling-ladder controls.

This is deliberately a config/queue generator only.  It never launches a run.
Recipe-dependent templates remain launch-disabled until their fixed-window
terminal-NLL selection dependency has been resolved and recorded.
"""

import hashlib
import json
import math
from pathlib import Path


DATA_DIR = "/root/userdata/MappingNetworks/data/finewebedu_20b"
OUT_DIR = "/root/userdata/MappingNetworks/outputs/y400_mai_scaling_ladder"
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
BASELINE_SCALE_CHOICES = (0.2, 0.3)
CANDIDATE_MAIN_LR_MULTIPLIERS = (0.5, 0.75, 1.0)
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
    "Select by terminal held-out NLL on fixed evaluation windows only; hard-reject "
    "numerical instability. Tie or preregistered-practical-equivalence results are "
    "all promoted to 5TPP confirmation rather than silently selecting one."
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
            "record the preregistered NLL band before launch; promote every tied/close "
            "recipe to 5TPP confirmation rather than silently choosing one"
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
        muon_adamw_lr_scale=0.3,
        hpo_stage="baseline_lr_screen_0p5tpp",
        candidate_scope="baseline_only_no_block_fht",
        selection_endpoint="terminal held-out NLL on fixed eval windows",
        instability_policy="hard reject NaN, Inf, divergence, or failed terminal evaluation",
        practical_equivalence_policy=(
            "record the preregistered NLL band before launch; promote every tied/close "
            "recipe to 5TPP confirmation rather than silently choosing one"
        ),
    )
    return config


def make_dense_scale_template(name: str, tier: str, scale: float) -> dict:
    config = make_config(name, tier, 0.5, launch_ready=False)
    config.update(recipe_template_fields(
        stage="baseline_fallback_scale_screen_0p5tpp",
        dependency=f"selected stable terminal-NLL LR from {tier} 0.5TPP baseline LR screen",
    ))
    config.update(
        muon_adamw_lr_scale=scale,
        hpo_stage="baseline_fallback_scale_screen_0p5tpp",
        candidate_scope="baseline_only_no_block_fht",
        launch_block_reason="BLOCKED_ON_LR_SELECTION",
    )
    return config


def make_dense_confirmation_template(name: str, tier: str) -> dict:
    config = make_config(name, tier, 5.0, launch_ready=False)
    config.update(recipe_template_fields(
        stage="baseline_recipe_confirmation_5tpp",
        dependency=f"selected or tied stable {tier} LR/fallback-scale recipe from 0.5TPP controls",
    ))
    config.update(
        muon_adamw_lr_scale=None,
        hpo_stage="baseline_recipe_confirmation_5tpp",
        candidate_scope="baseline_only_no_block_fht",
        confirmation_slots="promote one winner, or every recipe inside the preregistered practical-equivalence band",
        launch_block_reason="BLOCKED_ON_RECIPE_SELECTION",
    )
    return config


def make_dense_20tpp_template(name: str, tier: str) -> dict:
    config = make_config(name, tier, 20.0, launch_ready=False)
    config.update(recipe_template_fields(
        stage="registered_dense_baseline_20tpp",
        dependency=f"accepted stable terminal-NLL {tier} 5TPP baseline confirmation",
    ))
    config.update(
        muon_adamw_lr_scale=None,
        hpo_stage="registered_dense_baseline_20tpp",
        candidate_scope="baseline_only_no_block_fht",
        launch_block_reason="BLOCKED_ON_5TPP_BASELINE_CONFIRMATION",
    )
    return config


def make_fullattn_template(
    name: str,
    tier: str,
    tpp: float,
    *,
    main_lr_multiplier: float | None,
    stage: str,
) -> dict:
    config = make_config(name, tier, tpp, launch_ready=False)
    dense_name = f"y400_mai_{tier}_muon_20tpp_baseline"
    config.update(recipe_template_fields(
        stage=stage,
        dependency=f"accepted {dense_name} recipe and candidate 0.5TPP/5TPP selection chain",
    ))
    config.update(
        method="block_fht",
        muon_adamw_lr_scale=None,
        matched_dense_baseline=dense_name,
        inherit_dense_main_learning_rate_from=dense_name,
        inherit_dense_fallback_scale_from=dense_name,
        candidate_main_lr_multiplier=main_lr_multiplier,
        candidate_learning_rate_resolution=(
            "candidate learning_rate = candidate_main_lr_multiplier * accepted dense main learning rate"
            if main_lr_multiplier is not None
            else "fill with the selected stable full-attention 0.5TPP/5TPP recipe"
        ),
        block_fht_targets=FULL_ATTENTION_TARGETS,
        block_fht_layers=2,
        block_fht_latent_ratio=0.01,
        block_fht_match_gpt_init=True,
        block_fht_latent_init_std=0.02,
        block_fht_seed=1000,
        block_fht_cache_weights=True,
        candidate_scope="full_attention_only_qk_headwise_v_cproj_no_mlp_targets",
        dense_fit_gate_required=True,
        dense_fit_gate="BLOCKED_ON_DENSE_SCALING_FIT",
        dense_fit_artifact=None,
        dense_fit_artifact_sha256=None,
        dense_fit_coefficients=None,
        launch_block_reason="BLOCKED_ON_DENSE_SCALING_FIT",
    )
    return config


def write_config(name: str, config: dict) -> None:
    (CONFIG_DIR / f"{name}.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def clean_previous_generated_configs() -> None:
    """Refresh this generator's namespace without touching non-MAI configs."""
    for config_path in CONFIG_DIR.glob("y400_mai_*.json"):
        config_path.unlink()


def validate_generated_artifacts(rows: list[tuple]) -> None:
    """Fail generation if the registered ladder or hard ordering is malformed."""
    expected_configs = 48
    if len(rows) != expected_configs:
        raise AssertionError(f"expected {expected_configs} queue rows, got {len(rows)}")

    first_candidate = next(index for index, row in enumerate(rows) if row[4] == "block_fht")
    if any(row[4] != "baseline" for row in rows[:first_candidate]):
        raise AssertionError("only dense baseline stages may precede full-attention candidates")

    expected_eval_spec = eval_index_spec_sha256(REGISTERED_EVAL_BATCH_SIZE, REGISTERED_EVAL_ITERS)
    expected_eval_tokens_per_split = REGISTERED_EVAL_BATCH_SIZE * 1024 * REGISTERED_EVAL_ITERS
    for name, _, status, tier, method, _, _, launch_ready, dependency, *_ in rows:
        config_path = CONFIG_DIR / f"{name}.json"
        config = json.loads(config_path.read_text())
        if config["model_tier"] != tier or config["method"] != method:
            raise AssertionError(f"{name} queue/config identity mismatch")
        if config["launch_ready"] != (launch_ready == "true"):
            raise AssertionError(f"{name} queue/config launch gate mismatch")
        if (
            config["compile"] is not False
            or config["eval_batch_size"] != REGISTERED_EVAL_BATCH_SIZE
            or config["eval_iters"] != REGISTERED_EVAL_ITERS
            or config["eval_tokens_per_split"] != expected_eval_tokens_per_split
            or config["eval_total_tokens"] != 2 * expected_eval_tokens_per_split
            or config["fixed_eval_index_spec_sha256"] != expected_eval_spec
            or config["eval_protocol_id"] != REGISTERED_EVAL_PROTOCOL
            or config["checkpoint_wall_clock_seconds"] != 7200
            or config["data_manifest_sha256"] != PROVENANCE["data_manifest_sha256"]
        ):
            raise AssertionError(f"{name} misses the registered execution/evaluation settings")
        if not config["registered_resume_determinism_required"]:
            raise AssertionError(f"{name} must require deterministic registered resume")
        if config["n_embd"] // config["n_head"] != 64:
            raise AssertionError(f"{name} violates head dimension 64")
        expected_active = EXPECTED_MATERIALIZED_PARAM_COUNTS[tier]
        if (
            estimate_active_params(
                config["n_layer"], config["n_embd"], config["vocab_size"], config["block_size"]
            )
            != expected_active
        ):
            raise AssertionError(f"{name} uses the wrong materialized-parameter formula")
        if config["estimated_active_params"] != expected_active:
            raise AssertionError(f"{name} has the wrong materialized-parameter count")
        if (
            config.get("bias") is not False
            or config.get("dropout") != 0.0
            or config.get("tie_word_embeddings") is not True
        ):
            raise AssertionError(f"{name} must explicitly use no biases, zero dropout, and tied embeddings")
        expected_schedule = EXPECTED_TPP_SCHEDULES[tier][config["planned_tpp"]]
        actual_schedule = (config["planned_tokens"], config["max_iters"], config["scheduled_tokens"])
        if actual_schedule != expected_schedule:
            raise AssertionError(f"{name} has an incorrect regenerated TPP schedule")
        if config["scheduled_tpp"] != config["scheduled_tokens"] / expected_active:
            raise AssertionError(f"{name} has an incorrect scheduled TPP")
        if launch_ready == "true":
            if config.get("recipe_resolution_required") is True:
                raise AssertionError(f"{name} unexpectedly requires recipe resolution")
            unresolved = [
                field
                for field in ("learning_rate", "min_lr", "muon_adamw_lr_scale")
                if config.get(field) is None
            ]
            if unresolved:
                raise AssertionError(f"{name} has unresolved optimizer fields: {unresolved}")
        elif not config.get("recipe_resolution_required", False):
            raise AssertionError(f"{name} blocked template must require recipe resolution")
        if method == "block_fht":
            if status != "BLOCKED_ON_DENSE_SCALING_FIT" or launch_ready != "false":
                raise AssertionError(f"{name} must remain blocked on the accepted dense scaling fit")
            if dependency != "accepted immutable four-rung dense scaling-fit artifact pinned by SHA-256":
                raise AssertionError(f"{name} has an incomplete dense-fit dependency")
            if (
                config.get("dense_fit_gate_required") is not True
                or config.get("dense_fit_gate") != "BLOCKED_ON_DENSE_SCALING_FIT"
                or config.get("dense_fit_artifact") is not None
                or config.get("dense_fit_artifact_sha256") is not None
                or config.get("dense_fit_coefficients") is not None
            ):
                raise AssertionError(f"{name} has an incomplete executable dense-fit gate")
            if config["block_fht_targets"] != FULL_ATTENTION_TARGETS:
                raise AssertionError(f"{name} has the wrong full-attention target set")


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    clean_previous_generated_configs()
    rows = []

    def add(name: str, phase: str, status: str, config: dict, dependency: str) -> None:
        write_config(name, config)
        rows.append((
            name,
            phase,
            status,
            config["model_tier"],
            config["method"],
            config["planned_tokens"],
            f"{config['planned_tpp']:.6f}",
            str(config["launch_ready"]).lower(),
            dependency,
            SELECTION_NOTE,
            f"{REMOTE_CONFIG_DIR}/{name}.json",
            REPORTING_NOTE,
        ))

    # Queue all four rung LR screens first.  They are the only launch-ready rows.
    for tier in TIERS:
        for lr in BASELINE_SCREEN_LRS:
            name = f"y400_mai_{tier}_muon_0p5tpp_{lr_slug(lr)}"
            add(
                name,
                "dense_baseline_lr_screen_0p5tpp",
                "READY",
                make_dense_screen(name, tier, lr),
                "none; baseline-only fixed-window terminal-NLL screen",
            )

    # Once a rung's leading stable LR is known, compare the specified fallback scales.
    for tier in TIERS:
        for scale in BASELINE_SCALE_CHOICES:
            name = f"y400_mai_{tier}_muon_0p5tpp_scale{int(scale * 10):02d}"
            add(
                name,
                "dense_baseline_fallback_scale_screen_0p5tpp",
                "BLOCKED_ON_LR_SELECTION",
                make_dense_scale_template(name, tier, scale),
                f"selected stable {tier} LR from the 0.5TPP baseline screen",
            )

    for tier in TIERS:
        name = f"y400_mai_{tier}_muon_5tpp_baseline"
        add(
            name,
            "dense_baseline_recipe_confirmation_5tpp",
            "BLOCKED_ON_RECIPE_SELECTION",
            make_dense_confirmation_template(name, tier),
            f"selected or tied stable {tier} recipe from both 0.5TPP baseline controls",
        )

    for tier in TIERS:
        name = f"y400_mai_{tier}_muon_20tpp_baseline"
        add(
            name,
            "registered_dense_baseline_20tpp",
            "BLOCKED_ON_5TPP_BASELINE_CONFIRMATION",
            make_dense_20tpp_template(name, tier),
            f"accepted stable {tier} 5TPP dense baseline confirmation",
        )

    # Candidates remain disabled until a separately accepted immutable fit of all
    # four 20TPP dense terminal records is published and pinned in a resolved
    # launch config. Dense completion alone is deliberately not sufficient.
    all_dense_dependency = "accepted immutable four-rung dense scaling-fit artifact pinned by SHA-256"
    for tier in TIERS:
        for multiplier in CANDIDATE_MAIN_LR_MULTIPLIERS:
            name = f"y400_mai_{tier}_fullattn_blockfht_0p5tpp_{multiplier_slug(multiplier)}"
            add(
                name,
                "full_attention_blockfht_lr_screen_0p5tpp",
                "BLOCKED_ON_DENSE_SCALING_FIT",
                make_fullattn_template(
                    name,
                    tier,
                    0.5,
                    main_lr_multiplier=multiplier,
                    stage="full_attention_main_lr_screen_0p5tpp",
                ),
                all_dense_dependency,
            )
        name = f"y400_mai_{tier}_fullattn_blockfht_5tpp"
        add(
            name,
            "full_attention_blockfht_recipe_confirmation_5tpp",
            "BLOCKED_ON_DENSE_SCALING_FIT",
            make_fullattn_template(
                name,
                tier,
                5.0,
                main_lr_multiplier=None,
                stage="full_attention_recipe_confirmation_5tpp",
            ),
            all_dense_dependency,
        )
        name = f"y400_mai_{tier}_fullattn_blockfht_20tpp"
        add(
            name,
            "registered_full_attention_blockfht_20tpp",
            "BLOCKED_ON_DENSE_SCALING_FIT",
            make_fullattn_template(
                name,
                tier,
                20.0,
                main_lr_multiplier=None,
                stage="registered_full_attention_blockfht_20tpp",
            ),
            all_dense_dependency,
        )

    QUEUE_PATH.write_text(
        "name\tphase\tstatus\ttier\tmethod\tplanned_tokens\tplanned_tpp\tlaunch_ready\tdependency\tselection_notes\tconfig\treporting_note\n"
        + "".join("\t".join(map(str, row)) + "\n" for row in rows)
    )
    validate_generated_artifacts(rows)
    print(f"wrote {len(rows)} MAI scaling-ladder configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
