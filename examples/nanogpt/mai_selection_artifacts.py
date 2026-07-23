from __future__ import annotations

"""Immutable, hash-pinned selection artifacts for the registered MAI v2 ladder.

The command line intentionally creates only the two prospective selection
artifacts.  Terminal result records are produced by the registered execution
workflow and are validated here before they can influence a later rung.
"""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

try:  # Support both ``python -m`` and direct invocation of this utility.
    from examples.nanogpt.dense_scaling_fit import validate_dense_fit_artifact
except ModuleNotFoundError:  # pragma: no cover - exercised by direct invocation.
    from dense_scaling_fit import validate_dense_fit_artifact


POLICY_VERSION = "mai_ladder_selection_v2"
PRACTICAL_EQUIVALENCE_NLL = 0.02
REGISTERED_ADAMW_FALLBACK_SCALE = 0.3
TERMINAL_RESULT_SCHEMA_VERSION = "mai_terminal_result_v3"
CHECKPOINT_METADATA_SCHEMA_VERSION = "nanogpt_checkpoint_metadata_v2"
CHECKPOINT_SCHEMA_VERSION = "nanogpt_exact_resume_v2"
RANKING_ARTIFACT_SCHEMA_VERSION = "mai_zero_point_five_tpp_ranking_v3"
COMPARISON_ARTIFACT_SCHEMA_VERSION = "mai_five_tpp_comparison_v3"
RANKING_ARTIFACT_KIND = "mai_zero_point_five_tpp_ranking"
COMPARISON_ARTIFACT_KIND = "mai_five_tpp_top_two_comparison"
LEGACY_RUN_CONTRACT_SCHEMA_VERSION = "mai_registered_run_contract_v1"
RUN_CONTRACT_SCHEMA_VERSION = "mai_registered_run_contract_v2"
RUN_CONTRACT_FIELDS = (
    "model_seed",
    "train_data_seed",
    "n_layer",
    "n_embd",
    "n_head",
    "vocab_size",
    "block_size",
    "bias",
    "dropout",
    "tie_word_embeddings",
    "batch_size",
    "gradient_accumulation_steps",
    "dtype",
    "compile",
    "optimizer",
    "weight_decay",
    "beta1",
    "beta2",
    "muon_momentum",
    "muon_ns_steps",
)
# These are the complete BlockFHT controls that can alter construction or
# training of a registered v2 full-attention candidate.  Keep them separate
# from selection recipes: a recipe selects an optimizer setting, while these
# fields define the candidate method whose terminal result is eligible for
# ranking.  Generic model and optimizer controls remain in RUN_CONTRACT_FIELDS
# and selection_recipe respectively.
BLOCK_FHT_STRUCTURE_FIELDS = (
    "block_fht_targets",
    "block_fht_layers",
    "block_fht_latent_ratio",
    "block_fht_latent_ratios",
    "block_fht_match_gpt_init",
    "block_fht_latent_init_std",
    "block_fht_seed",
    "block_fht_modulation_alpha",
    "block_fht_modulation_centered",
    "block_fht_weight_scale",
    "block_fht_residual_base_scale",
    "block_fht_output_gain_targets",
    "block_fht_input_gain_targets",
    "block_fht_ffn_pregelu_gain",
    "block_fht_ffn_pregelu_bias",
    "block_fht_ffn_pregelu_bias_init",
    "block_fht_ffn_lowrank_rank",
    "block_fht_ffn_lowrank_scale",
    "block_fht_ffn_lowrank_init_std",
    "block_fht_ffn_spectral_rank",
    "block_fht_ffn_spectral_out_groups",
    "block_fht_ffn_spectral_in_groups",
    "block_fht_cproj_lowrank_rank",
    "block_fht_cproj_lowrank_scale",
    "block_fht_cproj_lowrank_init_std",
    "block_fht_cproj_lowrank_mode",
    "block_fht_cproj_lowrank_latent_ratio",
    "block_fht_cproj_lowrank_b_zero_init",
    "block_fht_cproj_lowrank_bias",
    "block_fht_cproj_tied_cfc_skip",
    "block_fht_cproj_tied_cfc_scale_init",
    "block_fht_cproj_tied_cfc_vector",
    "block_fht_cproj_quarter_diag",
    "block_fht_cproj_quarter_diag_scale_init",
    "block_fht_cproj_quarter_diag_init_std",
    "block_fht_cproj_spectral_resid_rank",
    "block_fht_cproj_spectral_resid_scale_init",
    "block_fht_cproj_spectral_resid_seed",
    "block_fht_ffn_postgelu_std_target",
    "block_fht_ffn_postgelu_std_lambda",
    "block_fht_cache_weights",
    "freeze_non_block_fht",
    "train_embeddings_when_frozen",
    "block_fht_latent_grad_normalize",
    "block_fht_latent_grad_target_rms",
    "mapping_stability_lambda",
    "mapping_stability_sigma",
    "mapping_stability_temperature",
    "mapping_norm_lambda",
    "mapping_norm_target_rms",
    "grad_clip",
)
# The full-attention v2 templates must emit this entire specification rather
# than relying on argparse defaults.  Terminal result contracts carry the same
# fields (and rank/compare require exact shared identity), so a selected launch
# cannot acquire a different control through a CLI flag or omitted config key.
REGISTERED_V2_BLOCK_FHT_METHOD_SPEC: dict[str, Any] = {
    "block_fht_targets": ["attn.c_attn.qk_headwise", "attn.c_attn.v", "attn.c_proj"],
    "block_fht_layers": 2,
    "block_fht_latent_ratio": 0.01,
    "block_fht_latent_ratios": None,
    "block_fht_match_gpt_init": True,
    "block_fht_latent_init_std": 0.02,
    "block_fht_seed": 1000,
    "block_fht_modulation_alpha": 0.0,
    "block_fht_modulation_centered": False,
    "block_fht_weight_scale": None,
    "block_fht_residual_base_scale": 0.0,
    "block_fht_output_gain_targets": [],
    "block_fht_input_gain_targets": [],
    "block_fht_ffn_pregelu_gain": False,
    "block_fht_ffn_pregelu_bias": False,
    "block_fht_ffn_pregelu_bias_init": 0.0,
    "block_fht_ffn_lowrank_rank": 0,
    "block_fht_ffn_lowrank_scale": 1.0,
    "block_fht_ffn_lowrank_init_std": 0.02,
    "block_fht_ffn_spectral_rank": 0,
    "block_fht_ffn_spectral_out_groups": 1,
    "block_fht_ffn_spectral_in_groups": 1,
    "block_fht_cproj_lowrank_rank": 0,
    "block_fht_cproj_lowrank_scale": 1.0,
    "block_fht_cproj_lowrank_init_std": 0.02,
    "block_fht_cproj_lowrank_mode": "dense",
    "block_fht_cproj_lowrank_latent_ratio": None,
    "block_fht_cproj_lowrank_b_zero_init": True,
    "block_fht_cproj_lowrank_bias": False,
    "block_fht_cproj_tied_cfc_skip": False,
    "block_fht_cproj_tied_cfc_scale_init": 0.0,
    "block_fht_cproj_tied_cfc_vector": True,
    "block_fht_cproj_quarter_diag": False,
    "block_fht_cproj_quarter_diag_scale_init": 0.0,
    "block_fht_cproj_quarter_diag_init_std": 0.02,
    "block_fht_cproj_spectral_resid_rank": 0,
    "block_fht_cproj_spectral_resid_scale_init": 0.0,
    "block_fht_cproj_spectral_resid_seed": 0,
    "block_fht_ffn_postgelu_std_target": 0.0,
    "block_fht_ffn_postgelu_std_lambda": 0.0,
    "block_fht_cache_weights": True,
    "freeze_non_block_fht": False,
    "train_embeddings_when_frozen": False,
    "block_fht_latent_grad_normalize": False,
    "block_fht_latent_grad_target_rms": 0.01,
    "mapping_stability_lambda": 0.0,
    "mapping_stability_sigma": 1e-3,
    "mapping_stability_temperature": 1.0,
    "mapping_norm_lambda": 0.0,
    "mapping_norm_target_rms": 0.03,
    "grad_clip": 1.0,
}
if set(REGISTERED_V2_BLOCK_FHT_METHOD_SPEC) != set(BLOCK_FHT_STRUCTURE_FIELDS):
    raise RuntimeError("registered v2 BlockFHT method specification is incomplete")
SHARED_IDENTITY_FIELDS = (
    "source_hashes",
    "data_manifest_sha256",
    "fixed_eval_indices_sha256",
    "eval_protocol_id",
    "run_contract",
)
REQUIRED_RECIPE_FIELDS = ("learning_rate", "min_lr", "muon_adamw_lr_scale")
REGISTERED_V2_OPTIMIZER_SETTINGS = {
    "optimizer": "muon",
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "muon_momentum": 0.95,
    "muon_ns_steps": 5,
    "muon_adamw_lr_scale": REGISTERED_ADAMW_FALLBACK_SCALE,
}
REGISTERED_V2_DETERMINISM_SETTINGS = {
    "registered_resume_determinism_required": True,
    "checkpoint_history": False,
    "save_checkpoint": True,
    "checkpoint_wall_clock_seconds": 7200,
}
# ``block_size`` is already part of RUN_CONTRACT_FIELDS. Together with the
# data directory and manifest identity, these fields are every configurable
# input to the ordered fixed-index construction.
FIXED_EVALUATION_CONFIG_FIELDS = (
    "data_dir",
    "fixed_eval_indices",
    "eval_protocol_id",
    "fixed_eval_indices_protocol",
    "eval_seed",
    "eval_batch_size",
    "eval_iters",
    "fixed_eval_index_spec_sha256",
)
REGISTERED_SCREEN_CANDIDATES = {
    ("baseline", "dense_screen_0p5tpp"): {
        "field": "learning_rate",
        "values": (0.0016, 0.0020, 0.0024),
    },
    ("baseline", "dense_recipe_screen_0p5tpp"): {
        "field": "learning_rate",
        "values": (0.0016, 0.0020, 0.0024),
    },
    ("block_fht", "full_attention_blockfht_screen_0p5tpp"): {
        "field": "candidate_main_lr_multiplier",
        "values": (0.5, 0.75, 1.0),
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be an exact SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be an exact SHA-256 hex string") from exc
    return value.lower()


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return number


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _exact_tpp(value: object, expected: float, name: str) -> float:
    number = _finite_number(value, name, positive=True)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must be exactly {expected:g} TPP")
    return number


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _string_list(value: object, name: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{name} must be" + (" a non-empty" if nonempty else " a") + " string list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be" + (" a non-empty" if nonempty else " a") + " string list")
    return list(value)


def _nonnegative_integer(value: object, name: str) -> int:
    number = _integer(value, name)
    if number < 0:
        raise ValueError(f"{name} must be >= 0")
    return number


def _nonnegative_number(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return number


def _optional_positive_number(value: object, name: str) -> float | None:
    return None if value is None else _finite_number(value, name, positive=True)


def _validate_v2_optimizer_config(config: dict[str, Any]) -> None:
    """Validate the concrete optimizer recipe before terminal acceptance.

    This mirrors registered recipe constraints without importing ``train.py``
    (and therefore without importing torch). A MAI terminal record must reflect
    a config that could have launched, not a fabricated JSON template.
    """
    optimizer = config.get("optimizer")
    if optimizer != REGISTERED_V2_OPTIMIZER_SETTINGS["optimizer"]:
        raise ValueError("MAI v2 launch config optimizer must be the registered muon setting")
    required = ["learning_rate", "min_lr", "weight_decay", "beta1", "beta2", "muon_adamw_lr_scale"]
    required.extend(("muon_momentum", "muon_ns_steps"))
    missing = [field for field in required if field not in config or config[field] is None]
    if missing:
        raise ValueError("terminal result launch config is missing optimizer field " + ", ".join(missing))

    learning_rate = _finite_number(config["learning_rate"], "config.learning_rate", positive=True)
    min_lr = _finite_number(config["min_lr"], "config.min_lr", positive=True)
    if not math.isclose(min_lr, 0.1 * learning_rate, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("terminal result launch config min_lr must equal 0.1 * learning_rate")
    _nonnegative_number(config["weight_decay"], "config.weight_decay")
    for field in ("beta1", "beta2"):
        value = _finite_number(config[field], f"config.{field}", positive=True)
        if value >= 1.0:
            raise ValueError(f"config.{field} must be < 1")
    fallback_scale = _finite_number(
        config["muon_adamw_lr_scale"], "config.muon_adamw_lr_scale", positive=True
    )
    if fallback_scale != REGISTERED_ADAMW_FALLBACK_SCALE:
        raise ValueError("MAI v2 launch config must retain muon_adamw_lr_scale=0.3")
    momentum = _finite_number(config["muon_momentum"], "config.muon_momentum", positive=True)
    if momentum >= 1.0:
        raise ValueError("config.muon_momentum must be < 1")
    _integer(config["muon_ns_steps"], "config.muon_ns_steps", positive=True)
    for field, expected in REGISTERED_V2_OPTIMIZER_SETTINGS.items():
        if config[field] != expected:
            raise ValueError(f"MAI v2 launch config must retain registered optimizer setting {field}")


def _validate_block_fht_dense_fit_gate(config: dict[str, Any]) -> None:
    """Require a real accepted dense-fit pin for a BlockFHT terminal result."""
    if config.get("dense_fit_gate_required") is not True:
        raise ValueError("BlockFHT terminal result launch config requires dense_fit_gate_required=true")
    artifact_name = config.get("dense_fit_artifact")
    if not isinstance(artifact_name, str) or not artifact_name:
        raise ValueError("BlockFHT terminal result launch config is missing immutable dense-fit artifact")
    expected_hash = _require_sha256(
        config.get("dense_fit_artifact_sha256"), "config.dense_fit_artifact_sha256"
    )
    artifact_path = Path(artifact_name)
    if not artifact_path.is_file():
        raise ValueError("BlockFHT terminal result dense-fit artifact is missing")
    if sha256_file(artifact_path) != expected_hash:
        raise ValueError("BlockFHT terminal result dense-fit artifact hash mismatch")
    artifact = _load_json_object(artifact_path, "BlockFHT terminal result dense-fit artifact")
    coefficients = validate_dense_fit_artifact(artifact)
    if config.get("dense_fit_coefficients") != coefficients:
        raise ValueError("BlockFHT terminal result dense-fit coefficients are not exactly pinned")


def _validate_latent_ratios(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("run_contract.block_fht_latent_ratios must be null or a string-to-positive-number object")
    normalized = {}
    for target, ratio in value.items():
        if not isinstance(target, str) or not target:
            raise ValueError("run_contract.block_fht_latent_ratios keys must be non-empty strings")
        normalized[target] = _finite_number(
            ratio, f"run_contract.block_fht_latent_ratios[{target!r}]", positive=True
        )
    return dict(sorted(normalized.items()))


def _validate_block_fht_structure(structure: object) -> dict[str, Any]:
    if not isinstance(structure, dict) or set(structure) != set(BLOCK_FHT_STRUCTURE_FIELDS):
        raise ValueError("run_contract BlockFHT structure has incompatible fields")
    boolean_fields = (
        "block_fht_match_gpt_init",
        "block_fht_modulation_centered",
        "block_fht_ffn_pregelu_gain",
        "block_fht_ffn_pregelu_bias",
        "block_fht_cproj_lowrank_b_zero_init",
        "block_fht_cproj_lowrank_bias",
        "block_fht_cproj_tied_cfc_skip",
        "block_fht_cproj_tied_cfc_vector",
        "block_fht_cproj_quarter_diag",
        "block_fht_cache_weights",
        "freeze_non_block_fht",
        "train_embeddings_when_frozen",
        "block_fht_latent_grad_normalize",
    )
    for field in boolean_fields:
        if not isinstance(structure[field], bool):
            raise ValueError(f"run_contract.{field} must be a boolean")

    normalized: dict[str, Any] = {
        "block_fht_targets": _string_list(
            structure["block_fht_targets"], "run_contract.block_fht_targets", nonempty=True
        ),
        "block_fht_latent_ratios": _validate_latent_ratios(structure["block_fht_latent_ratios"]),
        "block_fht_layers": _integer(
            structure["block_fht_layers"], "run_contract.block_fht_layers", positive=True
        ),
        "block_fht_latent_ratio": _finite_number(
            structure["block_fht_latent_ratio"], "run_contract.block_fht_latent_ratio", positive=True
        ),
        "block_fht_latent_init_std": _finite_number(
            structure["block_fht_latent_init_std"], "run_contract.block_fht_latent_init_std", positive=True
        ),
        "block_fht_seed": _nonnegative_integer(
            structure["block_fht_seed"], "run_contract.block_fht_seed"
        ),
        "block_fht_modulation_alpha": _finite_number(
            structure["block_fht_modulation_alpha"], "run_contract.block_fht_modulation_alpha"
        ),
        "block_fht_weight_scale": _optional_positive_number(
            structure["block_fht_weight_scale"], "run_contract.block_fht_weight_scale"
        ),
        "block_fht_residual_base_scale": _nonnegative_number(
            structure["block_fht_residual_base_scale"], "run_contract.block_fht_residual_base_scale"
        ),
        "block_fht_output_gain_targets": _string_list(
            structure["block_fht_output_gain_targets"], "run_contract.block_fht_output_gain_targets"
        ),
        "block_fht_input_gain_targets": _string_list(
            structure["block_fht_input_gain_targets"], "run_contract.block_fht_input_gain_targets"
        ),
        "block_fht_ffn_pregelu_bias_init": _finite_number(
            structure["block_fht_ffn_pregelu_bias_init"], "run_contract.block_fht_ffn_pregelu_bias_init"
        ),
        "block_fht_ffn_lowrank_rank": _nonnegative_integer(
            structure["block_fht_ffn_lowrank_rank"], "run_contract.block_fht_ffn_lowrank_rank"
        ),
        "block_fht_ffn_lowrank_scale": _finite_number(
            structure["block_fht_ffn_lowrank_scale"], "run_contract.block_fht_ffn_lowrank_scale", positive=True
        ),
        "block_fht_ffn_lowrank_init_std": _finite_number(
            structure["block_fht_ffn_lowrank_init_std"], "run_contract.block_fht_ffn_lowrank_init_std", positive=True
        ),
        "block_fht_ffn_spectral_rank": _nonnegative_integer(
            structure["block_fht_ffn_spectral_rank"], "run_contract.block_fht_ffn_spectral_rank"
        ),
        "block_fht_ffn_spectral_out_groups": _integer(
            structure["block_fht_ffn_spectral_out_groups"],
            "run_contract.block_fht_ffn_spectral_out_groups",
            positive=True,
        ),
        "block_fht_ffn_spectral_in_groups": _integer(
            structure["block_fht_ffn_spectral_in_groups"],
            "run_contract.block_fht_ffn_spectral_in_groups",
            positive=True,
        ),
        "block_fht_cproj_lowrank_rank": _nonnegative_integer(
            structure["block_fht_cproj_lowrank_rank"], "run_contract.block_fht_cproj_lowrank_rank"
        ),
        "block_fht_cproj_lowrank_scale": _finite_number(
            structure["block_fht_cproj_lowrank_scale"], "run_contract.block_fht_cproj_lowrank_scale", positive=True
        ),
        "block_fht_cproj_lowrank_init_std": _finite_number(
            structure["block_fht_cproj_lowrank_init_std"],
            "run_contract.block_fht_cproj_lowrank_init_std",
            positive=True,
        ),
        "block_fht_cproj_lowrank_latent_ratio": _optional_positive_number(
            structure["block_fht_cproj_lowrank_latent_ratio"],
            "run_contract.block_fht_cproj_lowrank_latent_ratio",
        ),
        "block_fht_cproj_tied_cfc_scale_init": _finite_number(
            structure["block_fht_cproj_tied_cfc_scale_init"],
            "run_contract.block_fht_cproj_tied_cfc_scale_init",
        ),
        "block_fht_cproj_quarter_diag_scale_init": _finite_number(
            structure["block_fht_cproj_quarter_diag_scale_init"],
            "run_contract.block_fht_cproj_quarter_diag_scale_init",
        ),
        "block_fht_cproj_quarter_diag_init_std": _finite_number(
            structure["block_fht_cproj_quarter_diag_init_std"],
            "run_contract.block_fht_cproj_quarter_diag_init_std",
            positive=True,
        ),
        "block_fht_cproj_spectral_resid_rank": _nonnegative_integer(
            structure["block_fht_cproj_spectral_resid_rank"],
            "run_contract.block_fht_cproj_spectral_resid_rank",
        ),
        "block_fht_cproj_spectral_resid_scale_init": _finite_number(
            structure["block_fht_cproj_spectral_resid_scale_init"],
            "run_contract.block_fht_cproj_spectral_resid_scale_init",
        ),
        "block_fht_cproj_spectral_resid_seed": _nonnegative_integer(
            structure["block_fht_cproj_spectral_resid_seed"],
            "run_contract.block_fht_cproj_spectral_resid_seed",
        ),
        "block_fht_ffn_postgelu_std_target": _nonnegative_number(
            structure["block_fht_ffn_postgelu_std_target"],
            "run_contract.block_fht_ffn_postgelu_std_target",
        ),
        "block_fht_ffn_postgelu_std_lambda": _nonnegative_number(
            structure["block_fht_ffn_postgelu_std_lambda"],
            "run_contract.block_fht_ffn_postgelu_std_lambda",
        ),
        "block_fht_latent_grad_target_rms": _finite_number(
            structure["block_fht_latent_grad_target_rms"],
            "run_contract.block_fht_latent_grad_target_rms",
            positive=True,
        ),
        "mapping_stability_lambda": _nonnegative_number(
            structure["mapping_stability_lambda"], "run_contract.mapping_stability_lambda"
        ),
        "mapping_stability_sigma": _finite_number(
            structure["mapping_stability_sigma"], "run_contract.mapping_stability_sigma", positive=True
        ),
        "mapping_stability_temperature": _finite_number(
            structure["mapping_stability_temperature"],
            "run_contract.mapping_stability_temperature",
            positive=True,
        ),
        "mapping_norm_lambda": _nonnegative_number(
            structure["mapping_norm_lambda"], "run_contract.mapping_norm_lambda"
        ),
        "mapping_norm_target_rms": _finite_number(
            structure["mapping_norm_target_rms"], "run_contract.mapping_norm_target_rms", positive=True
        ),
        "grad_clip": _nonnegative_number(structure["grad_clip"], "run_contract.grad_clip"),
    }
    if structure["block_fht_cproj_lowrank_mode"] not in {"dense", "block_fht"}:
        raise ValueError("run_contract.block_fht_cproj_lowrank_mode is unregistered")
    normalized["block_fht_cproj_lowrank_mode"] = structure["block_fht_cproj_lowrank_mode"]
    for field in boolean_fields:
        normalized[field] = structure[field]
    return {field: normalized[field] for field in BLOCK_FHT_STRUCTURE_FIELDS}


def _validate_run_contract(contract: object, *, method: str) -> dict[str, Any]:
    """Validate the registered launch shape carried by all terminal records."""
    if not isinstance(contract, dict):
        raise ValueError("terminal result run_contract is required")
    schema_version = contract.get("schema_version")
    if schema_version == LEGACY_RUN_CONTRACT_SCHEMA_VERSION:
        if method != "baseline":
            raise ValueError("BlockFHT terminal results require the v2 method-specific run_contract")
        required = {"schema_version", *RUN_CONTRACT_FIELDS}
    elif schema_version == RUN_CONTRACT_SCHEMA_VERSION:
        required = {"schema_version", "method_structure", *RUN_CONTRACT_FIELDS}
    else:
        raise ValueError("terminal result run_contract schema/version is incompatible")
    if set(contract) != required:
        raise ValueError("terminal result run_contract has incompatible fields")

    normalized: dict[str, Any] = {"schema_version": schema_version}
    for field in (
        "model_seed",
        "train_data_seed",
        "n_layer",
        "n_embd",
        "n_head",
        "vocab_size",
        "block_size",
        "batch_size",
        "gradient_accumulation_steps",
        "muon_ns_steps",
    ):
        normalized[field] = _integer(contract[field], f"run_contract.{field}", positive=field not in {
            "model_seed", "train_data_seed"
        })
    for field in ("bias", "tie_word_embeddings", "compile"):
        if not isinstance(contract[field], bool):
            raise ValueError(f"run_contract.{field} must be a boolean")
        normalized[field] = contract[field]
    if contract["dtype"] not in {"float32", "bfloat16", "float16"}:
        raise ValueError("run_contract.dtype is unregistered")
    normalized["dtype"] = contract["dtype"]
    if contract["optimizer"] not in {"adamw", "muon"}:
        raise ValueError("run_contract.optimizer is unregistered")
    normalized["optimizer"] = contract["optimizer"]
    normalized["dropout"] = _finite_number(contract["dropout"], "run_contract.dropout")
    if normalized["dropout"] < 0.0:
        raise ValueError("run_contract.dropout must be >= 0")
    normalized["weight_decay"] = _finite_number(
        contract["weight_decay"], "run_contract.weight_decay"
    )
    if normalized["weight_decay"] < 0.0:
        raise ValueError("run_contract.weight_decay must be >= 0")
    for field in ("beta1", "beta2", "muon_momentum"):
        normalized[field] = _finite_number(contract[field], f"run_contract.{field}", positive=True)
        if normalized[field] >= 1.0:
            raise ValueError(f"run_contract.{field} must be < 1")
    if schema_version == RUN_CONTRACT_SCHEMA_VERSION:
        method_structure = contract["method_structure"]
        if not isinstance(method_structure, dict) or method_structure.get("method") != method:
            raise ValueError("run_contract method-specific structure disagrees with terminal result method")
        if method == "baseline":
            if method_structure != {"method": "baseline"}:
                raise ValueError("run_contract baseline method-specific structure is incompatible")
            normalized["method_structure"] = {"method": "baseline"}
        else:
            if set(method_structure) != {"method", "block_fht"}:
                raise ValueError("run_contract BlockFHT method-specific structure is incompatible")
            normalized["method_structure"] = {
                "method": "block_fht",
                "block_fht": _validate_block_fht_structure(method_structure["block_fht"]),
            }
    return normalized


def _validate_identity(identity: object, *, method: str) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError("terminal result identity is required")
    required = ("config_sha256", *SHARED_IDENTITY_FIELDS)
    missing = [field for field in required if field not in identity]
    if missing:
        raise ValueError("terminal result identity is missing " + ", ".join(missing))
    normalized = {
        "config_sha256": _require_sha256(identity["config_sha256"], "identity.config_sha256"),
        "data_manifest_sha256": _require_sha256(
            identity["data_manifest_sha256"], "identity.data_manifest_sha256"
        ),
        "fixed_eval_indices_sha256": _require_sha256(
            identity["fixed_eval_indices_sha256"], "identity.fixed_eval_indices_sha256"
        ),
    }
    source_hashes = identity["source_hashes"]
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("identity.source_hashes must be a non-empty object")
    normalized_sources = {}
    for source_name, source_hash in source_hashes.items():
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("identity.source_hashes keys must be non-empty strings")
        normalized_sources[source_name] = _require_sha256(
            source_hash, f"identity.source_hashes[{source_name!r}]"
        )
    normalized["source_hashes"] = dict(sorted(normalized_sources.items()))
    protocol = identity["eval_protocol_id"]
    if not isinstance(protocol, str) or not protocol:
        raise ValueError("identity.eval_protocol_id must be a non-empty string")
    normalized["eval_protocol_id"] = protocol
    normalized["run_contract"] = _validate_run_contract(identity["run_contract"], method=method)
    return normalized


def _validate_candidate(candidate: object) -> dict[str, Any]:
    if not isinstance(candidate, dict) or set(candidate) != {"field", "value"}:
        raise ValueError("terminal result candidate must contain exactly field and value")
    field = candidate["field"]
    if field not in {"learning_rate", "candidate_main_lr_multiplier"}:
        raise ValueError("terminal result candidate field is unregistered")
    return {"field": field, "value": _finite_number(candidate["value"], "candidate.value", positive=True)}


def _validate_selection_recipe(recipe: object, candidate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(recipe, dict):
        raise ValueError("terminal result selection_recipe is required")
    required = set(REQUIRED_RECIPE_FIELDS) | {candidate["field"]}
    if set(recipe) != required:
        raise ValueError("terminal result selection_recipe has incompatible fields")
    normalized = {
        field: _finite_number(recipe[field], f"selection_recipe.{field}", positive=True)
        for field in sorted(required)
    }
    if normalized["muon_adamw_lr_scale"] != REGISTERED_ADAMW_FALLBACK_SCALE:
        raise ValueError("terminal result selection_recipe must retain the registered AdamW fallback scale")
    if not math.isclose(
        normalized["min_lr"],
        0.1 * normalized["learning_rate"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("terminal result selection_recipe min_lr must equal 0.1 * learning_rate")
    if normalized[candidate["field"]] != candidate["value"]:
        raise ValueError("terminal result candidate value disagrees with selection_recipe")
    return normalized


def _registered_screen_spec(method: object, stage: object) -> dict[str, Any]:
    spec = REGISTERED_SCREEN_CANDIDATES.get((method, stage))
    if spec is None:
        raise ValueError("screen-only terminal result method/stage is unregistered")
    return spec


def _validate_registered_screen_candidate(method: object, stage: object, candidate: dict[str, Any]) -> None:
    spec = _registered_screen_spec(method, stage)
    if candidate["field"] != spec["field"] or candidate["value"] not in spec["values"]:
        raise ValueError("screen-only terminal result candidate is not in the registered screen set")


def _require_full_registered_screen_set(records: list[dict[str, Any]], *, label: str) -> None:
    spec = _registered_screen_spec(records[0]["method"], records[0]["hpo_stage"])
    actual = {(record["candidate"]["field"], record["candidate"]["value"]) for record in records}
    expected = {(spec["field"], value) for value in spec["values"]}
    if actual != expected:
        raise ValueError(f"{label} must contain exactly the registered screen candidate set")


def _validate_confirmation_provenance(provenance: object, slot: str) -> dict[str, str]:
    if not isinstance(provenance, dict) or set(provenance) != {"path", "sha256", "schema_version", "slot"}:
        raise ValueError("5TPP confirmation result selection_provenance is required")
    path = provenance["path"]
    if not isinstance(path, str) or not path:
        raise ValueError("5TPP confirmation result ranking artifact path is required")
    if provenance["schema_version"] != RANKING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("5TPP confirmation result ranking artifact schema is incompatible")
    if provenance["slot"] != slot:
        raise ValueError("5TPP confirmation result ranking provenance slot is incompatible")
    return {
        "path": str(Path(path).resolve()),
        "sha256": _require_sha256(provenance["sha256"], "selection_provenance.sha256"),
        "schema_version": RANKING_ARTIFACT_SCHEMA_VERSION,
        "slot": slot,
    }


def _validate_completion_artifact_reference(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"terminal result completion.{name} must pin path and SHA-256")
    path = value["path"]
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ValueError(f"terminal result completion.{name}.path must be an absolute path")
    return {
        "path": str(Path(path)),
        "sha256": _require_sha256(value["sha256"], f"completion.{name}.sha256"),
    }


def _validate_completion(completion: object) -> dict[str, Any]:
    required = {
        "config",
        "status",
        "log",
        "checkpoint_metadata",
        "terminal_iteration",
        "terminal_train_loss",
        "terminal_val_loss",
    }
    if not isinstance(completion, dict) or set(completion) != required:
        raise ValueError("terminal result completion has incompatible fields")
    return {
        "config": _validate_completion_artifact_reference(completion["config"], "config"),
        "status": _validate_completion_artifact_reference(completion["status"], "status"),
        "log": _validate_completion_artifact_reference(completion["log"], "log"),
        "checkpoint_metadata": _validate_completion_artifact_reference(
            completion["checkpoint_metadata"], "checkpoint_metadata"
        ),
        "terminal_iteration": _nonnegative_integer(
            completion["terminal_iteration"], "completion.terminal_iteration"
        ),
        "terminal_train_loss": _finite_number(
            completion["terminal_train_loss"], "completion.terminal_train_loss"
        ),
        "terminal_val_loss": _finite_number(
            completion["terminal_val_loss"], "completion.terminal_val_loss"
        ),
    }


def validate_terminal_result(record: object) -> dict[str, Any]:
    """Validate one accepted, terminal MAI result record.

    The returned normalized object is the only representation embedded in a
    selection artifact, keeping fields that affect selection explicit.
    """
    if not isinstance(record, dict):
        raise ValueError("terminal result must be a JSON object")
    if record.get("schema_version") != TERMINAL_RESULT_SCHEMA_VERSION:
        raise ValueError("terminal result schema/version is incompatible")
    if record.get("acceptance_state") != "ACCEPTED":
        raise ValueError("terminal result is not ACCEPTED")
    if record.get("mai_ladder_policy_version") != POLICY_VERSION:
        raise ValueError("terminal result policy version is incompatible")
    tier = record.get("model_tier")
    method = record.get("method")
    stage = record.get("hpo_stage")
    role = record.get("ladder_role")
    slot = record.get("ladder_slot")
    if not isinstance(tier, str) or not tier:
        raise ValueError("terminal result model_tier is required")
    if method not in {"baseline", "block_fht"}:
        raise ValueError("terminal result method is unregistered")
    if not isinstance(stage, str) or not stage:
        raise ValueError("terminal result hpo_stage is required")
    if role not in {"screen_only", "confirmation", "selected_recipe"}:
        raise ValueError("terminal result ladder_role is unregistered")
    if not isinstance(slot, str) or not slot:
        raise ValueError("terminal result ladder_slot is required")
    provenance = record.get("selection_provenance")
    if role == "confirmation":
        provenance = _validate_confirmation_provenance(provenance, slot)
    elif provenance is not None:
        raise ValueError("non-confirmation terminal result must not carry selection_provenance")
    candidate = _validate_candidate(record.get("candidate"))
    recipe = _validate_selection_recipe(record.get("selection_recipe"), candidate)
    if role == "screen_only":
        _validate_registered_screen_candidate(method, stage, candidate)
    terminal_nll = _finite_number(record.get("terminal_held_out_nll"), "terminal_held_out_nll")
    completion = None
    if "completion" in record:
        completion = _validate_completion(record["completion"])
        if completion["terminal_val_loss"] != terminal_nll:
            raise ValueError("terminal result completion val loss disagrees with terminal_held_out_nll")
    normalized = {
        "schema_version": TERMINAL_RESULT_SCHEMA_VERSION,
        "acceptance_state": "ACCEPTED",
        "mai_ladder_policy_version": POLICY_VERSION,
        "model_tier": tier,
        "method": method,
        "hpo_stage": stage,
        "ladder_role": role,
        "ladder_slot": slot,
        "selection_provenance": provenance,
        "planned_tpp": _finite_number(record.get("planned_tpp"), "planned_tpp", positive=True),
        "candidate": candidate,
        "selection_recipe": recipe,
        "terminal_held_out_nll": terminal_nll,
        "identity": _validate_identity(record.get("identity"), method=method),
    }
    if completion is not None:
        normalized["completion"] = completion
    return normalized


def _shared_identity(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("selection artifact requires terminal records")
    reference = {field: records[0]["identity"][field] for field in SHARED_IDENTITY_FIELDS}
    for record in records[1:]:
        current = {field: record["identity"][field] for field in SHARED_IDENTITY_FIELDS}
        if current != reference:
            raise ValueError("terminal result records have incompatible shared identity fields")
    return _json_copy(reference)


def _require_common(records: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    reference = {field: records[0][field] for field in fields}
    for record in records[1:]:
        if any(record[field] != reference[field] for field in fields):
            raise ValueError("terminal result records do not share tier/method/stage/TPP")
    return reference


def _source_entry(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_path": str(path.resolve()),
        "record_sha256": sha256_file(path),
        "config_sha256": record["identity"]["config_sha256"],
        "record": record,
    }


def _load_source_entries(paths: list[Path]) -> list[dict[str, Any]]:
    entries = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"terminal result record is missing: {path}")
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"terminal result record is invalid JSON: {path}") from exc
        entries.append(_source_entry(path, validate_terminal_result(raw)))
    if len({entry["record_sha256"] for entry in entries}) != len(entries):
        raise ValueError("terminal result inputs must be distinct records")
    return entries


def _validate_source_entries(entries: object, expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError(f"selection artifact must contain exactly {expected_count} source result records")
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("selection artifact source record is invalid")
        path = entry.get("record_path")
        if not isinstance(path, str) or not path:
            raise ValueError("selection artifact source record path is required")
        record_hash = _require_sha256(entry.get("record_sha256"), "source record_sha256")
        record = validate_terminal_result(entry.get("record"))
        source_path = Path(path)
        if not source_path.is_file():
            raise ValueError("selection artifact pinned source result record is missing")
        if sha256_file(source_path) != record_hash:
            raise ValueError("selection artifact pinned source result record hash mismatch")
        try:
            source_record = validate_terminal_result(json.loads(source_path.read_text()))
        except json.JSONDecodeError as exc:
            raise ValueError("selection artifact pinned source result record is invalid JSON") from exc
        if source_record != record:
            raise ValueError("selection artifact pinned source result record content mismatch")
        if entry.get("config_sha256") != record["identity"]["config_sha256"]:
            raise ValueError("selection artifact source config identity is not pinned")
        normalized.append({
            "record_path": path,
            "record_sha256": record_hash,
            "config_sha256": record["identity"]["config_sha256"],
            "record": record,
        })
    if len({entry["record_sha256"] for entry in normalized}) != expected_count:
        raise ValueError("selection artifact source record hashes must be distinct")
    return normalized


def _rank_key(entry: dict[str, Any]) -> tuple[float, bytes, str]:
    record = entry["record"]
    return (
        record["terminal_held_out_nll"],
        canonical_json_bytes(record["candidate"]),
        record["identity"]["config_sha256"],
    )


def _ranked_descriptor(entry: dict[str, Any]) -> dict[str, Any]:
    record = entry["record"]
    return {
        "source_record_sha256": entry["record_sha256"],
        "source_config_sha256": record["identity"]["config_sha256"],
        "candidate": _json_copy(record["candidate"]),
        "selection_recipe": _json_copy(record["selection_recipe"]),
        "terminal_held_out_nll": record["terminal_held_out_nll"],
    }


def _validate_ranked_descriptor(descriptor: object, entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(descriptor, dict):
        raise ValueError("ranking artifact ranked slot is invalid")
    source_hash = _require_sha256(descriptor.get("source_record_sha256"), "ranked source_record_sha256")
    matches = [entry for entry in entries if entry["record_sha256"] == source_hash]
    if len(matches) != 1:
        raise ValueError("ranking artifact ranked slot does not reference a source record")
    expected = _ranked_descriptor(matches[0])
    if descriptor != expected:
        raise ValueError("ranking artifact ranked slot does not exactly pin its source record")
    return expected


def build_ranking_artifact(record_paths: list[Path]) -> dict[str, Any]:
    """Rank exactly three accepted 0.5TPP screen records for one tier/method."""
    if len(record_paths) != 3:
        raise ValueError("exactly three accepted 0.5TPP terminal result records are required")
    entries = _load_source_entries(record_paths)
    records = [entry["record"] for entry in entries]
    common = _require_common(records, ("model_tier", "method", "hpo_stage", "ladder_role", "planned_tpp"))
    if common["ladder_role"] != "screen_only":
        raise ValueError("ranking inputs must be 0.5TPP screen_only records")
    _exact_tpp(common["planned_tpp"], 0.5, "ranking input planned_tpp")
    _require_full_registered_screen_set(records, label="ranking inputs")
    shared = _shared_identity(records)
    ordered = sorted(entries, key=_rank_key)
    return {
        "schema_version": RANKING_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": RANKING_ARTIFACT_KIND,
        "state": "ACCEPTED",
        "mai_ladder_policy_version": POLICY_VERSION,
        "model_tier": common["model_tier"],
        "method": common["method"],
        "hpo_stage": common["hpo_stage"],
        "planned_tpp": 0.5,
        "shared_identity": shared,
        "source_records": entries,
        "ranking_rule": "terminal_held_out_nll_ascending_then_candidate_canonical_json",
        "ranked_slots": {
            "top1": _ranked_descriptor(ordered[0]),
            "top2": _ranked_descriptor(ordered[1]),
        },
    }


def validate_ranking_artifact(artifact: object) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ValueError("ranking artifact must be a JSON object")
    if artifact.get("schema_version") != RANKING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("ranking artifact schema/version is incompatible")
    if artifact.get("artifact_kind") != RANKING_ARTIFACT_KIND or artifact.get("state") != "ACCEPTED":
        raise ValueError("ranking artifact kind/state is incompatible")
    if artifact.get("mai_ladder_policy_version") != POLICY_VERSION:
        raise ValueError("ranking artifact policy version is incompatible")
    for field in ("model_tier", "hpo_stage"):
        if not isinstance(artifact.get(field), str) or not artifact[field]:
            raise ValueError(f"ranking artifact {field} is required")
    if artifact.get("method") not in {"baseline", "block_fht"}:
        raise ValueError("ranking artifact method is unregistered")
    _exact_tpp(artifact.get("planned_tpp"), 0.5, "ranking artifact planned_tpp")
    entries = _validate_source_entries(artifact.get("source_records"), 3)
    records = [entry["record"] for entry in entries]
    common = _require_common(records, ("model_tier", "method", "hpo_stage", "ladder_role", "planned_tpp"))
    if common["ladder_role"] != "screen_only":
        raise ValueError("ranking artifact inputs must be screen_only")
    _exact_tpp(common["planned_tpp"], 0.5, "ranking artifact input planned_tpp")
    if any(artifact[field] != common[field] for field in ("model_tier", "method", "hpo_stage")):
        raise ValueError("ranking artifact identity disagrees with source records")
    shared = _shared_identity(records)
    if artifact.get("shared_identity") != shared:
        raise ValueError("ranking artifact shared identity is incompatible")
    if artifact.get("ranking_rule") != "terminal_held_out_nll_ascending_then_candidate_canonical_json":
        raise ValueError("ranking artifact rule is incompatible")
    _require_full_registered_screen_set(records, label="ranking artifact source candidates")
    ranked_slots = artifact.get("ranked_slots")
    if not isinstance(ranked_slots, dict) or set(ranked_slots) != {"top1", "top2"}:
        raise ValueError("ranking artifact must contain exactly top1 and top2 slots")
    ordered = sorted(entries, key=_rank_key)
    for slot, expected_entry in (("top1", ordered[0]), ("top2", ordered[1])):
        descriptor = _validate_ranked_descriptor(ranked_slots[slot], entries)
        if descriptor != _ranked_descriptor(expected_entry):
            raise ValueError("ranking artifact top-two slots are stale or incorrectly ordered")
    return {
        "model_tier": artifact["model_tier"],
        "method": artifact["method"],
        "hpo_stage": artifact["hpo_stage"],
        "shared_identity": shared,
        "ranked_slots": _json_copy(ranked_slots),
    }


def _read_pinned_json(path_value: object, expected_hash: object, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is required")
    expected = _require_sha256(expected_hash, f"{label} SHA-256")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} hash mismatch")
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return path, loaded


def _comparison_source_descriptor(entry: dict[str, Any]) -> dict[str, Any]:
    record = entry["record"]
    return {
        "source_record_sha256": entry["record_sha256"],
        "source_config_sha256": record["identity"]["config_sha256"],
        "candidate": _json_copy(record["candidate"]),
        "selection_recipe": _json_copy(record["selection_recipe"]),
        "terminal_held_out_nll": record["terminal_held_out_nll"],
        "ladder_slot": record["ladder_slot"],
    }


def _require_ranked_confirmation_entries(
    entries: list[dict[str, Any]],
    ranking: dict[str, Any],
    ranking_reference: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    by_slot = {entry["record"]["ladder_slot"]: entry for entry in entries}
    if set(by_slot) != {"top1", "top2"}:
        raise ValueError("comparison inputs must be exactly the ranked top1 and top2 5TPP slots")
    for slot, entry in by_slot.items():
        record = entry["record"]
        ranked = ranking["ranked_slots"][slot]
        provenance = record["selection_provenance"]
        if (
            provenance != {
                "path": str(Path(ranking_reference["path"]).resolve()),
                "sha256": ranking_reference["sha256"],
                "schema_version": RANKING_ARTIFACT_SCHEMA_VERSION,
                "slot": slot,
            }
        ):
            raise ValueError("comparison input is not pinned to this ranking artifact and slot")
        if (
            record["candidate"] != ranked["candidate"]
            or record["selection_recipe"] != ranked["selection_recipe"]
        ):
            raise ValueError("comparison input is stale or not one of the ranked top-two recipes")
    return by_slot


def build_comparison_artifact(ranking_path: Path, record_paths: list[Path]) -> dict[str, Any]:
    """Compare exactly the two 5TPP confirmations authorized by a ranking artifact."""
    if len(record_paths) != 2:
        raise ValueError("exactly two ranked 5TPP terminal result records are required")
    ranking_hash = sha256_file(ranking_path) if ranking_path.is_file() else None
    _, raw_ranking = _read_pinned_json(ranking_path.as_posix(), ranking_hash, "ranking artifact")
    ranking = validate_ranking_artifact(raw_ranking)
    ranking_reference = {
        "path": str(ranking_path.resolve()),
        "sha256": ranking_hash,
        "schema_version": RANKING_ARTIFACT_SCHEMA_VERSION,
    }
    entries = _load_source_entries(record_paths)
    records = [entry["record"] for entry in entries]
    common = _require_common(records, ("model_tier", "method", "hpo_stage", "ladder_role", "planned_tpp"))
    if common["ladder_role"] != "confirmation":
        raise ValueError("comparison inputs must be 5TPP confirmation records")
    _exact_tpp(common["planned_tpp"], 5.0, "comparison input planned_tpp")
    if (
        common["model_tier"] != ranking["model_tier"]
        or common["method"] != ranking["method"]
    ):
        raise ValueError("comparison inputs disagree with ranking tier or method")
    shared = _shared_identity(records)
    if shared != ranking["shared_identity"]:
        raise ValueError("comparison inputs have incompatible shared identity with ranking")
    by_slot = _require_ranked_confirmation_entries(entries, ranking, ranking_reference)
    top1 = _comparison_source_descriptor(by_slot["top1"])
    top2 = _comparison_source_descriptor(by_slot["top2"])
    gap = abs(top1["terminal_held_out_nll"] - top2["terminal_held_out_nll"])
    if gap > PRACTICAL_EQUIVALENCE_NLL:
        selected_slot = "top1" if top1["terminal_held_out_nll"] < top2["terminal_held_out_nll"] else "top2"
        outcome = "LOWER_5TPP_NLL"
    else:
        selected_slot = "top1"
        outcome = "PRACTICAL_TIE"
    selected = top1 if selected_slot == "top1" else top2
    return {
        "schema_version": COMPARISON_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": COMPARISON_ARTIFACT_KIND,
        "state": "ACCEPTED",
        "mai_ladder_policy_version": POLICY_VERSION,
        "model_tier": common["model_tier"],
        "method": common["method"],
        "hpo_stage": common["hpo_stage"],
        "planned_tpp": 5.0,
        "shared_identity": shared,
        "ranking_artifact": ranking_reference,
        "source_records": entries,
        "ranked_confirmation_results": {"top1": top1, "top2": top2},
        "practical_equivalence_nll": PRACTICAL_EQUIVALENCE_NLL,
        "selection_rule": "lower_5tpp_nll_only_when_gap_strictly_exceeds_0.02_else_ranking_top1",
        "terminal_nll_gap": gap,
        "selected_slot": selected_slot,
        "selected_candidate": _json_copy(selected["candidate"]),
        "selected_selection_recipe": _json_copy(selected["selection_recipe"]),
        "selection_outcome": outcome,
    }


def _validate_comparison_descriptor(
    descriptor: object, entry: dict[str, Any], slot: str
) -> dict[str, Any]:
    expected = _comparison_source_descriptor(entry)
    if not isinstance(descriptor, dict) or descriptor != expected or descriptor.get("ladder_slot") != slot:
        raise ValueError("comparison artifact result descriptor is stale or incompatible")
    return expected


def validate_comparison_artifact(artifact: object) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ValueError("comparison artifact must be a JSON object")
    if artifact.get("schema_version") != COMPARISON_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("comparison artifact schema/version is incompatible")
    if artifact.get("artifact_kind") != COMPARISON_ARTIFACT_KIND or artifact.get("state") != "ACCEPTED":
        raise ValueError("comparison artifact kind/state is incompatible")
    if artifact.get("mai_ladder_policy_version") != POLICY_VERSION:
        raise ValueError("comparison artifact policy version is incompatible")
    for field in ("model_tier", "hpo_stage"):
        if not isinstance(artifact.get(field), str) or not artifact[field]:
            raise ValueError(f"comparison artifact {field} is required")
    if artifact.get("method") not in {"baseline", "block_fht"}:
        raise ValueError("comparison artifact method is unregistered")
    _exact_tpp(artifact.get("planned_tpp"), 5.0, "comparison artifact planned_tpp")
    ranking_reference = artifact.get("ranking_artifact")
    if not isinstance(ranking_reference, dict) or set(ranking_reference) != {"path", "sha256", "schema_version"}:
        raise ValueError("comparison artifact ranking reference is incomplete")
    if ranking_reference["schema_version"] != RANKING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("comparison artifact ranking schema is incompatible")
    _, raw_ranking = _read_pinned_json(
        ranking_reference["path"], ranking_reference["sha256"], "comparison ranking artifact"
    )
    ranking = validate_ranking_artifact(raw_ranking)
    entries = _validate_source_entries(artifact.get("source_records"), 2)
    records = [entry["record"] for entry in entries]
    common = _require_common(records, ("model_tier", "method", "hpo_stage", "ladder_role", "planned_tpp"))
    if common["ladder_role"] != "confirmation":
        raise ValueError("comparison artifact inputs must be confirmation records")
    _exact_tpp(common["planned_tpp"], 5.0, "comparison artifact input planned_tpp")
    if any(artifact[field] != common[field] for field in ("model_tier", "method", "hpo_stage")):
        raise ValueError("comparison artifact identity disagrees with source records")
    if artifact["model_tier"] != ranking["model_tier"] or artifact["method"] != ranking["method"]:
        raise ValueError("comparison artifact disagrees with ranking tier or method")
    shared = _shared_identity(records)
    if artifact.get("shared_identity") != shared or shared != ranking["shared_identity"]:
        raise ValueError("comparison artifact shared identity is incompatible")
    by_slot = _require_ranked_confirmation_entries(entries, ranking, ranking_reference)
    descriptors = artifact.get("ranked_confirmation_results")
    if not isinstance(descriptors, dict) or set(descriptors) != {"top1", "top2"}:
        raise ValueError("comparison artifact must contain exactly ranked top1 and top2 results")
    top1 = _validate_comparison_descriptor(descriptors["top1"], by_slot["top1"], "top1")
    top2 = _validate_comparison_descriptor(descriptors["top2"], by_slot["top2"], "top2")
    threshold = artifact.get("practical_equivalence_nll")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or float(threshold) != PRACTICAL_EQUIVALENCE_NLL:
        raise ValueError("comparison artifact practical-equivalence threshold must be exactly 0.02")
    if artifact.get("selection_rule") != "lower_5tpp_nll_only_when_gap_strictly_exceeds_0.02_else_ranking_top1":
        raise ValueError("comparison artifact selection rule is incompatible")
    gap = abs(top1["terminal_held_out_nll"] - top2["terminal_held_out_nll"])
    recorded_gap = _finite_number(artifact.get("terminal_nll_gap"), "comparison artifact terminal_nll_gap")
    if not math.isclose(recorded_gap, gap, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("comparison artifact terminal NLL gap is inconsistent")
    if gap > PRACTICAL_EQUIVALENCE_NLL:
        expected_slot = "top1" if top1["terminal_held_out_nll"] < top2["terminal_held_out_nll"] else "top2"
        expected_outcome = "LOWER_5TPP_NLL"
    else:
        expected_slot = "top1"
        expected_outcome = "PRACTICAL_TIE"
    selected = top1 if expected_slot == "top1" else top2
    if (
        artifact.get("selected_slot") != expected_slot
        or artifact.get("selection_outcome") != expected_outcome
        or artifact.get("selected_candidate") != selected["candidate"]
        or artifact.get("selected_selection_recipe") != selected["selection_recipe"]
    ):
        raise ValueError("comparison artifact violates the fixed registered 5TPP selection rule")
    return {
        "model_tier": artifact["model_tier"],
        "method": artifact["method"],
        "hpo_stage": artifact["hpo_stage"],
        "shared_identity": shared,
        "selected_slot": expected_slot,
        "selected_candidate": _json_copy(selected["candidate"]),
        "selected_selection_recipe": _json_copy(selected["selection_recipe"]),
    }


def _validate_v2_policy(config: dict[str, Any]) -> None:
    if config.get("mai_ladder_policy_version") != POLICY_VERSION:
        raise ValueError("MAI v2 launch config policy version is incompatible")
    threshold = config.get("practical_equivalence_nll")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or float(threshold) != PRACTICAL_EQUIVALENCE_NLL
    ):
        raise ValueError("MAI v2 launch config practical-equivalence threshold must be exactly 0.02")


def _validate_v2_determinism_config(config: dict[str, Any]) -> None:
    for field, expected in REGISTERED_V2_DETERMINISM_SETTINGS.items():
        if config.get(field) != expected:
            if field == "checkpoint_history":
                expected_text = "false"
            elif field == "save_checkpoint":
                expected_text = "true"
            else:
                expected_text = repr(expected)
            raise ValueError(f"MAI v2 launch config must retain {field}={expected_text}")


def _validate_v2_config_recipe_binding(
    config: dict[str, Any], candidate: object, recipe: object
) -> None:
    if not isinstance(candidate, dict) or set(candidate) != {"field", "value"}:
        raise ValueError("MAI v2 launch config has invalid selected candidate")
    if not isinstance(recipe, dict) or config.get("mai_selection_recipe") != recipe:
        raise ValueError("MAI v2 launch config selection recipe is not exactly pinned")
    if config.get("mai_selection_candidate") != candidate:
        raise ValueError("MAI v2 launch config selected candidate is not exactly pinned")
    field = candidate["field"]
    if field not in recipe or recipe[field] != candidate["value"]:
        raise ValueError("MAI v2 launch config selected candidate disagrees with recipe")
    for field, value in recipe.items():
        if config.get(field) != value:
            raise ValueError(f"MAI v2 launch config recipe field {field} disagrees with pinned artifact")


def _load_pinned_v2_selection_artifact(
    path_value: object,
    expected_hash: object,
    *,
    label: str,
    validator: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path, raw = _read_pinned_json(path_value, expected_hash, label)
    # Keep the path resolution inside the immutable artifact reader. The path
    # itself is deliberately not returned because the config pin is the only
    # launch authority.
    del path
    return raw, validator(raw)


def _validate_v2_shared_identity_binding(
    config: dict[str, Any],
    shared_identity: dict[str, Any],
    *,
    method: str,
    resolved_config: dict[str, Any] | None,
    runtime_source_hashes: dict[str, str] | None,
) -> None:
    """Bind selected 5/20TPP configs to immutable contract identities.

    The function intentionally accepts plain dictionaries only.  It is shared
    by the torch-free terminal builder and ``train.py``; callers provide the
    actual resolved runtime dictionary/source hashes when those are available.
    """
    if runtime_source_hashes is not None and runtime_source_hashes != shared_identity["source_hashes"]:
        raise ValueError("MAI v2 launch source identity disagrees with selection artifact")
    contract = shared_identity["run_contract"]
    expected_fields = ("method", *RUN_CONTRACT_FIELDS)
    for field in expected_fields:
        expected = method if field == "method" else contract[field]
        if not _same_typed_value(config.get(field), expected):
            raise ValueError(f"MAI v2 launch config run-contract field {field} disagrees with selection artifact")
        if resolved_config is not None and not _same_typed_value(resolved_config.get(field), expected):
            raise ValueError(
                f"MAI v2 resolved runtime run-contract field {field} disagrees with selection artifact"
            )
    method_structure = contract.get("method_structure")
    if method_structure is None:
        return
    if method_structure.get("method") != method:
        raise ValueError("MAI v2 method-specific structure disagrees with selection artifact")
    if method != "block_fht":
        return
    block_fht_structure = method_structure.get("block_fht")
    if not isinstance(block_fht_structure, dict):
        raise ValueError("MAI v2 BlockFHT selection identity is incomplete")
    for field in BLOCK_FHT_STRUCTURE_FIELDS:
        expected = block_fht_structure.get(field, object())
        if not _same_typed_value(config.get(field, object()), expected):
            raise ValueError(f"MAI v2 launch config run-contract field {field} disagrees with selection artifact")
        if resolved_config is not None and not _same_typed_value(resolved_config.get(field, object()), expected):
            raise ValueError(
                f"MAI v2 resolved runtime run-contract field {field} disagrees with selection artifact"
            )


def _validate_v2_runtime_config(
    config: dict[str, Any], resolved_config: dict[str, Any] | None
) -> None:
    if resolved_config is None:
        return
    runtime_fields = (
        *REGISTERED_V2_DETERMINISM_SETTINGS,
        *REGISTERED_V2_OPTIMIZER_SETTINGS,
        "learning_rate",
        "min_lr",
        *RUN_CONTRACT_FIELDS,
    )
    if config.get("method") == "block_fht":
        runtime_fields = (*runtime_fields, *BLOCK_FHT_STRUCTURE_FIELDS)
    for field in dict.fromkeys(runtime_fields):
        if field not in resolved_config or not _same_typed_value(resolved_config[field], config.get(field)):
            raise ValueError(f"MAI v2 resolved runtime field {field} disagrees with launch config")


def _validate_v2_confirmation_artifact(
    config: dict[str, Any],
    *,
    method: str,
    tier: object,
    resolved_config: dict[str, Any] | None,
    runtime_source_hashes: dict[str, str] | None,
) -> dict[str, Any]:
    _exact_tpp(config.get("planned_tpp"), 5.0, "MAI v2 5TPP confirmation")
    slot = config.get("ladder_slot")
    if slot not in {"top1", "top2"}:
        raise ValueError("MAI v2 5TPP config has invalid ranking slot")
    if config.get("zero_point_five_tpp_ranking_artifact_required") is not True:
        raise ValueError("MAI v2 5TPP config requires an immutable 0.5TPP ranking artifact")
    if config.get("zero_point_five_tpp_ranking_artifact_schema") != RANKING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("MAI v2 5TPP config has incompatible ranking artifact schema")
    raw, ranking = _load_pinned_v2_selection_artifact(
        config.get("zero_point_five_tpp_ranking_artifact"),
        config.get("zero_point_five_tpp_ranking_artifact_sha256"),
        label="MAI v2 0.5TPP ranking artifact",
        validator=validate_ranking_artifact,
    )
    if raw.get("schema_version") != RANKING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("MAI v2 5TPP config has incompatible ranking artifact schema")
    if (
        ranking["model_tier"] != tier
        or ranking["method"] != method
        or ranking["hpo_stage"] != config.get("zero_point_five_tpp_ranking_hpo_stage")
        or config.get("zero_point_five_tpp_ranking_tier") != tier
        or config.get("zero_point_five_tpp_ranking_method") != method
        or config.get("zero_point_five_tpp_ranking_slot") != slot
    ):
        raise ValueError("MAI v2 5TPP config disagrees with pinned ranking tier/method/stage/slot")
    ranked = ranking["ranked_slots"][slot]
    _validate_v2_config_recipe_binding(config, ranked["candidate"], ranked["selection_recipe"])
    _validate_v2_shared_identity_binding(
        config,
        ranking["shared_identity"],
        method=method,
        resolved_config=resolved_config,
        runtime_source_hashes=runtime_source_hashes,
    )
    return ranking["shared_identity"]


def _validate_v2_selected_recipe_artifact(
    config: dict[str, Any],
    *,
    method: str,
    tier: object,
    resolved_config: dict[str, Any] | None,
    runtime_source_hashes: dict[str, str] | None,
) -> dict[str, Any]:
    _exact_tpp(config.get("planned_tpp"), 20.0, "MAI v2 20TPP selected-recipe")
    if config.get("ladder_slot") != "selected_from_5tpp_comparison":
        raise ValueError("MAI v2 20TPP config has invalid comparison-selected slot")
    if config.get("five_tpp_comparison_artifact_required") is not True:
        raise ValueError("MAI v2 20TPP config requires an immutable 5TPP comparison artifact")
    if config.get("five_tpp_comparison_artifact_schema") != COMPARISON_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("MAI v2 20TPP config has incompatible comparison artifact schema")
    raw, comparison = _load_pinned_v2_selection_artifact(
        config.get("five_tpp_comparison_artifact"),
        config.get("five_tpp_comparison_artifact_sha256"),
        label="MAI v2 5TPP comparison artifact",
        validator=validate_comparison_artifact,
    )
    if raw.get("schema_version") != COMPARISON_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("MAI v2 20TPP config has incompatible comparison artifact schema")
    if (
        comparison["model_tier"] != tier
        or comparison["method"] != method
        or comparison["hpo_stage"] != config.get("five_tpp_comparison_hpo_stage")
        or config.get("five_tpp_comparison_tier") != tier
        or config.get("five_tpp_comparison_method") != method
    ):
        raise ValueError("MAI v2 20TPP config disagrees with pinned comparison tier/method/stage")
    _validate_v2_config_recipe_binding(
        config, comparison["selected_candidate"], comparison["selected_selection_recipe"]
    )
    _validate_v2_shared_identity_binding(
        config,
        comparison["shared_identity"],
        method=method,
        resolved_config=resolved_config,
        runtime_source_hashes=runtime_source_hashes,
    )
    return comparison["shared_identity"]


def validate_v2_launch_config(
    config: object,
    *,
    resolved_config: dict[str, Any] | None = None,
    runtime_source_hashes: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Validate the complete, torch-free registered MAI-v2 launch policy.

    Configs that do not opt into the MAI policy are intentionally ignored so
    the legacy training validation path remains available.  Both the terminal
    result builder and training entry point call this exact validator.
    """
    if not isinstance(config, dict):
        raise ValueError("launch config must be a JSON object")
    if "mai_ladder_policy_version" not in config:
        return None
    if resolved_config is not None and not isinstance(resolved_config, dict):
        raise ValueError("MAI v2 resolved runtime config must be an object")
    _validate_v2_policy(config)
    if config.get("launch_ready") is not True:
        raise ValueError("MAI v2 launch config must set launch_ready=true")
    if config.get("recipe_resolution_required") is not False:
        raise ValueError("MAI v2 launch config still requires recipe resolution")
    _validate_v2_determinism_config(config)
    _validate_v2_optimizer_config(config)

    method = config.get("method")
    tier = config.get("model_tier")
    role = config.get("ladder_role")
    if method not in {"baseline", "block_fht"}:
        raise ValueError("MAI v2 launch config method is unregistered")
    if not isinstance(tier, str) or not tier:
        raise ValueError("MAI v2 launch config model_tier is required")
    if role == "screen_only":
        _exact_tpp(config.get("planned_tpp"), 0.5, "MAI v2 screen")
        candidate_field = "learning_rate" if method == "baseline" else "candidate_main_lr_multiplier"
        candidate = {"field": candidate_field, "value": config.get(candidate_field)}
        recipe = {
            "learning_rate": config.get("learning_rate"),
            "min_lr": config.get("min_lr"),
            "muon_adamw_lr_scale": config.get("muon_adamw_lr_scale"),
        }
        if method == "block_fht":
            recipe[candidate_field] = config.get(candidate_field)
        normalized_candidate = _validate_candidate(candidate)
        _validate_selection_recipe(recipe, normalized_candidate)
        _validate_registered_screen_candidate(method, config.get("hpo_stage"), normalized_candidate)
        if method == "block_fht":
            _validate_block_fht_dense_fit_gate(config)
        _validate_v2_runtime_config(config, resolved_config)
        return None
    if role == "confirmation":
        shared_identity = _validate_v2_confirmation_artifact(
            config,
            method=method,
            tier=tier,
            resolved_config=resolved_config,
            runtime_source_hashes=runtime_source_hashes,
        )
    elif role == "selected_recipe":
        shared_identity = _validate_v2_selected_recipe_artifact(
            config,
            method=method,
            tier=tier,
            resolved_config=resolved_config,
            runtime_source_hashes=runtime_source_hashes,
        )
    else:
        raise ValueError("MAI v2 launch config ladder_role is unregistered")
    if method == "block_fht":
        _validate_block_fht_dense_fit_gate(config)
    _validate_v2_runtime_config(config, resolved_config)
    return shared_identity


_TERMINAL_EVALUATION_RE = re.compile(
    r"^\s*step\s+(?P<iteration>\d+)\s*:\s*train loss\s+(?P<train>[^,\s]+)\s*,\s*"
    r"val loss\s+(?P<val>\S+)\s*$",
    re.MULTILINE,
)


def _canonical_existing_file(path_value: Path | str, label: str) -> Path:
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path.resolve()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _same_typed_value(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _canonical_status_path(
    status: dict[str, Any],
    *,
    keys: tuple[str, ...],
    status_path: Path,
    label: str,
) -> Path:
    values = [(key, status[key]) for key in keys if key in status]
    if not values:
        raise ValueError(f"status is missing {label} path")
    canonical_paths = []
    for key, value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"status {key} must be a non-empty path string")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = status_path.parent / candidate
        canonical_paths.append(candidate.resolve())
    if any(path != canonical_paths[0] for path in canonical_paths[1:]):
        raise ValueError(f"status has conflicting {label} paths")
    return canonical_paths[0]


def _required_status_value(status: dict[str, Any], keys: tuple[str, ...], label: str) -> object:
    values = [(key, status[key]) for key in keys if key in status]
    if not values:
        raise ValueError(f"status is missing {label}")
    first = values[0][1]
    if any(not _same_typed_value(value, first) for _, value in values[1:]):
        raise ValueError(f"status has conflicting {label}")
    return first


def _validate_clean_status(
    status: dict[str, Any],
    *,
    status_path: Path,
    config_path: Path,
    log_path: Path,
    max_iters: int,
) -> None:
    state = _required_status_value(status, ("status", "state"), "completion state")
    if not isinstance(state, str) or state.lower() not in {"clean", "finished", "completed"}:
        raise ValueError("status is not clean/finished")
    if "classification" in status and status["classification"] != "clean":
        raise ValueError("status classification is not clean")
    exit_code = _required_status_value(status, ("exit_code", "train_exit_code"), "exit code")
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError("status exit code is not 0")
    for field in ("failed", "killed"):
        if field in status and status[field] is not False:
            raise ValueError(f"status is not clean: {field} is set")
    if "alive" in status and status["alive"] is not False:
        raise ValueError("status is not clean: alive is set")
    if "max_iters" in status:
        if type(status["max_iters"]) is not int or status["max_iters"] != max_iters:
            raise ValueError("status max_iters disagrees with launch config")
    if _canonical_status_path(
        status,
        keys=("config_path", "config"),
        status_path=status_path,
        label="config",
    ) != config_path:
        raise ValueError("status config path does not match supplied config")
    if _canonical_status_path(
        status,
        keys=("log_path", "log"),
        status_path=status_path,
        label="log",
    ) != log_path:
        raise ValueError("status log path does not match supplied log")


def _terminal_losses(log_path: Path, max_iters: int) -> tuple[float, float]:
    matches = list(_TERMINAL_EVALUATION_RE.finditer(log_path.read_text(errors="replace")))
    terminal = [match for match in matches if int(match.group("iteration")) == max_iters]
    if not terminal:
        if matches:
            raise ValueError(f"terminal evaluation iteration mismatch: expected step {max_iters}")
        raise ValueError(f"terminal evaluation is missing for step {max_iters}")
    if len(terminal) != 1:
        raise ValueError(f"duplicate terminal evaluation lines for step {max_iters}")
    match = terminal[0]
    try:
        train_loss = float(match.group("train"))
        val_loss = float(match.group("val"))
    except ValueError as exc:
        raise ValueError("terminal evaluation losses are invalid") from exc
    if not math.isfinite(train_loss) or not math.isfinite(val_loss):
        raise ValueError("terminal evaluation losses must be finite")
    return train_loss, val_loss


def _run_contract_from_resolved_config(resolved: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(resolved, dict):
        raise ValueError("checkpoint metadata run_identity.resolved_config is required")
    method = resolved.get("method")
    if method not in {"baseline", "block_fht"}:
        raise ValueError("checkpoint metadata resolved method is unregistered")
    missing = [field for field in RUN_CONTRACT_FIELDS if field not in resolved]
    if missing:
        raise ValueError("checkpoint metadata resolved config is missing " + ", ".join(missing))
    contract: dict[str, Any] = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        **{field: resolved[field] for field in RUN_CONTRACT_FIELDS},
        "method_structure": {"method": method},
    }
    if method == "block_fht":
        missing = [field for field in BLOCK_FHT_STRUCTURE_FIELDS if field not in resolved]
        if missing:
            raise ValueError("checkpoint metadata resolved BlockFHT config is missing " + ", ".join(missing))
        contract["method_structure"]["block_fht"] = {
            field: resolved[field] for field in BLOCK_FHT_STRUCTURE_FIELDS
        }
    return method, _validate_run_contract(contract, method=method)


def _require_config_matches_resolved(
    config: dict[str, Any], resolved: dict[str, Any], *, method: str
) -> None:
    fields = {
        "mai_ladder_policy_version",
        "model_tier",
        "method",
        "hpo_stage",
        "ladder_role",
        "ladder_slot",
        "planned_tpp",
        "max_iters",
        "learning_rate",
        "min_lr",
        "muon_adamw_lr_scale",
        "data_manifest_sha256",
        *FIXED_EVALUATION_CONFIG_FIELDS,
        *RUN_CONTRACT_FIELDS,
    }
    if method == "block_fht":
        fields.update({"candidate_main_lr_multiplier", *BLOCK_FHT_STRUCTURE_FIELDS})
    if config.get("ladder_role") == "confirmation":
        fields.update({
            "zero_point_five_tpp_ranking_artifact_required",
            "zero_point_five_tpp_ranking_artifact",
            "zero_point_five_tpp_ranking_artifact_sha256",
            "zero_point_five_tpp_ranking_artifact_schema",
            "zero_point_five_tpp_ranking_tier",
            "zero_point_five_tpp_ranking_method",
            "zero_point_five_tpp_ranking_hpo_stage",
            "zero_point_five_tpp_ranking_slot",
            "mai_selection_candidate",
            "mai_selection_recipe",
        })
    for field in sorted(fields):
        if field not in config or field not in resolved:
            raise ValueError(f"launch config/metadata resolved config is missing relevant field {field}")
        if not _same_typed_value(config[field], resolved[field]):
            raise ValueError(f"launch config field {field} disagrees with checkpoint metadata run contract")


def _validate_fixed_evaluation_identity_binding(
    evaluation: dict[str, Any],
    resolved: dict[str, Any],
    *,
    allow_missing_legacy_labels: bool = False,
) -> None:
    """Reject a checkpoint sidecar whose evaluation identity is self-contradictory."""
    bindings = (
        ("fixed_eval_indices", "fixed_eval_indices"),
        ("protocol", "eval_protocol_id"),
        ("eval_seed", "eval_seed"),
        ("eval_batch_size", "eval_batch_size"),
        ("eval_iters", "eval_iters"),
        ("block_size", "block_size"),
        ("fixed_eval_index_spec_sha256", "fixed_eval_index_spec_sha256"),
        ("fixed_eval_indices_protocol", "fixed_eval_indices_protocol"),
        # This is the runtime digest of materialized fixed index tensors. It
        # must be part of the resolved identity as well as evaluation metadata.
        ("fixed_eval_indices_sha256", "fixed_eval_indices_sha256"),
    )
    for evaluation_field, resolved_field in bindings:
        if evaluation_field not in evaluation or resolved_field not in resolved:
            if (
                allow_missing_legacy_labels
                and evaluation_field in {"fixed_eval_index_spec_sha256", "fixed_eval_indices_protocol"}
                and evaluation_field not in evaluation
                and resolved_field not in resolved
            ):
                continue
            raise ValueError(
                "checkpoint metadata run_identity evaluation/resolved config is missing "
                f"fixed-eval field {evaluation_field}"
            )
        if not _same_typed_value(evaluation[evaluation_field], resolved[resolved_field]):
            raise ValueError(
                "checkpoint metadata run_identity evaluation field "
                f"{evaluation_field} disagrees with resolved config field {resolved_field}"
            )


def _identity_from_checkpoint_metadata(
    metadata: dict[str, Any], config: dict[str, Any], max_iters: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA_VERSION:
        raise ValueError("checkpoint metadata schema/version is incompatible")
    if metadata.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint metadata checkpoint schema/version is incompatible")
    if metadata.get("checkpoint_file") != "ckpt.pt":
        raise ValueError("checkpoint metadata does not describe the exact-resume checkpoint")
    if type(metadata.get("next_iter")) is not int or metadata["next_iter"] != max_iters:
        raise ValueError("checkpoint metadata terminal iteration disagrees with launch config")
    if not isinstance(metadata.get("saved_at_unix"), (int, float)) or isinstance(metadata["saved_at_unix"], bool) or not math.isfinite(float(metadata["saved_at_unix"])):
        raise ValueError("checkpoint metadata saved_at_unix is invalid")
    run_identity = metadata.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ValueError("checkpoint metadata run_identity is required")
    resolved = run_identity.get("resolved_config")
    method, contract = _run_contract_from_resolved_config(resolved)
    assert isinstance(resolved, dict)
    if hashlib.sha256(canonical_json_bytes(resolved)).hexdigest() != _require_sha256(
        run_identity.get("config_sha256"), "checkpoint metadata run_identity.config_sha256"
    ):
        raise ValueError("checkpoint metadata run_identity resolved config SHA-256 mismatch")
    if metadata.get("config_sha256") != run_identity["config_sha256"]:
        raise ValueError("checkpoint metadata config SHA-256 disagrees with run_identity")
    manifest = run_identity.get("data_manifest")
    evaluation = run_identity.get("evaluation")
    if not isinstance(manifest, dict) or not isinstance(evaluation, dict):
        raise ValueError("checkpoint metadata run_identity data/evaluation identity is invalid")
    if evaluation.get("fixed_eval_indices") is not True:
        raise ValueError("checkpoint metadata run_identity lacks fixed evaluation indices")
    high_throughput_screen = (
        "mai_ladder_policy_version" not in config
        and config.get("method") == "baseline"
        and config.get("hpo_stage") == "dense_recipe_screen_0p5tpp"
        and config.get("ladder_role") == "screen_only"
    )
    _validate_fixed_evaluation_identity_binding(
        evaluation,
        resolved,
        allow_missing_legacy_labels=high_throughput_screen,
    )
    identity = _validate_identity({
        "config_sha256": run_identity.get("config_sha256"),
        "source_hashes": run_identity.get("source_hashes"),
        "data_manifest_sha256": manifest.get("sha256"),
        "fixed_eval_indices_sha256": evaluation.get("fixed_eval_indices_sha256"),
        "eval_protocol_id": evaluation.get("protocol"),
        "run_contract": contract,
    }, method=method)
    if high_throughput_screen:
        _validate_high_throughput_screen_config(config, resolved_config=resolved)
    else:
        _require_config_matches_resolved(config, resolved, method=method)
        if config.get("mai_ladder_policy_version") != POLICY_VERSION:
            raise ValueError("launch config is not a registered MAI v2 config")
    if config.get("data_manifest_sha256") != identity["data_manifest_sha256"]:
        raise ValueError("launch config data manifest disagrees with checkpoint metadata")
    if config.get("eval_protocol_id") != identity["eval_protocol_id"]:
        raise ValueError("launch config evaluation protocol disagrees with checkpoint metadata")
    if not high_throughput_screen:
        validate_v2_launch_config(
            config,
            resolved_config=resolved,
            runtime_source_hashes=run_identity.get("source_hashes"),
        )
    return identity, resolved


def _validate_high_throughput_screen_config(
    config: dict[str, Any], *, resolved_config: dict[str, Any] | None = None
) -> None:
    """Accept the hash-pinned MAI-v3 prefetch screens as ranking evidence.

    These screens predate the selection-policy metadata overlay, but their
    checkpoint sidecars bind the complete resolved runtime, source hashes,
    data manifest, and fixed-evaluation digest.  Only this exact registered
    stage is bridged; later rungs still require normal immutable artifacts.
    """
    if config.get("launch_ready") is not True or config.get("recipe_resolution_required") is not False:
        raise ValueError("MAI-v3 high-throughput screen is not launch-ready")
    if (
        config.get("method") != "baseline"
        or config.get("hpo_stage") != "dense_recipe_screen_0p5tpp"
        or config.get("ladder_role") != "screen_only"
        or config.get("model_tier") != "985m"
    ):
        raise ValueError("launch config is not the registered MAI-v3 high-throughput screen")
    _exact_tpp(config.get("planned_tpp"), 0.5, "MAI-v3 high-throughput screen")
    _validate_v2_determinism_config(config)
    _validate_v2_optimizer_config(config)
    candidate = {"field": "learning_rate", "value": config.get("learning_rate")}
    _validate_registered_screen_candidate(config.get("method"), config.get("hpo_stage"), candidate)
    if config.get("mfu_preflight_required") is not True or float(config.get("mfu_min_fraction", 0.0)) < 0.20:
        raise ValueError("MAI-v3 high-throughput screen lacks the mandatory 20% MFU gate")
    if resolved_config is not None:
        for field, value in config.items():
            if field not in resolved_config or not _same_typed_value(resolved_config[field], value):
                raise ValueError(
                    f"MAI-v3 high-throughput launch config field {field} disagrees with checkpoint metadata"
                )


def _confirmation_provenance_from_config(
    config: dict[str, Any], candidate: dict[str, Any], recipe: dict[str, Any]
) -> dict[str, str]:
    slot = config["ladder_slot"]
    if config.get("zero_point_five_tpp_ranking_artifact_required") is not True:
        raise ValueError("confirmation config does not require a 0.5TPP ranking artifact")
    if config.get("zero_point_five_tpp_ranking_artifact_schema") != RANKING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("confirmation config ranking artifact schema is incompatible")
    ranking_path, raw_ranking = _read_pinned_json(
        config.get("zero_point_five_tpp_ranking_artifact"),
        config.get("zero_point_five_tpp_ranking_artifact_sha256"),
        "confirmation ranking artifact",
    )
    ranking = validate_ranking_artifact(raw_ranking)
    if (
        ranking["model_tier"] != config.get("model_tier")
        or ranking["method"] != config.get("method")
        or ranking["hpo_stage"] != config.get("zero_point_five_tpp_ranking_hpo_stage")
        or config.get("zero_point_five_tpp_ranking_tier") != config.get("model_tier")
        or config.get("zero_point_five_tpp_ranking_method") != config.get("method")
        or config.get("zero_point_five_tpp_ranking_slot") != slot
    ):
        raise ValueError("confirmation config ranking provenance disagrees with its pinned artifact")
    ranked = ranking["ranked_slots"].get(slot)
    if ranked is None or ranked["candidate"] != candidate or ranked["selection_recipe"] != recipe:
        raise ValueError("confirmation config recipe is not the pinned ranked slot")
    return {
        "path": str(ranking_path.resolve()),
        "sha256": _require_sha256(
            config.get("zero_point_five_tpp_ranking_artifact_sha256"),
            "confirmation ranking artifact SHA-256",
        ),
        "schema_version": RANKING_ARTIFACT_SCHEMA_VERSION,
        "slot": slot,
    }


def _terminal_record_fields_from_config(
    config: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    method = config.get("method")
    role = config.get("ladder_role")
    if method not in {"baseline", "block_fht"} or role not in {"screen_only", "confirmation", "selected_recipe"}:
        raise ValueError("launch config is not a terminal MAI screen/confirmation/selected rung")
    candidate_field = "learning_rate" if method == "baseline" else "candidate_main_lr_multiplier"
    candidate = {"field": candidate_field, "value": config.get(candidate_field)}
    recipe = {
        "learning_rate": config.get("learning_rate"),
        "min_lr": config.get("min_lr"),
        "muon_adamw_lr_scale": config.get("muon_adamw_lr_scale"),
    }
    if method == "block_fht":
        recipe[candidate_field] = config.get(candidate_field)
    provenance = (
        _confirmation_provenance_from_config(config, candidate, recipe)
        if role == "confirmation"
        else None
    )
    return {
        "model_tier": config.get("model_tier"),
        "method": method,
        "hpo_stage": config.get("hpo_stage"),
        "ladder_role": role,
        "ladder_slot": config.get("ladder_slot"),
        "selection_provenance": provenance,
        "planned_tpp": config.get("planned_tpp"),
        "candidate": candidate,
        "selection_recipe": recipe,
        "identity": identity,
    }


def build_terminal_result(
    config_path: Path,
    status_path: Path,
    log_path: Path,
    checkpoint_metadata_path: Path,
    accept: bool,
) -> dict[str, Any]:
    """Build one accepted v3 terminal result from a finished MAI v2 run.

    This intentionally reads the JSON checkpoint sidecar only; it never loads
    the torch checkpoint.  The sidecar is published after the durable
    exact-resume checkpoint replacement and supplies the run identity.
    """
    config_path = _canonical_existing_file(config_path, "config")
    status_path = _canonical_existing_file(status_path, "status")
    log_path = _canonical_existing_file(log_path, "log")
    checkpoint_metadata_path = _canonical_existing_file(
        checkpoint_metadata_path, "checkpoint metadata"
    )
    if not accept:
        raise ValueError("terminal result requires explicit acceptance")
    config = _load_json_object(config_path, "config")
    if config.get("mai_ladder_policy_version") == POLICY_VERSION:
        validate_v2_launch_config(config)
    else:
        _validate_high_throughput_screen_config(config)
    max_iters = _integer(config.get("max_iters"), "config.max_iters", positive=True)
    status = _load_json_object(status_path, "status")
    _validate_clean_status(
        status,
        status_path=status_path,
        config_path=config_path,
        log_path=log_path,
        max_iters=max_iters,
    )
    metadata = _load_json_object(checkpoint_metadata_path, "checkpoint metadata")
    identity, _ = _identity_from_checkpoint_metadata(metadata, config, max_iters)
    train_loss, val_loss = _terminal_losses(log_path, max_iters)
    record = {
        "schema_version": TERMINAL_RESULT_SCHEMA_VERSION,
        "acceptance_state": "ACCEPTED",
        "mai_ladder_policy_version": POLICY_VERSION,
        **_terminal_record_fields_from_config(config, identity),
        "terminal_held_out_nll": val_loss,
        "completion": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "status": {"path": str(status_path), "sha256": sha256_file(status_path)},
            "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
            "checkpoint_metadata": {
                "path": str(checkpoint_metadata_path),
                "sha256": sha256_file(checkpoint_metadata_path),
            },
            "terminal_iteration": max_iters,
            "terminal_train_loss": train_loss,
            "terminal_val_loss": val_loss,
        },
    }
    return validate_terminal_result(record)


def write_immutable_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Atomically publish once, refusing replacement even under a race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable MAI selection artifact: {path}")
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is atomic and fails if any process published this path first.
        os.link(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rank = commands.add_parser("rank", help="create an immutable 0.5TPP top1/top2 ranking")
    rank.add_argument("--record", action="append", required=True, help="accepted 0.5TPP terminal-result JSON; provide exactly three")
    rank.add_argument("--output", required=True, help="new immutable ranking artifact path")
    compare = commands.add_parser("compare", help="create an immutable 5TPP top1/top2 comparison")
    compare.add_argument("--ranking", required=True, help="immutable 0.5TPP ranking artifact")
    compare.add_argument("--record", action="append", required=True, help="accepted ranked 5TPP terminal-result JSON; provide exactly two")
    compare.add_argument("--output", required=True, help="new immutable comparison artifact path")
    terminal = commands.add_parser(
        "terminal-result", help="create an immutable terminal result from a completed MAI v2 run"
    )
    terminal.add_argument("--config", required=True, help="the exact launch config JSON")
    terminal.add_argument("--status", required=True, help="finished status JSON pinning config_path and log_path")
    terminal.add_argument("--log", required=True, help="named training log containing the terminal evaluation")
    terminal.add_argument("--checkpoint-metadata", required=True, help="published ckpt.meta.json sidecar")
    terminal.add_argument("--output", required=True, help="new immutable terminal-result path")
    terminal.add_argument("--accept", action="store_true", help="explicitly accept the completed terminal result")
    args = parser.parse_args()
    if args.command == "rank":
        artifact = build_ranking_artifact([Path(value) for value in args.record])
        validate_ranking_artifact(artifact)
    elif args.command == "compare":
        artifact = build_comparison_artifact(Path(args.ranking), [Path(value) for value in args.record])
        validate_comparison_artifact(artifact)
    else:
        artifact = build_terminal_result(
            config_path=Path(args.config),
            status_path=Path(args.status),
            log_path=Path(args.log),
            checkpoint_metadata_path=Path(args.checkpoint_metadata),
            accept=args.accept,
        )
        validate_terminal_result(artifact)
    output = Path(args.output)
    write_immutable_artifact(output, artifact)
    print(json.dumps({"output": str(output.resolve()), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
