#!/usr/bin/env python3
"""Attribute a compact pair-VQ endpoint to forward state or shadow carry.

This is a terminal-checkpoint acquisition.  It does not train, refit, or
change codes.  The native compact weight W and decoded feedback E are
evaluated as W, W + E_fc, W + E_proj, and W + E on the same fixed windows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.model import GPT, GPTConfig, MultiOptimizer
from examples.nanogpt.muon_pair_vq import MuonPairVQ, MuonPairVQLinear
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_pair_vq_shadow_endpoint_plan_v1"
RESULT_SCHEMA = "mai_pair_vq_shadow_endpoint_result_v1"
REFRESH_PREFIX = "muon_matched_givens_refresh "
EVAL_RE = re.compile(
    r"^step (?P<step>\d+): train loss (?P<train>[0-9.eE+-]+), "
    r"val loss (?P<val>[0-9.eE+-]+)$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def parse_training_log(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]], list[dict[str, float]]]:
    refresh: list[dict[str, Any]] = []
    evaluations: dict[int, dict[str, float]] = {}
    performance: list[dict[str, float]] = []
    for raw in path.read_text().splitlines():
        if raw.startswith(REFRESH_PREFIX):
            refresh.append(json.loads(raw[len(REFRESH_PREFIX) :]))
            continue
        match = EVAL_RE.match(raw)
        if match:
            step = int(match.group("step"))
            evaluations[step] = {
                "train": float(match.group("train")),
                "val": float(match.group("val")),
            }
            continue
        if raw.startswith("perf iter="):
            row: dict[str, float] = {}
            for token in raw.split()[1:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
            performance.append(row)
    if not refresh:
        raise ValueError("training log contains no pair-VQ refresh diagnostics")
    return refresh, evaluations, performance


def _weighted_recovery(rows: list[dict[str, Any]], numerator: str, denominator: str) -> float:
    error = sum(float(row[numerator]) for row in rows)
    energy = sum(float(row[denominator]) for row in rows)
    return 1.0 - error / max(energy, 1e-30)


def summarize_refreshes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(int(row["optimizer_step"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for step in sorted(by_step):
        group = by_step[step]
        if len(group) != 24:
            raise ValueError(f"refresh {step} has {len(group)} records, expected 24")
        codec_weighted = _weighted_recovery(
            group, "feedback_quantization_residual_energy", "feedback_target_energy"
        )
        current_weighted = _weighted_recovery(
            group, "feedback_quantization_residual_energy", "current_request_energy"
        )
        realized_weighted = _weighted_recovery(
            group, "projection_residual_energy", "request_energy"
        )
        feedback_energy = sum(float(row["feedback_energy"]) for row in group)
        inferred_weight_energy = sum(
            float(row["feedback_energy"])
            / max(float(row["feedback_to_weight_energy_ratio"]), 1e-30)
            for row in group
            if float(row["feedback_to_weight_energy_ratio"]) > 0.0
        )
        summaries.append(
            {
                "step": step,
                "records": len(group),
                "codec_weighted_recovery": codec_weighted,
                "codec_worst_matrix_recovery": min(
                    float(row["feedback_codec_energy_recovery"]) for row in group
                ),
                "current_weighted_recovery": current_weighted,
                "current_worst_matrix_recovery": min(
                    float(row["conserved_requested_step_energy_recovery"])
                    for row in group
                ),
                "realized_weighted_recovery": realized_weighted,
                "realized_worst_matrix_recovery": min(
                    float(row["requested_step_energy_recovery"]) for row in group
                ),
                "feedback_energy": feedback_energy,
                "feedback_to_weight_energy_ratio": (
                    feedback_energy / max(inferred_weight_energy, 1e-30)
                ),
                "c_fc_current_worst": min(
                    float(row["conserved_requested_step_energy_recovery"])
                    for row in group
                    if int(row["out_features"]) > int(row["in_features"])
                ),
                "c_proj_current_worst": min(
                    float(row["conserved_requested_step_energy_recovery"])
                    for row in group
                    if int(row["out_features"]) < int(row["in_features"])
                ),
                "c_fc_realized_weighted": _weighted_recovery(
                    [
                        row
                        for row in group
                        if int(row["out_features"]) > int(row["in_features"])
                    ],
                    "projection_residual_energy",
                    "request_energy",
                ),
                "c_proj_realized_weighted": _weighted_recovery(
                    [
                        row
                        for row in group
                        if int(row["out_features"]) < int(row["in_features"])
                    ],
                    "projection_residual_energy",
                    "request_energy",
                ),
            }
        )
    return {
        "rows": summaries,
        "refresh_count": len(summaries),
        "minimum_codec_weighted_recovery": min(
            row["codec_weighted_recovery"] for row in summaries
        ),
        "minimum_codec_matrix_recovery": min(
            row["codec_worst_matrix_recovery"] for row in summaries
        ),
        "minimum_current_weighted_recovery": min(
            row["current_weighted_recovery"] for row in summaries
        ),
        "minimum_current_matrix_recovery": min(
            row["current_worst_matrix_recovery"] for row in summaries
        ),
        "minimum_realized_weighted_recovery": min(
            row["realized_weighted_recovery"] for row in summaries
        ),
        "terminal": summaries[-1],
    }


def performance_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    usable = [row for row in rows if row.get("iter", 0.0) >= 10.0]
    tokens = [row["tokens_per_s"] for row in usable if "tokens_per_s" in row]
    fractions = [
        row["opt_ms"] / row["iter_ms"]
        for row in usable
        if row.get("iter_ms", 0.0) > 0.0 and "opt_ms" in row
    ]
    peaks = [row["peak_mib"] for row in rows if "peak_mib" in row]
    return {
        "median_tokens_per_second": statistics.median(tokens),
        "median_optimizer_fraction": statistics.median(fractions),
        "maximum_peak_mib": max(peaks),
    }


def pair_optimizer(optimizer: MultiOptimizer) -> MuonPairVQ:
    matches = [item for item in optimizer.optimizers if isinstance(item, MuonPairVQ)]
    if len(matches) != 1:
        raise ValueError(f"expected one MuonPairVQ optimizer, found {len(matches)}")
    return matches[0]


def decoded_feedback(
    model: GPT, optimizer: MuonPairVQ
) -> dict[str, torch.Tensor]:
    decoded: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, MuonPairVQLinear):
            continue
        state = optimizer.state[module.weight]
        feedback = module.decode_feedback(
            state["feedback_levels"],
            state["feedback_codes"],
            state.get("feedback_center"),
        ).reshape_as(module.weight)
        decoded[name] = feedback.detach().float()
    if len(decoded) != 24:
        raise ValueError(f"expected 24 pair-VQ matrices, found {len(decoded)}")
    return decoded


@torch.no_grad()
def install_variant(
    model: GPT,
    base: dict[str, torch.Tensor],
    feedback: dict[str, torch.Tensor],
    variant: str,
) -> None:
    modules = dict(model.named_modules())
    for name, source in base.items():
        module = modules[name]
        assert isinstance(module, MuonPairVQLinear)
        module.weight.copy_(source)
        use = (
            variant == "full_shadow"
            or (variant == "c_fc_shadow" and name.endswith(".c_fc"))
            or (variant == "c_proj_shadow" and name.endswith(".c_proj"))
        )
        if use:
            module.weight.add_(feedback[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unexpected shadow endpoint plan schema")
    if args.device != "cuda":
        raise ValueError("registered acquisition requires CUDA")
    identities = plan["identities"]
    for key in ("checkpoint", "checkpoint_metadata", "training_log", "status"):
        item = identities[key]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"identity mismatch: {key}")
    repo_root = Path(__file__).resolve().parents[2]
    for key in ("source_config", "endpoint_plan", "parent_result", "gate_result"):
        item = identities[key]
        if sha256_file(repo_root / item["path"]) != item["sha256"]:
            raise ValueError(f"identity mismatch: {key}")
    manifest = Path(plan["measurement"]["data_dir"]) / "manifest.json"
    if sha256_file(manifest) != identities["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")
    status = json.loads(Path(identities["status"]["path"]).read_text())
    if status.get("state") != "finished" or int(status.get("exit_code", -1)) != 0:
        raise ValueError("endpoint did not finish cleanly")

    refresh_raw, evaluations, perf_raw = parse_training_log(
        Path(identities["training_log"]["path"])
    )
    trajectory = summarize_refreshes(refresh_raw)
    performance = performance_summary(perf_raw)
    source_config = json.loads((repo_root / identities["source_config"]["path"]).read_text())
    gates = source_config["endpoint_gate"]
    direction_gates = {
        "codec_weighted_pass": trajectory["minimum_codec_weighted_recovery"]
        >= float(gates["feedback_codec_weighted_recovery_min_every_logged_refresh"]),
        "codec_matrix_pass": trajectory["minimum_codec_matrix_recovery"]
        >= float(gates["feedback_codec_every_matrix_recovery_min_every_logged_refresh"]),
        "current_weighted_pass": trajectory["minimum_current_weighted_recovery"]
        >= float(gates["current_step_conservation_weighted_min_every_logged_refresh"]),
        "current_matrix_pass": trajectory["minimum_current_matrix_recovery"]
        >= float(gates["current_step_conservation_every_matrix_min_every_logged_refresh"]),
    }

    checkpoint = torch.load(
        identities["checkpoint"]["path"], map_location="cpu", weights_only=False
    )
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()
    optimizer = model.configure_optimizers(
        float(source_config["weight_decay"]),
        float(source_config["learning_rate"]),
        (float(source_config["beta1"]), float(source_config["beta2"])),
        "cuda",
        optimizer=str(source_config["optimizer"]),
        muon_momentum=float(source_config["muon_momentum"]),
        muon_ns_steps=int(source_config["muon_ns_steps"]),
        muon_adamw_lr_scale=float(source_config["muon_adamw_lr_scale"]),
    )
    assert isinstance(optimizer, MultiOptimizer)
    optimizer.load_state_dict(checkpoint["optimizer"])
    pair = pair_optimizer(optimizer)
    feedback = decoded_feedback(model, pair)
    modules = dict(model.named_modules())
    base = {
        name: modules[name].weight.detach().float().clone()
        for name in feedback
    }
    carry_rows = []
    for name in sorted(feedback):
        side = "c_fc" if name.endswith(".c_fc") else "c_proj"
        carry_energy = float(feedback[name].square().sum())
        weight_energy = float(base[name].square().sum())
        carry_rows.append(
            {
                "module": name,
                "side": side,
                "carry_energy": carry_energy,
                "weight_energy": weight_energy,
                "carry_to_weight_energy_ratio": carry_energy
                / max(weight_energy, 1e-30),
            }
        )

    measurement = plan["measurement"]
    fixed = make_fixed_eval_indices(
        Path(measurement["data_dir"]),
        int(measurement["eval_batch_size"]),
        int(measurement["block_size"]),
        int(measurement["eval_iters"]),
        int(measurement["eval_seed"]),
    )
    digest = fixed_eval_indices_digest(fixed)
    if digest != identities["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation digest mismatch")
    source = TokenBatchSource(Path(measurement["data_dir"]))
    validation: dict[str, float] = {}
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for variant in ("native", "c_fc_shadow", "c_proj_shadow", "full_shadow"):
        install_variant(model, base, feedback, variant)
        validation[variant] = evaluate_fixed_ce(
            model,
            data_dir=Path(measurement["data_dir"]),
            fixed_indices=fixed,
            split="val",
            eval_iters=int(measurement["eval_iters"]),
            eval_batch_size=int(measurement["eval_batch_size"]),
            block_size=int(measurement["block_size"]),
            device=args.device,
            dtype="bfloat16",
            source=source,
        )
        print(json.dumps({"variant": variant, "validation_ce": validation[variant]}), flush=True)
    install_variant(model, base, feedback, "native")

    terminal_log_ce = float(evaluations[max(evaluations)]["val"])
    parent_ce = float(plan["comparison"]["parent_validation_ce"])
    frozen = plan["frozen_gates"]
    native_reproduced = abs(validation["native"] - terminal_log_ce) <= float(
        frozen["native_reproduction_absolute_tolerance"]
    )
    full_gain = validation["native"] - validation["full_shadow"]
    full_gap = validation["full_shadow"] - parent_ce
    useful = full_gain >= float(frozen["minimum_causal_shadow_ce_improvement"])
    sufficient = full_gap <= float(frozen["maximum_full_shadow_parent_ce_gap"])
    if sufficient:
        classification = "SHADOW_CARRY_SUFFICIENT_BUT_NOT_REALIZED"
        next_action = (
            "Design a compact forward-shadow training gate and a terminal folding test; "
            "do not count the temporal carry as inference compression unless it can be folded."
        )
    elif useful:
        classification = "SHADOW_CARRY_USEFUL_BUT_INSUFFICIENT"
        next_action = (
            "Run a synchronized dense-shadow/projection replay to separate gradient-path "
            "drift from terminal projection capacity before changing the codec."
        )
    else:
        classification = "SHADOW_CARRY_NOT_FUNCTIONALLY_CAUSAL"
        next_action = (
            "Run a synchronized dense-shadow/projection replay; more carry bits or forward "
            "carry realization is not authorized."
        )
    result = {
        "schema_version": RESULT_SCHEMA,
        "recorded_at": time.time(),
        "classification": classification,
        "repository_commit": git_head(repo_root),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identities,
        "terminal": {
            "status": status,
            "fixed_evaluations": {str(k): v for k, v in sorted(evaluations.items())},
            "native_minus_parent_ce": terminal_log_ce - parent_ce,
            "loss_gate_passed": terminal_log_ce - parent_ce
            <= float(gates["terminal_candidate_minus_parent_ce_max"]),
        },
        "trajectory": trajectory,
        "direction_gates": direction_gates,
        "performance": performance,
        "carry": {
            "matrices": carry_rows,
            "aggregate_carry_to_weight_energy_ratio": sum(
                row["carry_energy"] for row in carry_rows
            )
            / max(sum(row["weight_energy"] for row in carry_rows), 1e-30),
        },
        "fixed_validation": {
            **validation,
            "terminal_log_ce": terminal_log_ce,
            "native_reproduced": native_reproduced,
            "full_shadow_improvement_ce": full_gain,
            "full_shadow_parent_gap_ce": full_gap,
            "c_fc_shadow_improvement_ce": validation["native"]
            - validation["c_fc_shadow"],
            "c_proj_shadow_improvement_ce": validation["native"]
            - validation["c_proj_shadow"],
            "fixed_eval_indices_sha256": digest,
        },
        "frozen_gate_outcomes": {
            "native_reproduced": native_reproduced,
            "shadow_functionally_useful": useful,
            "shadow_sufficient_to_close_parent_gate": sufficient,
        },
        "next_action": next_action,
        "wall_seconds": time.time() - started,
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    if not native_reproduced or not all(math.isfinite(v) for v in validation.values()):
        raise ValueError("fixed evaluation did not reproduce the registered endpoint")
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "classification": classification,
                "validation": validation,
                "full_shadow_improvement_ce": full_gain,
                "full_shadow_parent_gap_ce": full_gap,
                "result": str(args.output),
                "result_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
