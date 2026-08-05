#!/usr/bin/env python3
"""Causally attribute a dense-vs-generated attention endpoint gap.

The generated model is the backbone.  At every layer, its attention branch is
decomposed into three shape-compatible functions: causal probabilities, value
states, and output projection.  Each function can be supplied by either the
generated endpoint or a matched dense endpoint.  Evaluating all eight choices
gives an exact three-factor Shapley attribution in validation CE, while fixed
layer inputs give a matching normalized residual-output error attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_residual_compatibility import (
    fixed_validation_batches,
)
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.optimize_mlp_bilateral_endpoint_ce import (
    autocast_context,
    clear_frozen_base_cache,
    prepare_frozen_base_cache,
)


COMPONENTS = ("score", "value", "projection")


def load_endpoint_model(checkpoint_path: Path, device: str) -> GPT:
    """Load on CPU so deterministic CPU generators remain device-compatible."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def qkv(attn: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Materialize the two endpoint architectures used by this analysis."""
    bsz, seq_len, channels = x.shape
    if attn.qk_headwise_c_attn:
        qk_input = attn._apply_cayley_atlas(
            x,
            attn.qk_input_cayley,
            attn.qk_input_cayley_atlas,
            attn.active_cayley_atlas_stage,
        )
        value_input = attn._apply_cayley_atlas(
            x,
            attn.v_input_cayley,
            attn.v_input_cayley_atlas,
            attn.active_cayley_atlas_stage,
        )
        qk = attn.c_attn_qk_headwise(qk_input)
        qk = attn._apply_cayley_atlas(
            qk,
            attn.qk_output_cayley,
            attn.qk_output_cayley_atlas,
            attn.active_cayley_atlas_stage,
        )
        q, k = qk.split(attn.n_embd, dim=2)
        value = attn.c_attn_v(value_input)
        value = attn._apply_cayley_atlas(
            value,
            attn.v_output_cayley,
            attn.v_output_cayley_atlas,
            attn.active_cayley_atlas_stage,
        )
    elif attn.c_attn is not None:
        q, k, value = attn.c_attn(x).split(attn.n_embd, dim=2)
    else:
        raise ValueError(
            "endpoint attribution supports dense c_attn and qk-headwise "
            "generated attention only"
        )
    head_dim = channels // attn.n_head
    reshape = lambda value: value.view(  # noqa: E731
        bsz, seq_len, attn.n_head, head_dim
    ).transpose(1, 2)
    return reshape(q), reshape(k), reshape(value)


def causal_probabilities(attn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    q, k, _ = qkv(attn, x)
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    seq_len = x.shape[1]
    mask = torch.ones(
        (seq_len, seq_len), dtype=torch.bool, device=x.device
    ).tril()
    return F.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)


def value_states(attn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return qkv(attn, x)[2]


def project_attention(
    attn: torch.nn.Module, head_state: torch.Tensor
) -> torch.Tensor:
    bsz, heads, seq_len, head_dim = head_state.shape
    merged = head_state.transpose(1, 2).contiguous().view(
        bsz, seq_len, heads * head_dim
    )
    merged = attn._apply_cayley_atlas(
        merged,
        attn.cproj_input_cayley,
        attn.cproj_input_cayley_atlas,
        attn.active_cayley_atlas_stage,
    )
    output = attn.c_proj(merged)
    return attn._apply_cayley_atlas(
        output,
        attn.cproj_output_cayley,
        attn.cproj_output_cayley_atlas,
        attn.active_cayley_atlas_stage,
    )


def hybrid_attention(
    dense: torch.nn.Module,
    candidate: torch.nn.Module,
    x: torch.Tensor,
    mask: tuple[int, int, int],
) -> torch.Tensor:
    score_source = dense if mask[0] else candidate
    value_source = dense if mask[1] else candidate
    projection_source = dense if mask[2] else candidate
    probabilities = causal_probabilities(score_source, x)
    values = value_states(value_source, x)
    return project_attention(projection_source, probabilities @ values)


def mask_name(mask: tuple[int, int, int]) -> str:
    return "".join(str(value) for value in mask)


class HybridHooks:
    def __init__(
        self,
        dense: torch.nn.Module,
        candidate: torch.nn.Module,
        mask: tuple[int, int, int],
    ) -> None:
        self.handles = []
        for dense_block, candidate_block in zip(
            dense.transformer.h, candidate.transformer.h, strict=True
        ):
            dense_attn = dense_block.attn
            candidate_attn = candidate_block.attn

            def hook(_module, inputs, _output, d=dense_attn, c=candidate_attn):
                return hybrid_attention(d, c, inputs[0], mask)

            self.handles.append(candidate_attn.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def evaluate_ce(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    device: str,
) -> float:
    losses = []
    for tokens in batches:
        tokens = tokens.to(device)
        with autocast_context(device):
            _, loss = model(tokens[:, :-1], tokens[:, 1:])
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite or missing CE")
        losses.append(float(loss))
    return sum(losses) / len(losses)


def hybrid_ce(
    dense: torch.nn.Module,
    candidate: torch.nn.Module,
    batches: list[torch.Tensor],
    device: str,
    mask: tuple[int, int, int],
) -> float:
    hooks = HybridHooks(dense, candidate, mask)
    try:
        return evaluate_ce(candidate, batches, device)
    finally:
        hooks.close()


def shapley_improvements(
    losses: dict[tuple[int, int, int], float]
) -> dict[str, float]:
    """CE reduction assigned to dense substitution of each component."""
    result: dict[str, float] = {}
    n = len(COMPONENTS)
    for index, component in enumerate(COMPONENTS):
        contribution = 0.0
        for mask in losses:
            if mask[index]:
                continue
            size = sum(mask)
            weight = (
                math.factorial(size)
                * math.factorial(n - size - 1)
                / math.factorial(n)
            )
            added = list(mask)
            added[index] = 1
            contribution += weight * (losses[mask] - losses[tuple(added)])
        result[component] = contribution
    return result


def select_structural_gate(
    results: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    primary = results["primary"]
    confirmation = results["confirmation"]
    primary_shapley = primary["shapley_ce_improvement"]
    confirmation_shapley = confirmation["shapley_ce_improvement"]
    primary_top = max(COMPONENTS, key=lambda key: primary_shapley[key])
    confirmation_top = max(
        COMPONENTS, key=lambda key: confirmation_shapley[key]
    )
    minimum = float(protocol["minimum_stable_component_ce"])
    stable = (
        primary_top == confirmation_top
        and primary_shapley[primary_top] >= minimum
        and confirmation_shapley[confirmation_top] >= minimum
    )
    joint: dict[str, float] = {}
    for name, result in results.items():
        ce = result["hybrid_ce"]
        value = ce["000"] - ce["010"]
        projection = ce["000"] - ce["001"]
        value_projection = ce["000"] - ce["011"]
        joint[name] = value_projection - value - projection
    interaction_minimum = float(
        protocol["minimum_value_projection_interaction_ce"]
    )
    coupled = (
        not stable
        and joint["primary"] >= interaction_minimum
        and joint["confirmation"] >= interaction_minimum
    )
    if stable:
        classification = "STABLE_SINGLE_COMPONENT"
        selected = primary_top
    elif coupled:
        classification = "STABLE_VALUE_PROJECTION_INTERACTION"
        selected = "value_projection_coupling"
    else:
        classification = "NO_STABLE_ENDPOINT_COMPONENT"
        selected = None
    return {
        "classification": classification,
        "selected_component": selected,
        "primary_top_component": primary_top,
        "confirmation_top_component": confirmation_top,
        "minimum_stable_component_ce": minimum,
        "value_projection_interaction_ce": joint,
        "minimum_value_projection_interaction_ce": interaction_minimum,
        "rule": (
            "same top Shapley component with minimum CE improvement in both "
            "windows; otherwise admit a joint V/projection gate only when "
            "its super-additive interaction clears the fixed threshold in "
            "both windows"
        ),
        "automatic_training_authorized": False,
    }


@torch.no_grad()
def normalized_output_errors(
    dense: torch.nn.Module,
    candidate: torch.nn.Module,
    batches: list[torch.Tensor],
    device: str,
) -> dict[str, float]:
    """Compare layer attention outputs on candidate-backbone LN1 inputs."""
    totals = {mask: [0.0, 0.0] for mask in itertools.product((0, 1), repeat=3)}
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer, block in enumerate(candidate.transformer.h):
        def hook(_module, _inputs, output, index=layer):
            captured[index] = output.detach()
        handles.append(block.ln_1.register_forward_hook(hook))
    try:
        for tokens in batches:
            captured.clear()
            tokens = tokens.to(device)
            with autocast_context(device):
                candidate(tokens[:, :-1], None)
                for layer, (dense_block, candidate_block) in enumerate(zip(
                    dense.transformer.h, candidate.transformer.h, strict=True
                )):
                    x = captured[layer]
                    reference = hybrid_attention(
                        dense_block.attn, candidate_block.attn, x, (1, 1, 1)
                    ).float()
                    denominator = float(reference.square().sum())
                    for mask in totals:
                        output = hybrid_attention(
                            dense_block.attn, candidate_block.attn, x, mask
                        ).float()
                        totals[mask][0] += float((output - reference).square().sum())
                        totals[mask][1] += denominator
    finally:
        for handle in handles:
            handle.remove()
    return {
        mask_name(mask): numerator / max(denominator, 1e-30)
        for mask, (numerator, denominator) in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--dense-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != (
        "mai_124m_attention_dense_5tpp_replay_attribution_plan_v1"
    ):
        raise ValueError("unexpected attribution plan schema")
    manifest_sha256 = sha256(args.data_dir / "manifest.json")
    required_manifest = plan["replay"]["required_dataset_manifest_sha256"]
    if manifest_sha256 != required_manifest:
        raise ValueError("dataset manifest SHA-256 mismatch")
    protocol = plan["attribution"]
    dense_sha256 = sha256(args.dense_checkpoint)
    if dense_sha256 != protocol["required_dense_checkpoint_sha256"]:
        raise ValueError("dense checkpoint SHA-256 mismatch")
    candidate_sha256 = sha256(args.candidate_checkpoint)
    if candidate_sha256 != protocol["required_candidate_checkpoint_sha256"]:
        raise ValueError("candidate checkpoint SHA-256 mismatch")
    root = Path(__file__).resolve().parents[2]
    dense = load_endpoint_model(args.dense_checkpoint, args.device)
    candidate = load_endpoint_model(args.candidate_checkpoint, args.device)
    cached = prepare_frozen_base_cache(candidate, torch.bfloat16)
    windows = {
        name: fixed_validation_batches(
            args.data_dir,
            int(spec["batch_size"]),
            int(spec["block_size"]) + 1,
            int(spec["batches"]),
            int(spec["seed"]),
        )
        for name, spec in protocol["validation_windows"].items()
    }
    try:
        results: dict[str, Any] = {}
        for name, batches in windows.items():
            native_candidate = evaluate_ce(candidate, batches, args.device)
            native_dense = evaluate_ce(dense, batches, args.device)
            losses = {
                mask: hybrid_ce(dense, candidate, batches, args.device, mask)
                for mask in itertools.product((0, 1), repeat=3)
            }
            if abs(native_candidate - losses[(0, 0, 0)]) > float(
                protocol["maximum_native_manual_ce_delta"]
            ):
                raise RuntimeError(
                    f"{name} native/manual candidate CE disagreement"
                )
            shapley = shapley_improvements(losses)
            results[name] = {
                "token_sha256": tensor_sha256(torch.cat(batches)),
                "native_candidate_ce": native_candidate,
                "native_dense_ce": native_dense,
                "hybrid_ce": {
                    mask_name(mask): value for mask, value in losses.items()
                },
                "candidate_backbone_all_dense_attention_ce": losses[(1, 1, 1)],
                "attention_recoverable_ce": losses[(0, 0, 0)] - losses[(1, 1, 1)],
                "shapley_ce_improvement": shapley,
                "shapley_sum": sum(shapley.values()),
            }
        errors = normalized_output_errors(
            dense,
            candidate,
            windows[protocol["output_error_window"]][
                : int(protocol["output_error_batches"])
            ],
            args.device,
        )
    finally:
        clear_frozen_base_cache(candidate)
    decision = select_structural_gate(results, protocol)
    output = {
        "schema_version": "mai_124m_attention_endpoint_attribution_v1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": {
            "git_commit": git_commit(root),
            "entrypoint": "examples.nanogpt.analyze_attention_endpoint_attribution",
            "command": sys.argv,
            "plan": str(args.plan),
            "plan_sha256": sha256(args.plan),
            "dense_checkpoint": str(args.dense_checkpoint),
            "dense_checkpoint_sha256": dense_sha256,
            "candidate_checkpoint": str(args.candidate_checkpoint),
            "candidate_checkpoint_sha256": candidate_sha256,
            "dataset_manifest_sha256": manifest_sha256,
            "cached_block_fht_modules": cached,
            "device": args.device,
            "device_name": (
                torch.cuda.get_device_name(0)
                if args.device.startswith("cuda")
                else "cpu"
            ),
        },
        "protocol": protocol,
        "results": results,
        "normalized_attention_output_error": errors,
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
