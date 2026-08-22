#!/usr/bin/env python3
"""Test whether c_proj Pair-VQ fails because its codec is not neuron-major.

This is a zero-update representation oracle.  It projects the exact same dense
c_proj matrices through the production Pair-VQ weight/feedback budget in two
layouts:

* ``row_major`` pairs adjacent hidden channels for one residual coordinate;
* ``neuron_major`` transposes c_proj first, so pairs and FHT blocks stay inside
  one GELU neuron's residual-write vector.

No candidate state is installed in a training run.  A pass only authorizes a
production implementation followed by an exact-config MFU gate.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_layer_private_product_quantized_mlp import (
    InstalledQuantizedDenseMLP,
    QuantizedDenseMLPFamily,
    stack_optional_dense_gains,
)
from examples.nanogpt.analyze_mlp_activation_update_alignment import (
    ActivationCollector,
)
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_shared_mlp_endpoint_function import sha256_file
from examples.nanogpt.analyze_shared_mlp_exact_family_teacher_fit import (
    atomic_json,
    git_head,
)
from examples.nanogpt.muon_pair_vq import (
    MuonPairVQLinear,
    _decode_fractional_residual_lattice_feedback,
    _fit_fractional_residual_lattice_feedback_,
    _fractional_residual_lattice_feedback_layout,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    get_batch,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_124m_pair_vq_cproj_neuron_layout_oracle_plan_v1"
RESULT_SCHEMA = "mai_124m_pair_vq_cproj_neuron_layout_oracle_result_v1"


def matrix_to_pairs(weight: Tensor, *, layout: str) -> Tensor:
    if weight.ndim != 2 or weight.numel() % 2:
        raise ValueError("layout oracle expects an even rank-two matrix")
    if layout == "row_major":
        encoded = weight
    elif layout == "neuron_major":
        encoded = weight.T.contiguous()
    else:
        raise ValueError(f"unknown matrix layout: {layout}")
    return encoded.reshape(-1, 2)


def pairs_to_matrix(
    pairs: Tensor,
    *,
    shape: tuple[int, int],
    layout: str,
) -> Tensor:
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("decoded pairs must have shape [pairs, 2]")
    if pairs.numel() != math.prod(shape):
        raise ValueError("decoded pair count does not match matrix shape")
    if layout == "row_major":
        return pairs.reshape(shape)
    if layout == "neuron_major":
        return pairs.reshape(shape[1], shape[0]).T.contiguous()
    raise ValueError(f"unknown matrix layout: {layout}")


def action_metrics(hidden: Tensor, dense: Tensor, decoded: Tensor) -> dict[str, float]:
    if hidden.ndim != 2 or dense.ndim != 2 or decoded.shape != dense.shape:
        raise ValueError("invalid c_proj action shapes")
    target = hidden.float() @ dense.float().T
    candidate = hidden.float() @ decoded.float().T
    error = candidate - target
    target_energy = float(target.square().sum())
    candidate_energy = float(candidate.square().sum())
    error_energy = float(error.square().sum())
    dot = float((target * candidate).sum())
    return {
        "target_energy": target_energy,
        "candidate_energy": candidate_energy,
        "error_energy": error_energy,
        "energy_recovery": 1.0 - error_energy / max(target_energy, 1e-30),
        "cosine": dot
        / max(math.sqrt(target_energy * candidate_energy), 1e-30),
    }


def classify(
    *,
    bank_error_closures: list[float],
    layer_error_closures: list[float],
    dense_ce: float,
    row_major_ce: float,
    neuron_major_ce: float,
    equal_state_bytes: bool,
    thresholds: dict[str, float],
) -> tuple[str, dict[str, bool]]:
    row_gap = row_major_ce - dense_ce
    neuron_gap = neuron_major_ce - dense_ce
    gates = {
        "finite": all(
            math.isfinite(value)
            for value in (
                *bank_error_closures,
                *layer_error_closures,
                dense_ce,
                row_major_ce,
                neuron_major_ce,
            )
        ),
        "equal_state_bytes": bool(equal_state_bytes),
        "every_bank_action": min(bank_error_closures)
        >= float(thresholds["minimum_every_bank_action_error_closure"]),
        "every_layer_action": min(layer_error_closures)
        >= float(thresholds["minimum_every_layer_action_error_closure"]),
        "fixed_ce_improves": neuron_major_ce
        <= row_major_ce - float(thresholds["minimum_fixed_ce_improvement"]),
        "fixed_gap_closure": (
            1.0 - neuron_gap / max(row_gap, 1e-30)
            >= float(thresholds["minimum_fixed_ce_gap_closure"])
        ),
    }
    passed = all(gates.values())
    return (
        "CPROJ_NEURON_MAJOR_LAYOUT_AUTHORIZED"
        if passed
        else "CPROJ_NEURON_MAJOR_LAYOUT_REJECTED",
        gates,
    )


@torch.no_grad()
def reconstruct_cproj(
    weight: Tensor,
    *,
    layout: str,
    layer: int,
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, int | float]]:
    encoded = weight if layout == "row_major" else weight.T.contiguous()
    module = MuonPairVQLinear(
        int(encoded.shape[1]),
        int(encoded.shape[0]),
        bias=False,
        stages=1,
        base_seed=int(config["base_seed"]) + layer * 8192 + 4096,
        weight_std=float(config["weight_std"]),
        layer_id=layer,
        fast_residual=True,
        error_feedback=False,
        neighbor_candidates=int(config["neighbor_candidates"]),
        code_refresh_interval=int(config["code_refresh_interval"]),
    ).to(weight.device)
    for _sweep in range(int(config["projection_sweeps"])):
        module.project_requested_weight_(encoded, refresh_codes=True)

    base = module.weight.detach().float()
    residual_pairs = (encoded.float() - base).reshape(-1, 2)
    coordinate_bits = int(config["feedback_coordinate_bits"])
    block_size = int(config["feedback_block_size"])
    layout_spec = _fractional_residual_lattice_feedback_layout(
        int(encoded.numel()),
        coordinate_bits=coordinate_bits,
        block_size=block_size,
    )
    levels = torch.zeros(
        896 + (1 << coordinate_bits),
        device=weight.device,
        dtype=torch.float32,
    )
    packed = torch.zeros(
        layout_spec["total_bytes"],
        device=weight.device,
        dtype=torch.uint8,
    )
    # Hold the feedback transform seed fixed in logical c_proj coordinates so
    # the only changed property is matrix layout, not the random transform.
    feedback_seed = (
        int(config["feedback_seed_base"])
        + 8192 * layer
        + 131 * int(weight.shape[1])
        + 17 * int(weight.shape[0])
    )
    _fit_fractional_residual_lattice_feedback_(
        residual_pairs,
        levels,
        packed,
        seed=feedback_seed,
        coordinate_bits=coordinate_bits,
        block_size=block_size,
        lloyd_iterations=int(config["feedback_lloyd_iterations"]),
    )
    feedback = _decode_fractional_residual_lattice_feedback(
        levels,
        packed,
        element_count=int(encoded.numel()),
        seed=feedback_seed,
        coordinate_bits=coordinate_bits,
        block_size=block_size,
    ).reshape_as(encoded)
    decoded_encoded = base + feedback
    decoded = (
        decoded_encoded
        if layout == "row_major"
        else decoded_encoded.T.contiguous()
    )
    state = {
        "codec_bytes": int(module.persistent_codec_bytes),
        "feedback_bytes": int(
            packed.numel() * packed.element_size()
            + levels.numel() * levels.element_size()
        ),
        "persistent_model_bytes": int(
            module.persistent_codec_bytes
            + packed.numel() * packed.element_size()
            + levels.numel() * levels.element_size()
        ),
        "weight_energy_recovery": float(
            1.0
            - (decoded - weight.float()).square().sum()
            / weight.float().square().sum().clamp_min(1e-30)
        ),
    }
    del module, levels, packed, feedback, base, residual_pairs
    return decoded, state


@torch.no_grad()
def collect_hidden_bank(
    model,
    *,
    data_dir: Path,
    source: TokenBatchSource,
    indices: Tensor,
    batch_size: int,
    block_size: int,
    sample_cap: int,
    device: str,
) -> dict[int, Tensor]:
    layers = list(range(len(model.transformer.h)))
    collector = ActivationCollector(model, layers, sample_cap)
    try:
        x, _y = get_batch(
            data_dir,
            "val",
            batch_size,
            block_size,
            device,
            indices=indices,
            source=source,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(x, None)
        if not collector.complete():
            raise RuntimeError("activation bank did not reach the sample cap")
        return {
            layer: collector.tensor(layer, "post_gelu").to(device)
            for layer in layers
        }
    finally:
        collector.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA or args.device != "cuda":
        raise ValueError("unexpected plan schema or device")
    for item in plan["causal_results"]:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"causal result identity mismatch: {item['path']}")
    checkpoint = Path(plan["identities"]["dense_parent_checkpoint"]["path"])
    if sha256_file(checkpoint) != plan["identities"]["dense_parent_checkpoint"]["sha256"]:
        raise ValueError("dense parent checkpoint identity mismatch")
    data_dir = Path(plan["identities"]["data_dir"])
    if sha256_file(data_dir / "manifest.json") != plan["identities"]["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    teacher = load_model(checkpoint, args.device)
    blocks = list(teacher.transformer.h)
    dense_fc = torch.stack([block.mlp.c_fc.weight.detach().float() for block in blocks])
    dense_proj = torch.stack([block.mlp.c_proj.weight.detach().float() for block in blocks])
    pre_gain, output_log_gain = stack_optional_dense_gains(blocks)

    row_major, neuron_major = [], []
    state_rows: list[dict[str, Any]] = []
    layer_limit = 1 if args.preflight_only else len(blocks)
    for layer in range(layer_limit):
        row, row_state = reconstruct_cproj(
            dense_proj[layer],
            layout="row_major",
            layer=layer,
            config=plan["codec"],
        )
        neuron, neuron_state = reconstruct_cproj(
            dense_proj[layer],
            layout="neuron_major",
            layer=layer,
            config=plan["codec"],
        )
        row_major.append(row)
        neuron_major.append(neuron)
        state_rows.append(
            {"layer": layer, "row_major": row_state, "neuron_major": neuron_state}
        )
        print(json.dumps({"projected_layer": layer, **state_rows[-1]}), flush=True)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "complete",
                    "one_layer_two_layout_seconds": time.time() - started,
                    "projected_full_oracle_seconds": (time.time() - started) * len(blocks),
                    "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                },
                sort_keys=True,
            )
        )
        return

    row_major_tensor = torch.stack(row_major)
    neuron_major_tensor = torch.stack(neuron_major)
    fixed = make_fixed_eval_indices(
        data_dir,
        int(plan["measurement"]["batch_size"]),
        int(plan["measurement"]["block_size"]),
        int(plan["measurement"]["registered_fixed_eval_batches"]),
        int(plan["measurement"]["fixed_eval_seed"]),
    )
    digest = fixed_eval_indices_digest(fixed)
    if digest != plan["identities"]["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation inventory mismatch")
    source = TokenBatchSource(data_dir)

    banks: dict[str, Any] = {}
    layer_closures: list[float] = []
    for bank_index in plan["measurement"]["activation_bank_indices"]:
        hidden = collect_hidden_bank(
            teacher,
            data_dir=data_dir,
            source=source,
            indices=fixed["val"][int(bank_index)],
            batch_size=int(plan["measurement"]["batch_size"]),
            block_size=int(plan["measurement"]["block_size"]),
            sample_cap=int(plan["measurement"]["activation_sample_cap"]),
            device=args.device,
        )
        layers: dict[str, Any] = {}
        row_error = 0.0
        neuron_error = 0.0
        for layer in range(len(blocks)):
            row_metrics = action_metrics(
                hidden[layer], dense_proj[layer], row_major_tensor[layer]
            )
            neuron_metrics = action_metrics(
                hidden[layer], dense_proj[layer], neuron_major_tensor[layer]
            )
            closure = 1.0 - neuron_metrics["error_energy"] / max(
                row_metrics["error_energy"], 1e-30
            )
            row_error += row_metrics["error_energy"]
            neuron_error += neuron_metrics["error_energy"]
            layer_closures.append(closure)
            layers[str(layer)] = {
                "row_major": row_metrics,
                "neuron_major": neuron_metrics,
                "action_error_closure": closure,
            }
        banks[str(bank_index)] = {
            "layers": layers,
            "action_error_closure": 1.0 - neuron_error / max(row_error, 1e-30),
        }
        del hidden

    losses: dict[str, float] = {}
    for name, c_proj in (
        ("dense", dense_proj),
        ("row_major", row_major_tensor),
        ("neuron_major", neuron_major_tensor),
    ):
        model = load_model(checkpoint, args.device)
        if name != "dense":
            family = QuantizedDenseMLPFamily(
                c_fc=dense_fc,
                c_proj=c_proj,
                pre_gain=pre_gain,
                output_log_gain=output_log_gain,
            ).to(args.device)
            for layer in range(family.layers):
                model.transformer.h[layer].mlp = InstalledQuantizedDenseMLP(
                    family, layer
                )
        losses[name] = evaluate_fixed_ce(
            model,
            data_dir=data_dir,
            fixed_indices=fixed,
            split="val",
            eval_iters=int(plan["measurement"]["scored_fixed_eval_batches"]),
            eval_batch_size=int(plan["measurement"]["batch_size"]),
            block_size=int(plan["measurement"]["block_size"]),
            device=args.device,
            dtype="bfloat16",
            source=source,
        )
        del model
        if name != "dense":
            del family

    bank_closures = [value["action_error_closure"] for value in banks.values()]
    state_equal = all(
        row["row_major"]["persistent_model_bytes"]
        == row["neuron_major"]["persistent_model_bytes"]
        for row in state_rows
    )
    classification, gates = classify(
        bank_error_closures=bank_closures,
        layer_error_closures=layer_closures,
        dense_ce=losses["dense"],
        row_major_ce=losses["row_major"],
        neuron_major_ce=losses["neuron_major"],
        equal_state_bytes=state_equal,
        thresholds=plan["thresholds"],
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "passed": all(gates.values()),
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": plan["identities"],
        "codec": plan["codec"],
        "state": state_rows,
        "fixed_eval_indices_sha256": digest,
        "validation_cross_entropy": losses,
        "action_banks": banks,
        "minimum_layer_action_error_closure": min(layer_closures),
        "minimum_bank_action_error_closure": min(bank_closures),
        "gates": gates,
        "thresholds": plan["thresholds"],
        "authorization": (
            plan["authorization_on_pass"]
            if all(gates.values())
            else plan["authorization_on_fail"]
        ),
        "maximum_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "classification": classification,
                "losses": losses,
                "minimum_bank_action_error_closure": min(bank_closures),
                "minimum_layer_action_error_closure": min(layer_closures),
                "gates": gates,
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
