#!/usr/bin/env python3
"""Localize four-bit pair-VQ fixed-CE error to c_fc or c_proj."""
from __future__ import annotations

import argparse
import json
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
from examples.nanogpt.analyze_mlp_cproj_bilateral_endpoint_fixed_eval import (
    evaluate_fixed_ce,
)
from examples.nanogpt.analyze_shared_mlp_endpoint_function import sha256_file
from examples.nanogpt.analyze_shared_mlp_exact_family_teacher_fit import (
    atomic_json,
    git_head,
)
from examples.nanogpt.train import (
    TokenBatchSource,
    fixed_eval_indices_digest,
    make_fixed_eval_indices,
)


PLAN_SCHEMA = "mai_pair_vq_side_localization_plan_v1"
RESULT_SCHEMA = "mai_pair_vq_side_localization_result_v1"


@torch.no_grad()
def decode_pair_vq_side(
    *,
    codebooks: Tensor,
    codes: Tensor,
    shapes: list[torch.Size],
    device: str,
) -> Tensor:
    decoded = []
    if int(codebooks.shape[0]) != len(shapes) or int(codes.shape[0]) != len(shapes):
        raise ValueError("pair-VQ layer count mismatch")
    for layer, shape in enumerate(shapes):
        vectors = codebooks[layer].to(device=device, dtype=torch.float32).index_select(
            0, codes[layer].to(device=device, dtype=torch.long)
        )
        if vectors.numel() != int(shape.numel()):
            raise ValueError("pair-VQ decoded shape mismatch")
        decoded.append(vectors.reshape(shape))
    return torch.stack(decoded)


def _dominance(
    *,
    c_fc_gap: float,
    c_proj_gap: float,
    both_gap: float,
) -> dict[str, Any]:
    threshold = 0.60 * both_gap
    fc = c_fc_gap >= 1.5 * c_proj_gap and c_fc_gap >= threshold
    proj = c_proj_gap >= 1.5 * c_fc_gap and c_proj_gap >= threshold
    dominant = "c_fc" if fc else "c_proj" if proj else "neither"
    interaction = both_gap - c_fc_gap - c_proj_gap
    return {
        "dominant_side": dominant,
        "c_fc_to_c_proj_gap_ratio": c_fc_gap / max(c_proj_gap, 1e-30),
        "c_proj_to_c_fc_gap_ratio": c_proj_gap / max(c_fc_gap, 1e-30),
        "dominance_minimum_gap": threshold,
        "both_minus_side_gap_sum": interaction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA or args.device != "cuda":
        raise ValueError("unexpected plan schema or device")
    causal = plan["causal_basis"]
    if sha256_file(Path(causal["pair_vq_result"])) != causal[
        "pair_vq_result_sha256"
    ]:
        raise ValueError("pair-VQ causal result identity mismatch")
    identity = plan["identities"]
    for key in ("dense_teacher_checkpoint", "pair_vq_state", "pair_vq_raw_result"):
        if sha256_file(Path(identity[key]["path"])) != identity[key]["sha256"]:
            raise ValueError(f"identity mismatch: {key}")
    data_dir = Path("/mnt/ssd-data/orj/MappingNetworks/data/finewebedu_20b")
    if sha256_file(data_dir / "manifest.json") != identity["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest identity mismatch")

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    teacher_path = Path(identity["dense_teacher_checkpoint"]["path"])
    teacher = load_model(teacher_path, args.device)
    blocks = list(teacher.transformer.h)
    fc_shapes = [block.mlp.c_fc.weight.shape for block in blocks]
    proj_shapes = [block.mlp.c_proj.weight.shape for block in blocks]
    dense_fc = torch.stack([block.mlp.c_fc.weight.detach().float() for block in blocks])
    dense_proj = torch.stack(
        [block.mlp.c_proj.weight.detach().float() for block in blocks]
    )
    pre_gain, output_log_gain = stack_optional_dense_gains(blocks)
    compact = torch.load(
        identity["pair_vq_state"]["path"], map_location="cpu", weights_only=False
    )
    quantized_fc = decode_pair_vq_side(
        codebooks=compact["c_fc_codebook"],
        codes=compact["c_fc_codes"],
        shapes=fc_shapes,
        device=args.device,
    )
    quantized_proj = decode_pair_vq_side(
        codebooks=compact["c_proj_codebook"],
        codes=compact["c_proj_codes"],
        shapes=proj_shapes,
        device=args.device,
    )

    measurement = plan["measurement"]
    fixed = make_fixed_eval_indices(
        data_dir,
        int(measurement["fixed_eval_batch_size"]),
        int(measurement["fixed_eval_block_size"]),
        int(measurement["fixed_eval_batches"]),
        int(measurement["fixed_eval_seed"]),
    )
    digest = fixed_eval_indices_digest(fixed)
    if digest != identity["fixed_eval_indices_sha256"]:
        raise ValueError("fixed evaluation digest mismatch")
    source = TokenBatchSource(data_dir)

    variants = {
        "dense": (dense_fc, dense_proj),
        "c_fc_only": (quantized_fc, dense_proj),
        "c_proj_only": (dense_fc, quantized_proj),
        "both": (quantized_fc, quantized_proj),
    }
    losses: dict[str, float] = {}
    for name, (c_fc, c_proj) in variants.items():
        model = load_model(teacher_path, args.device)
        if name != "dense":
            family = QuantizedDenseMLPFamily(
                c_fc=c_fc,
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
            eval_iters=int(measurement["fixed_eval_batches"]),
            eval_batch_size=int(measurement["fixed_eval_batch_size"]),
            block_size=int(measurement["fixed_eval_block_size"]),
            device=args.device,
            dtype="bfloat16",
            source=source,
        )
        print(json.dumps({"variant": name, "validation_ce": losses[name]}), flush=True)
        del model
        if name != "dense":
            del family

    gaps = {name: value - losses["dense"] for name, value in losses.items()}
    dominance = _dominance(
        c_fc_gap=gaps["c_fc_only"],
        c_proj_gap=gaps["c_proj_only"],
        both_gap=gaps["both"],
    )
    expected = json.loads(Path(identity["pair_vq_raw_result"]["path"]).read_text())
    rule = plan["frozen_decision_rule"]
    consistency = {
        "dense_absolute_error": abs(
            losses["dense"] - float(expected["teacher_validation_cross_entropy"])
        ),
        "both_gap_absolute_error": abs(
            gaps["both"] - float(expected["fixed_validation_cross_entropy_gap"])
        ),
    }
    consistency["passed"] = bool(
        consistency["dense_absolute_error"] <= 0.002
        and consistency["both_gap_absolute_error"] <= 0.002
    )
    if not consistency["passed"]:
        classification = "PAIR_VQ_SIDE_LOCALIZATION_INCONSISTENT"
    else:
        classification = (
            "PAIR_VQ_SIDE_LOCALIZATION_" + dominance["dominant_side"].upper()
        )
    result = {
        "schema_version": RESULT_SCHEMA,
        "classification": classification,
        "repository_commit": git_head(Path(__file__).resolve().parents[2]),
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "identities": identity,
        "fixed_eval_indices_sha256": digest,
        "validation_cross_entropy": losses,
        "validation_cross_entropy_gaps": gaps,
        "dominance": dominance,
        "consistency": consistency,
        "frozen_decision_rule": rule,
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
                "gaps": gaps,
                "dominance": dominance,
                "consistency": consistency,
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
