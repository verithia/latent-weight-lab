#!/usr/bin/env python3
"""Independent production-shape replay for the hierarchical B64/L16 fitter."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.muon_pair_vq import (
    _decode_fractional_lattice_feedback,
    _decode_fractional_residual_lattice_feedback,
    _fit_fractional_residual_lattice_feedback_,
    _fit_fractional_residual_lattice_feedback_batch_,
    _fit_scalar_codebook,
    _fit_scalar_codebooks_batched,
    _fractional_lattice_feedback_layout,
    _fractional_lattice_feedback_segments,
    _fractional_residual_lattice_feedback_layout,
    _signed_block_fht,
    _unpack_fixed_width_codes,
)


def assigned_codes(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(
        values.contiguous(), (levels[:-1] + levels[1:]) * 0.5
    )


def role_difference(
    values: list[torch.Tensor],
    serial_levels: list[torch.Tensor],
    candidate_levels: list[torch.Tensor],
) -> dict[str, float | int]:
    max_centroid = 0.0
    mismatches = 0
    total = 0
    for source, reference, candidate in zip(
        values, serial_levels, candidate_levels, strict=True
    ):
        max_centroid = max(
            max_centroid, float((reference - candidate).abs().max())
        )
        reference_codes = assigned_codes(source, reference)
        candidate_codes = assigned_codes(source, candidate)
        mismatches += int((reference_codes != candidate_codes).sum())
        total += source.numel()
    return {
        "centroid_max_abs": max_centroid,
        "code_mismatches": mismatches,
        "code_mismatch_fraction": mismatches / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("production-shape integration replay requires CUDA")
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    element_count = 2_359_296
    entries: list[dict[str, Any]] = []
    serial_entries: list[dict[str, Any]] = []
    for index in range(24):
        is_cfc = index % 2 == 0
        coordinate_bits = 5 if is_cfc else 4
        block_size = 64 if is_cfc else 32
        iterations = 16 if is_cfc else 4
        seed = 20261025 + 8192 * (index // 2) + 131 * (
            768 if is_cfc else 3072
        ) + 17 * (3072 if is_cfc else 768)
        generator = torch.Generator(device="cuda").manual_seed(20261101 + index)
        vectors = (
            torch.randn(
                element_count // 2,
                2,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
            * (0.001 + 0.00002 * index)
        ).contiguous()
        layout = _fractional_residual_lattice_feedback_layout(
            element_count,
            coordinate_bits=coordinate_bits,
            block_size=block_size,
        )
        level_shape = 896 + (1 << coordinate_bits)
        entries.append(
            {
                "vectors": vectors,
                "levels": torch.zeros(level_shape, device="cuda"),
                "packed": torch.zeros(
                    layout["total_bytes"], device="cuda", dtype=torch.uint8
                ),
                "seed": seed,
                "coordinate_bits": coordinate_bits,
                "block_size": block_size,
                "lloyd_iterations": iterations,
            }
        )
        serial_entries.append(
            {
                **{key: value for key, value in entries[-1].items() if key not in {"levels", "packed"}},
                "levels": torch.zeros(level_shape, device="cuda"),
                "packed": torch.zeros(
                    layout["total_bytes"], device="cuda", dtype=torch.uint8
                ),
            }
        )

    serial_changes = []
    for entry in serial_entries:
        serial_changes.append(
            _fit_fractional_residual_lattice_feedback_(
                entry["vectors"],
                entry["levels"],
                entry["packed"],
                seed=entry["seed"],
                coordinate_bits=entry["coordinate_bits"],
                block_size=entry["block_size"],
                lloyd_iterations=entry["lloyd_iterations"],
            )
        )
    candidate_changes = _fit_fractional_residual_lattice_feedback_batch_(entries)

    role_values: dict[str, list[torch.Tensor]] = {
        "base_gains": [],
        "base_coordinates": [],
        "refined_coordinates": [],
        "cfc_residual_gains": [],
        "cfc_residual_coordinates": [],
        "cproj_residual_gains": [],
        "cproj_residual_coordinates": [],
    }
    role_serial_levels: dict[str, list[torch.Tensor]] = {
        role: [] for role in role_values
    }
    role_candidate_levels: dict[str, list[torch.Tensor]] = {
        role: [] for role in role_values
    }
    selectors_valid = True
    decoded_finite = True
    maximum_decoded_relative_l2 = 0.0
    maximum_decoded_absolute = 0.0
    shapes_and_dtypes_exact = True
    packed_streams_valid = True
    for serial, candidate in zip(serial_entries, entries, strict=True):
        transformed = _signed_block_fht(
            serial["vectors"], block_size=32, seed=serial["seed"]
        )
        gains = transformed.square().mean(dim=1).sqrt().clamp_min(1e-30)
        serial_gain_codes = assigned_codes(gains.log(), serial["levels"][:256])
        decoded_gains = serial["levels"][:256].index_select(
            0, serial_gain_codes
        ).exp()
        normalized = (transformed / decoded_gains[:, None]).reshape(-1)
        role_values["base_gains"].append(gains.log())
        role_values["base_coordinates"].append(normalized)
        role_values["refined_coordinates"].append(normalized)
        role_serial_levels["base_gains"].append(serial["levels"][:256])
        role_serial_levels["base_coordinates"].append(serial["levels"][256:384])
        role_serial_levels["refined_coordinates"].append(serial["levels"][384:640])
        role_candidate_levels["base_gains"].append(candidate["levels"][:256])
        role_candidate_levels["base_coordinates"].append(candidate["levels"][256:384])
        role_candidate_levels["refined_coordinates"].append(candidate["levels"][384:640])

        coordinate_bits = serial["coordinate_bits"]
        block_size = serial["block_size"]
        layout = _fractional_residual_lattice_feedback_layout(
            element_count,
            coordinate_bits=coordinate_bits,
            block_size=block_size,
        )
        serial_fractional_base = _decode_fractional_lattice_feedback(
            serial["levels"][:640],
            serial["packed"][: layout["base_bytes"]],
            element_count=element_count,
            seed=serial["seed"],
        )
        serial_decoded = _decode_fractional_residual_lattice_feedback(
            serial["levels"],
            serial["packed"],
            element_count=element_count,
            seed=serial["seed"],
            coordinate_bits=coordinate_bits,
            block_size=block_size,
        )
        residual = _signed_block_fht(
            serial["vectors"] - serial_fractional_base,
            block_size=block_size,
            seed=serial["seed"],
        )
        residual_gains = residual.square().mean(dim=1).sqrt().clamp_min(1e-30)
        residual_gain_codes = assigned_codes(
            residual_gains.log(), serial["levels"][640:896]
        )
        residual_decoded_gains = serial["levels"][640:896].index_select(
            0, residual_gain_codes
        ).exp()
        residual_normalized = (
            residual / residual_decoded_gains[:, None]
        ).reshape(-1)
        prefix = "cfc" if coordinate_bits == 5 else "cproj"
        role_values[f"{prefix}_residual_gains"].append(residual_gains.log())
        role_values[f"{prefix}_residual_coordinates"].append(residual_normalized)
        role_serial_levels[f"{prefix}_residual_gains"].append(
            serial["levels"][640:896]
        )
        role_serial_levels[f"{prefix}_residual_coordinates"].append(
            serial["levels"][896 : 896 + (1 << coordinate_bits)]
        )
        role_candidate_levels[f"{prefix}_residual_gains"].append(
            candidate["levels"][640:896]
        )
        role_candidate_levels[f"{prefix}_residual_coordinates"].append(
            candidate["levels"][896 : 896 + (1 << coordinate_bits)]
        )

        candidate_decoded = _decode_fractional_residual_lattice_feedback(
            candidate["levels"],
            candidate["packed"],
            element_count=element_count,
            seed=candidate["seed"],
            coordinate_bits=coordinate_bits,
            block_size=block_size,
        )
        decoded_finite = decoded_finite and bool(
            torch.isfinite(serial_decoded).all()
            and torch.isfinite(candidate_decoded).all()
        )
        delta = candidate_decoded - serial_decoded
        maximum_decoded_absolute = max(
            maximum_decoded_absolute, float(delta.abs().max())
        )
        maximum_decoded_relative_l2 = max(
            maximum_decoded_relative_l2,
            float(
                delta.square().sum().sqrt()
                / serial_decoded.square().sum().sqrt().clamp_min(1e-30)
            ),
        )
        base_layout = _fractional_lattice_feedback_layout(element_count)
        for packed in (serial["packed"], candidate["packed"]):
            base_stop = layout["base_bytes"]
            _, _, selector_stream, _ = _fractional_lattice_feedback_segments(
                packed[:base_stop], element_count=element_count
            )
            selector = _unpack_fixed_width_codes(
                selector_stream, bits=1, count=base_layout["block_count"]
            )
            selectors_valid = selectors_valid and int(selector.sum()) == base_layout[
                "selected_blocks"
            ]
        shapes_and_dtypes_exact = shapes_and_dtypes_exact and (
            candidate["levels"].shape == serial["levels"].shape
            and candidate["levels"].dtype == torch.float32
            and candidate["packed"].shape == serial["packed"].shape
            and candidate["packed"].dtype == torch.uint8
        )
        packed_streams_valid = packed_streams_valid and (
            candidate["packed"].numel() == layout["total_bytes"]
        )

    end_to_end_roles = {
        role: role_difference(
            role_values[role],
            role_serial_levels[role],
            role_candidate_levels[role],
        )
        for role in role_values
    }
    direct_candidate_levels: dict[str, list[torch.Tensor]] = {}
    serial_exact_roles = {
        "base_gains",
        "refined_coordinates",
        "cfc_residual_gains",
        "cproj_residual_gains",
    }
    for role, values in role_values.items():
        if role in serial_exact_roles:
            outputs = [
                _fit_scalar_codebook(
                    source,
                    level_count=role_serial_levels[role][index].numel(),
                    iterations=(16 if role == "cfc_residual_gains" else 4),
                )
                for index, source in enumerate(values)
            ]
            direct_candidate_levels[role] = [output[0] for output in outputs]
        else:
            source_matrix = torch.stack(values)
            level_count = role_serial_levels[role][0].numel()
            iterations = 16 if role == "cfc_residual_coordinates" else 4
            fitted, _codes = _fit_scalar_codebooks_batched(
                source_matrix,
                level_count=level_count,
                iterations=iterations,
                hierarchical=role != "base_coordinates",
            )
            direct_candidate_levels[role] = list(fitted)
    roles = {
        role: role_difference(
            role_values[role],
            role_serial_levels[role],
            direct_candidate_levels[role],
        )
        for role in role_values
    }
    scalar_correctness = all(
        result["centroid_max_abs"] <= 0.00015
        and result["code_mismatch_fraction"] <= 0.0001
        for result in roles.values()
    )
    result = {
        "schema_version": "mai_124m_pair_vq_hierarchical_integration_replay_v1",
        "recorded_at": "2026-08-26",
        "source_commit": commit,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(),
        "roles": roles,
        "end_to_end_role_drift": end_to_end_roles,
        "scalar_correctness_passed": scalar_correctness,
        "persistent_state_shapes_and_dtypes_exact": shapes_and_dtypes_exact,
        "selectors_valid": selectors_valid,
        "packed_streams_valid": packed_streams_valid,
        "decoded_outputs_finite": decoded_finite,
        "maximum_decoded_relative_l2_vs_serial": maximum_decoded_relative_l2,
        "maximum_decoded_absolute_vs_serial": maximum_decoded_absolute,
        "serial_code_changes": serial_changes,
        "candidate_code_changes": candidate_changes,
        "gate_passed": bool(
            scalar_correctness
            and shapes_and_dtypes_exact
            and selectors_valid
            and packed_streams_valid
            and decoded_finite
        ),
        "automatic_training": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
