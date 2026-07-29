from __future__ import annotations

import torch

from examples.nanogpt.analyze_mlp_blockfht_polar_projection import (
    aggregate_rows,
    production_cproj_geometry,
    project_blockfht_tangent,
)
from latent_weight_lab.block_fht import block_fht_slice


def test_projection_is_exact_on_generator_range() -> None:
    latent = torch.tensor([0.25, -0.75])
    size = 4
    direction = block_fht_slice(latent, size, 1, 17, 0, size).reshape(2, 2)
    projected = project_blockfht_tangent(
        direction,
        latent_shape=(2,),
        size=size,
        layers=1,
        seed=17,
    )
    torch.testing.assert_close(projected, direction)


def test_registered_geometry_and_decision() -> None:
    geometry = production_cproj_geometry(
        {
            "n_embd": 768,
            "block_fht_latent_ratio": 0.01,
            "block_fht_latent_ratios": None,
            "block_fht_muon_latent_targets": ["mlp.c_proj"],
            "block_fht_muon_latent_rows": 154,
            "block_fht_layers": 2,
            "block_fht_seed": 1000,
        }
    )
    assert geometry["latent_shape"] == (154, 154)
    assert geometry["complete_blocks"] == 72
    rows = [
        {
            "target_chord_fro": 1.0,
            "raw_positive_step_line_recovery": 0.16,
            "projected_positive_step_line_recovery": 0.11,
            "projected_cosine": 0.2,
            "applied_direction_energy_retained": 0.01,
            "projector_idempotence_relative_error": 1e-6,
        }
        for _ in range(20)
    ]
    result = aggregate_rows(rows)
    assert result["admitted"]
    assert result["positive_projected_cells"] == 20
