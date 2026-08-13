import json
from pathlib import Path

import pytest
import torch

from examples.nanogpt.analyze_sparse_moe_paired_fisher_krylov_decoder_oracle import (
    FisherKrylovDecoder,
    LatentTuple,
    PairedEmpiricalFisher,
    PairedTensor,
    ProceduralPairedMap,
    coordinate_accounting,
    fit_decoder,
    latent_dot,
    pair_dot,
    route_and_sample_with_probabilities,
)
from examples.nanogpt.analyze_sparse_moe_rolling_tangent_oracle import LayerState


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "examples/nanogpt/configs/selection_artifacts/124m_sparse_moe_paired_fisher_krylov_decoder_oracle_plan.json"


def tiny_pair(seed: int = 7) -> PairedTensor:
    generator = torch.Generator().manual_seed(seed)
    return PairedTensor(
        torch.randn(2, 5, 3, generator=generator) * 0.1,
        torch.randn(2, 3, 5, generator=generator) * 0.1,
    )


def tiny_fisher(seed: int = 11) -> PairedEmpiricalFisher:
    generator = torch.Generator().manual_seed(seed)
    return PairedEmpiricalFisher(
        torch.randn(2, 9, 3, generator=generator),
        torch.rand(2, 9, generator=generator).mul_(0.5).add_(0.5),
        tiny_pair(seed + 1),
        device="cpu",
    )


def tiny_plan() -> dict:
    plan = json.loads(PLAN_PATH.read_text())
    plan["source"].update(
        {"tensor_layers": 1, "layers": [0], "num_experts": 2,
         "input_width": 3, "expert_hidden_width": 5}
    )
    plan["candidate"]["coordinate_split_by_polynomial_order"] = [3, 2, 2]
    return plan


def test_registered_coordinate_accounting_is_exactly_above_200x() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    counts = coordinate_accounting(plan)
    assert counts["per_expert"] == 11776
    assert counts["per_layer"] == 94208
    assert counts["compact"] == 1130496
    assert counts["dense"] == 226492416
    assert counts["compression"] == pytest.approx(200.34782608695653)
    amendment = plan["implementation_protocol_amendment_before_implementation_or_candidate_values"]
    assert amendment["procedural_map"]["block_fht_layers"] == 3
    assert amendment["linear_solver"]["maximum_iterations"] == 64


def test_paired_fisher_jvp_vjp_are_adjoint_and_fisher_is_psd() -> None:
    fisher = tiny_fisher()
    left = tiny_pair(19)
    generator = torch.Generator().manual_seed(23)
    cotangent = torch.randn(2, 9, 3, generator=generator)
    jvp_vjp_error = (fisher.jvp(left) * cotangent).sum() - pair_dot(
        left, fisher.vjp(cotangent)
    )
    assert float(jvp_vjp_error.abs()) < 2e-5
    right = tiny_pair(29)
    symmetry = pair_dot(left, fisher.apply(right)) - pair_dot(fisher.apply(left), right)
    assert float(symmetry.abs()) < 2e-5
    assert float(pair_dot(left, fisher.apply(left))) >= -1e-6


def test_procedural_map_adjoint_and_registered_normalization() -> None:
    mapping = ProceduralPairedMap(
        experts=1, input_width=3, hidden_width=4, latent_width=3,
        layers=3, seed=41, layer=0, bank_index=0,
        polynomial_order=0, device="cpu",
    )
    coordinates = torch.randn(1, 3, generator=torch.Generator().manual_seed(43))
    cotangent = tiny_pair(47)
    cotangent = PairedTensor(cotangent.c_fc[:1, :4], cotangent.c_proj[:1, :, :4])
    left = pair_dot(mapping.apply(coordinates), cotangent)
    right = (coordinates * mapping.adjoint(cotangent)).sum()
    torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)

    energy = 0.0
    for coordinate in range(3):
        basis = torch.zeros(1, 3)
        basis[0, coordinate] = 1.0
        image = mapping.apply(basis)
        energy += float(pair_dot(image, image))
    assert energy / 3.0 == pytest.approx(1.0, rel=2e-6, abs=2e-6)


@pytest.mark.parametrize("candidate", [False, True])
def test_krylov_decoder_adjoint(candidate: bool) -> None:
    plan = tiny_plan()
    fisher = tiny_fisher(53)
    fisher.estimate_largest_eigenvalue(
        tiny_pair(59), seed=61, iterations=8, epsilon=1e-8
    )
    decoder = FisherKrylovDecoder(
        plan, fisher, bank_index=0, layer=0,
        candidate=candidate, device="cpu",
    )
    generator = torch.Generator().manual_seed(67)
    coordinates = LatentTuple(tuple(
        torch.randn(2, width, generator=generator) for width in decoder.widths
    ))
    cotangent = tiny_pair(71)
    left = pair_dot(decoder.apply(coordinates), cotangent)
    right = latent_dot(coordinates, decoder.adjoint(cotangent))
    torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-5)


def test_identity_preconditioned_cg_reduces_registered_metric_error() -> None:
    plan = tiny_plan()
    fisher = tiny_fisher(73)
    fisher.estimate_largest_eigenvalue(
        tiny_pair(79), seed=83, iterations=8, epsilon=1e-8
    )
    decoder = FisherKrylovDecoder(
        plan, fisher, bank_index=0, layer=0,
        candidate=False, device="cpu",
    )
    generator = torch.Generator().manual_seed(89)
    true_coordinates = LatentTuple(tuple(
        torch.randn(2, width, generator=generator) * 0.1
        for width in decoder.widths
    ))
    target = decoder.apply(true_coordinates)
    _solution, diagnostics = fit_decoder(
        decoder, target, relative_damping=1e-6,
        maximum_iterations=64, tolerance=0.05, trace_seed=97,
    )
    assert diagnostics["relative_normal_residual"] <= 0.05
    assert diagnostics["fisher_metric_recovery"] > 0.9


def test_routed_sampling_preserves_selected_probability() -> None:
    state = LayerState(
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]),
        torch.zeros(3, 4, 2),
        torch.zeros(3, 2, 4),
    )
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    ).repeat(8, 1)
    sampled, probabilities, counts = route_and_sample_with_probabilities(
        state, inputs, top_k=2, samples_per_expert=4, seed=101
    )
    assert sampled.shape == (3, 4, 2)
    assert probabilities.shape == (3, 4)
    assert min(counts) >= 4
    assert bool(((probabilities > 0.0) & (probabilities < 1.0)).all())
