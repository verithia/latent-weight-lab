import torch

from examples.nanogpt.moe_paired_geometry import (
    apply_neuron_permutation,
    complete_expert_similarity,
    functional_atom_similarity,
    match_paired_neurons,
    maximum_weight_assignment,
)


def test_joint_neuron_match_recovers_exact_hidden_permutation() -> None:
    generator = torch.Generator().manual_seed(17)
    c_fc = torch.randn(7, 5, generator=generator)
    c_proj = torch.randn(5, 7, generator=generator)
    activations = torch.randn(128, 5, generator=generator)
    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    permuted_fc = c_fc.index_select(0, permutation)
    permuted_proj = c_proj.index_select(1, permutation)

    match = match_paired_neurons(
        c_fc,
        c_proj,
        permuted_fc,
        permuted_proj,
        activations,
    )
    aligned_fc, aligned_proj = apply_neuron_permutation(
        permuted_fc,
        permuted_proj,
        match.permutation,
    )
    assert torch.equal(aligned_fc, c_fc)
    assert torch.equal(aligned_proj, c_proj)
    assert match.mean_similarity > 0.99999


def test_functional_similarity_is_not_an_independent_weight_cosine() -> None:
    c_fc = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    c_proj = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    activations = torch.tensor([[2.0, 0.0], [-1.0, 0.0], [0.5, 0.0]])
    similarity = functional_atom_similarity(
        c_fc,
        c_proj,
        c_fc,
        c_proj,
        activations,
    )
    assert torch.allclose(similarity.diag(), torch.ones(2), atol=1e-6)
    assert similarity[0, 1] < 0.5


def test_complete_expert_similarity_recovers_joint_expert_permutation() -> None:
    generator = torch.Generator().manual_seed(23)
    router = torch.randn(3, 4, generator=generator)
    c_fc = torch.randn(3, 6, 4, generator=generator)
    c_proj = torch.randn(3, 4, 6, generator=generator)
    activations = torch.randn(64, 4, generator=generator)
    permutation = torch.tensor([2, 0, 1])
    similarity = complete_expert_similarity(
        router,
        c_fc,
        c_proj,
        router.index_select(0, permutation),
        c_fc.index_select(0, permutation),
        c_proj.index_select(0, permutation),
        activations,
    )
    assignment = maximum_weight_assignment(similarity)
    assert torch.equal(
        permutation.index_select(0, assignment),
        torch.arange(3),
    )
