from __future__ import annotations

import torch

from examples.nanogpt.analyze_sparse_moe_cfc_global_tt_oracle import (
    capped_bond_ranks,
    coordinate_count,
    dense_to_physical,
    materialize_expert_matrix,
    materialize_tt,
    physical_to_dense,
    randomized_tt_svd,
    result_authorization,
)


def test_registered_budget_is_exactly_over_200x() -> None:
    modes = [12, 8, 6, 3, 4, 4, 4, 4, 4, 4, 4, 4]
    ranks = capped_bond_ranks(modes, 150)
    assert ranks == [1, 12, 96, 150, 150, 150, 150, 150, 150, 64, 16, 4, 1]
    assert coordinate_count(modes, ranks) == 566028
    assert (12 * 8 * 1536 * 768) / coordinate_count(modes, ranks) > 200


def test_equal_budget_separated_control() -> None:
    interleaved = [12, 8, 6, 3, 4, 4, 4, 4, 4, 4, 4, 4]
    separated = [12, 8, 6, 4, 4, 4, 4, 3, 4, 4, 4, 4]
    assert coordinate_count(interleaved, capped_bond_ranks(interleaved, 150)) == (
        coordinate_count(separated, capped_bond_ranks(separated, 150))
    )


def test_physical_roundtrip_for_both_orders() -> None:
    generator = torch.Generator().manual_seed(3)
    dense = torch.randn(2, 3, 8, 6, generator=generator)
    for interleaved in (False, True):
        physical = dense_to_physical(
            dense, [2, 2, 2], [3, 2, 1], interleaved=interleaved
        )
        recovered = physical_to_dense(
            physical, [2, 2, 2], [3, 2, 1], interleaved=interleaved
        )
        assert torch.equal(recovered, dense)


def test_full_rank_tt_svd_reconstructs_and_slices_small_tensor() -> None:
    generator = torch.Generator().manual_seed(7)
    dense = torch.randn(2, 3, 4, 4, generator=generator)
    physical = dense_to_physical(dense, [2, 2], [2, 2], interleaved=True)
    modes = list(physical.shape)
    ranks = capped_bond_ranks(modes, 64)
    cores, diagnostics = randomized_tt_svd(
        physical,
        modes,
        ranks,
        seed=11,
        oversample=2,
        power_iterations=1,
    )
    reconstructed = physical_to_dense(
        materialize_tt(cores), [2, 2], [2, 2], interleaved=True
    )
    torch.testing.assert_close(reconstructed, dense, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        materialize_expert_matrix(
            cores, 1, 2, [2, 2], [2, 2], interleaved=True
        ),
        dense[1, 2],
        atol=2e-5,
        rtol=2e-5,
    )
    assert diagnostics


def test_truncated_tt_is_seeded_finite_and_compact() -> None:
    generator = torch.Generator().manual_seed(13)
    physical = torch.randn(3, 4, 5, 6, generator=generator)
    modes = list(physical.shape)
    ranks = capped_bond_ranks(modes, 3)
    left, _ = randomized_tt_svd(
        physical,
        modes,
        ranks,
        seed=17,
        oversample=1,
        power_iterations=1,
    )
    right, _ = randomized_tt_svd(
        physical,
        modes,
        ranks,
        seed=17,
        oversample=1,
        power_iterations=1,
    )
    assert sum(core.numel() for core in left) == coordinate_count(modes, ranks)
    assert all(torch.isfinite(core).all() for core in left)
    for first, second in zip(left, right):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_authorization_requires_separate_training_gate() -> None:
    passed = result_authorization(True)
    rejected = result_authorization(False)
    assert passed["implementation"]
    assert passed["initialization_fit_shadow"]
    assert not passed["language_model_training"]
    assert not passed["mfu_preflight"]
    assert not rejected["implementation"]
