import torch

from examples.nanogpt.analyze_mlp_cfc_generator_spectrum import (
    compact_generator_cosine,
    compressed_left_generator,
    generator_inner,
    orbit_generator_coordinates,
    spectrum_metrics,
)


def _skew(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(size, size, generator=generator)
    return 0.5 * (values - values.T)


def test_orbit_coordinates_reconstruct_bilateral_tangent() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(12, 5, generator=generator)
    left = _skew(12, 11)
    right = _skew(5, 13)
    residual = left @ weight + weight @ right
    compact = orbit_generator_coordinates(weight, residual)
    assert float(compact["reconstruction_error"]) < 2e-5
    assert torch.allclose(
        compact["bilateral_update"], residual, atol=2e-5, rtol=2e-5
    )


def test_compressed_left_spectrum_matches_materialized_generator() -> None:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(10, 4, generator=generator)
    residual = torch.randn(10, 4, generator=generator)
    u, _s, _vh = torch.linalg.svd(weight, full_matrices=False)
    compact = orbit_generator_coordinates(weight, residual)
    core = compact["left_core"]
    perpendicular = compact["left_perpendicular"]
    materialized = (
        u @ core @ u.T
        + perpendicular @ u.T
        - u @ perpendicular.T
    )
    compressed = compressed_left_generator(core, perpendicular)
    expected = torch.linalg.svdvals(materialized)
    observed = torch.linalg.svdvals(compressed)
    assert torch.allclose(
        expected[: observed.numel()], observed, atol=2e-5, rtol=2e-5
    )


def test_compact_inner_matches_materialized_inner() -> None:
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(11, 5, generator=generator)
    u, _s, _vh = torch.linalg.svd(weight, full_matrices=False)
    first = orbit_generator_coordinates(
        weight, torch.randn(11, 5, generator=generator)
    )
    second = orbit_generator_coordinates(
        weight, torch.randn(11, 5, generator=generator)
    )

    def materialize(values: dict[str, torch.Tensor]) -> torch.Tensor:
        core = values["left_core"]
        perpendicular = values["left_perpendicular"]
        return u @ core @ u.T + perpendicular @ u.T - u @ perpendicular.T

    expected = (materialize(first) * materialize(second)).sum()
    observed = generator_inner(
        first["left_core"],
        first["left_perpendicular"],
        second["left_core"],
        second["left_perpendicular"],
    )
    assert torch.allclose(expected, observed, atol=2e-5, rtol=2e-5)
    assert abs(compact_generator_cosine(first, first, "left") - 1.0) < 1e-6
    assert abs(compact_generator_cosine(first, first, "right") - 1.0) < 1e-6


def test_spectrum_metrics_report_skew_pairs_and_energy_ranks() -> None:
    values = torch.tensor([4.0, 4.0, 2.0, 2.0, 1.0, 1.0])
    result = spectrum_metrics(values)
    assert result["rank50"] == 2
    assert result["rank90"] == 4
    assert result["paired_singular_max_relative_error"] == 0.0
    assert abs(float(result["stable_rank"]) - 2.625) < 1e-12
