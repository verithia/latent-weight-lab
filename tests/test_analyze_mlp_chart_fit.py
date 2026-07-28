import torch

from examples.nanogpt.analyze_mlp_chart_fit import (
    exact_block_fht_projection,
    resolved_latent_dim,
    shared_hidden_radial_projection,
)
from latent_weight_lab.block_fht import block_fht_slice


def test_exact_block_fht_projection_matches_explicit_projector() -> None:
    latent_dim, latent_shape = resolved_latent_dim(8, 0.25, 0)
    assert latent_dim == 2
    basis = []
    for column in range(latent_dim):
        latent = torch.zeros(latent_dim)
        latent[column] = 1.0
        basis.append(block_fht_slice(latent, 8, 2, 17, 0, 8))
    matrix = torch.stack(basis, dim=1)
    delta = torch.arange(8, dtype=torch.float32)
    explicit = matrix @ torch.linalg.lstsq(matrix, delta).solution
    observed = exact_block_fht_projection(
        delta,
        latent_dim=latent_dim,
        latent_shape=latent_shape,
        layers=2,
        seed=17,
    )
    assert torch.allclose(
        torch.tensor(observed["projection_energy_fraction"]),
        explicit.square().sum() / delta.square().sum(),
        atol=1e-6,
        rtol=1e-6,
    )


def test_shared_hidden_radial_projection_recovers_exact_gain() -> None:
    generator = torch.Generator().manual_seed(7)
    fc = torch.randn(6, 3, generator=generator)
    proj = torch.randn(3, 6, generator=generator)
    gain = torch.linspace(-0.2, 0.3, 6)
    metrics = shared_hidden_radial_projection(
        fc,
        gain[:, None] * fc,
        proj,
        proj * gain[None, :],
    )
    assert abs(metrics["projection_energy_fraction"] - 1.0) < 1e-6
    assert abs(metrics["gain_rms"] - float(gain.square().mean().sqrt())) < 1e-6
