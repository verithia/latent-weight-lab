import torch

from examples.nanogpt.analyze_mlp_cfc_output_general import (
    output_general_components,
)


def test_output_general_components_reconstruct_arbitrary_residual() -> None:
    generator = torch.Generator().manual_seed(31)
    weight = torch.randn(12, 5, generator=generator)
    residual = torch.randn(12, 5, generator=generator) * 1e-3
    components, diagnostics, spectra = output_general_components(
        weight, residual
    )
    assert diagnostics["general_reconstruction_error"] < 2e-5
    assert diagnostics["component_reconstruction_error"] < 2e-5
    assert diagnostics["skew_shear_bilateral_error"] < 2e-5
    assert torch.allclose(
        components["output_general"], residual, atol=2e-7, rtol=2e-5
    )
    assert torch.allclose(
        components["output_skew"] + components["output_symmetric"],
        residual,
        atol=2e-7,
        rtol=2e-5,
    )
    assert set(spectra) == {"general", "skew", "symmetric"}


def test_skew_and_symmetric_components_have_expected_operator_symmetry() -> None:
    generator = torch.Generator().manual_seed(37)
    weight = torch.randn(10, 4, generator=generator)
    residual = torch.randn(10, 4, generator=generator) * 1e-4
    u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    left_factor = residual @ vh.T / singular[None, :]
    operator = left_factor @ u.T
    expected_skew = 0.5 * (operator - operator.T) @ weight
    expected_symmetric = 0.5 * (operator + operator.T) @ weight
    components, _diagnostics, _spectra = output_general_components(
        weight, residual
    )
    assert torch.allclose(
        components["output_skew"], expected_skew, atol=2e-7, rtol=2e-5
    )
    assert torch.allclose(
        components["output_symmetric"],
        expected_symmetric,
        atol=2e-7,
        rtol=2e-5,
    )
