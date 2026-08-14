from __future__ import annotations

import torch

from examples.nanogpt.analyze_postbranch_fc_oracle import (
    apply_fc,
    fit_fc,
    normalized_recovery,
)


def test_fit_fc_recovers_known_linear_map() -> None:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(256, 8, generator=generator)
    feature_jvps = torch.randn(128, 8, generator=generator)
    target_matrix = torch.randn(8, 8, generator=generator)
    targets = apply_fc(features, target_matrix)
    target_jvps = apply_fc(feature_jvps, target_matrix)
    fitted, metrics = fit_fc(
        features,
        targets,
        feature_jvps,
        target_jvps,
        ridge_ratio=1e-8,
    )
    assert metrics["ridge"] > 0.0
    assert normalized_recovery(
        apply_fc(features, fitted), targets
    ) > 0.99999
    assert torch.allclose(
        fitted, target_matrix, atol=2e-4, rtol=2e-4
    )


def test_apply_fc_identity_is_exact() -> None:
    values = torch.randn(4, 7)
    identity = torch.eye(7)
    assert torch.equal(apply_fc(values, None), values)
    assert torch.equal(apply_fc(values, identity), values)
