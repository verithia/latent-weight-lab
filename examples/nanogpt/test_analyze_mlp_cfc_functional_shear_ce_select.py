from examples.nanogpt.analyze_mlp_cfc_functional_shear_ce_select import select_beta
from examples.nanogpt.analyze_mlp_cfc_functional_shear_radius import scale_name


def test_select_beta_chooses_lowest_loss() -> None:
    rows = []
    for beta, loss in ((0.125, 5.0), (0.25, 4.9), (0.375, 4.95)):
        rows.extend(
            {"candidate": scale_name(beta, prefix="functional_mix"), "loss": loss}
            for _ in range(3)
        )
    selected, summaries = select_beta(
        rows,
        betas=[0.125, 0.25, 0.375],
        tie_tolerance=1e-8,
    )
    assert selected == 0.25
    assert summaries[scale_name(0.25, prefix="functional_mix")]["mean"] == 4.9


def test_select_beta_uses_smallest_within_tolerance() -> None:
    rows = []
    for beta, loss in ((0.125, 5.0), (0.25, 5.0 - 5e-9)):
        rows.extend(
            {"candidate": scale_name(beta, prefix="functional_mix"), "loss": loss}
            for _ in range(3)
        )
    selected, _ = select_beta(
        rows,
        betas=[0.125, 0.25],
        tie_tolerance=1e-8,
    )
    assert selected == 0.125
