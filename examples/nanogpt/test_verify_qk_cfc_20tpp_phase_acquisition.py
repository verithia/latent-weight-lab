from pathlib import Path

import pytest

from examples.nanogpt.verify_qk_cfc_20tpp_phase_acquisition import (
    parse_logged_losses,
)


REQUIRED = [0, 2373, 4746, 7119, 9489]


def test_parse_logged_losses_uses_registered_20tpp_steps(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            f"step {step}: train loss {4.0 - index * 0.1:.4f}, "
            f"val loss {4.1 - index * 0.1:.4f}"
            for index, step in enumerate(REQUIRED)
        )
        + "\n"
    )
    parsed = parse_logged_losses(log, REQUIRED)
    assert sorted(parsed) == REQUIRED
    assert parsed[9489]["val"] == pytest.approx(3.7)


def test_parse_logged_losses_fails_closed_on_missing_registered_step(
    tmp_path: Path,
) -> None:
    log = tmp_path / "train.log"
    log.write_text("step 0: train loss 10.0, val loss 10.1\n")
    with pytest.raises(ValueError, match="2373"):
        parse_logged_losses(log, REQUIRED)
