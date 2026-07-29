import pytest
import torch

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.optimize_mlp_pregelu_frame_endpoint_ce import (
    CERTIFICATE_SCHEMA,
    capture_pregelu_state,
    install_pregelu_frames,
    select_decision,
    validate_mfu_certificate,
)


def make_rows(
    values: dict[str, tuple[float, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, (baseline, selected) in values.items():
        rows.extend(
            [
                {"update": 0, "split": split, "ce": baseline},
                {"update": 120, "split": split, "ce": selected},
            ]
        )
    return rows


def test_select_decision_requires_marginal_gain_on_three_splits() -> None:
    rows = make_rows(
        {
            "primary": (5.7, 5.69),
            "confirmation": (5.8, 5.79),
            "audit": (5.9, 5.89),
        }
    )
    result = select_decision(rows, 0.005)
    assert result["selected_update"] == 120
    assert result["decision"] == "POSITIVE_PRE_GELU_FRAME_CAPACITY"

    rows[-1]["ce"] = 5.898
    result = select_decision(rows, 0.005)
    assert result["decision"] == "REJECT_PRE_GELU_FRAME_CAPACITY"


def test_select_decision_rejects_identity_as_primary_winner() -> None:
    rows = make_rows(
        {
            "primary": (5.7, 5.71),
            "confirmation": (5.8, 5.79),
            "audit": (5.9, 5.89),
        }
    )
    result = select_decision(rows, 0.005)
    assert result["selected_update"] == 0
    assert result["decision"] == "REJECT_PRE_GELU_FRAME_CAPACITY"


def test_install_pregelu_frames_is_identity_and_freezes_base() -> None:
    torch.manual_seed(314)
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=1,
            n_embd=8,
            bias=False,
        )
    )
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    baseline = model(tokens)[0].detach()
    parameters = install_pregelu_frames(
        model,
        stages=2,
        rotation_block_size=4,
        basis_block_size=8,
        basis_seed=41,
        per_layer_seed_offset=64,
        coordinate_scale=4.0,
    )
    torch.testing.assert_close(model(tokens)[0], baseline)
    assert set(parameters) == {
        "layer.0.pregelu_rotation",
        "layer.1.pregelu_rotation",
    }
    assert all(parameter.requires_grad for parameter in parameters.values())
    selected = {id(parameter) for parameter in parameters.values()}
    assert all(
        parameter.requires_grad == (id(parameter) in selected)
        for parameter in model.parameters()
    )
    state = capture_pregelu_state(model)
    assert set(state) == set(parameters)
    assert all(torch.count_nonzero(value) == 0 for value in state.values())


def test_validate_mfu_certificate_is_identity_bound() -> None:
    identity = {"identity_sha256": "abc"}
    certificate = {
        "schema_version": CERTIFICATE_SCHEMA,
        "identity": identity,
        "measurement": {"mfu_fraction": 0.21},
        "stability": {
            "ce_increase": 0.01,
            "maximum_ce_increase": 0.1,
        },
        "passed": True,
    }
    validate_mfu_certificate(certificate, identity, 0.2)
    with pytest.raises(ValueError, match="does not match"):
        validate_mfu_certificate(
            certificate, {"identity_sha256": "def"}, 0.2
        )
    certificate["measurement"]["mfu_fraction"] = 0.19
    with pytest.raises(ValueError, match="failed"):
        validate_mfu_certificate(certificate, identity, 0.2)
