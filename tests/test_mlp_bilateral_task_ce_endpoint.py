from argparse import Namespace

import pytest
import torch

from examples.nanogpt.optimize_mlp_bilateral_endpoint_ce import (
    indices_digest,
    protocol_identity,
    select_decision,
    validate_mfu_certificate,
)


def test_indices_digest_is_shape_and_order_sensitive() -> None:
    values = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long)
    assert indices_digest(values) == indices_digest(values.clone())
    assert indices_digest(values) != indices_digest(values.flip(-1))
    assert indices_digest(values) != indices_digest(values.reshape(2, 2))


def test_select_decision_requires_both_validation_splits() -> None:
    rows = [
        {"update": 0, "split": "primary", "ce": 5.7},
        {"update": 30, "split": "primary", "ce": 5.68},
        {"update": 0, "split": "confirmation", "ce": 5.8},
        {"update": 30, "split": "confirmation", "ce": 5.79},
    ]
    selected = select_decision(rows, 0.005)
    assert selected["selected_update"] == 30
    assert selected["decision"] == (
        "POSITIVE_TASK_CONDITIONED_CHART_CAPACITY"
    )

    rows[-1]["ce"] = 5.799
    selected = select_decision(rows, 0.005)
    assert selected["decision"] == "REJECT_BILATERAL_CHART_TASK_CAPACITY"


def test_select_decision_rejects_identity_as_best() -> None:
    rows = [
        {"update": 0, "split": "primary", "ce": 5.7},
        {"update": 30, "split": "primary", "ce": 5.71},
        {"update": 0, "split": "confirmation", "ce": 5.8},
        {"update": 30, "split": "confirmation", "ce": 5.79},
    ]
    selected = select_decision(rows, 0.005)
    assert selected["selected_update"] == 0
    assert selected["decision"] == "REJECT_BILATERAL_CHART_TASK_CAPACITY"


def test_validate_mfu_certificate_is_identity_bound() -> None:
    identity = {"identity_sha256": "abc"}
    certificate = {
        "schema_version": "mai_124m_mlp_bilateral_task_ce_mfu_v1",
        "identity": identity,
        "measurement": {"mfu_fraction": 0.21},
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


def test_protocol_identity_hash_covers_update_shape(
    tmp_path, monkeypatch
) -> None:
    plan_dir = (
        tmp_path
        / "examples/nanogpt/configs/selection_artifacts"
    )
    plan_dir.mkdir(parents=True)
    (plan_dir / "124m_mlp_bilateral_task_ce_endpoint_plan.json").write_text(
        "{}"
    )
    source = tmp_path / "examples/nanogpt"
    source.mkdir(parents=True, exist_ok=True)
    fake_script = source / "optimize_mlp_bilateral_endpoint_ce.py"
    fake_script.write_text("# test")
    monkeypatch.setattr(
        "examples.nanogpt.optimize_mlp_bilateral_endpoint_ce.__file__",
        str(fake_script),
    )
    monkeypatch.setattr(
        "examples.nanogpt.optimize_mlp_bilateral_endpoint_ce.git_head",
        lambda root: "commit",
    )
    args = Namespace(
        initial_output_log_gain=0.0,
        batch_size=32,
        gradient_accumulation_steps=8,
        block_size=1024,
        learning_rate=0.00072,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.0,
    )
    first = protocol_identity(
        args,
        root=tmp_path,
        checkpoint_sha256="checkpoint",
        manifest_sha256="manifest",
        device_name="gpu",
    )
    args.batch_size = 16
    second = protocol_identity(
        args,
        root=tmp_path,
        checkpoint_sha256="checkpoint",
        manifest_sha256="manifest",
        device_name="gpu",
    )
    assert first["identity_sha256"] != second["identity_sha256"]
