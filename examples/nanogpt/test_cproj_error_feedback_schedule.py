from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from examples.nanogpt import train


class DummyOptimizer:
    def __init__(self) -> None:
        self.param_groups = [
            {
                "error_feedback": True,
                "error_feedback_decay": 1.0,
                "cproj_error_feedback_decay_schedule": True,
            },
            {
                "error_feedback": True,
                "error_feedback_decay": 0.75,
            },
        ]


def test_replay_boundary_is_preserved_exactly() -> None:
    fraction = 120.0 / 238.0
    assert train.scheduled_cproj_error_feedback_decay(
        iter_num=119,
        max_iters=238,
        decay_before=1.0,
        decay_after=0.5,
        switch_fraction=fraction,
    ) == 1.0
    assert train.scheduled_cproj_error_feedback_decay(
        iter_num=120,
        max_iters=238,
        decay_before=1.0,
        decay_after=0.5,
        switch_fraction=fraction,
    ) == 0.5


def test_schedule_scales_with_horizon_and_only_updates_tagged_group() -> None:
    optimizer = DummyOptimizer()
    fraction = 120.0 / 238.0
    assert train.apply_cproj_error_feedback_decay_schedule(
        optimizer,
        iter_num=4,
        max_iters=9,
        decay_before=1.0,
        decay_after=0.5,
        switch_fraction=fraction,
    ) == 1.0
    assert optimizer.param_groups[0]["error_feedback_decay"] == 1.0
    assert optimizer.param_groups[1]["error_feedback_decay"] == 0.75
    assert train.apply_cproj_error_feedback_decay_schedule(
        optimizer,
        iter_num=5,
        max_iters=9,
        decay_before=1.0,
        decay_after=0.5,
        switch_fraction=fraction,
    ) == 0.5
    assert optimizer.param_groups[0]["error_feedback_decay"] == 0.5
    assert optimizer.param_groups[1]["error_feedback_decay"] == 0.75


def test_resume_recomputes_decay_from_iteration() -> None:
    optimizer = DummyOptimizer()
    optimizer.param_groups[0]["error_feedback_decay"] = 1.0
    train.apply_cproj_error_feedback_decay_schedule(
        optimizer,
        iter_num=180,
        max_iters=238,
        decay_before=1.0,
        decay_after=0.5,
        switch_fraction=120.0 / 238.0,
    )
    assert optimizer.param_groups[0]["error_feedback_decay"] == 0.5


@pytest.mark.parametrize(
    ("after", "fraction"),
    ((0.5, None), (None, 0.5), (-0.1, 0.5), (1.1, 0.5), (0.5, 0.0), (0.5, 1.0)),
)
def test_invalid_schedule_arguments_are_rejected(after, fraction) -> None:
    argv = (
        [
            "train.py",
            "--method",
            "block_fht",
            "--optimizer",
            "muon",
            "--block-fht-targets",
            "mlp.c_proj",
            "--block-fht-mlp-cproj-muon-matched-givens",
            "--block-fht-mlp-cproj-muon-matched-givens-error-feedback",
        ]
        + ([] if after is None else [
            "--block-fht-mlp-cproj-muon-matched-givens-error-feedback-decay-after",
            str(after),
        ])
        + ([] if fraction is None else [
            "--block-fht-mlp-cproj-muon-matched-givens-error-feedback-switch-fraction",
            str(fraction),
        ])
    )
    with patch.object(sys, "argv", argv), pytest.raises(ValueError):
        train.parse_args()


def test_schedule_requires_error_feedback() -> None:
    argv = [
        "train.py",
        "--method",
        "block_fht",
        "--data-dir",
        "/tmp/data",
        "--out-dir",
        "/tmp/out",
        "--optimizer",
        "muon",
        "--block-fht-targets",
        "mlp.c_proj",
        "--block-fht-mlp-cproj-muon-matched-givens",
        "--block-fht-mlp-cproj-muon-matched-givens-error-feedback-decay-after",
        "0.5",
        "--block-fht-mlp-cproj-muon-matched-givens-error-feedback-switch-fraction",
        str(120.0 / 238.0),
    ]
    with patch.object(sys, "argv", argv), pytest.raises(ValueError):
        train.parse_args()


@pytest.mark.parametrize("value", ("0", "-1", "nan", "inf"))
def test_feedback_nominal_step_cap_rejects_invalid_values(value: str) -> None:
    argv = [
        "train.py",
        "--method",
        "block_fht",
        "--optimizer",
        "muon",
        "--block-fht-targets",
        "mlp.c_proj",
        "--block-fht-mlp-cproj-muon-matched-givens",
        "--block-fht-mlp-cproj-muon-matched-givens-error-feedback",
        "--block-fht-mlp-cproj-muon-matched-givens-error-feedback-max-nominal-steps",
        value,
    ]
    with patch.object(sys, "argv", argv), pytest.raises(ValueError):
        train.parse_args()


def test_feedback_nominal_step_cap_requires_error_feedback() -> None:
    argv = [
        "train.py",
        "--method",
        "block_fht",
        "--optimizer",
        "muon",
        "--block-fht-targets",
        "mlp.c_proj",
        "--block-fht-mlp-cproj-muon-matched-givens",
        "--block-fht-mlp-cproj-muon-matched-givens-error-feedback-max-nominal-steps",
        "192",
    ]
    with patch.object(sys, "argv", argv), pytest.raises(ValueError):
        train.parse_args()


def test_feedback_nominal_step_cap_parses_when_enabled() -> None:
    argv = [
        "train.py",
        "--method",
        "block_fht",
        "--data-dir",
        "/tmp/data",
        "--out-dir",
        "/tmp/out",
        "--optimizer",
        "muon",
        "--block-fht-targets",
        "mlp.c_proj",
        "--block-fht-mlp-cproj-muon-matched-givens",
        "--block-fht-mlp-cproj-muon-matched-givens-error-feedback",
        "--block-fht-mlp-cproj-muon-matched-givens-error-feedback-max-nominal-steps",
        "192",
    ]
    with patch.object(sys, "argv", argv):
        args = train.parse_args()
    assert (
        args.block_fht_mlp_cproj_muon_matched_givens_error_feedback_max_nominal_steps
        == 192.0
    )
