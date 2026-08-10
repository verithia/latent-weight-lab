from examples.nanogpt.watch_remote_moe_paired_acquisition import (
    parse_steps,
    snapshot_stem,
)


def test_steps_and_snapshot_identity_are_canonical() -> None:
    assert parse_steps("9495,2374,4748,2374") == [2374, 4748, 9495]
    assert snapshot_stem(2374) == "step_002374_moe_paired_l0_l5_l11"
