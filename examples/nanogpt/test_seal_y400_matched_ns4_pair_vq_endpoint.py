from examples.nanogpt.seal_y400_matched_ns4_pair_vq_endpoint import (
    classification,
    json_equivalent,
)


def test_matched_ns4_pair_vq_classification_is_fail_closed() -> None:
    assert (
        classification(False, False)
        == "INVALID_MATCHED_NS4_PAIR_VQ_124M_0P5TPP"
    )
    assert (
        classification(True, False)
        == "REJECT_MATCHED_NS4_PAIR_VQ_124M_0P5TPP"
    )
    assert (
        classification(True, True)
        == "PASS_MATCHED_NS4_PAIR_VQ_124M_0P5TPP"
    )


def test_checkpoint_and_json_sidecar_container_types_are_equivalent() -> None:
    assert json_equivalent({"betas": (0.9, 0.95)}, {"betas": [0.9, 0.95]})
