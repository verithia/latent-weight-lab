from examples.nanogpt.seal_y400_matched_ns4_pair_vq_endpoint import classification


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
