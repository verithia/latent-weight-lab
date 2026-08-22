from examples.nanogpt.seal_pair_vq_lazy_retraction_5tpp import classification


def test_lazy_retraction_5tpp_classification_is_fail_closed() -> None:
    assert (
        classification(False, False, False)
        == "INVALID_LAZY_RETRACTION_124M_5TPP"
    )
    assert (
        classification(True, True, True)
        == "PASS_LAZY_RETRACTION_124M_5TPP"
    )
    assert (
        classification(True, False, False)
        == "REJECT_LAZY_RETRACTION_124M_5TPP"
    )
    assert (
        classification(True, True, False)
        == "REJECT_LAZY_RETRACTION_124M_5TPP"
    )
