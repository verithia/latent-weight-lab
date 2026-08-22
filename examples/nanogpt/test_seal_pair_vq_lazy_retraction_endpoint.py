from examples.nanogpt.seal_pair_vq_lazy_retraction_endpoint import classification


def test_lazy_retraction_classification_is_fail_closed() -> None:
    assert (
        classification(False, 5.0, 5.411)
        == "INVALID_LAZY_RETRACTION_124M_0P5TPP"
    )
    assert (
        classification(True, 5.369017124176025, 5.411)
        == "PASS_LAZY_RETRACTION_124M_0P5TPP"
    )
    assert (
        classification(True, 5.412, 5.411)
        == "REJECT_LAZY_RETRACTION_124M_0P5TPP"
    )
