from examples.nanogpt.seal_pair_vq_mlp_single_family_5tpp import (
    classification,
    dense_control_is_unmodified,
)


def test_classification_is_semantic_only_after_valid_seal() -> None:
    assert classification(False, "mlp.c_fc", 0.0, 0.01).startswith("INVALID_")
    assert classification(True, "mlp.c_fc", 0.0099, 0.01) == (
        "PAIR_VQ_C_FC_NEGLIGIBLE_124M_5TPP"
    )
    assert classification(True, "mlp.c_proj", 0.0101, 0.01) == (
        "PAIR_VQ_C_PROJ_MATERIAL_124M_5TPP"
    )


def test_dense_control_rejects_hidden_structured_representations() -> None:
    config = {"block_fht_targets": ["attn.c_attn.qk_headwise"]}
    assert dense_control_is_unmodified(config, "mlp.c_fc")
    assert dense_control_is_unmodified(config, "mlp.c_proj")
    assert not dense_control_is_unmodified(
        {"block_fht_targets": ["mlp.c_fc"]}, "mlp.c_fc"
    )
    assert not dense_control_is_unmodified(
        {"block_fht_mlp_cproj_muon_matched_givens": True}, "mlp.c_proj"
    )
