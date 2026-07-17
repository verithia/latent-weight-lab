from examples.nanogpt.model import GPTConfig, make_linear
from latent_weight_lab.block_fht import BlockFHTLinear


def test_spectral_config_applies_only_to_mlp_c_fc():
    config = GPTConfig(block_fht=True, block_fht_targets=("mlp.c_fc", "attn.c_proj"), block_fht_ffn_spectral_rank=2, block_fht_ffn_spectral_out_groups=2, block_fht_ffn_spectral_in_groups=2)
    cfc = make_linear(8, 16, False, config, "mlp.c_fc", 0)
    attn = make_linear(8, 8, False, config, "attn.c_proj", 1)
    assert isinstance(cfc, BlockFHTLinear) and cfc.spectral_core is not None
    assert isinstance(attn, BlockFHTLinear) and attn.spectral_core is None
