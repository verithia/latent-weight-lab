import pytest
import torch
import torch.nn.functional as F

from latent_weight_lab.block_fht import (
    BlockFHT,
    BlockFHTLinear,
    ProductFHTLinear,
    block_fht_slice_torch,
    flush_block_fht_weight_cache,
    prepare_block_fht_weight_cache,
    restore_block_fht_weight_cache,
    signs_for,
    sign_word_for,
    suspend_block_fht_weight_cache,
)
from examples.nanogpt.model import (
    Block,
    FixedFHTMix,
    GPT,
    GPTConfig,
    LearnedFHTBlockOrthogonalOutputMix,
    LearnedGivensOutputMix,
    MLP,
    freeze_non_block_fht,
)


def test_sign_word_uses_32_positions():
    bits = sign_word_for(seed=999, block=3, layer=2, word=4)
    signs = [1 if ((bits >> bit) & 1) else -1 for bit in range(32)]
    assert len(signs) == 32
    assert set(signs) <= {-1, 1}


def test_vectorized_signs_match_scalar_hash_bits():
    size = 65
    got = signs_for(torch.empty(1), block=3, layer=2, seed=999, block_size=size)
    expected = torch.tensor(
        [1.0 if ((sign_word_for(999, 3, 2, pos >> 5) >> (pos & 31)) & 1) else -1.0 for pos in range(size)]
    )
    assert torch.equal(got, expected)


def test_slice_matches_full_forward_cpu():
    bfht = BlockFHT(7, size=31, layers=2, seed=123)
    sliced = bfht.slice(3, 17)
    full = bfht()
    assert torch.allclose(sliced, full[3:17])


def test_product_fht_linear_initializes_at_target_scale_without_dense_basis():
    torch.manual_seed(7)
    layer = ProductFHTLinear(
        16,
        8,
        factors=3,
        seed=41,
        weight_std=0.05,
        bias=True,
    )
    assert layer.weight.shape == (8, 16)
    assert layer.product_log_diagonals.shape == (3, 16)
    assert layer.product_output_log_gain.shape == (8,)
    assert layer.trainable_scalar_count == 3 * 16 + 8 + 8
    assert not any(
        parameter.shape == layer.weight.shape
        for parameter in layer.parameters()
    )
    assert layer.weight.std().item() == pytest.approx(0.05, rel=0.2)


def test_product_fht_cache_vjp_matches_live_gradient_without_preconditioning():
    torch.manual_seed(11)
    live = ProductFHTLinear(
        8,
        4,
        factors=2,
        seed=97,
        weight_std=0.04,
        weight_space_muon=False,
        natural_gradient=False,
    )
    cached = ProductFHTLinear(
        8,
        4,
        factors=2,
        seed=97,
        weight_std=0.04,
        weight_space_muon=False,
        natural_gradient=False,
    )
    cached.load_state_dict(live.state_dict())
    live_input = torch.randn(3, 8, requires_grad=True)
    cached_input = live_input.detach().clone().requires_grad_(True)
    live(live_input).square().mean().backward()
    cached.materialize_weight_cache()
    cached(cached_input).square().mean().backward()
    cached.flush_weight_cache_to_factor_grad()
    torch.testing.assert_close(
        cached_input.grad,
        live_input.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        cached.product_log_diagonals.grad,
        live.product_log_diagonals.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        cached.product_output_log_gain.grad,
        live.product_output_log_gain.grad,
        atol=1e-6,
        rtol=1e-5,
    )


def test_product_fht_weight_space_muon_projects_finite_factor_gradients():
    torch.manual_seed(13)
    layer = ProductFHTLinear(
        8,
        4,
        factors=2,
        seed=101,
        weight_std=0.04,
        weight_space_muon=True,
        natural_gradient=True,
    )
    layer.materialize_weight_cache()
    layer(torch.randn(3, 8)).square().mean().backward()
    layer.flush_weight_cache_to_factor_grad()
    assert layer.weight_space_momentum_buffer.abs().sum() > 0
    assert layer.product_log_diagonals.grad is not None
    assert torch.isfinite(layer.product_log_diagonals.grad).all()
    assert layer.product_log_diagonals.grad.abs().sum() > 0
    assert layer.product_output_log_gain.grad is not None
    assert torch.isfinite(layer.product_output_log_gain.grad).all()


def test_product_fht_cproj_is_target_selective_and_sgd_owned():
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_fht=True,
            block_fht_targets=("attn.c_proj", "mlp.c_proj"),
            block_fht_latent_ratio=1.0,
            block_fht_cproj_product_fht_factors=2,
        )
    )
    attention = model.transformer.h[0].attn.c_proj
    cproj = model.transformer.h[0].mlp.c_proj
    assert isinstance(attention, BlockFHTLinear)
    assert isinstance(cproj, ProductFHTLinear)
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=2.4e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    factor_ids = {
        id(cproj.product_log_diagonals),
        id(cproj.product_output_log_gain),
    }
    owners = [
        child
        for child in optimizer.optimizers
        if any(
            id(parameter) in factor_ids
            for group in child.param_groups
            for parameter in group["params"]
        )
    ]
    assert len(owners) == 1
    assert owners[0].__class__.__name__ == "SGD"
    assert {
        id(parameter)
        for group in owners[0].param_groups
        for parameter in group["params"]
    } == factor_ids


def test_backward_cpu():
    bfht = BlockFHT(7, size=31, layers=2, seed=123)
    loss = bfht.slice(3, 17).square().sum()
    loss.backward()
    assert bfht.latent.grad is not None
    assert bfht.latent.grad.shape == bfht.latent.shape


def test_external_latent_parameter():
    latent = torch.nn.Parameter(torch.randn(8))
    bfht = BlockFHT(latent, size=40, layers=1, seed=5)
    assert bfht.latent is latent


def test_supported_cuda_block_range_metadata():
    small = BlockFHT(32, size=64, layers=1, seed=1)
    large = BlockFHT((1 << 23) - 17, size=1 << 23, layers=1, seed=1)
    assert small.block_size == 32
    assert large.block_size == 1 << 23


def test_reference_function():
    latent = torch.randn(8, requires_grad=True)
    out = block_fht_slice_torch(latent, size=40, layers=2, seed=7, start=5, stop=23)
    assert out.shape == (18,)
    out.square().sum().backward()
    assert latent.grad is not None


def test_block_fht_linear_matches_materialized_weight():
    layer = BlockFHTLinear(5, 3, bias=True, latent_dim=8, layers=2, seed=11)
    x = torch.randn(4, 5)
    out = layer(x)
    expected = F.linear(x, layer.weight, layer.bias)
    assert torch.allclose(out, expected)


def test_block_fht_linear_cached_grad_matches_dynamic():
    torch.manual_seed(123)
    dynamic = BlockFHTLinear(5, 3, bias=True, latent_dim=8, layers=2, seed=11)
    cached = BlockFHTLinear(5, 3, bias=True, latent_dim=8, layers=2, seed=11)
    cached.load_state_dict(dynamic.state_dict())
    x = torch.randn(4, 5)

    dynamic_loss = dynamic(x).square().mean()
    dynamic_loss.backward()

    prepare_block_fht_weight_cache(cached)
    cached_loss = cached(x).square().mean()
    cached_loss.backward()
    flush_block_fht_weight_cache(cached)

    assert torch.allclose(cached_loss, dynamic_loss)
    assert torch.allclose(cached.generator.latent.grad, dynamic.generator.latent.grad, atol=1e-6)
    assert torch.allclose(cached.bias.grad, dynamic.bias.grad, atol=1e-6)


def test_block_fht_linear_cached_grad_matches_dynamic_with_channel_gains():
    torch.manual_seed(321)
    dynamic = BlockFHTLinear(
        5,
        3,
        bias=True,
        latent_dim=8,
        layers=2,
        seed=13,
        output_gain=True,
        input_gain=True,
        modulation_alpha=1e-3,
        modulation_centered=True,
    )
    with torch.no_grad():
        dynamic.output_gain.copy_(torch.tensor([0.7, 1.1, 1.3]))
        dynamic.input_gain.copy_(torch.tensor([0.8, 1.2, 0.9, 1.4, 0.6]))
    cached = BlockFHTLinear(
        5,
        3,
        bias=True,
        latent_dim=8,
        layers=2,
        seed=13,
        output_gain=True,
        input_gain=True,
        modulation_alpha=1e-3,
        modulation_centered=True,
    )
    cached.load_state_dict(dynamic.state_dict())
    x = torch.randn(4, 5)

    dynamic_loss = dynamic(x).square().mean()
    dynamic_loss.backward()

    prepare_block_fht_weight_cache(cached)
    assert cached._cached_weight is not None
    cached_loss = cached(x).square().mean()
    cached_loss.backward()
    flush_block_fht_weight_cache(cached)

    assert torch.allclose(cached_loss, dynamic_loss)
    assert torch.allclose(cached.generator.latent.grad, dynamic.generator.latent.grad, atol=1e-6)
    assert torch.allclose(cached.bias.grad, dynamic.bias.grad, atol=1e-6)
    assert torch.allclose(cached.output_gain.grad, dynamic.output_gain.grad, atol=1e-6)
    assert torch.allclose(cached.input_gain.grad, dynamic.input_gain.grad, atol=1e-6)


def test_quadratic_chart_cached_gradient_matches_dynamic_gradient():
    torch.manual_seed(327)
    dynamic = BlockFHTLinear(
        5,
        3,
        bias=True,
        latent_dim=8,
        layers=2,
        seed=17,
        quadratic_scale=0.5,
    )
    cached = BlockFHTLinear(
        5,
        3,
        bias=True,
        latent_dim=8,
        layers=2,
        seed=17,
        quadratic_scale=0.5,
    )
    cached.load_state_dict(dynamic.state_dict())
    x = torch.randn(4, 5)

    dynamic_loss = dynamic(x).square().mean()
    dynamic_loss.backward()

    prepare_block_fht_weight_cache(cached)
    cached_loss = cached(x).square().mean()
    cached_loss.backward()
    flush_block_fht_weight_cache(cached)

    torch.testing.assert_close(cached_loss, dynamic_loss)
    torch.testing.assert_close(
        cached.generator.latent.grad,
        dynamic.generator.latent.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        cached.bias.grad,
        dynamic.bias.grad,
        atol=1e-6,
        rtol=1e-5,
    )


def test_quadratic_chart_is_nonlinear_without_extra_parameters():
    torch.manual_seed(329)
    linear = BlockFHTLinear(5, 3, latent_dim=8, layers=2, seed=19)
    curved = BlockFHTLinear(
        5,
        3,
        latent_dim=8,
        layers=2,
        seed=19,
        quadratic_scale=0.5,
    )
    curved.load_state_dict(linear.state_dict())
    assert sum(parameter.numel() for parameter in curved.parameters()) == sum(
        parameter.numel() for parameter in linear.parameters()
    )

    with torch.no_grad():
        z = curved.generator.latent.detach().clone()
        curved.generator.latent.copy_(z)
        positive = curved.weight.clone()
        curved.generator.latent.copy_(-z)
        negative = curved.weight.clone()
        curved.generator.latent.zero_()
        origin = curved.weight.clone()
    second_difference = positive + negative - 2.0 * origin
    assert second_difference.norm() > 0


def test_quadratic_chart_preserves_initial_weight_scale():
    torch.manual_seed(331)
    linear = BlockFHTLinear(
        64,
        128,
        latent_ratio=0.1,
        layers=2,
        seed=23,
    )
    curved = BlockFHTLinear(
        64,
        128,
        latent_ratio=0.1,
        layers=2,
        seed=23,
        quadratic_scale=0.5,
    )
    curved.load_state_dict(linear.state_dict())
    relative_std = curved.weight.std() / linear.weight.std()
    assert 0.95 < relative_std < 1.05


def test_quadratic_chart_is_target_selective_in_gpt():
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_fht=True,
            block_fht_targets=("attn.c_proj", "mlp.c_proj"),
            block_fht_latent_ratio=1.0,
            block_fht_quadratic_targets=("mlp.c_proj",),
            block_fht_quadratic_scale=0.5,
        )
    )
    assert model.transformer.h[0].attn.c_proj.quadratic_scale == 0.0
    assert model.transformer.h[0].mlp.c_proj.quadratic_scale == 0.5


def test_zero_initialized_affine_delta_starts_at_frozen_dense_base():
    torch.manual_seed(332)
    layer = BlockFHTLinear(
        5,
        3,
        bias=True,
        latent_dim=8,
        latent_shape=(2, 4),
        layers=2,
        seed=19,
        latent_init_std=0.02,
        weight_scale=0.25,
        residual_base_scale=1.0,
        residual_base_std=0.01,
        residual_delta_zero_init=True,
    )
    assert layer.residual_base_weight is not None
    assert torch.count_nonzero(layer.generator.latent) == 0
    torch.testing.assert_close(layer.weight, layer.residual_base_weight)
    inputs = torch.randn(7, 5)
    loss = layer(inputs).square().mean()
    loss.backward()
    assert layer.generator.latent.grad is not None
    assert torch.count_nonzero(layer.generator.latent.grad) > 0
    assert layer.residual_base_weight.grad is None


def test_affine_delta_is_target_selective_in_gpt():
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_fht=True,
            block_fht_targets=("attn.c_proj", "mlp.c_proj"),
            block_fht_latent_ratio=1.0,
            block_fht_match_gpt_init=True,
            block_fht_affine_delta_targets=("mlp.c_proj",),
            block_fht_affine_delta_scale=1.0,
        )
    )
    attention = model.transformer.h[0].attn.c_proj
    projection = model.transformer.h[0].mlp.c_proj
    assert attention.residual_base_weight is None
    assert projection.residual_base_weight is not None
    assert torch.count_nonzero(attention.generator.latent) > 0
    assert torch.count_nonzero(projection.generator.latent) == 0
    torch.testing.assert_close(projection.weight, projection.residual_base_weight)


@pytest.mark.parametrize(
    ("config_update", "message"),
    [
        (
            {
                "block_fht_targets": ("attn.c_proj",),
                "block_fht_affine_delta_targets": ("mlp.c_proj",),
            },
            "must also be BlockFHT targets",
        ),
        (
            {
                "block_fht_targets": ("mlp.c_proj",),
                "block_fht_affine_delta_targets": ("mlp.c_proj",),
                "block_fht_residual_base_scale": 0.5,
            },
            "legacy global residual base",
        ),
        (
            {
                "block_fht_targets": ("mlp.c_proj",),
                "block_fht_affine_delta_targets": ("mlp.c_proj",),
                "block_fht_affine_delta_scale": 0.0,
            },
            "must be positive and finite",
        ),
    ],
)
def test_affine_delta_configuration_is_validated(config_update, message):
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_fht=True,
        **config_update,
    )
    with pytest.raises(ValueError, match=message):
        GPT(config)


def test_affine_delta_cached_gradient_matches_dynamic_gradient():
    torch.manual_seed(334)
    dynamic = BlockFHTLinear(
        5,
        3,
        bias=True,
        latent_dim=8,
        latent_shape=(2, 4),
        layers=2,
        seed=23,
        residual_base_scale=0.5,
        residual_base_std=0.01,
        residual_delta_zero_init=True,
    )
    cached = BlockFHTLinear(
        5,
        3,
        bias=True,
        latent_dim=8,
        latent_shape=(2, 4),
        layers=2,
        seed=23,
        residual_base_scale=0.5,
        residual_base_std=0.01,
        residual_delta_zero_init=True,
    )
    cached.load_state_dict(dynamic.state_dict())
    inputs = torch.randn(7, 5)
    dynamic_loss = dynamic(inputs).square().mean()
    dynamic_loss.backward()
    prepare_block_fht_weight_cache(cached)
    assert cached._cached_weight is not None
    cached_loss = cached(inputs).square().mean()
    cached_loss.backward()
    flush_block_fht_weight_cache(cached)
    torch.testing.assert_close(cached_loss, dynamic_loss)
    torch.testing.assert_close(
        cached.generator.latent.grad,
        dynamic.generator.latent.grad,
        atol=1e-6,
        rtol=1e-5,
    )


def test_matrix_latent_matches_flat_latent_forward_and_gradient():
    torch.manual_seed(333)
    flat = BlockFHTLinear(5, 3, latent_dim=8, layers=2, seed=19)
    matrix = BlockFHTLinear(
        5,
        3,
        latent_dim=8,
        latent_shape=(2, 4),
        layers=2,
        seed=19,
    )
    with torch.no_grad():
        matrix.generator.latent.copy_(flat.generator.latent.reshape(2, 4))
    x_flat = torch.randn(4, 5, requires_grad=True)
    x_matrix = x_flat.detach().clone().requires_grad_(True)

    flat_loss = flat(x_flat).square().mean()
    matrix_loss = matrix(x_matrix).square().mean()
    flat_loss.backward()
    matrix_loss.backward()

    torch.testing.assert_close(matrix_loss, flat_loss)
    torch.testing.assert_close(x_matrix.grad, x_flat.grad)
    torch.testing.assert_close(
        matrix.generator.latent.grad.reshape(-1),
        flat.generator.latent.grad,
    )


def test_matrix_latent_cached_gradient_matches_dynamic_gradient():
    torch.manual_seed(335)
    dynamic = BlockFHTLinear(
        5,
        3,
        bias=True,
        latent_dim=8,
        latent_shape=(2, 4),
        layers=2,
        seed=23,
    )
    cached = BlockFHTLinear(
        5,
        3,
        bias=True,
        latent_dim=8,
        latent_shape=(2, 4),
        layers=2,
        seed=23,
    )
    cached.load_state_dict(dynamic.state_dict())
    x = torch.randn(4, 5)

    dynamic_loss = dynamic(x).square().mean()
    dynamic_loss.backward()
    prepare_block_fht_weight_cache(cached)
    cached_loss = cached(x).square().mean()
    cached_loss.backward()
    flush_block_fht_weight_cache(cached)

    torch.testing.assert_close(cached_loss, dynamic_loss)
    torch.testing.assert_close(
        cached.generator.latent.grad,
        dynamic.generator.latent.grad,
    )


def test_suspended_cache_keeps_ce_grad_and_live_perturbation_grad():
    """A stability forward must bypass only the unperturbed CE cache."""
    torch.manual_seed(456)
    dynamic = BlockFHTLinear(5, 3, bias=True, latent_dim=8, layers=2, seed=17)
    hybrid = BlockFHTLinear(5, 3, bias=True, latent_dim=8, layers=2, seed=17)
    hybrid.load_state_dict(dynamic.state_dict())
    x = torch.randn(4, 5)
    perturbation = torch.randn_like(dynamic.generator.latent) * 1e-3

    dynamic_base = dynamic(x).square().mean()
    dynamic_base.backward()
    with torch.no_grad():
        dynamic.generator.latent.add_(perturbation)
    dynamic_perturbed = dynamic(x).square().mean()
    dynamic_perturbed.backward()
    with torch.no_grad():
        dynamic.generator.latent.sub_(perturbation)

    prepare_block_fht_weight_cache(hybrid)
    hybrid_base = hybrid(x).square().mean()
    hybrid_base.backward()
    suspended = suspend_block_fht_weight_cache(hybrid)
    with torch.no_grad():
        hybrid.generator.latent.add_(perturbation)
    hybrid_perturbed = hybrid(x).square().mean()
    hybrid_perturbed.backward()
    with torch.no_grad():
        hybrid.generator.latent.sub_(perturbation)
    restore_block_fht_weight_cache(suspended)
    flush_block_fht_weight_cache(hybrid)

    assert torch.allclose(hybrid_base, dynamic_base)
    assert torch.allclose(hybrid_perturbed, dynamic_perturbed)
    assert torch.allclose(hybrid.generator.latent.grad, dynamic.generator.latent.grad, atol=1e-6)
    assert torch.allclose(hybrid.bias.grad, dynamic.bias.grad, atol=1e-6)


def test_postgelu_spread_reduces_without_full_activation_cast():
    mlp = MLP(GPTConfig(n_embd=4, n_head=1, block_fht_ffn_postgelu_std_target=0.15), layer_id=0)
    values = torch.randn(3, 5, 16, requires_grad=True)
    mlp.last_postgelu = values
    actual = mlp.postgelu_spread_loss()
    assert actual is not None
    expected_std = values.float().reshape(-1, values.shape[-1]).std(dim=0, unbiased=False)
    expected = torch.relu(expected_std.new_tensor(0.15) - expected_std).square().mean()
    assert torch.allclose(actual, expected)
    actual.backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_mlp_cproj_teacher_alignment_is_zero_at_identity_and_differentiable():
    config = GPTConfig(
        n_embd=8,
        n_head=1,
        block_fht_mlp_output_block_rotation_stages=1,
        block_fht_mlp_output_block_rotation_size=4,
        block_fht_mlp_output_block_rotation_basis_size=8,
        block_fht_mlp_residual_output_gain=True,
    )
    mlp = MLP(config, layer_id=0)
    teacher_weight = mlp.c_proj.weight.detach().clone()
    mlp.set_cproj_teacher_weight(teacher_weight)
    values = torch.randn(2, 3, 8)

    mlp(values)
    identity_loss = mlp.cproj_teacher_alignment_loss()
    assert identity_loss is not None
    assert torch.allclose(identity_loss, torch.zeros_like(identity_loss))

    assert mlp.residual_output_log_gain is not None
    with torch.no_grad():
        mlp.residual_output_log_gain.add_(0.1)
    mlp.zero_grad(set_to_none=True)
    mlp(values)
    shifted_loss = mlp.cproj_teacher_alignment_loss()
    assert shifted_loss is not None and shifted_loss > 0
    with torch.no_grad():
        activated = mlp.gelu(mlp.c_fc(values))
        student_weight = mlp._materialize_charted_cproj_weight(
            mlp.c_proj.weight
        )
        expected = F.mse_loss(
            F.linear(activated, student_weight),
            F.linear(activated, teacher_weight),
        )
    assert torch.allclose(shifted_loss, expected)
    shifted_loss.backward()
    assert mlp.residual_output_log_gain.grad is not None
    assert torch.isfinite(mlp.residual_output_log_gain.grad).all()


def test_forward_fused_matches_materialized_with_both_gains():
    layer = BlockFHTLinear(5, 3, bias=True, latent_dim=8, layers=2, seed=11, output_gain=True, input_gain=True)
    x = torch.randn(4, 5)
    assert torch.allclose(layer.forward_fused(x), F.linear(x, layer.weight, layer.bias))


def test_freeze_restores_input_gain_trainability():
    layer = BlockFHTLinear(5, 3, latent_dim=8, layers=2, output_gain=True, input_gain=True)
    freeze_non_block_fht(torch.nn.Sequential(layer), train_embeddings=False)
    assert layer.input_gain.requires_grad and layer.output_gain.requires_grad


def test_mlp_shared_hidden_gain_is_identity_initialized_and_trainable():
    torch.manual_seed(334)
    base = MLP(GPTConfig(n_embd=4, n_head=1), layer_id=0)
    paired = MLP(
        GPTConfig(
            n_embd=4,
            n_head=1,
            block_fht_mlp_shared_hidden_gain=True,
        ),
        layer_id=0,
    )
    paired.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(3, 5, 4)
    torch.testing.assert_close(paired(x), base(x))
    paired(x).square().mean().backward()
    assert paired.shared_hidden_log_gain.grad is not None
    assert paired.shared_hidden_log_gain.grad.abs().sum() > 0


def test_selected_block_fht_latent_is_matrix_and_muon_owned():
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_fht=True,
            block_fht_targets=("attn.c_proj", "mlp.c_proj"),
            block_fht_latent_ratio=1.0,
            block_fht_muon_latent_targets=("mlp.c_proj",),
            block_fht_muon_latent_rows=4,
        )
    )
    attn_latent = model.transformer.h[0].attn.c_proj.generator.latent
    mlp_latent = model.transformer.h[0].mlp.c_proj.generator.latent
    assert attn_latent.ndim == 1
    assert mlp_latent.ndim == 2
    assert mlp_latent.shape[0] == 4
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=2.4e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_adamw_lr_scale=0.3,
    )
    muon_parameters = [
        parameter
        for child in optimizer.optimizers
        if child.__class__.__name__ == "Muon"
        for group in child.param_groups
        for parameter in group["params"]
    ]
    assert any(parameter is mlp_latent for parameter in muon_parameters)
    assert not any(parameter is attn_latent for parameter in muon_parameters)


def test_mlp_chart_lr_scale_only_scales_chart_adamw_group():
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_fht=True,
            block_fht_targets=("mlp.c_proj",),
            block_fht_latent_ratio=1.0,
            block_fht_muon_latent_targets=("mlp.c_proj",),
            block_fht_muon_latent_rows=4,
            block_fht_mlp_pregelu_block_rotation_stages=1,
            block_fht_mlp_pregelu_block_rotation_size=4,
            block_fht_mlp_pregelu_block_rotation_basis_size=8,
            block_fht_mlp_hidden_block_rotation_stages=1,
            block_fht_mlp_hidden_block_rotation_size=4,
            block_fht_mlp_hidden_block_rotation_basis_size=8,
            block_fht_mlp_output_block_rotation_stages=1,
            block_fht_mlp_output_block_rotation_size=4,
            block_fht_mlp_output_block_rotation_basis_size=8,
            block_fht_mlp_residual_output_gain=True,
        )
    )
    mlp = model.transformer.h[0].mlp
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=2.4e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_adamw_lr_scale=0.3,
        block_fht_mlp_chart_lr_scale=5.0,
        block_fht_mlp_pregelu_chart_lr_scale=0.1,
    )
    adamw = next(
        child
        for child in optimizer.optimizers
        if child.__class__.__name__ == "AdamW"
    )
    chart_parameters = {
        id(mlp.hidden_block_rotation.coordinates),
        id(mlp.output_block_rotation.coordinates),
        id(mlp.residual_output_log_gain),
    }
    pregelu_chart_parameters = {
        id(mlp.pregelu_block_rotation.coordinates),
    }
    chart_group = next(
        group
        for group in adamw.param_groups
        if any(id(parameter) in chart_parameters for parameter in group["params"])
    )
    regular_group = next(
        group
        for group in adamw.param_groups
        if not any(
            id(parameter) in chart_parameters | pregelu_chart_parameters
            for parameter in group["params"]
        )
    )
    pregelu_chart_group = next(
        group
        for group in adamw.param_groups
        if any(
            id(parameter) in pregelu_chart_parameters
            for parameter in group["params"]
        )
    )
    assert chart_group["lr_scale"] == pytest.approx(1.5)
    assert pregelu_chart_group["lr_scale"] == pytest.approx(0.03)
    assert regular_group["lr_scale"] == pytest.approx(0.3)
    assert {
        id(parameter) for parameter in chart_group["params"]
    }.issuperset(chart_parameters)
    assert {
        id(parameter) for parameter in pregelu_chart_group["params"]
    } == pregelu_chart_parameters
    assert {
        id(parameter) for parameter in chart_group["params"]
    }.isdisjoint(pregelu_chart_parameters)


def test_spectral_zero_correction_matches_same_seed_block_fht():
    base = BlockFHTLinear(8, 12, bias=True, latent_dim=8, layers=2, seed=9)
    structured = BlockFHTLinear(8, 12, bias=True, latent_dim=8, layers=2, seed=9, spectral_rank=2, spectral_out_groups=3, spectral_in_groups=2)
    structured.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(3, 8)
    assert torch.allclose(structured.weight, base.weight)
    assert torch.allclose(structured(x), base(x))


def test_spectral_core_and_group_gains_receive_gradients_and_disable_cache():
    layer = BlockFHTLinear(8, 12, latent_dim=8, layers=2, seed=9, spectral_rank=2, spectral_out_groups=3, spectral_in_groups=2)
    with torch.no_grad():
        layer.spectral_core[0, 0] = 0.1
    layer.materialize_weight_cache()
    assert layer._cached_weight is None
    loss = layer(torch.randn(4, 8)).square().mean()
    loss.backward()
    assert torch.isfinite(layer.spectral_core.grad).all()
    assert torch.isfinite(layer.spectral_log_out_gain.grad).all()
    assert torch.isfinite(layer.spectral_log_in_gain.grad).all()
    freeze_non_block_fht(torch.nn.Sequential(layer), train_embeddings=False)
    assert layer.spectral_core.requires_grad and layer.spectral_log_out_gain.requires_grad and layer.spectral_log_in_gain.requires_grad


def test_cproj_fixed_basis_spectrum_is_zero_function_but_trainable_at_scale_one():
    torch.manual_seed(789)
    base = MLP(GPTConfig(n_embd=4, n_head=1), layer_id=0)
    structured = MLP(
        GPTConfig(
            n_embd=4,
            n_head=1,
            block_fht_cproj_spectral_resid_rank=2,
            block_fht_cproj_spectral_resid_scale_init=1.0,
            block_fht_cproj_spectral_resid_seed=17001,
        ),
        layer_id=0,
    )
    structured.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(3, 5, 4)

    expected = base(x)
    actual = structured(x)
    assert torch.allclose(actual, expected)

    actual.square().mean().backward()
    assert structured.cproj_spectral_resid_diag.grad is not None
    assert torch.isfinite(structured.cproj_spectral_resid_diag.grad).all()
    assert structured.cproj_spectral_resid_diag.grad.abs().sum() > 0
    assert structured.cproj_spectral_resid_scale.grad is not None
    assert torch.equal(
        structured.cproj_spectral_resid_scale.grad,
        torch.zeros_like(structured.cproj_spectral_resid_scale.grad),
    )


def test_cproj_fixed_basis_muon_matrix_preserves_diagonal_forward_and_uses_matrix_parameter():
    torch.manual_seed(789)
    vector = MLP(
        GPTConfig(
            n_embd=4,
            n_head=1,
            block_fht_cproj_spectral_resid_rank=2,
            block_fht_cproj_spectral_resid_scale_init=1.0,
            block_fht_cproj_spectral_resid_seed=17001,
        ),
        layer_id=0,
    )
    matrix = MLP(
        GPTConfig(
            n_embd=4,
            n_head=1,
            block_fht_cproj_spectral_resid_rank=2,
            block_fht_cproj_spectral_resid_scale_init=1.0,
            block_fht_cproj_spectral_resid_seed=17001,
            block_fht_cproj_spectral_resid_muon_matrix=True,
        ),
        layer_id=0,
    )
    shared_state = vector.state_dict()
    shared_state.pop("cproj_spectral_resid_diag")
    matrix.load_state_dict(shared_state, strict=False)
    with torch.no_grad():
        values = torch.tensor([0.25, -0.125])
        vector.cproj_spectral_resid_diag.copy_(values)
        matrix.cproj_spectral_resid_diag.copy_(values.view(1, -1))
    assert vector.cproj_spectral_resid_diag.ndim == 1
    assert matrix.cproj_spectral_resid_diag.ndim == 2
    x = torch.randn(3, 5, 4)
    torch.testing.assert_close(matrix(x), vector(x))


def test_cproj_fixed_basis_muon_matrix_is_assigned_to_muon_optimizer():
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_fht_cproj_spectral_resid_rank=4,
            block_fht_cproj_spectral_resid_scale_init=1.0,
            block_fht_cproj_spectral_resid_seed=17001,
            block_fht_cproj_spectral_resid_muon_matrix=True,
        )
    )
    diagonal = model.transformer.h[0].mlp.cproj_spectral_resid_diag
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=2.4e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_adamw_lr_scale=0.3,
    )
    muon_parameters = [
        parameter
        for child in optimizer.optimizers
        if child.__class__.__name__ == "Muon"
        for group in child.param_groups
        for parameter in group["params"]
    ]
    assert any(parameter is diagonal for parameter in muon_parameters)


def test_cproj_fixed_basis_full_core_is_zero_function_trainable_and_muon_owned():
    torch.manual_seed(790)
    base = MLP(GPTConfig(n_embd=4, n_head=1), layer_id=0)
    structured = MLP(
        GPTConfig(
            n_embd=4,
            n_head=1,
            block_fht_cproj_spectral_resid_rank=2,
            block_fht_cproj_spectral_resid_scale_init=1.0,
            block_fht_cproj_spectral_resid_seed=17001,
            block_fht_cproj_spectral_resid_full_core=True,
        ),
        layer_id=0,
    )
    structured.load_state_dict(base.state_dict(), strict=False)
    core = structured.cproj_spectral_resid_diag
    assert core.shape == (2, 2)
    x = torch.randn(3, 5, 4)
    torch.testing.assert_close(structured(x), base(x))
    structured(x).square().mean().backward()
    assert core.grad is not None
    assert torch.isfinite(core.grad).all()
    assert core.grad.abs().sum() > 0

    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_fht_cproj_spectral_resid_rank=4,
            block_fht_cproj_spectral_resid_scale_init=1.0,
            block_fht_cproj_spectral_resid_seed=17001,
            block_fht_cproj_spectral_resid_full_core=True,
        )
    )
    model_core = model.transformer.h[0].mlp.cproj_spectral_resid_diag
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=2.4e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_adamw_lr_scale=0.3,
    )
    muon_parameters = [
        parameter
        for child in optimizer.optimizers
        if child.__class__.__name__ == "Muon"
        for group in child.param_groups
        for parameter in group["params"]
    ]
    assert any(parameter is model_core for parameter in muon_parameters)


def test_cproj_fixed_basis_full_core_matches_explicit_basis_weight():
    layer = MLP(
        GPTConfig(
            n_embd=4,
            n_head=1,
            block_fht_cproj_spectral_resid_rank=2,
            block_fht_cproj_spectral_resid_scale_init=0.75,
            block_fht_cproj_spectral_resid_seed=17001,
            block_fht_cproj_spectral_resid_full_core=True,
        ),
        layer_id=0,
    )
    with torch.no_grad():
        layer.cproj_spectral_resid_diag.copy_(
            torch.tensor([[0.25, -0.125], [0.5, 0.375]])
        )
    x = torch.randn(2, 3, 4)
    hidden = layer.c_fc(x)
    activated = layer.gelu(hidden)
    base = layer.c_proj(activated)
    effective = (
        layer.cproj_spectral_resid_out_basis
        @ layer.cproj_spectral_resid_diag
        @ layer.cproj_spectral_resid_in_basis.T
    )
    expected = base + 0.75 * torch.nn.functional.linear(activated, effective)
    torch.testing.assert_close(layer(x), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_fht_mix_constructs_under_ambient_cuda_device():
    with torch.device("cuda:0"):
        mix = FixedFHTMix(8, 17001)
        basis = mix.basis_columns(4)
    assert mix.signs.device.type == "cpu"
    assert basis.device.type == "cpu"
    mix.to("cuda:0")
    output = mix(torch.randn(2, 8, device="cuda:0"))
    assert output.device.type == "cuda"


def test_fixed_fht_basis_columns_match_both_lowrank_projection_sides():
    torch.manual_seed(2468)
    mix = FixedFHTMix(5, seed=23)
    rank = 3
    basis = mix.basis_columns(rank)

    inputs = torch.randn(7, 5)
    assert torch.allclose(inputs.matmul(basis), mix(inputs)[..., :rank], atol=1e-6)

    coefficients = torch.randn(7, rank)
    padded = F.pad(coefficients, (0, 5 - rank))
    assert torch.allclose(coefficients.matmul(basis.transpose(0, 1)), mix(padded), atol=1e-6)


def test_learned_givens_output_mix_is_identity_at_initialization() -> None:
    mix = LearnedGivensOutputMix(8, stages=3, seed=19)
    values = torch.randn(2, 5, 8)
    torch.testing.assert_close(mix(values), values)
    assert mix.angles.ndim == 1


def test_learned_givens_output_mix_preserves_norm_and_has_angle_gradient() -> None:
    mix = LearnedGivensOutputMix(8, stages=3, seed=23)
    with torch.no_grad():
        mix.angles.copy_(torch.linspace(-0.4, 0.5, mix.angles.numel()))
    values = torch.randn(4, 7, 8)
    output = mix(values)
    torch.testing.assert_close(output.norm(dim=-1), values.norm(dim=-1), atol=1e-5, rtol=1e-5)
    output[..., 0].sum().backward()
    assert mix.angles.grad is not None
    assert float(mix.angles.grad.abs().sum()) > 0.0


def test_fht_block_orthogonal_mix_is_identity_and_norm_preserving() -> None:
    mix = LearnedFHTBlockOrthogonalOutputMix(
        features=16,
        stages=2,
        rotation_block_size=4,
        basis_block_size=8,
        seed=29,
    )
    values = torch.randn(5, 7, 16)
    torch.testing.assert_close(mix(values), values, atol=2e-6, rtol=2e-6)
    assert mix.coordinates.ndim == 1
    with torch.no_grad():
        mix.coordinates.normal_(std=0.1)
    output = mix(values)
    torch.testing.assert_close(output, values @ mix.matrix(values))
    torch.testing.assert_close(
        output.norm(dim=-1),
        values.norm(dim=-1),
        atol=2e-5,
        rtol=2e-5,
    )
    output[..., 0].sum().backward()
    assert mix.coordinates.grad is not None
    assert float(mix.coordinates.grad.abs().sum()) > 0.0


def test_fht_block_orthogonal_inverse_recovers_inputs_and_gradients() -> None:
    mix = LearnedFHTBlockOrthogonalOutputMix(16, 3, 4, 8, 29)
    values = torch.randn(3, 5, 16, requires_grad=True)
    with torch.no_grad():
        mix.coordinates.normal_(std=0.1)
    reconstructed = mix.inverse(mix(values))
    torch.testing.assert_close(
        reconstructed,
        values,
        atol=3e-5,
        rtol=3e-5,
    )
    reconstructed.square().mean().backward()
    assert values.grad is not None
    assert mix.coordinates.grad is not None


def test_fht_block_orthogonal_coordinate_scale_scales_identity_tangent() -> None:
    base = LearnedFHTBlockOrthogonalOutputMix(
        16, 2, 4, 8, 31, coordinate_scale=1.0
    )
    scaled = LearnedFHTBlockOrthogonalOutputMix(
        16, 2, 4, 8, 31, coordinate_scale=4.0
    )
    values = torch.randn(3, 16)
    torch.testing.assert_close(base(values), scaled(values))
    base(values)[:, 0].sum().backward()
    scaled(values)[:, 0].sum().backward()
    torch.testing.assert_close(
        scaled.coordinates.grad,
        4.0 * base.coordinates.grad,
        atol=1e-6,
        rtol=1e-6,
    )


def test_mlp_folds_block_rotation_and_output_gain_into_cproj_weight() -> None:
    torch.manual_seed(303)
    config = GPTConfig(
        n_embd=8,
        n_head=1,
        block_fht_mlp_output_block_rotation_stages=2,
        block_fht_mlp_output_block_rotation_size=4,
        block_fht_mlp_output_block_rotation_basis_size=8,
        block_fht_mlp_residual_output_gain=True,
        block_fht_mlp_residual_output_gain_scale=3.0,
    )
    mlp = MLP(config, layer_id=0)
    values = torch.randn(2, 3, 8)
    with torch.no_grad():
        mlp.output_block_rotation.coordinates.normal_(std=0.05)
        mlp.residual_output_log_gain.normal_(std=0.03)
    activated = mlp.gelu(mlp.c_fc(values))
    gain = (3.0 * mlp.residual_output_log_gain).exp()
    expected_weight = mlp.output_block_rotation(
        mlp.c_proj.weight.transpose(0, 1) * gain
    ).transpose(0, 1)
    expected = F.linear(activated, expected_weight, mlp.c_proj.bias)
    torch.testing.assert_close(mlp(values), expected)
    mlp(values).square().mean().backward()
    assert mlp.output_block_rotation.coordinates.grad is not None
    assert mlp.residual_output_log_gain.grad is not None


def test_mlp_folds_hidden_rotation_and_gain_into_cproj_weight() -> None:
    torch.manual_seed(305)
    config = GPTConfig(
        n_embd=8,
        n_head=1,
        block_fht_mlp_hidden_block_rotation_stages=2,
        block_fht_mlp_hidden_block_rotation_size=4,
        block_fht_mlp_hidden_block_rotation_basis_size=8,
        block_fht_mlp_hidden_gain=True,
        block_fht_mlp_hidden_gain_scale=3.0,
    )
    mlp = MLP(config, layer_id=0)
    values = torch.randn(2, 3, 8)
    with torch.no_grad():
        mlp.hidden_block_rotation.coordinates.normal_(std=0.05)
        mlp.hidden_log_gain.normal_(std=0.03)
    activated = mlp.gelu(mlp.c_fc(values))
    rotation = mlp.hidden_block_rotation.matrix(mlp.c_proj.weight)
    gain = (3.0 * mlp.hidden_log_gain).exp()
    expected_weight = (
        mlp.c_proj.weight @ rotation.transpose(0, 1)
    ) * gain
    expected = F.linear(activated, expected_weight, mlp.c_proj.bias)
    torch.testing.assert_close(mlp(values), expected)
    mlp(values).square().mean().backward()
    assert mlp.hidden_block_rotation.coordinates.grad is not None
    assert mlp.hidden_log_gain.grad is not None


def install_pregelu_frame(model: GPT) -> None:
    for layer_id, block in enumerate(model.transformer.h):
        block.mlp.pregelu_block_rotation = (
            LearnedFHTBlockOrthogonalOutputMix(
                features=4 * model.config.n_embd,
                stages=2,
                rotation_block_size=4,
                basis_block_size=8,
                seed=311 + layer_id,
                coordinate_scale=3.0,
            )
        )


def test_mlp_folds_pregelu_frame_into_cfc_weight() -> None:
    torch.manual_seed(309)
    config = GPTConfig(n_embd=8, n_head=1, bias=False)
    mlp = MLP(config, layer_id=0)
    mlp.pregelu_block_rotation = LearnedFHTBlockOrthogonalOutputMix(
        features=4 * config.n_embd,
        stages=2,
        rotation_block_size=4,
        basis_block_size=8,
        seed=310,
        coordinate_scale=3.0,
    )
    values = torch.randn(2, 3, config.n_embd)
    identity_expected = mlp.c_proj(mlp.gelu(mlp.c_fc(values)))
    torch.testing.assert_close(mlp(values), identity_expected)

    with torch.no_grad():
        mlp.pregelu_block_rotation.coordinates.normal_(std=0.03)
    explicit = mlp.c_proj(
        mlp.gelu(mlp.pregelu_block_rotation(mlp.c_fc(values)))
    )
    torch.testing.assert_close(mlp(values), explicit)


@pytest.mark.parametrize("retain_graph", [False, True])
def test_cached_charted_cfc_matches_live_forward_and_gradients(
    retain_graph: bool,
) -> None:
    torch.manual_seed(312)
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_embd=8,
        bias=False,
        block_fht_mlp_pregelu_block_rotation_stages=2,
        block_fht_mlp_pregelu_block_rotation_size=4,
        block_fht_mlp_pregelu_block_rotation_basis_size=8,
        block_fht_mlp_pregelu_block_rotation_coordinate_scale=3.0,
        block_fht_mlp_pregelu_block_rotation_seed=311,
        block_fht_mlp_pregelu_cache_retain_graph=retain_graph,
    )
    live = GPT(config)
    cached = GPT(config)
    with torch.no_grad():
        live.transformer.h[
            0
        ].mlp.pregelu_block_rotation.coordinates.normal_(std=0.02)
    cached.load_state_dict(live.state_dict())
    inputs = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, inputs.shape)

    live_loss = live(inputs, targets)[1]
    assert live_loss is not None
    live_loss.backward()

    cached.prepare_block_fht_cache()
    cached_mlp = cached.transformer.h[0].mlp
    assert cached_mlp._cached_charted_cfc_weight is not None
    assert (
        cached_mlp._cached_charted_cfc_graph_weight is not None
    ) == retain_graph
    cached_loss = cached(inputs, targets)[1]
    assert cached_loss is not None
    cached_loss.backward()
    cached.flush_block_fht_cache()
    assert cached_mlp._cached_charted_cfc_weight is None
    assert cached_mlp._cached_charted_cfc_graph_weight is None

    torch.testing.assert_close(cached_loss, live_loss)
    torch.testing.assert_close(
        cached_mlp.c_fc.weight.grad,
        live.transformer.h[0].mlp.c_fc.weight.grad,
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        cached_mlp.pregelu_block_rotation.coordinates.grad,
        live.transformer.h[
            0
        ].mlp.pregelu_block_rotation.coordinates.grad,
        atol=2e-6,
        rtol=2e-5,
    )


def test_cached_charted_cfc_can_skip_frozen_base_vjp() -> None:
    torch.manual_seed(313)
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_embd=8,
        bias=False,
    )
    model = GPT(config)
    install_pregelu_frame(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mlp = model.transformer.h[0].mlp
    coordinates = mlp.pregelu_block_rotation.coordinates
    coordinates.requires_grad_(True)
    model.prepare_block_fht_cache()
    assert mlp._cached_charted_cfc_weight is not None
    inputs = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, inputs.shape)
    loss = model(inputs, targets)[1]
    assert loss is not None
    loss.backward()
    mlp.flush_charted_cfc_cache(project_base_gradient=False)
    assert mlp.c_fc.weight.grad is None
    assert coordinates.grad is not None


def test_cached_charted_cproj_matches_live_forward_and_gradients() -> None:
    torch.manual_seed(307)
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_embd=8,
        bias=False,
        block_fht=True,
        block_fht_targets=("mlp.c_proj",),
        block_fht_latent_ratio=0.25,
        block_fht_mlp_hidden_block_rotation_stages=2,
        block_fht_mlp_hidden_block_rotation_size=4,
        block_fht_mlp_hidden_block_rotation_basis_size=8,
        block_fht_mlp_hidden_block_rotation_coordinate_scale=2.0,
        block_fht_mlp_hidden_gain=True,
        block_fht_mlp_hidden_gain_scale=3.0,
        block_fht_mlp_hidden_log_gain_init=0.05,
        block_fht_mlp_output_block_rotation_stages=2,
        block_fht_mlp_output_block_rotation_size=4,
        block_fht_mlp_output_block_rotation_basis_size=8,
        block_fht_mlp_output_block_rotation_coordinate_scale=3.0,
        block_fht_mlp_residual_output_gain=True,
        block_fht_mlp_residual_output_gain_scale=2.0,
        block_fht_mlp_residual_output_log_gain_init=0.1,
    )
    live = GPT(config)
    cached = GPT(config)
    cached.load_state_dict(live.state_dict())
    with torch.no_grad():
        for model in (live, cached):
            mlp = model.transformer.h[0].mlp
            mlp.hidden_block_rotation.coordinates.normal_(std=0.02)
            mlp.hidden_log_gain.add_(
                torch.linspace(-0.01, 0.01, 4 * config.n_embd)
            )
            mlp.output_block_rotation.coordinates.normal_(std=0.03)
            mlp.residual_output_log_gain.add_(
                torch.linspace(-0.02, 0.02, config.n_embd)
            )
    cached.load_state_dict(live.state_dict())
    inputs = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, inputs.shape)

    live_loss = live(inputs, targets)[1]
    assert live_loss is not None
    live_loss.backward()

    cached.prepare_block_fht_cache()
    cached_mlp = cached.transformer.h[0].mlp
    assert cached_mlp._cached_charted_cproj_weight is not None
    cached_loss = cached(inputs, targets)[1]
    assert cached_loss is not None
    cached_loss.backward()
    cached.flush_block_fht_cache()
    assert cached_mlp._cached_charted_cproj_weight is None

    torch.testing.assert_close(cached_loss, live_loss)
    torch.testing.assert_close(
        cached_mlp.c_proj.generator.latent.grad,
        live.transformer.h[0].mlp.c_proj.generator.latent.grad,
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        cached_mlp.hidden_block_rotation.coordinates.grad,
        live.transformer.h[0].mlp.hidden_block_rotation.coordinates.grad,
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        cached_mlp.hidden_log_gain.grad,
        live.transformer.h[0].mlp.hidden_log_gain.grad,
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        cached_mlp.output_block_rotation.coordinates.grad,
        live.transformer.h[0].mlp.output_block_rotation.coordinates.grad,
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        cached_mlp.residual_output_log_gain.grad,
        live.transformer.h[0].mlp.residual_output_log_gain.grad,
        atol=2e-6,
        rtol=2e-5,
    )


def test_cached_charted_cproj_can_skip_frozen_base_vjp() -> None:
    torch.manual_seed(308)
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_embd=8,
        bias=False,
        block_fht=True,
        block_fht_targets=("mlp.c_proj",),
        block_fht_latent_ratio=0.25,
        block_fht_mlp_hidden_block_rotation_stages=1,
        block_fht_mlp_hidden_block_rotation_size=4,
        block_fht_mlp_hidden_block_rotation_basis_size=8,
        block_fht_mlp_hidden_gain=True,
        block_fht_mlp_output_block_rotation_stages=1,
        block_fht_mlp_output_block_rotation_size=4,
        block_fht_mlp_output_block_rotation_basis_size=8,
        block_fht_mlp_residual_output_gain=True,
    )
    model = GPT(config)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mlp = model.transformer.h[0].mlp
    chart = (
        mlp.hidden_block_rotation.coordinates,
        mlp.hidden_log_gain,
        mlp.output_block_rotation.coordinates,
        mlp.residual_output_log_gain,
    )
    for parameter in chart:
        parameter.requires_grad_(True)
    model.prepare_block_fht_cache()
    base_weight = mlp.c_proj._cached_weight
    assert base_weight is not None
    base_weight.requires_grad_(False)
    inputs = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, inputs.shape)
    loss = model(inputs, targets)[1]
    assert loss is not None
    loss.backward()
    mlp.flush_charted_cproj_cache(project_base_gradient=False)
    assert base_weight.grad is None
    assert mlp.c_proj.generator.latent.grad is None
    assert all(parameter.grad is not None for parameter in chart)


def test_mlp_residual_output_gain_init_is_in_effective_log_coordinates() -> None:
    mlp = MLP(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_residual_output_gain=True,
            block_fht_mlp_residual_output_gain_scale=4.0,
            block_fht_mlp_residual_output_log_gain_init=0.125,
        ),
        layer_id=0,
    )
    assert mlp.residual_output_log_gain is not None
    torch.testing.assert_close(
        4.0 * mlp.residual_output_log_gain,
        torch.full((8,), 0.125),
    )


def test_mlp_residual_conditioned_output_gate_is_identity_and_dynamic() -> None:
    torch.manual_seed(490)
    base = Block(
        GPTConfig(
            n_embd=8,
            n_head=1,
        ),
        layer_id=0,
    )
    paired = Block(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_residual_conditioned_output_gate=True,
        ),
        layer_id=0,
    )
    paired.load_state_dict(base.state_dict(), strict=False)
    values = torch.randn(2, 3, 8)
    torch.testing.assert_close(paired(values), base(values))

    paired(values).sum().backward()
    mlp = paired.mlp
    assert mlp.residual_conditioned_output_slope is not None
    assert mlp.residual_conditioned_output_bias is not None
    assert mlp.residual_conditioned_output_slope.grad is not None
    assert mlp.residual_conditioned_output_bias.grad is not None
    assert mlp.residual_conditioned_output_slope.grad.abs().sum() > 0
    assert mlp.residual_conditioned_output_bias.grad.abs().sum() > 0


def test_mlp_residual_conditioned_output_gate_selects_layers() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=3,
            n_head=1,
            n_embd=8,
            block_fht_mlp_residual_conditioned_output_gate=True,
            block_fht_mlp_residual_conditioned_output_gate_layers=(0, 2),
        )
    )
    assert (
        model.transformer.h[0].mlp.residual_conditioned_output_slope
        is not None
    )
    assert (
        model.transformer.h[1].mlp.residual_conditioned_output_slope
        is None
    )
    assert (
        model.transformer.h[2].mlp.residual_conditioned_output_slope
        is not None
    )


def test_mlp_residual_conditioned_output_gate_can_reuse_static_gain() -> None:
    mlp = MLP(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_residual_conditioned_output_gate=True,
            block_fht_mlp_residual_conditioned_output_gate_bias=False,
        ),
        layer_id=0,
    )
    assert mlp.residual_conditioned_output_slope is not None
    assert mlp.residual_conditioned_output_bias is None
    condition = torch.randn(2, 3, 8)
    modulation = mlp.residual_conditioned_output_modulation(condition)
    assert modulation is not None
    torch.testing.assert_close(modulation, torch.zeros_like(modulation))
    modulation.sum().backward()
    assert mlp.residual_conditioned_output_slope.grad is not None
    assert mlp.residual_conditioned_output_slope.grad.abs().sum() > 0


def test_mlp_residual_conditioned_fixed_basis_gate_matches_diagnostic() -> None:
    from examples.nanogpt.analyze_mlp_conditioned_gate_alignment import (
        FixedBasisBilinearOutputGate,
    )

    torch.manual_seed(491)
    mlp = MLP(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_residual_conditioned_output_gate=True,
            block_fht_mlp_residual_conditioned_output_gate_fixed_basis=True,
            block_fht_mlp_residual_conditioned_output_gate_basis_block_size=8,
            block_fht_mlp_residual_conditioned_output_gate_basis_seed=123,
        ),
        layer_id=0,
    )
    diagnostic = FixedBasisBilinearOutputGate(
        8,
        basis_block_size=8,
        seed=123,
    )
    assert mlp.residual_conditioned_output_slope is not None
    assert mlp.residual_conditioned_output_bias is not None
    with torch.no_grad():
        mlp.residual_conditioned_output_slope.copy_(
            torch.randn_like(mlp.residual_conditioned_output_slope)
        )
        mlp.residual_conditioned_output_bias.copy_(
            torch.randn_like(mlp.residual_conditioned_output_bias)
        )
        diagnostic.slope.copy_(mlp.residual_conditioned_output_slope)
        diagnostic.bias.copy_(mlp.residual_conditioned_output_bias)
    condition = torch.randn(2, 3, 8)
    update = torch.randn(2, 3, 8)
    actual = mlp.apply_residual_conditioned_output_gate(condition, update)
    expected = diagnostic(condition, update)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert mlp.residual_conditioned_output_slope.grad is not None
    assert mlp.residual_conditioned_output_bias.grad is not None


def test_mlp_residual_conditioned_untied_fixed_basis_matches_diagnostic() -> None:
    from examples.nanogpt.analyze_mlp_conditioned_gate_alignment import (
        UntiedFixedBasisBilinearOutputGate,
    )

    torch.manual_seed(503)
    mlp = MLP(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_residual_conditioned_output_gate=True,
            block_fht_mlp_residual_conditioned_output_gate_fixed_basis=True,
            block_fht_mlp_residual_conditioned_output_gate_untied_bases=True,
            block_fht_mlp_residual_conditioned_output_gate_basis_block_size=8,
            block_fht_mlp_residual_conditioned_output_gate_basis_seed=123,
            block_fht_mlp_residual_conditioned_output_gate_update_basis_seed=456,
            block_fht_mlp_residual_conditioned_output_gate_output_basis_seed=789,
        ),
        layer_id=0,
    )
    diagnostic = UntiedFixedBasisBilinearOutputGate(
        8,
        basis_block_size=8,
        condition_seed=123,
        update_seed=456,
        output_seed=789,
    )
    assert mlp.residual_conditioned_output_slope is not None
    assert mlp.residual_conditioned_output_bias is not None
    with torch.no_grad():
        mlp.residual_conditioned_output_slope.copy_(
            torch.randn_like(mlp.residual_conditioned_output_slope)
        )
        mlp.residual_conditioned_output_bias.copy_(
            torch.randn_like(mlp.residual_conditioned_output_bias)
        )
        diagnostic.slope.copy_(mlp.residual_conditioned_output_slope)
        diagnostic.bias.copy_(mlp.residual_conditioned_output_bias)
    condition = torch.randn(2, 3, 8)
    update = torch.randn(2, 3, 8)
    actual = mlp.apply_residual_conditioned_output_gate(condition, update)
    expected = diagnostic(condition, update)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert mlp.residual_conditioned_output_slope.grad is not None
    assert mlp.residual_conditioned_output_bias.grad is not None


def test_mlp_postgelu_conditioned_gate_matches_diagnostic() -> None:
    from examples.nanogpt.analyze_mlp_conditioned_gate_alignment import (
        PostGeluConditionedBilinearOutputGate,
    )

    torch.manual_seed(509)
    mlp = MLP(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_residual_conditioned_output_gate=True,
            block_fht_mlp_residual_conditioned_output_gate_bias=False,
            block_fht_mlp_residual_conditioned_output_gate_fixed_basis=True,
            block_fht_mlp_residual_conditioned_output_gate_untied_bases=True,
            block_fht_mlp_residual_conditioned_output_gate_basis_block_size=8,
            block_fht_mlp_residual_conditioned_output_gate_basis_seed=123,
            block_fht_mlp_residual_conditioned_output_gate_update_basis_seed=456,
            block_fht_mlp_residual_conditioned_output_gate_output_basis_seed=789,
            block_fht_mlp_conditioned_output_gate_source="postgelu",
            block_fht_mlp_conditioned_output_gate_projection_seed=101112,
        ),
        layer_id=0,
    )
    diagnostic = PostGeluConditionedBilinearOutputGate(
        8,
        basis_block_size=8,
        condition_seed=123,
        update_seed=456,
        output_seed=789,
        projection_seed=101112,
    )
    assert mlp.residual_conditioned_output_slope is not None
    with torch.no_grad():
        mlp.residual_conditioned_output_slope.copy_(
            torch.randn_like(mlp.residual_conditioned_output_slope)
        )
        diagnostic.slope.copy_(mlp.residual_conditioned_output_slope)
    activated = torch.randn(2, 3, 32)
    update = torch.randn(2, 3, 8)
    actual = mlp.apply_residual_conditioned_output_gate(
        mlp.postgelu_conditioned_output_condition(activated),
        update,
    )
    expected = diagnostic(activated, update)
    torch.testing.assert_close(actual, expected)


def test_mlp_postgelu_conditioned_gate_is_identity_and_dynamic() -> None:
    torch.manual_seed(521)
    base = MLP(GPTConfig(n_embd=8, n_head=1), layer_id=0)
    conditioned = MLP(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_residual_conditioned_output_gate=True,
            block_fht_mlp_residual_conditioned_output_gate_bias=False,
            block_fht_mlp_residual_conditioned_output_gate_fixed_basis=True,
            block_fht_mlp_residual_conditioned_output_gate_untied_bases=True,
            block_fht_mlp_residual_conditioned_output_gate_basis_block_size=8,
            block_fht_mlp_conditioned_output_gate_source="postgelu",
        ),
        layer_id=0,
    )
    conditioned.load_state_dict(base.state_dict(), strict=False)
    values = torch.randn(2, 3, 8)

    torch.testing.assert_close(conditioned(values), base(values))
    conditioned(values).square().mean().backward()
    assert conditioned.residual_conditioned_output_slope is not None
    assert conditioned.residual_conditioned_output_slope.grad is not None
    assert conditioned.residual_conditioned_output_slope.grad.abs().sum() > 0


def test_mlp_postgelu_hidden_self_gate_is_identity_and_dynamic() -> None:
    torch.manual_seed(523)
    base = MLP(GPTConfig(n_embd=8, n_head=1), layer_id=0)
    conditioned = MLP(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_postgelu_hidden_self_gate=True,
            block_fht_mlp_postgelu_hidden_self_gate_basis_block_size=8,
        ),
        layer_id=0,
    )
    conditioned.load_state_dict(base.state_dict(), strict=False)
    values = torch.randn(2, 3, 8)
    torch.testing.assert_close(conditioned(values), base(values))

    conditioned(values).square().mean().backward()
    assert conditioned.postgelu_hidden_self_slope is not None
    assert conditioned.postgelu_hidden_self_slope.grad is not None
    assert conditioned.postgelu_hidden_self_slope.grad.abs().sum() > 0


def test_mlp_postgelu_hidden_self_gate_selects_layers() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=3,
            n_head=1,
            n_embd=8,
            block_fht_mlp_postgelu_hidden_self_gate=True,
            block_fht_mlp_postgelu_hidden_self_gate_layers=(0, 2),
            block_fht_mlp_postgelu_hidden_self_gate_basis_block_size=8,
        )
    )
    assert model.transformer.h[0].mlp.postgelu_hidden_self_slope is not None
    assert model.transformer.h[1].mlp.postgelu_hidden_self_slope is None
    assert model.transformer.h[2].mlp.postgelu_hidden_self_slope is not None


def test_mlp_postgelu_hidden_self_gate_matches_diagnostic() -> None:
    from examples.nanogpt.analyze_mlp_conditioned_gate_alignment import (
        PostGeluHiddenSelfBilinearGate,
    )

    torch.manual_seed(541)
    mlp = MLP(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_postgelu_hidden_self_gate=True,
            block_fht_mlp_postgelu_hidden_self_gate_scale=0.75,
            block_fht_mlp_postgelu_hidden_self_gate_basis_block_size=8,
            block_fht_mlp_postgelu_hidden_self_gate_condition_basis_seed=123,
            block_fht_mlp_postgelu_hidden_self_gate_update_basis_seed=456,
            block_fht_mlp_postgelu_hidden_self_gate_output_basis_seed=789,
            block_fht_mlp_postgelu_hidden_self_gate_rms_epsilon=1e-5,
        ),
        layer_id=0,
    )
    diagnostic = PostGeluHiddenSelfBilinearGate(
        32,
        scale=0.75,
        basis_block_size=8,
        condition_seed=123,
        update_seed=456,
        output_seed=789,
        rms_epsilon=1e-5,
    )
    assert mlp.postgelu_hidden_self_slope is not None
    with torch.no_grad():
        mlp.postgelu_hidden_self_slope.copy_(
            torch.randn_like(mlp.postgelu_hidden_self_slope)
        )
        diagnostic.slope.copy_(mlp.postgelu_hidden_self_slope)
        diagnostic.bias.zero_()
    activated = torch.randn(2, 3, 32)
    actual = mlp.apply_postgelu_hidden_self_gate(activated)
    expected = diagnostic(activated)
    torch.testing.assert_close(actual, expected)


def test_mlp_multihead_postgelu_hidden_self_gate_matches_diagnostic() -> None:
    from examples.nanogpt.analyze_mlp_conditioned_gate_alignment import (
        MultiHeadPostGeluHiddenSelfBilinearGate,
    )

    torch.manual_seed(547)
    mlp = MLP(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_postgelu_hidden_self_gate=True,
            block_fht_mlp_postgelu_hidden_self_gate_scale=0.75,
            block_fht_mlp_postgelu_hidden_self_gate_heads=2,
            block_fht_mlp_postgelu_hidden_self_gate_head_seed_stride=1000003,
            block_fht_mlp_postgelu_hidden_self_gate_basis_block_size=8,
            block_fht_mlp_postgelu_hidden_self_gate_condition_basis_seed=123,
            block_fht_mlp_postgelu_hidden_self_gate_update_basis_seed=456,
            block_fht_mlp_postgelu_hidden_self_gate_output_basis_seed=789,
            block_fht_mlp_postgelu_hidden_self_gate_rms_epsilon=1e-5,
        ),
        layer_id=0,
    )
    diagnostic = MultiHeadPostGeluHiddenSelfBilinearGate(
        32,
        2,
        scale=0.75,
        basis_block_size=8,
        condition_seed=123,
        update_seed=456,
        output_seed=789,
        head_seed_stride=1000003,
        rms_epsilon=1e-5,
    )
    assert mlp.postgelu_hidden_self_slope is not None
    with torch.no_grad():
        mlp.postgelu_hidden_self_slope.copy_(
            torch.randn_like(mlp.postgelu_hidden_self_slope)
        )
        diagnostic.slope.copy_(mlp.postgelu_hidden_self_slope)
        diagnostic.bias.zero_()
    activated = torch.randn(2, 3, 32)
    actual = mlp.apply_postgelu_hidden_self_gate(activated)
    expected = diagnostic(activated)
    torch.testing.assert_close(actual, expected)
    assert mlp.postgelu_hidden_condition_permutation.shape == (2, 32)
    assert not torch.equal(
        mlp.postgelu_hidden_condition_permutation[0],
        mlp.postgelu_hidden_condition_permutation[1],
    )


def test_mlp_postgelu_conditioned_gate_requires_untied_fixed_bases() -> None:
    with pytest.raises(
        ValueError,
        match="requires fixed_basis and untied_bases",
    ):
        MLP(
            GPTConfig(
                n_embd=8,
                n_head=1,
                block_fht_mlp_residual_conditioned_output_gate=True,
                block_fht_mlp_conditioned_output_gate_source="postgelu",
            ),
            layer_id=0,
        )


def test_mlp_residual_conditioned_untied_bases_require_fixed_basis() -> None:
    with pytest.raises(ValueError, match="untied_bases requires fixed_basis"):
        MLP(
            GPTConfig(
                n_embd=8,
                n_head=1,
                block_fht_mlp_residual_conditioned_output_gate=True,
                block_fht_mlp_residual_conditioned_output_gate_untied_bases=True,
            ),
            layer_id=0,
        )


def test_mlp_residual_conditioned_fixed_basis_gate_validates_block_size() -> None:
    with pytest.raises(ValueError, match="power of two dividing n_embd"):
        MLP(
            GPTConfig(
                n_embd=8,
                n_head=1,
                block_fht_mlp_residual_conditioned_output_gate=True,
                block_fht_mlp_residual_conditioned_output_gate_fixed_basis=True,
                block_fht_mlp_residual_conditioned_output_gate_basis_block_size=3,
            ),
            layer_id=0,
        )


def test_freeze_keeps_block_output_chart_trainable() -> None:
    mlp = MLP(
        GPTConfig(
            n_embd=8,
            n_head=1,
            block_fht_mlp_hidden_block_rotation_stages=1,
            block_fht_mlp_hidden_block_rotation_size=4,
            block_fht_mlp_hidden_block_rotation_basis_size=8,
            block_fht_mlp_hidden_gain=True,
            block_fht_mlp_output_block_rotation_stages=1,
            block_fht_mlp_output_block_rotation_size=4,
            block_fht_mlp_output_block_rotation_basis_size=8,
            block_fht_mlp_residual_output_gain=True,
            block_fht_mlp_residual_conditioned_output_gate=True,
        ),
        layer_id=0,
    )
    freeze_non_block_fht(mlp, train_embeddings=False)
    assert mlp.hidden_block_rotation.coordinates.requires_grad
    assert mlp.hidden_log_gain.requires_grad
    assert mlp.output_block_rotation.coordinates.requires_grad
    assert mlp.residual_output_log_gain.requires_grad
    assert mlp.residual_conditioned_output_slope.requires_grad
    assert mlp.residual_conditioned_output_bias.requires_grad
    assert not mlp.c_proj.weight.requires_grad


def test_shared_hidden_gain_coordinate_scale_preserves_identity_and_scales_tangent() -> None:
    base = MLP(GPTConfig(n_embd=4, n_head=1, block_fht_mlp_shared_hidden_gain=True), layer_id=0)
    scaled = MLP(
        GPTConfig(
            n_embd=4,
            n_head=1,
            block_fht_mlp_shared_hidden_gain=True,
            block_fht_mlp_shared_hidden_gain_scale=4.0,
        ),
        layer_id=0,
    )
    scaled.load_state_dict(base.state_dict())
    values = torch.randn(2, 3, 4)
    torch.testing.assert_close(base(values), scaled(values))
    base(values).sum().backward()
    scaled(values).sum().backward()
    assert base.shared_hidden_log_gain is not None
    assert scaled.shared_hidden_log_gain is not None
    torch.testing.assert_close(
        scaled.shared_hidden_log_gain.grad,
        4.0 * base.shared_hidden_log_gain.grad,
    )
