from __future__ import annotations

import copy
from argparse import Namespace
from unittest.mock import patch

import torch

import examples.nanogpt.muon_pair_vq as muon_pair_vq_module
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.muon_pair_vq import (
    MuonPairVQ,
    MuonPairVQLinear,
    _block_gain_axis_adaptation_counterfactual,
    _block_fht_free_pair_vq_counterfactual,
    _block_fht_fractional_lattice_counterfactual,
    _block_fht_gain_lattice_counterfactual,
    _conditional_polar_pair_diagnostics,
    _decode_conditional_polar_pair_codec,
    _decode_free_pair_vq_rvq2,
    _decode_residual_conditional_polar_pair_codec,
    _decode_polar_pair_codec,
    _decode_rvq_pair_codec,
    _fit_conditional_polar_pair_codec_,
    _fit_free_pair_vq_rvq2_,
    _fit_residual_conditional_polar_pair_codec_,
    _fit_polar_pair_codec_,
    _fit_rvq_pair_codec_,
    _fit_scalar_codebook,
    _fit_stochastic_cartesian_pair_codec_,
    _fit_block_local_uniform_stochastic_cartesian_pair_codec_,
    _fractional_lattice_feedback_layout,
    _fractional_residual_lattice_feedback_layout,
    _fractional_residual_lattice_source_decomposition,
    _nearest_cartesian_codes,
    _nearest_codes_exact,
    _normal_cartesian_codebook,
    _pack_fixed_width_codes,
    _unpack_fixed_width_codes,
)
from examples.nanogpt.train import make_checkpoint, pair_vq_model_kwargs


def make_module(
    *,
    in_features: int = 8,
    out_features: int = 6,
    stages: int = 2,
    seed: int = 101,
    fast_residual: bool = False,
    stochastic_fast_retraction: bool = False,
    stochastic_fast_fht_block_size: int = 0,
    stochastic_fast_uniform_levels: bool = False,
    stochastic_fast_block_local_levels: bool = False,
    error_feedback: bool = False,
    forward_visible_feedback: bool = False,
    fp16_ambient_momentum: bool = False,
    fp16_ambient_reference_probe_steps: tuple[int, ...] = (),
    feedback_codec: str = "cartesian4x4",
    feedback_output_group_size: int = 0,
    feedback_residual_probe_steps: tuple[int, ...] = (),
    feedback_residual_probe_layers: tuple[int, ...] = (),
    feedback_residual_probe_lloyd_iterations: tuple[int, ...] = (),
    lazy_retraction_interval: int = 1,
    lazy_retraction_forced_steps: tuple[int, ...] = (),
) -> MuonPairVQLinear:
    return MuonPairVQLinear(
        in_features,
        out_features,
        bias=False,
        stages=stages,
        base_seed=seed,
        weight_std=0.02,
        layer_id=3,
        fast_residual=fast_residual,
        stochastic_fast_retraction=stochastic_fast_retraction,
        stochastic_fast_fht_block_size=stochastic_fast_fht_block_size,
        stochastic_fast_uniform_levels=stochastic_fast_uniform_levels,
        stochastic_fast_block_local_levels=stochastic_fast_block_local_levels,
        error_feedback=error_feedback,
        forward_visible_feedback=forward_visible_feedback,
        fp16_ambient_momentum=fp16_ambient_momentum,
        fp16_ambient_reference_probe_steps=(
            fp16_ambient_reference_probe_steps
        ),
        feedback_codec=feedback_codec,
        feedback_output_group_size=feedback_output_group_size,
        feedback_residual_probe_steps=feedback_residual_probe_steps,
        feedback_residual_probe_layers=feedback_residual_probe_layers,
        feedback_residual_probe_lloyd_iterations=(
            feedback_residual_probe_lloyd_iterations
        ),
        lazy_retraction_interval=lazy_retraction_interval,
        lazy_retraction_forced_steps=lazy_retraction_forced_steps,
        neighbor_candidates=16,
        code_refresh_interval=8,
    )


def make_optimizer(module: MuonPairVQLinear) -> MuonPairVQ:
    return MuonPairVQ(
        [module], lr=0.01, momentum=0.5, weight_decay=0.1, ns_steps=1
    )


def _reference_dynamic_fit_scalar_codebook(
    values: torch.Tensor,
    *,
    level_count: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(torch.finfo(torch.float32).tiny)
    probabilities = (
        torch.arange(level_count, device=values.device, dtype=torch.float32) + 0.5
    ) / level_count
    levels = mean + std * (2.0**0.5) * torch.erfinv(2.0 * probabilities - 1.0)
    for _iteration in range(iterations):
        codes = torch.bucketize(
            values.contiguous(), (levels[:-1] + levels[1:]) * 0.5
        )
        sums = torch.zeros_like(levels)
        sums.index_add_(0, codes, values)
        counts = torch.bincount(codes, minlength=level_count)
        live = counts > 0
        levels[live] = sums[live] / counts[live]
        levels = levels.sort().values
    codes = torch.bucketize(
        values.contiguous(), (levels[:-1] + levels[1:]) * 0.5
    )
    return levels, codes


def test_static_scalar_centroids_match_dynamic_reference_with_empty_codes() -> None:
    values = torch.cat(
        (
            torch.full((2048,), -1.25),
            torch.linspace(-0.01, 0.01, 4096),
            torch.full((2048,), 2.5),
        )
    )
    reference_levels, reference_codes = _reference_dynamic_fit_scalar_codebook(
        values,
        level_count=256,
        iterations=4,
    )
    levels, codes = _fit_scalar_codebook(
        values,
        level_count=256,
        iterations=4,
    )
    assert torch.equal(levels, reference_levels)
    assert torch.equal(codes, reference_codes)


def test_stochastic_cartesian_retraction_is_replay_exact_and_unbiased() -> None:
    values = torch.linspace(-0.9, 0.9, 8192).reshape(-1, 2)
    values[0] = torch.tensor([-7.0, 11.0])
    values[-1] = torch.tensor([13.0, -9.0])
    levels_a = torch.zeros(2, 16)
    levels_b = torch.zeros(2, 16)
    codes_a = torch.zeros(values.shape[0], dtype=torch.uint8)
    codes_b = torch.zeros(values.shape[0], dtype=torch.uint8)
    changes_a, diagnostics_a = _fit_stochastic_cartesian_pair_codec_(
        values, levels_a, codes_a, seed=1901
    )
    changes_b, diagnostics_b = _fit_stochastic_cartesian_pair_codec_(
        values, levels_b, codes_b, seed=1901
    )
    assert changes_a > 0
    assert changes_b == changes_a
    assert torch.equal(codes_b, codes_a)
    torch.testing.assert_close(levels_b, levels_a, rtol=0.0, atol=0.0)
    assert diagnostics_b == diagnostics_a
    assert diagnostics_a["stochastic_fast_expected_bias_recovery"] > 0.999999
    assert diagnostics_a["stochastic_fast_boundary_clipped_values"] == 0
    torch.testing.assert_close(levels_a[:, 0], values.amin(dim=0))
    torch.testing.assert_close(levels_a[:, -1], values.amax(dim=0))


def test_stochastic_fast_retraction_resume_is_exact_without_rng_state() -> None:
    module = make_module(
        fast_residual=True,
        stochastic_fast_retraction=True,
        stochastic_fast_fht_block_size=16,
        error_feedback=True,
    )
    optimizer = make_optimizer(module)
    torch.manual_seed(1903)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = make_module(
        fast_residual=True,
        stochastic_fast_retraction=True,
        stochastic_fast_fht_block_size=16,
        error_feedback=True,
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    assert torch.equal(restored.fast_codes, module.fast_codes)
    assert not any("rng" in key for key in restored.state_dict())


def test_stochastic_fast_fht_retraction_roundtrips_and_keeps_byte_accounting() -> None:
    raw = make_module(
        fast_residual=True,
        stochastic_fast_retraction=True,
        error_feedback=True,
    )
    transformed = make_module(
        fast_residual=True,
        stochastic_fast_retraction=True,
        stochastic_fast_fht_block_size=16,
        error_feedback=True,
    )
    transformed.load_state_dict(raw.state_dict(), strict=True)
    before = transformed.persistent_codec_bytes
    torch.manual_seed(1907)
    requested = transformed.weight.detach() + 0.01 * torch.randn_like(
        transformed.weight
    )
    diagnostics = transformed.project_requested_weight_(
        requested, refresh_codes=True
    )
    assert diagnostics["stochastic_fast_fht_block_size"] == 16
    assert diagnostics["stochastic_fast_expected_bias_recovery"] > 0.999999
    assert diagnostics["stochastic_fast_boundary_clipped_values"] == 0
    assert transformed.persistent_codec_bytes == before
    torch.testing.assert_close(
        transformed.decode_weight(), transformed.weight, rtol=0.0, atol=0.0
    )


def test_uniform_stochastic_levels_are_bracketed_and_reduce_tail_variance() -> None:
    torch.manual_seed(1913)
    values = torch.randn(32768, 2)
    values[0] = torch.tensor([-8.0, 9.0])
    values[1] = torch.tensor([7.0, -10.0])
    lloyd_levels = torch.zeros(2, 16)
    uniform_levels = torch.zeros(2, 16)
    lloyd_codes = torch.zeros(values.shape[0], dtype=torch.uint8)
    uniform_codes = torch.zeros_like(lloyd_codes)
    _, lloyd = _fit_stochastic_cartesian_pair_codec_(
        values, lloyd_levels, lloyd_codes, seed=1917
    )
    _, uniform = _fit_stochastic_cartesian_pair_codec_(
        values,
        uniform_levels,
        uniform_codes,
        seed=1917,
        uniform_levels=True,
    )
    assert uniform["stochastic_fast_expected_bias_recovery"] > 0.999999
    assert uniform["stochastic_fast_boundary_clipped_values"] == 0
    assert uniform["stochastic_fast_uniform_levels"] == 1
    assert (
        uniform["stochastic_fast_sampling_variance_ratio"]
        < lloyd["stochastic_fast_sampling_variance_ratio"]
    )
    expected_steps = (
        values.amax(dim=0) - values.amin(dim=0)
    ) / 15.0
    torch.testing.assert_close(
        uniform_levels[:, 1] - uniform_levels[:, 0], expected_steps
    )
    torch.testing.assert_close(
        uniform_levels[:, 1:] - uniform_levels[:, :-1],
        expected_steps[:, None].expand(-1, 15),
    )


def test_block_local_uniform_stochastic_levels_reduce_global_tail_penalty() -> None:
    torch.manual_seed(1931)
    values = torch.randn(4096, 2)
    values[0] = torch.tensor([-16.0, 15.0])
    global_levels = torch.zeros(2, 16)
    global_codes = torch.zeros(values.shape[0], dtype=torch.uint8)
    local_codes = torch.zeros_like(global_codes)
    local_bounds = torch.zeros(values.shape[0] // 128, 2, 2)
    _, global_diagnostics = _fit_stochastic_cartesian_pair_codec_(
        values,
        global_levels,
        global_codes,
        seed=1933,
        uniform_levels=True,
    )
    _, local_diagnostics = (
        _fit_block_local_uniform_stochastic_cartesian_pair_codec_(
            values,
            local_bounds,
            local_codes,
            pairs_per_group=128,
            seed=1933,
        )
    )
    assert local_diagnostics["stochastic_fast_expected_bias_recovery"] > 0.999999
    assert local_diagnostics["stochastic_fast_boundary_clipped_values"] == 0
    assert local_diagnostics["stochastic_fast_block_local_groups"] == 32
    assert (
        local_diagnostics["stochastic_fast_sampling_variance_ratio"]
        < 0.5 * global_diagnostics["stochastic_fast_sampling_variance_ratio"]
    )
    torch.testing.assert_close(local_bounds[:, :, 0], values.reshape(32, 128, 2).amin(dim=1))
    torch.testing.assert_close(local_bounds[:, :, 1], values.reshape(32, 128, 2).amax(dim=1))


def test_block_local_stochastic_fast_state_is_compact_and_resume_exact() -> None:
    module = make_module(
        fast_residual=True,
        stochastic_fast_retraction=True,
        stochastic_fast_fht_block_size=16,
        stochastic_fast_uniform_levels=True,
        stochastic_fast_block_local_levels=True,
        error_feedback=True,
    )
    expected_bounds_bytes = (module.element_count // 16) * 4 * 4
    assert module.fast_group_bounds.numel() * 4 == expected_bounds_bytes
    optimizer = make_optimizer(module)
    torch.manual_seed(1937)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = make_module(
        fast_residual=True,
        stochastic_fast_retraction=True,
        stochastic_fast_fht_block_size=16,
        stochastic_fast_uniform_levels=True,
        stochastic_fast_block_local_levels=True,
        error_feedback=True,
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    assert torch.equal(restored.fast_codes, module.fast_codes)
    assert torch.equal(restored.fast_group_bounds, module.fast_group_bounds)


def make_fractional_module(
    *,
    seed: int = 1721,
    feedback_codec: str = "fractional_lattice_q7q8_b32_p25",
) -> MuonPairVQLinear:
    return MuonPairVQLinear(
        16,
        16,
        bias=False,
        stages=1,
        base_seed=seed,
        weight_std=0.02,
        layer_id=2,
        fast_residual=False,
        error_feedback=True,
        feedback_codec=feedback_codec,
        feedback_output_group_size=0,
        neighbor_candidates=16,
        code_refresh_interval=8,
    )


def make_adaptive_fractional_module(
    *,
    in_features: int,
    out_features: int,
    seed: int,
    feedback_codec: str = "fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
) -> MuonPairVQLinear:
    return MuonPairVQLinear(
        in_features,
        out_features,
        bias=False,
        stages=1,
        base_seed=seed,
        weight_std=0.02,
        layer_id=2,
        fast_residual=False,
        error_feedback=True,
        feedback_codec=feedback_codec,
        neighbor_candidates=16,
        code_refresh_interval=8,
    )


def test_residual_conditional_polar_probe_reports_geometry_and_convergence() -> None:
    module = make_module(
        error_feedback=True,
        feedback_codec="conditional_polar16x16_rvq2",
        feedback_residual_probe_steps=(0,),
        feedback_residual_probe_lloyd_iterations=(3, 6),
    )
    optimizer = make_optimizer(module)
    torch.manual_seed(1701)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()[0]
    assert 0.0 <= diagnostics["feedback_residual_polar_angular_error_fraction"] <= 1.0
    assert 0.0 <= diagnostics["feedback_residual_polar_radial_error_fraction"] <= 1.0
    assert diagnostics["feedback_residual_active_codes"] > 0
    assert diagnostics["feedback_residual_lloyd3_codec_energy_recovery"] > 0.0
    assert diagnostics["feedback_residual_lloyd6_codec_energy_recovery"] > 0.0
    assert diagnostics["feedback_residual_center_energy_ratio"] >= 0.0


def test_free_pair_vq_rvq2_closes_gaussian_residual_with_free_directions() -> None:
    torch.manual_seed(1703)
    vectors = torch.randn(8192, 2)
    codebooks = torch.zeros(2, 256, 2)
    codes = torch.zeros(2, vectors.shape[0], dtype=torch.uint8)
    _fit_free_pair_vq_rvq2_(
        vectors,
        codebooks,
        codes,
        neighbor_candidates=16,
    )
    decoded = _decode_free_pair_vq_rvq2(codebooks, codes)
    recovery = 1.0 - float((vectors - decoded).square().sum()) / float(
        vectors.square().sum()
    )
    coarse = codebooks[0].index_select(0, codes[0].long())
    residual_target = vectors - coarse
    residual = codebooks[1].index_select(0, codes[1].long())
    residual_recovery = 1.0 - float(
        (residual_target - residual).square().sum()
    ) / float(residual_target.square().sum())
    assert recovery > 0.999
    assert residual_recovery > 0.985
    assert int(codes[0].unique().numel()) > 200
    assert int(codes[1].unique().numel()) > 200


def test_block_fht_free_pair_probe_preserves_energy_and_reports_capacity() -> None:
    torch.manual_seed(1704)
    vectors = torch.randn(4096, 2)
    diagnostics = _block_fht_free_pair_vq_counterfactual(
        vectors,
        block_size=16,
        seed=1704,
    )
    assert diagnostics["parseval_relative_error"] < 1e-6
    assert diagnostics["full_recovery"] > 0.99
    assert diagnostics["residual_recovery"] > 0.98
    assert diagnostics["stage1_active_codes"] > 192
    assert diagnostics["stage2_active_codes"] > 192


def test_block_fht_gain_lattice_reports_physical_rate_and_recovery() -> None:
    torch.manual_seed(1706)
    vectors = torch.randn(4096, 2)
    diagnostics = _block_fht_gain_lattice_counterfactual(
        vectors,
        block_size=64,
        coordinate_bits=6,
        seed=1706,
    )
    assert diagnostics["parseval_relative_error"] < 1e-6
    assert diagnostics["physical_bits_per_weight"] == 6.125
    assert diagnostics["full_recovery"] > 0.99
    assert diagnostics["coordinate_active_codes"] == 64
    # The fixture contains exactly 128 gain blocks, so occupancy cannot exceed
    # 128 even with a 256-level gain codebook.  Requiring a majority of those
    # blocks to remain distinct still catches a collapsed gain quantizer.
    assert diagnostics["gain_active_codes"] > 64


def test_block_gain_axis_adaptation_reports_all_oracle_arms() -> None:
    torch.manual_seed(1707)
    base = torch.randn(1024, 8)
    # Correlate the fixture axes so the KLT has a real gauge to diagnose.
    source = base + 0.4 * base.roll(1, dims=1)
    diagnostics = _block_gain_axis_adaptation_counterfactual(
        source.reshape(-1, 2),
        block_size=8,
        coordinate_bits=6,
        seed=1707,
    )
    assert diagnostics["physical_bits_per_weight"] == 7.0
    assert diagnostics["fht_parseval_relative_error"] < 1e-6
    assert diagnostics["klt_parseval_relative_error"] < 1e-6
    for arm in ("fht_global", "fht_axis", "klt_global", "klt_axis"):
        assert diagnostics[f"{arm}_full_recovery"] > 0.99
        assert diagnostics[f"{arm}_coordinate_active_codes_min"] > 48
        assert diagnostics[f"{arm}_gain_active_codes"] > 64


def test_block_fht_fractional_lattice_rate_and_monotonic_recovery() -> None:
    torch.manual_seed(1708)
    vectors = torch.randn(4096, 2)
    diagnostics = _block_fht_fractional_lattice_counterfactual(
        vectors,
        block_size=32,
        base_coordinate_bits=7,
        refinement_fractions=(0.125, 0.5),
        seed=1708,
    )
    assert diagnostics["parseval_relative_error"] < 1e-6
    assert diagnostics["base_active_codes"] == 128
    assert diagnostics["refined_active_codes"] > 192
    assert diagnostics["p0p125_physical_bits_per_weight"] == 7.40625
    assert diagnostics["p0p5_physical_bits_per_weight"] == 7.78125
    assert (
        diagnostics["base_full_recovery"]
        < diagnostics["p0p125_full_recovery"]
        < diagnostics["p0p5_full_recovery"]
        < diagnostics["uniform_refined_full_recovery"]
    )


def test_fixed_width_pack_roundtrip_is_bit_exact() -> None:
    torch.manual_seed(1710)
    for bits, count in ((1, 19), (7, 256)):
        codes = torch.randint(0, 1 << bits, (count,), dtype=torch.uint8)
        packed = _pack_fixed_width_codes(codes, bits=bits)
        decoded = _unpack_fixed_width_codes(
            packed,
            bits=bits,
            count=count,
        )
        assert torch.equal(decoded, codes)
        assert packed.numel() == ((count + 7) // 8) * bits


def test_fractional_lattice_feedback_uses_exact_packed_rate() -> None:
    module = make_fractional_module()
    optimizer = make_optimizer(module)
    torch.manual_seed(1711)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()[0]
    state = optimizer.state[module.weight]
    layout = _fractional_lattice_feedback_layout(module.element_count)
    assert layout["total_bytes"] == 241
    assert 8.0 * layout["total_bytes"] / module.element_count == 7.53125
    assert state["feedback_levels"].shape == (640,)
    assert state["feedback_levels"].dtype == torch.float32
    assert state["feedback_codes"].shape == (241,)
    assert state["feedback_codes"].dtype == torch.uint8
    assert module.compact_feedback_bytes == 241 + 640 * 4
    assert "feedback_center" not in state
    assert diagnostics["feedback_codec_energy_recovery"] > 0.99
    assert not any(
        value.numel() == module.element_count
        and value.dtype == torch.float32
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def test_fractional_lattice_feedback_resume_is_bit_exact_for_next_step() -> None:
    torch.manual_seed(1712)
    module = make_fractional_module(seed=1723)
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_fractional_module(seed=1725)
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(
        restored_state["feedback_codes"], original_state["feedback_codes"]
    )
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )


def test_fractional_residual_lattice_uses_exact_rate_and_improves_recovery() -> None:
    base = make_fractional_module(seed=1727)
    residual = make_fractional_module(
        seed=1727,
        feedback_codec="fractional_lattice_q7q8_b32_p25_rq4",
    )
    base_optimizer = make_optimizer(base)
    residual_optimizer = make_optimizer(residual)
    torch.manual_seed(1729)
    gradient = torch.randn_like(base.weight)
    base.weight.grad = gradient.clone()
    residual.weight.grad = gradient.clone()
    base_optimizer.step()
    residual_optimizer.step()
    base_diagnostics = base_optimizer.consume_diagnostics()[0]
    residual_diagnostics = residual_optimizer.consume_diagnostics()[0]
    state = residual_optimizer.state[residual.weight]
    layout = _fractional_residual_lattice_feedback_layout(
        residual.element_count
    )
    assert layout["total_bytes"] == 377
    assert 8.0 * layout["total_bytes"] / residual.element_count == 11.78125
    assert state["feedback_levels"].shape == (912,)
    assert state["feedback_codes"].shape == (377,)
    assert residual.compact_feedback_bytes == 377 + 912 * 4
    assert (
        residual_diagnostics["feedback_codec_energy_recovery"]
        > base_diagnostics["feedback_codec_energy_recovery"]
    )
    assert residual_diagnostics["feedback_codec_energy_recovery"] > 0.9999
    assert not any(
        value.numel() == residual.element_count
        and value.dtype == torch.float32
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def test_fractional_residual_lattice_resume_is_bit_exact_for_next_step() -> None:
    torch.manual_seed(1731)
    module = make_fractional_module(
        seed=1733,
        feedback_codec="fractional_lattice_q7q8_b32_p25_rq4",
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_fractional_module(
        seed=1735,
        feedback_codec="fractional_lattice_q7q8_b32_p25_rq4",
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(
        restored_state["feedback_codes"], original_state["feedback_codes"]
    )
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )


def test_family_adaptive_residual_lattice_uses_q5_only_for_cfc() -> None:
    q4_cfc = MuonPairVQLinear(
        16,
        32,
        bias=False,
        stages=1,
        base_seed=1737,
        weight_std=0.02,
        layer_id=2,
        fast_residual=False,
        error_feedback=True,
        feedback_codec="fractional_lattice_q7q8_b32_p25_rq4",
        neighbor_candidates=16,
        code_refresh_interval=8,
    )
    adaptive_cfc = make_adaptive_fractional_module(
        in_features=16,
        out_features=32,
        seed=1737,
    )
    adaptive_cproj = make_adaptive_fractional_module(
        in_features=32,
        out_features=16,
        seed=1737,
    )
    q4_optimizer = make_optimizer(q4_cfc)
    adaptive_optimizer = make_optimizer(adaptive_cfc)
    torch.manual_seed(1739)
    gradient = torch.randn_like(q4_cfc.weight)
    q4_cfc.weight.grad = gradient.clone()
    adaptive_cfc.weight.grad = gradient.clone()
    q4_optimizer.step()
    adaptive_optimizer.step()
    q4_diagnostics = q4_optimizer.consume_diagnostics()[0]
    adaptive_diagnostics = adaptive_optimizer.consume_diagnostics()[0]

    cfc_layout = _fractional_residual_lattice_feedback_layout(
        adaptive_cfc.element_count,
        coordinate_bits=5,
    )
    cproj_layout = _fractional_residual_lattice_feedback_layout(
        adaptive_cproj.element_count,
        coordinate_bits=4,
    )
    assert cfc_layout["total_bytes"] == 818
    assert cproj_layout["total_bytes"] == 754
    assert 8.0 * cfc_layout["total_bytes"] / adaptive_cfc.element_count == 12.78125
    assert 8.0 * cproj_layout["total_bytes"] / adaptive_cproj.element_count == 11.78125
    assert adaptive_cfc.feedback_level_shape == (928,)
    assert adaptive_cproj.feedback_level_shape == (912,)
    assert adaptive_cfc.feedback_code_shape == (818,)
    assert adaptive_cproj.feedback_code_shape == (754,)
    assert (
        adaptive_diagnostics["feedback_codec_energy_recovery"]
        > q4_diagnostics["feedback_codec_energy_recovery"]
    )
    assert adaptive_diagnostics["feedback_codec_energy_recovery"] > 0.9999


def test_family_adaptive_residual_lattice_resume_is_bit_exact() -> None:
    torch.manual_seed(1741)
    module = make_adaptive_fractional_module(
        in_features=16,
        out_features=32,
        seed=1743,
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = make_adaptive_fractional_module(
        in_features=16,
        out_features=32,
        seed=1745,
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(
        restored_state["feedback_codes"], original_state["feedback_codes"]
    )
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )


def test_b64_lloyd16_cfc_codec_uses_family_specific_rate_and_replays() -> None:
    codec = "fractional_lattice_q7q8_b32_p25_rq4_cfcq5b64l16"
    cfc = make_adaptive_fractional_module(
        in_features=16,
        out_features=32,
        seed=1747,
        feedback_codec=codec,
    )
    cproj = make_adaptive_fractional_module(
        in_features=32,
        out_features=16,
        seed=1749,
        feedback_codec=codec,
    )
    assert cfc.feedback_residual_lattice_coordinate_bits == 5
    assert cfc.feedback_residual_lattice_block_size == 64
    assert cfc.feedback_residual_lattice_lloyd_iterations == 16
    assert cproj.feedback_residual_lattice_coordinate_bits == 4
    assert cproj.feedback_residual_lattice_block_size == 32
    assert cproj.feedback_residual_lattice_lloyd_iterations == 4
    assert cfc.feedback_code_shape == (810,)
    assert cproj.feedback_code_shape == (754,)
    assert 8.0 * cfc.feedback_code_shape[0] / cfc.element_count == 12.65625
    assert 8.0 * cproj.feedback_code_shape[0] / cproj.element_count == 11.78125

    optimizer = make_optimizer(cfc)
    torch.manual_seed(1751)
    for _step in range(2):
        cfc.weight.grad = torch.randn_like(cfc.weight)
        optimizer.step()
    model_state = copy.deepcopy(cfc.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    restored = make_adaptive_fractional_module(
        in_features=16,
        out_features=32,
        seed=1753,
        feedback_codec=codec,
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(cfc.weight)
    cfc.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, cfc.weight, rtol=0.0, atol=0.0)
    assert torch.equal(
        restored_optimizer.state[restored.weight]["feedback_codes"],
        optimizer.state[cfc.weight]["feedback_codes"],
    )


def test_fractional_residual_source_decomposition_reports_fixed_source_axes() -> None:
    torch.manual_seed(1747)
    values = torch.randn(256, 2)
    diagnostics = _fractional_residual_lattice_source_decomposition(
        values,
        seed=1749,
        block_sizes=(16, 32),
        coordinate_bits=(4, 5),
        lloyd_iterations=(2, 4),
        axis_block_size=32,
        axis_coordinate_bits=5,
    )
    assert 0.0 < diagnostics["innovation_energy_ratio"] < 1.0
    assert diagnostics["b32_q5_lloyd4_quantgain_innovation_recovery"] > 0.0
    assert diagnostics["b32_q5_lloyd4_exactgain_innovation_recovery"] > 0.0
    assert diagnostics["b32_q5_lloyd4_quantgain_axis_innovation_recovery"] > 0.0


def test_fractional_residual_source_probe_runs_only_on_selected_cfc_layer() -> None:
    module = MuonPairVQLinear(
        16,
        32,
        bias=False,
        stages=1,
        base_seed=1751,
        weight_std=0.02,
        layer_id=2,
        error_feedback=True,
        feedback_codec="fractional_lattice_q7q8_b32_p25_rq4_cfcq5",
        feedback_residual_probe_steps=(0,),
        feedback_residual_probe_layers=(2,),
        feedback_residual_probe_lloyd_iterations=(2,),
        feedback_lattice_probe_block_sizes=(16, 32),
        feedback_lattice_probe_coordinate_bits=(4, 5),
        feedback_axis_adaptation_probe_block_size=32,
        feedback_axis_adaptation_probe_coordinate_bits=5,
        neighbor_candidates=16,
        code_refresh_interval=8,
    )
    optimizer = make_optimizer(module)
    torch.manual_seed(1753)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()[0]
    assert "feedback_source_innovation_energy_ratio" in diagnostics
    assert (
        "feedback_source_b32_q5_lloyd2_quantgain_axis_innovation_recovery"
        in diagnostics
    )


def test_free_pair_vq_rvq2_feedback_is_compact_and_reports_exact_regret() -> None:
    torch.manual_seed(1705)
    module = make_module(
        stages=1,
        error_feedback=True,
        feedback_codec="free_vq256_rvq2",
        feedback_residual_probe_steps=(0,),
    )
    optimizer = make_optimizer(module)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()[0]
    state = optimizer.state[module.weight]
    assert state["feedback_levels"].shape == (2, 256, 2)
    assert state["feedback_codes"].shape == (2, module.element_count // 2)
    assert state["feedback_codes"].dtype == torch.uint8
    assert "feedback_center" not in state
    assert module.compact_feedback_bytes == module.element_count + 4096
    assert diagnostics["feedback_residual_codec_energy_recovery"] > 0.98
    assert diagnostics["feedback_exact_same_codebook_energy_recovery"] >= (
        diagnostics["feedback_codec_energy_recovery"] - 1e-7
    )
    assert diagnostics["feedback_local_assignment_recovery_gap"] >= -1e-7


def test_free_pair_vq_rvq2_resume_is_bit_exact_for_next_step() -> None:
    torch.manual_seed(1707)
    module = make_module(
        stages=1,
        seed=1709,
        error_feedback=True,
        feedback_codec="free_vq256_rvq2",
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(
        stages=1,
        seed=1711,
        error_feedback=True,
        feedback_codec="free_vq256_rvq2",
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(
        restored_state["feedback_codes"], original_state["feedback_codes"]
    )
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )


def test_codec_state_excludes_transient_dense_weight() -> None:
    module = make_module()
    state = module.state_dict()
    assert set(state) == {
        "codebooks",
        "codes",
        "fast_levels",
        "fast_codes",
        "fast_group_bounds",
        "optimizer_step",
    }
    assert state["codebooks"].dtype == torch.float32
    assert state["codes"].dtype == torch.uint8
    assert "weight" not in state
    assert module.persistent_codec_bytes == 2 * 256 * 2 * 4 + 2 * 24 + 8


def test_fast_residual_repairs_small_step_tangent() -> None:
    torch.manual_seed(1019)
    module = make_module(stages=2, fast_residual=True)
    requested = module.weight + 1e-4 * torch.randn_like(module.weight)
    diagnostics = module.project_requested_weight_(requested, refresh_codes=True)
    assert diagnostics["requested_step_energy_recovery"] > 0.8
    assert diagnostics["requested_update_cosine"] > 0.9
    assert diagnostics["fast_code_changes"] > 0
    assert module.fast_levels.shape == (2, 16)
    assert module.fast_codes.numel() == module.element_count // 2


def test_projection_moves_toward_request_and_refreshes_codes() -> None:
    module = make_module(stages=1)
    old_codes = module.codes.clone()
    requested = module.weight + 0.05 * torch.randn_like(module.weight)
    diagnostics = module.project_requested_weight_(requested, refresh_codes=True)
    assert diagnostics["requested_step_energy_recovery"] > 0.0
    assert diagnostics["requested_update_cosine"] > 0.0
    assert diagnostics["code_changes"] > 0
    assert not torch.equal(module.codes, old_codes)


def test_optimizer_state_is_only_compact_code_momentum() -> None:
    module = make_module()
    optimizer = make_optimizer(module)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    state = optimizer.state[module.weight]
    assert set(state) == {"compact_momentum"}
    assert state["compact_momentum"].shape == module.codebooks.shape
    assert state["compact_momentum"].numel() == 2 * 256 * 2


def test_fp16_ambient_momentum_uses_persisted_rounded_recurrence() -> None:
    torch.manual_seed(20260822)
    module = make_module(fp16_ambient_momentum=True)
    optimizer = make_optimizer(module)
    first_gradient = torch.randn_like(module.weight)
    module.weight.grad = first_gradient.clone()
    optimizer.step()
    state = optimizer.state[module.weight]
    assert set(state) == {"ambient_momentum"}
    assert state["ambient_momentum"].dtype == torch.float16
    assert state["ambient_momentum"].shape == module.weight.shape
    expected_first = first_gradient.to(torch.float16)
    torch.testing.assert_close(
        state["ambient_momentum"], expected_first, rtol=0.0, atol=0.0
    )

    second_gradient = torch.randn_like(module.weight)
    expected_second = (
        0.5 * expected_first.float() + second_gradient.float()
    ).to(torch.float16)
    module.weight.grad = second_gradient.clone()
    optimizer.step()
    torch.testing.assert_close(
        state["ambient_momentum"], expected_second, rtol=0.0, atol=0.0
    )
    assert "compact_momentum" not in state
    assert module.optimizer_momentum_bytes == module.element_count * 2


def test_fp16_ambient_momentum_feeds_persisted_state_to_muon(
    monkeypatch,
) -> None:
    torch.manual_seed(20260824)
    module = make_module(fp16_ambient_momentum=True)
    optimizer = make_optimizer(module)
    observed: list[torch.Tensor] = []

    def capture_muon(requested_gradient: torch.Tensor, *, steps: int):
        assert steps == 1
        observed.append(requested_gradient.clone())
        return torch.zeros_like(requested_gradient)

    monkeypatch.setattr(muon_pair_vq_module, "muon_update", capture_muon)
    gradient = torch.randn_like(module.weight)
    expected_momentum = gradient.to(torch.float16)
    expected_request = gradient.float() + 0.5 * expected_momentum.float()
    module.weight.grad = gradient.clone()
    optimizer.step()
    assert len(observed) == 1
    torch.testing.assert_close(
        observed[0], expected_request, rtol=0.0, atol=0.0
    )


def test_fp16_ambient_reference_probe_is_nonpersistent_and_reports_polar() -> None:
    torch.manual_seed(20260825)
    module = make_module(
        fp16_ambient_momentum=True,
        fp16_ambient_reference_probe_steps=(0, 1),
    )
    optimizer = make_optimizer(module)
    for expected_step in (0, 1):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()
        assert len(diagnostics) == 1
        row = diagnostics[0]
        assert row["optimizer_step"] == expected_step
        assert row["ambient_reference_polar_cosine"] > 0.999
        assert row["ambient_reference_positive_line_recovery"] > 0.998
        assert row["ambient_reference_prepolar_cosine"] > 0.999
    state = optimizer.state[module.weight]
    assert set(state) == {"ambient_momentum"}
    serialized_state = optimizer.state_dict()["state"]
    assert len(serialized_state) == 1
    assert set(next(iter(serialized_state.values()))) == {"ambient_momentum"}


def test_pair_coded_feedback_conserves_discarded_motion_compactly() -> None:
    torch.manual_seed(127)
    module = make_module(stages=1, error_feedback=True)
    optimizer = make_optimizer(module)
    diagnostics = None
    for _step in range(12):
        module.weight.grad = 1e-3 * torch.randn_like(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["error_feedback"] == 1
    assert diagnostics["feedback_codec_energy_recovery"] > 0.95
    assert diagnostics["conserved_requested_step_energy_recovery"] > 0.90
    assert diagnostics["feedback_code_changes"] > 0
    state = optimizer.state[module.weight]
    assert set(state) == {
        "compact_momentum",
        "feedback_levels",
        "feedback_codes",
    }
    assert state["feedback_levels"].shape == (2, 16)
    assert state["feedback_levels"].dtype == torch.float32
    assert state["feedback_codes"].shape == (module.element_count // 2,)
    assert state["feedback_codes"].dtype == torch.uint8
    assert module.compact_feedback_bytes == module.element_count // 2 + 128
    assert all(value.numel() != module.element_count for value in state.values())


def test_pair_coded_feedback_resume_is_bit_exact_for_next_step() -> None:
    torch.manual_seed(131)
    module = make_module(
        stages=1,
        seed=137,
        error_feedback=True,
        feedback_output_group_size=2,
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(
        stages=1,
        seed=139,
        error_feedback=True,
        feedback_output_group_size=2,
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    restored_state = restored_optimizer.state[restored.weight]
    assert restored_state["feedback_codes"].dtype == torch.uint8
    assert restored_state["feedback_levels"].dtype == torch.float32

    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    assert torch.equal(restored.codes, module.codes)
    torch.testing.assert_close(
        restored.codebooks, module.codebooks, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(
        restored_state["feedback_codes"], original_state["feedback_codes"]
    )
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )


def test_forward_visible_feedback_materializes_compact_virtual_center() -> None:
    torch.manual_seed(20260822)
    module = make_module(
        in_features=32,
        out_features=32,
        stages=1,
        error_feedback=True,
        forward_visible_feedback=True,
    )
    optimizer = make_optimizer(module)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()[0]
    state = optimizer.state[module.weight]
    base = module.decode_weight()
    feedback = module.decode_feedback(
        state["feedback_levels"], state["feedback_codes"]
    ).reshape_as(module.weight)
    assert torch.count_nonzero(feedback) > 0
    torch.testing.assert_close(
        module.weight, base + feedback, rtol=0.0, atol=0.0
    )
    assert diagnostics["forward_visible_feedback"] == 1
    assert diagnostics["projection_target_includes_prior_feedback"] == 0
    assert diagnostics["virtual_weight_matches_compact_state"] == 1
    assert module.storage_accounting()["persistent_training_bytes"] == (
        module.persistent_codec_bytes
        + module.compact_momentum_bytes
        + module.compact_feedback_bytes
    )


def test_forward_visible_feedback_resume_reconstructs_virtual_weight_exactly() -> None:
    torch.manual_seed(20260823)
    module = make_module(
        in_features=32,
        out_features=32,
        stages=1,
        seed=20260824,
        error_feedback=True,
        forward_visible_feedback=True,
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    expected_virtual = module.weight.detach().clone()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(
        in_features=32,
        out_features=32,
        stages=1,
        seed=20260825,
        error_feedback=True,
        forward_visible_feedback=True,
    )
    restored.load_state_dict(model_state, strict=True)
    assert not torch.equal(restored.weight, expected_virtual)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    torch.testing.assert_close(
        restored.weight, expected_virtual, rtol=0.0, atol=0.0
    )

    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(
        restored.weight, module.weight, rtol=0.0, atol=0.0
    )


def test_lazy_retraction_carries_ambient_updates_until_compact_boundary() -> None:
    torch.manual_seed(20260826)
    module = make_module(
        in_features=32,
        out_features=32,
        stages=1,
        error_feedback=True,
        forward_visible_feedback=True,
        fp16_ambient_momentum=True,
        lazy_retraction_interval=8,
    )
    optimizer = make_optimizer(module)
    initial_codes = module.codes.detach().clone()
    initial_codebooks = module.codebooks.detach().clone()

    for expected_step in range(1, 8):
        before = module.weight.detach().clone()
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()[0]
        assert diagnostics["lazy_retraction"] == 1
        assert diagnostics["retracted"] == 0
        assert diagnostics["compact_boundary"] == 0
        assert diagnostics["requested_step_energy_recovery"] == 1.0
        assert not torch.equal(module.weight, before)
        assert torch.equal(module.codes, initial_codes)
        torch.testing.assert_close(
            module.codebooks, initial_codebooks, rtol=0.0, atol=0.0
        )
        assert int(module.optimizer_step) == expected_step
        assert optimizer.compact_boundary_ready is False

    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()[0]
    assert diagnostics["retracted"] == 1
    assert diagnostics["compact_boundary"] == 1
    assert diagnostics["refresh_codes"] == 1
    assert int(module.optimizer_step) == 8
    assert optimizer.compact_boundary_ready is True
    state = optimizer.state[module.weight]
    compact_weight = module.decode_weight() + module.decode_feedback(
        state["feedback_levels"], state["feedback_codes"]
    ).reshape_as(module.weight)
    torch.testing.assert_close(
        module.weight, compact_weight, rtol=0.0, atol=0.0
    )


def test_lazy_retraction_forces_registered_boundary_and_resumes_exactly() -> None:
    torch.manual_seed(20260827)
    module = make_module(
        in_features=32,
        out_features=32,
        stages=1,
        seed=20260828,
        error_feedback=True,
        forward_visible_feedback=True,
        fp16_ambient_momentum=True,
        lazy_retraction_interval=8,
        lazy_retraction_forced_steps=(3,),
    )
    optimizer = make_optimizer(module)
    for expected_retracted in (0, 0, 1):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()[0]
        assert diagnostics["retracted"] == expected_retracted
    assert int(module.optimizer_step) == 3
    assert optimizer.compact_boundary_ready is True
    expected_weight = module.weight.detach().clone()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(
        in_features=32,
        out_features=32,
        stages=1,
        seed=20260829,
        error_feedback=True,
        forward_visible_feedback=True,
        fp16_ambient_momentum=True,
        lazy_retraction_interval=8,
        lazy_retraction_forced_steps=(3,),
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    torch.testing.assert_close(
        restored.weight, expected_weight, rtol=0.0, atol=0.0
    )
    assert restored_optimizer.compact_boundary_ready is True


def test_checkpoint_rejects_unretracted_lazy_ambient_carry() -> None:
    class UnretractedOptimizer:
        compact_boundary_ready = False

    try:
        make_checkpoint(
            raw_model=None,  # type: ignore[arg-type]
            optimizer=UnretractedOptimizer(),  # type: ignore[arg-type]
            scaler=None,  # type: ignore[arg-type]
            gpt_config=GPTConfig(),
            next_iter=1,
            best_val_loss=1.0,
            pending_train_data_generator_state=None,
            run_identity={},
            reason="test",
        )
    except RuntimeError as error:
        assert "compact retraction boundary" in str(error)
    else:
        raise AssertionError("nonboundary checkpoint was not rejected")


def test_legacy_error_feedback_forward_remains_base_projection() -> None:
    torch.manual_seed(20260826)
    module = make_module(stages=1, fast_residual=True, error_feedback=True)
    optimizer = make_optimizer(module)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()[0]
    torch.testing.assert_close(
        module.weight, module.decode_weight(), rtol=0.0, atol=0.0
    )
    assert diagnostics["forward_visible_feedback"] == 0
    assert diagnostics["projection_target_includes_prior_feedback"] == 1


def test_output_grouped_feedback_is_compact_and_conserves_motion() -> None:
    torch.manual_seed(139)
    module = make_module(
        stages=1,
        error_feedback=True,
        feedback_output_group_size=2,
    )
    optimizer = make_optimizer(module)
    diagnostics = None
    for _step in range(12):
        row_scale = torch.linspace(0.2, 2.0, module.out_features)[:, None]
        module.weight.grad = row_scale * torch.randn_like(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["feedback_codec_energy_recovery"] > 0.95
    assert diagnostics["conserved_requested_step_energy_recovery"] > 0.90
    state = optimizer.state[module.weight]
    assert state["feedback_levels"].shape == (3, 2, 16)
    assert state["feedback_codes"].dtype == torch.uint8
    expected_bytes = module.element_count // 2 + 3 * 2 * 16 * 4
    assert module.compact_feedback_bytes == expected_bytes
    assert all(value.numel() != module.element_count for value in state.values())


def test_polar_feedback_preserves_pair_direction_at_same_code_rate() -> None:
    torch.manual_seed(149)
    module = make_module(
        stages=1,
        error_feedback=True,
        feedback_codec="polar32x8",
    )
    optimizer = make_optimizer(module)
    diagnostics = None
    for _step in range(12):
        module.weight.grad = 1e-3 * torch.randn_like(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["feedback_codec_energy_recovery"] > 0.98
    state = optimizer.state[module.weight]
    assert state["feedback_levels"].shape == (8,)
    assert state["feedback_center"].shape == (2,)
    assert state["feedback_codes"].dtype == torch.uint8
    assert module.compact_feedback_bytes == module.element_count // 2 + 40
    assert all(value.numel() != module.element_count for value in state.values())


def test_polar_feedback_resume_is_bit_exact_for_next_step() -> None:
    torch.manual_seed(151)
    module = make_module(
        stages=1,
        seed=157,
        error_feedback=True,
        feedback_codec="polar32x8",
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(
        stages=1,
        seed=163,
        error_feedback=True,
        feedback_codec="polar32x8",
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(restored_state["feedback_codes"], original_state["feedback_codes"])
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        restored_state["feedback_center"],
        original_state["feedback_center"],
        rtol=0.0,
        atol=0.0,
    )


def test_conditional_polar_models_direction_dependent_radius() -> None:
    torch.manual_seed(165)
    angle_indices = torch.randint(0, 32, (32768,))
    directions = torch.stack(
        (
            torch.cos(angle_indices.float() * (2.0 * torch.pi / 32.0)),
            torch.sin(angle_indices.float() * (2.0 * torch.pi / 32.0)),
        ),
        dim=1,
    )
    scales = 1.0 + 0.75 * torch.cos(
        angle_indices.float() * (4.0 * torch.pi / 32.0)
    )
    radii = scales * torch.exp(0.4 * torch.randn_like(scales))
    vectors = radii[:, None] * directions

    shared_levels = torch.zeros(8)
    shared_center = torch.zeros(2)
    shared_codes = torch.zeros(vectors.shape[0], dtype=torch.uint8)
    _fit_polar_pair_codec_(
        vectors, shared_levels, shared_center, shared_codes
    )
    shared = _decode_polar_pair_codec(shared_levels, shared_center, shared_codes)
    shared_recovery = 1.0 - float((vectors - shared).square().sum()) / float(
        vectors.square().sum()
    )

    conditional_levels = torch.zeros(32, 8)
    conditional_center = torch.zeros(2)
    conditional_codes = torch.zeros(vectors.shape[0], dtype=torch.uint8)
    _fit_conditional_polar_pair_codec_(
        vectors, conditional_levels, conditional_center, conditional_codes
    )
    conditional = _decode_conditional_polar_pair_codec(
        conditional_levels, conditional_center, conditional_codes
    )
    conditional_recovery = 1.0 - float(
        (vectors - conditional).square().sum()
    ) / float(vectors.square().sum())
    assert conditional_recovery > 0.985
    assert conditional_recovery > shared_recovery + 0.01


def test_conditional_polar_feedback_state_is_compact() -> None:
    torch.manual_seed(166)
    module = make_module(
        stages=1,
        error_feedback=True,
        feedback_codec="conditional_polar32x8",
    )
    optimizer = make_optimizer(module)
    module.weight.grad = 1e-3 * torch.randn_like(module.weight)
    optimizer.step()
    state = optimizer.state[module.weight]
    assert state["feedback_levels"].shape == (32, 8)
    assert state["feedback_center"].shape == (2,)
    assert state["feedback_codes"].dtype == torch.uint8
    assert module.compact_feedback_bytes == module.element_count // 2 + 1032
    assert all(value.numel() != module.element_count for value in state.values())


def test_conditional_polar16x16_reallocates_bits_to_heavy_tail_radius() -> None:
    torch.manual_seed(171)
    angle_indices = torch.randint(0, 16, (65536,))
    directions = torch.stack(
        (
            torch.cos(angle_indices.float() * (2.0 * torch.pi / 16.0)),
            torch.sin(angle_indices.float() * (2.0 * torch.pi / 16.0)),
        ),
        dim=1,
    )
    radii = torch.exp(0.9 * torch.randn(angle_indices.shape[0]))
    vectors = radii[:, None] * directions

    levels32x8 = torch.zeros(32, 8)
    center32x8 = torch.zeros(2)
    codes32x8 = torch.zeros(vectors.shape[0], dtype=torch.uint8)
    _fit_conditional_polar_pair_codec_(
        vectors, levels32x8, center32x8, codes32x8
    )
    decoded32x8 = _decode_conditional_polar_pair_codec(
        levels32x8, center32x8, codes32x8
    )

    levels16x16 = torch.zeros(16, 16)
    center16x16 = torch.zeros(2)
    codes16x16 = torch.zeros(vectors.shape[0], dtype=torch.uint8)
    _fit_conditional_polar_pair_codec_(
        vectors, levels16x16, center16x16, codes16x16
    )
    decoded16x16 = _decode_conditional_polar_pair_codec(
        levels16x16, center16x16, codes16x16
    )
    recovery32x8 = 1.0 - float((vectors - decoded32x8).square().sum()) / float(
        vectors.square().sum()
    )
    recovery16x16 = 1.0 - float((vectors - decoded16x16).square().sum()) / float(
        vectors.square().sum()
    )
    assert recovery16x16 > 0.98
    assert recovery16x16 > recovery32x8 + 0.02

    module = make_module(
        stages=1,
        error_feedback=True,
        feedback_codec="conditional_polar16x16",
    )
    optimizer = make_optimizer(module)
    module.weight.grad = 1e-3 * torch.randn_like(module.weight)
    optimizer.step()
    state = optimizer.state[module.weight]
    assert state["feedback_levels"].shape == (16, 16)
    assert state["feedback_center"].shape == (2,)
    assert state["feedback_codes"].dtype == torch.uint8
    assert module.compact_feedback_bytes == module.element_count // 2 + 1032


def test_residual_conditional_polar16x16_closes_fine_error() -> None:
    torch.manual_seed(172)
    angle_indices = torch.randint(0, 16, (65536,))
    directions = torch.stack(
        (
            torch.cos(angle_indices.float() * (2.0 * torch.pi / 16.0)),
            torch.sin(angle_indices.float() * (2.0 * torch.pi / 16.0)),
        ),
        dim=1,
    )
    radii = torch.exp(0.9 * torch.randn(angle_indices.shape[0]))
    vectors = radii[:, None] * directions + 0.01 * torch.randn_like(directions)

    coarse_levels = torch.zeros(16, 16)
    coarse_center = torch.zeros(2)
    coarse_codes = torch.zeros(vectors.shape[0], dtype=torch.uint8)
    _fit_conditional_polar_pair_codec_(
        vectors, coarse_levels, coarse_center, coarse_codes
    )
    coarse = _decode_conditional_polar_pair_codec(
        coarse_levels, coarse_center, coarse_codes
    )
    coarse_error = float((vectors - coarse).square().sum())

    levels = torch.zeros(2, 16, 16)
    center = torch.zeros(2, 2)
    codes = torch.zeros(2, vectors.shape[0], dtype=torch.uint8)
    _fit_residual_conditional_polar_pair_codec_(
        vectors, levels, center, codes
    )
    decoded = _decode_residual_conditional_polar_pair_codec(
        levels, center, codes
    )
    residual_error = float((vectors - decoded).square().sum())
    assert residual_error < 0.05 * coarse_error
    assert 1.0 - residual_error / float(vectors.square().sum()) > 0.999

    module = make_module(
        stages=1,
        error_feedback=True,
        feedback_codec="conditional_polar16x16_rvq2",
    )
    optimizer = make_optimizer(module)
    module.weight.grad = 1e-3 * torch.randn_like(module.weight)
    optimizer.step()
    state = optimizer.state[module.weight]
    assert state["feedback_levels"].shape == (2, 16, 16)
    assert state["feedback_center"].shape == (2, 2)
    assert state["feedback_codes"].shape == (2, module.element_count // 2)
    assert state["feedback_codes"].dtype == torch.uint8
    assert module.compact_feedback_bytes == module.element_count + 2064
    assert all(
        value.dtype == torch.uint8 or value.numel() != module.element_count
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def test_residual_conditional_polar16x16_resume_is_exact() -> None:
    torch.manual_seed(174)
    module = make_module(
        stages=1,
        seed=181,
        error_feedback=True,
        feedback_codec="conditional_polar16x16_rvq2",
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(
        stages=1,
        seed=183,
        error_feedback=True,
        feedback_codec="conditional_polar16x16_rvq2",
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(
        restored_state["feedback_codes"], original_state["feedback_codes"]
    )
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        restored_state["feedback_center"],
        original_state["feedback_center"],
        rtol=0.0,
        atol=0.0,
    )


def test_conditional_polar_diagnostics_form_an_orthogonal_decomposition() -> None:
    torch.manual_seed(168)
    vectors = torch.randn(32768, 2)
    levels = torch.zeros(32, 8)
    center = torch.zeros(2)
    codes = torch.zeros(vectors.shape[0], dtype=torch.uint8)
    _fit_conditional_polar_pair_codec_(vectors, levels, center, codes)
    diagnostics = _conditional_polar_pair_diagnostics(
        vectors, levels, center, codes
    )
    assert diagnostics["feedback_polar_decomposition_relative_error"] < 1e-5
    assert abs(
        diagnostics["feedback_polar_angular_error_fraction"]
        + diagnostics["feedback_polar_radial_error_fraction"]
        - 1.0
    ) < 1e-5
    assert diagnostics["feedback_active_codes"] > 200
    assert diagnostics["feedback_active_angles"] == 32
    assert diagnostics["feedback_active_radii"] == 8


def test_rvq_feedback_learns_joint_pair_atoms_at_same_code_rate() -> None:
    torch.manual_seed(167)
    module = make_module(
        stages=1,
        error_feedback=True,
        feedback_codec="rvq4x4",
    )
    optimizer = make_optimizer(module)
    diagnostics = None
    for _step in range(12):
        base = torch.randn(module.out_features, module.in_features // 2, 1)
        paired = torch.cat((base, 0.7 * base + 0.2 * torch.randn_like(base)), dim=2)
        module.weight.grad = 1e-3 * paired.reshape_as(module.weight)
        optimizer.step()
        diagnostics = optimizer.consume_diagnostics()[0]
    assert diagnostics is not None
    assert diagnostics["feedback_codec_energy_recovery"] > 0.98
    state = optimizer.state[module.weight]
    assert state["feedback_levels"].shape == (2, 16, 2)
    assert state["feedback_center"].shape == (2,)
    assert state["feedback_codes"].dtype == torch.uint8
    assert module.compact_feedback_bytes == module.element_count // 2 + 264
    assert all(value.numel() != module.element_count for value in state.values())


def test_rvq_first_fit_uses_both_stage_assignments() -> None:
    torch.manual_seed(171)
    first = torch.randn(16384)
    second = 0.8 * first + 0.3 * torch.randn_like(first)
    vectors = torch.stack((first, second), dim=1)
    codebooks = torch.zeros(2, 16, 2)
    center = torch.zeros(2)
    codes = torch.zeros(vectors.shape[0], dtype=torch.uint8)
    _fit_rvq_pair_codec_(vectors, codebooks, center, codes)
    decoded = _decode_rvq_pair_codec(codebooks, center, codes)
    recovery = 1.0 - float((vectors - decoded).square().sum()) / float(
        vectors.square().sum()
    )
    assert recovery > 0.98
    assert int((codes % 16).unique().numel()) > 8


def test_rvq_feedback_resume_is_bit_exact_for_next_step() -> None:
    torch.manual_seed(173)
    module = make_module(
        stages=1,
        seed=179,
        error_feedback=True,
        feedback_codec="rvq4x4",
    )
    optimizer = make_optimizer(module)
    for _step in range(3):
        module.weight.grad = torch.randn_like(module.weight)
        optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(
        stages=1,
        seed=181,
        error_feedback=True,
        feedback_codec="rvq4x4",
    )
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    original_state = optimizer.state[module.weight]
    restored_state = restored_optimizer.state[restored.weight]
    assert torch.equal(restored_state["feedback_codes"], original_state["feedback_codes"])
    torch.testing.assert_close(
        restored_state["feedback_levels"],
        original_state["feedback_levels"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        restored_state["feedback_center"],
        original_state["feedback_center"],
        rtol=0.0,
        atol=0.0,
    )


def test_model_and_optimizer_resume_are_bit_exact_for_next_step() -> None:
    torch.manual_seed(103)
    module = make_module(seed=107)
    optimizer = make_optimizer(module)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(seed=109)
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)

    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    assert torch.equal(restored.codes, module.codes)
    torch.testing.assert_close(
        restored.codebooks, module.codebooks, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        restored_optimizer.state[restored.weight]["compact_momentum"],
        optimizer.state[module.weight]["compact_momentum"],
        rtol=0.0,
        atol=0.0,
    )


def test_fp16_ambient_momentum_resume_is_bit_exact_for_next_step() -> None:
    torch.manual_seed(20260823)
    module = make_module(seed=20260824, fp16_ambient_momentum=True)
    optimizer = make_optimizer(module)
    module.weight.grad = torch.randn_like(module.weight)
    optimizer.step()
    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    restored = make_module(seed=20260825, fp16_ambient_momentum=True)
    restored.load_state_dict(model_state, strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(optimizer_state)
    gradient = torch.randn_like(module.weight)
    module.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(restored.weight, module.weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        restored_optimizer.state[restored.weight]["ambient_momentum"],
        optimizer.state[module.weight]["ambient_momentum"],
        rtol=0.0,
        atol=0.0,
    )


def test_device_style_migration_preserves_weight_leaf() -> None:
    module = make_module()
    module._apply(lambda tensor: tensor.clone())
    assert module.weight.is_leaf and module.weight.requires_grad
    make_optimizer(module)


def test_cartesian_initialization_is_exact_nearest_neighbor() -> None:
    torch.manual_seed(113)
    vectors = torch.randn(4096, 2) * 0.02
    codebook = _normal_cartesian_codebook(0.02, device=torch.device("cpu"))
    assert torch.equal(
        _nearest_cartesian_codes(vectors, codebook),
        _nearest_codes_exact(vectors, codebook),
    )


def test_gpt_routes_complete_mlp_and_optimizer_through_pair_vq() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=(),
            block_fht_mlp_pair_vq=True,
            block_fht_mlp_pair_vq_neighbor_candidates=16,
            block_fht_mlp_pair_vq_code_refresh_interval=8,
            block_fht_mlp_pair_vq_error_feedback=True,
        )
    )
    mlp = model.transformer.h[0].mlp
    assert isinstance(mlp.c_fc, MuonPairVQLinear)
    assert isinstance(mlp.c_proj, MuonPairVQLinear)
    assert mlp.c_fc.stages == 2
    assert mlp.c_proj.stages == 1
    assert mlp.c_fc.fast_residual is True
    assert mlp.c_proj.fast_residual is False
    assert mlp.c_fc.error_feedback is True
    assert mlp.c_proj.error_feedback is True
    stats = model.mlp_pair_vq_stats()
    assert stats["modules"] == 2
    assert stats["dense_master_weight"] == "disabled"
    assert stats["dense_optimizer_momentum"] == "disabled"
    assert stats["dense_ambient_error_buffer"] == "disabled"
    assert stats["compact_feedback_bytes"] > 0
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=1,
    )
    pair_optimizers = [
        item for item in optimizer.optimizers if isinstance(item, MuonPairVQ)
    ]
    assert len(pair_optimizers) == 1
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert int(mlp.c_fc.optimizer_step) == 1
    assert int(mlp.c_proj.optimizer_step) == 1


def test_gpt_routes_pair_vq_to_one_mlp_matrix_family() -> None:
    for target in ("mlp.c_fc", "mlp.c_proj"):
        model = GPT(
            GPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=1,
                n_head=2,
                n_embd=8,
                bias=False,
                block_fht=True,
                block_fht_targets=(),
                block_fht_mlp_pair_vq=True,
                block_fht_mlp_pair_vq_targets=(target,),
                block_fht_mlp_pair_vq_error_feedback=True,
            )
        )
        mlp = model.transformer.h[0].mlp
        selected = mlp.c_fc if target == "mlp.c_fc" else mlp.c_proj
        control = mlp.c_proj if target == "mlp.c_fc" else mlp.c_fc
        assert isinstance(selected, MuonPairVQLinear)
        assert isinstance(control, torch.nn.Linear)
        stats = model.mlp_pair_vq_stats()
        assert stats["modules"] == 1
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=0.001,
            betas=(0.9, 0.95),
            device_type="cpu",
            optimizer="muon",
            muon_momentum=0.95,
            muon_ns_steps=1,
        )
        pair_optimizer = next(
            item
            for item in optimizer.optimizers
            if isinstance(item, MuonPairVQ)
        )
        assert len(pair_optimizer.param_groups[0]["params"]) == 1


def test_gpt_routes_optional_fast_residual_through_cproj() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=(),
            block_fht_mlp_pair_vq=True,
            block_fht_mlp_pair_vq_error_feedback=True,
            block_fht_mlp_pair_vq_cproj_fast_residual=True,
            block_fht_mlp_pair_vq_stochastic_fast_retraction=True,
            block_fht_mlp_pair_vq_stochastic_fast_fht_block_size=8,
            block_fht_mlp_pair_vq_stochastic_fast_uniform_levels=True,
            block_fht_mlp_pair_vq_feedback_codec="cartesian4x4",
            block_fht_mlp_pair_vq_feedback_output_group_size=2,
        )
    )
    mlp = model.transformer.h[0].mlp
    assert isinstance(mlp.c_proj, MuonPairVQLinear)
    assert mlp.c_proj.stages == 1
    assert mlp.c_proj.fast_residual is True
    assert mlp.c_proj.stochastic_fast_retraction is True
    assert mlp.c_fc.stochastic_fast_retraction is True
    assert mlp.c_proj.stochastic_fast_fht_block_size == 8
    assert mlp.c_fc.stochastic_fast_fht_block_size == 8
    assert mlp.c_proj.stochastic_fast_uniform_levels is True
    assert mlp.c_fc.stochastic_fast_uniform_levels is True
    assert mlp.c_proj.error_feedback is True
    assert mlp.c_proj.feedback_output_group_size == 2
    assert mlp.c_fc.feedback_output_group_size == 2
    pair_count = mlp.c_proj.element_count // mlp.c_proj.vector_length
    assert mlp.c_proj.fast_codes.numel() == pair_count
    assert mlp.c_proj.fast_levels.shape == (2, 16)
    assert all(
        value.numel() != mlp.c_proj.element_count
        for value in mlp.c_proj.state_dict().values()
    )


def test_gpt_routes_residual_probe_without_persistent_dense_state() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=(),
            block_fht_mlp_pair_vq=True,
            block_fht_mlp_pair_vq_error_feedback=True,
            block_fht_mlp_pair_vq_feedback_codec=(
                "conditional_polar16x16_rvq2"
            ),
            block_fht_mlp_pair_vq_feedback_residual_probe_steps=(0, 8),
            block_fht_mlp_pair_vq_feedback_residual_probe_lloyd_iterations=(
                3,
                6,
                12,
            ),
        )
    )
    for block in model.transformer.h:
        for module in (block.mlp.c_fc, block.mlp.c_proj):
            assert isinstance(module, MuonPairVQLinear)
            assert module.feedback_residual_probe_steps == (0, 8)
            assert module.feedback_residual_probe_lloyd_iterations == (3, 6, 12)
            state = module.state_dict()
            assert "weight" not in state
            assert not any(
                value.numel() == module.element_count
                and value.dtype == torch.float32
                for value in state.values()
            )


def test_pair_vq_training_boundary_forwards_cproj_fast_residual() -> None:
    namespace = Namespace(
        block_fht_mlp_pair_vq=True,
        block_fht_mlp_pair_vq_seed=20261020,
        block_fht_mlp_pair_vq_neighbor_candidates=16,
        block_fht_mlp_pair_vq_code_refresh_interval=8,
        block_fht_mlp_pair_vq_error_feedback=True,
        block_fht_mlp_pair_vq_hierarchical_feedback_fit=True,
        block_fht_mlp_pair_vq_forward_visible_feedback=True,
        block_fht_mlp_pair_vq_cproj_fast_residual=True,
        block_fht_mlp_pair_vq_stochastic_fast_retraction=False,
        block_fht_mlp_pair_vq_stochastic_fast_fht_block_size=0,
        block_fht_mlp_pair_vq_stochastic_fast_uniform_levels=False,
        block_fht_mlp_pair_vq_feedback_codec="polar32x8",
        block_fht_mlp_pair_vq_feedback_output_group_size=0,
    )
    kwargs = pair_vq_model_kwargs(namespace)
    assert kwargs == {
        "block_fht_attn_pair_vq": False,
        "block_fht_attn_pair_vq_seed": 20261121,
        "block_fht_mlp_pair_vq": True,
        "block_fht_mlp_pair_vq_targets": ("mlp.c_fc", "mlp.c_proj"),
        "block_fht_mlp_pair_vq_seed": 20261020,
        "block_fht_mlp_pair_vq_neighbor_candidates": 16,
        "block_fht_mlp_pair_vq_code_refresh_interval": 8,
        "block_fht_mlp_pair_vq_error_feedback": True,
        "block_fht_mlp_pair_vq_hierarchical_feedback_fit": True,
        "block_fht_mlp_pair_vq_forward_visible_feedback": True,
        "block_fht_mlp_pair_vq_fp16_ambient_momentum": False,
        "block_fht_mlp_pair_vq_fp16_reserved_escape_granularity": "",
        "block_fht_mlp_pair_vq_fp16_ambient_reference_probe_steps": (),
        "block_fht_mlp_pair_vq_lazy_retraction_interval": 1,
        "block_fht_mlp_pair_vq_lazy_retraction_forced_steps": (),
        "block_fht_mlp_pair_vq_cproj_fast_residual": True,
        "block_fht_mlp_pair_vq_stochastic_fast_retraction": False,
        "block_fht_mlp_pair_vq_stochastic_fast_fht_block_size": 0,
        "block_fht_mlp_pair_vq_stochastic_fast_uniform_levels": False,
        "block_fht_mlp_pair_vq_stochastic_fast_block_local_levels": False,
        "block_fht_mlp_pair_vq_feedback_codec": "polar32x8",
        "block_fht_mlp_pair_vq_feedback_output_group_size": 0,
        "block_fht_mlp_pair_vq_feedback_residual_probe_steps": (),
        "block_fht_mlp_pair_vq_feedback_residual_probe_layers": (),
        "block_fht_mlp_pair_vq_feedback_residual_probe_lloyd_iterations": (),
        "block_fht_mlp_pair_vq_feedback_transform_probe_block_sizes": (),
        "block_fht_mlp_pair_vq_feedback_lattice_probe_block_sizes": (),
        "block_fht_mlp_pair_vq_feedback_lattice_probe_coordinate_bits": (),
        "block_fht_mlp_pair_vq_feedback_axis_adaptation_probe_block_size": 0,
        "block_fht_mlp_pair_vq_feedback_axis_adaptation_probe_coordinate_bits": 7,
        "block_fht_mlp_pair_vq_feedback_fractional_probe_block_size": 0,
        "block_fht_mlp_pair_vq_feedback_fractional_probe_base_coordinate_bits": 7,
        "block_fht_mlp_pair_vq_feedback_fractional_probe_refinement_fractions": (),
    }
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=8,
        bias=False,
        block_fht=True,
        block_fht_targets=(),
        **kwargs,
    )
    model = GPT(config)
    assert model.transformer.h[0].mlp.c_proj.fast_residual is True
    assert model.transformer.h[0].mlp.c_proj.feedback_codec == "polar32x8"
    assert model.config.block_fht_mlp_pair_vq_hierarchical_feedback_fit is True


def test_pair_vq_optimizer_boundary_forwards_hierarchical_feedback_fit() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=(),
            block_fht_mlp_pair_vq=True,
            block_fht_mlp_pair_vq_error_feedback=True,
            block_fht_mlp_pair_vq_hierarchical_feedback_fit=True,
        )
    )
    fake_optimizer = type(
        "FakePairVQOptimizer",
        (),
        {"param_groups": [{"params": []}]},
    )()
    with patch(
        "examples.nanogpt.model.MuonPairVQ",
        return_value=fake_optimizer,
    ) as constructor:
        model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=0.001,
            betas=(0.9, 0.95),
            device_type="cpu",
            optimizer="muon",
            muon_momentum=0.95,
            muon_ns_steps=1,
        )
    assert constructor.call_args.kwargs["hierarchical_feedback_fit"] is True


def test_gpt_routes_attention_value_and_output_through_pair_vq() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=("attn.c_attn.qk_headwise",),
            block_fht_output_gain_targets=("attn.c_attn.qk_headwise",),
            block_fht_attn_pair_vq=True,
            block_fht_attn_pair_vq_seed=20261121,
            block_fht_mlp_pair_vq=True,
            block_fht_mlp_pair_vq_error_feedback=True,
            block_fht_mlp_pair_vq_forward_visible_feedback=True,
            block_fht_mlp_pair_vq_fp16_ambient_momentum=True,
            block_fht_mlp_pair_vq_cproj_fast_residual=True,
        )
    )
    block = model.transformer.h[0]
    modules = (
        block.attn.c_attn_v,
        block.attn.c_proj,
        block.mlp.c_fc,
        block.mlp.c_proj,
    )
    assert all(isinstance(module, MuonPairVQLinear) for module in modules)
    assert block.attn.c_attn_v.stages == 2
    assert block.attn.c_proj.stages == 1
    assert block.attn.c_attn_v.fast_residual is True
    assert block.attn.c_proj.fast_residual is True
    assert block.attn.c_attn_v.reserved_escape_scope == "c_fc"
    assert block.attn.c_proj.reserved_escape_scope == "c_proj"
    assert all("weight" not in module.state_dict() for module in modules)
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=1,
    )
    pair_optimizers = [
        item for item in optimizer.optimizers if isinstance(item, MuonPairVQ)
    ]
    assert len(pair_optimizers) == 1
    assert len(pair_optimizers[0].param_groups[0]["params"]) == 4
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert all(int(module.optimizer_step) == 1 for module in modules)


def test_attention_pair_vq_rejects_competing_v_or_output_representation() -> None:
    for target in ("attn.c_attn.v", "attn.c_proj"):
        try:
            GPT(
                GPTConfig(
                    block_size=8,
                    vocab_size=32,
                    n_layer=1,
                    n_head=2,
                    n_embd=8,
                    bias=False,
                    block_fht=True,
                    block_fht_targets=("attn.c_attn.qk_headwise", target),
                    block_fht_attn_pair_vq=True,
                )
            )
        except ValueError as exc:
            assert "Pair-VQ" in str(exc)
        else:
            raise AssertionError(f"expected conflict rejection for {target}")


def test_gpt_routes_fp16_ambient_pair_vq_momentum_and_accounts_state() -> None:
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            block_fht=True,
            block_fht_targets=(),
            block_fht_mlp_pair_vq=True,
            block_fht_mlp_pair_vq_error_feedback=True,
            block_fht_mlp_pair_vq_forward_visible_feedback=True,
            block_fht_mlp_pair_vq_fp16_ambient_momentum=True,
        )
    )
    mlp = model.transformer.h[0].mlp
    modules = (mlp.c_fc, mlp.c_proj)
    assert all(
        isinstance(module, MuonPairVQLinear)
        and module.fp16_ambient_momentum
        for module in modules
    )
    stats = model.mlp_pair_vq_stats()
    elements = sum(module.element_count for module in modules)
    assert stats["compact_momentum_bytes"] == 0
    assert stats["ambient_momentum_bytes"] == 2 * elements
    assert stats["optimizer_momentum_bytes"] == 2 * elements
    assert stats["persistent_training_bytes"] == (
        stats["codec_bytes"]
        + stats["ambient_momentum_bytes"]
        + stats["compact_feedback_bytes"]
    )
    assert stats["dense_optimizer_momentum"] == "fp16_ambient"

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.001,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=1,
    )
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    pair_optimizer = next(
        item for item in optimizer.optimizers if isinstance(item, MuonPairVQ)
    )
    assert len(pair_optimizer.state) == 2
    for state in pair_optimizer.state.values():
        assert "ambient_momentum" in state
        assert state["ambient_momentum"].dtype == torch.float16
        assert "compact_momentum" not in state
