from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

import pytest

from examples.nanogpt.analyze_mlp_cfc_multistage_directed import (
    fit_multistage_directed_sparse_mixer,
)
from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.train import parse_args, require_block_fht_native_extension
from examples.nanogpt.muon_matched_givens import (
    MuonDirectedProduct,
    MuonDirectedProductLinear,
    MuonMatchedGivens,
    batched_multistage_directed_sparse_update,
    decode_blockwise_int8_optimizer_state,
    encode_blockwise_int8_optimizer_state,
)


def make_module(
    *,
    layer_id: int = 0,
    error_feedback: bool = False,
    error_feedback_decay: float = 1.0,
) -> MuonDirectedProductLinear:
    torch.manual_seed(301 + layer_id)
    return MuonDirectedProductLinear(
        4,
        8,
        bias=False,
        incoming_schedule=(1, 1, 1),
        ridge_ratio=1e-6,
        chunk_size=3,
        family_radius_ratio=0.65,
        error_feedback=error_feedback,
        error_feedback_decay=error_feedback_decay,
        weight_std=0.02,
        layer_id=layer_id,
    )


def make_optimizer(
    modules: list[MuonDirectedProductLinear],
    **kwargs,
) -> MuonDirectedProduct:
    return MuonDirectedProduct(
        modules,
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.1,
        ns_steps=5,
        **kwargs,
    )


def make_gpt_config() -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
        block_fht=True,
        block_fht_targets=("mlp.c_proj",),
        block_fht_mlp_cproj_muon_matched_givens=True,
        block_fht_mlp_cproj_muon_matched_givens_stages=1,
        block_fht_mlp_cproj_muon_matched_givens_residual_stages=0,
        block_fht_mlp_cproj_muon_matched_givens_neighbors=2,
        block_fht_mlp_cproj_muon_matched_givens_refresh_interval=3,
        block_fht_mlp_cfc_directed_product=True,
        block_fht_mlp_cfc_directed_product_schedule=(1, 1, 1),
        block_fht_mlp_cfc_directed_product_ridge_ratio=1e-6,
        block_fht_mlp_cfc_directed_product_chunk_size=5,
        block_fht_mlp_cfc_directed_product_family_radius_ratio=0.65,
        block_fht_mlp_cfc_directed_product_error_feedback=False,
        block_fht_mlp_cfc_directed_product_error_feedback_decay=1.0,
    )


def test_batched_solver_matches_registered_scalar_solver() -> None:
    torch.manual_seed(307)
    source = torch.randn(2, 6, 8)
    target = torch.randn_like(source) * 0.1
    schedule = (2, 1, 1)
    actual, rows = batched_multistage_directed_sparse_update(
        source,
        target,
        incoming_schedule=schedule,
        ridge_ratio=1e-6,
        chunk_size=3,
    )
    expected = []
    for member in range(source.shape[0]):
        prediction, _row = fit_multistage_directed_sparse_mixer(
            source[member],
            target[member],
            incoming_schedule=list(schedule),
            ridge_ratio=1e-6,
            chunk_size=3,
        )
        expected.append(prediction)
    assert torch.allclose(actual, torch.stack(expected), atol=2e-5, rtol=2e-5)
    assert [row["incoming_per_target"] for row in rows] == list(schedule)


def test_module_exposes_only_sparse_coordinate_budget() -> None:
    module = make_module()
    assert module.coordinate_count == 24
    assert not isinstance(module.weight, torch.nn.Parameter)
    assert set(module.state_dict()) == {"weight", "optimizer_step"}
    assert all(
        tensor.shape != (module.out_features, module.out_features)
        for tensor in module.state_dict().values()
    )


def test_forward_gradient_and_family_radius_are_finite() -> None:
    modules = [make_module(layer_id=index) for index in range(2)]
    optimizer = make_optimizer(modules)
    before = [module.weight.clone() for module in modules]
    loss = sum(module(torch.randn(3, 4)).square().mean() for module in modules)
    loss.backward()
    assert all(torch.isfinite(module.weight.grad).all() for module in modules)
    optimizer.step()
    diagnostics = optimizer.consume_diagnostics()
    actual_radius = torch.stack(
        [
            (module.weight - old).float().square().sum()
            for module, old in zip(modules, before, strict=True)
        ]
    ).sum().sqrt()
    assert torch.isfinite(actual_radius)
    assert torch.allclose(
        actual_radius,
        torch.tensor(diagnostics[0]["target_family_fro"]),
        atol=1e-7,
        rtol=2e-5,
    )
    assert all(row["coordinates"] == 24 for row in diagnostics)
    assert all(int(module.optimizer_step) == 1 for module in modules)


def test_optimizer_resume_is_exact() -> None:
    modules = [make_module(layer_id=index) for index in range(2)]
    optimizer = make_optimizer(modules)
    generator = torch.Generator().manual_seed(313)
    for module in modules:
        module.weight.grad = torch.randn(
            module.weight.shape, generator=generator
        )
    optimizer.step()
    module_states = [copy.deepcopy(module.state_dict()) for module in modules]
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    resumed = [make_module(layer_id=index) for index in range(2)]
    resumed_optimizer = make_optimizer(resumed)
    for module, state in zip(resumed, module_states, strict=True):
        module.load_state_dict(state)
    resumed_optimizer.load_state_dict(optimizer_state)

    next_gradients = [
        torch.randn(module.weight.shape, generator=generator)
        for module in modules
    ]
    for module, gradient in zip(modules, next_gradients, strict=True):
        module.weight.grad = gradient.clone()
    for module, gradient in zip(resumed, next_gradients, strict=True):
        module.weight.grad = gradient.clone()
    optimizer.step()
    resumed_optimizer.step()
    for original, restored in zip(modules, resumed, strict=True):
        assert torch.equal(original.weight, restored.weight)
        assert torch.equal(original.optimizer_step, restored.optimizer_step)


def test_blockwise_int8_optimizer_state_codec_is_deterministic() -> None:
    source = torch.randn(5, 7, generator=torch.Generator().manual_seed(1301))
    quantized, scales = encode_blockwise_int8_optimizer_state(
        source, block_size=8
    )
    second_quantized, second_scales = encode_blockwise_int8_optimizer_state(
        source, block_size=8
    )
    decoded = decode_blockwise_int8_optimizer_state(
        quantized, scales, block_size=8
    )
    assert quantized.dtype == torch.int8
    assert scales.dtype == torch.float16
    assert scales.numel() == 5
    assert torch.equal(quantized, second_quantized)
    assert torch.equal(scales, second_scales)
    assert torch.max(torch.abs(decoded - source)) <= scales.float().max()


def test_compact_directed_state_persists_and_resumes_exactly() -> None:
    modules = [
        make_module(layer_id=index, error_feedback=True)
        for index in range(2)
    ]
    optimizer = make_optimizer(
        modules,
        momentum_state_dtype="float16",
        feedback_state_codec="int8_blockwise",
        feedback_state_block_size=8,
    )
    generator = torch.Generator().manual_seed(1303)
    for module in modules:
        module.weight.grad = torch.randn(
            module.weight.shape, generator=generator
        )
    optimizer.step()
    for module in modules:
        state = optimizer.state[module.weight]
        assert state["momentum_buffer"].dtype == torch.float16
        assert state["compression_residual"].dtype == torch.int8
        assert state["compression_residual_block_scale"].dtype == torch.float16

    module_states = [copy.deepcopy(module.state_dict()) for module in modules]
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    resumed = [
        make_module(layer_id=index, error_feedback=True)
        for index in range(2)
    ]
    resumed_optimizer = make_optimizer(
        resumed,
        momentum_state_dtype="float16",
        feedback_state_codec="int8_blockwise",
        feedback_state_block_size=8,
    )
    for module, state in zip(resumed, module_states, strict=True):
        module.load_state_dict(state)
    resumed_optimizer.load_state_dict(optimizer_state)
    for module in resumed:
        state = resumed_optimizer.state[module.weight]
        assert state["momentum_buffer"].dtype == torch.float16
        assert state["compression_residual"].dtype == torch.int8
        assert state["compression_residual_block_scale"].dtype == torch.float16

    gradients = [
        torch.randn(module.weight.shape, generator=generator)
        for module in modules
    ]
    for original, restored, gradient in zip(
        modules, resumed, gradients, strict=True
    ):
        original.weight.grad = gradient.clone()
        restored.weight.grad = gradient.clone()
    optimizer.step()
    resumed_optimizer.step()
    for original, restored in zip(modules, resumed, strict=True):
        assert torch.equal(original.weight, restored.weight)
        for key in (
            "momentum_buffer",
            "compression_residual",
            "compression_residual_block_scale",
        ):
            assert torch.equal(
                optimizer.state[original.weight][key],
                resumed_optimizer.state[restored.weight][key],
            )


def test_explicit_float32_codec_preserves_default_update() -> None:
    control = [make_module(layer_id=0, error_feedback=True)]
    explicit = [make_module(layer_id=0, error_feedback=True)]
    explicit[0].load_state_dict(copy.deepcopy(control[0].state_dict()))
    optimizers = (
        make_optimizer(control),
        make_optimizer(
            explicit,
            momentum_state_dtype="float32",
            feedback_state_codec="float32",
            feedback_state_block_size=4096,
        ),
    )
    generator = torch.Generator().manual_seed(1307)
    for _ in range(2):
        gradient = torch.randn(control[0].weight.shape, generator=generator)
        control[0].weight.grad = gradient.clone()
        explicit[0].weight.grad = gradient.clone()
        for optimizer in optimizers:
            optimizer.step()
    assert torch.equal(control[0].weight, explicit[0].weight)
    for key in ("momentum_buffer", "compression_residual"):
        assert torch.equal(
            optimizers[0].state[control[0].weight][key],
            optimizers[1].state[explicit[0].weight][key],
        )


def test_error_feedback_carries_exact_compression_residual() -> None:
    modules = [
        make_module(layer_id=index, error_feedback=True)
        for index in range(2)
    ]
    optimizer = make_optimizer(modules)
    generator = torch.Generator().manual_seed(314)
    first_gradients = [
        torch.randn(module.weight.shape, generator=generator)
        for module in modules
    ]
    for module, gradient in zip(modules, first_gradients, strict=True):
        module.weight.grad = gradient
    optimizer.step()
    first_rows = optimizer.consume_diagnostics()
    assert all(row["error_feedback"] is True for row in first_rows)
    assert all(row["feedback_input_fro"] == 0.0 for row in first_rows)
    assert all(row["feedback_output_fro"] > 0.0 for row in first_rows)
    stored = [
        optimizer.state[module.weight]["compression_residual"].clone()
        for module in modules
    ]

    for module in modules:
        module.weight.grad = torch.zeros_like(module.weight)
    optimizer.step()
    second_rows = optimizer.consume_diagnostics()
    for row, residual in zip(second_rows, stored, strict=True):
        assert torch.allclose(
            torch.tensor(row["feedback_input_fro"]),
            residual.norm(),
            atol=1e-7,
            rtol=2e-5,
        )


def test_error_feedback_first_step_matches_uncompensated_path() -> None:
    baseline = [make_module(layer_id=index) for index in range(2)]
    feedback = [
        make_module(layer_id=index, error_feedback=True)
        for index in range(2)
    ]
    for source, target in zip(baseline, feedback, strict=True):
        target.load_state_dict(copy.deepcopy(source.state_dict()))
    baseline_optimizer = make_optimizer(baseline)
    feedback_optimizer = make_optimizer(feedback)
    generator = torch.Generator().manual_seed(316)
    gradients = [
        torch.randn(module.weight.shape, generator=generator)
        for module in baseline
    ]
    for plain, compensated, gradient in zip(
        baseline, feedback, gradients, strict=True
    ):
        plain.weight.grad = gradient.clone()
        compensated.weight.grad = gradient.clone()
    baseline_optimizer.step()
    feedback_optimizer.step()
    for plain, compensated in zip(baseline, feedback, strict=True):
        assert torch.equal(plain.weight, compensated.weight)
        assert torch.equal(
            baseline_optimizer.state[plain.weight]["momentum_buffer"],
            feedback_optimizer.state[compensated.weight]["momentum_buffer"],
        )


def test_error_feedback_optimizer_resume_is_exact() -> None:
    modules = [
        make_module(layer_id=index, error_feedback=True)
        for index in range(2)
    ]
    optimizer = make_optimizer(modules)
    generator = torch.Generator().manual_seed(315)
    for module in modules:
        module.weight.grad = torch.randn(
            module.weight.shape, generator=generator
        )
    optimizer.step()
    module_states = [copy.deepcopy(module.state_dict()) for module in modules]
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    resumed = [
        make_module(layer_id=index, error_feedback=True)
        for index in range(2)
    ]
    resumed_optimizer = make_optimizer(resumed)
    for module, state in zip(resumed, module_states, strict=True):
        module.load_state_dict(state)
    resumed_optimizer.load_state_dict(optimizer_state)

    next_gradients = [
        torch.randn(module.weight.shape, generator=generator)
        for module in modules
    ]
    for original, restored, gradient in zip(
        modules, resumed, next_gradients, strict=True
    ):
        original.weight.grad = gradient.clone()
        restored.weight.grad = gradient.clone()
    optimizer.step()
    resumed_optimizer.step()
    for original, restored in zip(modules, resumed, strict=True):
        assert torch.equal(original.weight, restored.weight)
        assert torch.equal(original.optimizer_step, restored.optimizer_step)
        assert torch.equal(
            optimizer.state[original.weight]["compression_residual"],
            resumed_optimizer.state[restored.weight]["compression_residual"],
        )


def test_gpt_wiring_optimizer_assignment_and_stats() -> None:
    config = make_gpt_config()
    config.block_fht_mlp_cfc_directed_product_error_feedback = True
    config.block_fht_mlp_cproj_muon_matched_givens_error_feedback = True
    config.block_fht_mlp_muon_momentum_state_dtype = "float16"
    config.block_fht_mlp_error_feedback_state_codec = "int8_blockwise"
    config.block_fht_mlp_error_feedback_state_block_size = 8
    model = GPT(config)
    modules = [block.mlp.c_fc for block in model.transformer.h]
    assert all(isinstance(module, MuonDirectedProductLinear) for module in modules)
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    directed = next(
        item
        for item in optimizer.optimizers
        if isinstance(item, MuonDirectedProduct)
    )
    cproj = next(
        item for item in optimizer.optimizers if isinstance(item, MuonMatchedGivens)
    )
    assert optimizer.optimizers.index(directed) < optimizer.optimizers.index(cproj)
    for selected in (directed, cproj):
        group = selected.param_groups[0]
        assert group["momentum_state_dtype"] == "float16"
        assert group["feedback_state_codec"] == "int8_blockwise"
        assert group["feedback_state_block_size"] == 8
    tokens = torch.randint(0, 32, (2, 8))
    _logits, loss = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    optimizer.step()
    assert all(int(module.optimizer_step) == 1 for module in modules)
    # Per layer: 3*32 directed c_fc coordinates and 1*16 c_proj angles.
    assert model.block_fht_stats()["latent"] == 2 * (96 + 16)


def test_compact_mlp_state_does_not_leak_into_attention_optimizer() -> None:
    config = make_gpt_config()
    config.block_fht_targets = ("attn.c_proj", "mlp.c_proj")
    config.block_fht_attn_muon_matched_givens_targets = ("attn.c_proj",)
    config.block_fht_attn_muon_matched_givens_stages = 1
    config.block_fht_attn_muon_matched_givens_neighbors = 2
    config.block_fht_attn_muon_matched_givens_fast_matching = True
    config.block_fht_mlp_cfc_directed_product_error_feedback = True
    config.block_fht_mlp_cproj_muon_matched_givens_error_feedback = True
    config.block_fht_mlp_muon_momentum_state_dtype = "float16"
    config.block_fht_mlp_error_feedback_state_codec = "int8_blockwise"
    config.block_fht_mlp_error_feedback_state_block_size = 8
    model = GPT(config)
    composite = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
    )
    matched = [
        optimizer
        for optimizer in composite.optimizers
        if isinstance(optimizer, MuonMatchedGivens)
    ]
    assert len(matched) == 2
    attention_weights = {
        id(block.attn.c_proj.weight) for block in model.transformer.h
    }
    attention = next(
        optimizer
        for optimizer in matched
        if id(optimizer.param_groups[0]["params"][0]) in attention_weights
    )
    mlp = next(optimizer for optimizer in matched if optimizer is not attention)
    assert attention.param_groups[0]["momentum_state_dtype"] == "float32"
    assert attention.param_groups[0]["feedback_state_codec"] == "float32"
    assert mlp.param_groups[0]["momentum_state_dtype"] == "float16"
    assert mlp.param_groups[0]["feedback_state_codec"] == "int8_blockwise"


def test_directed_cfc_preserves_dense_paired_seed_initialization() -> None:
    dense_config = make_gpt_config()
    dense_config.block_fht_mlp_cfc_directed_product = False
    directed_config = copy.deepcopy(dense_config)
    directed_config.block_fht_mlp_cfc_directed_product = True
    torch.manual_seed(317)
    dense = GPT(dense_config)
    torch.manual_seed(317)
    directed = GPT(directed_config)
    for dense_block, directed_block in zip(
        dense.transformer.h, directed.transformer.h, strict=True
    ):
        assert torch.equal(
            dense_block.mlp.c_fc.weight,
            directed_block.mlp.c_fc.weight,
        )


def test_directed_cfc_accepts_dense_cproj_control_and_rejects_unqualified_generated_cproj(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    joint = json.loads(
        (
            root
            / "examples/nanogpt/configs/"
            "pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_"
            "cfcdecay1_cprojdecay0p5_5tpp_lr24e4_v2.json"
        ).read_text()
    )
    dense_control = {
        key: value
        for key, value in joint.items()
        if not key.startswith("block_fht_mlp_cproj_")
    }
    dense_control["block_fht_targets"] = [
        target
        for target in dense_control["block_fht_targets"]
        if target != "mlp.c_proj"
    ]
    dense_path = tmp_path / "dense_cproj_control.json"
    dense_path.write_text(json.dumps(dense_control))
    with patch.object(
        sys, "argv", ["train.py", "--config", str(dense_path)]
    ):
        parsed = parse_args()
    assert parsed.block_fht_mlp_cfc_directed_product is True
    assert "mlp.c_proj" not in parsed.block_fht_targets
    assert parsed.block_fht_mlp_cproj_muon_matched_givens is False

    generated_unqualified = dict(dense_control)
    generated_unqualified["block_fht_targets"] = [
        *dense_control["block_fht_targets"],
        "mlp.c_proj",
    ]
    generated_path = tmp_path / "generated_unqualified_cproj.json"
    generated_path.write_text(json.dumps(generated_unqualified))
    with patch.object(
        sys, "argv", ["train.py", "--config", str(generated_path)]
    ):
        with pytest.raises(ValueError, match="either dense c_proj"):
            parse_args()


def test_compact_mlp_optimizer_state_requires_qualified_paired_path(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (
            root
            / "examples/nanogpt/configs/"
            "pro6_mai_v3_124m_repairedfullattn_plus_fullmlp_"
            "cfcdecay1_cprojdecay0p5_5tpp_lr24e4_v2.json"
        ).read_text()
    )
    config.update(
        {
            "block_fht_mlp_muon_momentum_state_dtype": "float16",
            "block_fht_mlp_error_feedback_state_codec": "int8_blockwise",
            "block_fht_mlp_error_feedback_state_block_size": 4096,
        }
    )
    accepted_path = tmp_path / "compact_state.json"
    accepted_path.write_text(json.dumps(config))
    with patch.object(
        sys, "argv", ["train.py", "--config", str(accepted_path)]
    ):
        parsed = parse_args()
    assert parsed.block_fht_mlp_muon_momentum_state_dtype == "float16"
    assert parsed.block_fht_mlp_error_feedback_state_codec == "int8_blockwise"

    config["block_fht_mlp_cfc_directed_product_error_feedback"] = False
    invalid_path = tmp_path / "compact_state_without_cfc_feedback.json"
    invalid_path.write_text(json.dumps(config))
    with patch.object(
        sys, "argv", ["train.py", "--config", str(invalid_path)]
    ):
        with pytest.raises(ValueError, match="compact MLP optimizer state"):
            parse_args()


def test_native_extension_guard_fails_closed(monkeypatch) -> None:
    from latent_weight_lab import block_fht as block_fht_module

    assert require_block_fht_native_extension(False) is False
    monkeypatch.setattr(block_fht_module, "_load_block_fht_ext", lambda: object())
    assert require_block_fht_native_extension(True) is True
    monkeypatch.setattr(block_fht_module, "_load_block_fht_ext", lambda: None)
    monkeypatch.setattr(
        block_fht_module,
        "_BLOCK_FHT_EXT_ERROR",
        RuntimeError("missing native test backend"),
    )
    with pytest.raises(RuntimeError, match="missing native test backend"):
        require_block_fht_native_extension(True)
