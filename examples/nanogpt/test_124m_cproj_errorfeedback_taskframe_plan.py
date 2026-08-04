from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from examples.nanogpt.model import GPT, GPTConfig
from examples.nanogpt.mfu_preflight import make_preflight_config
from examples.nanogpt.muon_matched_givens import MuonMatchedGivensLinear
from examples.nanogpt.train import schedule_mlp_task_frame_gradients


REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO
    / "examples/nanogpt/configs/"
    "pro6_mai_v3_124m_fullattn_plus_mlp_cproj_"
    "errorfeedback_taskframe_0p5tpp.json"
)
PLAN_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_errorfeedback_task_frame_causal_plan.json"
)
MFU_RESULT_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_errorfeedback_task_frame_0p5tpp_mfu_result.json"
)
PRELAUNCH_METADATA_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_errorfeedback_task_frame_prelaunch_run_metadata.json"
)
RESULT_PATH = (
    REPO
    / "examples/nanogpt/configs/selection_artifacts/"
    "124m_mlp_cproj_errorfeedback_task_frame_0p5tpp_result.json"
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_tiny_model() -> GPT:
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            bias=False,
            dropout=0.0,
            block_fht=True,
            block_fht_targets=("mlp.c_proj",),
            block_fht_mlp_cproj_muon_matched_givens=True,
            block_fht_mlp_cproj_muon_matched_givens_stages=1,
            block_fht_mlp_cproj_muon_matched_givens_residual_stages=1,
            block_fht_mlp_cproj_muon_matched_givens_neighbors=2,
            block_fht_mlp_cproj_muon_matched_givens_refresh_interval=1,
            block_fht_mlp_cproj_muon_matched_givens_fast_fresh=True,
            block_fht_mlp_cproj_muon_matched_givens_error_feedback=True,
            block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay=1.0,
            block_fht_mlp_pregelu_block_rotation_stages=1,
            block_fht_mlp_pregelu_block_rotation_size=4,
            block_fht_mlp_pregelu_block_rotation_basis_size=8,
            block_fht_mlp_pregelu_block_rotation_coordinate_scale=2.0,
            block_fht_mlp_pregelu_cache_retain_graph=False,
            block_fht_mlp_hidden_block_rotation_stages=1,
            block_fht_mlp_hidden_block_rotation_size=4,
            block_fht_mlp_hidden_block_rotation_basis_size=8,
            block_fht_mlp_hidden_block_rotation_coordinate_scale=2.0,
            block_fht_mlp_hidden_gain=False,
            block_fht_mlp_output_rotation_stages=0,
            block_fht_mlp_output_block_rotation_stages=1,
            block_fht_mlp_output_block_rotation_size=4,
            block_fht_mlp_output_block_rotation_basis_size=8,
            block_fht_mlp_output_block_rotation_coordinate_scale=2.0,
            block_fht_mlp_residual_output_gain=False,
        )
    )


def make_optimizer(model: GPT):
    return model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=0.0024,
        betas=(0.9, 0.95),
        device_type="cpu",
        optimizer="muon",
        muon_momentum=0.95,
        muon_ns_steps=2,
        muon_adamw_lr_scale=0.3,
        block_fht_mlp_chart_lr_scale=0.1,
        block_fht_mlp_pregelu_chart_lr_scale=0.1,
    )


def frame_coordinates(model: GPT) -> tuple[torch.nn.Parameter, ...]:
    mlp = model.transformer.h[0].mlp
    assert mlp.pregelu_block_rotation is not None
    assert mlp.hidden_block_rotation is not None
    assert mlp.output_block_rotation is not None
    return (
        mlp.pregelu_block_rotation.coordinates,
        mlp.hidden_block_rotation.coordinates,
        mlp.output_block_rotation.coordinates,
    )


def random_batch(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randint(32, (2, 8), generator=generator),
        torch.randint(32, (2, 8), generator=generator),
    )


def train_step(
    model: GPT,
    optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    verify_routes: bool = False,
    iter_num: int | None = None,
    task_frame_start_iter: int = 0,
) -> int:
    optimizer.zero_grad(set_to_none=True)
    model.prepare_block_fht_cache(dtype=torch.float32)
    _, loss = model(*batch)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    model.flush_block_fht_cache()
    held = 0
    if iter_num is not None:
        held = schedule_mlp_task_frame_gradients(
            model,
            iter_num=iter_num,
            start_iter=task_frame_start_iter,
        )
    mlp = model.transformer.h[0].mlp
    assert isinstance(mlp.c_proj, MuonMatchedGivensLinear)
    if verify_routes:
        for coordinates in frame_coordinates(model):
            assert coordinates.grad is not None
            assert torch.isfinite(coordinates.grad).all()
            assert torch.count_nonzero(coordinates.grad) > 0
        assert mlp.c_proj.weight.grad is not None
        assert torch.isfinite(mlp.c_proj.weight.grad).all()
        assert torch.count_nonzero(mlp.c_proj.weight.grad) > 0
        assert mlp.c_fc.weight.grad is not None
        assert torch.isfinite(mlp.c_fc.weight.grad).all()
        assert torch.count_nonzero(mlp.c_fc.weight.grad) > 0
    optimizer.step()
    return held


def assert_nested_equal(left: Any, right: Any) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def test_plan_binds_exact_config_and_frozen_causal_decision() -> None:
    plan = load(PLAN_PATH)
    config = load(CONFIG_PATH)
    assert sha256(CONFIG_PATH) == plan["identity"]["config_sha256"]
    assert plan["identity"]["implementation_commit"] == config["implementation_commit"]
    assert config["data_manifest_sha256"] == plan["identity"]["dataset_manifest_sha256"]
    assert config["fixed_eval_index_runtime_digest"].endswith(
        plan["identity"]["fixed_eval_indices_sha256"]
    )
    assert config["max_iters"] == config["lr_decay_iters"] == 238
    assert config["planned_tokens"] == config["scheduled_tokens"] == 62_390_272
    assert config["preregistered_decision_rule"]["pass_validation_ce_maximum"] == 5.522365207672119
    assert plan["decision_rule"]["automatic_larger_rung_authorized"] is False
    assert plan["execution_policy"]["queue_entry"] is False
    assert plan["performance_and_correctness_gate"]["watchdog"] is False
    command = plan["execution_policy"]["mfu_command"]
    assert command[command.index("--warmup-updates") + 1] == "1"
    assert command[command.index("--timed-updates") + 1] == "8"


def test_candidate_preserves_parent_and_adds_only_registered_frames() -> None:
    config = load(CONFIG_PATH)
    assert config["block_fht_targets"] == [
        "attn.c_attn.qk_headwise",
        "attn.c_attn.v",
        "attn.c_proj",
        "mlp.c_proj",
    ]
    assert config["block_fht_mlp_cproj_muon_matched_givens_stages"] == 64
    assert config["block_fht_mlp_cproj_muon_matched_givens_residual_stages"] == 24
    assert config["block_fht_mlp_cproj_muon_matched_givens_error_feedback"] is True
    assert config["block_fht_mlp_cproj_muon_matched_givens_error_feedback_decay"] == 1.0
    assert config["block_fht_mlp_pregelu_block_rotation_stages"] == 2
    assert config["block_fht_mlp_hidden_block_rotation_stages"] == 2
    assert config["block_fht_mlp_output_rotation_stages"] == 0
    assert config["block_fht_mlp_output_block_rotation_stages"] == 4
    assert config["block_fht_mlp_pregelu_chart_lr_scale"] == 0.1
    assert config["block_fht_mlp_chart_lr_scale"] == 0.1
    assert 0.0024 * 0.3 * 0.1 == 0.000072
    assert config["block_fht_mlp_hidden_gain"] is False
    assert config["block_fht_mlp_residual_output_gain"] is False
    assert config["block_fht_mlp_shared_hidden_gain"] is False


def test_mfu_gate_and_prelaunch_metadata_bind_the_run() -> None:
    result = load(MFU_RESULT_PATH)
    metadata = load(PRELAUNCH_METADATA_PATH)
    assert result["passed"] is True
    assert result["measurement"]["mfu_fraction"] >= 0.2
    assert result["measurement"]["timed_updates"] == 8
    assert result["correctness"]["interrupted_resume_optimizer_bitwise_equal"] is True
    assert sha256(CONFIG_PATH) == result["config"]["sha256"]
    assert sha256(PLAN_PATH) == result["plan"]["sha256"]
    assert sha256(MFU_RESULT_PATH) == metadata["identity"]["mfu_result_sha256"]
    assert metadata["run"]["entrypoint"] == "examples.nanogpt.train"
    assert metadata["entrypoint"] == "examples.nanogpt.train"
    assert metadata["repository"]["git_commit"] == result["execution"]["git_commit"]
    assert metadata["config"]["sha256"] == result["config"]["sha256"]
    assert metadata["dataset_manifest"]["sha256"] == result["stability"]["dataset_manifest_sha256"]
    assert metadata["launch_attempts"][0]["scientific_updates_persisted"] == 0
    assert metadata["run"]["git_commit"] == result["execution"]["git_commit"]
    assert metadata["run"]["direct_foreground_polling"] is True
    assert metadata["run"]["watchdog"] is False
    assert metadata["protocol"]["larger_rung_authorized"] is False


def test_three_frame_gradient_routes_and_interrupted_resume_are_exact() -> None:
    torch.manual_seed(20260890)
    template = make_tiny_model()
    initial_model_state = copy.deepcopy(template.state_dict())

    uninterrupted = make_tiny_model()
    uninterrupted.load_state_dict(copy.deepcopy(initial_model_state))
    uninterrupted_optimizer = make_optimizer(uninterrupted)
    uninterrupted_generator = torch.Generator().manual_seed(20260891)
    train_step(
        uninterrupted,
        uninterrupted_optimizer,
        random_batch(uninterrupted_generator),
        verify_routes=True,
    )
    train_step(
        uninterrupted,
        uninterrupted_optimizer,
        random_batch(uninterrupted_generator),
    )

    interrupted = make_tiny_model()
    interrupted.load_state_dict(copy.deepcopy(initial_model_state))
    interrupted_optimizer = make_optimizer(interrupted)
    interrupted_generator = torch.Generator().manual_seed(20260891)
    train_step(
        interrupted,
        interrupted_optimizer,
        random_batch(interrupted_generator),
        verify_routes=True,
    )
    checkpoint_model = copy.deepcopy(interrupted.state_dict())
    checkpoint_optimizer = copy.deepcopy(interrupted_optimizer.state_dict())
    checkpoint_data_rng = interrupted_generator.get_state().clone()
    checkpoint_torch_rng = torch.get_rng_state().clone()

    resumed = make_tiny_model()
    resumed.load_state_dict(checkpoint_model)
    resumed_optimizer = make_optimizer(resumed)
    resumed_optimizer.load_state_dict(checkpoint_optimizer)
    resumed_generator = torch.Generator()
    resumed_generator.set_state(checkpoint_data_rng)
    torch.set_rng_state(checkpoint_torch_rng)
    train_step(
        resumed,
        resumed_optimizer,
        random_batch(resumed_generator),
    )

    assert_nested_equal(uninterrupted.state_dict(), resumed.state_dict())
    assert_nested_equal(
        uninterrupted_optimizer.state_dict(),
        resumed_optimizer.state_dict(),
    )
    assert torch.equal(
        uninterrupted_generator.get_state(),
        resumed_generator.get_state(),
    )
    for left, right in zip(
        frame_coordinates(uninterrupted),
        frame_coordinates(resumed),
        strict=True,
    ):
        assert torch.equal(left, right)


def test_terminal_result_rejects_directionally_misaligned_transfer() -> None:
    result = load(RESULT_PATH)
    assert result["classification"] == "REJECT_CAUSAL_TASK_FRAME_DIRECTION_MISALIGNED"
    assert sha256(CONFIG_PATH) == result["config"]["sha256"]
    assert sha256(PLAN_PATH) == result["plan"]["sha256"]
    assert sha256(MFU_RESULT_PATH) == result["mfu_result"]["sha256"]
    assert result["identity"]["checkpoint_schema"] == "nanogpt_exact_resume_v2"
    assert result["identity"]["checkpoint_next_iter"] == 238
    assert result["loss"]["candidate_minus_parent_ce"] > 0.0
    assert result["loss"]["candidate_minus_pass_threshold_ce"] > 0.0
    assert result["decision"]["passed"] is False
    assert result["decision"]["automatic_rerun_authorized"] is False
    for field in (
        "pregelu_global_cosine",
        "postgelu_hidden_global_cosine",
        "residual_output_global_cosine",
    ):
        assert abs(result["direction_diagnosis"][field]) < 0.05


def test_delayed_boundary_holds_all_frames_then_resumes_exactly() -> None:
    torch.manual_seed(20260892)
    template = make_tiny_model()
    initial_model_state = copy.deepcopy(template.state_dict())

    uninterrupted = make_tiny_model()
    uninterrupted.load_state_dict(copy.deepcopy(initial_model_state))
    uninterrupted_optimizer = make_optimizer(uninterrupted)
    uninterrupted_generator = torch.Generator().manual_seed(20260893)
    held = train_step(
        uninterrupted,
        uninterrupted_optimizer,
        random_batch(uninterrupted_generator),
        iter_num=119,
        task_frame_start_iter=120,
    )
    assert held == 3
    assert all(
        torch.count_nonzero(parameter) == 0
        for parameter in frame_coordinates(uninterrupted)
    )
    assert all(
        parameter not in suboptimizer.state
        for suboptimizer in uninterrupted_optimizer.optimizers
        for parameter in frame_coordinates(uninterrupted)
    )
    checkpoint_model = copy.deepcopy(uninterrupted.state_dict())
    checkpoint_optimizer = copy.deepcopy(uninterrupted_optimizer.state_dict())
    checkpoint_data_rng = uninterrupted_generator.get_state().clone()
    checkpoint_torch_rng = torch.get_rng_state().clone()

    train_step(
        uninterrupted,
        uninterrupted_optimizer,
        random_batch(uninterrupted_generator),
        verify_routes=True,
        iter_num=120,
        task_frame_start_iter=120,
    )
    assert all(
        torch.count_nonzero(parameter) > 0
        for parameter in frame_coordinates(uninterrupted)
    )

    resumed = make_tiny_model()
    resumed.load_state_dict(checkpoint_model)
    resumed_optimizer = make_optimizer(resumed)
    resumed_optimizer.load_state_dict(checkpoint_optimizer)
    resumed_generator = torch.Generator()
    resumed_generator.set_state(checkpoint_data_rng)
    torch.set_rng_state(checkpoint_torch_rng)
    train_step(
        resumed,
        resumed_optimizer,
        random_batch(resumed_generator),
        verify_routes=True,
        iter_num=120,
        task_frame_start_iter=120,
    )

    assert_nested_equal(uninterrupted.state_dict(), resumed.state_dict())
    assert_nested_equal(
        uninterrupted_optimizer.state_dict(),
        resumed_optimizer.state_dict(),
    )
    assert torch.equal(
        uninterrupted_generator.get_state(),
        resumed_generator.get_state(),
    )


def test_mfu_preflight_moves_delayed_boundary_before_timed_updates() -> None:
    source = load(CONFIG_PATH)
    source["block_fht_mlp_task_frame_start_iter"] = 120
    with tempfile.TemporaryDirectory() as raw:
        preflight = make_preflight_config(
            source,
            Path(raw),
            warmups=1,
            timed=8,
        )
    assert preflight["block_fht_mlp_task_frame_start_iter"] == 1
    assert preflight["perf_warmup_iters"] == 1
    assert preflight["max_iters"] == 9
