from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from examples.nanogpt.dense_pair_vq_shadow import DensePairVQShadowObserver


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(4, 16, bias=False)
        self.c_proj = nn.Linear(16, 4, bias=False)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMLP()


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = nn.ModuleList([TinyBlock()])


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = TinyTransformer()


def test_shadow_tracks_without_mutating_dense_and_restores_install(tmp_path) -> None:
    config = {
        "block_fht_mlp_pair_vq": True,
        "block_fht_mlp_pair_vq_seed": 10,
        "block_fht_mlp_pair_vq_neighbor_candidates": 16,
        "block_fht_mlp_pair_vq_code_refresh_interval": 8,
        "block_fht_mlp_pair_vq_cproj_fast_residual": True,
    }
    config_path = tmp_path / "source.json"
    config_path.write_text(json.dumps(config))
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    model = TinyModel()
    originals = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    observer = DensePairVQShadowObserver(
        model,
        source_config_path=config_path,
        source_config_sha256=digest,
        result_path=tmp_path / "result.json",
        device="cpu",
    )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, originals[name])
    before = observer.projection_metrics()["aggregate"][
        "weighted_energy_recovery"
    ]
    with torch.no_grad():
        model.transformer.h[0].mlp.c_fc.weight.add_(0.001)
        model.transformer.h[0].mlp.c_proj.weight.sub_(0.001)
    modified = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    update = observer.update(optimizer_step=0)
    assert update["refresh_codes"]
    after = observer.projection_metrics()["aggregate"][
        "weighted_energy_recovery"
    ]
    assert torch.isfinite(torch.tensor([before, after])).all()
    with observer.installed():
        for name, dense in observer._dense_modules.items():
            torch.testing.assert_close(
                dense.weight,
                observer._shadow_modules[name].weight,
                rtol=0.0,
                atol=0.0,
            )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, modified[name])


def test_registered_plan_binds_exact_dense_replay_config() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(
        (
            root
            / "examples/nanogpt/configs/selection_artifacts/124m_dense_pairvq_shadow_replay_plan.json"
        ).read_text()
    )
    registered = plan["registered_run"]
    config_path = root / registered["config"]
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == registered[
        "config_sha256"
    ]
    config = json.loads(config_path.read_text())
    assert config["pair_vq_dense_shadow_replay"] is True
    assert config["block_fht_targets"] == ["attn.c_attn.qk_headwise"]
    assert config["max_iters"] == 238
    assert config["fixed_eval_indices"] is True
    assert config["save_checkpoint"] is False
    assert config["mfu_min_fraction"] >= 0.20
