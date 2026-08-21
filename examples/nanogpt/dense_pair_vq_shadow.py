"""Nonintervening pair-VQ observer for a dense MLP training trajectory.

The observer projects each dense MLP checkpoint into the production pair-VQ
model family after every optimizer step.  Its weights are used only inside a
temporary evaluation context; they never affect gradients or dense updates.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from examples.nanogpt.muon_pair_vq import MuonPairVQLinear


RESULT_SCHEMA = "mai_dense_pair_vq_shadow_replay_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class DensePairVQShadowObserver:
    """Track dense MLP weights with the live pair-VQ projection operator."""

    def __init__(
        self,
        model: nn.Module,
        *,
        source_config_path: Path,
        source_config_sha256: str,
        result_path: Path,
        device: str,
    ) -> None:
        if sha256_file(source_config_path) != source_config_sha256:
            raise ValueError("pair-VQ shadow source-config identity mismatch")
        source = json.loads(source_config_path.read_text())
        if not bool(source.get("block_fht_mlp_pair_vq", False)):
            raise ValueError("shadow source config must enable full MLP pair-VQ")
        self.source_config_path = source_config_path
        self.source_config_sha256 = source_config_sha256
        self.result_path = result_path
        self.device = device
        self.code_refresh_interval = int(
            source["block_fht_mlp_pair_vq_code_refresh_interval"]
        )
        self.records: list[dict[str, Any]] = []
        self.projection_history: list[dict[str, Any]] = []
        self.last_projection: dict[str, Any] | None = None
        self._dense_modules: dict[str, nn.Module] = {}
        self._shadow_modules: dict[str, MuonPairVQLinear] = {}

        blocks = list(model.transformer.h)
        base_seed = int(source["block_fht_mlp_pair_vq_seed"])
        for layer, block in enumerate(blocks):
            for side, stages, seed_offset, fast_residual in (
                ("c_fc", 2, 0, True),
                (
                    "c_proj",
                    1,
                    4096,
                    bool(source["block_fht_mlp_pair_vq_cproj_fast_residual"]),
                ),
            ):
                dense = getattr(block.mlp, side)
                if isinstance(dense, MuonPairVQLinear):
                    raise ValueError("dense shadow replay requires a dense MLP parent")
                weight = dense.weight
                name = f"transformer.h.{layer}.mlp.{side}"
                shadow = MuonPairVQLinear(
                    int(weight.shape[1]),
                    int(weight.shape[0]),
                    bias=False,
                    stages=stages,
                    base_seed=base_seed + layer * 8192 + seed_offset,
                    weight_std=(
                        0.02
                        if side == "c_fc"
                        else 0.02 / (2 * len(blocks)) ** 0.5
                    ),
                    layer_id=layer,
                    fast_residual=fast_residual,
                    stochastic_fast_retraction=bool(
                        source.get(
                            "block_fht_mlp_pair_vq_stochastic_fast_retraction",
                            False,
                        )
                    ),
                    error_feedback=False,
                    neighbor_candidates=int(
                        source["block_fht_mlp_pair_vq_neighbor_candidates"]
                    ),
                    code_refresh_interval=self.code_refresh_interval,
                ).to(device)
                shadow.project_requested_weight_(weight, refresh_codes=True)
                shadow.optimizer_step.zero_()
                self._dense_modules[name] = dense
                self._shadow_modules[name] = shadow
        expected_modules = 2 * len(blocks)
        if len(self._shadow_modules) != expected_modules:
            raise ValueError(
                "expected "
                f"{expected_modules} MLP shadow matrices, "
                f"found {len(self._shadow_modules)}"
            )

    @property
    def persistent_matrix_bytes(self) -> int:
        return sum(
            module.persistent_codec_bytes
            for module in self._shadow_modules.values()
        )

    @torch.no_grad()
    def projection_metrics(self) -> dict[str, Any]:
        rows = []
        for name, shadow in self._shadow_modules.items():
            target = self._dense_modules[name].weight.detach().float()
            decoded = shadow.weight.detach().float()
            error = float((decoded - target).square().sum())
            energy = float(target.square().sum())
            rows.append(
                {
                    "module": name,
                    "side": "c_fc" if name.endswith(".c_fc") else "c_proj",
                    "squared_error": error,
                    "squared_target_norm": energy,
                    "weight_energy_recovery": 1.0
                    - error / max(energy, 1e-30),
                }
            )

        def aggregate(selected: list[dict[str, Any]]) -> dict[str, float]:
            error = sum(float(row["squared_error"]) for row in selected)
            energy = sum(float(row["squared_target_norm"]) for row in selected)
            return {
                "weighted_energy_recovery": 1.0
                - error / max(energy, 1e-30),
                "worst_matrix_energy_recovery": min(
                    float(row["weight_energy_recovery"]) for row in selected
                ),
            }

        return {
            "aggregate": aggregate(rows),
            "c_fc": aggregate([row for row in rows if row["side"] == "c_fc"]),
            "c_proj": aggregate(
                [row for row in rows if row["side"] == "c_proj"]
            ),
            "matrices": rows,
        }

    @torch.no_grad()
    def update(self, *, optimizer_step: int) -> dict[str, Any]:
        refresh = int(optimizer_step) % self.code_refresh_interval == 0
        rows = []
        for name, shadow in self._shadow_modules.items():
            row = shadow.project_requested_weight_(
                self._dense_modules[name].weight,
                refresh_codes=refresh,
            )
            rows.append({"module": name, **row})
        def aggregate(selected: list[dict[str, Any]]) -> dict[str, float]:
            request_energy = sum(
                float(row["request_energy"]) for row in selected
            )
            residual_energy = sum(
                float(row["projection_residual_energy"]) for row in selected
            )
            return {
                "request_weighted_energy_recovery": 1.0
                - residual_energy / max(request_energy, 1e-30),
                "worst_matrix_requested_step_energy_recovery": min(
                    float(row["requested_step_energy_recovery"])
                    for row in selected
                ),
            }

        self.last_projection = {
            "optimizer_step": int(optimizer_step),
            "refresh_codes": refresh,
            "aggregate": aggregate(rows),
            "c_fc": aggregate(
                [row for row in rows if str(row["module"]).endswith(".c_fc")]
            ),
            "c_proj": aggregate(
                [
                    row
                    for row in rows
                    if str(row["module"]).endswith(".c_proj")
                ]
            ),
            "code_changes": sum(int(row["code_changes"]) for row in rows),
            "matrices": rows,
        }
        if refresh:
            self.projection_history.append(
                {
                    key: value
                    for key, value in self.last_projection.items()
                    if key != "matrices"
                }
            )
        return self.last_projection

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        with torch.no_grad():
            backups = {
                name: module.weight.detach().clone()
                for name, module in self._dense_modules.items()
            }
            for name, module in self._dense_modules.items():
                module.weight.copy_(self._shadow_modules[name].weight)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, module in self._dense_modules.items():
                    module.weight.copy_(backups[name])

    def record_evaluation(
        self,
        *,
        step: int,
        dense_losses: dict[str, float],
        shadow_losses: dict[str, float],
        run_identity_sha256: str,
        fixed_eval_indices_sha256: str,
        terminal: bool,
    ) -> dict[str, Any]:
        record = {
            "step": int(step),
            "dense_losses": dense_losses,
            "shadow_losses": shadow_losses,
            "shadow_minus_dense_validation_ce": float(shadow_losses["val"])
            - float(dense_losses["val"]),
            "projection": self.projection_metrics(),
            "last_projection_update": self.last_projection,
        }
        self.records.append(record)
        payload = {
            "schema_version": RESULT_SCHEMA,
            "status": "finished" if terminal else "running",
            "source_config": {
                "path": str(self.source_config_path),
                "sha256": self.source_config_sha256,
            },
            "run_identity_sha256": run_identity_sha256,
            "fixed_eval_indices_sha256": fixed_eval_indices_sha256,
            "persistent_pair_vq_matrix_bytes": self.persistent_matrix_bytes,
            "projection_refresh_history": self.projection_history,
            "records": self.records,
        }
        atomic_json(self.result_path, payload)
        return record
