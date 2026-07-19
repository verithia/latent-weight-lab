from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.model import GPT, GPTConfig, freeze_non_block_fht
from examples.nanogpt.dense_scaling_fit import (
    ARTIFACT_SCHEMA_VERSION,
    sha256_file,
    validate_dense_fit_artifact,
)
from examples.nanogpt.mai_selection_artifacts import validate_v2_launch_config
from latent_weight_lab import BlockFHTLinear


CHECKPOINT_SCHEMA_VERSION = "nanogpt_exact_resume_v2"
CHECKPOINT_METADATA_SCHEMA_VERSION = "nanogpt_checkpoint_metadata_v2"
CHECKPOINT_FILENAME = "ckpt.pt"
CHECKPOINT_METADATA_FILENAME = "ckpt.meta.json"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return all resolved settings that affect a reproducible training state.

    ``--config`` is invocation provenance rather than a semantic setting, while
    ``--init-from`` must change from scratch to resume. Both are recorded
    separately and intentionally excluded from the identity hash.
    """
    resolved = {
        key: value
        for key, value in vars(args).items()
        if key not in {"config", "init_from"} and not key.startswith("_")
    }
    if resolved.get("model_seed") is None:
        resolved["model_seed"] = 1337
    return resolved


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "examples/nanogpt/train.py",
        root / "examples/nanogpt/model.py",
        root / "latent_weight_lab/block_fht.py",
    )
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in paths
    }


def data_manifest_identity(data_dir: Path) -> dict[str, str | None]:
    manifest = data_dir / "manifest.json"
    return {
        "path": str(manifest.resolve()),
        "sha256": sha256_file(manifest) if manifest.is_file() else None,
    }


def execution_provenance_from_environment(*, required: bool) -> dict[str, Any] | None:
    """Load the immutable launcher envelope without making it resume identity.

    A resumed process receives a new literal argv but remains part of the same
    training identity.  Therefore this evidence is checkpoint metadata rather
    than a field compared by ``validate_resume_checkpoint``.
    """
    path_value = os.environ.get("EXPERIMENT_PROVENANCE_PATH")
    expected_sha256 = os.environ.get("EXPERIMENT_PROVENANCE_SHA256")
    if not path_value and not expected_sha256:
        if required:
            raise ValueError("registered launch requires EXPERIMENT_PROVENANCE_PATH and SHA-256")
        return None
    if not path_value or not expected_sha256:
        raise ValueError("experiment provenance path/SHA-256 environment is incomplete")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ValueError(f"experiment provenance file is missing: {path}")
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("experiment provenance SHA-256 does not match launcher binding")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("experiment provenance JSON is invalid") from exc
    required_fields = ("repository", "entrypoint", "command", "config", "dataset_manifest")
    if not isinstance(payload, dict) or any(field not in payload for field in required_fields):
        raise ValueError("experiment provenance is missing required fields")
    repository = payload["repository"]
    if not isinstance(repository, dict) or not isinstance(repository.get("git_commit"), str):
        raise ValueError("experiment provenance Git revision is invalid")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "git_commit": repository["git_commit"],
        "entrypoint": payload["entrypoint"],
        "command": payload["command"],
        "config": payload["config"],
        "dataset_manifest": payload["dataset_manifest"],
    }


def evaluation_identity(args: argparse.Namespace, fixed_eval_digest: str | None) -> dict[str, Any]:
    protocol = args.eval_protocol_id or (
        "fixed_eval_indices_v2" if args.fixed_eval_indices else "unseeded_eval_v1"
    )
    return {
        "protocol": protocol,
        "fixed_eval_indices": bool(args.fixed_eval_indices),
        "fixed_eval_indices_sha256": fixed_eval_digest,
        "fixed_eval_index_spec_sha256": getattr(args, "fixed_eval_index_spec_sha256", None),
        "fixed_eval_indices_protocol": getattr(args, "fixed_eval_indices_protocol", None),
        "eval_seed": args.eval_seed,
        "eval_batch_size": args.eval_batch_size,
        "eval_iters": args.eval_iters,
        "block_size": args.block_size,
    }


def build_run_identity(args: argparse.Namespace, data_dir: Path, fixed_eval_digest: str | None) -> dict[str, Any]:
    execution_provenance_from_environment(
        required=bool(getattr(args, "prelaunch_provenance_requirements", None))
    )
    resolved = resolved_config(args)
    manifest = data_manifest_identity(data_dir)
    expected_manifest_hash = getattr(args, "data_manifest_sha256", None)
    if expected_manifest_hash is not None and manifest["sha256"] != expected_manifest_hash:
        raise ValueError("data manifest hash does not match the registered config")
    if getattr(args, "registered_resume_determinism_required", False) and manifest["sha256"] is None:
        raise ValueError("registered deterministic resume requires data_dir/manifest.json")
    evaluation = evaluation_identity(args, fixed_eval_digest)
    # Bind the values actually used to build fixed windows into the resolved
    # config hash.  This prevents a checkpoint sidecar from pairing one
    # evaluation description with a different resolved launch configuration.
    resolved["eval_protocol_id"] = evaluation["protocol"]
    resolved["fixed_eval_indices_sha256"] = evaluation["fixed_eval_indices_sha256"]
    return {
        "resolved_config": resolved,
        "config_sha256": hashlib.sha256(canonical_json_bytes(resolved)).hexdigest(),
        "source_hashes": source_hashes(),
        "data_manifest": manifest,
        "evaluation": evaluation,
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "cpu_torch_rng_state": torch.get_rng_state().detach().clone(),
        "cuda_rng_states": (
            [state.detach().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else None
        ),
        "python_random_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }


def restore_rng_state(checkpoint: dict[str, Any], *, device_type: str) -> None:
    required = ("cpu_torch_rng_state", "cuda_rng_states", "python_random_state", "numpy_rng_state")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError("checkpoint is missing RNG state: " + ", ".join(missing))
    cpu_state = checkpoint["cpu_torch_rng_state"]
    if not isinstance(cpu_state, torch.Tensor):
        raise ValueError("checkpoint CPU torch RNG state is invalid")
    torch.set_rng_state(cpu_state.detach().to(device="cpu", dtype=torch.uint8))
    try:
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint Python or NumPy RNG state is invalid") from exc
    cuda_states = checkpoint["cuda_rng_states"]
    if device_type == "cuda":
        if cuda_states is None:
            raise ValueError("checkpoint is missing CUDA RNG states")
        if not isinstance(cuda_states, (list, tuple)) or not all(isinstance(state, torch.Tensor) for state in cuda_states):
            raise ValueError("checkpoint CUDA RNG states are invalid")
        torch.cuda.set_rng_state_all([state.detach().to(device="cpu", dtype=torch.uint8) for state in cuda_states])


def _fsync_directory(directory: Path) -> None:
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Durably replace only the latest checkpoint, then publish its metadata."""
    if path.name != CHECKPOINT_FILENAME:
        raise ValueError(f"exact-resume checkpoint path must be named {CHECKPOINT_FILENAME}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # The temp file is in ``path.parent``, so replace is same-filesystem and atomic.
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    # Metadata is deliberately written only after the checkpoint replacement is
    # durable. It is informational; the embedded checkpoint metadata remains
    # authoritative if a process dies between these two operations.
    _atomic_write_json(path.parent / CHECKPOINT_METADATA_FILENAME, {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "checkpoint_schema_version": checkpoint["schema_version"],
        "checkpoint_file": path.name,
        "next_iter": checkpoint["next_iter"],
        "saved_at_unix": checkpoint["saved_at_unix"],
        "config_sha256": checkpoint["run_identity"]["config_sha256"],
        "execution_provenance": checkpoint.get("execution_provenance"),
        # The JSON sidecar is intentionally complete enough for the terminal
        # result builder to audit a finished run without importing torch or
        # loading the checkpoint. The checkpoint remains authoritative for
        # exact resume state.
        "run_identity": checkpoint["run_identity"],
    })


def make_checkpoint(
    *,
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    gpt_config: GPTConfig,
    next_iter: int,
    best_val_loss: float,
    pending_train_data_generator_state: torch.Tensor | None,
    run_identity: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    model_state = raw_model.state_dict()
    if any("_cached_weight" in name for name in model_state):
        raise RuntimeError("BlockFHT cache must not be serialized in an exact-resume checkpoint")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "model_config": asdict(gpt_config),
        "next_iter": int(next_iter),
        "best_val_loss": float(best_val_loss),
        "train_data_generator_state": pending_train_data_generator_state,
        "train_data_generator_state_protocol": (
            "dedicated_cpu_generator_pre_current_batch_v1"
            if pending_train_data_generator_state is not None
            else None
        ),
        "run_identity": run_identity,
        "execution_provenance": execution_provenance_from_environment(required=False),
        "saved_at_unix": time.time(),
        "checkpoint_reason": reason,
        "block_fht_cache_state": "flushed_not_serialized",
        **capture_rng_state(),
    }


def validate_resume_checkpoint(
    checkpoint: object,
    *,
    run_identity: dict[str, Any],
    expected_model_config: GPTConfig,
    registered_resume_required: bool,
) -> dict[str, Any]:
    """Reject incomplete or identity-incompatible resume state before CUDA/output setup."""
    if not isinstance(checkpoint, dict):
        raise ValueError("resume checkpoint is invalid")
    required = (
        "schema_version", "model", "optimizer", "grad_scaler", "model_config", "next_iter",
        "best_val_loss", "train_data_generator_state", "run_identity", "saved_at_unix", "block_fht_cache_state",
        "cpu_torch_rng_state", "cuda_rng_states", "python_random_state", "numpy_rng_state",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError("resume checkpoint is missing required state: " + ", ".join(missing))
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("resume checkpoint schema/version is incompatible")
    if checkpoint["block_fht_cache_state"] != "flushed_not_serialized":
        raise ValueError("resume checkpoint BlockFHT cache state is incompatible")
    saved_identity = checkpoint["run_identity"]
    if not isinstance(saved_identity, dict):
        raise ValueError("resume checkpoint run identity is invalid")
    for field, message in (
        ("config_sha256", "config identity"),
        ("source_hashes", "source identity"),
        ("data_manifest", "data identity"),
        ("evaluation", "evaluation identity"),
    ):
        if saved_identity.get(field) != run_identity.get(field):
            raise ValueError(f"resume checkpoint {message} mismatch")
    if saved_identity.get("resolved_config") != run_identity.get("resolved_config"):
        raise ValueError("resume checkpoint resolved config mismatch")
    if checkpoint["model_config"] != asdict(expected_model_config):
        raise ValueError("resume checkpoint model config mismatch")
    if not isinstance(checkpoint["next_iter"], int) or checkpoint["next_iter"] < 0:
        raise ValueError("resume checkpoint next_iter is invalid")
    if not isinstance(checkpoint["best_val_loss"], (int, float)):
        raise ValueError("resume checkpoint best_val_loss is invalid")
    if registered_resume_required and checkpoint["train_data_generator_state"] is None:
        raise ValueError("registered resume checkpoint is missing pre-current-batch train-data generator state")
    if not isinstance(checkpoint["cpu_torch_rng_state"], torch.Tensor):
        raise ValueError("resume checkpoint CPU torch RNG state is invalid")
    return checkpoint


def make_cpu_generator(seed: int | None) -> torch.Generator | None:
    """Create an opt-in CPU RNG stream without touching PyTorch's global RNG."""
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def make_split_cpu_generators(seed: int) -> dict[str, torch.Generator]:
    """Create stable, independent CPU RNG streams for train and validation."""
    generators = {}
    for split_index, split in enumerate(("train", "val")):
        generator = make_cpu_generator(int(seed) + split_index)
        assert generator is not None
        generators[split] = generator
    return generators


def generator_state(generator: torch.Generator | None) -> torch.Tensor | None:
    """Return a checkpoint-safe copy of a dedicated CPU generator state."""
    if generator is None:
        return None
    return generator.get_state().detach().clone()


def restore_generator_state(generator: torch.Generator | None, state: torch.Tensor | None) -> None:
    """Restore a dedicated CPU generator from a checkpointed state."""
    if generator is None:
        raise ValueError("checkpoint has train-data generator state but --train-data-seed is not configured")
    if state is None:
        raise ValueError("checkpoint is missing train-data generator state")
    if not isinstance(state, torch.Tensor):
        raise ValueError("checkpoint train-data generator state is invalid")
    generator.set_state(state.detach().to(device="cpu", dtype=torch.uint8))


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def get_batch(
    data_dir: Path,
    split: str,
    batch_size: int,
    block_size: int,
    device: str,
    *,
    generator: torch.Generator | None = None,
    indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    if indices is None:
        ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    else:
        ix = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if ix.numel() != batch_size:
            raise ValueError(f"expected {batch_size} {split} batch indices, got {ix.numel()}")
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if "cuda" in device:
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def make_fixed_eval_indices(
    data_dir: Path,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    eval_seed: int,
) -> dict[str, torch.Tensor]:
    """Build reproducible evaluation windows using split-local CPU RNG streams."""
    indices = {}
    generators = make_split_cpu_generators(eval_seed)
    for split in ("train", "val"):
        data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        indices[split] = torch.randint(
            len(data) - block_size,
            (eval_iters, batch_size),
            generator=generators[split],
        )
    return indices


def fixed_eval_indices_digest(indices: dict[str, torch.Tensor]) -> str:
    """Return a platform-stable digest for the ordered fixed evaluation windows."""
    digest = hashlib.sha256()
    digest.update(b"fixed_eval_indices_v2\0")
    for split in ("train", "val"):
        values = indices[split].detach().to(device="cpu", dtype=torch.int64).contiguous()
        digest.update(split.encode("ascii") + b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(values.numpy(), dtype="<i8").tobytes())
    return digest.hexdigest()


def validate_dense_fit_gate(config: dict, args: argparse.Namespace) -> None:
    """Validate a pinned accepted dense fit before any candidate can allocate CUDA."""
    needs_dense_fit = bool(config.get("dense_fit_gate_required", False))
    if not needs_dense_fit:
        return
    artifact_name = config.get("dense_fit_artifact")
    expected_hash = config.get("dense_fit_artifact_sha256")
    pinned_coefficients = config.get("dense_fit_coefficients")
    if not isinstance(artifact_name, str) or not artifact_name:
        raise ValueError("launch-ready candidate is missing immutable dense-fit artifact")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("launch-ready candidate is missing dense-fit artifact SHA-256")
    artifact_path = Path(artifact_name)
    if not artifact_path.is_file():
        raise ValueError("launch-ready candidate dense-fit artifact is missing")
    actual_hash = sha256_file(artifact_path)
    if actual_hash != expected_hash:
        raise ValueError("launch-ready candidate dense-fit artifact hash mismatch")
    try:
        artifact = json.loads(artifact_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("launch-ready candidate dense-fit artifact is invalid JSON") from exc
    coefficients = validate_dense_fit_artifact(artifact)
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("launch-ready candidate dense-fit artifact schema/version is incompatible")
    if not isinstance(pinned_coefficients, dict) or pinned_coefficients != coefficients:
        raise ValueError("launch-ready candidate dense-fit coefficients are not exactly pinned")


def validate_mai_selection_runtime_identity(
    config: dict,
    args: argparse.Namespace,
    *,
    data_dir: Path,
    fixed_eval_digest: str | None,
    expected_shared_identity: dict[str, Any] | None,
) -> None:
    """Bind a resolved selection artifact to the actual launch data/eval identity."""
    if expected_shared_identity is None:
        return
    if not args.fixed_eval_indices or fixed_eval_digest is None:
        raise ValueError("selected MAI launch requires fixed evaluation indices")
    if config.get("eval_protocol_id") != expected_shared_identity["eval_protocol_id"]:
        raise ValueError("selected MAI config evaluation protocol disagrees with selection artifact")
    if fixed_eval_digest != expected_shared_identity["fixed_eval_indices_sha256"]:
        raise ValueError("selected MAI runtime fixed-eval digest disagrees with selection artifact")
    actual_manifest = data_manifest_identity(data_dir)["sha256"]
    if (
        config.get("data_manifest_sha256") != expected_shared_identity["data_manifest_sha256"]
        or actual_manifest != expected_shared_identity["data_manifest_sha256"]
    ):
        raise ValueError("selected MAI data manifest disagrees with selection artifact")


def validate_launch_config(config: dict, args: argparse.Namespace) -> dict[str, Any] | None:
    """Reject unresolved registered templates before model or CUDA initialization.

    Missing fields preserve legacy CLI/default behavior.  Only explicitly
    blocked templates and optimizer values resolved to ``null`` are rejected.
    """
    if "mai_ladder_policy_version" in config:
        # The artifact module is deliberately torch-free and is also called by
        # the terminal-result builder. Keep MAI-v2 policy decisions there so a
        # result cannot be accepted for a config training would reject.
        return validate_v2_launch_config(
            config,
            resolved_config=vars(args),
            runtime_source_hashes=source_hashes(),
        )
    if config.get("launch_ready") is False:
        raise ValueError("config is launch-blocked: launch_ready=false")
    if config.get("recipe_resolution_required") is True:
        raise ValueError("config requires recipe resolution before launch")
    if (
        config.get("registered_resume_determinism_required") is True
        and getattr(args, "checkpoint_history", False)
    ):
        raise ValueError("checkpoint history is unsupported; exact resume retains latest ckpt.pt only")
    if config.get("registered_resume_determinism_required") is True:
        if not getattr(args, "save_checkpoint", False):
            raise ValueError("registered deterministic runs require save_checkpoint=true")
        if float(getattr(args, "checkpoint_wall_clock_seconds", 7200.0)) != 7200.0:
            raise ValueError("registered deterministic runs require checkpoint_wall_clock_seconds=7200")

    required_by_optimizer = {
        "adamw": ("learning_rate", "min_lr", "weight_decay", "beta1", "beta2"),
        "muon": (
            "learning_rate",
            "min_lr",
            "weight_decay",
            "beta1",
            "beta2",
            "muon_momentum",
            "muon_ns_steps",
            "muon_adamw_lr_scale",
        ),
    }
    if args.optimizer not in required_by_optimizer:
        raise ValueError("config has unresolved or unsupported optimizer: " + repr(args.optimizer))
    unresolved = [
        field
        for field in required_by_optimizer[args.optimizer]
        if getattr(args, field, None) is None
    ]
    if unresolved:
        raise ValueError(
            "config has unresolved required optimizer fields: " + ", ".join(unresolved)
        )
    validate_dense_fit_gate(config, args)
    return None


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data_dir: Path,
    args,
    ctx,
    cache_model: GPT | None = None,
    cache_dtype: torch.dtype | None = None,
    fixed_eval_indices: dict[str, torch.Tensor] | None = None,
    eval_generators: dict[str, torch.Generator] | None = None,
) -> dict[str, float]:
    model.eval()
    out = {}
    if cache_model is not None:
        cache_model.prepare_block_fht_cache(dtype=cache_dtype)
    try:
        for split in ["train", "val"]:
            losses = torch.zeros(args.eval_iters)
            for idx in range(args.eval_iters):
                batch_indices = None if fixed_eval_indices is None else fixed_eval_indices[split][idx]
                generator = None if eval_generators is None else eval_generators[split]
                x, y = get_batch(
                    data_dir,
                    split,
                    args.eval_batch_size,
                    args.block_size,
                    args.device,
                    generator=generator,
                    indices=batch_indices,
                )
                with ctx:
                    _, loss = model(x, y)
                losses[idx] = loss.item()
            out[split] = float(losses.mean())
    finally:
        if cache_model is not None:
            cache_model.flush_block_fht_cache()
    model.train()
    return out


def cosine_lr(iter_num: int, args) -> float:
    if iter_num < args.warmup_iters:
        return args.learning_rate * (iter_num + 1) / (args.warmup_iters + 1)
    if iter_num > args.lr_decay_iters:
        return args.min_lr
    ratio = (iter_num - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def apply_scheduled_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr * float(group.get("lr_scale", 1.0))


def block_fht_latents(model: GPT) -> list[torch.Tensor]:
    return [module.generator.latent for module in model.modules() if isinstance(module, BlockFHTLinear)]


def latent_rms_hinge_loss(model: GPT, target: float) -> torch.Tensor:
    losses = []
    for latent in block_fht_latents(model):
        rms = latent.float().square().mean().sqrt()
        losses.append(torch.relu(rms - float(target)).square())
    if not losses:
        return next(model.parameters()).new_zeros(())
    return torch.stack(losses).mean()


def perturb_block_fht_latents(model: GPT, sigma: float) -> list[torch.Tensor]:
    noises = []
    with torch.no_grad():
        for latent in block_fht_latents(model):
            noise = torch.randn_like(latent) * float(sigma)
            latent.add_(noise)
            noises.append(noise)
    return noises


def restore_block_fht_latents(model: GPT, noises: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for latent, noise in zip(block_fht_latents(model), noises, strict=True):
            latent.sub_(noise)


def logits_kl_stability_loss(logits: torch.Tensor, perturbed_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = float(temperature)
    reference = F.softmax(logits.detach().float() / temp, dim=-1)
    perturbed_log_probs = F.log_softmax(perturbed_logits.float() / temp, dim=-1)
    return F.kl_div(perturbed_log_probs, reference, reduction="batchmean") * (temp * temp)


def iter_logits_kl_stability_backward_chunks(
    logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    temperature: float,
    chunk_rows: int,
    token_rows: int = 0,
):
    """Yield exact KL values/output gradients without a full FP32 logits copy.

    ``F.kl_div(..., reduction='batchmean') * T^2`` has derivative
    ``T * (softmax(perturbed/T) - softmax(reference/T)) / batch`` with respect
    to perturbed logits. Computing it in bounded rows retains the complete
    token objective while avoiding simultaneous full FP32 logits tensors. A
    positive ``token_rows`` uses evenly distributed token rows and rescales the
    result to a fixed-position estimator of the full-token objective.
    """
    if logits.shape != perturbed_logits.shape or logits.ndim != 3:
        raise ValueError("stability logits must be matching [batch, sequence, vocab] tensors")
    if chunk_rows <= 0:
        raise ValueError("stability chunk_rows must be positive")
    if token_rows < 0:
        raise ValueError("stability token_rows must be non-negative")
    temp = float(temperature)
    if temp <= 0.0 or not math.isfinite(temp):
        raise ValueError("stability temperature must be finite and positive")
    batch_size, sequence_length, vocab_size = perturbed_logits.shape
    flat_reference = logits.detach().reshape(-1, vocab_size)
    flat_perturbed = perturbed_logits.reshape(-1, vocab_size)
    total_rows = flat_perturbed.shape[0]
    if 0 < token_rows < total_rows:
        # Each row is a token position. Data batches remain independently
        # sampled, while this evenly distributed fixed selection avoids an
        # additional RNG stream/checkpoint obligation for the estimator.
        positions = ((torch.arange(token_rows, device=flat_perturbed.device) + 0.5) * total_rows / token_rows).to(torch.long)
        flat_reference = flat_reference.index_select(0, positions)
        flat_perturbed = flat_perturbed.index_select(0, positions)
        normalization = float(sequence_length) / token_rows
    else:
        normalization = 1.0 / batch_size
    for start in range(0, flat_perturbed.shape[0], int(chunk_rows)):
        stop = min(flat_perturbed.shape[0], start + int(chunk_rows))
        # No autograd graph is needed for the analytic output gradient. The
        # caller immediately backpropagates it through the original output
        # slice, freeing each FP32 chunk before the next one.
        with torch.no_grad():
            reference = F.softmax(flat_reference[start:stop].float() / temp, dim=-1)
            perturbed_log_probs = F.log_softmax(flat_perturbed[start:stop].float() / temp, dim=-1)
            value = F.kl_div(perturbed_log_probs, reference, reduction="sum") * (temp * temp * normalization)
            output_gradient = (perturbed_log_probs.exp() - reference) * (temp * normalization)
        yield flat_perturbed[start:stop], value, output_gradient.to(dtype=perturbed_logits.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", required=False)
    parser.add_argument("--out-dir", required=False)
    parser.add_argument("--init-from", choices=["scratch", "resume"], default="scratch")
    parser.add_argument("--method", choices=["baseline", "block_fht"], default="baseline")
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--train-data-seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--fixed-eval-indices", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-protocol-id", default=None)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-wall-clock-seconds", type=float, default=7200.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--lr-decay-iters", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adamw-lr-scale", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--block-fht-latent-ratio", type=float, default=0.01)
    parser.add_argument("--block-fht-latent-ratios", type=json.loads, default=None)
    parser.add_argument("--block-fht-layers", type=int, default=2)
    parser.add_argument("--block-fht-targets", nargs="+", default=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"])
    parser.add_argument("--block-fht-latent-init-std", type=float, default=0.02)
    parser.add_argument("--block-fht-modulation-alpha", type=float, default=0.0)
    parser.add_argument("--block-fht-modulation-centered", action="store_true")
    parser.add_argument("--block-fht-match-gpt-init", action="store_true")
    parser.add_argument("--block-fht-weight-scale", type=float, default=None)
    parser.add_argument("--block-fht-residual-base-scale", type=float, default=0.0)
    parser.add_argument("--block-fht-output-gain-targets", nargs="+", default=[])
    parser.add_argument("--block-fht-input-gain-targets", nargs="+", default=[])
    parser.add_argument("--block-fht-ffn-pregelu-gain", action="store_true")
    parser.add_argument("--block-fht-ffn-pregelu-bias", action="store_true")
    parser.add_argument("--block-fht-ffn-pregelu-bias-init", type=float, default=0.0)
    parser.add_argument("--block-fht-ffn-lowrank-rank", type=int, default=0)
    parser.add_argument("--block-fht-ffn-lowrank-scale", type=float, default=1.0)
    parser.add_argument("--block-fht-ffn-lowrank-init-std", type=float, default=0.02)
    parser.add_argument("--block-fht-ffn-spectral-rank", type=int, default=0)
    parser.add_argument("--block-fht-ffn-spectral-out-groups", type=int, default=1)
    parser.add_argument("--block-fht-ffn-spectral-in-groups", type=int, default=1)
    parser.add_argument("--block-fht-cproj-lowrank-rank", type=int, default=0)
    parser.add_argument("--block-fht-cproj-lowrank-scale", type=float, default=1.0)
    parser.add_argument("--block-fht-cproj-lowrank-init-std", type=float, default=0.02)
    parser.add_argument("--block-fht-cproj-lowrank-mode", choices=["dense", "block_fht"], default="dense")
    parser.add_argument("--block-fht-cproj-lowrank-latent-ratio", type=float, default=None)
    parser.add_argument("--block-fht-cproj-lowrank-b-zero-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-fht-cproj-lowrank-bias", action="store_true")
    parser.add_argument("--block-fht-cproj-tied-cfc-skip", action="store_true")
    parser.add_argument("--block-fht-cproj-tied-cfc-scale-init", type=float, default=0.0)
    parser.add_argument("--block-fht-cproj-tied-cfc-vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-fht-cproj-quarter-diag", action="store_true")
    parser.add_argument("--block-fht-cproj-quarter-diag-scale-init", type=float, default=0.0)
    parser.add_argument("--block-fht-cproj-quarter-diag-init-std", type=float, default=0.02)
    parser.add_argument("--block-fht-cproj-spectral-resid-rank", type=int, default=0)
    parser.add_argument("--block-fht-cproj-spectral-resid-scale-init", type=float, default=0.0)
    parser.add_argument("--block-fht-cproj-spectral-resid-seed", type=int, default=0)
    parser.add_argument("--block-fht-ffn-postgelu-std-target", type=float, default=0.0)
    parser.add_argument("--block-fht-ffn-postgelu-std-lambda", type=float, default=0.0)
    parser.add_argument("--block-fht-seed", type=int, default=1000)
    parser.add_argument("--block-fht-cache-weights", action="store_true")
    parser.add_argument("--freeze-non-block-fht", action="store_true")
    parser.add_argument("--train-embeddings-when-frozen", action="store_true")
    parser.add_argument("--tie-word-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--block-fht-latent-grad-normalize", action="store_true")
    parser.add_argument("--block-fht-latent-grad-target-rms", type=float, default=0.01)
    parser.add_argument("--mapping-stability-lambda", type=float, default=0.0)
    parser.add_argument("--mapping-stability-sigma", type=float, default=1e-3)
    parser.add_argument("--mapping-stability-temperature", type=float, default=1.0)
    parser.add_argument("--mapping-stability-microbatches", type=int, default=0)
    parser.add_argument("--mapping-stability-chunk-rows", type=int, default=2048)
    parser.add_argument("--mapping-stability-token-rows", type=int, default=0)
    parser.add_argument("--mapping-norm-lambda", type=float, default=0.0)
    parser.add_argument("--mapping-norm-target-rms", type=float, default=0.03)
    parser.add_argument("--perf-profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--perf-warmup-iters", type=int, default=5)
    parser.add_argument("--perf-log-interval", type=int, default=10)
    namespace = parser.parse_args()
    config = load_config(namespace.config)
    for key, value in config.items():
        setattr(namespace, key.replace("-", "_"), value)
    if namespace.eval_batch_size is None:
        namespace.eval_batch_size = namespace.batch_size
    namespace._mai_selection_shared_identity = validate_launch_config(config, namespace)
    namespace._launch_config = config
    if namespace.data_dir is None or namespace.out_dir is None:
        raise ValueError("--data-dir and --out-dir are required, either as args or config keys")
    if namespace.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be > 0")
    if namespace.perf_log_interval <= 0:
        raise ValueError("--perf-log-interval must be > 0")
    if namespace.perf_warmup_iters < 0:
        raise ValueError("--perf-warmup-iters must be >= 0")
    if namespace.checkpoint_wall_clock_seconds <= 0:
        raise ValueError("--checkpoint-wall-clock-seconds must be > 0")
    if namespace.mapping_stability_microbatches < 0:
        raise ValueError("--mapping-stability-microbatches must be >= 0")
    if namespace.mapping_stability_microbatches > namespace.gradient_accumulation_steps:
        raise ValueError("--mapping-stability-microbatches cannot exceed gradient accumulation steps")
    if namespace.mapping_stability_chunk_rows <= 0:
        raise ValueError("--mapping-stability-chunk-rows must be positive")
    if namespace.mapping_stability_token_rows < 0:
        raise ValueError("--mapping-stability-token-rows must be non-negative")
    return namespace


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    model_seed = 1337 if args.model_seed is None else int(args.model_seed)
    train_data_generator = make_cpu_generator(args.train_data_seed)
    registered_resume_required = bool(getattr(args, "registered_resume_determinism_required", False))
    if registered_resume_required and train_data_generator is None:
        raise ValueError(
            "registered deterministic resume requires a configured --train-data-seed"
        )
    if args.fixed_eval_indices and args.eval_seed is None:
        raise ValueError("--fixed-eval-indices requires --eval-seed")
    fixed_eval_indices = None
    fixed_eval_digest = None
    eval_generators = None
    if args.fixed_eval_indices:
        fixed_eval_indices = make_fixed_eval_indices(
            data_dir,
            args.eval_batch_size,
            args.block_size,
            args.eval_iters,
            int(args.eval_seed),
        )
        fixed_eval_digest = fixed_eval_indices_digest(fixed_eval_indices)
    elif args.eval_seed is not None:
        eval_generators = make_split_cpu_generators(int(args.eval_seed))
    validate_mai_selection_runtime_identity(
        args._launch_config,
        args,
        data_dir=data_dir,
        fixed_eval_digest=fixed_eval_digest,
        expected_shared_identity=args._mai_selection_shared_identity,
    )
    rng_protocol_enabled = any(
        value is not None
        for value in (args.model_seed, args.train_data_seed, args.eval_seed, args.eval_protocol_id)
    ) or args.fixed_eval_indices
    if rng_protocol_enabled:
        eval_protocol_id = args.eval_protocol_id or (
            "fixed_eval_indices_v2" if args.fixed_eval_indices else "seeded_random_eval_v1"
        )
        print(
            "rng_eval_metadata "
            + json.dumps(
                {
                    "eval_iters": args.eval_iters,
                    "eval_batch_size": args.eval_batch_size,
                    "eval_tokens_per_split": args.eval_batch_size * args.block_size * args.eval_iters,
                    "eval_total_tokens": 2 * args.eval_batch_size * args.block_size * args.eval_iters,
                    "eval_protocol_id": eval_protocol_id,
                    "eval_seed": args.eval_seed,
                    "fixed_eval_indices_sha256": fixed_eval_digest,
                    "model_seed": model_seed,
                    "train_data_seed": args.train_data_seed,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    gpt_config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        block_fht=args.method == "block_fht",
        block_fht_targets=tuple(args.block_fht_targets),
        block_fht_latent_ratio=args.block_fht_latent_ratio,
        block_fht_latent_ratios=args.block_fht_latent_ratios,
        block_fht_layers=args.block_fht_layers,
        block_fht_seed=args.block_fht_seed,
        block_fht_latent_init_std=args.block_fht_latent_init_std,
        block_fht_modulation_alpha=args.block_fht_modulation_alpha,
        block_fht_modulation_centered=args.block_fht_modulation_centered,
        block_fht_match_gpt_init=args.block_fht_match_gpt_init,
        block_fht_weight_scale=args.block_fht_weight_scale,
        block_fht_residual_base_scale=args.block_fht_residual_base_scale,
        block_fht_output_gain_targets=tuple(args.block_fht_output_gain_targets),
        block_fht_input_gain_targets=tuple(args.block_fht_input_gain_targets),
        block_fht_ffn_pregelu_gain=args.block_fht_ffn_pregelu_gain,
        block_fht_ffn_pregelu_bias=args.block_fht_ffn_pregelu_bias,
        block_fht_ffn_pregelu_bias_init=args.block_fht_ffn_pregelu_bias_init,
        block_fht_ffn_lowrank_rank=args.block_fht_ffn_lowrank_rank,
        block_fht_ffn_lowrank_scale=args.block_fht_ffn_lowrank_scale,
        block_fht_ffn_lowrank_init_std=args.block_fht_ffn_lowrank_init_std,
        block_fht_ffn_spectral_rank=args.block_fht_ffn_spectral_rank,
        block_fht_ffn_spectral_out_groups=args.block_fht_ffn_spectral_out_groups,
        block_fht_ffn_spectral_in_groups=args.block_fht_ffn_spectral_in_groups,
        block_fht_cproj_lowrank_rank=args.block_fht_cproj_lowrank_rank,
        block_fht_cproj_lowrank_scale=args.block_fht_cproj_lowrank_scale,
        block_fht_cproj_lowrank_init_std=args.block_fht_cproj_lowrank_init_std,
        block_fht_cproj_lowrank_mode=args.block_fht_cproj_lowrank_mode,
        block_fht_cproj_lowrank_latent_ratio=args.block_fht_cproj_lowrank_latent_ratio,
        block_fht_cproj_lowrank_b_zero_init=args.block_fht_cproj_lowrank_b_zero_init,
        block_fht_cproj_lowrank_bias=args.block_fht_cproj_lowrank_bias,
        block_fht_cproj_tied_cfc_skip=args.block_fht_cproj_tied_cfc_skip,
        block_fht_cproj_tied_cfc_scale_init=args.block_fht_cproj_tied_cfc_scale_init,
        block_fht_cproj_tied_cfc_vector=args.block_fht_cproj_tied_cfc_vector,
        block_fht_cproj_quarter_diag=args.block_fht_cproj_quarter_diag,
        block_fht_cproj_quarter_diag_scale_init=args.block_fht_cproj_quarter_diag_scale_init,
        block_fht_cproj_quarter_diag_init_std=args.block_fht_cproj_quarter_diag_init_std,
        block_fht_cproj_spectral_resid_rank=args.block_fht_cproj_spectral_resid_rank,
        block_fht_cproj_spectral_resid_scale_init=args.block_fht_cproj_spectral_resid_scale_init,
        block_fht_cproj_spectral_resid_seed=args.block_fht_cproj_spectral_resid_seed,
        block_fht_ffn_postgelu_std_target=args.block_fht_ffn_postgelu_std_target,
        tie_word_embeddings=args.tie_word_embeddings,
    )

    # Resume identity is built and checked before creating an output directory,
    # initializing CUDA, or constructing a model on the target device.
    run_identity = build_run_identity(args, data_dir, fixed_eval_digest)
    checkpoint = None
    if args.init_from == "resume":
        checkpoint_path = out_dir / CHECKPOINT_FILENAME
        if not checkpoint_path.is_file():
            raise ValueError(f"resume checkpoint is missing: {checkpoint_path}")
        checkpoint = validate_resume_checkpoint(
            torch.load(checkpoint_path, map_location="cpu", weights_only=False),
            run_identity=run_identity,
            expected_model_config=gpt_config,
            registered_resume_required=registered_resume_required,
        )
        if checkpoint["train_data_generator_state"] is not None:
            restore_generator_state(train_data_generator, checkpoint["train_data_generator_state"])

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    torch.manual_seed(model_seed)
    iter_num = 0
    best_val_loss = 1e9
    if args.init_from == "resume":
        assert checkpoint is not None
        model = GPT(gpt_config)
        model.load_state_dict(checkpoint["model"])
        iter_num = int(checkpoint["next_iter"])
        best_val_loss = float(checkpoint["best_val_loss"])
    else:
        model = GPT(gpt_config)
    model.to(args.device)
    if args.method == "block_fht" and args.freeze_non_block_fht:
        freeze_non_block_fht(model, train_embeddings=args.train_embeddings_when_frozen)
    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
        device_type,
        optimizer=args.optimizer,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        muon_adamw_lr_scale=args.muon_adamw_lr_scale,
    )
    if args.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16")
    if args.init_from == "resume":
        scaler.load_state_dict(checkpoint["grad_scaler"])
        restore_rng_state(checkpoint, device_type=device_type)
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    use_weight_cache = (
        args.method == "block_fht"
        and args.block_fht_cache_weights
    )
    if use_weight_cache and float(args.mapping_stability_lambda) != 0.0:
        print("block_fht: cached CE weights with live-latent stability perturbation forwards")
    tokens_per_iter = args.batch_size * args.block_size * args.gradient_accumulation_steps
    print(f"tokens per iteration: {tokens_per_iter:,}")
    print(f"model_config: {asdict(gpt_config)}")
    total_params = sum(param.numel() for param in raw_model.parameters())
    trainable_params = sum(param.numel() for param in raw_model.parameters() if param.requires_grad)
    print(f"parameters: total={total_params:,} trainable={trainable_params:,}")
    if args.method == "block_fht":
        stats = raw_model.block_fht_stats()
        print(
            "block_fht: "
            f"modules={stats['modules']} generated={stats['generated']:,} latent={stats['latent']:,}"
        )

    # This is the state immediately before ``x, y``.  Evaluation checkpoints
    # persist it so a continuation replays the pending current batch exactly.
    pending_train_data_generator_state = generator_state(train_data_generator)
    x, y = get_batch(
        data_dir,
        "train",
        args.batch_size,
        args.block_size,
        args.device,
        generator=train_data_generator,
    )
    t0 = time.perf_counter()
    perf_peak_reset = False
    if checkpoint is not None:
        saved_at = checkpoint["saved_at_unix"]
        if not isinstance(saved_at, (int, float)) or not math.isfinite(float(saved_at)):
            raise ValueError("resume checkpoint saved_at_unix is invalid")
        elapsed_since_checkpoint = max(0.0, time.time() - float(saved_at))
        last_checkpoint_success_monotonic = time.monotonic() - elapsed_since_checkpoint
    else:
        last_checkpoint_success_monotonic = -math.inf

    def save_exact_resume_checkpoint(reason: str, *, next_iter_value: int) -> None:
        nonlocal checkpoint, last_checkpoint_success_monotonic
        # Cache-backed BlockFHT gradients must be flushed before serializing;
        # calls occur only at evaluation or outer optimizer boundaries.
        raw_model.flush_block_fht_cache()
        checkpoint = make_checkpoint(
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            gpt_config=gpt_config,
            next_iter=next_iter_value,
            best_val_loss=best_val_loss,
            pending_train_data_generator_state=pending_train_data_generator_state,
            run_identity=run_identity,
            reason=reason,
        )
        atomic_save_checkpoint(out_dir / CHECKPOINT_FILENAME, checkpoint)
        # Coalesced wall-clock safety advances only after the atomic replacement
        # (and its post-replacement metadata publication) succeeds.
        last_checkpoint_success_monotonic = time.monotonic()

    def perf_sync() -> None:
        if args.perf_profile and device_type == "cuda":
            torch.cuda.synchronize()

    def perf_now() -> float:
        perf_sync()
        return time.perf_counter()

    while True:
        eval_ms = 0.0
        checkpoint_succeeded_this_iteration = False
        lr = cosine_lr(iter_num, args)
        apply_scheduled_lr(optimizer, lr)
        # The final post-update iteration must be evaluated even when it falls
        # between periodic intervals; the OR avoids a duplicate at a boundary.
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
            cache_model = raw_model if use_weight_cache else None
            eval_start = perf_now() if args.perf_profile else 0.0
            losses = estimate_loss(
                model,
                data_dir,
                args,
                ctx,
                cache_model=cache_model,
                cache_dtype=ptdtype,
                fixed_eval_indices=fixed_eval_indices,
                eval_generators=eval_generators,
            )
            if args.perf_profile:
                eval_ms = (perf_now() - eval_start) * 1000.0
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
            if args.save_checkpoint:
                save_exact_resume_checkpoint("evaluation", next_iter_value=iter_num)
                checkpoint_succeeded_this_iteration = True
        if iter_num >= args.max_iters:
            break

        perf_active = bool(args.perf_profile and iter_num >= args.perf_warmup_iters)
        if perf_active and device_type == "cuda" and not perf_peak_reset:
            torch.cuda.reset_peak_memory_stats()
            perf_peak_reset = True
        iter_start = perf_now() if args.perf_profile else 0.0
        prepare_cache_ms = 0.0
        forward_backward_ms = 0.0
        flush_cache_ms = 0.0
        grad_postprocess_ms = 0.0
        optimizer_ms = 0.0
        data_ms = 0.0

        ce_accum = None
        stability_accum = 0.0
        norm_accum = 0.0
        postgelu_accum = 0.0
        if use_weight_cache:
            section_start = perf_now() if args.perf_profile else 0.0
            raw_model.prepare_block_fht_cache(dtype=ptdtype)
            if args.perf_profile:
                prepare_cache_ms += (perf_now() - section_start) * 1000.0
        # A zero value preserves the legacy objective: evaluate stability on
        # every accumulated microbatch.  A positive value is an explicitly
        # labelled stochastic estimator of that mean regularizer, evaluated on
        # the first N independently sampled microbatches and normalized by N.
        stability_microbatches = (
            args.gradient_accumulation_steps
            if args.mapping_stability_microbatches == 0
            else args.mapping_stability_microbatches
        )
        for microbatch_index in range(args.gradient_accumulation_steps):
            section_start = perf_now() if args.perf_profile else 0.0
            with ctx:
                logits, loss = model(x, y)
                if float(args.mapping_norm_lambda) != 0.0:
                    norm_loss = latent_rms_hinge_loss(raw_model, args.mapping_norm_target_rms)
                    loss = loss + float(args.mapping_norm_lambda) * norm_loss
                    norm_accum += float(norm_loss.detach().item())
                if float(args.block_fht_ffn_postgelu_std_lambda) != 0.0:
                    postgelu_loss = raw_model.postgelu_spread_loss()
                    loss = loss + float(args.block_fht_ffn_postgelu_std_lambda) * postgelu_loss
                    postgelu_accum += float(postgelu_loss.detach().item())
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (
                float(args.mapping_stability_lambda) != 0.0
                and microbatch_index < stability_microbatches
            ):
                # CE used the unperturbed cache above. Suspend it only around
                # this second forward so the KL term differentiates through
                # the actual perturbed latent weights; restore it before the
                # end-of-step cache flush projects CE gradients to latents.
                suspended_cache = raw_model.suspend_block_fht_cache() if use_weight_cache else []
                noises = perturb_block_fht_latents(raw_model, args.mapping_stability_sigma)
                try:
                    with ctx:
                        perturbed_logits, _ = model(x, None)
                    available_rows = perturbed_logits.shape[0] * perturbed_logits.shape[1]
                    selected_rows = (
                        args.mapping_stability_token_rows
                        if 0 < args.mapping_stability_token_rows < available_rows
                        else available_rows
                    )
                    chunks_remaining = math.ceil(selected_rows / args.mapping_stability_chunk_rows)
                    stability_value = 0.0
                    for output_slice, chunk_value, output_gradient in iter_logits_kl_stability_backward_chunks(
                        logits,
                        perturbed_logits,
                        args.mapping_stability_temperature,
                        args.mapping_stability_chunk_rows,
                        args.mapping_stability_token_rows,
                    ):
                        chunks_remaining -= 1
                        output_gradient.mul_(float(args.mapping_stability_lambda) / stability_microbatches)
                        scaler.scale(output_slice).backward(
                            gradient=output_gradient,
                            retain_graph=chunks_remaining > 0,
                        )
                        stability_value += float(chunk_value.item())
                    stability_accum += stability_value
                finally:
                    restore_block_fht_latents(raw_model, noises)
                    if suspended_cache:
                        raw_model.restore_block_fht_cache(suspended_cache)
            ce_loss = loss.detach() * args.gradient_accumulation_steps
            ce_accum = ce_loss if ce_accum is None else ce_accum + ce_loss
            if args.perf_profile:
                forward_backward_ms += (perf_now() - section_start) * 1000.0
                section_start = perf_now()
            pending_train_data_generator_state = generator_state(train_data_generator)
            x, y = get_batch(
                data_dir,
                "train",
                args.batch_size,
                args.block_size,
                args.device,
                generator=train_data_generator,
            )
            if args.perf_profile:
                data_ms += (perf_now() - section_start) * 1000.0
        if use_weight_cache:
            section_start = perf_now() if args.perf_profile else 0.0
            raw_model.flush_block_fht_cache()
            if args.perf_profile:
                flush_cache_ms += (perf_now() - section_start) * 1000.0
        section_start = perf_now() if args.perf_profile else 0.0
        if args.grad_clip != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if args.block_fht_latent_grad_normalize:
            for latent in block_fht_latents(raw_model):
                if latent.grad is None:
                    continue
                grad_rms = latent.grad.float().square().mean().sqrt().clamp_min(1e-12)
                latent.grad.mul_(float(args.block_fht_latent_grad_target_rms) / grad_rms)
        if args.perf_profile:
            grad_postprocess_ms += (perf_now() - section_start) * 1000.0
            section_start = perf_now()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if args.perf_profile:
            optimizer_ms += (perf_now() - section_start) * 1000.0
        if perf_active and (iter_num % args.perf_log_interval == 0 or eval_ms > 0.0):
            iter_ms = (perf_now() - iter_start) * 1000.0
            tokens_per_second = tokens_per_iter / max(iter_ms / 1000.0, 1e-12)
            other_ms = iter_ms - prepare_cache_ms - forward_backward_ms - flush_cache_ms - grad_postprocess_ms - optimizer_ms - data_ms
            peak_mib = 0.0
            if device_type == "cuda":
                peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                "perf "
                f"iter={iter_num} "
                f"tokens_per_s={tokens_per_second:.2f} "
                f"iter_ms={iter_ms:.2f} "
                f"prepare_ms={prepare_cache_ms:.2f} "
                f"fwbw_ms={forward_backward_ms:.2f} "
                f"train_compute_ms={forward_backward_ms:.2f} "
                f"flush_ms={flush_cache_ms:.2f} "
                f"grad_ms={grad_postprocess_ms:.2f} "
                f"opt_ms={optimizer_ms:.2f} "
                f"data_ms={data_ms:.2f} "
                f"other_ms={other_ms:.2f} "
                f"eval_ms={eval_ms:.2f} "
                f"peak_mib={peak_mib:.2f}"
            )
        t1 = time.perf_counter()
        if iter_num % args.log_interval == 0:
            ce_value = float((ce_accum / args.gradient_accumulation_steps).item()) if ce_accum is not None else 0.0
            msg = f"iter {iter_num}: loss {ce_value:.4f}, time {(t1 - t0) * 1000:.2f}ms"
            if float(args.mapping_stability_lambda) != 0.0:
                msg += f", stability {stability_accum / stability_microbatches:.6f}"
            if float(args.mapping_norm_lambda) != 0.0:
                msg += f", norm {norm_accum / args.gradient_accumulation_steps:.6f}"
            if float(args.block_fht_ffn_postgelu_std_lambda) != 0.0:
                msg += f", postgelu {postgelu_accum / args.gradient_accumulation_steps:.6f}"
            print(msg)
        t0 = t1
        # This check runs once per outer optimizer step, never per microbatch.
        # Evaluation saves have already reset the coalesced wall-clock deadline.
        if (
            args.save_checkpoint
            and not checkpoint_succeeded_this_iteration
            and time.monotonic() - last_checkpoint_success_monotonic >= args.checkpoint_wall_clock_seconds
        ):
            save_exact_resume_checkpoint("wall_clock_safety", next_iter_value=iter_num + 1)
        iter_num += 1


if __name__ == "__main__":
    main()
