"""Test whether a teacher-fitted MLP chart improves complete-model CE.

The bilateral ``c_proj`` oracle recovers much of the attention-only teacher's
MLP output error on held-out activations, but causal CE training does not
realize that capacity.  This endpoint diagnostic fits the exact production
chart to frozen teacher MLP outputs, then rolls the fitted charts into the
complete plain-``c_proj`` language model and measures CE on deterministic fit
and holdout token windows.

No language-model parameter update is applied.  The teacher-fitted chart is
an oracle diagnostic, not a deployable teacher-free method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from examples.nanogpt.analyze_cproj_manifold import load_model
from examples.nanogpt.analyze_mlp_activation_chart_oracle import (
    prediction_metrics,
)
from examples.nanogpt.analyze_mlp_chart_gradient_alignment import (
    SplitData,
    chart_parameters,
    collect_split,
    load_chart_model,
)
from examples.nanogpt.model import GPT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def parse_layers(value: str) -> list[int]:
    layers = [int(part) for part in value.split(",") if part.strip()]
    if not layers or len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError(
            "layers must be a non-empty list of unique integers"
        )
    return layers


def capture_chart_state(
    model: GPT, layers: list[int]
) -> dict[str, torch.Tensor]:
    return {
        key: parameter.detach().float().cpu().clone()
        for key, parameter in chart_parameters(model, layers).items()
    }


def restore_chart_state(
    model: GPT,
    layers: list[int],
    state: dict[str, torch.Tensor],
) -> None:
    parameters = chart_parameters(model, layers)
    if set(parameters) != set(state):
        raise ValueError("chart state does not match model coordinates")
    with torch.no_grad():
        for key, parameter in parameters.items():
            parameter.copy_(
                state[key].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


def combine_chart_states(
    initial: dict[str, torch.Tensor],
    fitted: dict[str, torch.Tensor],
    selected_layers: list[int],
) -> dict[str, torch.Tensor]:
    if set(initial) != set(fitted):
        raise ValueError("initial and fitted chart states do not match")
    prefixes = tuple(f"layer.{int(layer)}." for layer in selected_layers)
    return {
        key: (
            fitted[key].clone()
            if key.startswith(prefixes)
            else initial[key].clone()
        )
        for key in initial
    }


def chart_prediction(
    model: GPT,
    layer: int,
    pre_gelu: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    mlp = model.transformer.h[layer].mlp
    charted_weight = mlp._materialize_charted_cproj_weight(weight)
    return F.linear(F.gelu(pre_gelu), charted_weight, bias)


def fit_layer_chart(
    model: GPT,
    layer: int,
    fit: SplitData,
    holdout: SplitData,
    *,
    device: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    selected = {
        key: parameter
        for key, parameter in chart_parameters(model, [layer]).items()
    }
    optimizer = torch.optim.Adam(
        list(selected.values()), lr=float(learning_rate)
    )
    train_pre = fit.pre_gelu[layer].to(device)
    train_target = fit.teacher_mlp_out[layer].to(device)
    holdout_pre = holdout.pre_gelu[layer].to(device)
    holdout_target = holdout.teacher_mlp_out[layer].to(device)
    weight = fit.cproj_weight[layer].to(device)
    holdout_weight = holdout.cproj_weight[layer].to(device)
    bias = fit.cproj_bias[layer]
    bias = bias.to(device) if bias is not None else None
    holdout_bias = holdout.cproj_bias[layer]
    holdout_bias = (
        holdout_bias.to(device) if holdout_bias is not None else None
    )
    with torch.no_grad():
        train_identity = chart_prediction(
            model, layer, train_pre, weight, bias
        )
        holdout_identity = chart_prediction(
            model,
            layer,
            holdout_pre,
            holdout_weight,
            holdout_bias,
        )

    generator = torch.Generator(device=train_pre.device)
    generator.manual_seed(int(seed) + int(layer) * 1009)
    for _ in range(int(steps)):
        indices = torch.randint(
            train_pre.shape[0],
            (min(int(batch_size), train_pre.shape[0]),),
            generator=generator,
            device=train_pre.device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = chart_prediction(
            model,
            layer,
            train_pre.index_select(0, indices),
            weight,
            bias,
        )
        target = train_target.index_select(0, indices)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_prediction = chart_prediction(
            model, layer, train_pre, weight, bias
        )
        holdout_prediction = chart_prediction(
            model,
            layer,
            holdout_pre,
            holdout_weight,
            holdout_bias,
        )
    row: dict[str, object] = {
        "layer": int(layer),
        "parameter_count": int(
            sum(parameter.numel() for parameter in selected.values())
        ),
    }
    row.update(
        {
            f"fit_{key}": value
            for key, value in prediction_metrics(
                train_target, train_prediction, train_identity
            ).items()
        }
    )
    row.update(
        {
            f"holdout_{key}": value
            for key, value in prediction_metrics(
                holdout_target,
                holdout_prediction,
                holdout_identity,
            ).items()
        }
    )
    for key, parameter in selected.items():
        group = key.rsplit(".", 1)[-1]
        row[f"{group}_rms"] = float(
            parameter.detach().float().square().mean().sqrt()
        )
    return row


def evaluate_model_ce(
    model: GPT,
    batches: list[torch.Tensor],
    device: str,
) -> float:
    model.eval()
    cache_dtype = (
        torch.bfloat16 if device.startswith("cuda") else torch.float32
    )
    model.flush_block_fht_cache()
    with torch.no_grad():
        model.prepare_block_fht_cache(dtype=cache_dtype)
        losses: list[float] = []
        try:
            for tokens in batches:
                tokens = tokens.to(device)
                inputs = tokens[:, :-1].contiguous()
                targets = tokens[:, 1:].contiguous()
                context = (
                    torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    )
                    if device.startswith("cuda")
                    else torch.autocast(
                        device_type="cpu", enabled=False
                    )
                )
                with context:
                    _, loss = model(inputs, targets)
                if loss is None:
                    raise RuntimeError("model returned no CE loss")
                losses.append(float(loss))
        finally:
            model.flush_block_fht_cache()
    return float(np.mean(losses))


def evaluate_variant(
    model: GPT,
    layers: list[int],
    state: dict[str, torch.Tensor],
    fit: SplitData,
    holdout: SplitData,
    device: str,
    *,
    name: str,
    selected_layers: list[int],
) -> dict[str, object]:
    restore_chart_state(model, layers, state)
    fit_ce = evaluate_model_ce(model, fit.batches, device)
    holdout_ce = evaluate_model_ce(model, holdout.batches, device)
    print(
        f"variant={name} fit_ce={fit_ce:.6f} "
        f"holdout_ce={holdout_ce:.6f}",
        flush=True,
    )
    return {
        "variant": name,
        "selected_layers": ",".join(str(layer) for layer in selected_layers),
        "fit_ce": fit_ce,
        "holdout_ce": holdout_ce,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", required=True, type=Path)
    parser.add_argument("--plain-cproj", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=parse_layers(",".join(str(layer) for layer in range(12))),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--sample-cap", type=int, default=2048)
    parser.add_argument("--fit-token-seed", type=int, default=20260716)
    parser.add_argument("--holdout-token-seed", type=int, default=20260717)
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--fit-batch-size", type=int, default=256)
    parser.add_argument("--fit-learning-rate", type=float, default=0.02)
    parser.add_argument("--fit-seed", type=int, default=314159)
    parser.add_argument(
        "--initial-output-log-gain", type=float, default=0.0
    )
    parser.add_argument("--minimum-ce-gain", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    layers = list(args.layers)
    if args.fit_steps <= 0 or args.fit_batch_size <= 0:
        raise ValueError("fit steps and batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    print("collecting deterministic fit split", flush=True)
    fit = collect_split(
        name="fit",
        seed=args.fit_token_seed,
        attention_only=args.attention_only,
        plain_cproj=args.plain_cproj,
        data_dir=args.data_dir,
        layers=layers,
        batch_size=args.batch_size,
        block_size=args.block_size,
        batches_count=args.batches,
        sample_cap=args.sample_cap,
        device=args.device,
    )
    print("collecting deterministic holdout split", flush=True)
    holdout = collect_split(
        name="holdout",
        seed=args.holdout_token_seed,
        attention_only=args.attention_only,
        plain_cproj=args.plain_cproj,
        data_dir=args.data_dir,
        layers=layers,
        batch_size=args.batch_size,
        block_size=args.block_size,
        batches_count=args.batches,
        sample_cap=args.sample_cap,
        device=args.device,
    )

    print("loading production bilateral chart model", flush=True)
    model = load_chart_model(
        args.plain_cproj,
        args.device,
        layers,
        args.initial_output_log_gain,
    )
    initial_state = capture_chart_state(model, layers)
    fit_rows: list[dict[str, object]] = []
    for layer in layers:
        print(f"fitting layer={layer}", flush=True)
        fit_rows.append(
            fit_layer_chart(
                model,
                layer,
                fit,
                holdout,
                device=args.device,
                steps=args.fit_steps,
                batch_size=args.fit_batch_size,
                learning_rate=args.fit_learning_rate,
                seed=args.fit_seed,
            )
        )
    fitted_state = capture_chart_state(model, layers)

    ce_rows: list[dict[str, object]] = []
    ce_rows.append(
        evaluate_variant(
            model,
            layers,
            initial_state,
            fit,
            holdout,
            args.device,
            name="plain_cproj_identity",
            selected_layers=[],
        )
    )
    for layer in layers:
        ce_rows.append(
            evaluate_variant(
                model,
                layers,
                combine_chart_states(initial_state, fitted_state, [layer]),
                fit,
                holdout,
                args.device,
                name=f"fitted_layer_{layer}",
                selected_layers=[layer],
            )
        )
    ce_rows.append(
        evaluate_variant(
            model,
            layers,
            fitted_state,
            fit,
            holdout,
            args.device,
            name="fitted_all_layers",
            selected_layers=layers,
        )
    )

    print("evaluating attention-only reference", flush=True)
    teacher = load_model(args.attention_only, args.device)
    teacher_row = {
        "variant": "attention_only_reference",
        "selected_layers": "",
        "fit_ce": evaluate_model_ce(teacher, fit.batches, args.device),
        "holdout_ce": evaluate_model_ce(
            teacher, holdout.batches, args.device
        ),
    }
    ce_rows.append(teacher_row)
    del teacher
    if "cuda" in args.device:
        torch.cuda.empty_cache()

    identity = ce_rows[0]
    single_rows = [
        row for row in ce_rows if str(row["variant"]).startswith("fitted_layer_")
    ]
    best_fit = min(single_rows, key=lambda row: float(row["fit_ce"]))
    all_layers = next(
        row for row in ce_rows if row["variant"] == "fitted_all_layers"
    )
    identity_fit = float(identity["fit_ce"])
    identity_holdout = float(identity["holdout_ce"])
    for row in ce_rows:
        row["fit_gain_vs_identity"] = identity_fit - float(row["fit_ce"])
        row["holdout_gain_vs_identity"] = (
            identity_holdout - float(row["holdout_ce"])
        )
    all_positive = (
        float(all_layers["holdout_gain_vs_identity"])
        >= args.minimum_ce_gain
    )
    fit_selected_positive = (
        float(best_fit["fit_gain_vs_identity"]) >= args.minimum_ce_gain
        and float(best_fit["holdout_gain_vs_identity"])
        >= args.minimum_ce_gain
    )
    if all_positive:
        decision = "POSITIVE_TASK_DIRECTION_ALL_LAYER_ROLLOUT"
    elif fit_selected_positive:
        decision = "POSITIVE_BUT_NONCOMPOSITIONAL_SINGLE_LAYER_DIRECTION"
    else:
        decision = "REJECT_TEACHER_MLP_REPAIR_AS_TASK_DIRECTION"

    fit_csv = args.output / "bilateral_endpoint_mlp_fit.csv"
    ce_csv = args.output / "bilateral_endpoint_full_lm_ce.csv"
    state_path = args.output / "bilateral_endpoint_chart_state.pt"
    write_csv(fit_csv, fit_rows)
    write_csv(ce_csv, ce_rows)
    torch.save(
        {
            "schema_version": "bilateral_endpoint_chart_state_v1",
            "layers": layers,
            "initial_output_log_gain": args.initial_output_log_gain,
            "initial_state": initial_state,
            "fitted_state": fitted_state,
        },
        state_path,
    )

    root = Path(__file__).resolve().parents[2]
    plan_path = (
        root
        / "examples/nanogpt/configs/selection_artifacts/"
        "124m_mlp_bilateral_endpoint_ce_oracle_plan.json"
    )
    manifest_path = args.data_dir / "manifest.json"
    summary = {
        "schema_version": "mai_124m_mlp_bilateral_endpoint_ce_oracle_v1",
        "scientific_scope": (
            "teacher-guided endpoint oracle; no language-model update"
        ),
        "repository_commit": git_head(root),
        "command": sys.argv,
        "source": {
            "path": str(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "plan": {
            "path": str(plan_path),
            "sha256": sha256(plan_path),
        },
        "attention_only": {
            "path": str(args.attention_only),
            "sha256": sha256(args.attention_only),
        },
        "plain_cproj": {
            "path": str(args.plain_cproj),
            "sha256": sha256(args.plain_cproj),
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "protocol": {
            "layers": layers,
            "fit_steps": args.fit_steps,
            "fit_batch_size": args.fit_batch_size,
            "fit_learning_rate": args.fit_learning_rate,
            "fit_seed": args.fit_seed,
            "sample_cap": args.sample_cap,
            "fit_token_seed": args.fit_token_seed,
            "fit_token_sha256": fit.token_sha256,
            "holdout_token_seed": args.holdout_token_seed,
            "holdout_token_sha256": holdout.token_sha256,
            "full_lm_batches_per_split": args.batches,
            "full_lm_batch_size": args.batch_size,
            "full_lm_block_size": args.block_size,
            "initial_output_log_gain": args.initial_output_log_gain,
            "language_model_parameter_updates": 0,
        },
        "artifacts": {
            "mlp_fit_csv": {
                "path": str(fit_csv),
                "sha256": sha256(fit_csv),
            },
            "full_lm_ce_csv": {
                "path": str(ce_csv),
                "sha256": sha256(ce_csv),
            },
            "chart_state": {
                "path": str(state_path),
                "sha256": sha256(state_path),
            },
        },
        "fit_rows": fit_rows,
        "ce_rows": ce_rows,
        "selection": {
            "minimum_ce_gain": args.minimum_ce_gain,
            "best_single_layer_by_fit": best_fit["variant"],
            "best_single_layer_fit_gain": best_fit[
                "fit_gain_vs_identity"
            ],
            "best_single_layer_holdout_gain": best_fit[
                "holdout_gain_vs_identity"
            ],
            "all_layer_fit_gain": all_layers["fit_gain_vs_identity"],
            "all_layer_holdout_gain": all_layers[
                "holdout_gain_vs_identity"
            ],
            "decision": decision,
            "launch_causal_training": bool(
                all_positive or fit_selected_positive
            ),
        },
    }
    summary_path = args.output / "bilateral_endpoint_ce_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["selection"], indent=2, sort_keys=True))
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
