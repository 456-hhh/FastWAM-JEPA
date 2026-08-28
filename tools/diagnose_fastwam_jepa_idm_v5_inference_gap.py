from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.models.wan22.v5_contract import (  # noqa: E402
    ACTION_HORIZON,
    TOKENS_PER_TEMPORAL_GROUP,
    VISUAL_TOKEN_COUNT,
    VJEPA_DIM,
    canonicalize_v5_batch,
)
from fastwam_jepa_v5_data import (  # noqa: E402
    autocast_context,
    build_v5_loader,
    compose_cfg,
    json_file,
    load_v5_model_checkpoint,
    precision_dtypes,
    provenance_paths,
    require_file,
    seed_everything,
)


PRODUCTION_VISUAL_CONFIG = {
    "num_layers": 30,
    "hidden_dim": 768,
    "ffn_dim": 3072,
    "num_heads": 24,
    "attn_head_dim": 128,
    "vjepa_dim": VJEPA_DIM,
    "text_dim": 4096,
    "spatial_pool_size": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the V5 teacher-forced versus sampled-future inference gap."
    )
    parser.add_argument("--checkpoint", required=True, help="V5 Stage3 step5000 checkpoint.")
    parser.add_argument("--release-checkpoint", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--libero-data-root", required=True)
    parser.add_argument("--vjepa-repo", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint-key", default=None)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", choices=("-1_1", "0_1"), default="-1_1")
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-visual-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite tensor detected in {name}.")


def _finite_float(name: str, value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"Nonfinite scalar detected in {name}: {result}.")
    return result


def _latent_gap_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError(
            f"Latent gap shape mismatch: predicted={tuple(predicted.shape)} "
            f"target={tuple(target.shape)}."
        )
    if predicted.ndim != 3:
        raise ValueError(f"Latent gap inputs must be [B,N,D], got {tuple(predicted.shape)}.")
    _require_finite("predicted_latent", predicted)
    _require_finite("target_latent", target)
    predicted_float = predicted.float()
    target_float = target.float()
    difference = predicted_float - target_float
    predicted_flat = predicted_float.flatten(start_dim=1)
    target_flat = target_float.flatten(start_dim=1)
    predicted_norms = torch.linalg.vector_norm(predicted_flat, dim=1)
    target_norms = torch.linalg.vector_norm(target_flat, dim=1)
    denominator = predicted_norms * target_norms
    if bool((denominator <= 0).any()):
        raise ValueError("Latent cosine is undefined because a sample has zero norm.")
    cosine = (predicted_flat * target_flat).sum(dim=1) / denominator
    predicted_norm = predicted_norms.mean()
    target_norm = target_norms.mean()
    if not bool(target_norm > 0):
        raise ValueError("Latent norm ratio is undefined because the GT norm is zero.")
    return {
        "mse": _finite_float("latent_mse", difference.square().mean()),
        "l1": _finite_float("latent_l1", difference.abs().mean()),
        "cosine": _finite_float("latent_cosine", cosine.mean()),
        "predicted_norm": _finite_float("predicted_latent_norm", predicted_norm),
        "gt_norm": _finite_float("gt_latent_norm", target_norm),
        "norm_ratio": _finite_float("latent_norm_ratio", predicted_norm / target_norm),
    }


def _cache_tensor_gap(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError(
            f"K/V cache shape mismatch: predicted={tuple(predicted.shape)} "
            f"target={tuple(target.shape)}."
        )
    _require_finite("predicted_kv", predicted)
    _require_finite("target_kv", target)
    predicted_float = predicted.float()
    target_float = target.float()
    predicted_norm = torch.linalg.vector_norm(predicted_float)
    target_norm = torch.linalg.vector_norm(target_float)
    if not bool(predicted_norm > 0) or not bool(target_norm > 0):
        raise ValueError("K/V cache gap is undefined because a cache tensor has zero norm.")
    difference_norm = torch.linalg.vector_norm(predicted_float - target_float)
    cosine = (predicted_float * target_float).sum() / (predicted_norm * target_norm)
    return {
        "relative_l2": _finite_float("kv_relative_l2", difference_norm / target_norm),
        "cosine": _finite_float("kv_cosine", cosine),
    }


def _summarize_kv_cache_gap(
    predicted_cache: list[dict[str, torch.Tensor]],
    gt_cache: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    if len(predicted_cache) != len(gt_cache) or not predicted_cache:
        raise ValueError(
            f"K/V cache layer count mismatch: predicted={len(predicted_cache)} "
            f"gt={len(gt_cache)}."
        )
    if len(predicted_cache) % 3 != 0:
        raise ValueError("V5 K/V layer count must divide evenly into early/middle/late groups.")
    per_layer = []
    for layer_index, (predicted_layer, gt_layer) in enumerate(
        zip(predicted_cache, gt_cache)
    ):
        if set(predicted_layer) != {"k", "v"} or set(gt_layer) != {"k", "v"}:
            raise ValueError(f"Layer {layer_index} cache must contain exactly K and V tensors.")
        k_gap = _cache_tensor_gap(predicted_layer["k"], gt_layer["k"])
        v_gap = _cache_tensor_gap(predicted_layer["v"], gt_layer["v"])
        per_layer.append(
            {
                "layer": layer_index,
                "k_relative_l2": k_gap["relative_l2"],
                "k_cosine": k_gap["cosine"],
                "v_relative_l2": v_gap["relative_l2"],
                "v_cosine": v_gap["cosine"],
            }
        )
    group_size = len(per_layer) // 3
    groups = {}
    metric_names = ("k_relative_l2", "k_cosine", "v_relative_l2", "v_cosine")
    for group_index, group_name in enumerate(("early", "middle", "late")):
        start = group_index * group_size
        selected = per_layer[start : start + group_size]
        groups[group_name] = {
            metric_name: sum(item[metric_name] for item in selected) / len(selected)
            for metric_name in metric_names
        }
    return {"per_layer": per_layer, "groups": groups}


def _weighted_action_mse(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_timestep: torch.Tensor,
    action_is_pad: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    expected_action_shape = (int(prediction.shape[0]), ACTION_HORIZON, 7)
    if tuple(prediction.shape) != expected_action_shape or tuple(target.shape) != expected_action_shape:
        raise ValueError(
            f"Action prediction/target must both be {expected_action_shape}, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if tuple(action_timestep.shape) != (expected_action_shape[0],):
        raise ValueError("Action timestep must be [B].")
    if tuple(action_is_pad.shape) != expected_action_shape[:2]:
        raise ValueError("action_is_pad must be [B,16].")
    _require_finite("action_prediction", prediction)
    _require_finite("action_flow_target", target)
    per_token = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=-1)
    valid = ~action_is_pad.to(device=prediction.device, dtype=torch.bool)
    valid_count = valid.sum(dim=1)
    if bool((valid_count == 0).any()):
        raise ValueError("Inference-gap batch contains an all-padding action sample.")
    per_sample = (per_token * valid).sum(dim=1) / valid_count
    weight = scheduler.training_weight(action_timestep).to(
        device=prediction.device, dtype=per_sample.dtype
    )
    if tuple(weight.shape) != (expected_action_shape[0],):
        raise ValueError(f"Action training weight must be [B], got {tuple(weight.shape)}.")
    loss = (per_sample * weight).mean()
    _require_finite("weighted_action_mse", loss)
    return loss


def _evaluate_action_condition(
    *,
    model,
    visual_condition: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    noisy_action: torch.Tensor,
    action_timestep: torch.Tensor,
    action_target: torch.Tensor,
    action_is_pad: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    if tuple(visual_condition.shape[1:]) != (VISUAL_TOKEN_COUNT, VJEPA_DIM):
        raise ValueError(
            f"Clean visual condition must be [B,{VISUAL_TOKEN_COUNT},{VJEPA_DIM}], "
            f"got {tuple(visual_condition.shape)}."
        )
    cache = model._prefill_visual_cache(visual_condition, context, context_mask)
    prediction = model._predict_action_with_cache(
        noisy_action,
        action_timestep,
        context,
        context_mask,
        cache,
    )
    loss = _weighted_action_mse(
        prediction=prediction,
        target=action_target,
        action_timestep=action_timestep,
        action_is_pad=action_is_pad,
        scheduler=model.action_train_scheduler,
    )
    return loss, cache


def _mean_std(values: list[float], *, name: str) -> dict[str, float]:
    if not values:
        raise ValueError(f"Cannot summarize empty metric {name}.")
    tensor = torch.tensor(values, dtype=torch.float64)
    _require_finite(name, tensor)
    return {
        "mean": _finite_float(f"{name}_mean", tensor.mean()),
        "std": _finite_float(f"{name}_std", tensor.std(unbiased=False)),
    }


def _flatten_metrics(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_metrics(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, float):
        output[prefix] = value


def _overall_summary(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch_results:
        raise ValueError("No inference-gap batches were evaluated.")
    flattened = []
    for record in batch_results:
        values: dict[str, float] = {}
        _flatten_metrics("latent_gap", record["latent_gap"], values)
        _flatten_metrics("action_gap", record["action_gap"], values)
        _flatten_metrics("kv_cache_gap.groups", record["kv_cache_gap"]["groups"], values)
        flattened.append(values)
    expected_keys = set(flattened[0])
    if any(set(values) != expected_keys for values in flattened[1:]):
        raise ValueError("Per-batch scalar metric keys are inconsistent.")
    scalar_metrics = {
        key: _mean_std([values[key] for values in flattened], name=key)
        for key in sorted(expected_keys)
    }

    layer_count = len(batch_results[0]["kv_cache_gap"]["per_layer"])
    if any(len(record["kv_cache_gap"]["per_layer"]) != layer_count for record in batch_results):
        raise ValueError("Per-batch K/V layer counts are inconsistent.")
    per_layer = []
    for layer_index in range(layer_count):
        layer_summary: dict[str, Any] = {"layer": layer_index}
        for metric_name in (
            "k_relative_l2",
            "k_cosine",
            "v_relative_l2",
            "v_cosine",
        ):
            layer_summary[metric_name] = _mean_std(
                [
                    record["kv_cache_gap"]["per_layer"][layer_index][metric_name]
                    for record in batch_results
                ],
                name=f"layer_{layer_index}_{metric_name}",
            )
        per_layer.append(layer_summary)
    return {
        "num_batches": len(batch_results),
        "num_samples": sum(int(record["batch_size"]) for record in batch_results),
        "scalar_metrics": scalar_metrics,
        "per_layer_kv_metrics": per_layer,
    }


def _validate_batch(batch: dict[str, torch.Tensor], *, batch_size: int) -> None:
    required = ("video", "action", "context", "context_mask", "proprio", "action_is_pad")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"Inference-gap canonical batch is missing required fields: {missing}.")
    expected_shapes = {
        "video": (batch_size, 3, 5, 224, 448),
        "action": (batch_size, ACTION_HORIZON, 7),
        "proprio": (batch_size, 8),
        "action_is_pad": (batch_size, ACTION_HORIZON),
    }
    for name, expected_shape in expected_shapes.items():
        if tuple(batch[name].shape) != expected_shape:
            raise ValueError(f"{name} must be {expected_shape}, got {tuple(batch[name].shape)}.")
    if batch["context"].ndim != 3 or tuple(batch["context"].shape[::2]) != (
        batch_size,
        4096,
    ):
        raise ValueError(f"context must be [B,L,4096], got {tuple(batch['context'].shape)}.")
    if tuple(batch["context_mask"].shape) != tuple(batch["context"].shape[:2]):
        raise ValueError("context_mask must exactly match context [B,L].")
    if batch["context_mask"].dtype != torch.bool or batch["action_is_pad"].dtype != torch.bool:
        raise ValueError("context_mask and action_is_pad must be boolean tensors.")
    for name in ("video", "action", "context", "proprio"):
        _require_finite(name, batch[name])


def _validate_production_model(model, metadata: dict[str, Any]) -> None:
    visual_config = metadata.get("visual_config")
    if not isinstance(visual_config, dict):
        raise ValueError("Stage3 checkpoint metadata is missing visual_config.")
    mismatches = {
        key: (visual_config.get(key), expected)
        for key, expected in PRODUCTION_VISUAL_CONFIG.items()
        if visual_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint is not the V5 production architecture: {mismatches}.")
    if bool(getattr(model.vjepa_encoder, "dummy", True)):
        raise RuntimeError("Inference-gap diagnosis forbids dummy V-JEPA.")
    if not bool(getattr(model.vjepa_encoder, "freeze", False)):
        raise RuntimeError("Inference-gap diagnosis requires frozen V-JEPA.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Inference-gap diagnosis requires every model parameter to be frozen.")
    parameter_devices = {parameter.device for parameter in model.parameters()}
    if len(parameter_devices) != 1 or next(iter(parameter_devices)).type != "cuda":
        raise RuntimeError(f"Inference-gap model must be entirely on CUDA, got {parameter_devices}.")
    parameter_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if parameter_dtypes != {torch.bfloat16}:
        raise RuntimeError(f"Inference-gap model must use BF16 parameters, got {parameter_dtypes}.")


def _print_batch_summary(record: dict[str, Any]) -> None:
    future = record["latent_gap"]["combined_future"]
    action = record["action_gap"]
    groups = record["kv_cache_gap"]["groups"]
    print(
        f"batch={record['batch_index']} future_mse={future['mse']:.6e} "
        f"future_cosine={future['cosine']:.6f} "
        f"action_loss_gt={action['action_loss_gt_future']:.6e} "
        f"action_loss_pred={action['action_loss_pred_future']:.6e} "
        f"action_loss_ratio={action['action_loss_ratio']:.6f} "
        f"kv_rel_l2_early={groups['early']['k_relative_l2']:.6f} "
        f"kv_rel_l2_middle={groups['middle']['k_relative_l2']:.6f} "
        f"kv_rel_l2_late={groups['late']['k_relative_l2']:.6f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if min(
        args.num_batches,
        args.batch_size,
        args.num_visual_inference_steps,
        args.num_workers + 1,
        args.seed,
    ) <= 0:
        raise ValueError(
            "num-batches, batch-size, visual inference steps, seed must be positive; "
            "num-workers must be nonnegative."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Inference-gap diagnosis requires CUDA; CPU fallback is disabled.")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("--device must be an explicit CUDA device such as cuda:0.")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Inference-gap diagnosis requires CUDA BF16 support.")
    parameter_dtype, autocast_dtype = precision_dtypes(args.precision)
    if parameter_dtype != torch.bfloat16 or autocast_dtype != torch.bfloat16:
        raise RuntimeError("Inference-gap diagnosis is fixed to BF16 precision.")
    seed_everything(args.seed)

    cfg = compose_cfg(args.config_name, args.task)
    loader, _ = build_v5_loader(
        cfg,
        libero_data_root=args.libero_data_root,
        dataset_stats_path=args.dataset_stats_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        ddp_enabled=False,
        world_size=1,
        rank=0,
    )
    paths = provenance_paths(args, rank=0)
    checkpoint_path = require_file(args.checkpoint, name="--checkpoint")
    model, metadata, checkpoint_payload = load_v5_model_checkpoint(
        args,
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        expected_stage="stage3",
        device=device,
        dtype=parameter_dtype,
        provenance=paths,
    )
    if checkpoint_payload.get("global_step") != 5000:
        raise ValueError(
            "Inference-gap diagnosis requires the V5 Stage3 step5000 checkpoint, "
            f"got global_step={checkpoint_payload.get('global_step')}."
        )
    _validate_production_model(model, metadata)
    model.eval()
    del checkpoint_payload

    batch_results: list[dict[str, Any]] = []
    iterator = iter(loader)
    with torch.inference_mode():
        for batch_index in range(args.num_batches):
            try:
                raw_batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"LIBERO loader ended before requested batch {batch_index + 1}."
                ) from exc
            batch = canonicalize_v5_batch(raw_batch, device=device, dtype=parameter_dtype)
            _validate_batch(batch, batch_size=args.batch_size)
            with autocast_context(autocast_dtype):
                z0 = model.encode_current(batch["video"])
                gt_z1, gt_z2 = model.encode_future_gt(batch["video"])
                context, context_mask = model.build_base_context(
                    batch["context"], batch["context_mask"], batch["proprio"]
                )
                sampled = model.infer_future_jepa(
                    z0,
                    context,
                    context_mask,
                    num_inference_steps=args.num_visual_inference_steps,
                    seed=args.seed,
                )
                predicted_z1 = sampled["z1"]
                predicted_z2 = sampled["z2"]
                for name, tensor, expected_shape in (
                    (
                        "z0",
                        z0,
                        (args.batch_size, TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM),
                    ),
                    (
                        "gt_z1",
                        gt_z1,
                        (args.batch_size, TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM),
                    ),
                    (
                        "gt_z2",
                        gt_z2,
                        (args.batch_size, TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM),
                    ),
                    (
                        "predicted_z1",
                        predicted_z1,
                        (args.batch_size, TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM),
                    ),
                    (
                        "predicted_z2",
                        predicted_z2,
                        (args.batch_size, TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM),
                    ),
                ):
                    if tuple(tensor.shape) != expected_shape:
                        raise ValueError(f"{name} must be {expected_shape}, got {tuple(tensor.shape)}.")
                    _require_finite(name, tensor)

                gt_future = torch.cat((gt_z1, gt_z2), dim=1)
                predicted_future = torch.cat((predicted_z1, predicted_z2), dim=1)
                gt_visual_condition = torch.cat((z0, gt_future), dim=1)
                predicted_visual_condition = torch.cat((z0, predicted_future), dim=1)

                action = batch["action"]
                action_timestep = model.action_train_scheduler.sample_training_t(
                    batch_size=args.batch_size,
                    device=device,
                    dtype=action.dtype,
                )
                action_noise = torch.randn_like(action)
                noisy_action = model.action_train_scheduler.add_noise(
                    action, action_noise, action_timestep
                )
                action_target = model.action_train_scheduler.training_target(
                    action, action_noise, action_timestep
                )
                action_loss_gt, gt_cache = _evaluate_action_condition(
                    model=model,
                    visual_condition=gt_visual_condition,
                    context=context,
                    context_mask=context_mask,
                    noisy_action=noisy_action,
                    action_timestep=action_timestep,
                    action_target=action_target,
                    action_is_pad=batch["action_is_pad"],
                )
                action_loss_pred, predicted_cache = _evaluate_action_condition(
                    model=model,
                    visual_condition=predicted_visual_condition,
                    context=context,
                    context_mask=context_mask,
                    noisy_action=noisy_action,
                    action_timestep=action_timestep,
                    action_target=action_target,
                    action_is_pad=batch["action_is_pad"],
                )

            gt_loss_value = _finite_float("action_loss_gt_future", action_loss_gt)
            predicted_loss_value = _finite_float(
                "action_loss_pred_future", action_loss_pred
            )
            if gt_loss_value <= 0:
                raise ValueError("GT-future weighted action MSE must be positive.")
            batch_record = {
                "batch_index": batch_index,
                "batch_size": args.batch_size,
                "latent_gap": {
                    "z1": _latent_gap_metrics(predicted_z1, gt_z1),
                    "z2": _latent_gap_metrics(predicted_z2, gt_z2),
                    "combined_future": _latent_gap_metrics(predicted_future, gt_future),
                },
                "action_gap": {
                    "action_loss_gt_future": gt_loss_value,
                    "action_loss_pred_future": predicted_loss_value,
                    "action_loss_ratio": _finite_float(
                        "action_loss_ratio", predicted_loss_value / gt_loss_value
                    ),
                    "action_timestep_mean": _finite_float(
                        "action_timestep_mean", action_timestep.float().mean()
                    ),
                    "action_noise_norm": _finite_float(
                        "action_noise_norm",
                        torch.linalg.vector_norm(action_noise.float(), dim=(1, 2)).mean(),
                    ),
                },
                "kv_cache_gap": _summarize_kv_cache_gap(predicted_cache, gt_cache),
            }
            batch_results.append(batch_record)
            _print_batch_summary(batch_record)

    overall = _overall_summary(batch_results)
    output_path = Path(args.output_json).expanduser().resolve()
    payload = {
        "diagnostic": "fastwam_jepa_idm_v5_inference_gap",
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": 5000,
        "precision": args.precision,
        "device": str(device),
        "seed": args.seed,
        "num_visual_inference_steps": args.num_visual_inference_steps,
        "production_visual_config": dict(PRODUCTION_VISUAL_CONFIG),
        "batches": batch_results,
        "overall": overall,
    }
    json_file(output_path, payload)
    scalar_metrics = overall["scalar_metrics"]
    print(
        "overall "
        f"future_mse={scalar_metrics['latent_gap.combined_future.mse']['mean']:.6e} "
        f"future_cosine={scalar_metrics['latent_gap.combined_future.cosine']['mean']:.6f} "
        f"action_loss_ratio={scalar_metrics['action_gap.action_loss_ratio']['mean']:.6f} "
        f"output_json={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
