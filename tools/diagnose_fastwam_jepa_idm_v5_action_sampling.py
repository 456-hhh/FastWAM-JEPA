from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT,
    PROJECT_ROOT / "tools",
    PROJECT_ROOT / "experiments" / "libero",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_fastwam_jepa_idm_v2_predictor_value as rollout_utils  # noqa: E402
from diagnose_fastwam_jepa_idm_v5_inference_gap import (  # noqa: E402
    PRODUCTION_VISUAL_CONFIG,
    _evaluate_action_condition,
    _latent_gap_metrics,
    _validate_batch,
    _validate_production_model,
)
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


ACTION_DIMENSION_NAMES = ("dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper")
EXEC_HORIZON = 4
SIGNIFICANT_FUTURE_GAP_RATIO = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose V5 full action sampling with GT versus sampled future tokens."
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
    parser.add_argument("--num-action-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _require_finite(name: str, value: torch.Tensor | np.ndarray) -> None:
    finite = torch.isfinite(value).all() if torch.is_tensor(value) else np.isfinite(value).all()
    if not bool(finite):
        raise FloatingPointError(f"Nonfinite value detected in {name}.")


def _finite_float(name: str, value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"Nonfinite scalar detected in {name}: {result}.")
    return result


def _validate_actions(
    prediction: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor
) -> torch.Tensor:
    expected_shape = (int(prediction.shape[0]), ACTION_HORIZON, len(ACTION_DIMENSION_NAMES))
    if tuple(prediction.shape) != expected_shape or tuple(target.shape) != expected_shape:
        raise ValueError(
            f"Action tensors must both be {expected_shape}, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if tuple(action_is_pad.shape) != expected_shape[:2] or action_is_pad.dtype != torch.bool:
        raise ValueError(f"action_is_pad must be boolean {expected_shape[:2]}.")
    _require_finite("action_prediction", prediction)
    _require_finite("action_target", target)
    valid = ~action_is_pad.to(device=prediction.device)
    if bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("Every action sample must contain at least one valid target step.")
    return valid


def _masked_action_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: torch.Tensor,
    *,
    horizon: int = ACTION_HORIZON,
) -> dict[str, Any]:
    valid = _validate_actions(prediction, target, action_is_pad)
    if horizon <= 0 or horizon > ACTION_HORIZON:
        raise ValueError(f"horizon must be in [1,{ACTION_HORIZON}], got {horizon}.")
    error = prediction.float()[:, :horizon] - target.float()[:, :horizon]
    valid = valid[:, :horizon]
    if not bool(valid.any()):
        raise ValueError(f"No valid action targets exist in the first {horizon} steps.")
    expanded_valid = valid.unsqueeze(-1).expand_as(error)
    selected = error[expanded_valid]
    per_horizon_mae = []
    for step in range(horizon):
        step_valid = valid[:, step]
        if not bool(step_valid.any()):
            raise ValueError(f"No valid action target exists at horizon step {step}.")
        per_horizon_mae.append(
            _finite_float(f"horizon_{step}_mae", error[step_valid, step].abs().mean())
        )
    per_action_dimension_mae = {
        name: _finite_float(f"{name}_mae", error[..., index][valid].abs().mean())
        for index, name in enumerate(ACTION_DIMENSION_NAMES)
    }
    return {
        "mse": _finite_float("normalized_action_mse", selected.square().mean()),
        "mae": _finite_float("normalized_action_mae", selected.abs().mean()),
        "valid_action_steps": int(valid.sum().item()),
        "per_horizon_step_mae": per_horizon_mae,
        "per_action_dimension_mae": per_action_dimension_mae,
    }


def _action_pair_metrics(
    first: torch.Tensor, second: torch.Tensor, action_is_pad: torch.Tensor
) -> dict[str, float]:
    valid = _validate_actions(first, second, action_is_pad)
    first_selected = first.float()[valid].flatten()
    second_selected = second.float()[valid].flatten()
    first_norm = torch.linalg.vector_norm(first_selected)
    second_norm = torch.linalg.vector_norm(second_selected)
    if not bool(first_norm > 0) or not bool(second_norm > 0):
        raise ValueError("Sampled-action cosine is undefined because an action has zero norm.")
    difference = first_selected - second_selected
    return {
        "mse": _finite_float("sample_pair_mse", difference.square().mean()),
        "mae": _finite_float("sample_pair_mae", difference.abs().mean()),
        "cosine": _finite_float(
            "sample_pair_cosine", (first_selected * second_selected).sum() / (first_norm * second_norm)
        ),
    }


def _per_dimension_distribution(
    action: torch.Tensor | np.ndarray, action_is_pad: torch.Tensor
) -> dict[str, dict[str, float]]:
    tensor = torch.as_tensor(action, dtype=torch.float64, device="cpu")
    expected_shape = (int(action_is_pad.shape[0]), ACTION_HORIZON, len(ACTION_DIMENSION_NAMES))
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Distribution action must be {expected_shape}, got {tuple(tensor.shape)}.")
    if tuple(action_is_pad.shape) != expected_shape[:2] or action_is_pad.dtype != torch.bool:
        raise ValueError(f"action_is_pad must be boolean {expected_shape[:2]}.")
    _require_finite("denormalized_action", tensor)
    valid = ~action_is_pad.to(device="cpu")
    if not bool(valid.any()):
        raise ValueError("Cannot summarize an all-padding action batch.")
    result = {}
    for index, name in enumerate(ACTION_DIMENSION_NAMES):
        values = tensor[..., index][valid]
        result[name] = {
            "mean": _finite_float(f"{name}_mean", values.mean()),
            "std": _finite_float(f"{name}_std", values.std(unbiased=False)),
            "min": _finite_float(f"{name}_min", values.min()),
            "max": _finite_float(f"{name}_max", values.max()),
        }
    return result


def _mean_std(values: list[float], *, name: str) -> dict[str, float]:
    if not values:
        raise ValueError(f"Cannot summarize empty metric {name}.")
    tensor = torch.tensor(values, dtype=torch.float64)
    _require_finite(name, tensor)
    return {
        "mean": _finite_float(f"{name}_mean", tensor.mean()),
        "std": _finite_float(f"{name}_std", tensor.std(unbiased=False)),
    }


def _diagnostic_hint(
    *,
    gt_first4_mae: float,
    sampled_first4_mae: float,
    zero_first4_mae: float,
) -> dict[str, Any]:
    values = (gt_first4_mae, sampled_first4_mae, zero_first4_mae)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"Diagnostic MAE inputs must be finite and positive, got {values}.")
    ratio = sampled_first4_mae / gt_first4_mae
    if gt_first4_mae >= zero_first4_mae:
        category = "A"
        conclusion = (
            "GT-future full action sampling does not beat the zero-normalized-action baseline; "
            "ActionDiT, action flow sampling, or the action scheduler path is the primary suspect."
        )
    elif ratio >= SIGNIFICANT_FUTURE_GAP_RATIO:
        category = "B"
        conclusion = (
            "GT-future action sampling beats the zero baseline, but sampled future degrades "
            "first-4 MAE substantially; future generation/inference gap is the primary suspect."
        )
    else:
        category = "C"
        conclusion = (
            "Offline GT-future sampling beats the zero baseline without a large sampled-future "
            "penalty; inspect rollout inputs, denormalization, gripper handling, execution horizon, "
            "and closed-loop distribution shift."
        )
    return {
        "category": category,
        "conclusion": conclusion,
        "gt_first4_mae": gt_first4_mae,
        "sampled_first4_mae": sampled_first4_mae,
        "zero_first4_mae": zero_first4_mae,
        "sampled_to_gt_first4_mae_ratio": ratio,
        "significant_future_gap_ratio": SIGNIFICANT_FUTURE_GAP_RATIO,
        "gt_poor_rule": "gt_first4_mae >= zero_normalized_action_first4_mae",
    }


def _require_action_processor(loader):
    dataset = loader.dataset
    if not hasattr(dataset, "lerobot_dataset"):
        raise TypeError("V5 LIBERO loader dataset must expose lerobot_dataset.")
    processor = getattr(dataset.lerobot_dataset, "processor", None)
    if processor is None:
        raise RuntimeError("V5 LIBERO dataset has no fitted action processor.")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("V5 action denormalization requires one merged action key.")
    action_key = action_meta[0]["key"]
    processor.normalizer.normalizers["action"][action_key]
    return processor


def _denormalize_action(action: torch.Tensor, processor) -> np.ndarray:
    denormalized = rollout_utils._denormalize_action(action, processor)
    expected_shape = (int(action.shape[0]), ACTION_HORIZON, len(ACTION_DIMENSION_NAMES))
    if tuple(denormalized.shape) != expected_shape:
        raise ValueError(
            f"Formal action denormalization must return {expected_shape}, "
            f"got {tuple(denormalized.shape)}."
        )
    _require_finite("formal_denormalized_action", denormalized)
    return denormalized


def _validate_latent(name: str, value: torch.Tensor, batch_size: int, token_count: int) -> None:
    expected_shape = (batch_size, token_count, VJEPA_DIM)
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} must be {expected_shape}, got {tuple(value.shape)}.")
    _require_finite(name, value)


def _print_batch(record: dict[str, Any]) -> None:
    gt = record["gt_future_full_sampling"]
    predicted = record["sampled_future_full_sampling"]
    print(
        f"batch={record['batch_index']} gt_mse={gt['mse']:.6e} gt_mae={gt['mae']:.6e} "
        f"sampled_mse={predicted['mse']:.6e} sampled_mae={predicted['mae']:.6e} "
        f"mae_ratio={record['full_sampling_mae_ratio']:.6f} "
        f"first4_ratio={record['first4']['mae_ratio']:.6f} "
        f"one_step_ratio={record['one_step_flow']['loss_ratio']:.6f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if min(
        args.num_batches,
        args.batch_size,
        args.num_visual_inference_steps,
        args.num_action_inference_steps,
        args.seed,
    ) <= 0 or args.num_workers < 0:
        raise ValueError(
            "num-batches, batch-size, visual/action inference steps, and seed must be "
            "positive; num-workers must be nonnegative."
        )
    required_protocol = {
        "num_batches": 20,
        "batch_size": 4,
        "num_visual_inference_steps": 10,
        "num_action_inference_steps": 10,
        "seed": 42,
    }
    protocol_mismatches = {
        name: (getattr(args, name), expected)
        for name, expected in required_protocol.items()
        if getattr(args, name) != expected
    }
    if protocol_mismatches:
        raise ValueError(
            f"V5 action-sampling diagnosis requires the fixed protocol: {protocol_mismatches}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("V5 action-sampling diagnosis requires CUDA; CPU fallback is disabled.")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("--device must be an explicit CUDA device such as cuda:0.")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("V5 action-sampling diagnosis requires CUDA BF16 support.")
    parameter_dtype, autocast_dtype = precision_dtypes(args.precision)
    if parameter_dtype != torch.bfloat16 or autocast_dtype != torch.bfloat16:
        raise RuntimeError("V5 action-sampling diagnosis is fixed to BF16 precision.")
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
    processor = _require_action_processor(loader)
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
            "V5 action-sampling diagnosis requires the Stage3 step5000 checkpoint, "
            f"got global_step={checkpoint_payload.get('global_step')}."
        )
    _validate_production_model(model, metadata)
    model.eval()
    del checkpoint_payload

    batch_results: list[dict[str, Any]] = []
    normalized_target: list[torch.Tensor] = []
    normalized_gt: list[torch.Tensor] = []
    normalized_sampled: list[torch.Tensor] = []
    padding_masks: list[torch.Tensor] = []
    denormalized_target: list[np.ndarray] = []
    denormalized_gt: list[np.ndarray] = []
    denormalized_sampled: list[np.ndarray] = []
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
                sampled_future = model.infer_future_jepa(
                    z0,
                    context,
                    context_mask,
                    num_inference_steps=args.num_visual_inference_steps,
                    seed=args.seed,
                )
                sampled_z1 = sampled_future["z1"]
                sampled_z2 = sampled_future["z2"]
                for name, value in (
                    ("z0", z0),
                    ("gt_z1", gt_z1),
                    ("gt_z2", gt_z2),
                    ("sampled_z1", sampled_z1),
                    ("sampled_z2", sampled_z2),
                ):
                    _validate_latent(name, value, args.batch_size, TOKENS_PER_TEMPORAL_GROUP)

                gt_visual = torch.cat((z0, gt_z1, gt_z2), dim=1)
                sampled_visual = torch.cat((z0, sampled_z1, sampled_z2), dim=1)
                if tuple(gt_visual.shape) != (args.batch_size, VISUAL_TOKEN_COUNT, VJEPA_DIM):
                    raise ValueError(f"GT visual condition has invalid shape {tuple(gt_visual.shape)}.")
                if tuple(sampled_visual.shape) != tuple(gt_visual.shape):
                    raise ValueError("GT and sampled visual conditions must have identical shapes.")

                initial_action_noise = model._seeded_noise(
                    (args.batch_size, ACTION_HORIZON, len(ACTION_DIMENSION_NAMES)),
                    seed=args.seed,
                    device=device,
                    dtype=parameter_dtype,
                )
                noise_reference = initial_action_noise.clone()
                gt_action_sample = model.infer_action(
                    visual_tokens=gt_visual,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=args.num_action_inference_steps,
                    seed=args.seed,
                    initial_action_noise=initial_action_noise,
                )
                if not torch.equal(initial_action_noise, noise_reference):
                    raise RuntimeError("GT-future action inference mutated initial_action_noise.")
                sampled_action_sample = model.infer_action(
                    visual_tokens=sampled_visual,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=args.num_action_inference_steps,
                    seed=args.seed,
                    initial_action_noise=initial_action_noise,
                )
                if not torch.equal(initial_action_noise, noise_reference):
                    raise RuntimeError("Sampled-future action inference mutated initial_action_noise.")
                _validate_actions(gt_action_sample, batch["action"], batch["action_is_pad"])
                _validate_actions(sampled_action_sample, batch["action"], batch["action_is_pad"])

                action_timestep = model.action_train_scheduler.sample_training_t(
                    batch_size=args.batch_size,
                    device=device,
                    dtype=batch["action"].dtype,
                )
                action_noise = torch.randn_like(batch["action"])
                noisy_action = model.action_train_scheduler.add_noise(
                    batch["action"], action_noise, action_timestep
                )
                action_target = model.action_train_scheduler.training_target(
                    batch["action"], action_noise, action_timestep
                )
                one_step_gt, _ = _evaluate_action_condition(
                    model=model,
                    visual_condition=gt_visual,
                    context=context,
                    context_mask=context_mask,
                    noisy_action=noisy_action,
                    action_timestep=action_timestep,
                    action_target=action_target,
                    action_is_pad=batch["action_is_pad"],
                )
                one_step_sampled, _ = _evaluate_action_condition(
                    model=model,
                    visual_condition=sampled_visual,
                    context=context,
                    context_mask=context_mask,
                    noisy_action=noisy_action,
                    action_timestep=action_timestep,
                    action_target=action_target,
                    action_is_pad=batch["action_is_pad"],
                )

            gt_metrics = _masked_action_metrics(
                gt_action_sample, batch["action"], batch["action_is_pad"]
            )
            sampled_metrics = _masked_action_metrics(
                sampled_action_sample, batch["action"], batch["action_is_pad"]
            )
            gt_first4 = _masked_action_metrics(
                gt_action_sample,
                batch["action"],
                batch["action_is_pad"],
                horizon=EXEC_HORIZON,
            )
            sampled_first4 = _masked_action_metrics(
                sampled_action_sample,
                batch["action"],
                batch["action_is_pad"],
                horizon=EXEC_HORIZON,
            )
            if gt_metrics["mse"] <= 0 or gt_metrics["mae"] <= 0 or gt_first4["mae"] <= 0:
                raise ValueError("GT-future full-sampling errors must be positive.")
            one_step_gt_value = _finite_float("one_step_gt_loss", one_step_gt)
            one_step_sampled_value = _finite_float("one_step_sampled_loss", one_step_sampled)
            if one_step_gt_value <= 0:
                raise ValueError("GT-future one-step flow loss must be positive.")

            target_denormalized = _denormalize_action(batch["action"], processor)
            gt_denormalized = _denormalize_action(gt_action_sample, processor)
            sampled_denormalized = _denormalize_action(sampled_action_sample, processor)
            batch_record = {
                "batch_index": batch_index,
                "batch_size": args.batch_size,
                "initial_action_noise_max_abs_diff": _finite_float(
                    "initial_action_noise_max_abs_diff",
                    (initial_action_noise - noise_reference).abs().max(),
                ),
                "future_gap": {
                    "z1": _latent_gap_metrics(sampled_z1, gt_z1),
                    "z2": _latent_gap_metrics(sampled_z2, gt_z2),
                    "combined": _latent_gap_metrics(
                        torch.cat((sampled_z1, sampled_z2), dim=1),
                        torch.cat((gt_z1, gt_z2), dim=1),
                    ),
                },
                "gt_future_full_sampling": gt_metrics,
                "sampled_future_full_sampling": sampled_metrics,
                "full_sampling_mse_ratio": _finite_float(
                    "full_sampling_mse_ratio", sampled_metrics["mse"] / gt_metrics["mse"]
                ),
                "full_sampling_mae_ratio": _finite_float(
                    "full_sampling_mae_ratio", sampled_metrics["mae"] / gt_metrics["mae"]
                ),
                "gt_vs_sampled_action": _action_pair_metrics(
                    gt_action_sample, sampled_action_sample, batch["action_is_pad"]
                ),
                "first4": {
                    "gt_future_mae": gt_first4["mae"],
                    "sampled_future_mae": sampled_first4["mae"],
                    "mae_ratio": _finite_float(
                        "first4_mae_ratio", sampled_first4["mae"] / gt_first4["mae"]
                    ),
                    "gt_future_per_action_dimension_mae": gt_first4[
                        "per_action_dimension_mae"
                    ],
                    "sampled_future_per_action_dimension_mae": sampled_first4[
                        "per_action_dimension_mae"
                    ],
                },
                "one_step_flow": {
                    "gt_future_loss": one_step_gt_value,
                    "sampled_future_loss": one_step_sampled_value,
                    "loss_ratio": _finite_float(
                        "one_step_loss_ratio", one_step_sampled_value / one_step_gt_value
                    ),
                },
                "denormalized_distribution": {
                    "target": _per_dimension_distribution(
                        target_denormalized, batch["action_is_pad"]
                    ),
                    "gt_future_prediction": _per_dimension_distribution(
                        gt_denormalized, batch["action_is_pad"]
                    ),
                    "sampled_future_prediction": _per_dimension_distribution(
                        sampled_denormalized, batch["action_is_pad"]
                    ),
                },
            }
            batch_results.append(batch_record)
            _print_batch(batch_record)

            normalized_target.append(batch["action"].detach().float().cpu())
            normalized_gt.append(gt_action_sample.detach().float().cpu())
            normalized_sampled.append(sampled_action_sample.detach().float().cpu())
            padding_masks.append(batch["action_is_pad"].detach().cpu())
            denormalized_target.append(target_denormalized)
            denormalized_gt.append(gt_denormalized)
            denormalized_sampled.append(sampled_denormalized)

    all_target = torch.cat(normalized_target, dim=0)
    all_gt = torch.cat(normalized_gt, dim=0)
    all_sampled = torch.cat(normalized_sampled, dim=0)
    all_padding = torch.cat(padding_masks, dim=0)
    all_target_denormalized = np.concatenate(denormalized_target, axis=0)
    all_gt_denormalized = np.concatenate(denormalized_gt, axis=0)
    all_sampled_denormalized = np.concatenate(denormalized_sampled, axis=0)

    overall_gt = _masked_action_metrics(all_gt, all_target, all_padding)
    overall_sampled = _masked_action_metrics(all_sampled, all_target, all_padding)
    overall_gt_first4 = _masked_action_metrics(
        all_gt, all_target, all_padding, horizon=EXEC_HORIZON
    )
    overall_sampled_first4 = _masked_action_metrics(
        all_sampled, all_target, all_padding, horizon=EXEC_HORIZON
    )
    zero_first4 = _masked_action_metrics(
        torch.zeros_like(all_target), all_target, all_padding, horizon=EXEC_HORIZON
    )
    diagnosis = _diagnostic_hint(
        gt_first4_mae=overall_gt_first4["mae"],
        sampled_first4_mae=overall_sampled_first4["mae"],
        zero_first4_mae=zero_first4["mae"],
    )
    overall = {
        "num_batches": len(batch_results),
        "num_samples": int(all_target.shape[0]),
        "gt_future_full_sampling": overall_gt,
        "sampled_future_full_sampling": overall_sampled,
        "full_sampling_mse_ratio": _finite_float(
            "overall_mse_ratio", overall_sampled["mse"] / overall_gt["mse"]
        ),
        "full_sampling_mae_ratio": _finite_float(
            "overall_mae_ratio", overall_sampled["mae"] / overall_gt["mae"]
        ),
        "gt_vs_sampled_action": _action_pair_metrics(all_gt, all_sampled, all_padding),
        "first4": {
            "gt_future_mae": overall_gt_first4["mae"],
            "sampled_future_mae": overall_sampled_first4["mae"],
            "mae_ratio": _finite_float(
                "overall_first4_mae_ratio",
                overall_sampled_first4["mae"] / overall_gt_first4["mae"],
            ),
            "gt_future_per_action_dimension_mae": overall_gt_first4[
                "per_action_dimension_mae"
            ],
            "sampled_future_per_action_dimension_mae": overall_sampled_first4[
                "per_action_dimension_mae"
            ],
            "zero_normalized_action_mae": zero_first4["mae"],
        },
        "one_step_flow": {
            key: _mean_std(
                [record["one_step_flow"][key] for record in batch_results], name=key
            )
            for key in ("gt_future_loss", "sampled_future_loss", "loss_ratio")
        },
        "future_gap": {
            metric: _mean_std(
                [record["future_gap"]["combined"][metric] for record in batch_results],
                name=f"future_gap_{metric}",
            )
            for metric in ("mse", "l1", "cosine", "predicted_norm", "gt_norm", "norm_ratio")
        },
        "per_batch_metric_mean_std": {
            metric_name: _mean_std(
                [extractor(record) for record in batch_results], name=metric_name
            )
            for metric_name, extractor in {
                "gt_future_mse": lambda record: record["gt_future_full_sampling"]["mse"],
                "gt_future_mae": lambda record: record["gt_future_full_sampling"]["mae"],
                "sampled_future_mse": lambda record: record[
                    "sampled_future_full_sampling"
                ]["mse"],
                "sampled_future_mae": lambda record: record[
                    "sampled_future_full_sampling"
                ]["mae"],
                "full_sampling_mse_ratio": lambda record: record[
                    "full_sampling_mse_ratio"
                ],
                "full_sampling_mae_ratio": lambda record: record[
                    "full_sampling_mae_ratio"
                ],
                "first4_gt_mae": lambda record: record["first4"]["gt_future_mae"],
                "first4_sampled_mae": lambda record: record["first4"][
                    "sampled_future_mae"
                ],
                "first4_mae_ratio": lambda record: record["first4"]["mae_ratio"],
                "gt_vs_sampled_mse": lambda record: record["gt_vs_sampled_action"]["mse"],
                "gt_vs_sampled_mae": lambda record: record["gt_vs_sampled_action"]["mae"],
                "gt_vs_sampled_cosine": lambda record: record["gt_vs_sampled_action"][
                    "cosine"
                ],
            }.items()
        },
        "denormalized_distribution": {
            "target": _per_dimension_distribution(all_target_denormalized, all_padding),
            "gt_future_prediction": _per_dimension_distribution(all_gt_denormalized, all_padding),
            "sampled_future_prediction": _per_dimension_distribution(
                all_sampled_denormalized, all_padding
            ),
        },
        "diagnosis": diagnosis,
    }
    payload = {
        "diagnostic": "fastwam_jepa_idm_v5_action_sampling",
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": 5000,
        "precision": args.precision,
        "device": str(device),
        "seed": args.seed,
        "num_visual_inference_steps": args.num_visual_inference_steps,
        "num_action_inference_steps": args.num_action_inference_steps,
        "production_visual_config": dict(PRODUCTION_VISUAL_CONFIG),
        "action_dimension_names": list(ACTION_DIMENSION_NAMES),
        "metrics_use_action_padding_mask": True,
        "denormalization": "production processor normalizer.backward; no clip or gripper transform",
        "batches": batch_results,
        "overall": overall,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    json_file(output_path, payload)

    print(
        "overall "
        f"gt_mse={overall_gt['mse']:.6e} gt_mae={overall_gt['mae']:.6e} "
        f"sampled_mse={overall_sampled['mse']:.6e} "
        f"sampled_mae={overall_sampled['mae']:.6e} "
        f"mse_ratio={overall['full_sampling_mse_ratio']:.6f} "
        f"mae_ratio={overall['full_sampling_mae_ratio']:.6f} "
        f"first4_gt_mae={overall_gt_first4['mae']:.6e} "
        f"first4_sampled_mae={overall_sampled_first4['mae']:.6e} "
        f"diagnosis={diagnosis['category']} output_json={output_path}",
        flush=True,
    )
    print(f"diagnostic_hint={diagnosis['conclusion']}", flush=True)
    for name in ACTION_DIMENSION_NAMES:
        target_stats = overall["denormalized_distribution"]["target"][name]
        gt_stats = overall["denormalized_distribution"]["gt_future_prediction"][name]
        sampled_stats = overall["denormalized_distribution"]["sampled_future_prediction"][name]
        print(
            f"denormalized_dim={name} "
            f"target_mean={target_stats['mean']:.6e} target_std={target_stats['std']:.6e} "
            f"target_min={target_stats['min']:.6e} target_max={target_stats['max']:.6e} "
            f"gt_mean={gt_stats['mean']:.6e} gt_std={gt_stats['std']:.6e} "
            f"gt_min={gt_stats['min']:.6e} gt_max={gt_stats['max']:.6e} "
            f"sampled_mean={sampled_stats['mean']:.6e} sampled_std={sampled_stats['std']:.6e} "
            f"sampled_min={sampled_stats['min']:.6e} sampled_max={sampled_stats['max']:.6e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
