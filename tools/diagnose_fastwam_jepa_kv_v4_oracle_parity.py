#!/usr/bin/env python3
"""Diagnose release FastWAM, oracle-KV, and student-KV action parity."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.models.wan22.fastwam_jepa_kv_v4 import (  # noqa: E402
    CONTEXT_MASK_MODE_BASELINE,
    FastWAMJEPAKVV4,
    encode_causal_dual_camera_tokens,
    validate_checkpoint_context_mask_mode,
)
from train_fastwam_jepa_kv_v4_stage1_distill import (  # noqa: E402
    build_teacher,
    build_vjepa_encoder,
    camera_order_from_cfg,
    compose_cfg,
    precision_dtypes,
    require_dir,
    require_file,
    sha256_file,
    teacher_current_frame_cache,
)
from evaluate_fastwam_jepa_kv_v4_libero_rollout import (  # noqa: E402
    build_generator,
    configure_mujoco,
    load_payload,
    load_raw_text_context,
    postprocess_action,
    resolve_device,
    resolve_path,
)

RELEASE_CHECKPOINT = (
    "/ML-vePFS/protected/jinlei/challenge/dd_old_20260730/FastWAM/"
    "checkpoints/fastwam_release/libero_uncond_2cam224.pt"
)
RELEASE_STATS = (
    "/ML-vePFS/protected/jinlei/challenge/dd_old_20260730/FastWAM/"
    "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
)
ACTION_DIM_NAMES = ("dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare release FastWAM with oracle and student v4 KV denoising."
    )
    parser.add_argument("--v4-checkpoint", type=str, required=True)
    parser.add_argument("--vjepa-repo", type=str, required=True)
    parser.add_argument("--vjepa-checkpoint", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--release-checkpoint", type=str, default=RELEASE_CHECKPOINT)
    parser.add_argument("--dataset-stats-path", type=str, default=RELEASE_STATS)
    parser.add_argument("--config-name", type=str, default="sim_libero")
    parser.add_argument("--task", type=str, default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--libero-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--text-embedding-cache-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--wait-steps", type=int, default=30)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--rollout-oracle", action="store_true")
    parser.add_argument("--allow-checkpoint-mismatch", action="store_true")
    parser.add_argument("--vjepa-checkpoint-key", type=str, default=None)
    parser.add_argument("--vjepa-model-name", type=str, default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", choices=("-1_1", "0_1"), default="-1_1")
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--vjepa-autocast-dtype", type=str, default=None)
    parser.add_argument("--vjepa-strict-load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mujoco-gl", choices=("egl", "osmesa", "glfw"), default="egl")
    parser.add_argument("--pyopengl-platform", default="egl")
    parser.add_argument("--egl-device-id", type=int, default=0)
    parser.add_argument("--egl-import-device-id", default=None)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("This parity protocol requires --seed 42")
    if args.action_horizon != 32:
        raise ValueError("This parity protocol requires --action-horizon 32")
    if args.num_inference_steps != 10:
        raise ValueError("This parity protocol requires --num-inference-steps 10")
    if args.replan_steps != 10:
        raise ValueError("This parity protocol requires --replan-steps 10")
    if args.wait_steps != 30:
        raise ValueError("This parity protocol requires --wait-steps 30")
    return args


def _as_batched_context(
    context: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    context = context.to(device=device, dtype=dtype)
    context_mask = context_mask.to(device=device, dtype=torch.bool)
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    if context.ndim != 3 or context_mask.shape != context.shape[:2]:
        raise ValueError(
            f"Invalid context shapes: context={tuple(context.shape)}, "
            f"mask={tuple(context_mask.shape)}"
        )
    if not bool(context_mask.all().item()):
        raise ValueError("Parity requires baseline_all_true context-mask policy")
    return context, context_mask


def _append_proprio(
    teacher: torch.nn.Module,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    action_context, action_mask = teacher._append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=proprio,
    )
    return action_context, action_mask.to(dtype=torch.bool)


def _make_initial_noise(seed: int, action_horizon: int, action_dim: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(
        (1, action_horizon, action_dim),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )


def _requested_randn_shape(args: Sequence[Any]) -> Optional[Tuple[int, ...]]:
    if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
        return tuple(int(value) for value in args[0])
    try:
        return tuple(int(value) for value in args)
    except (TypeError, ValueError):
        return None


@contextmanager
def _force_initial_action_noise(initial_noise: torch.Tensor) -> Iterable[Dict[str, bool]]:
    """Inject one explicit action-noise tensor into the unmodified baseline API."""
    original_randn = torch.randn
    state = {"used": False}
    expected_shape = tuple(initial_noise.shape)

    def forced_randn(*size: Any, **kwargs: Any) -> torch.Tensor:
        if not state["used"] and _requested_randn_shape(size) == expected_shape:
            state["used"] = True
            device = kwargs.get("device", initial_noise.device)
            dtype = kwargs.get("dtype", initial_noise.dtype)
            return initial_noise.to(device=device, dtype=dtype).clone()
        return original_randn(*size, **kwargs)

    with mock.patch.object(torch, "randn", new=forced_randn):
        yield state
    if not state["used"]:
        raise RuntimeError("Baseline infer_action did not request the expected initial noise")


def _cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32)


def _scalar_timestep(timestep: torch.Tensor | float) -> float:
    return float(torch.as_tensor(timestep).detach().to(dtype=torch.float32).flatten()[0].item())


def _run_baseline_with_trace(
    teacher: torch.nn.Module,
    *,
    image: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor,
    initial_noise: torch.Tensor,
    seed: int,
    action_horizon: int,
    num_inference_steps: int,
    sigma_shift: Optional[float],
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Sequence[Mapping[str, torch.Tensor]]]:
    records: List[Dict[str, Any]] = []
    baseline_cache: Optional[Sequence[Mapping[str, torch.Tensor]]] = None
    original_predict = teacher._predict_action_noise_with_cache
    scheduler = teacher.infer_action_scheduler
    original_step = scheduler.step

    def traced_predict(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal baseline_cache
        prediction = original_predict(*args, **kwargs)
        noisy_action = kwargs.get("latents_action")
        timestep = kwargs.get("timestep_action")
        cache = kwargs.get("video_kv_cache")
        if noisy_action is None or timestep is None or cache is None:
            raise RuntimeError("Unexpected release FastWAM action predictor signature")
        if baseline_cache is None:
            baseline_cache = cache
        records.append(
            {
                "timestep": _scalar_timestep(timestep),
                "latent_before": _cpu_float(noisy_action),
                "pred": _cpu_float(prediction),
            }
        )
        return prediction

    def traced_step(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = original_step(*args, **kwargs)
        if not records or "latent_after" in records[-1]:
            raise RuntimeError("Unexpected release FastWAM scheduler call order")
        records[-1]["latent_after"] = _cpu_float(result)
        return result

    with _force_initial_action_noise(initial_noise) as noise_state:
        with mock.patch.object(teacher, "_predict_action_noise_with_cache", new=traced_predict):
            with mock.patch.object(scheduler, "step", new=traced_step):
                final_action = teacher.infer_action(
                    prompt=None,
                    input_image=image,
                    context=context,
                    context_mask=context_mask,
                    proprio=proprio,
                    action_horizon=action_horizon,
                    num_inference_steps=num_inference_steps,
                    sigma_shift=sigma_shift,
                    seed=seed,
                    rand_device="cpu",
                )
        if not noise_state["used"]:
            raise RuntimeError("Explicit initial action noise was not used")

    if baseline_cache is None:
        raise RuntimeError("Baseline infer_action did not expose a video KV cache")
    if len(records) != num_inference_steps:
        raise ValueError(f"Expected {num_inference_steps} baseline steps, got {len(records)}")
    if any("latent_after" not in record for record in records):
        raise RuntimeError("A baseline denoise step is missing its updated action latent")
    return _cpu_float(final_action["action"]), records, baseline_cache


def _run_cache_denoise(
    action_runner: FastWAMJEPAKVV4,
    *,
    video_kv_cache: Sequence[Mapping[str, torch.Tensor]],
    context: torch.Tensor,
    context_mask: torch.Tensor,
    initial_noise: torch.Tensor,
    timesteps: Sequence[torch.Tensor],
    deltas: Sequence[torch.Tensor],
    scheduler: Any,
    device: torch.device,
    dtype: torch.dtype,
    record_steps: bool = True,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    latent = initial_noise.to(device=device, dtype=dtype).clone()
    records: List[Dict[str, Any]] = []
    action_runner.reset_debug_counters()
    for timestep, delta in zip(timesteps, deltas):
        timestep = torch.as_tensor(timestep, device=device, dtype=dtype).unsqueeze(0)
        delta = torch.as_tensor(delta, device=device, dtype=dtype)
        prediction = action_runner.predict_action_noise_with_cache(
            noisy_action=latent,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
        )
        updated = scheduler.step(prediction, delta, latent)
        if record_steps:
            records.append(
                {
                    "timestep": _scalar_timestep(timestep),
                    "latent_before": _cpu_float(latent),
                    "pred": _cpu_float(prediction),
                    "latent_after": _cpu_float(updated),
                }
            )
        latent = updated
    return _cpu_float(latent), records


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(dtype=torch.float32).reshape(1, -1)
    right = right.to(dtype=torch.float32).reshape(1, -1)
    return float(F.cosine_similarity(left, right, dim=1, eps=1e-12).item())


def _pair_metrics(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    left = left.to(dtype=torch.float32)
    right = right.to(dtype=torch.float32)
    difference = left - right
    return {
        "mse": float(difference.square().mean().item()),
        "cosine": _cosine(left, right),
        "max_abs_diff": float(difference.abs().max().item()),
    }


def _tensor_summary(tensor: torch.Tensor) -> Dict[str, float]:
    tensor = tensor.to(dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "norm": float(tensor.norm().item()),
    }


def _per_dimension_mse(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    difference = left.to(dtype=torch.float32) - right.to(dtype=torch.float32)
    values = difference.square().reshape(-1, difference.shape[-1]).mean(dim=0)
    if values.numel() != len(ACTION_DIM_NAMES):
        raise ValueError(f"Expected {len(ACTION_DIM_NAMES)} action dimensions, got {values.numel()}")
    return {name: float(value.item()) for name, value in zip(ACTION_DIM_NAMES, values)}


def _cache_layer_metrics(
    left_cache: Sequence[Mapping[str, torch.Tensor]],
    right_cache: Sequence[Mapping[str, torch.Tensor]],
) -> List[Dict[str, Any]]:
    if len(left_cache) != len(right_cache):
        raise ValueError(f"Cache layer mismatch: {len(left_cache)} vs {len(right_cache)}")
    metrics: List[Dict[str, Any]] = []
    for layer_index, (left, right) in enumerate(zip(left_cache, right_cache)):
        left_k, left_v = _cpu_float(left["k"]), _cpu_float(left["v"])
        right_k, right_v = _cpu_float(right["k"]), _cpu_float(right["v"])
        if left_k.shape != right_k.shape or left_v.shape != right_v.shape:
            raise ValueError(
                f"Cache shape mismatch at layer {layer_index}: "
                f"K {tuple(left_k.shape)} vs {tuple(right_k.shape)}, "
                f"V {tuple(left_v.shape)} vs {tuple(right_v.shape)}"
            )
        k_metrics = _pair_metrics(left_k, right_k)
        v_metrics = _pair_metrics(left_v, right_v)
        metrics.append(
            {
                "layer": layer_index,
                "k_shape": list(left_k.shape),
                "v_shape": list(left_v.shape),
                "k_mse": k_metrics["mse"],
                "k_cosine": k_metrics["cosine"],
                "v_mse": v_metrics["mse"],
                "v_cosine": v_metrics["cosine"],
            }
        )
    return metrics


def _summarize_layer_metrics(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        raise ValueError("Cannot summarize an empty cache")

    def compact(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: row[key]
            for key in ("layer", "k_mse", "k_cosine", "v_mse", "v_cosine")
        }

    return {
        "first": compact(metrics[0]),
        "middle": compact(metrics[len(metrics) // 2]),
        "last": compact(metrics[-1]),
        "mean": {
            key: float(np.mean([float(row[key]) for row in metrics]))
            for key in ("k_mse", "k_cosine", "v_mse", "v_cosine")
        },
    }


def _build_per_step_metrics(
    baseline: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    student: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not (len(baseline) == len(oracle) == len(student)):
        raise ValueError(
            f"Denoise trace lengths differ: {len(baseline)}, {len(oracle)}, {len(student)}"
        )
    output: List[Dict[str, Any]] = []
    for step, (base_row, oracle_row, student_row) in enumerate(
        zip(baseline, oracle, student)
    ):
        timesteps = (
            float(base_row["timestep"]),
            float(oracle_row["timestep"]),
            float(student_row["timestep"]),
        )
        if max(timesteps) - min(timesteps) > 1e-6:
            raise ValueError(f"Timestep mismatch at denoise step {step}: {timesteps}")
        base_oracle_pred = _pair_metrics(base_row["pred"], oracle_row["pred"])
        base_oracle_latent = _pair_metrics(
            base_row["latent_after"], oracle_row["latent_after"]
        )
        oracle_student_pred = _pair_metrics(oracle_row["pred"], student_row["pred"])
        oracle_student_latent = _pair_metrics(
            oracle_row["latent_after"], student_row["latent_after"]
        )
        output.append(
            {
                "step": step,
                "timestep": timesteps[0],
                "baseline_pred": _tensor_summary(base_row["pred"]),
                "oracle_pred": _tensor_summary(oracle_row["pred"]),
                "student_pred": _tensor_summary(student_row["pred"]),
                "baseline_oracle_pred_mse": base_oracle_pred["mse"],
                "baseline_oracle_pred_cosine": base_oracle_pred["cosine"],
                "baseline_oracle_latent_action_mse": base_oracle_latent["mse"],
                "baseline_oracle_latent_action_max_abs_diff": base_oracle_latent[
                    "max_abs_diff"
                ],
                "oracle_student_pred_mse": oracle_student_pred["mse"],
                "oracle_student_pred_cosine": oracle_student_pred["cosine"],
                "oracle_student_latent_action_mse": oracle_student_latent["mse"],
                "oracle_student_latent_action_max_abs_diff": oracle_student_latent[
                    "max_abs_diff"
                ],
            }
        )
    return output


def _load_processor(cfg: Any, stats_path: Path) -> Any:
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats_path))
    return processor


def _capture_observation(args: argparse.Namespace, cfg: Any) -> Tuple[Any, str, Any]:
    from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_env

    env, task_description = get_libero_env(
        task_id=args.task_id,
        suite_name=args.libero_suite,
        resolution=cfg.data.train.processor.resolution,
        seed=args.seed,
    )
    initial_states = env.base_env._get_initial_states()
    if not 0 <= args.init_state_index < len(initial_states):
        env.close()
        raise IndexError(
            f"init-state-index {args.init_state_index} outside [0, {len(initial_states)})"
        )
    env.reset()
    observation = env.set_init_state(initial_states[args.init_state_index])
    for _ in range(args.wait_steps):
        observation, _, done, _ = env.step(get_libero_dummy_action())
        if done:
            env.close()
            raise RuntimeError("LIBERO episode ended during the wait protocol")
    return env, task_description, observation


def _validate_v4_payload(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
    *,
    release_checkpoint: Path,
    stats_path: Path,
    vjepa_checkpoint: Path,
    camera_order: Sequence[str],
) -> None:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("v4 checkpoint metadata must be a mapping")
    validate_checkpoint_context_mask_mode(
        metadata,
        CONTEXT_MASK_MODE_BASELINE,
        checkpoint_name="v4 checkpoint",
    )
    if metadata.get("input_policy") != "single_current_frame_duplicated_to_2":
        raise ValueError("v4 checkpoint has an incompatible input_policy")
    if metadata.get("selected_frame_index") != 0:
        raise ValueError("v4 checkpoint must select current frame index 0")
    expected_camera_order = tuple(str(name) for name in camera_order)
    recorded_camera_order = tuple(metadata.get("camera_order", ()))
    if recorded_camera_order != expected_camera_order:
        raise ValueError(
            f"Expected camera order {expected_camera_order}, got {recorded_camera_order}"
        )
    hashes = {
        "release action checkpoint": (
            metadata.get("action_checkpoint_sha256")
            or metadata.get("teacher_fastwam_checkpoint_sha256"),
            sha256_file(release_checkpoint),
        ),
        "dataset stats": (metadata.get("dataset_stats_sha256"), sha256_file(stats_path)),
        "V-JEPA checkpoint": (
            metadata.get("vjepa_checkpoint_sha256"),
            sha256_file(vjepa_checkpoint),
        ),
    }
    mismatches = [
        f"{name}: checkpoint={recorded!r}, runtime={actual!r}"
        for name, (recorded, actual) in hashes.items()
        if recorded != actual
    ]
    if mismatches and not args.allow_checkpoint_mismatch:
        raise ValueError("Checkpoint provenance mismatch: " + "; ".join(mismatches))
    for mismatch in mismatches:
        print(f"WARNING {mismatch}")


def _build_action_runner(
    args: argparse.Namespace,
    cfg: Any,
    teacher: torch.nn.Module,
    generator: torch.nn.Module,
    vjepa_encoder: torch.nn.Module,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> FastWAMJEPAKVV4:
    runner = FastWAMJEPAKVV4(
        action_expert=teacher.action_expert,
        vjepa_encoder=vjepa_encoder,
        kv_generator=generator,
        camera_order=camera_order_from_cfg(cfg),
        proprio_dim=None,
        action_horizon=args.action_horizon,
        action_train_shift=float(cfg.model.action_scheduler.train_shift),
        action_infer_shift=float(cfg.model.action_scheduler.infer_shift),
        action_num_train_timesteps=int(cfg.model.action_scheduler.num_train_timesteps),
        freeze_vjepa=True,
        freeze_action=True,
        freeze_proprio=True,
        context_mask_mode=CONTEXT_MASK_MODE_BASELINE,
        device=device,
        torch_dtype=dtype,
    )
    return runner.eval()


@contextmanager
def _forbid_module_forward(*modules: torch.nn.Module) -> Iterable[None]:
    def forbidden(_module: torch.nn.Module, _inputs: Tuple[Any, ...]) -> None:
        raise RuntimeError("Oracle rollout attempted to call V-JEPA or the KV generator")

    handles = [module.register_forward_pre_hook(forbidden) for module in modules]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def _run_oracle_rollout(
    args: argparse.Namespace,
    cfg: Any,
    *,
    teacher: torch.nn.Module,
    action_runner: FastWAMJEPAKVV4,
    processor: Any,
    timesteps: Sequence[torch.Tensor],
    deltas: Sequence[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_env
    from tools.evaluate_fastwam_jepa_idm_v2_predictor_value import (
        _obs_to_model_input,
        get_max_steps,
    )

    env, task_description = get_libero_env(
        task_id=0,
        suite_name="libero_spatial",
        resolution=cfg.data.train.processor.resolution,
        seed=42,
    )
    initial_states = env.base_env._get_initial_states()
    if len(initial_states) < 5:
        env.close()
        raise ValueError(f"Oracle rollout requires 5 initial states, found {len(initial_states)}")
    rollout_context, rollout_mask = load_raw_text_context(
        task_description, cfg, override_dir=args.text_embedding_cache_dir
    )
    rollout_context, rollout_mask = _as_batched_context(
        rollout_context, rollout_mask, device=device, dtype=dtype
    )
    max_steps = get_max_steps("libero_spatial", None)
    initial_noise = _make_initial_noise(42, args.action_horizon, teacher.action_expert.action_dim)
    episodes: List[Dict[str, Any]] = []
    try:
        for episode_index in range(5):
            env.reset()
            observation = env.set_init_state(initial_states[episode_index])
            for _ in range(30):
                observation, _, done, _ = env.step(get_libero_dummy_action())
                if done:
                    raise RuntimeError(
                        f"Oracle rollout episode {episode_index} ended during wait"
                    )
            success = False
            env_steps = 0
            replans = 0
            while env_steps < max_steps and not success:
                video_size = cfg.data.train.video_size
                image, proprio, _ = _obs_to_model_input(
                    observation,
                    cfg=cfg,
                    processor=processor,
                    width=int(video_size[1]),
                    height=int(video_size[0]),
                    device=device,
                    dtype=dtype,
                )
                action_context, action_mask = _append_proprio(
                    teacher, rollout_context, rollout_mask, proprio
                )
                exact_cache, _ = teacher_current_frame_cache(
                    teacher,
                    image.unsqueeze(2),
                    action_context,
                    action_mask,
                )
                normalized_action, _ = _run_cache_denoise(
                    action_runner,
                    video_kv_cache=exact_cache,
                    context=action_context,
                    context_mask=action_mask,
                    initial_noise=initial_noise,
                    timesteps=timesteps,
                    deltas=deltas,
                    scheduler=teacher.infer_action_scheduler,
                    device=device,
                    dtype=dtype,
                    record_steps=False,
                )
                action_chunk = postprocess_action(
                    normalized_action.squeeze(0), processor, binarize_gripper=True
                )
                replans += 1
                for action in action_chunk[:10]:
                    observation, _, done, _ = env.step(action)
                    env_steps += 1
                    if done:
                        success = True
                        break
                    if env_steps >= max_steps:
                        break
            episodes.append(
                {
                    "episode": episode_index,
                    "success": bool(success),
                    "env_steps": env_steps,
                    "replans": replans,
                }
            )
            print(
                f"oracle_rollout episode={episode_index} success={int(success)} "
                f"env_steps={env_steps} replans={replans}"
            )
    finally:
        env.close()
    successes = sum(int(row["success"]) for row in episodes)
    return {
        "success_rate": successes / len(episodes),
        "successes": successes,
        "episodes": episodes,
        "protocol": {
            "suite": "libero_spatial",
            "task_id": 0,
            "episodes": 5,
            "seed": 42,
            "num_inference_steps": 10,
            "action_horizon": 32,
            "replan_steps": 10,
            "wait_steps": 30,
            "binarize_gripper": True,
            "vjepa_calls": 0,
            "kv_generator_calls": 0,
            "task_description": task_description,
        },
    }


def _print_final_metrics(
    name: str,
    metrics: Mapping[str, float],
    per_dimension: Mapping[str, float],
) -> None:
    print(
        f"{name} mse={metrics['mse']:.8e} cosine={metrics['cosine']:.8f} "
        f"max_abs_diff={metrics['max_abs_diff']:.8e}"
    )
    print(
        name
        + " per_dim_mse "
        + " ".join(f"{key}={value:.8e}" for key, value in per_dimension.items())
    )


def main() -> None:
    args = parse_args()
    configure_mujoco(args)

    from omegaconf import OmegaConf
    from fastwam.utils.pytorch_utils import set_global_seed
    from tools.evaluate_fastwam_jepa_idm_v2_predictor_value import _obs_to_model_input

    set_global_seed(args.seed, get_worker_init_fn=False)
    device = resolve_device(args.device)
    dtype, _ = precision_dtypes(args.precision, device)
    release_checkpoint = require_file(
        str(resolve_path(args.release_checkpoint)), name="release checkpoint"
    )
    stats_path = require_file(str(resolve_path(args.dataset_stats_path)), name="dataset stats")
    v4_checkpoint = require_file(str(resolve_path(args.v4_checkpoint)), name="v4 checkpoint")
    vjepa_checkpoint = require_file(
        str(resolve_path(args.vjepa_checkpoint)), name="V-JEPA checkpoint"
    )
    vjepa_repo = require_dir(str(resolve_path(args.vjepa_repo)), name="V-JEPA repository")
    output_json = resolve_path(args.output_json)

    cfg = compose_cfg(args.config_name, args.task)
    OmegaConf.update(
        cfg,
        "EVALUATION.context_mask_mode",
        CONTEXT_MASK_MODE_BASELINE,
        merge=False,
    )
    payload = load_payload(v4_checkpoint)
    camera_order = camera_order_from_cfg(cfg)
    _validate_v4_payload(
        args,
        payload,
        release_checkpoint=release_checkpoint,
        stats_path=stats_path,
        vjepa_checkpoint=vjepa_checkpoint,
        camera_order=camera_order,
    )
    checkpoint_args = payload.get("args", {})
    if not isinstance(checkpoint_args, Mapping):
        checkpoint_args = {}
    for field in (
        "vjepa_model_name",
        "vjepa_img_size",
        "vjepa_input_range",
        "vjepa_tubelet_size",
        "vjepa_dim",
        "vjepa_checkpoint_key",
        "vjepa_autocast_dtype",
        "vjepa_strict_load",
    ):
        if field in checkpoint_args:
            setattr(args, field, checkpoint_args[field])
    args.vjepa_repo = str(vjepa_repo)
    args.vjepa_checkpoint = str(vjepa_checkpoint)

    teacher = build_teacher(
        cfg,
        checkpoint_path=release_checkpoint,
        device=device,
        dtype=dtype,
        rank=0,
    )
    teacher.eval()
    if type(teacher).__name__ != "FastWAM":
        raise TypeError(f"Release checkpoint must be loaded as FastWAM, got {type(teacher).__name__}")
    processor = _load_processor(cfg, stats_path)
    generator = build_generator(payload, device=device, dtype=dtype)
    vjepa_encoder, vjepa_load_report = build_vjepa_encoder(
        args, device=device, dtype=dtype, rank=0
    )
    action_runner = _build_action_runner(
        args,
        cfg,
        teacher,
        generator,
        vjepa_encoder,
        dtype=dtype,
        device=device,
    )
    if action_runner.action_expert is not teacher.action_expert:
        raise RuntimeError("A/B/C do not share the exact same release ActionDiT object")
    if "action_expert" in payload:
        print("checkpoint_action_expert=present_but_ignored_for_release_action_parity")

    env, task_description, observation = _capture_observation(args, cfg)
    try:
        video_size = cfg.data.train.video_size
        image, proprio, _ = _obs_to_model_input(
            observation,
            cfg=cfg,
            processor=processor,
            width=int(video_size[1]),
            height=int(video_size[0]),
            device=device,
            dtype=dtype,
        )
    finally:
        env.close()
    context, context_mask = load_raw_text_context(
        task_description,
        cfg,
        override_dir=args.text_embedding_cache_dir,
    )
    context, context_mask = _as_batched_context(
        context, context_mask, device=device, dtype=dtype
    )
    action_context, action_mask = _append_proprio(
        teacher, context, context_mask, proprio
    )

    timesteps, deltas = teacher.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=args.num_inference_steps,
        shift_override=args.sigma_shift,
        device=device,
        dtype=dtype,
    )
    timesteps = list(timesteps)
    deltas = list(deltas)
    initial_noise = _make_initial_noise(args.seed, args.action_horizon, teacher.action_expert.action_dim)

    with torch.inference_mode():
        exact_cache, teacher_grid = teacher_current_frame_cache(
            teacher,
            image.unsqueeze(2),
            action_context,
            action_mask,
        )
        baseline_final, baseline_trace, baseline_internal_cache = _run_baseline_with_trace(
            teacher,
            image=image,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            initial_noise=initial_noise,
            seed=args.seed,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
        )
        baseline_initial_diff = float(
            (baseline_trace[0]["latent_before"] - initial_noise).abs().max().item()
        )
        if baseline_initial_diff != 0.0:
            raise ValueError(
                f"Baseline did not use the explicit initial noise: max diff={baseline_initial_diff}"
            )
        oracle_final, oracle_trace = _run_cache_denoise(
            action_runner,
            video_kv_cache=exact_cache,
            context=action_context,
            context_mask=action_mask,
            initial_noise=initial_noise,
            timesteps=timesteps,
            deltas=deltas,
            scheduler=teacher.infer_action_scheduler,
            device=device,
            dtype=dtype,
        )
        visual_tokens = encode_causal_dual_camera_tokens(
            vjepa_encoder,
            image,
            camera_order=camera_order_from_cfg(cfg),
        )
        student_cache = generator(visual_tokens, action_context, action_mask)
        student_final, student_trace = _run_cache_denoise(
            action_runner,
            video_kv_cache=student_cache,
            context=action_context,
            context_mask=action_mask,
            initial_noise=initial_noise,
            timesteps=timesteps,
            deltas=deltas,
            scheduler=teacher.infer_action_scheduler,
            device=device,
            dtype=dtype,
        )

    baseline_oracle = _pair_metrics(baseline_final, oracle_final)
    oracle_student = _pair_metrics(oracle_final, student_final)
    baseline_oracle_per_dim = _per_dimension_mse(baseline_final, oracle_final)
    oracle_student_per_dim = _per_dimension_mse(oracle_final, student_final)
    per_step_metrics = _build_per_step_metrics(
        baseline_trace, oracle_trace, student_trace
    )
    teacher_student_kv = _cache_layer_metrics(exact_cache, student_cache)
    baseline_oracle_kv = _cache_layer_metrics(baseline_internal_cache, exact_cache)
    teacher_student_kv_summary = _summarize_layer_metrics(teacher_student_kv)
    baseline_oracle_kv_summary = _summarize_layer_metrics(baseline_oracle_kv)

    expected_grid = (1, 7, 14)
    if tuple(int(value) for value in teacher_grid) != expected_grid:
        raise ValueError(f"Expected teacher grid {expected_grid}, got {teacher_grid}")
    if len(exact_cache) != 30 or len(student_cache) != 30:
        raise ValueError(
            f"Expected 30 cache layers, got teacher={len(exact_cache)}, student={len(student_cache)}"
        )
    if action_runner.video_seq_len != 98:
        raise ValueError(f"Expected video_seq_len=98, got {action_runner.video_seq_len}")

    action_parameter_count = sum(
        parameter.numel() for parameter in teacher.action_expert.parameters()
    )
    print(f"teacher_type={type(teacher).__name__}")
    print(f"teacher_grid={tuple(teacher_grid)}")
    print(f"video_seq_len={action_runner.video_seq_len}")
    print(f"cache_layer_count={len(exact_cache)}")
    print(f"action_context_shape={tuple(action_context.shape)}")
    print(
        "action_dit_parameter_load_parity="
        f"shared_object_true parameter_count={action_parameter_count}"
    )
    print(
        "scheduler_timesteps="
        + json.dumps([_scalar_timestep(value) for value in timesteps])
    )
    print(f"initial_noise_max_abs_diff={baseline_initial_diff:.1f}")
    for row in teacher_student_kv:
        print(
            f"cache_layer={row['layer']:02d} K_shape={tuple(row['k_shape'])} "
            f"V_shape={tuple(row['v_shape'])} K_mse={row['k_mse']:.8e} "
            f"K_cos={row['k_cosine']:.8f} V_mse={row['v_mse']:.8e} "
            f"V_cos={row['v_cosine']:.8f}"
        )
    print("teacher_student_kv_summary=" + json.dumps(teacher_student_kv_summary))
    print("baseline_internal_oracle_kv_summary=" + json.dumps(baseline_oracle_kv_summary))
    for row in per_step_metrics:
        print(
            f"denoise_step={row['step']:02d} timestep={row['timestep']:.7g} "
            f"A_B_pred_mse={row['baseline_oracle_pred_mse']:.8e} "
            f"A_B_pred_cos={row['baseline_oracle_pred_cosine']:.8f} "
            f"A_B_latent_mse={row['baseline_oracle_latent_action_mse']:.8e} "
            f"A_B_latent_max={row['baseline_oracle_latent_action_max_abs_diff']:.8e} "
            f"B_C_pred_mse={row['oracle_student_pred_mse']:.8e} "
            f"B_C_pred_cos={row['oracle_student_pred_cosine']:.8f} "
            f"B_C_latent_mse={row['oracle_student_latent_action_mse']:.8e} "
            f"B_C_latent_max={row['oracle_student_latent_action_max_abs_diff']:.8e}"
        )
    _print_final_metrics(
        "baseline_vs_oracle", baseline_oracle, baseline_oracle_per_dim
    )
    _print_final_metrics("oracle_vs_student", oracle_student, oracle_student_per_dim)

    oracle_rollout: Optional[Dict[str, Any]] = None
    if args.rollout_oracle:
        with _forbid_module_forward(vjepa_encoder, generator):
            oracle_rollout = _run_oracle_rollout(
                args,
                cfg,
                teacher=teacher,
                action_runner=action_runner,
                processor=processor,
                timesteps=timesteps,
                deltas=deltas,
                device=device,
                dtype=dtype,
            )
        print(f"oracle_rollout_success_rate={oracle_rollout['success_rate']:.6f}")

    result = {
        "protocol": {
            "release_checkpoint": str(release_checkpoint),
            "dataset_stats_path": str(stats_path),
            "v4_checkpoint": str(v4_checkpoint),
            "vjepa_checkpoint": str(vjepa_checkpoint),
            "libero_suite": args.libero_suite,
            "task_id": args.task_id,
            "init_state_index": args.init_state_index,
            "task_description": task_description,
            "seed": args.seed,
            "action_horizon": args.action_horizon,
            "num_inference_steps": args.num_inference_steps,
            "sigma_shift": args.sigma_shift,
            "precision": args.precision,
            "context_mask_policy": CONTEXT_MASK_MODE_BASELINE,
            "same_explicit_initial_noise": True,
            "same_release_action_expert_object": True,
        },
        "structure": {
            "teacher_type": type(teacher).__name__,
            "teacher_grid": list(teacher_grid),
            "video_seq_len": action_runner.video_seq_len,
            "cache_layer_count": len(exact_cache),
            "cache_k_shape": list(exact_cache[0]["k"].shape),
            "cache_v_shape": list(exact_cache[0]["v"].shape),
            "action_context_shape": list(action_context.shape),
            "action_parameter_count": action_parameter_count,
            "vjepa_load_report": vjepa_load_report,
            "scheduler_timesteps": [_scalar_timestep(value) for value in timesteps],
            "initial_noise_max_abs_diff": baseline_initial_diff,
        },
        "baseline_oracle_final_mse": baseline_oracle["mse"],
        "baseline_oracle_final_cosine": baseline_oracle["cosine"],
        "baseline_oracle_final_max_abs_diff": baseline_oracle["max_abs_diff"],
        "baseline_oracle_per_dimension_mse": baseline_oracle_per_dim,
        "oracle_student_final_mse": oracle_student["mse"],
        "oracle_student_final_cosine": oracle_student["cosine"],
        "oracle_student_final_max_abs_diff": oracle_student["max_abs_diff"],
        "oracle_student_per_dimension_mse": oracle_student_per_dim,
        "per_step_metrics": per_step_metrics,
        "per_layer_kv_metrics": teacher_student_kv,
        "per_layer_kv_summary": teacher_student_kv_summary,
        "baseline_internal_oracle_kv_summary": baseline_oracle_kv_summary,
        "oracle_rollout_success_rate": (
            None if oracle_rollout is None else oracle_rollout["success_rate"]
        ),
        "oracle_rollout": oracle_rollout,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    print(f"output_json={output_json}")


if __name__ == "__main__":
    main()
