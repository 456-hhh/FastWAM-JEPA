from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
LIBERO_EXP_ROOT = PROJECT_ROOT / "experiments" / "libero"
for _path in (SRC_ROOT, PROJECT_ROOT, LIBERO_EXP_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import evaluate_fastwam_jepa_idm_v2_predictor_value as rollout_base
import train_fastwam_jepa_idm_v2_stage2_libero as stage2_train

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FUTURE_SOURCES = ("predicted", "no_future", "oracle")
SHAPE_CRITICAL_CONFIG_KEYS = {
    "current_frame_count",
    "future_frame_count",
    "resolved_num_future_tokens",
    "future_predictor_layers",
    "future_predictor_hidden_dim",
    "future_predictor_heads",
    "vjepa_img_size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU LIBERO rollout/eval for FastWAM-JEPA-IDM v2 Stage 2 "
            "checkpoints. V-JEPA2 is loaded separately and kept frozen."
        )
    )
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--vjepa-checkpoint", required=True)
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument(
        "--libero-suite", default="libero_spatial", choices=LIBERO_SUITES
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--future-source", default="predicted", choices=FUTURE_SOURCES)
    parser.add_argument(
        "--use-proprio", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--current-frame-count", type=int, default=4)
    parser.add_argument("--future-frame-count", type=int, default=4)
    parser.add_argument("--num-future-tokens", default="auto")
    parser.add_argument("--future-predictor-layers", type=int, default=24)
    parser.add_argument("--future-predictor-hidden-dim", type=int, default=1024)
    parser.add_argument("--future-predictor-heads", type=int, default=16)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--lambda-future", type=float, default=0.05)
    parser.add_argument("--lambda-cos", type=float, default=0.1)
    parser.add_argument("--exec-horizon", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--save-videos", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--task-id", default="all", help="LIBERO task id or 'all'.")
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument("--vjepa-repo", default=stage2_train.DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", default="-1_1", choices=["-1_1", "0_1"])
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--adapter-current-tokens", type=int, default=64)
    parser.add_argument("--adapter-future-tokens", type=int, default=64)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument(
        "--binarize-gripper", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--oracle-future-mode", default="snapshot_dummy", choices=["snapshot_dummy"]
    )
    parser.add_argument("--mujoco-gl", default="egl", choices=["egl", "osmesa", "glfw"])
    parser.add_argument("--pyopengl-platform", default="egl")
    parser.add_argument("--egl-device-id", type=int, default=0)
    parser.add_argument("--egl-import-device-id", default=None)
    return parser.parse_args()


def resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def require_file(path_value: str | Path | None, *, name: str) -> Path:
    path = resolve_path(path_value)
    if path is None or not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")
    return path


def require_dir(path_value: str | Path | None, *, name: str) -> Path:
    path = resolve_path(path_value)
    if path is None or not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist or is not a directory: {path}")
    return path


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_arg))


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            f"Stage 2 checkpoint payload must be dict, got {type(payload)}."
        )
    return payload


def merged_checkpoint_config(payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("args", "stage2_temporal_config", "config"):
        value = payload.get(key)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _warn_config_diff(key: str, ckpt_value: Any, cli_value: Any) -> None:
    print(
        f"WARNING checkpoint {key}={ckpt_value!r} differs from CLI {key}={cli_value!r}",
        flush=True,
    )


def resolve_and_validate_stage2_config(
    args: argparse.Namespace, checkpoint_config: dict[str, Any]
) -> None:
    requested_tokens = stage2_train._parse_auto_int(
        args.num_future_tokens, name="--num-future-tokens"
    )
    checkpoint_tokens = _optional_int(
        checkpoint_config.get(
            "resolved_num_future_tokens", checkpoint_config.get("num_future_tokens")
        )
    )
    resolved_tokens = (
        requested_tokens if requested_tokens is not None else checkpoint_tokens
    )
    if resolved_tokens is None:
        resolved_tokens = 512
    args.resolved_num_future_tokens = int(resolved_tokens)
    args.num_future_tokens = str(int(resolved_tokens))

    comparisons = {
        "current_frame_count": int(args.current_frame_count),
        "future_frame_count": int(args.future_frame_count),
        "resolved_num_future_tokens": int(args.resolved_num_future_tokens),
        "future_predictor_layers": int(args.future_predictor_layers),
        "future_predictor_hidden_dim": int(args.future_predictor_hidden_dim),
        "future_predictor_heads": int(args.future_predictor_heads),
        "vjepa_img_size": int(args.vjepa_img_size),
    }
    for key, cli_value in comparisons.items():
        ckpt_value = _optional_int(checkpoint_config.get(key))
        if ckpt_value is None:
            print(f"WARNING Stage 2 checkpoint config is missing {key}", flush=True)
            continue
        if int(ckpt_value) != int(cli_value):
            _warn_config_diff(key, ckpt_value, cli_value)
            if key in SHAPE_CRITICAL_CONFIG_KEYS:
                raise ValueError(
                    f"Shape-critical Stage 2 config mismatch for {key}: "
                    f"checkpoint={ckpt_value}, CLI={cli_value}."
                )

    checkpoint_use_proprio = _optional_bool(checkpoint_config.get("use_proprio"))
    if checkpoint_use_proprio is not None and bool(checkpoint_use_proprio) != bool(
        args.use_proprio
    ):
        _warn_config_diff("use_proprio", checkpoint_use_proprio, bool(args.use_proprio))
        if bool(args.use_proprio) and not bool(checkpoint_use_proprio):
            raise ValueError(
                "CLI enables proprio, but the Stage 2 checkpoint config says use_proprio=False."
            )

    action_dim = _optional_int(checkpoint_config.get("action_dim"))
    if action_dim is not None and action_dim != 7:
        raise ValueError(
            f"Stage 2 checkpoint action_dim must be 7 for LIBERO, got {action_dim}."
        )
    if int(args.current_frame_count) != 4 or int(args.future_frame_count) != 4:
        raise ValueError(
            "Stage 2 LIBERO rollout requires current_frame_count=4 and future_frame_count=4."
        )
    if int(args.resolved_num_future_tokens) != 512:
        raise ValueError(
            "Stage 2 LIBERO rollout expects resolved_num_future_tokens=512 for 4 V-JEPA2 frames; "
            f"got {args.resolved_num_future_tokens}."
        )
    if int(args.future_predictor_hidden_dim) != 1024:
        raise ValueError(
            "Stage 2 LIBERO rollout expects future_predictor_hidden_dim=1024."
        )
    if (
        int(args.future_predictor_layers) != 24
        or int(args.future_predictor_heads) != 16
    ):
        raise ValueError(
            "Stage 2 LIBERO rollout expects future_predictor_layers=24 and heads=16."
        )
    if bool(args.use_proprio) is False:
        print(
            "WARNING running Stage 2 checkpoint with proprio disabled by CLI.",
            flush=True,
        )


def apply_libero_data_root(cfg: DictConfig, args: argparse.Namespace) -> None:
    if args.libero_data_root is None:
        return
    root = require_dir(args.libero_data_root, name="--libero-data-root")
    if (root / "meta").exists() or (root / "data").exists():
        dataset_dirs = [root]
    else:
        dataset_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.endswith("_lerobot")
        )
        if not dataset_dirs:
            raise FileNotFoundError(
                f"--libero-data-root did not contain *_lerobot dataset dirs: {root}"
            )
    cfg.data.train.dataset_dirs = [str(path) for path in dataset_dirs]


def resolve_dataset_stats_path(cfg: DictConfig, args: argparse.Namespace) -> Path:
    candidates: list[Path] = []
    explicit = (
        resolve_path(args.dataset_stats_path) if args.dataset_stats_path else None
    )
    if explicit is not None:
        candidates.append(explicit)

    cfg_stats = None
    if OmegaConf.select(cfg, "EVALUATION.dataset_stats_path") is not None:
        cfg_stats = OmegaConf.select(cfg, "EVALUATION.dataset_stats_path")
    elif OmegaConf.select(cfg, "data.train.dataset_stats_path") is not None:
        cfg_stats = OmegaConf.select(cfg, "data.train.dataset_stats_path")
    if cfg_stats:
        cfg_stats_path = resolve_path(str(cfg_stats))
        if cfg_stats_path is not None:
            candidates.append(cfg_stats_path)

    stage2_checkpoint = resolve_path(args.stage2_checkpoint)
    if stage2_checkpoint is not None:
        for parent in list(stage2_checkpoint.parents)[:5]:
            candidates.append(parent / "dataset_stats.json")

    if args.libero_data_root is not None:
        root = resolve_path(args.libero_data_root)
        if root is not None:
            candidates.append(root / "dataset_stats.json")
            if root.exists() and root.is_dir():
                for child in sorted(
                    path
                    for path in root.iterdir()
                    if path.is_dir() and path.name.endswith("_lerobot")
                ):
                    candidates.append(child / "dataset_stats.json")

    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Pass --dataset-stats-path or place it near --stage2-checkpoint."
    )


def extract_module_state(
    payload: dict[str, Any],
    *,
    direct_keys: tuple[str, ...],
    model_prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor] | None:
    return stage2_train.extract_nested_state(
        payload,
        direct_keys=direct_keys,
        model_prefixes=model_prefixes,
    )


def _first_items(items: list[Any], limit: int = 10) -> list[Any]:
    return items[:limit]


def load_module_exact(
    module: torch.nn.Module,
    state_dict: dict[str, Any] | None,
    *,
    name: str,
    required: bool,
) -> dict[str, Any]:
    if not state_dict:
        if required:
            raise ValueError(f"Stage 2 checkpoint is missing required {name} weights.")
        return {
            "loaded_keys_count": 0,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatch": [],
        }

    cleaned = stage2_train.strip_prefixes(
        {key: value for key, value in state_dict.items() if torch.is_tensor(value)}
    )
    own_state = module.state_dict()
    missing = [key for key in own_state if key not in cleaned]
    unexpected = [key for key in cleaned if key not in own_state]
    shape_mismatch = [
        {
            "key": key,
            "source_shape": tuple(cleaned[key].shape),
            "target_shape": tuple(own_state[key].shape),
        }
        for key in cleaned
        if key in own_state and tuple(cleaned[key].shape) != tuple(own_state[key].shape)
    ]
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"Stage 2 {name} state_dict mismatch: "
            f"missing={_first_items(missing)} unexpected={_first_items(unexpected)} "
            f"shape_mismatch={_first_items(shape_mismatch)}"
        )
    module.load_state_dict(cleaned, strict=True)
    return {
        "loaded_keys_count": len(cleaned),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatch": [],
    }


def load_stage2_modules(
    model: torch.nn.Module,
    payload: dict[str, Any],
    *,
    use_proprio: bool,
) -> dict[str, Any]:
    stats = {
        "action_expert": load_module_exact(
            model.action_expert,
            extract_module_state(
                payload,
                direct_keys=("action_expert", "action", "action_state_dict"),
                model_prefixes=("action_expert.", "module.action_expert."),
            ),
            name="action_expert",
            required=True,
        ),
        "future_predictor": load_module_exact(
            model.future_predictor,
            extract_module_state(
                payload,
                direct_keys=("future_predictor", "predictor", "predictor_state_dict"),
                model_prefixes=("future_predictor.", "module.future_predictor."),
            ),
            name="future_predictor",
            required=True,
        ),
        "jepa_adapter": load_module_exact(
            model.jepa_adapter,
            extract_module_state(
                payload,
                direct_keys=("jepa_adapter", "adapter", "adapter_state_dict"),
                model_prefixes=("jepa_adapter.", "module.jepa_adapter."),
            ),
            name="jepa_adapter",
            required=True,
        ),
    }
    if use_proprio:
        if model.proprio_encoder is None:
            raise RuntimeError("use_proprio=True but model.proprio_encoder is None.")
        stats["proprio_encoder"] = load_module_exact(
            model.proprio_encoder,
            extract_module_state(
                payload,
                direct_keys=("proprio_encoder", "proprio_projection"),
                model_prefixes=("proprio_encoder.", "module.proprio_encoder."),
            ),
            name="proprio_encoder",
            required=True,
        )
    else:
        stats["proprio_encoder"] = {"loaded_keys_count": 0, "disabled_by_cli": True}
    return stats


def build_stage2_model(
    *,
    cfg: DictConfig,
    args: argparse.Namespace,
    payload: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    from fastwam.models.wan22.action_dit import ActionDiT

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):
        raise ValueError(f"cfg.model must resolve to dict, got {type(model_cfg)}.")
    action_cfg = dict(model_cfg["action_dit_config"])
    if int(action_cfg["action_dim"]) != 7:
        raise ValueError(
            f"LIBERO action_dim must be 7, got {action_cfg['action_dim']}."
        )

    action_expert = ActionDiT(**action_cfg)
    vjepa_encoder = stage2_train.build_vjepa_encoder(args, device=device, dtype=dtype)
    model = stage2_train.build_model(
        cfg=cfg,
        args=args,
        vjepa_encoder=vjepa_encoder,
        action_expert=action_expert,
        device=device,
        dtype=dtype,
    )
    load_stats = load_stage2_modules(model, payload, use_proprio=bool(args.use_proprio))
    model.eval()
    model.requires_grad_(False)
    model.vjepa_encoder.eval()
    model.vjepa_encoder.requires_grad_(False)
    return model, load_stats


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _frame_to_rgb(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        for key in ("image", "agentview_image", "wrist_image"):
            if key in frame:
                frame = frame[key]
                break
        else:
            raise ValueError(
                "Video frame dict must contain image, agentview_image, or wrist_image."
            )
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3:
        raise ValueError(f"Video frame must be HWC RGB-like, got shape {array.shape}.")
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Video frame must have 3 channels, got shape {array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _warn_video_issue(prefix: str, exc: Exception) -> None:
    message = str(exc).replace("\n", " ")[:300]
    print(f"WARNING {prefix}: {type(exc).__name__}: {message}", flush=True)


def summarize_task(rows: list[dict[str, Any]], *, description: str) -> dict[str, Any]:
    successes = sum(1 for row in rows if bool(row["success"]))
    returns = [float(row["return"]) for row in rows]
    lengths = [float(row["length"]) for row in rows]
    total = len(rows)
    return {
        "description": description,
        "episodes": int(total),
        "successes": int(successes),
        "success_rate": float(successes / max(total, 1)),
        "avg_episode_length": _mean(lengths),
        "avg_return": _mean(returns),
    }


def predict_env_action_chunk(
    *,
    env,
    obs: dict,
    model: torch.nn.Module,
    processor,
    cfg: DictConfig,
    args: argparse.Namespace,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_horizon: int,
    input_w: int,
    input_h: int,
    episode_seed: int,
) -> tuple[np.ndarray, dict]:
    return rollout_base.predict_env_action_chunk(
        env=env,
        obs=obs,
        model=model,
        processor=processor,
        cfg=cfg,
        args=args,
        context=context,
        context_mask=context_mask,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        episode_seed=episode_seed,
        mode=str(args.future_source),
    )


def run_rollout_episode(
    *,
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor,
    cfg: DictConfig,
    args: argparse.Namespace,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_horizon: int,
    input_w: int,
    input_h: int,
    task_id: int,
    episode_idx: int,
    video_dir: Path | None,
) -> dict[str, Any]:
    from experiments.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_image,
        save_rollout_video,
    )

    setattr(
        rollout_base.predict_env_action_chunk,
        "_frame_histories",
        {
            str(args.future_source): rollout_base.FrameHistory(
                frame_count=int(args.current_frame_count)
            )
        },
    )
    env.reset()
    obs = env.set_init_state(initial_state)
    replay_images = []
    pending_actions: list[list[float]] = []
    episode_return = 0.0
    success = False
    length = 0
    video_path = None
    max_policy_steps = rollout_base.get_max_steps(
        str(args.libero_suite), args.max_steps
    )
    max_total_steps = int(max_policy_steps) + int(args.num_steps_wait)

    for step_idx in range(max_total_steps):
        if step_idx < int(args.num_steps_wait):
            action_to_env = get_libero_dummy_action()
        else:
            if not pending_actions:
                action, imgs = predict_env_action_chunk(
                    env=env,
                    obs=obs,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    args=args,
                    context=context,
                    context_mask=context_mask,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    episode_seed=int(args.effective_seed)
                    + int(task_id) * 100_000
                    + int(episode_idx),
                )
                if bool(args.save_videos):
                    try:
                        replay_images.append(_frame_to_rgb(imgs))
                    except Exception as exc:
                        _warn_video_issue("failed to capture rollout frame", exc)
                pending_actions = action[: int(args.exec_horizon)].tolist()
            action_to_env = pending_actions.pop(0)

        obs, reward, done, _ = env.step(action_to_env)
        if (
            bool(args.save_videos)
            and step_idx >= int(args.num_steps_wait)
            and pending_actions
        ):
            try:
                replay_images.append(_frame_to_rgb(get_libero_image(obs)))
            except Exception as exc:
                _warn_video_issue("failed to capture rollout frame", exc)
        episode_return += float(np.asarray(reward).reshape(-1)[0])
        length += 1
        if done:
            success = True
            break

    if bool(args.save_videos):
        if video_dir is None:
            raise RuntimeError(
                "--save-videos was enabled but no video_dir was provided."
            )
        video_dir.mkdir(parents=True, exist_ok=True)
        if replay_images:
            try:
                video_path = save_rollout_video(
                    video_dir,
                    replay_images,
                    f"{args.future_source}_task{task_id}_episode{episode_idx}",
                    success=success,
                    task_description=task_description,
                    fps=20,
                )
            except Exception as exc:
                _warn_video_issue("failed to save rollout video", exc)
                video_path = None
        else:
            print("WARNING no rollout frames captured; skipping video save", flush=True)


    return {
        "future_source": str(args.future_source),
        "task_id": int(task_id),
        "episode": int(episode_idx),
        "success": bool(success),
        "return": float(episode_return),
        "length": int(length),
        "video_path": video_path,
    }


def run_rollout_eval(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    args: argparse.Namespace,
    output_json: Path,
    load_stats: dict[str, Any],
) -> dict[str, Any]:
    from libero.libero import benchmark

    from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils.pytorch_utils import set_global_seed

    safe_seed = int(args.seed)
    if safe_seed <= 0:
        print("WARNING --seed must be >0 for FastWAM set_global_seed; using seed=1", flush=True)
        safe_seed = 1
    args.effective_seed = safe_seed
    set_global_seed(safe_seed, get_worker_init_fn=False)
    stats_path = resolve_dataset_stats_path(cfg, args)
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    action_horizon = int(args.action_horizon)
    if action_horizon <= 0:
        raise ValueError(f"--action-horizon must be positive, got {action_horizon}.")
    if int(args.exec_horizon) <= 0 or int(args.exec_horizon) > action_horizon:
        raise ValueError(
            f"--exec-horizon must be in [1, action_horizon], got {args.exec_horizon} vs {action_horizon}."
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[str(args.libero_suite)]()
    task_ids = rollout_base.get_task_ids(task_suite, str(args.task_id))
    video_dir = output_json.parent / "videos" if bool(args.save_videos) else None
    per_episode: list[dict[str, Any]] = []
    per_task: dict[str, Any] = {}

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = list(task_suite.get_task_init_states(task_id))
        while len(initial_states) < int(args.num_episodes):
            initial_states.extend(
                initial_states[: int(args.num_episodes) - len(initial_states)]
            )
        env, task_description = get_libero_env(
            task, LIBERO_ENV_RESOLUTION, safe_seed
        )
        prompt = DEFAULT_PROMPT.format(task=task_description)
        context, context_mask = rollout_base.load_cached_text_context(prompt, cfg)
        task_rows: list[dict[str, Any]] = []
        print(
            f"rollout_task_id={task_id} future_source={args.future_source}", flush=True
        )
        try:
            for episode_idx in range(int(args.num_episodes)):
                result = run_rollout_episode(
                    env=env,
                    initial_state=initial_states[episode_idx],
                    task_description=task_description,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    args=args,
                    context=context,
                    context_mask=context_mask,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    task_id=int(task_id),
                    episode_idx=int(episode_idx),
                    video_dir=video_dir,
                )
                result["task_description"] = task_description
                task_rows.append(result)
                per_episode.append(result)
                print(
                    f"episode task={task_id} idx={episode_idx} success={result['success']} "
                    f"return={result['return']:.4f} length={result['length']}",
                    flush=True,
                )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        per_task[str(task_id)] = summarize_task(task_rows, description=task_description)

    success_count = sum(1 for row in per_episode if bool(row["success"]))
    returns = [float(row["return"]) for row in per_episode]
    lengths = [float(row["length"]) for row in per_episode]
    total = len(per_episode)
    results = {
        "stage2_checkpoint": str(resolve_path(args.stage2_checkpoint)),
        "vjepa_checkpoint": str(resolve_path(args.vjepa_checkpoint)),
        "future_source": str(args.future_source),
        "libero_suite": str(args.libero_suite),
        "num_episodes": int(args.num_episodes),
        "seed": int(args.seed),
        "effective_seed": int(args.effective_seed),
        "overall_success_rate": float(success_count / max(total, 1)),
        "per_task": per_task,
        "avg_episode_length": _mean(lengths),
        "avg_return": _mean(returns),
        "successes": int(success_count),
        "total_episodes": int(total),
        "dataset_stats_path": str(stats_path),
        "use_proprio": bool(args.use_proprio),
        "current_frame_count": int(args.current_frame_count),
        "future_frame_count": int(args.future_frame_count),
        "resolved_num_future_tokens": int(args.resolved_num_future_tokens),
        "exec_horizon": int(args.exec_horizon),
        "action_horizon": int(action_horizon),
        "save_videos": bool(args.save_videos),
        "oracle_note": (
            "oracle uses existing simulator snapshot_dummy future peek; it is a debug baseline, "
            "not a causal deployable policy."
            if str(args.future_source) == "oracle"
            else None
        ),
        "load_stats": load_stats,
        "episodes": per_episode,
    }
    return results


def main() -> None:
    args = parse_args()
    args.task_suite = args.libero_suite
    args.rollout_modes = args.future_source
    args.save_rollout_video = bool(args.save_videos)
    if int(args.num_episodes) <= 0:
        raise ValueError("--num-episodes must be positive.")

    rollout_base.configure_mujoco_env(args)
    device = resolve_device(str(args.device))
    dtype, _ = stage2_train.precision_to_dtype(str(args.precision), device)
    stage2_checkpoint = require_file(args.stage2_checkpoint, name="--stage2-checkpoint")
    payload = load_checkpoint_payload(stage2_checkpoint)
    checkpoint_config = merged_checkpoint_config(payload)
    resolve_and_validate_stage2_config(args, checkpoint_config)

    cfg = stage2_train.compose_cfg(str(args.config_name), str(args.task))
    apply_libero_data_root(cfg, args)
    model, load_stats = build_stage2_model(
        cfg=cfg,
        args=args,
        payload=payload,
        device=device,
        dtype=dtype,
    )

    output_json = resolve_path(args.output_json)
    if output_json is None:
        raise ValueError("--output-json is required.")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    results = run_rollout_eval(
        cfg=cfg,
        model=model,
        args=args,
        output_json=output_json,
        load_stats=load_stats,
    )
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=rollout_base.NumpyEncoder)
    print(
        f"saved_results={output_json} overall_success_rate={results['overall_success_rate']:.4f} "
        f"avg_episode_length={results['avg_episode_length']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
