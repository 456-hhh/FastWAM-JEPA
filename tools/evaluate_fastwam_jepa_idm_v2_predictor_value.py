from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
LIBERO_EXP_ROOT = PROJECT_ROOT / "experiments" / "libero"
for path in (SRC_ROOT, PROJECT_ROOT, LIBERO_EXP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam_jepa_runtime_guard import configure_runtime_stability

DEFAULT_ACTION_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM/runs/libero_joint_2cam224_1e-4/"
    "libero_joint_4gpu_20k_20260524_171732/checkpoints/weights/step_020000.pt"
)
DEFAULT_VJEPA_REPO = "/data1/Johnny/challenge/dd/FastWAM_jepa/external/vjepa2"
DEFAULT_VJEPA_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM_jepa/checkpoints/vjepa2/vitg.pt"
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class FrameHistory:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = int(frame_count)
        self.frames: list[torch.Tensor] = []

    def append(self, frame: torch.Tensor) -> None:
        if frame.ndim != 4 or frame.shape[0] != 1 or frame.shape[1] != 3:
            raise ValueError(f"Frame must be [1, 3, H, W], got {tuple(frame.shape)}.")
        self.frames.append(frame.detach())
        self.frames = self.frames[-self.frame_count :]

    def as_video(self) -> torch.Tensor:
        if not self.frames:
            raise ValueError("FrameHistory is empty.")
        frames = list(self.frames)
        while len(frames) < self.frame_count:
            frames.insert(0, frames[0].clone())
        return torch.stack(frames[-self.frame_count :], dim=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline predictor-value evaluation for FastWAM-JEPA-IDM v2. "
            "Compares oracle_future, predicted_future, and no_future on the same real batch."
        )
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_joint_2cam224_1e-4")
    parser.add_argument("--eval-mode", default="loss_proxy", choices=["loss_proxy", "rollout"])
    parser.add_argument("--action-checkpoint", default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--vjepa-repo", default=DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-checkpoint", default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", default="-1_1", choices=["-1_1", "0_1"])
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disable-wsl-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-log-level", default="INFO", choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--runtime-log-path", default=None)
    parser.add_argument("--runtime-log-max-mb", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--output-dir", default="evaluate_results/fastwam_jepa_idm_v2_predictor_value")
    parser.add_argument("--current-frame-count", type=int, default=2)
    parser.add_argument("--future-frame-count", type=int, default=2)
    parser.add_argument("--num-future-tokens", type=int, default=256)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--adapter-current-tokens", type=int, default=16)
    parser.add_argument("--adapter-future-tokens", type=int, default=16)
    parser.add_argument("--future-predictor-layers", type=int, default=2)
    parser.add_argument("--future-predictor-heads", type=int, default=8)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", default="0", help="Task id or `all`.")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--rollout-modes", default="oracle,predicted,no_future")
    parser.add_argument(
        "--oracle-future-mode",
        default="snapshot_dummy",
        choices=["snapshot_dummy"],
        help=(
            "Oracle rollout peeks by snapshotting the simulator, stepping dummy actions "
            "to collect future frames, then restoring the original state. This is a "
            "simulator-peek baseline, not a causal deployable policy."
        ),
    )
    parser.add_argument("--save-rollout-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--binarize-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument(
        "--mujoco-gl",
        default="egl",
        choices=["egl", "osmesa", "glfw"],
        help="MuJoCo GL backend used only by --eval-mode rollout.",
    )
    parser.add_argument(
        "--pyopengl-platform",
        default="egl",
        choices=["egl", "osmesa", "glfw"],
        help="PyOpenGL platform used only by --eval-mode rollout.",
    )
    parser.add_argument(
        "--egl-device-id",
        type=int,
        default=0,
        help="Runtime EGL device id. With CUDA_VISIBLE_DEVICES=7 this is usually 0.",
    )
    parser.add_argument(
        "--egl-import-device-id",
        default=None,
        help=(
            "Temporary MUJOCO_EGL_DEVICE_ID used only while importing robosuite. "
            "Defaults to the first CUDA_VISIBLE_DEVICES entry."
        ),
    )
    return parser.parse_args()


def resolve_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def seed_everything(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def compose_cfg(config_name: str, task: str) -> DictConfig:
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir = (PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=[f"task={task}"])


def resolve_dataset_dirs(cfg: DictConfig) -> None:
    dataset_dirs = cfg.data.train.get("dataset_dirs")
    if dataset_dirs is None:
        raise ValueError("`cfg.data.train.dataset_dirs` is required.")

    resolved_dirs: list[str] = []
    print("Resolved dataset_dirs:", flush=True)
    for dataset_dir in dataset_dirs:
        path = Path(str(dataset_dir))
        abs_path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        print(f"  {abs_path}", flush=True)
        if not abs_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {abs_path}")
        if not abs_path.is_dir():
            raise FileNotFoundError(f"Dataset path is not a directory: {abs_path}")
        resolved_dirs.append(str(abs_path))

    cfg.data.train.dataset_dirs = resolved_dirs


def build_loader(cfg: DictConfig, *, batch_size: int, num_workers: int) -> DataLoader:
    resolve_dataset_dirs(cfg)
    dataset = instantiate(cfg.data.train)
    print(f"Dataset: {type(dataset).__name__}, len={len(dataset)}", flush=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def assert_batch_video(sample: dict[str, Any], *, current_frames: int, future_frames: int) -> None:
    video = sample.get("video")
    if not torch.is_tensor(video):
        raise ValueError("Batch is missing tensor `sample['video']`.")
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(
            "`sample['video']` must be [B, 3, T, H, W]. "
            f"Got {tuple(video.shape)}. This script does not silently permute."
        )
    required_frames = int(current_frames) + int(future_frames)
    if int(video.shape[2]) < required_frames:
        raise ValueError(
            "`sample['video']` does not contain enough frames: "
            f"T={int(video.shape[2])}, required at least {required_frames}."
        )


def _state_dict_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key in ("mot", "model", "model_state_dict", "state_dict", "module"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    candidates.append(payload)
    return candidates


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key[len("module.") :] if isinstance(key, str) and key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def extract_action_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    prefixes = (
        "mixtures.action.",
        "dit.mixtures.action.",
        "mot.mixtures.action.",
        "module.mixtures.action.",
        "module.dit.mixtures.action.",
        "module.mot.mixtures.action.",
        "action_expert.",
        "model.action_expert.",
        "module.action_expert.",
    )
    for state in _state_dict_candidates(payload):
        for prefix in prefixes:
            action_state = {
                key[len(prefix) :]: value
                for key, value in state.items()
                if isinstance(key, str) and key.startswith(prefix)
            }
            if action_state:
                return action_state
    raise ValueError("Could not find ActionDiT weights in checkpoint.")


def extract_predictor_state_dict(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("future_predictor", "predictor", "predictor_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return strip_module_prefix(value)

    for state_key in ("model", "model_state_dict", "state_dict"):
        state = payload.get(state_key)
        if isinstance(state, dict):
            filtered = {}
            for key, value in strip_module_prefix(state).items():
                if isinstance(key, str) and key.startswith("future_predictor."):
                    filtered[key[len("future_predictor.") :]] = value
            if filtered:
                return filtered

    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return strip_module_prefix(payload)

    raise ValueError("Could not find JepaFuturePredictor weights in checkpoint.")


def build_action_expert(
    *,
    action_cfg: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from fastwam.models.wan22.action_dit import ActionDiT

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Action checkpoint does not exist: {checkpoint_path}")
    action_expert = ActionDiT(**action_cfg)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Action checkpoint payload must be a dict, got {type(payload)}.")
    action_expert.load_state_dict(extract_action_state_dict(payload), strict=True)
    return action_expert.to(device=device, dtype=dtype)


def load_predictor_checkpoint(model: torch.nn.Module, checkpoint_path: Path, *, device: torch.device) -> str:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Predictor checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Predictor checkpoint must be a dict, got {type(payload)}.")
    state = extract_predictor_state_dict(payload)
    missing, unexpected = model.future_predictor.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "Unexpected predictor load_state_dict result with strict=True: "
            f"missing={missing}, unexpected={unexpected}."
        )
    step = payload.get("step", "unknown")
    return f"loaded_checkpoint:{checkpoint_path}:step={step}"


def build_model(
    *,
    cfg: DictConfig,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from fastwam.models.vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper
    from fastwam.models.wan22.fastwam_jepa_idm import FastWAMJEPAIDM

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):
        raise ValueError(f"`cfg.model` must resolve to dict, got {type(model_cfg)}.")
    action_cfg = dict(model_cfg["action_dit_config"])

    action_expert = build_action_expert(
        action_cfg=action_cfg,
        checkpoint_path=resolve_path(args.action_checkpoint),
        device=device,
        dtype=dtype,
    )
    action_expert.eval()
    action_expert.requires_grad_(False)

    vjepa_encoder = VJepaEncoderWrapper(
        dummy=False,
        model_name=args.vjepa_model_name,
        external_repo_path=str(resolve_path(args.vjepa_repo)),
        checkpoint_path=str(resolve_path(args.vjepa_checkpoint)),
        pretrained=False,
        vjepa_dim=int(args.vjepa_dim),
        num_tokens=int(args.num_future_tokens),
        freeze=True,
        normalize_tokens=False,
        img_size=int(args.vjepa_img_size),
        input_range=str(args.vjepa_input_range),
        tubelet_size=int(args.vjepa_tubelet_size),
        frame_encoding_mode="clip_or_repeat",
    )

    action_scheduler_cfg = model_cfg.get("action_scheduler", {})
    proprio_dim = model_cfg.get("proprio_dim")
    model = FastWAMJEPAIDM(
        action_expert=action_expert,
        vjepa_encoder=vjepa_encoder,
        action_dim=int(action_cfg["action_dim"]),
        hidden_dim=int(action_cfg["hidden_dim"]),
        vjepa_dim=int(args.vjepa_dim),
        num_future_tokens=int(args.num_future_tokens),
        text_dim=int(action_cfg["text_dim"]),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=None,
        torch_dtype=dtype,
        action_train_shift=float(action_scheduler_cfg.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler_cfg.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(action_scheduler_cfg.get("num_train_timesteps", 1000)),
        lambda_action=1.0,
        lambda_future=0.0,
        current_frame_count=int(args.current_frame_count),
        future_frame_count=int(args.future_frame_count),
        adapter_current_tokens=int(args.adapter_current_tokens),
        adapter_future_tokens=int(args.adapter_future_tokens),
        future_predictor_layers=int(args.future_predictor_layers),
        future_predictor_heads=int(args.future_predictor_heads),
        future_source="oracle",
    ).to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    predictor_source = load_predictor_checkpoint(
        model,
        resolve_path(args.predictor_checkpoint),
        device=device,
    )
    model.future_predictor.eval()
    model.future_predictor.requires_grad_(False)
    print(f"predictor_weight_source={predictor_source}", flush=True)
    return model


def run_mode(
    *,
    model: torch.nn.Module,
    sample: dict[str, Any],
    mode: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.future_source = mode
    seed_everything(seed, device)
    with torch.no_grad():
        loss_total, loss_dict = model.training_loss(sample)
    if not torch.isfinite(loss_total):
        raise RuntimeError(f"loss_total is not finite for mode={mode}.")
    return {
        "mode": mode,
        "loss_total": float(loss_total.detach().item()),
        "loss_action": float(loss_dict["loss_action"]),
        "loss_future_jepa": float(loss_dict["loss_future_jepa"]),
        "oracle_vs_predicted_gap": float(loss_dict["oracle_vs_predicted_gap"]),
        "predictor_used": model.last_forward_shapes.get("predictor_used"),
        "shapes": dict(model.last_forward_shapes),
    }


def compute_metrics(results_by_mode: dict[str, dict[str, Any]]) -> dict[str, float]:
    oracle_loss = float(results_by_mode["oracle"]["loss_action"])
    predicted_loss = float(results_by_mode["predicted"]["loss_action"])
    no_future_loss = float(results_by_mode["no_future"]["loss_action"])
    predictor_value_score = oracle_loss - predicted_loss
    stability_gap = predicted_loss - no_future_loss
    predictor_contribution_score = no_future_loss - predicted_loss
    return {
        "oracle_loss_action": oracle_loss,
        "predicted_loss_action": predicted_loss,
        "no_future_loss_action": no_future_loss,
        "oracle_vs_predicted_gap": float(results_by_mode["predicted"]["oracle_vs_predicted_gap"]),
        "predicted_action_loss_delta_vs_oracle": predicted_loss - oracle_loss,
        "no_future_action_loss_delta_vs_oracle": no_future_loss - oracle_loss,
        "predictor_value_score": predictor_value_score,
        "stability_gap": stability_gap,
        "predictor_contribution_score": predictor_contribution_score,
    }


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(sum(values) / max(len(values), 1))


def first_cuda_visible_device() -> str | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return None
    for item in visible_devices.split(","):
        item = item.strip()
        if item and item != "-1":
            return item
    return None


def configure_mujoco_env(args: argparse.Namespace) -> str | None:
    os.environ["MUJOCO_GL"] = str(args.mujoco_gl)
    os.environ["PYOPENGL_PLATFORM"] = str(args.pyopengl_platform)
    egl_import_device_id = None
    if str(args.mujoco_gl) == "egl":
        egl_import_device_id = (
            str(args.egl_import_device_id)
            if args.egl_import_device_id is not None
            else first_cuda_visible_device()
        )
        if egl_import_device_id is None:
            egl_import_device_id = str(int(args.egl_device_id))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_import_device_id)
    return egl_import_device_id


def get_task_ids(task_suite, task_id_arg: str) -> list[int]:
    if str(task_id_arg).lower() == "all":
        return list(range(int(task_suite.n_tasks)))
    return [int(task_id_arg)]


def get_max_steps(task_suite_name: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def parse_rollout_modes(modes_arg: str) -> tuple[str, ...]:
    modes = tuple(mode.strip() for mode in str(modes_arg).split(",") if mode.strip())
    valid = {"oracle", "predicted", "no_future"}
    invalid = [mode for mode in modes if mode not in valid]
    if invalid:
        raise ValueError(f"Invalid rollout modes {invalid}; expected subset of {sorted(valid)}.")
    if not modes:
        raise ValueError("`--rollout-modes` must contain at least one mode.")
    return modes


def resolve_dataset_stats_path(cfg: DictConfig, args: argparse.Namespace) -> Path:
    candidates: list[Path] = []
    explicit = resolve_path(args.dataset_stats_path) if args.dataset_stats_path else None
    if explicit is not None:
        candidates.append(explicit)
    for checkpoint_arg in (args.predictor_checkpoint, args.action_checkpoint):
        checkpoint = resolve_path(checkpoint_arg)
        if checkpoint is not None:
            for parent in list(checkpoint.parents)[:5]:
                candidates.append(parent / "dataset_stats.json")
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Pass --dataset-stats-path explicitly."
    )


def load_cached_text_context(prompt: str, cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = cfg.data.train.get("text_embedding_cache_dir")
    if cache_dir is None:
        raise ValueError("cfg.data.train.text_embedding_cache_dir is required.")
    cache_dir = resolve_path(str(cache_dir))
    context_len = int(cfg.data.train.get("context_len", 128))
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{hashed}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing cached text context: {cache_path}. Run scripts/precompute_text_embeds.py first."
        )
    payload = torch.load(cache_path, map_location="cpu")
    context = payload["context"]
    context_mask = payload["mask"].bool()
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask)
    return context, context_mask


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)),
        resample=Image.BILINEAR,
    )
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _extract_sim_state(obs: dict) -> np.ndarray:
    from experiments.libero.libero_utils import quat2axisangle

    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def _normalize_proprio(proprio: np.ndarray, processor) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]
    state_batch = {
        "state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}
    }
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor,
    width: int,
    height: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    from experiments.libero.libero_utils import get_libero_image

    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(
                f"shape_meta.images[{camera_idx}].shape must be [C, H, W], got {shape}."
            )
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = int(processor.num_output_cameras)
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(
            f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}."
        )

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    if actual_h != expected_h or actual_w != expected_w:
        raise ValueError(
            "Input image size mismatch after per-camera resize + concat: "
            f"got (H, W)=({actual_h}, {actual_w}), expected ({expected_h}, {expected_w})."
        )

    image = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    image = image * (2.0 / 255.0) - 1.0
    proprio = _normalize_proprio(_extract_sim_state(obs), processor)
    return image, proprio, imgs


def _denormalize_action(action: torch.Tensor, processor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}.")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _snapshot_env_state(env) -> dict[str, Any]:
    sim = getattr(env, "sim", None)
    if sim is None:
        raise RuntimeError("Oracle future peek requires env.sim, but this environment has no `sim` attribute.")
    env_attrs: dict[str, Any] = {}
    for name in ("timestep", "_timestep", "cur_time", "done"):
        if hasattr(env, name):
            value = getattr(env, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            env_attrs[name] = value
    if hasattr(sim, "get_state") and hasattr(sim, "set_state"):
        return {"kind": "sim_get_state", "sim": sim, "env": env, "env_attrs": env_attrs, "state": sim.get_state()}

    data = getattr(sim, "data", None)
    model = getattr(sim, "model", None)
    if data is None:
        raise RuntimeError("Oracle future peek could not snapshot env.sim.data.")
    snapshot: dict[str, Any] = {"kind": "mujoco_arrays", "sim": sim, "env": env, "env_attrs": env_attrs}
    for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
        value = getattr(data, name, None)
        if value is not None:
            snapshot[name] = value.copy()
    if hasattr(data, "time"):
        snapshot["time"] = float(data.time)
    if model is not None and hasattr(model, "opt"):
        snapshot["model"] = model
    return snapshot


def _restore_env_state(snapshot: dict[str, Any]) -> None:
    sim = snapshot["sim"]
    env = snapshot.get("env")
    for name, value in snapshot.get("env_attrs", {}).items():
        if env is not None and hasattr(env, name):
            current = getattr(env, name)
            if isinstance(current, np.ndarray) and isinstance(value, np.ndarray):
                current[...] = value
            else:
                setattr(env, name, value)
    if snapshot["kind"] == "sim_get_state":
        sim.set_state(snapshot["state"])
        if hasattr(sim, "forward"):
            sim.forward()
        return

    data = getattr(sim, "data", None)
    if data is None:
        raise RuntimeError("Oracle future peek could not restore env.sim.data.")
    for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
        if name in snapshot and hasattr(data, name):
            getattr(data, name)[:] = snapshot[name]
    if "time" in snapshot and hasattr(data, "time"):
        data.time = snapshot["time"]
    if hasattr(sim, "forward"):
        sim.forward()


def collect_oracle_future_video(
    *,
    env,
    obs: dict,
    cfg: DictConfig,
    processor,
    args: argparse.Namespace,
    input_w: int,
    input_h: int,
    model: torch.nn.Module,
) -> torch.Tensor:
    from experiments.libero.libero_utils import get_libero_dummy_action

    if str(args.oracle_future_mode) != "snapshot_dummy":
        raise ValueError(f"Unsupported oracle_future_mode: {args.oracle_future_mode}")

    snapshot = _snapshot_env_state(env)
    frames: list[torch.Tensor] = []
    try:
        peek_obs = obs
        for _ in range(int(args.future_frame_count)):
            peek_obs, _, done, _ = env.step(get_libero_dummy_action())
            image, _, _ = _obs_to_model_input(
                peek_obs,
                cfg=cfg,
                processor=processor,
                width=input_w,
                height=input_h,
                device=model._runtime_device(),
                dtype=model.torch_dtype,
            )
            frames.append(image)
            if done:
                break
    finally:
        _restore_env_state(snapshot)

    if not frames:
        image, _, _ = _obs_to_model_input(
            obs,
            cfg=cfg,
            processor=processor,
            width=input_w,
            height=input_h,
            device=model._runtime_device(),
            dtype=model.torch_dtype,
        )
        frames.append(image)
    while len(frames) < int(args.future_frame_count):
        frames.append(frames[-1].clone())
    return torch.stack(frames[: int(args.future_frame_count)], dim=2)


@torch.no_grad()
def sample_action_jepa_idm(
    *,
    model: torch.nn.Module,
    current_video: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor,
    action_horizon: int,
    future_source: str,
    oracle_future_video: torch.Tensor | None,
    num_inference_steps: int,
    sigma_shift: float | None,
    seed: int | None,
    rand_device: str,
) -> dict[str, torch.Tensor]:
    original_source = model.future_source
    model.future_source = future_source
    try:
        device = model._runtime_device()
        current_video = current_video.to(device=device, dtype=model.torch_dtype)
        context = context.to(device=device, dtype=model.torch_dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        condition_context, condition_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=device, dtype=model.torch_dtype),
        )

        current_jepa_tokens = model._encode_jepa_video(current_video)
        predictor_used = future_source == "predicted"
        if predictor_used:
            future_out = model.future_predictor(
                current_jepa_tokens=current_jepa_tokens,
                condition_context=condition_context,
                condition_mask=condition_mask,
            )
            adapter_future_tokens = future_out["pred_future_tokens"]
        elif future_source == "oracle":
            if oracle_future_video is None:
                raise ValueError("future_source='oracle' requires oracle_future_video.")
            adapter_future_tokens = model._encode_jepa_video(
                oracle_future_video.to(device=device, dtype=model.torch_dtype)
            )
        elif future_source == "no_future":
            adapter_future_tokens = None
        else:
            raise ValueError(f"Unexpected future_source={future_source!r}.")

        action_context, action_context_mask = model.jepa_adapter(
            current_jepa_tokens=current_jepa_tokens,
            future_jepa_tokens=adapter_future_tokens,
            base_context=condition_context,
            base_context_mask=condition_mask,
        )

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, int(action_horizon), int(model.action_dim)),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=device, dtype=model.torch_dtype)

        timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=device)
            pred_action = model.action_expert(
                action_tokens=latents_action,
                timestep=timestep,
                context=action_context,
                context_mask=action_context_mask,
            )
            latents_action = model.infer_action_scheduler.step(pred_action, step_delta, latents_action)

        model.last_forward_shapes = {
            "future_source": future_source,
            "current_jepa_tokens": tuple(current_jepa_tokens.shape),
            "action_context": tuple(action_context.shape),
            "action_context_mask": tuple(action_context_mask.shape),
            "pred_action": tuple(latents_action.shape),
            "predictor_used": "true" if predictor_used else "false",
        }
        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
    finally:
        model.future_source = original_source


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
    mode: str,
) -> tuple[np.ndarray, dict]:
    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model._runtime_device(),
        dtype=model.torch_dtype,
    )
    if not hasattr(predict_env_action_chunk, "_frame_histories"):
        raise RuntimeError("Frame history storage was not initialized.")
    histories = getattr(predict_env_action_chunk, "_frame_histories")
    frame_history: FrameHistory = histories[mode]
    frame_history.append(image)
    current_video = frame_history.as_video()

    oracle_future_video = None
    if mode == "oracle":
        oracle_future_video = collect_oracle_future_video(
            env=env,
            obs=obs,
            cfg=cfg,
            processor=processor,
            args=args,
            input_w=input_w,
            input_h=input_h,
            model=model,
        )

    pred = sample_action_jepa_idm(
        model=model,
        current_video=current_video,
        context=context,
        context_mask=context_mask,
        proprio=proprio,
        action_horizon=action_horizon,
        future_source=mode,
        oracle_future_video=oracle_future_video,
        num_inference_steps=int(args.num_inference_steps),
        sigma_shift=args.sigma_shift,
        seed=episode_seed,
        rand_device=str(args.rand_device),
    )
    action = _denormalize_action(pred["action"], processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1

    from experiments.libero.libero_utils import invert_gripper_action

    action = invert_gripper_action(action)
    if bool(args.binarize_gripper):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs


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
    mode: str,
    video_dir: Path,
) -> dict[str, Any]:
    from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_image, save_rollout_video

    max_steps = get_max_steps(args.task_suite, args.max_steps)
    if bool(args.dry_run):
        max_steps = min(max_steps, 10)
    setattr(
        predict_env_action_chunk,
        "_frame_histories",
        {mode: FrameHistory(frame_count=int(args.current_frame_count))},
    )

    env.reset()
    obs = env.set_init_state(initial_state)
    replay_images = []
    pending_actions: list[list[float]] = []
    episode_return = 0.0
    success = False
    length = 0
    max_total_steps = max_steps + int(args.num_steps_wait)

    for t in range(max_total_steps):
        if t < int(args.num_steps_wait):
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
                    episode_seed=int(args.seed) + task_id * 100_000 + episode_idx,
                    mode=mode,
                )
                if bool(args.save_rollout_video):
                    replay_images.append(imgs.copy())
                pending_actions = action[: int(args.replan_steps)].tolist()
            action_to_env = pending_actions.pop(0)

        obs, reward, done, _ = env.step(action_to_env)
        if bool(args.save_rollout_video) and t >= int(args.num_steps_wait) and pending_actions:
            replay_images.append(get_libero_image(obs))
        episode_return += float(np.asarray(reward).reshape(-1)[0])
        length += 1
        if done:
            success = True
            break

    video_path = None
    if bool(args.save_rollout_video):
        video_path = save_rollout_video(
            video_dir,
            replay_images,
            f"{mode}_task{task_id}_episode{episode_idx}",
            success=success,
            task_description=task_description,
        )

    return {
        "mode": mode,
        "task_id": int(task_id),
        "episode": int(episode_idx),
        "success": bool(success),
        "return": float(episode_return),
        "length": int(length),
        "video_path": video_path,
    }


def summarize_rollout_mode(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    successes = sum(1 for row in rows if row["success"])
    returns = [float(row["return"]) for row in rows]
    lengths = [float(row["length"]) for row in rows]
    success_rate = float(successes / max(total, 1))
    return {
        "episodes": float(total),
        "successes": float(successes),
        "success_rate": success_rate,
        "task_completion_rate": success_rate,
        "episode_return": float(np.mean(returns)) if returns else 0.0,
        "avg_episode_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def run_rollout_eval(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    from libero.libero import benchmark

    from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils.pytorch_utils import set_global_seed

    set_global_seed(int(args.seed), get_worker_init_fn=False)
    stats_path = resolve_dataset_stats_path(cfg, args)
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    action_horizon = int(args.action_horizon) if args.action_horizon else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}.")

    modes = parse_rollout_modes(args.rollout_modes)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[str(args.task_suite)]()
    task_ids = get_task_ids(task_suite, str(args.task_id))
    if bool(args.dry_run):
        args.num_episodes = 1

    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    per_episode: list[dict[str, Any]] = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = list(task_suite.get_task_init_states(task_id))
        while len(initial_states) < int(args.num_episodes):
            initial_states.extend(initial_states[: int(args.num_episodes) - len(initial_states)])
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, int(args.seed))
        prompt = DEFAULT_PROMPT.format(task=task_description)
        context, context_mask = load_cached_text_context(prompt, cfg)

        print(f"rollout_task_id={task_id} description={task_description}", flush=True)
        try:
            for episode_idx in range(int(args.num_episodes)):
                for mode in modes:
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
                        mode=mode,
                        video_dir=video_dir,
                    )
                    result["task_description"] = task_description
                    per_episode.append(result)
                    print(
                        f"mode={mode} task={task_id} episode={episode_idx} "
                        f"success={result['success']} return={result['return']:.4f} "
                        f"length={result['length']}",
                        flush=True,
                    )
        finally:
            if hasattr(env, "close"):
                env.close()

    by_mode = {
        mode: summarize_rollout_mode([row for row in per_episode if row["mode"] == mode])
        for mode in modes
    }
    predicted_sr = by_mode.get("predicted", {}).get("success_rate", math.nan)
    no_future_sr = by_mode.get("no_future", {}).get("success_rate", math.nan)
    oracle_sr = by_mode.get("oracle", {}).get("success_rate", math.nan)
    summary = {
        "modes": by_mode,
        "predictor_policy_gain": float(predicted_sr - no_future_sr),
        "oracle_upper_bound": float(oracle_sr),
        "oracle_future_note": (
            "oracle mode uses simulator snapshot_dummy peek and is not a causal deployable policy."
        ),
        "num_tasks": float(len(task_ids)),
        "num_episodes_per_task": float(args.num_episodes),
    }
    return {"summary": summary, "episodes": per_episode}


def main() -> None:
    args = parse_args()
    if args.eval_mode == "rollout":
        egl_import_device_id = configure_mujoco_env(args)
    else:
        egl_import_device_id = None
    runtime_status = configure_runtime_stability(
        disable_wsl_fallback=args.disable_wsl_fallback,
        log_level=args.runtime_log_level,
        log_path=args.runtime_log_path,
        max_log_mb=args.runtime_log_max_mb,
    )
    print(f"runtime_safe_mode={runtime_status['safe_mode']}", flush=True)
    print(f"runtime_disable_wsl_fallback={runtime_status['disable_wsl_fallback']}", flush=True)
    print(f"runtime_sandbox_failure_seen={runtime_status['sandbox_failure_seen']}", flush=True)
    print(f"runtime_log_level={runtime_status['log_level']}", flush=True)
    print(f"runtime_disk_log_enabled={runtime_status['disk_log_enabled']}", flush=True)
    print(f"runtime_log_path={runtime_status['log_path']}", flush=True)
    print(f"runtime_log_rotation_max_mb={runtime_status['log_rotation_max_mb']}", flush=True)
    if args.num_batches <= 0:
        raise ValueError(f"`--num-batches` must be positive, got {args.num_batches}.")
    if args.batch_size <= 0:
        raise ValueError(f"`--batch-size` must be positive, got {args.batch_size}.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed, device)
    output_dir = resolve_path(args.output_dir)
    if output_dir is None:
        raise ValueError("`--output-dir` is required.")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_cfg(args.config_name, args.task)
    model = build_model(cfg=cfg, args=args, device=device, dtype=dtype)

    if args.eval_mode == "rollout":
        if str(args.mujoco_gl) == "egl":
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(args.egl_device_id))
            print(
                " ".join(
                    [
                        "mujoco_rendering=egl",
                        f"import_device_id={egl_import_device_id}",
                        f"runtime_device_id={os.environ['MUJOCO_EGL_DEVICE_ID']}",
                        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
                    ]
                ),
                flush=True,
            )
        print("evaluation_protocol=libero_policy_rollout_three_modes", flush=True)
        print(f"action_checkpoint={resolve_path(args.action_checkpoint)}", flush=True)
        print(f"predictor_checkpoint={resolve_path(args.predictor_checkpoint)}", flush=True)
        print(f"vjepa_checkpoint={resolve_path(args.vjepa_checkpoint)}", flush=True)
        print(f"rollout_modes={parse_rollout_modes(args.rollout_modes)}", flush=True)
        print(f"oracle_future_mode={args.oracle_future_mode}", flush=True)
        print(f"seed={args.seed}", flush=True)
        start_time = time.time()
        rollout_results = run_rollout_eval(
            cfg=cfg,
            model=model,
            args=args,
            output_dir=output_dir,
        )
        rollout_results["args"] = vars(args)
        rollout_results["duration_sec"] = float(time.time() - start_time)
        results_path = output_dir / "policy_rollout_results.json"
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(rollout_results, f, indent=2, cls=NumpyEncoder)
        summary = rollout_results["summary"]
        print("rollout_summary:", flush=True)
        for mode, metrics in summary["modes"].items():
            print(
                f"  mode={mode} success_rate={metrics['success_rate']:.4f} "
                f"task_completion_rate={metrics['task_completion_rate']:.4f} "
                f"episode_return={metrics['episode_return']:.4f}",
                flush=True,
            )
        print(f"predictor_policy_gain={summary['predictor_policy_gain']:.4f}", flush=True)
        print(f"oracle_upper_bound={summary['oracle_upper_bound']:.4f}", flush=True)
        print(f"saved_results={results_path}", flush=True)
        return

    loader = build_loader(cfg, batch_size=args.batch_size, num_workers=args.num_workers)

    print("evaluation_protocol=offline_same_batch_three_modes", flush=True)
    print(f"action_checkpoint={resolve_path(args.action_checkpoint)}", flush=True)
    print(f"predictor_checkpoint={resolve_path(args.predictor_checkpoint)}", flush=True)
    print(f"vjepa_checkpoint={resolve_path(args.vjepa_checkpoint)}", flush=True)
    print(f"seed={args.seed}", flush=True)

    per_batch: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    modes = ("oracle", "predicted", "no_future")
    for batch_idx, sample in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        if not isinstance(sample, dict):
            raise ValueError(f"Expected dataloader batch dict, got {type(sample)}.")
        assert_batch_video(
            sample,
            current_frames=args.current_frame_count,
            future_frames=args.future_frame_count,
        )

        compare_seed = int(args.seed + 10_000 + batch_idx)
        results_by_mode = {
            mode: run_mode(
                model=model,
                sample=sample,
                mode=mode,
                seed=compare_seed,
                device=device,
            )
            for mode in modes
        }
        metrics = compute_metrics(results_by_mode)
        metric_rows.append(metrics)
        per_batch.append(
            {
                "batch_idx": batch_idx,
                "seed": compare_seed,
                "modes": results_by_mode,
                "metrics": metrics,
            }
        )
        print(
            " ".join(
                [
                    f"batch={batch_idx}",
                    f"oracle_loss_action={metrics['oracle_loss_action']:.6f}",
                    f"predicted_loss_action={metrics['predicted_loss_action']:.6f}",
                    f"no_future_loss_action={metrics['no_future_loss_action']:.6f}",
                    f"oracle_vs_predicted_gap={metrics['oracle_vs_predicted_gap']:.6f}",
                    f"predicted_delta_vs_oracle={metrics['predicted_action_loss_delta_vs_oracle']:.6f}",
                    f"no_future_delta_vs_oracle={metrics['no_future_action_loss_delta_vs_oracle']:.6f}",
                    f"predictor_value_score={metrics['predictor_value_score']:.6f}",
                    f"stability_gap={metrics['stability_gap']:.6f}",
                    f"predictor_contribution_score={metrics['predictor_contribution_score']:.6f}",
                ]
            ),
            flush=True,
        )

    if not metric_rows:
        raise RuntimeError("No batches were evaluated.")

    metric_keys = sorted(metric_rows[0].keys())
    summary = {key: mean_metric(metric_rows, key) for key in metric_keys}
    summary["num_batches"] = float(len(metric_rows))
    print("summary:", flush=True)
    for key in metric_keys:
        print(f"  {key}={summary[key]:.6f}", flush=True)

    results = {
        "args": vars(args),
        "summary": summary,
        "per_batch": per_batch,
    }
    output_path = output_dir / "predictor_value_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved_results={output_path}", flush=True)


if __name__ == "__main__":
    main()
