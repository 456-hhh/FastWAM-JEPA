from __future__ import annotations

# This script is based on train_fastwam_jepa_idm_v2_stage2_libero.py and is the
# v3 Stage5 entry point for z_task-to-ActionDiT integration.

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (SRC_ROOT, PROJECT_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam_jepa_runtime_guard import configure_runtime_stability
from fastwam.models.wan22.pairwise_stage1_compat import (
    PairwiseStage1TextActionCompatWrapper,
    load_stage1_text_action_checkpoint,
)
from fastwam.models.wan22.z_task_adapter import (
    ZTaskContextAdapter,
    append_z_task_to_context,
)
from fastwam.training.pairwise_joint_loss import (
    PairwiseJointLossWeights,
    combine_stage6_losses,
)


DEFAULT_VJEPA_REPO = "/data1/Johnny/challenge/dd/FastWAM_jepa/external/vjepa2"
DEFAULT_VJEPA_CHECKPOINT = "/data1/Johnny/challenge/dd/FastWAM_jepa/checkpoints/vjepa2/vitg.pt"
DEFAULT_STAGE1_DIR = (
    "runs/v2_stage1_predictor_mixed_fast_robotwinfull_24layer_ddp4_0123_"
    "pergpu32_global128_h1024_10k"
)
DEFAULT_ACTION_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM/runs/libero_idm_2cam224_1e-4/"
    "2026-05-18_01-22-59/checkpoints/weights/step_020000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastWAM-JEPA-IDM v3 Stage5 z_task-to-action training."
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument("--stage1-checkpoint", default="auto")
    parser.add_argument("--action-checkpoint", default=DEFAULT_ACTION_CHECKPOINT)
    parser.add_argument("--vjepa-repo", default=DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-checkpoint", default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", default="-1_1", choices=["-1_1", "0_1"])
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--current-frame-count", type=int, default=4)
    parser.add_argument("--future-frame-count", type=int, default=4)
    parser.add_argument("--num-future-tokens", default="auto")
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--future-predictor-layers", type=int, default=24)
    parser.add_argument("--future-predictor-hidden-dim", type=int, default=1024)
    parser.add_argument("--future-predictor-heads", type=int, default=16)
    parser.add_argument("--adapter-current-tokens", type=int, default=64)
    parser.add_argument("--adapter-future-tokens", type=int, default=64)
    parser.add_argument("--use-proprio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda-future", type=float, default=0.1)
    parser.add_argument("--lambda-cos", type=float, default=0.0)
    parser.add_argument(
        "--use-z-task-token",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Stage5 z_task token injection into ActionDiT context.",
    )
    parser.add_argument(
        "--stage1-text-action-checkpoint",
        default=None,
        help="Checkpoint from v3 Stage1 text-action latent training. Required when --use-z-task-token is enabled.",
    )
    parser.add_argument(
        "--freeze-pairwise-latent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the Stage1 pairwise latent wrapper when injecting z_task.",
    )
    parser.add_argument(
        "--z-task-dim",
        type=int,
        default=1024,
        help="Latent dimension of z_task / z_task_token.",
    )
    parser.add_argument(
        "--z-task-gate-init",
        type=float,
        default=-4.0,
        help="Initial logit for the learnable z_task context gate.",
    )
    parser.add_argument(
        "--lambda-vlp-to-a",
        type=float,
        default=0.0,
        help="Weight for pairwise VLP/text-to-action latent loss in Stage6.",
    )
    parser.add_argument(
        "--lambda-va-to-l",
        type=float,
        default=0.0,
        help="Weight for pairwise VA-to-language latent loss in Stage6.",
    )
    parser.add_argument(
        "--pairwise-tau",
        type=float,
        default=0.07,
        help="Temperature for the Stage1 text-action compatibility wrapper.",
    )
    parser.add_argument("--lr-adapter", type=float, default=1.0e-4)
    parser.add_argument("--lr-proprio", type=float, default=1.0e-4)
    parser.add_argument("--lr-predictor", type=float, default=2.0e-5)
    parser.add_argument("--lr-action", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--freeze-vjepa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output-dir", default="runs/fastwam_jepa_idm_v2_stage2_libero")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-rank0-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-wsl-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--runtime-log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--runtime-log-path", default=None)
    parser.add_argument("--runtime-log-max-mb", type=int, default=100)
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


def resolve_stage1_checkpoint(path_value: str | Path | None) -> Path:
    if path_value is None or str(path_value).strip().lower() in {"", "auto"}:
        base = resolve_path(DEFAULT_STAGE1_DIR)
        candidates = [
            base / "checkpoint_step_010000.pt",
            base / "checkpoint_step_008000.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not auto-resolve Stage 1 checkpoint. Tried: "
            + ", ".join(str(path) for path in candidates)
        )
    return require_file(path_value, name="--stage1-checkpoint")


def init_distributed_from_env() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP mode requires CUDA because this script uses NCCL.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return True, world_size, rank, local_rank, torch.device(f"cuda:{local_rank}")
    return False, world_size, rank, local_rank, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_rank0(rank: int) -> bool:
    return int(rank) == 0


def rank0_print(rank: int, *args: Any, **kwargs: Any) -> None:
    if is_rank0(rank):
        print(*args, **kwargs)


def unwrap_ddp(module: torch.nn.Module | None) -> torch.nn.Module | None:
    if module is None:
        return None
    return module.module if isinstance(module, DDP) else module


def precision_to_dtype(precision: str, device: torch.device) -> tuple[torch.dtype, torch.dtype | None]:
    if precision == "fp32" or device.type != "cuda":
        return torch.float32, None
    if precision == "fp16":
        return torch.float16, torch.float16
    if precision == "bf16":
        return torch.bfloat16, torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def compose_cfg(config_name: str, task: str) -> DictConfig:
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir = (PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=[f"task={task}"])


def resolve_libero_dataset_dirs(cfg: DictConfig, *, override_root: str | None, rank: int) -> list[str]:
    if override_root is not None:
        root = require_dir(override_root, name="--libero-data-root")
        if (root / "meta").exists() or (root / "data").exists():
            paths = [root]
        else:
            paths = sorted(path for path in root.iterdir() if path.is_dir() and path.name.endswith("_lerobot"))
            if not paths:
                raise FileNotFoundError(
                    f"--libero-data-root did not contain *_lerobot dataset dirs: {root}"
                )
    else:
        raw_dirs = list(cfg.data.train.dataset_dirs)
        paths = []
        for item in raw_dirs:
            path = Path(str(item))
            paths.append(path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve())
    for path in paths:
        rank0_print(rank, f"resolved_libero_dataset_dir={path}", flush=True)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"LIBERO dataset dir does not exist: {path}")
        if "robotwin" in path.as_posix().lower():
            raise ValueError(f"Stage 2 is LIBERO-only; got RoboTwin-like dataset path: {path}")
    cfg.data.train.dataset_dirs = [str(path) for path in paths]
    return [str(path) for path in paths]


def build_libero_loader(
    cfg: DictConfig,
    *,
    args: argparse.Namespace,
    ddp_enabled: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    resolve_libero_dataset_dirs(cfg, override_root=args.libero_data_root, rank=rank)
    dataset = instantiate(cfg.data.train)
    rank0_print(rank, f"libero_dataset_type={type(dataset).__name__} len={len(dataset)}", flush=True)
    sampler = None
    shuffle = True
    if ddp_enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(args.seed),
            drop_last=True,
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader, sampler


def _parse_auto_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "auto", "none", "unset", "0"}:
            return None
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer or 'auto', got {value!r}.") from exc
    else:
        parsed = int(value)
    return None if parsed <= 0 else int(parsed)


def provisional_num_tokens(args: argparse.Namespace) -> int:
    return _parse_auto_int(args.num_future_tokens, name="--num-future-tokens") or 256


def resize_video(video: torch.Tensor, *, size: int) -> torch.Tensor:
    if int(video.shape[-1]) == int(size) and int(video.shape[-2]) == int(size):
        return video
    bsz, channels, frames, height, width = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, height, width)
    flat = F.interpolate(flat.float(), size=(int(size), int(size)), mode="bilinear", align_corners=False)
    return flat.to(dtype=video.dtype).reshape(bsz, frames, channels, int(size), int(size)).permute(0, 2, 1, 3, 4)


def canonicalize_libero_batch(
    batch: dict[str, Any],
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if not isinstance(batch, dict):
        raise ValueError(f"Expected LIBERO dataloader batch dict, got {type(batch)}.")
    video = batch["video"]
    if not torch.is_tensor(video) or video.ndim != 5 or int(video.shape[1]) != 3:
        raise ValueError(f"LIBERO batch video must be [B, 3, T, H, W], got {tuple(video.shape)}.")
    current_t = int(args.current_frame_count)
    future_t = int(args.future_frame_count)
    total_t = current_t + future_t
    if int(video.shape[2]) < total_t:
        raise ValueError(
            f"LIBERO video has T={video.shape[2]}, but Stage 2 requires at least {total_t} frames."
        )
    current_video = video[:, :, :current_t]
    future_video = video[:, :, current_t:total_t]
    current_video = resize_video(current_video, size=int(args.vjepa_img_size))
    future_video = resize_video(future_video, size=int(args.vjepa_img_size))
    if tuple(current_video.shape[1:]) != (3, current_t, int(args.vjepa_img_size), int(args.vjepa_img_size)):
        raise ValueError(f"current_video shape check failed: {tuple(current_video.shape)}")
    if tuple(future_video.shape[1:]) != (3, future_t, int(args.vjepa_img_size), int(args.vjepa_img_size)):
        raise ValueError(f"future_video shape check failed: {tuple(future_video.shape)}")

    action = batch["action"]
    if not torch.is_tensor(action) or action.ndim != 3 or int(action.shape[-1]) != 7:
        raise ValueError(f"Stage 2 LIBERO action labels must be [B, T_a, 7], got {tuple(action.shape)}.")
    context = batch["context"]
    context_mask = batch["context_mask"]
    if not torch.is_tensor(context) or not torch.is_tensor(context_mask):
        raise ValueError("LIBERO batch must contain tensor context/context_mask.")
    if context.ndim != 3 or context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
        raise ValueError(
            f"context/context_mask must be [B, L, D]/[B, L], got {tuple(context.shape)} and {tuple(context_mask.shape)}."
        )

    result: dict[str, Any] = {
        "video": current_video.to(device=device, dtype=dtype, non_blocking=True),
        "future_video": future_video.to(device=device, dtype=dtype, non_blocking=True),
        "action": action.to(device=device, dtype=dtype, non_blocking=True),
        "context": context.to(device=device, dtype=dtype, non_blocking=True),
        "context_mask": context_mask.to(device=device, dtype=torch.bool, non_blocking=True),
        "source_name": "libero",
    }
    action_is_pad = batch.get("action_is_pad")
    if torch.is_tensor(action_is_pad):
        if tuple(action_is_pad.shape) != tuple(action.shape[:2]):
            raise ValueError(
                f"action_is_pad must be [B, T_a], got {tuple(action_is_pad.shape)} vs {tuple(action.shape[:2])}."
            )
        result["action_is_pad"] = action_is_pad.to(device=device, dtype=torch.bool, non_blocking=True)
    if args.use_proprio:
        proprio = batch.get("proprio")
        if not torch.is_tensor(proprio):
            raise ValueError("--use-proprio is enabled, but LIBERO batch has no proprio tensor.")
        if proprio.ndim == 3:
            current_proprio = proprio[:, 0, :]
        elif proprio.ndim == 2:
            current_proprio = proprio
        else:
            raise ValueError(f"proprio must be [B, T, D] or [B, D], got {tuple(proprio.shape)}.")
        result["proprio"] = current_proprio.to(device=device, dtype=dtype, non_blocking=True)
    return result


def _state_dict_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key in ("mot", "model", "model_state_dict", "state_dict", "module"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    candidates.append(payload)
    return candidates


def strip_prefixes(state_dict: dict[str, Any], prefixes: tuple[str, ...] = ("module.",)) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


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
                if isinstance(key, str) and key.startswith(prefix) and torch.is_tensor(value)
            }
            if action_state:
                return strip_prefixes(action_state)
    raise ValueError("Could not find ActionDiT weights in checkpoint.")


def extract_nested_state(
    payload: dict[str, Any],
    *,
    direct_keys: tuple[str, ...],
    model_prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor] | None:
    for key in direct_keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return strip_prefixes({k: v for k, v in value.items() if torch.is_tensor(v)})
    for state_key in ("model", "model_state_dict", "state_dict", "module"):
        state = payload.get(state_key)
        if not isinstance(state, dict):
            continue
        stripped = strip_prefixes(state)
        for prefix in model_prefixes:
            filtered = {
                key[len(prefix) :]: value
                for key, value in stripped.items()
                if isinstance(key, str) and key.startswith(prefix) and torch.is_tensor(value)
            }
            if filtered:
                return filtered
    return None


def load_shape_matching(
    module: torch.nn.Module,
    state_dict: dict[str, Any] | None,
    *,
    name: str,
    rank: int,
    required: bool,
) -> dict[str, Any]:
    if not state_dict:
        if required:
            raise ValueError(f"No state_dict found for {name}.")
        rank0_print(rank, f"{name}_load skipped: no compatible state_dict found", flush=True)
        return {"loaded_keys_count": 0, "skipped_keys_count": 0, "missing_keys": [], "unexpected_keys": []}
    own_state = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    shape_mismatch: list[dict[str, Any]] = []
    for key, value in strip_prefixes(state_dict).items():
        if not torch.is_tensor(value) or key not in own_state:
            skipped.append(str(key))
            continue
        if tuple(value.shape) != tuple(own_state[key].shape):
            skipped.append(str(key))
            shape_mismatch.append(
                {
                    "key": str(key),
                    "source_shape": tuple(value.shape),
                    "target_shape": tuple(own_state[key].shape),
                }
            )
            continue
        matched[str(key)] = value.to(dtype=own_state[key].dtype)
    missing, unexpected = module.load_state_dict(matched, strict=False)
    stats = {
        "loaded_keys_count": len(matched),
        "skipped_keys_count": len(skipped),
        "shape_mismatch_count": len(shape_mismatch),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "shape_mismatch": shape_mismatch[:20],
    }
    rank0_print(
        rank,
        f"{name}_load loaded_keys={len(matched)} skipped_keys={len(skipped)} "
        f"shape_mismatch={len(shape_mismatch)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if shape_mismatch:
        rank0_print(rank, f"{name}_shape_mismatch_first20={shape_mismatch[:20]}", flush=True)
    if required and not matched:
        raise RuntimeError(f"{name} checkpoint did not load any compatible keys.")
    return stats


def checkpoint_config(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("stage1_temporal_config", "config", "args"):
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def validate_stage1_config(stage1_payload: dict[str, Any], args: argparse.Namespace, *, rank: int) -> None:
    cfg = checkpoint_config(stage1_payload)
    rank0_print(rank, f"stage1_checkpoint_config={cfg}", flush=True)
    comparisons = {
        "current_frame_count": int(args.current_frame_count),
        "future_frame_count": int(args.future_frame_count),
        "resolved_num_future_tokens": int(args.resolved_num_future_tokens),
        "future_predictor_layers": int(args.future_predictor_layers),
        "future_predictor_hidden_dim": int(args.future_predictor_hidden_dim),
        "future_predictor_heads": int(args.future_predictor_heads),
    }
    for key, expected in comparisons.items():
        got = cfg.get(key)
        if got is None:
            rank0_print(rank, f"WARNING stage1 checkpoint missing config key {key}", flush=True)
            continue
        if int(got) != int(expected):
            raise ValueError(f"Stage 1 checkpoint config mismatch for {key}: checkpoint={got}, expected={expected}.")


def build_action_expert(
    *,
    action_cfg: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    from fastwam.models.wan22.action_dit import ActionDiT

    action_expert = ActionDiT(**action_cfg)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Action checkpoint payload must be dict, got {type(payload)}.")
    action_state = extract_action_state_dict(payload)
    stats = load_shape_matching(action_expert, action_state, name="action_dit", rank=rank, required=True)
    return action_expert.to(device=device, dtype=dtype), stats


def build_vjepa_encoder(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    from fastwam.models.vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper

    encoder = VJepaEncoderWrapper(
        dummy=False,
        model_name=str(args.vjepa_model_name),
        external_repo_path=str(require_dir(args.vjepa_repo, name="--vjepa-repo")),
        checkpoint_path=str(require_file(args.vjepa_checkpoint, name="--vjepa-checkpoint")),
        pretrained=False,
        vjepa_dim=int(args.vjepa_dim),
        num_tokens=provisional_num_tokens(args),
        freeze=True,
        normalize_tokens=False,
        img_size=int(args.vjepa_img_size),
        input_range=str(args.vjepa_input_range),
        tubelet_size=int(args.vjepa_tubelet_size),
        frame_encoding_mode="clip_or_repeat",
    ).to(device=device, dtype=dtype)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def infer_token_counts(
    *,
    vjepa_encoder: torch.nn.Module,
    first_batch: dict[str, Any],
    args: argparse.Namespace,
    rank: int,
) -> tuple[int, int, int]:
    with torch.no_grad():
        current_tokens = vjepa_encoder(first_batch["video"])
        future_tokens = vjepa_encoder(first_batch["future_video"])
    current_count = int(current_tokens.shape[1])
    future_count = int(future_tokens.shape[1])
    if current_count != future_count:
        raise ValueError(
            "RoPE full-transformer predictor requires current/future token counts to match, "
            f"got current={current_count}, future={future_count}."
        )
    requested = _parse_auto_int(args.num_future_tokens, name="--num-future-tokens")
    resolved = future_count if requested is None else int(requested)
    if resolved != future_count:
        raise ValueError(
            f"--num-future-tokens={resolved} does not match V-JEPA future token count={future_count}. "
            "Use --num-future-tokens auto."
        )
    args.resolved_num_future_tokens = int(resolved)
    rank0_print(rank, f"current_frame_count={args.current_frame_count}", flush=True)
    rank0_print(rank, f"future_frame_count={args.future_frame_count}", flush=True)
    rank0_print(rank, f"current_token_count={current_count}", flush=True)
    rank0_print(rank, f"future_token_count={future_count}", flush=True)
    rank0_print(rank, f"resolved_num_future_tokens={resolved}", flush=True)
    if int(args.current_frame_count) == 4 and current_count != 512:
        raise ValueError(f"Expected 512 current V-JEPA tokens for 4 frames, got {current_count}.")
    if int(args.future_frame_count) == 4 and future_count != 512:
        raise ValueError(f"Expected 512 future V-JEPA tokens for 4 frames, got {future_count}.")
    return current_count, future_count, int(resolved)


def build_model(
    *,
    cfg: DictConfig,
    args: argparse.Namespace,
    vjepa_encoder: torch.nn.Module,
    action_expert: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from fastwam.models.vjepa.jepa_fastwam_adapter import JepaToFastWAMAdapter
    from fastwam.models.vjepa.jepa_future_predictor import JepaFuturePredictor
    from fastwam.models.wan22.fastwam_jepa_idm import FastWAMJEPAIDM

    action_cfg = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
    scheduler_cfg = OmegaConf.to_container(cfg.model.action_scheduler, resolve=True)
    text_dim = int(action_cfg["text_dim"])
    proprio_dim = int(OmegaConf.select(cfg, "model.proprio_dim")) if args.use_proprio else None
    future_predictor = JepaFuturePredictor(
        vjepa_dim=int(args.vjepa_dim),
        hidden_dim=int(args.future_predictor_hidden_dim),
        num_future_tokens=int(args.resolved_num_future_tokens),
        text_dim=text_dim,
        num_layers=int(args.future_predictor_layers),
        num_heads=int(args.future_predictor_heads),
    )
    adapter = JepaToFastWAMAdapter(
        vjepa_dim=int(args.vjepa_dim),
        text_dim=text_dim,
        num_current_context_tokens=int(args.adapter_current_tokens),
        num_future_context_tokens=int(args.adapter_future_tokens),
    )
    model = FastWAMJEPAIDM(
        action_expert=action_expert,
        vjepa_encoder=vjepa_encoder,
        future_predictor=future_predictor,
        jepa_adapter=adapter,
        action_dim=int(action_cfg["action_dim"]),
        hidden_dim=int(action_cfg["hidden_dim"]),
        vjepa_dim=int(args.vjepa_dim),
        num_future_tokens=int(args.resolved_num_future_tokens),
        text_dim=text_dim,
        proprio_dim=proprio_dim,
        device=None,
        torch_dtype=dtype,
        action_train_shift=float(scheduler_cfg.get("train_shift", 5.0)),
        action_infer_shift=float(scheduler_cfg.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(scheduler_cfg.get("num_train_timesteps", 1000)),
        lambda_action=1.0,
        lambda_future=float(args.lambda_future),
        current_frame_count=int(args.current_frame_count),
        future_frame_count=int(args.future_frame_count),
        adapter_current_tokens=int(args.adapter_current_tokens),
        adapter_future_tokens=int(args.adapter_future_tokens),
        future_predictor_layers=int(args.future_predictor_layers),
        future_predictor_heads=int(args.future_predictor_heads),
        future_source="predicted",
    )
    if bool(args.use_z_task_token):
        if args.stage1_text_action_checkpoint is None:
            raise ValueError("--stage1-text-action-checkpoint is required when --use-z-task-token is enabled.")
        if int(args.z_task_dim) != 1024:
            raise ValueError(
                "Current Stage1 text-action compatibility wrapper outputs z_task_dim=1024; "
                f"got --z-task-dim={args.z_task_dim}."
            )
        if float(args.pairwise_tau) <= 0.0:
            raise ValueError(f"--pairwise-tau must be positive, got {args.pairwise_tau}.")

        pairwise_latent = PairwiseStage1TextActionCompatWrapper(
            tau=float(args.pairwise_tau),
        )
        load_stats = load_stage1_text_action_checkpoint(
            pairwise_latent,
            args.stage1_text_action_checkpoint,
            strict=True,
        )
        setattr(model, "pairwise_latent", pairwise_latent)
        setattr(model, "pairwise_latent_load_stats", load_stats)

        z_task_adapter = ZTaskContextAdapter(
            z_task_dim=int(args.z_task_dim),
            context_dim=text_dim,
            gate_init=float(args.z_task_gate_init),
        )
        setattr(model, "z_task_adapter", z_task_adapter)
    else:
        setattr(model, "pairwise_latent", None)
        setattr(model, "pairwise_latent_load_stats", {})
        setattr(model, "z_task_adapter", None)
    return model.to(device=device, dtype=dtype)


def load_stage1_checkpoint(model: torch.nn.Module, checkpoint_path: Path, *, args: argparse.Namespace, rank: int) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Stage 1 checkpoint payload must be dict, got {type(payload)}.")
    validate_stage1_config(payload, args, rank=rank)
    stats = {
        "checkpoint": str(checkpoint_path),
        "future_predictor": load_shape_matching(
            model.future_predictor,
            extract_nested_state(
                payload,
                direct_keys=("future_predictor", "predictor", "predictor_state_dict"),
                model_prefixes=("future_predictor.", "module.future_predictor."),
            ),
            name="stage1_future_predictor",
            rank=rank,
            required=True,
        ),
        "jepa_adapter": load_shape_matching(
            model.jepa_adapter,
            extract_nested_state(
                payload,
                direct_keys=("jepa_adapter", "adapter", "adapter_state_dict"),
                model_prefixes=("jepa_adapter.", "module.jepa_adapter."),
            ),
            name="stage1_jepa_adapter",
            rank=rank,
            required=False,
        ),
    }
    if model.proprio_encoder is not None:
        stats["proprio_encoder"] = load_shape_matching(
            model.proprio_encoder,
            extract_nested_state(
                payload,
                direct_keys=("proprio_encoder", "proprio_projection"),
                model_prefixes=("proprio_encoder.", "module.proprio_encoder."),
            ),
            name="stage1_proprio_encoder",
            rank=rank,
            required=False,
        )
    return stats


def configure_trainability(model: torch.nn.Module, *, args: argparse.Namespace) -> list[dict[str, Any]]:
    model.vjepa_encoder.requires_grad_(False)
    model.future_predictor.requires_grad_(True)
    model.jepa_adapter.requires_grad_(True)
    model.action_expert.requires_grad_(True)
    if model.proprio_encoder is not None:
        model.proprio_encoder.requires_grad_(True)
    z_task_adapter = getattr(model, "z_task_adapter", None)
    if z_task_adapter is not None:
        z_task_adapter.requires_grad_(True)
    pairwise_latent = getattr(model, "pairwise_latent", None)
    if pairwise_latent is not None:
        if bool(args.freeze_pairwise_latent):
            pairwise_latent.requires_grad_(False)
        else:
            pairwise_latent.requires_grad_(True)
    groups = [
        {"name": "adapter", "params": [p for p in model.jepa_adapter.parameters() if p.requires_grad]},
        {"name": "future_predictor", "params": [p for p in model.future_predictor.parameters() if p.requires_grad]},
        {"name": "action_dit", "params": [p for p in model.action_expert.parameters() if p.requires_grad]},
    ]
    if model.proprio_encoder is not None:
        groups.append({"name": "proprio", "params": [p for p in model.proprio_encoder.parameters() if p.requires_grad]})
    if z_task_adapter is not None:
        groups.append({"name": "z_task_adapter", "params": [p for p in z_task_adapter.parameters() if p.requires_grad]})
    if pairwise_latent is not None and not bool(args.freeze_pairwise_latent):
        groups.append({"name": "pairwise_latent", "params": [p for p in pairwise_latent.parameters() if p.requires_grad]})
    return groups


def future_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()


def stage2_forward_loss(model: torch.nn.Module, sample: dict[str, Any], *, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    module = unwrap_ddp(model)
    if module is None:
        raise RuntimeError("DDP unwrap failed.")
    video = sample["video"]
    future_video = sample["future_video"]
    action = sample["action"]
    context = sample["context"]
    context_mask = sample["context_mask"]
    proprio = sample.get("proprio")
    if tuple(video.shape[1:]) != (3, int(args.current_frame_count), int(args.vjepa_img_size), int(args.vjepa_img_size)):
        raise ValueError(f"current_video shape must be [B,3,4,256,256], got {tuple(video.shape)}.")
    if tuple(future_video.shape[1:]) != (3, int(args.future_frame_count), int(args.vjepa_img_size), int(args.vjepa_img_size)):
        raise ValueError(f"future_video shape must be [B,3,4,256,256], got {tuple(future_video.shape)}.")
    if action.ndim != 3 or int(action.shape[-1]) != 7:
        raise ValueError(f"LIBERO action must be [B, T_a, 7], got {tuple(action.shape)}.")
    if module.proprio_encoder is not None:
        if proprio is None or proprio.ndim != 2:
            raise ValueError(
                "Stage 2 proprio must be current-only [B, D_p]; future proprio is not allowed. "
                f"Got {None if proprio is None else tuple(proprio.shape)}."
            )

    condition_context, condition_mask = module._append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=proprio,
    )
    current_jepa_tokens = module._encode_jepa_video(video)
    target_future_jepa_tokens = module._encode_jepa_video(future_video).detach()
    if int(current_jepa_tokens.shape[1]) != int(args.resolved_num_future_tokens):
        raise ValueError(
            f"current_token_count={current_jepa_tokens.shape[1]} does not match resolved_num_future_tokens={args.resolved_num_future_tokens}."
        )
    if tuple(current_jepa_tokens.shape) != tuple(target_future_jepa_tokens.shape):
        raise ValueError(
            f"current/future V-JEPA token shapes must match, got {tuple(current_jepa_tokens.shape)} vs {tuple(target_future_jepa_tokens.shape)}."
        )

    future_out = module.future_predictor(
        current_jepa_tokens=current_jepa_tokens,
        condition_context=condition_context,
        condition_mask=condition_mask,
    )
    pred_future_jepa_tokens = future_out["pred_future_tokens"]
    if tuple(pred_future_jepa_tokens.shape) != tuple(target_future_jepa_tokens.shape):
        raise ValueError(
            "pred_future_jepa_tokens shape mismatch, "
            f"got {tuple(pred_future_jepa_tokens.shape)} vs {tuple(target_future_jepa_tokens.shape)}."
        )
    action_context, action_context_mask = module.jepa_adapter(
        current_jepa_tokens=current_jepa_tokens,
        future_jepa_tokens=pred_future_jepa_tokens,
        base_context=condition_context,
        base_context_mask=condition_mask,
    )
    pairwise_out: dict[str, torch.Tensor] = {}
    z_task_context_token = None
    z_task_gate_value = None

    if getattr(module, "pairwise_latent", None) is not None and getattr(module, "z_task_adapter", None) is not None:
        pairwise_out = module.pairwise_latent.forward_train(
            world_tokens=current_jepa_tokens,
            text_tokens=context,
            text_mask=context_mask,
            proprio=proprio,
            action_chunk=action,
            tau=float(args.pairwise_tau),
        )

        z_task_token = pairwise_out.get("z_task_token")
        if z_task_token is None:
            raise RuntimeError("pairwise_latent.forward_train(...) must return `z_task_token`.")

        z_task_context_token = module.z_task_adapter(z_task_token)
        action_context, action_context_mask = append_z_task_to_context(
            context=action_context,
            context_mask=action_context_mask,
            z_task_context_token=z_task_context_token,
        )
        z_task_gate_value = module.z_task_adapter.gate().detach()

    batch_size = int(action.shape[0])
    noise_action = torch.randn_like(action)
    timestep_action = module.train_action_scheduler.sample_training_t(
        batch_size=batch_size,
        device=action.device,
        dtype=action.dtype,
    )
    noisy_action = module.train_action_scheduler.add_noise(action, noise_action, timestep_action)
    target_action = module.train_action_scheduler.training_target(action, noise_action, timestep_action)
    pred_action = module.action_expert(
        action_tokens=noisy_action,
        timestep=timestep_action,
        context=action_context,
        context_mask=action_context_mask,
    )
    action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
    action_is_pad = sample.get("action_is_pad")
    if action_is_pad is not None:
        valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    else:
        action_loss_per_sample = action_loss_token.mean(dim=1)
    action_weight = module.train_action_scheduler.training_weight(timestep_action).to(
        device=action_loss_per_sample.device,
        dtype=action_loss_per_sample.dtype,
    )
    loss_action = (action_loss_per_sample * action_weight).mean()
    loss_future_l1 = F.l1_loss(pred_future_jepa_tokens.float(), target_future_jepa_tokens.float())
    loss_future_cos = future_cosine_loss(pred_future_jepa_tokens, target_future_jepa_tokens)
    loss_future_jepa = loss_future_l1 + float(args.lambda_cos) * loss_future_cos
    loss_total, joint_loss_items = combine_stage6_losses(
        loss_action=loss_action,
        loss_future_jepa=loss_future_jepa,
        loss_vlp_to_a=pairwise_out.get("loss_vlp_to_a") if pairwise_out else None,
        loss_va_to_l=pairwise_out.get("loss_va_to_l") if pairwise_out else None,
        weights=PairwiseJointLossWeights(
            lambda_future=float(args.lambda_future),
            lambda_vlp_to_a=float(args.lambda_vlp_to_a),
            lambda_va_to_l=float(args.lambda_va_to_l),
        ),
    )
    module.last_forward_shapes = {
        "current_jepa_tokens": tuple(current_jepa_tokens.shape),
        "target_future_jepa_tokens": tuple(target_future_jepa_tokens.shape),
        "pred_future_jepa_tokens": tuple(pred_future_jepa_tokens.shape),
        "action_context": tuple(action_context.shape),
        "pred_action": tuple(pred_action.shape),
        "proprio": None if proprio is None else tuple(proprio.shape),
        "use_z_task_token": bool(pairwise_out),
        "z_task_token": tuple(pairwise_out["z_task_token"].shape) if pairwise_out and "z_task_token" in pairwise_out else None,
        "z_task_context_token": tuple(z_task_context_token.shape) if z_task_context_token is not None else None,
        "z_task_context_tokens_appended": int(z_task_context_token.shape[1]) if z_task_context_token is not None else 0,
        "action_context_after_z_task": tuple(action_context.shape),
    }
    loss_dict = {
        "loss_total": loss_total.detach(),
        "loss_action": loss_action.detach(),
        "loss_future_jepa": loss_future_jepa.detach(),
        "loss_future_l1": loss_future_l1.detach(),
        "loss_future_cos": loss_future_cos.detach(),
        "loss_vlp_to_a": joint_loss_items["loss_vlp_to_a"],
        "loss_va_to_l": joint_loss_items["loss_va_to_l"],
    }
    if pairwise_out and "retrieval_acc_vlp_to_a" in pairwise_out:
        loss_dict["retrieval_acc_vlp_to_a"] = pairwise_out["retrieval_acc_vlp_to_a"].detach()
    if z_task_gate_value is not None:
        loss_dict["z_task_gate"] = z_task_gate_value.detach()
    return loss_total, loss_dict


def reduce_loss_dict(loss_dict: dict[str, torch.Tensor], *, ddp_enabled: bool, world_size: int) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for key, value in loss_dict.items():
        tensor = value.detach().float()
        if ddp_enabled:
            tensor = tensor.clone()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / float(world_size)
        reduced[key] = float(tensor.item())
    return reduced


def trainable_count(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def total_count(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters())


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    loss_dict: dict[str, float],
    load_stats: dict[str, Any],
) -> Path:
    module = unwrap_ddp(model)
    if module is None:
        raise RuntimeError("Cannot save an empty model.")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_step_{int(step):06d}.pt"
    payload = {
        "model": module.state_dict(),
        "future_predictor": module.future_predictor.state_dict(),
        "jepa_adapter": module.jepa_adapter.state_dict(),
        "proprio_encoder": None if module.proprio_encoder is None else module.proprio_encoder.state_dict(),
        "action_expert": module.action_expert.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "args": vars(args),
        "config": {
            "current_frame_count": int(args.current_frame_count),
            "future_frame_count": int(args.future_frame_count),
            "resolved_num_future_tokens": int(args.resolved_num_future_tokens),
            "future_predictor_layers": int(args.future_predictor_layers),
            "future_predictor_hidden_dim": int(args.future_predictor_hidden_dim),
            "future_predictor_heads": int(args.future_predictor_heads),
            "vjepa_img_size": int(args.vjepa_img_size),
            "use_proprio": bool(args.use_proprio),
            "action_dim": 7,
        },
        "loss_dict": dict(loss_dict),
        "load_stats": load_stats,
    }
    if getattr(module, "pairwise_latent", None) is not None:
        payload["pairwise_latent"] = module.pairwise_latent.state_dict()
    if getattr(module, "z_task_adapter", None) is not None:
        payload["z_task_adapter"] = module.z_task_adapter.state_dict()
    if getattr(module, "pairwise_latent_load_stats", None) is not None:
        payload["pairwise_latent_load_stats"] = getattr(module, "pairwise_latent_load_stats")
    torch.save(payload, path)
    return path


def main() -> None:
    args = parse_args()
    ddp_enabled, world_size, rank, local_rank, device = init_distributed_from_env()
    args._rank = rank
    args._world_size = world_size
    args._local_rank = local_rank
    args._rank_seed = int(args.seed) + rank * 100003
    configure_runtime_stability(
        disable_wsl_fallback=bool(args.disable_wsl_fallback),
        log_level=str(args.runtime_log_level),
        log_path=args.runtime_log_path,
        max_log_mb=int(args.runtime_log_max_mb),
    )
    if int(args.current_frame_count) != 4 or int(args.future_frame_count) != 4:
        raise ValueError("Stage 2 v2 currently requires current_frame_count=4 and future_frame_count=4.")
    if int(args.vjepa_img_size) != 256:
        raise ValueError("Stage 2 v2 expects V-JEPA img_size=256.")
    if float(args.lambda_future) < 0.0 or float(args.lambda_cos) < 0.0:
        raise ValueError("--lambda-future and --lambda-cos must be non-negative.")
    if not bool(args.freeze_vjepa):
        raise ValueError("Stage 2 must keep V-JEPA2 frozen. Do not pass --no-freeze-vjepa.")
    if not bool(args.use_z_task_token):
        if float(args.lambda_vlp_to_a) != 0.0 or float(args.lambda_va_to_l) != 0.0:
            raise ValueError(
                "--lambda-vlp-to-a / --lambda-va-to-l require --use-z-task-token in this Stage5 script."
            )
    if bool(args.use_z_task_token) and args.stage1_text_action_checkpoint is None:
        raise ValueError("--stage1-text-action-checkpoint is required when --use-z-task-token is enabled.")
    seed_everything(int(args._rank_seed))
    param_dtype, autocast_dtype = precision_to_dtype(str(args.precision), device)

    cfg = compose_cfg(str(args.config_name), str(args.task))
    if "robotwin" in str(args.task).lower():
        raise ValueError(f"Stage 2 is LIBERO-only, got task={args.task!r}.")
    loader, sampler = build_libero_loader(
        cfg,
        args=args,
        ddp_enabled=ddp_enabled,
        rank=rank,
        world_size=world_size,
    )
    data_iter = iter(loader)
    raw_first = next(data_iter)
    first_batch = canonicalize_libero_batch(raw_first, args=args, device=device, dtype=param_dtype)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):
        raise ValueError("cfg.model must resolve to a dict.")
    action_cfg = dict(model_cfg["action_dit_config"])
    if int(action_cfg["action_dim"]) != 7:
        raise ValueError(f"LIBERO action_dim must be 7, got {action_cfg['action_dim']}.")
    action_checkpoint = require_file(args.action_checkpoint, name="--action-checkpoint")
    stage1_checkpoint = resolve_stage1_checkpoint(args.stage1_checkpoint)

    vjepa_encoder = build_vjepa_encoder(args, device=device, dtype=param_dtype)
    current_token_count, future_token_count, resolved_tokens = infer_token_counts(
        vjepa_encoder=vjepa_encoder,
        first_batch=first_batch,
        args=args,
        rank=rank,
    )
    action_expert, action_load_stats = build_action_expert(
        action_cfg=action_cfg,
        checkpoint_path=action_checkpoint,
        device=device,
        dtype=param_dtype,
        rank=rank,
    )
    model = build_model(
        cfg=cfg,
        args=args,
        vjepa_encoder=vjepa_encoder,
        action_expert=action_expert,
        device=device,
        dtype=param_dtype,
    )
    stage1_load_stats = load_stage1_checkpoint(model, stage1_checkpoint, args=args, rank=rank)
    groups = configure_trainability(model, args=args)
    if any(param.requires_grad for param in model.vjepa_encoder.parameters()):
        raise RuntimeError("V-JEPA2 encoder is not frozen.")
    param_groups = []
    lr_by_name = {
        "adapter": float(args.lr_adapter),
        "proprio": float(args.lr_proprio),
        "future_predictor": float(args.lr_predictor),
        "action_dit": float(args.lr_action),
        "z_task_adapter": float(args.lr_adapter),
        "pairwise_latent": float(args.lr_adapter),
    }
    for group in groups:
        params = list(group["params"])
        if params:
            param_groups.append(
                {
                    "params": params,
                    "lr": lr_by_name[str(group["name"])],
                    "weight_decay": float(args.weight_decay),
                    "name": str(group["name"]),
                }
            )
    if not param_groups:
        raise RuntimeError("No trainable parameters were configured.")
    optimizer = torch.optim.AdamW(param_groups)

    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )
    module = unwrap_ddp(model)
    if module is None:
        raise RuntimeError("DDP unwrap failed after wrapping.")
    output_dir = resolve_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir is required.")
    load_stats = {
        "action_checkpoint": str(action_checkpoint),
        "stage1_checkpoint": str(stage1_checkpoint),
        "action": action_load_stats,
        "stage1": stage1_load_stats,
    }

    rank0_print(rank, f"ddp_enabled={ddp_enabled}", flush=True)
    rank0_print(rank, f"world_size={world_size}", flush=True)
    rank0_print(rank, f"rank={rank}", flush=True)
    rank0_print(rank, f"local_rank={local_rank}", flush=True)
    rank0_print(rank, f"per_gpu_batch_size={args.batch_size}", flush=True)
    rank0_print(rank, f"grad_accum_steps={args.grad_accum_steps}", flush=True)
    rank0_print(rank, f"effective_global_batch_size={int(args.batch_size) * world_size * int(args.grad_accum_steps)}", flush=True)
    rank0_print(rank, f"action_checkpoint={action_checkpoint}", flush=True)
    rank0_print(rank, f"stage1_checkpoint={stage1_checkpoint}", flush=True)
    rank0_print(rank, f"vjepa_checkpoint={require_file(args.vjepa_checkpoint, name='--vjepa-checkpoint')}", flush=True)
    rank0_print(rank, f"proprio_enabled={bool(args.use_proprio)}", flush=True)
    rank0_print(rank, f"stage5_use_z_task_token={bool(args.use_z_task_token)}", flush=True)
    if bool(args.use_z_task_token):
        rank0_print(rank, f"stage1_text_action_checkpoint={args.stage1_text_action_checkpoint}", flush=True)
        rank0_print(rank, f"freeze_pairwise_latent={bool(args.freeze_pairwise_latent)}", flush=True)
        rank0_print(rank, f"z_task_dim={args.z_task_dim}", flush=True)
        rank0_print(rank, f"z_task_gate_init={args.z_task_gate_init}", flush=True)
        rank0_print(rank, f"lambda_vlp_to_a={args.lambda_vlp_to_a}", flush=True)
        rank0_print(rank, f"lambda_va_to_l={args.lambda_va_to_l}", flush=True)
        rank0_print(
            rank,
            f"pairwise_latent_load_stats={getattr(module, 'pairwise_latent_load_stats', {})}",
            flush=True,
        )
    rank0_print(rank, f"current_token_count={current_token_count} future_token_count={future_token_count} resolved_num_future_tokens={resolved_tokens}", flush=True)
    rank0_print(rank, f"trainable_adapter_params={trainable_count(module.jepa_adapter)} total={total_count(module.jepa_adapter)}", flush=True)
    rank0_print(rank, f"trainable_proprio_params={trainable_count(module.proprio_encoder)} total={total_count(module.proprio_encoder)}", flush=True)
    rank0_print(rank, f"trainable_predictor_params={trainable_count(module.future_predictor)} total={total_count(module.future_predictor)}", flush=True)
    rank0_print(rank, f"trainable_action_dit_params={trainable_count(module.action_expert)} total={total_count(module.action_expert)}", flush=True)
    rank0_print(rank, f"frozen_vjepa_params={total_count(module.vjepa_encoder)}", flush=True)
    rank0_print(rank, f"first_batch_shapes video={tuple(first_batch['video'].shape)} future_video={tuple(first_batch['future_video'].shape)} action={tuple(first_batch['action'].shape)} context={tuple(first_batch['context'].shape)} proprio={None if first_batch.get('proprio') is None else tuple(first_batch['proprio'].shape)}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    grad_accum_steps = int(args.grad_accum_steps)
    pending_first = first_batch
    last_loss: dict[str, float] = {}
    start_time = time.time()
    accum_start = time.time()

    while update_step < int(args.steps):
        if sampler is not None and micro_step % max(len(loader), 1) == 0:
            sampler.set_epoch(micro_step // max(len(loader), 1))
        if pending_first is not None:
            batch = pending_first
            pending_first = None
        else:
            try:
                raw_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                raw_batch = next(data_iter)
            batch = canonicalize_libero_batch(raw_batch, args=args, device=device, dtype=param_dtype)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            loss_total, loss_dict = stage2_forward_loss(model, batch, args=args)
            scaled_loss = loss_total / float(grad_accum_steps)
        finite_flag = torch.tensor(1 if torch.isfinite(loss_total).item() else 0, device=device, dtype=torch.int32)
        if ddp_enabled:
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
        if int(finite_flag.item()) != 1:
            raise RuntimeError(f"Non-finite loss detected at update_step={update_step + 1}.")
        scaled_loss.backward()
        micro_step += 1

        if micro_step % grad_accum_steps != 0:
            continue

        if float(args.max_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [param for group in param_groups for param in group["params"]],
                float(args.max_grad_norm),
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

        reduced_loss = reduce_loss_dict(loss_dict, ddp_enabled=ddp_enabled, world_size=world_size)
        last_loss = dict(reduced_loss)
        if is_rank0(rank) and (update_step == 1 or update_step % int(args.log_every) == 0):
            elapsed = max(time.time() - accum_start, 1.0e-6)
            samples = int(args.batch_size) * int(world_size) * int(args.grad_accum_steps) * max(update_step if update_step == 1 else int(args.log_every), 1)
            samples_per_sec = samples / elapsed
            lr_msg = ",".join(f"{group.get('name','group')}:{group['lr']:.2e}" for group in optimizer.param_groups)
            mem = ""
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                mem = f" gpu_memory=allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB"
            print(
                " ".join(
                    [
                        f"step={update_step}",
                        f"loss_total={reduced_loss['loss_total']:.6f}",
                        f"loss_action={reduced_loss['loss_action']:.6f}",
                        f"loss_future_jepa={reduced_loss['loss_future_jepa']:.6f}",
                        f"loss_future_l1={reduced_loss['loss_future_l1']:.6f}",
                        f"loss_future_cos={reduced_loss['loss_future_cos']:.6f}",
                        f"loss_vlp_to_a={reduced_loss.get('loss_vlp_to_a', 0.0):.6f}",
                        f"loss_va_to_l={reduced_loss.get('loss_va_to_l', 0.0):.6f}",
                        f"retrieval_acc_vlp_to_a={reduced_loss.get('retrieval_acc_vlp_to_a', 0.0):.6f}",
                        f"z_task_gate={reduced_loss.get('z_task_gate', 0.0):.6f}",
                        f"lr={lr_msg}",
                        f"iter_time_sec={elapsed:.3f}",
                        f"samples_per_sec={samples_per_sec:.3f}",
                        mem,
                    ]
                ),
                flush=True,
            )
            accum_start = time.time()

        if is_rank0(rank) and int(args.save_every) > 0 and update_step % int(args.save_every) == 0:
            path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=output_dir / "checkpoints",
                step=update_step,
                args=args,
                loss_dict=last_loss,
                load_stats=load_stats,
            )
            print(f"saved_checkpoint={path}", flush=True)

    if is_rank0(rank):
        path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=output_dir / "checkpoints",
            step=update_step,
            args=args,
            loss_dict=last_loss,
            load_stats=load_stats,
        )
        summary = {
            "steps": update_step,
            "elapsed_sec": time.time() - start_time,
            "last_loss": last_loss,
            "checkpoint": str(path),
            "current_token_count": current_token_count,
            "future_token_count": future_token_count,
            "resolved_num_future_tokens": resolved_tokens,
            "proprio_enabled": bool(args.use_proprio),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "stage2_libero_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"saved_final_checkpoint={path}", flush=True)
        print(f"saved_summary={output_dir / 'stage2_libero_summary.json'}", flush=True)

    if ddp_enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
