from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vjepa.jepa_kv_cache_generator import (
    JepaKVCacheGenerator,
    kv_cache_distillation_loss,
)
from ..vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper
from .action_dit import ActionDiT
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


SELECTED_FRAME_INDEX = 0
CONTEXT_MASK_MODES = ("baseline_all_true", "cached_real_mask")


def normalize_context_mask_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in CONTEXT_MASK_MODES:
        raise ValueError(
            f"Unsupported context mask mode {mode!r}; expected one of {CONTEXT_MASK_MODES}."
        )
    return normalized


def prepare_v4_context(
    context: torch.Tensor,
    real_context_mask: torch.Tensor,
    *,
    mode: str = "baseline_all_true",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero cached padding embeddings, then apply the requested v4 mask protocol."""
    mode = normalize_context_mask_mode(mode)
    if context.ndim not in (2, 3):
        raise ValueError(f"Context must be [L,D] or [B,L,D], got {tuple(context.shape)}.")
    if real_context_mask.ndim != context.ndim - 1:
        raise ValueError(
            "Context mask rank must be one less than context rank, got "
            f"{tuple(real_context_mask.shape)} vs {tuple(context.shape)}."
        )
    if tuple(real_context_mask.shape) != tuple(context.shape[:-1]):
        raise ValueError(
            "Context mask shape must match context token axes, got "
            f"{tuple(real_context_mask.shape)} vs {tuple(context.shape[:-1])}."
        )
    real_mask = real_context_mask.to(device=context.device, dtype=torch.bool)
    if bool((real_mask.reshape(-1, real_mask.shape[-1]).sum(dim=1) == 0).any()):
        raise ValueError("Every context sample must contain at least one valid text token.")
    prepared = context.clone().masked_fill(~real_mask.unsqueeze(-1), 0.0)
    if mode == "baseline_all_true":
        output_mask = torch.ones_like(real_mask, dtype=torch.bool)
    else:
        output_mask = real_mask.clone()
    return prepared, output_mask


def prepare_baseline_compatible_context(
    context: torch.Tensor,
    real_context_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return prepare_v4_context(context, real_context_mask, mode="baseline_all_true")


def validate_checkpoint_context_mask_mode(
    metadata: Mapping[str, Any],
    requested_mode: str,
    *,
    checkpoint_name: str,
    allow_mismatch: bool = False,
) -> str:
    requested = normalize_context_mask_mode(requested_mode)
    checkpoint_mode = metadata.get("context_mask_mode")
    if checkpoint_mode is None:
        raise ValueError(f"{checkpoint_name} metadata is missing context_mask_mode.")
    checkpoint_mode = normalize_context_mask_mode(str(checkpoint_mode))
    if checkpoint_mode != requested and not allow_mismatch:
        raise ValueError(
            f"{checkpoint_name} was trained with {checkpoint_mode}, "
            f"current run requests {requested}."
        )
    return checkpoint_mode


def extract_causal_current_frame(video: torch.Tensor) -> torch.Tensor:
    """Select only decision-time frame zero without inspecting future frames."""
    if video.ndim == 5:
        if video.shape[2] < 1:
            raise ValueError("`video` must contain frame index 0.")
        current = video[:, :, SELECTED_FRAME_INDEX]
    elif video.ndim == 4:
        current = video
    else:
        raise ValueError(
            "`video` must be [B,3,T,H,W] or [B,3,H,W], "
            f"got {tuple(video.shape)}."
        )
    if current.shape[1] != 3:
        raise ValueError(f"Current frame must have three RGB channels, got {current.shape[1]}.")
    return current


def build_duplicated_vjepa_clip(current_frame: torch.Tensor) -> torch.Tensor:
    if current_frame.ndim != 4 or current_frame.shape[1] != 3:
        raise ValueError(
            "`current_frame` must be [B,3,H,W], "
            f"got {tuple(current_frame.shape)}."
        )
    clip = current_frame.unsqueeze(2).repeat(1, 1, 2, 1, 1)
    if not torch.equal(clip[:, :, 0], clip[:, :, 1]):
        raise RuntimeError("Duplicated V-JEPA frames must be elementwise identical.")
    return clip


def split_dual_camera_current_frame(
    current_frame: torch.Tensor,
    *,
    camera_order: Sequence[str],
    image_size: int,
) -> list[torch.Tensor]:
    if len(camera_order) != 2:
        raise ValueError(f"v4 requires exactly two camera keys, got {list(camera_order)}.")
    height, width = int(current_frame.shape[-2]), int(current_frame.shape[-1])
    if width % 2 != 0:
        raise ValueError(f"Horizontal camera collage width must be even, got {width}.")
    camera_width = width // 2
    if camera_width != height:
        raise ValueError(
            "Each camera must be square before V-JEPA preprocessing; "
            f"collage HxW={height}x{width} gives camera HxW={height}x{camera_width}."
        )
    frames = list(current_frame.split(camera_width, dim=-1))
    if len(frames) != 2:
        raise RuntimeError(f"Expected two camera frames, got {len(frames)}.")
    return [
        F.interpolate(frame, size=(image_size, image_size), mode="bilinear", align_corners=False)
        for frame in frames
    ]


def flatten_horizontal_camera_grids(camera_grids: Sequence[torch.Tensor]) -> torch.Tensor:
    """Flatten equal camera grids in the teacher's horizontal row-major order."""
    if not camera_grids:
        raise ValueError("At least one camera grid is required.")
    reference_shape = tuple(camera_grids[0].shape)
    if len(reference_shape) != 4:
        raise ValueError(f"Camera grids must be [B,H,W,D], got {reference_shape}.")
    for index, grid in enumerate(camera_grids):
        if tuple(grid.shape) != reference_shape:
            raise ValueError(
                f"Camera grid {index} shape {tuple(grid.shape)} does not match {reference_shape}."
            )
    joint_grid = torch.cat(list(camera_grids), dim=2)
    return joint_grid.reshape(joint_grid.shape[0], -1, joint_grid.shape[-1])


def _pair(value: Any, *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            return int(value[0]), int(value[1])
        if len(value) == 3:
            return int(value[-2]), int(value[-1])
    raise ValueError(f"Could not resolve two-dimensional {name} from {value!r}.")


def _resolve_vjepa_dense_grid(
    encoder: nn.Module,
    *,
    token_count: int,
    clip_frames: int,
) -> tuple[int, int]:
    tubelet_size = int(getattr(encoder, "tubelet_size", 0))
    if tubelet_size <= 0 or clip_frames % tubelet_size != 0:
        raise ValueError(
            f"Invalid V-JEPA tubelet configuration: frames={clip_frames}, tubelet={tubelet_size}."
        )
    temporal_grid = clip_frames // tubelet_size
    if temporal_grid != 1:
        raise ValueError(
            "v4 requires exactly one temporal tubelet from duplicated current frames, "
            f"got {temporal_grid}."
        )

    if bool(getattr(encoder, "dummy", False)):
        side = int(math.isqrt(token_count))
        if side * side != token_count:
            raise ValueError(
                "Dummy V-JEPA tokens must form a dense square spatial grid, "
                f"got N={token_count}."
            )
        return side, side

    core = getattr(encoder, "encoder", None)
    patch_embed = getattr(core, "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(core, "patch_size", None)
    if patch_size is None:
        raise ValueError(
            "Real V-JEPA encoder does not expose patch_size; dense token layout cannot be verified."
        )
    patch_h, patch_w = _pair(patch_size, name="V-JEPA patch_size")
    image_size = getattr(encoder, "img_size", None)
    if image_size is None:
        raise ValueError("V-JEPA wrapper must expose img_size for dense-grid validation.")
    image_h, image_w = _pair(image_size, name="V-JEPA img_size")
    if image_h % patch_h != 0 or image_w % patch_w != 0:
        raise ValueError(
            f"V-JEPA image size {(image_h, image_w)} is not divisible by patch {(patch_h, patch_w)}."
        )
    grid_h, grid_w = image_h // patch_h, image_w // patch_w
    expected_tokens = temporal_grid * grid_h * grid_w
    if token_count != expected_tokens:
        raise ValueError(
            "V-JEPA output is not the expected dense patch sequence; refusing to truncate "
            f"possible CLS/register tokens. expected={expected_tokens}, actual={token_count}, "
            f"grid={temporal_grid}x{grid_h}x{grid_w}."
        )
    return grid_h, grid_w


def encode_causal_dual_camera_tokens(
    vjepa_encoder: nn.Module,
    current_frame: torch.Tensor,
    *,
    camera_order: Sequence[str],
    output_grid: tuple[int, int] = (7, 7),
) -> tuple[torch.Tensor, dict[str, Any]]:
    image_size = int(getattr(vjepa_encoder, "img_size", 0))
    if image_size <= 0:
        raise ValueError("V-JEPA encoder must expose a positive img_size.")
    camera_frames = split_dual_camera_current_frame(
        current_frame,
        camera_order=camera_order,
        image_size=image_size,
    )
    camera_clips = [build_duplicated_vjepa_clip(frame) for frame in camera_frames]
    encoder_input = torch.cat(camera_clips, dim=0)
    if not torch.equal(encoder_input[:, :, 0], encoder_input[:, :, 1]):
        raise RuntimeError("All V-JEPA clips must duplicate the same current frame.")

    if bool(getattr(vjepa_encoder, "freeze", False)):
        with torch.no_grad():
            encoded = vjepa_encoder(encoder_input)
    else:
        encoded = vjepa_encoder(encoder_input)
    if encoded.ndim != 3 or encoded.shape[0] != encoder_input.shape[0]:
        raise ValueError(
            "V-JEPA output must be [2B,N,D], got "
            f"{tuple(encoded.shape)} for input {tuple(encoder_input.shape)}."
        )
    grid_h, grid_w = _resolve_vjepa_dense_grid(
        vjepa_encoder,
        token_count=int(encoded.shape[1]),
        clip_frames=int(encoder_input.shape[2]),
    )
    if int(encoded.shape[1]) != grid_h * grid_w:
        raise ValueError("Only one dense temporal tubelet is supported by v4.")

    batch_size = int(current_frame.shape[0])
    per_camera = encoded.reshape(2, batch_size, grid_h, grid_w, encoded.shape[-1])
    pooled_grids: list[torch.Tensor] = []
    for camera_idx in range(2):
        tokens = per_camera[camera_idx].permute(0, 3, 1, 2)
        tokens = F.adaptive_avg_pool2d(tokens, output_grid)
        pooled_grids.append(tokens.permute(0, 2, 3, 1))
    visual_tokens = flatten_horizontal_camera_grids(pooled_grids)
    expected_seq = 2 * int(output_grid[0]) * int(output_grid[1])
    if int(visual_tokens.shape[1]) != expected_seq:
        raise RuntimeError(
            f"Dual-camera pooled sequence must contain {expected_seq} tokens, got {visual_tokens.shape[1]}."
        )
    debug = {
        "selected_frame_index": SELECTED_FRAME_INDEX,
        "camera_order": tuple(str(name) for name in camera_order),
        "camera_frame_shapes": tuple(tuple(frame.shape) for frame in camera_frames),
        "vjepa_clip_shape": tuple(encoder_input.shape),
        "vjepa_dense_grid": (grid_h, grid_w),
        "visual_tokens_shape": tuple(visual_tokens.shape),
        "duplicated_frames_equal": True,
    }
    return visual_tokens, debug


def validate_teacher_cache_row_major(
    cache: list[dict[str, torch.Tensor]],
    *,
    grid_size: Sequence[int],
    num_cameras: int = 2,
) -> list[dict[str, torch.Tensor]]:
    if len(grid_size) != 3:
        raise ValueError(f"Teacher grid_size must be (T,H,W), got {tuple(grid_size)}.")
    grid_t, grid_h, grid_w = (int(value) for value in grid_size)
    if grid_t != 1 or grid_w % num_cameras != 0:
        raise ValueError(
            "Teacher current-frame cache must be one temporal grid with evenly split cameras, "
            f"got grid={grid_t}x{grid_h}x{grid_w}."
        )
    for layer_idx, layer in enumerate(cache):
        for key in ("k", "v"):
            if key not in layer:
                raise ValueError(f"Teacher layer {layer_idx} is missing {key!r}.")
            value = layer[key]
            if int(value.shape[1]) != grid_t * grid_h * grid_w:
                raise ValueError(
                    f"Teacher layer {layer_idx} {key} does not match grid {tuple(grid_size)}."
                )
    return cache


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset_stats(
    *,
    expected_sha256: str,
    dataset_stats_path: str | Path,
    allow_mismatch: bool = False,
) -> str:
    actual = sha256_file(dataset_stats_path)
    if actual != str(expected_sha256) and not allow_mismatch:
        raise ValueError(
            "Dataset stats SHA256 mismatch: "
            f"checkpoint={expected_sha256}, current={actual}, path={dataset_stats_path}."
        )
    return actual


class FastWAMJEPAKVV4(nn.Module):
    """Causal single-frame V-JEPA student for original FastWAM action-cache denoising."""

    def __init__(
        self,
        *,
        action_expert: ActionDiT,
        vjepa_encoder: VJepaEncoderWrapper,
        kv_generator: JepaKVCacheGenerator,
        camera_order: Sequence[str],
        proprio_dim: Optional[int] = 8,
        action_horizon: int = 32,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        freeze_vjepa: bool = True,
        freeze_action: bool = True,
        freeze_proprio: bool = True,
        context_mask_mode: str = "baseline_all_true",
        device: Optional[str | torch.device] = None,
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if len(camera_order) != 2:
            raise ValueError(f"v4 requires exactly two cameras, got {list(camera_order)}.")
        self.action_expert = action_expert
        self.vjepa_encoder = vjepa_encoder
        self.kv_generator = kv_generator
        self.camera_order = tuple(str(name) for name in camera_order)
        self.action_dim = int(action_expert.action_dim)
        self.text_dim = int(action_expert.text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.action_horizon = int(action_horizon)
        self.video_seq_len = int(kv_generator.video_seq_len)
        self.context_mask_mode = normalize_context_mask_mode(context_mask_mode)
        self.torch_dtype = torch_dtype
        self.mot = MoT(
            mixtures={"action": action_expert},
            mot_checkpoint_mixed_attn=False,
            external_video_cache_only=True,
        )
        if len(action_expert.blocks) != kv_generator.num_layers:
            raise ValueError("KV generator num_layers must match ActionDiT blocks.")
        if int(action_expert.num_heads) != kv_generator.num_heads:
            raise ValueError("KV generator num_heads must match ActionDiT.")
        if int(action_expert.attn_head_dim) != kv_generator.attn_head_dim:
            raise ValueError("KV generator head_dim must match ActionDiT.")
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {self.action_horizon}.")

        self.proprio_encoder = (
            None if self.proprio_dim is None else nn.Linear(self.proprio_dim, self.text_dim)
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(action_num_train_timesteps),
            shift=float(action_train_shift),
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(action_num_train_timesteps),
            shift=float(action_infer_shift),
        )
        self.freeze_vjepa = bool(freeze_vjepa)
        self.freeze_action = bool(freeze_action)
        self.freeze_proprio = bool(freeze_proprio)
        if self.freeze_vjepa:
            self.vjepa_encoder.requires_grad_(False)
            self.vjepa_encoder.eval()
        if self.freeze_action:
            self.action_expert.requires_grad_(False)
            self.action_expert.eval()
        if self.freeze_proprio and self.proprio_encoder is not None:
            self.proprio_encoder.requires_grad_(False)
            self.proprio_encoder.eval()

        self.debug_counts = {"vjepa_forward": 0, "kv_generator_forward": 0, "action_forward": 0}
        self.last_debug: dict[str, Any] = {"selected_frame_index": SELECTED_FRAME_INDEX}
        self.last_inference_timing: dict[str, float] = {}
        self._infer_cache_ids: list[int] = []
        if device is not None:
            self.to(device=device, dtype=torch_dtype)

    def train(self, mode: bool = True) -> "FastWAMJEPAKVV4":
        super().train(mode)
        if self.freeze_vjepa:
            self.vjepa_encoder.eval()
        if self.freeze_action:
            self.action_expert.eval()
        if self.freeze_proprio and self.proprio_encoder is not None:
            self.proprio_encoder.eval()
        return self

    def _runtime_device(self) -> torch.device:
        return next(self.kv_generator.parameters()).device

    @staticmethod
    def extract_causal_current_frame(video: torch.Tensor) -> torch.Tensor:
        return extract_causal_current_frame(video)

    @staticmethod
    def build_duplicated_vjepa_clip(current_frame: torch.Tensor) -> torch.Tensor:
        return build_duplicated_vjepa_clip(current_frame)

    def reset_debug_counters(self) -> None:
        for key in self.debug_counts:
            self.debug_counts[key] = 0
        self.kv_generator.reset_debug_counters()
        self._infer_cache_ids = []

    def _prepare_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context.shape[-1] != self.text_dim:
            raise ValueError(f"Context must be [B,L,{self.text_dim}], got {tuple(context.shape)}.")
        if context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError("A real [B,L] context mask matching context is required by v4.")
        device = self._runtime_device()
        context = context.to(device=device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        context, context_mask = prepare_v4_context(
            context,
            context_mask,
            mode=self.context_mask_mode,
        )

        if self.proprio_encoder is None:
            if proprio is not None:
                raise ValueError("Proprio was supplied but the v4 proprio encoder is disabled.")
            return context, context_mask
        if proprio is None:
            raise ValueError("Proprio is required when the v4 proprio encoder is enabled.")
        if proprio.ndim == 3:
            proprio = proprio[:, SELECTED_FRAME_INDEX]
        if proprio.ndim != 2 or proprio.shape != (context.shape[0], self.proprio_dim):
            raise ValueError(
                f"Proprio must be [B,{self.proprio_dim}], got {tuple(proprio.shape)}."
            )
        token = self.proprio_encoder(
            proprio.to(device=device, dtype=self.torch_dtype).unsqueeze(1)
        )
        token_mask = torch.ones((context.shape[0], 1), dtype=torch.bool, device=device)
        return torch.cat([context, token], dim=1), torch.cat([context_mask, token_mask], dim=1)

    def encode_current_frame(self, video: torch.Tensor) -> torch.Tensor:
        current = extract_causal_current_frame(video).to(
            device=self._runtime_device(), dtype=self.torch_dtype
        )
        self.debug_counts["vjepa_forward"] += 1
        visual_tokens, debug = encode_causal_dual_camera_tokens(
            self.vjepa_encoder,
            current,
            camera_order=self.camera_order,
        )
        self.last_debug.update(debug)
        return visual_tokens.to(device=self._runtime_device(), dtype=self.torch_dtype)

    def generate_video_kv_cache(
        self,
        video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        visual_tokens = self.encode_current_frame(video)
        self.debug_counts["kv_generator_forward"] += 1
        cache = self.kv_generator(visual_tokens, context, context_mask)
        self._validate_cache(cache, batch_size=int(visual_tokens.shape[0]))
        return cache

    def _validate_cache(
        self,
        cache: list[dict[str, torch.Tensor]],
        *,
        batch_size: int,
    ) -> None:
        if len(cache) != len(self.action_expert.blocks):
            raise ValueError(
                f"Expected {len(self.action_expert.blocks)} cache layers, got {len(cache)}."
            )
        expected = (batch_size, self.video_seq_len, self.kv_generator.cache_dim)
        for layer_idx, layer in enumerate(cache):
            for key in ("k", "v"):
                if key not in layer or tuple(layer[key].shape) != expected:
                    shape = None if key not in layer else tuple(layer[key].shape)
                    raise ValueError(
                        f"Cache layer {layer_idx} {key} must be {expected}, got {shape}."
                    )

    @staticmethod
    def _action_attention_mask(
        *,
        video_seq_len: int,
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        total = int(video_seq_len) + int(action_seq_len)
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = True
        mask[video_seq_len:, :] = True
        return mask

    def predict_action_noise_with_cache(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        self._validate_cache(video_kv_cache, batch_size=int(noisy_action.shape[0]))
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._action_attention_mask(
            video_seq_len=self.video_seq_len,
            action_seq_len=int(noisy_action.shape[1]),
            device=noisy_action.device,
        )
        self.debug_counts["action_forward"] += 1
        self._infer_cache_ids.append(id(video_kv_cache))
        tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=self.video_seq_len,
        )
        return self.action_expert.post_dit(tokens, action_pre)

    def training_loss(
        self,
        sample: dict[str, torch.Tensor],
        *,
        teacher_video_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
        lambda_kv: float = 0.0,
        kv_lambda_cos: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        required = ("video", "action", "context", "context_mask")
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"v4 sample is missing keys: {missing}.")
        device = self._runtime_device()
        context, context_mask = self._prepare_context(
            sample["context"], sample["context_mask"], sample.get("proprio")
        )
        cache = self.generate_video_kv_cache(sample["video"], context, context_mask)

        action = sample["action"]
        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action must be [B,T,{self.action_dim}], got {tuple(action.shape)}."
            )
        if action.shape[1] < self.action_horizon:
            raise ValueError(
                f"Action target needs {self.action_horizon} steps, got {action.shape[1]}."
            )
        action = action[:, : self.action_horizon].to(device=device, dtype=self.torch_dtype)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad[:, : self.action_horizon].to(device=device, dtype=torch.bool)

        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=int(action.shape[0]),
            device=device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise, timestep)
        target = self.train_action_scheduler.training_target(action, noise, timestep)
        prediction = self.predict_action_noise_with_cache(
            noisy_action, timestep, context, context_mask, cache
        )
        per_token = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=-1)
        if action_is_pad is None:
            per_sample = per_token.mean(dim=1)
        else:
            valid = (~action_is_pad).to(dtype=per_token.dtype)
            per_sample = (per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        weight = self.train_action_scheduler.training_weight(timestep).to(
            device=device, dtype=per_sample.dtype
        )
        loss_action = (per_sample * weight).mean()

        if float(lambda_kv) > 0.0:
            if teacher_video_kv_cache is None:
                raise ValueError("lambda_kv > 0 requires a teacher video K/V cache.")
            loss_kv, kv_metrics = kv_cache_distillation_loss(
                cache, teacher_video_kv_cache, lambda_cos=float(kv_lambda_cos)
            )
            loss_k = kv_metrics["loss_k"]
            loss_v = kv_metrics["loss_v"]
            loss_cos = kv_metrics["loss_cos"]
            cos_first = kv_metrics["cos_first"]
            cos_middle = kv_metrics["cos_middle"]
            cos_last = kv_metrics["cos_last"]
        else:
            loss_kv = loss_action.new_zeros(())
            loss_k = loss_action.new_zeros(())
            loss_v = loss_action.new_zeros(())
            loss_cos = loss_action.new_zeros(())
            cos_first = loss_action.new_zeros(())
            cos_middle = loss_action.new_zeros(())
            cos_last = loss_action.new_zeros(())
        total = loss_action + float(lambda_kv) * loss_kv
        return total, {
            "loss_total": float(total.detach()),
            "loss_action": float(loss_action.detach()),
            "loss_kv": float(loss_kv.detach()),
            "loss_k": float(loss_k.detach()),
            "loss_v": float(loss_v.detach()),
            "loss_cos": float(loss_cos.detach()),
            "cos_first": float(cos_first.detach()),
            "cos_middle": float(cos_middle.detach()),
            "cos_last": float(cos_last.detach()),
        }

    def forward(self, *args: Any, **kwargs: Any):
        return self.training_loss(*args, **kwargs)

    @torch.no_grad()
    def infer_action(
        self,
        *,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> torch.Tensor:
        replan_start = time.perf_counter()
        self.eval()
        self.reset_debug_counters()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        current = extract_causal_current_frame(input_image)
        if current.shape[0] != 1:
            raise ValueError(f"infer_action requires batch size 1, got {current.shape[0]}.")
        context, context_mask = self._prepare_context(context, context_mask, proprio)
        if context.shape[0] != 1:
            raise ValueError(f"infer_action context batch must be 1, got {context.shape[0]}.")
        runtime_device = self._runtime_device()

        def sync_device() -> None:
            if runtime_device.type == "cuda":
                torch.cuda.synchronize(runtime_device)

        sync_device()
        vjepa_start = time.perf_counter()
        visual_tokens = self.encode_current_frame(current)
        sync_device()
        vjepa_ms = (time.perf_counter() - vjepa_start) * 1000.0

        kv_start = time.perf_counter()
        self.debug_counts["kv_generator_forward"] += 1
        cache = self.kv_generator(visual_tokens, context, context_mask)
        self._validate_cache(cache, batch_size=int(visual_tokens.shape[0]))
        sync_device()
        kv_generator_ms = (time.perf_counter() - kv_start) * 1000.0

        horizon = self.action_horizon if action_horizon is None else int(action_horizon)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn(
            (1, horizon, self.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self._runtime_device(), dtype=self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self._runtime_device(),
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        action_start = time.perf_counter()
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.unsqueeze(0).to(device=action.device, dtype=action.dtype)
            prediction = self.predict_action_noise_with_cache(
                action, timestep, context, context_mask, cache
            )
            action = self.infer_action_scheduler.step(prediction, step_delta, action)
        sync_device()
        action_denoise_ms = (time.perf_counter() - action_start) * 1000.0

        if self.debug_counts["vjepa_forward"] != 1:
            raise RuntimeError(f"Expected one V-JEPA forward, got {self.debug_counts['vjepa_forward']}.")
        if self.debug_counts["kv_generator_forward"] != 1:
            raise RuntimeError(
                f"Expected one KV generator forward, got {self.debug_counts['kv_generator_forward']}."
            )
        if self.debug_counts["action_forward"] != int(num_inference_steps):
            raise RuntimeError(
                "Action forward count must equal inference steps, got "
                f"{self.debug_counts['action_forward']} vs {num_inference_steps}."
            )
        if len(set(self._infer_cache_ids)) != 1:
            raise RuntimeError("Action denoising did not reuse the same cache object.")
        self.last_inference_timing = {
            "vjepa_ms": float(vjepa_ms),
            "kv_generator_ms": float(kv_generator_ms),
            "action_denoise_ms": float(action_denoise_ms),
            "total_replan_ms": float((time.perf_counter() - replan_start) * 1000.0),
        }
        self.last_debug["timing_ms"] = dict(self.last_inference_timing)
        return action[0].detach().to(device="cpu", dtype=torch.float32)

    def load_student_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        dataset_stats_path: str | Path,
        allow_stats_mismatch: bool = False,
    ) -> dict[str, Any]:
        payload = torch.load(Path(checkpoint_path), map_location=self._runtime_device())
        if not isinstance(payload, dict):
            raise ValueError("v4 checkpoint payload must be a dict.")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or "dataset_stats_sha256" not in metadata:
            raise ValueError("v4 checkpoint is missing dataset stats SHA256 metadata.")
        validate_dataset_stats(
            expected_sha256=str(metadata["dataset_stats_sha256"]),
            dataset_stats_path=dataset_stats_path,
            allow_mismatch=allow_stats_mismatch,
        )
        self.kv_generator.load_state_dict(payload["kv_generator"], strict=True)
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        if self.proprio_encoder is not None:
            if payload.get("proprio_encoder") is None:
                raise ValueError("v4 checkpoint is missing proprio_encoder weights.")
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        return payload
