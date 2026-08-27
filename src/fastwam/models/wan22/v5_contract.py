from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


V5_VERSION = "v5"
ACTION_HORIZON = 16
EXEC_HORIZON_DEFAULT = 4
DATASET_VIDEO_INDICES = (0, 1, 2, 3, 4)
RAW_OBSERVATION_OFFSETS = (0, 4, 8, 12, 16)
VISUAL_STRIDE = 4
CAMERA_ORDER = ("agentview", "wrist")
CAMERA_HEIGHT = 224
CAMERA_WIDTH = 224
JOINT_WIDTH = 448
VJEPA_SPATIAL_SIZE = 16
SPATIAL_POOL_SIZE = 6
VJEPA_DIM = 1408
CURRENT_TOKEN_COUNT_PER_CAMERA = 256
FUTURE_TOKEN_COUNT_PER_CAMERA = 512
TOKENS_PER_CAMERA_GROUP = SPATIAL_POOL_SIZE * SPATIAL_POOL_SIZE
TOKENS_PER_TEMPORAL_GROUP = len(CAMERA_ORDER) * TOKENS_PER_CAMERA_GROUP
VISUAL_TOKEN_COUNT = 3 * TOKENS_PER_TEMPORAL_GROUP


def temporal_metadata() -> dict[str, Any]:
    return {
        "version": V5_VERSION,
        "action_horizon": ACTION_HORIZON,
        "exec_horizon_default": EXEC_HORIZON_DEFAULT,
        "dataset_video_indices": list(DATASET_VIDEO_INDICES),
        "raw_observation_offsets": list(RAW_OBSERVATION_OFFSETS),
        "visual_stride": VISUAL_STRIDE,
        "camera_order": list(CAMERA_ORDER),
        "flatten_order": "temporal_camera_spatial_row_major",
    }


def split_dual_camera_video(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if video.ndim != 5:
        raise ValueError(f"V5 video must be [B,3,T,224,448], got {tuple(video.shape)}.")
    if int(video.shape[1]) != 3:
        raise ValueError(f"V5 video must have 3 RGB channels, got {video.shape[1]}.")
    if tuple(video.shape[-2:]) != (CAMERA_HEIGHT, JOINT_WIDTH):
        raise ValueError(
            "V5 requires an exact horizontal 224x448 agentview/wrist layout, "
            f"got {tuple(video.shape[-2:])}."
        )
    agentview = video[..., :CAMERA_WIDTH]
    wrist = video[..., CAMERA_WIDTH:]
    if tuple(agentview.shape[-2:]) != (CAMERA_HEIGHT, CAMERA_WIDTH):
        raise RuntimeError("Agentview split did not produce 224x224 frames.")
    if tuple(wrist.shape[-2:]) != (CAMERA_HEIGHT, CAMERA_WIDTH):
        raise RuntimeError("Wrist split did not produce 224x224 frames.")
    return agentview.contiguous(), wrist.contiguous()


def build_vjepa_clips(video: torch.Tensor) -> dict[str, torch.Tensor]:
    if int(video.shape[2]) < len(DATASET_VIDEO_INDICES):
        raise ValueError(
            "V5 requires video indices [0,1,2,3,4], "
            f"but video has only {video.shape[2]} timesteps."
        )
    agentview, wrist = split_dual_camera_video(video[:, :, :5])
    return {
        "agentview_current": agentview[:, :, 0:1].repeat(1, 1, 2, 1, 1),
        "wrist_current": wrist[:, :, 0:1].repeat(1, 1, 2, 1, 1),
        "agentview_future": agentview[:, :, 1:5],
        "wrist_future": wrist[:, :, 1:5],
    }


def _pool_camera_tokens(
    tokens: torch.Tensor,
    *,
    temporal_groups: int,
    expected_tokens: int,
    expected_dim: int,
    spatial_pool_size: int,
) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"V-JEPA tokens must be [B,N,D], got {tuple(tokens.shape)}.")
    if int(tokens.shape[1]) != expected_tokens or int(tokens.shape[2]) != expected_dim:
        raise ValueError(
            "V-JEPA token contract mismatch: "
            f"expected [B,{expected_tokens},{expected_dim}], got {tuple(tokens.shape)}."
        )
    spatial_tokens = VJEPA_SPATIAL_SIZE * VJEPA_SPATIAL_SIZE
    if expected_tokens != temporal_groups * spatial_tokens:
        raise RuntimeError("V5 temporal/spatial token constants are inconsistent.")
    batch_size = int(tokens.shape[0])
    # Official PatchEmbed3D returns Conv3d [B,D,T,H,W] flattened over T,H,W,
    # so contiguous token groups are temporal-major and spatial row-major.
    grid = tokens.reshape(
        batch_size, temporal_groups, VJEPA_SPATIAL_SIZE, VJEPA_SPATIAL_SIZE, expected_dim
    )
    grid = grid.permute(0, 1, 4, 2, 3).reshape(
        batch_size * temporal_groups,
        expected_dim,
        VJEPA_SPATIAL_SIZE,
        VJEPA_SPATIAL_SIZE,
    )
    pooled = F.adaptive_avg_pool2d(grid.float(), (spatial_pool_size, spatial_pool_size))
    pooled = pooled.to(dtype=tokens.dtype).reshape(
        batch_size, temporal_groups, expected_dim, spatial_pool_size, spatial_pool_size
    )
    return pooled.permute(0, 1, 3, 4, 2).contiguous()


def pool_dual_camera_vjepa_tokens(
    agent_tokens: torch.Tensor,
    wrist_tokens: torch.Tensor,
    *,
    temporal_groups: int,
    vjepa_dim: int = VJEPA_DIM,
    spatial_pool_size: int = SPATIAL_POOL_SIZE,
) -> torch.Tensor:
    if temporal_groups not in (1, 2):
        raise ValueError(f"V5 supports one or two V-JEPA tubelet groups, got {temporal_groups}.")
    expected = (
        CURRENT_TOKEN_COUNT_PER_CAMERA if temporal_groups == 1 else FUTURE_TOKEN_COUNT_PER_CAMERA
    )
    agent_grid = _pool_camera_tokens(
        agent_tokens,
        temporal_groups=temporal_groups,
        expected_tokens=expected,
        expected_dim=vjepa_dim,
        spatial_pool_size=spatial_pool_size,
    )
    wrist_grid = _pool_camera_tokens(
        wrist_tokens,
        temporal_groups=temporal_groups,
        expected_tokens=expected,
        expected_dim=vjepa_dim,
        spatial_pool_size=spatial_pool_size,
    )
    # Sequence contract: time-major, then camera-major, then 6x6 row-major.
    joint = torch.stack((agent_grid, wrist_grid), dim=2)
    batch_size = int(joint.shape[0])
    return joint.reshape(
        batch_size,
        temporal_groups,
        2 * spatial_pool_size * spatial_pool_size,
        vjepa_dim,
    )


def canonicalize_v5_batch(
    batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    required = ("video", "action", "context", "context_mask", "proprio")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"V5 LIBERO batch is missing keys: {missing}.")
    video = batch["video"]
    action = batch["action"]
    context = batch["context"]
    context_mask = batch["context_mask"]
    proprio = batch["proprio"]
    split_dual_camera_video(video)
    if int(video.shape[2]) < 5:
        raise ValueError(f"V5 video must contain at least 5 timesteps, got {video.shape[2]}.")
    if action.ndim != 3 or int(action.shape[1]) < ACTION_HORIZON or int(action.shape[2]) != 7:
        raise ValueError(f"V5 action must contain [B,16,7], got {tuple(action.shape)}.")
    if context.ndim != 3 or int(context.shape[-1]) != 4096:
        raise ValueError(f"V5 context must be [B,L,4096], got {tuple(context.shape)}.")
    if context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
        raise ValueError("V5 context_mask must be [B,L] and match context.")
    context_mask = context_mask.to(dtype=torch.bool)
    if bool((context_mask.sum(dim=1) == 0).any()):
        raise ValueError("Every V5 context sample must contain a valid text token.")
    if proprio.ndim != 3 or int(proprio.shape[-1]) != 8 or int(proprio.shape[1]) < 1:
        raise ValueError(f"V5 proprio must contain [B,T,8], got {tuple(proprio.shape)}.")
    result = {
        "video": video[:, :, :5].to(device=device, dtype=dtype, non_blocking=True),
        "action": action[:, :ACTION_HORIZON].to(device=device, dtype=dtype, non_blocking=True),
        "context": context.to(device=device, dtype=dtype, non_blocking=True),
        "context_mask": context_mask.to(device=device, non_blocking=True),
        "proprio": proprio[:, 0].to(device=device, dtype=dtype, non_blocking=True),
    }
    if "action_is_pad" in batch:
        action_is_pad = batch["action_is_pad"]
        if action_is_pad.ndim != 2 or int(action_is_pad.shape[1]) < ACTION_HORIZON:
            raise ValueError("V5 action_is_pad must contain [B,16].")
        result["action_is_pad"] = action_is_pad[:, :ACTION_HORIZON].to(
            device=device, dtype=torch.bool, non_blocking=True
        )
    return result
