from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class JepaKVCacheGenerator(nn.Module):
    """Generate final-space, layer-wise video K/V caches from V-JEPA tokens."""

    def __init__(
        self,
        *,
        input_dim: int = 1408,
        context_dim: int = 4096,
        hidden_dim: int = 1024,
        num_layers: int = 30,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        video_seq_len: int = 98,
        layer_rank: int = 16,
        num_cameras: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.cache_dim = self.num_heads * self.attn_head_dim
        self.video_seq_len = int(video_seq_len)
        self.layer_rank = int(layer_rank)
        self.num_cameras = int(num_cameras)
        self.forward_calls = 0

        positive = {
            "input_dim": self.input_dim,
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "attn_head_dim": self.attn_head_dim,
            "video_seq_len": self.video_seq_len,
            "layer_rank": self.layer_rank,
            "num_cameras": self.num_cameras,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"JepaKVCacheGenerator dimensions must be positive: {invalid}.")
        if self.video_seq_len % self.num_cameras != 0:
            raise ValueError(
                "`video_seq_len` must divide evenly across cameras, got "
                f"{self.video_seq_len} and {self.num_cameras}."
            )
        tokens_per_camera = self.video_seq_len // self.num_cameras
        self.camera_grid_side = int(math.isqrt(tokens_per_camera))
        if self.camera_grid_side**2 != tokens_per_camera:
            raise ValueError(
                "Each camera token sequence must form a square grid for horizontal row-major "
                f"layout, got {tokens_per_camera} tokens per camera."
            )

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.target_position = nn.Parameter(
            torch.randn(1, self.video_seq_len, self.hidden_dim) / self.hidden_dim**0.5
        )
        self.camera_type = nn.Parameter(
            torch.randn(self.num_cameras, self.hidden_dim) / self.hidden_dim**0.5
        )

        self.context_norm = nn.LayerNorm(self.context_dim)
        self.film = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
        )
        self.base_k = nn.Linear(self.hidden_dim, self.cache_dim)
        self.base_v = nn.Linear(self.hidden_dim, self.cache_dim)
        self.rank_k = nn.Linear(self.hidden_dim, self.layer_rank)
        self.rank_v = nn.Linear(self.hidden_dim, self.layer_rank)
        self.layer_k = nn.Parameter(
            torch.empty(self.num_layers, self.layer_rank, self.cache_dim)
        )
        self.layer_v = nn.Parameter(
            torch.empty(self.num_layers, self.layer_rank, self.cache_dim)
        )
        nn.init.normal_(self.layer_k, std=0.02 / self.layer_rank**0.5)
        nn.init.normal_(self.layer_v, std=0.02 / self.layer_rank**0.5)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def reset_debug_counters(self) -> None:
        self.forward_calls = 0

    def _camera_embedding(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        camera_ids = torch.arange(self.num_cameras, device=device).repeat_interleave(
            self.camera_grid_side
        )
        camera_ids = camera_ids.repeat(self.camera_grid_side)
        return self.camera_type.to(device=device, dtype=dtype)[camera_ids].unsqueeze(0)

    def _masked_context(self, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"`context` must be [B,L,{self.context_dim}], got {tuple(context.shape)}."
            )
        if context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError(
                "`context_mask` must match context [B,L], got "
                f"{tuple(context_mask.shape)} vs {tuple(context.shape[:2])}."
            )
        mask = context_mask.to(device=context.device, dtype=torch.bool)
        valid_count = mask.sum(dim=1)
        if bool((valid_count == 0).any()):
            raise ValueError("Every sample must contain at least one valid context token.")
        weights = mask.to(dtype=context.dtype).unsqueeze(-1)
        pooled = (context * weights).sum(dim=1) / valid_count.to(context.dtype).unsqueeze(-1)
        return self.context_norm(pooled)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        self.forward_calls += 1
        if visual_tokens.ndim != 3 or tuple(visual_tokens.shape[1:]) != (
            self.video_seq_len,
            self.input_dim,
        ):
            raise ValueError(
                "`visual_tokens` must be "
                f"[B,{self.video_seq_len},{self.input_dim}], got {tuple(visual_tokens.shape)}."
            )
        if context.shape[0] != visual_tokens.shape[0]:
            raise ValueError("Visual and context batch sizes must match.")

        h = self.input_projection(self.input_norm(visual_tokens))
        h = h + self.target_position.to(device=h.device, dtype=h.dtype)
        h = h + self._camera_embedding(device=h.device, dtype=h.dtype)
        scale, shift = self.film(self._masked_context(context, context_mask)).chunk(2, dim=-1)
        h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        base_k = self.base_k(h)
        base_v = self.base_v(h)
        rank_k = self.rank_k(h)
        rank_v = self.rank_v(h)
        cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            delta_k = torch.einsum("bsr,rd->bsd", rank_k, self.layer_k[layer_idx])
            delta_v = torch.einsum("bsr,rd->bsd", rank_v, self.layer_v[layer_idx])
            cache.append({"k": base_k + delta_k, "v": base_v + delta_v})
        return cache


def kv_cache_distillation_loss(
    student_cache: list[dict[str, torch.Tensor]],
    teacher_cache: list[dict[str, torch.Tensor]],
    *,
    lambda_cos: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(student_cache) != len(teacher_cache) or not student_cache:
        raise ValueError(
            "Student and teacher cache lists must have the same non-zero length, got "
            f"{len(student_cache)} and {len(teacher_cache)}."
        )
    loss_k: list[torch.Tensor] = []
    loss_v: list[torch.Tensor] = []
    cosine: list[torch.Tensor] = []
    for layer_idx, (student, teacher) in enumerate(zip(student_cache, teacher_cache)):
        for key in ("k", "v"):
            if key not in student or key not in teacher:
                raise ValueError(f"Cache layer {layer_idx} is missing {key!r}.")
            if student[key].shape != teacher[key].shape:
                raise ValueError(
                    f"Cache layer {layer_idx} {key} shape mismatch: "
                    f"{tuple(student[key].shape)} vs {tuple(teacher[key].shape)}."
                )
        student_k = student["k"].float()
        student_v = student["v"].float()
        teacher_k = teacher["k"].detach().float()
        teacher_v = teacher["v"].detach().float()
        loss_k.append(torch.nn.functional.smooth_l1_loss(student_k, teacher_k))
        loss_v.append(torch.nn.functional.smooth_l1_loss(student_v, teacher_v))
        cos_k = torch.nn.functional.cosine_similarity(student_k, teacher_k, dim=-1).mean()
        cos_v = torch.nn.functional.cosine_similarity(student_v, teacher_v, dim=-1).mean()
        cosine.append(0.5 * (cos_k + cos_v))

    mean_k = torch.stack(loss_k).mean()
    mean_v = torch.stack(loss_v).mean()
    mean_cosine = torch.stack(cosine).mean()
    loss_cos = 1.0 - mean_cosine
    total = mean_k + mean_v + float(lambda_cos) * loss_cos
    middle = len(cosine) // 2
    return total, {
        "loss_k": mean_k,
        "loss_v": mean_v,
        "loss_cos": loss_cos,
        "cos_first": cosine[0],
        "cos_middle": cosine[middle],
        "cos_last": cosine[-1],
    }
