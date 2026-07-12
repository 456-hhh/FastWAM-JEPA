from __future__ import annotations

import torch
from torch import nn

from fastwam.models.wan22.pairwise_conditional_latent_v3 import (
    ActionEncoder,
    FusionVLP,
    LanguageProjector,
    LearnedQueryCrossAttentionPool,
    ProprioProjector,
    VisionProjector,
    contrastive_loss,
)


def _require_shape(
    value: torch.Tensor,
    shape: tuple[int | None, ...],
    *,
    name: str,
) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)}.")
    if value.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dims, got {tuple(value.shape)}.")
    for dim, expected in enumerate(shape):
        if expected is not None and int(value.shape[dim]) != int(expected):
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")


class FusionVA(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        num_tokens: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_tokens = int(num_tokens)
        self.vision_embedding = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.action_embedding = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(num_heads),
            dim_feedforward=self.latent_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.latent_dim),
        )
        self.language_pool = LearnedQueryCrossAttentionPool(
            num_queries=self.num_tokens,
            dim=self.latent_dim,
            num_heads=int(num_heads),
            num_layers=1,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, z_v: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        expected = (None, self.num_tokens, self.latent_dim)
        _require_shape(z_v, expected, name="z_v")
        _require_shape(z_a, expected, name="z_a")
        if int(z_v.shape[0]) != int(z_a.shape[0]):
            raise ValueError(f"z_v/z_a batch sizes differ: {z_v.shape[0]} vs {z_a.shape[0]}.")
        if z_v.device != z_a.device:
            raise ValueError(f"z_v/z_a devices differ: {z_v.device} vs {z_a.device}.")
        if z_v.dtype != z_a.dtype:
            raise ValueError(f"z_v/z_a dtypes differ: {z_v.dtype} vs {z_a.dtype}.")

        fused = torch.cat(
            [
                z_v + self.vision_embedding.to(device=z_v.device, dtype=z_v.dtype),
                z_a + self.action_embedding.to(device=z_a.device, dtype=z_a.dtype),
            ],
            dim=1,
        )
        fused = self.encoder(fused)
        q_l = self.output_norm(self.language_pool(fused))
        _require_shape(q_l, expected, name="q_l")
        return q_l


class Stage4VLPVAActionModel(nn.Module):
    def __init__(
        self,
        *,
        raw_vjepa_tokens: int = 512,
        vjepa_dim: int = 1408,
        context_tokens: int = 128,
        action_horizon: int = 32,
        proprio_dim: int = 8,
    ) -> None:
        super().__init__()
        self.raw_vjepa_tokens = int(raw_vjepa_tokens)
        self.vjepa_dim = int(vjepa_dim)
        self.context_tokens = int(context_tokens)
        self.action_horizon = int(action_horizon)
        self.proprio_dim = int(proprio_dim)

        self.language_projector = LanguageProjector()
        self.action_encoder = ActionEncoder()
        self.vision_projector = VisionProjector(
            input_dim=self.vjepa_dim,
            token_count=self.raw_vjepa_tokens,
        )
        self.proprio_projector = ProprioProjector(proprio_dim=self.proprio_dim)
        self.fusion_vlp = FusionVLP()
        self.fusion_va = FusionVA()

    def forward(
        self,
        *,
        current_jepa_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        tau: float,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(current_jepa_tokens.shape[0])
        _require_shape(
            current_jepa_tokens,
            (None, self.raw_vjepa_tokens, self.vjepa_dim),
            name="current_jepa_tokens",
        )
        _require_shape(
            context,
            (batch_size, self.context_tokens, 4096),
            name="context",
        )
        _require_shape(
            context_mask,
            (batch_size, self.context_tokens),
            name="context_mask",
        )
        if context_mask.dtype != torch.bool:
            raise ValueError(f"context_mask must be bool with True=valid, got {context_mask.dtype}.")
        _require_shape(proprio, (batch_size, self.proprio_dim), name="proprio")
        _require_shape(action, (batch_size, self.action_horizon, 7), name="action")

        z_v = self.vision_projector(current_jepa_tokens)
        z_l = self.language_projector(context, text_mask=context_mask)
        z_p = self.proprio_projector(proprio)
        z_task = self.fusion_vlp(z_v, z_l, z_p)
        z_a = self.action_encoder(action)
        q_l = self.fusion_va(z_v, z_a)

        expected_latent = (batch_size, 4, 1024)
        _require_shape(z_v, expected_latent, name="z_v")
        _require_shape(z_l, expected_latent, name="z_l")
        _require_shape(z_p, (batch_size, 1, 1024), name="z_p")
        _require_shape(z_task, expected_latent, name="z_task")
        _require_shape(z_a, expected_latent, name="z_a")
        _require_shape(q_l, expected_latent, name="q_l")

        loss_vlp_a, retrieval_vlp_a = contrastive_loss(z_task, z_a, tau=float(tau))
        loss_va_l, retrieval_va_l = contrastive_loss(q_l, z_l, tau=float(tau))
        return {
            "z_v": z_v,
            "z_l": z_l,
            "z_p": z_p,
            "z_task": z_task,
            "z_a": z_a,
            "q_l": q_l,
            "loss_vlp_a": loss_vlp_a,
            "loss_va_l": loss_va_l,
            "retrieval_vlp_a": retrieval_vlp_a,
            "retrieval_va_l": retrieval_va_l,
        }


__all__ = ["FusionVA", "Stage4VLPVAActionModel"]
