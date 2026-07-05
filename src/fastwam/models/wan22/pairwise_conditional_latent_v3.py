from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class CrossAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = None
        if context_mask is not None:
            if context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
                raise ValueError(
                    "context_mask must be [B, L] and match context, "
                    f"got {tuple(context_mask.shape)} vs {tuple(context.shape)}."
                )
            key_padding_mask = ~context_mask.to(dtype=torch.bool, device=context.device)

        context_norm = self.context_norm(context)
        cross_out, _ = self.cross_attn(
            query=self.query_norm(queries),
            key=context_norm,
            value=context_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + cross_out
        self_out, _ = self.self_attn(
            query=self.self_norm(queries),
            key=self.self_norm(queries),
            value=self.self_norm(queries),
            need_weights=False,
        )
        queries = queries + self_out
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class LearnedQueryCrossAttentionPool(nn.Module):
    def __init__(
        self,
        *,
        num_queries: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError(f"context must be [B, L, D], got {tuple(context.shape)}.")
        queries = self.queries.expand(context.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context, context_mask=context_mask)
        return self.output_norm(queries)


class LanguageProjector(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int = 4096,
        latent_dim: int = 1024,
        num_queries: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.text_dim = int(text_dim)
        self.latent_dim = int(latent_dim)
        self.num_queries = int(num_queries)
        self.input_norm = nn.LayerNorm(self.text_dim)
        self.input_proj = nn.Linear(self.text_dim, self.latent_dim)
        self.pool = LearnedQueryCrossAttentionPool(
            num_queries=self.num_queries,
            dim=self.latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_tokens.ndim != 3 or int(text_tokens.shape[1]) != 128 or int(text_tokens.shape[2]) != self.text_dim:
            raise ValueError(
                f"text_tokens must be [B, 128, {self.text_dim}], got {tuple(text_tokens.shape)}."
            )
        x = self.input_proj(self.input_norm(text_tokens))
        return self.pool(x, context_mask=text_mask)


class ActionEncoder(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int = 7,
        action_horizon: int = 32,
        hidden_dim: int = 512,
        latent_dim: int = 1024,
        num_queries: int = 4,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        query_heads: int = 8,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.input_proj = nn.Linear(self.action_dim, self.hidden_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.action_horizon, self.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=temporal_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.pool = LearnedQueryCrossAttentionPool(
            num_queries=num_queries,
            dim=self.hidden_dim,
            num_heads=query_heads,
            num_layers=1,
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.latent_dim)
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 3 or int(action.shape[1]) != self.action_horizon or int(action.shape[2]) != self.action_dim:
            raise ValueError(
                f"action must be [B, {self.action_horizon}, {self.action_dim}], got {tuple(action.shape)}."
            )
        x = self.input_proj(action) + self.temporal_pos.to(device=action.device, dtype=action.dtype)
        x = self.temporal_encoder(x)
        x = self.pool(x)
        return self.output_norm(self.output_proj(x))


class VisionProjector(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 1408,
        token_count: int = 512,
        latent_dim: int = 1024,
        num_queries: int = 4,
        num_heads: int = 8,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_count = int(token_count)
        self.latent_dim = int(latent_dim)
        self.num_queries = int(num_queries)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, self.latent_dim)
        self.pool = LearnedQueryCrossAttentionPool(
            num_queries=self.num_queries,
            dim=self.latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(
        self,
        current_jepa_tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            current_jepa_tokens.ndim != 3
            or int(current_jepa_tokens.shape[1]) != self.token_count
            or int(current_jepa_tokens.shape[2]) != self.input_dim
        ):
            raise ValueError(
                f"current_jepa_tokens must be [B, {self.token_count}, {self.input_dim}], "
                f"got {tuple(current_jepa_tokens.shape)}."
            )
        x = self.input_proj(self.input_norm(current_jepa_tokens))
        return self.pool(x, context_mask=token_mask)


class FusionVL(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        num_tokens: int = 4,
        num_layers: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_tokens = int(num_tokens)
        self.emb_v = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.emb_l = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=int(self.latent_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.latent_dim),
        )
        self.action_pool = LearnedQueryCrossAttentionPool(
            num_queries=self.num_tokens,
            dim=self.latent_dim,
            num_heads=num_heads,
            num_layers=1,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, z_v: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        expected_tail = (self.num_tokens, self.latent_dim)
        if z_v.ndim != 3 or tuple(z_v.shape[1:]) != expected_tail:
            raise ValueError(f"z_v must be [B, {self.num_tokens}, {self.latent_dim}], got {tuple(z_v.shape)}.")
        if z_l.ndim != 3 or tuple(z_l.shape[1:]) != expected_tail:
            raise ValueError(f"z_l must be [B, {self.num_tokens}, {self.latent_dim}], got {tuple(z_l.shape)}.")
        if int(z_v.shape[0]) != int(z_l.shape[0]):
            raise ValueError(f"z_v and z_l batch sizes must match, got {z_v.shape[0]} vs {z_l.shape[0]}.")
        fused = torch.cat([z_v + self.emb_v, z_l + self.emb_l], dim=1)
        fused = self.transformer(fused)
        return self.output_norm(self.action_pool(fused))


class ProprioProjector(nn.Module):
    def __init__(
        self,
        *,
        proprio_dim: int = 8,
        hidden_dim: int = 256,
        latent_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(self.proprio_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.ndim != 2 or int(proprio.shape[1]) != self.proprio_dim:
            raise ValueError(f"proprio must be [B, {self.proprio_dim}], got {tuple(proprio.shape)}.")
        return self.net(proprio).unsqueeze(1)


class FusionVLP(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        num_tokens: int = 4,
        num_heads: int = 8,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_tokens = int(num_tokens)
        self.emb_v = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.emb_l = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.emb_p = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.task_pool = LearnedQueryCrossAttentionPool(
            num_queries=self.num_tokens,
            dim=self.latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, z_v: torch.Tensor, z_l: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        expected_tail = (self.num_tokens, self.latent_dim)
        if z_v.ndim != 3 or tuple(z_v.shape[1:]) != expected_tail:
            raise ValueError(f"z_v must be [B, {self.num_tokens}, {self.latent_dim}], got {tuple(z_v.shape)}.")
        if z_l.ndim != 3 or tuple(z_l.shape[1:]) != expected_tail:
            raise ValueError(f"z_l must be [B, {self.num_tokens}, {self.latent_dim}], got {tuple(z_l.shape)}.")
        if z_p.ndim != 3 or tuple(z_p.shape[1:]) != (1, self.latent_dim):
            raise ValueError(f"z_p must be [B, 1, {self.latent_dim}], got {tuple(z_p.shape)}.")
        if int(z_v.shape[0]) != int(z_l.shape[0]) or int(z_v.shape[0]) != int(z_p.shape[0]):
            raise ValueError(f"z_v/z_l/z_p batch sizes must match, got {z_v.shape[0]}, {z_l.shape[0]}, {z_p.shape[0]}.")
        fused = torch.cat([z_v + self.emb_v, z_l + self.emb_l, z_p + self.emb_p], dim=1)
        return self.output_norm(self.task_pool(fused))


class TextToActionHead(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        num_tokens: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_tokens = int(num_tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=int(self.latent_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.latent_dim),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, int(self.latent_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(self.latent_dim * mlp_ratio), self.latent_dim),
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, z_l: torch.Tensor) -> torch.Tensor:
        if z_l.ndim != 3 or int(z_l.shape[1]) != self.num_tokens or int(z_l.shape[2]) != self.latent_dim:
            raise ValueError(
                f"z_l must be [B, {self.num_tokens}, {self.latent_dim}], got {tuple(z_l.shape)}."
            )
        x = self.transformer(z_l)
        x = x + self.mlp(x)
        return self.output_norm(x)


def _mean_pool_normalize(latent: torch.Tensor, *, name: str) -> torch.Tensor:
    if latent.ndim != 3 or int(latent.shape[1]) != 4:
        raise ValueError(f"{name} must be [B, 4, D], got {tuple(latent.shape)}.")
    pooled = latent.mean(dim=1)
    return F.normalize(pooled, dim=-1, eps=1.0e-6)


def retrieval_accuracy(q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    q_norm = _mean_pool_normalize(q, name="q")
    z_norm = _mean_pool_normalize(z, name="z")
    logits = q_norm @ z_norm.transpose(0, 1)
    return retrieval_accuracy_from_logits(logits)


def retrieval_accuracy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or int(logits.shape[0]) != int(logits.shape[1]):
        raise ValueError(f"logits must be square [B, B], got {tuple(logits.shape)}.")
    labels = torch.arange(logits.shape[0], device=logits.device)
    row_acc = (logits.argmax(dim=1) == labels).float().mean()
    col_acc = (logits.argmax(dim=0) == labels).float().mean()
    return 0.5 * (row_acc + col_acc)


def contrastive_loss(
    q: torch.Tensor,
    z: torch.Tensor,
    *,
    tau: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    if float(tau) <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}.")
    q_norm = _mean_pool_normalize(q, name="q")
    z_norm = _mean_pool_normalize(z, name="z")
    logits = (q_norm @ z_norm.transpose(0, 1)) / float(tau)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)
    acc = retrieval_accuracy_from_logits(logits.detach())
    return loss, acc


def latent_norms(**latents: torch.Tensor) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in latents.items():
        result[name] = float(value.detach().float().norm(dim=-1).mean().item())
    return result


__all__ = [
    "ActionEncoder",
    "FusionVL",
    "FusionVLP",
    "LanguageProjector",
    "LearnedQueryCrossAttentionPool",
    "ProprioProjector",
    "TextToActionHead",
    "VisionProjector",
    "contrastive_loss",
    "latent_norms",
    "retrieval_accuracy",
    "retrieval_accuracy_from_logits",
]
