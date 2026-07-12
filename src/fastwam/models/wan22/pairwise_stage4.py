from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from fastwam.models.wan22 import pairwise_conditional_latent_v3 as latent_v3
from fastwam.models.wan22.pairwise_conditional_latent_v3 import (
    ActionEncoder,
    LanguageProjector,
    LearnedQueryCrossAttentionPool,
    TextToActionHead,
    contrastive_loss,
)


VisionProjectorBase = getattr(latent_v3, "VisionProjector", None)
FusionVLBase = getattr(latent_v3, "FusionVL", None)


def mean_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tokens):
        raise TypeError(f"`tokens` must be a torch.Tensor, got {type(tokens)}.")
    if tokens.ndim != 3:
        raise ValueError(f"`tokens` must be [B, K, D], got shape {tuple(tokens.shape)}.")
    if int(tokens.shape[1]) <= 0:
        raise ValueError(f"`tokens` must contain at least one token, got shape {tuple(tokens.shape)}.")
    return tokens.mean(dim=1)


def _require_shape(value: torch.Tensor, shape: tuple[int | None, ...], *, name: str) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"`{name}` must be a torch.Tensor, got {type(value)}.")
    if value.ndim != len(shape):
        raise ValueError(f"`{name}` must have {len(shape)} dims, got shape {tuple(value.shape)}.")
    for idx, expected in enumerate(shape):
        if expected is not None and int(value.shape[idx]) != int(expected):
            raise ValueError(
                f"`{name}` shape mismatch at dim {idx}: got {tuple(value.shape)} vs expected {shape}."
            )


class _FallbackVisionProjector(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 1408,
        token_count: int = 512,
        latent_dim: int = 1024,
        num_queries: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.world_dim = int(input_dim)
        self.token_count = int(token_count)
        self.latent_dim = int(latent_dim)
        self.num_queries = int(num_queries)
        self.input_norm = nn.LayerNorm(self.world_dim)
        self.input_proj = nn.Linear(self.world_dim, self.latent_dim)
        self.pool = LearnedQueryCrossAttentionPool(
            num_queries=self.num_queries,
            dim=self.latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(self, world_tokens: torch.Tensor) -> torch.Tensor:
        _require_shape(world_tokens, (None, self.token_count, self.world_dim), name="world_tokens")
        x = self.input_proj(self.input_norm(world_tokens))
        return self.pool(x)


class _FallbackFusionVL(nn.Module):
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
        self.type_embed = nn.Parameter(torch.zeros(1, 2, self.latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=self.latent_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.latent_dim),
        )
        self.pool = LearnedQueryCrossAttentionPool(
            num_queries=self.num_tokens,
            dim=self.latent_dim,
            num_heads=num_heads,
            num_layers=1,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, z_v: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
        _require_shape(z_v, (None, self.num_tokens, self.latent_dim), name="z_v")
        _require_shape(z_l, (None, self.num_tokens, self.latent_dim), name="z_l")
        tokens = torch.cat(
            [
                z_v + self.type_embed[:, 0:1, :].to(device=z_v.device, dtype=z_v.dtype),
                z_l + self.type_embed[:, 1:2, :].to(device=z_l.device, dtype=z_l.dtype),
            ],
            dim=1,
        )
        tokens = self.encoder(tokens)
        return self.output_norm(self.pool(tokens))


class ProprioProjector(nn.Module):
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
        self.input_proj = nn.LazyLinear(self.latent_dim)
        self.input_norm = nn.LayerNorm(self.latent_dim)
        self.pool = LearnedQueryCrossAttentionPool(
            num_queries=self.num_tokens,
            dim=self.latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(proprio):
            raise TypeError(f"`proprio` must be a torch.Tensor, got {type(proprio)}.")
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        elif proprio.ndim != 3:
            raise ValueError(f"`proprio` must be [B, D] or [B, T, D], got shape {tuple(proprio.shape)}.")
        x = self.input_proj(proprio)
        x = self.input_norm(x)
        return self.pool(x)


class FusionVLP(nn.Module):
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
        self.type_embed = nn.Parameter(torch.zeros(1, 3, self.latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=self.latent_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.latent_dim),
        )
        self.pool = LearnedQueryCrossAttentionPool(
            num_queries=self.num_tokens,
            dim=self.latent_dim,
            num_heads=num_heads,
            num_layers=1,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def forward(
        self,
        *,
        z_v: torch.Tensor,
        z_l: torch.Tensor,
        z_p: torch.Tensor,
        residual_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require_shape(z_v, (None, self.num_tokens, self.latent_dim), name="z_v")
        _require_shape(z_l, (None, self.num_tokens, self.latent_dim), name="z_l")
        _require_shape(z_p, (None, self.num_tokens, self.latent_dim), name="z_p")
        tokens = torch.cat(
            [
                z_v + self.type_embed[:, 0:1, :].to(device=z_v.device, dtype=z_v.dtype),
                z_l + self.type_embed[:, 1:2, :].to(device=z_l.device, dtype=z_l.dtype),
                z_p + self.type_embed[:, 2:3, :].to(device=z_p.device, dtype=z_p.dtype),
            ],
            dim=1,
        )
        tokens = self.encoder(tokens)
        pooled = self.pool(tokens)
        if residual_tokens is not None:
            _require_shape(
                residual_tokens,
                (None, self.num_tokens, self.latent_dim),
                name="residual_tokens",
            )
            pooled = pooled + residual_tokens
        return self.output_norm(pooled)


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
        self.type_embed = nn.Parameter(torch.zeros(1, 2, self.latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=self.latent_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.latent_dim),
        )
        self.pool = LearnedQueryCrossAttentionPool(
            num_queries=self.num_tokens,
            dim=self.latent_dim,
            num_heads=num_heads,
            num_layers=1,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, *, z_v: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        _require_shape(z_v, (None, self.num_tokens, self.latent_dim), name="z_v")
        _require_shape(z_a, (None, self.num_tokens, self.latent_dim), name="z_a")
        tokens = torch.cat(
            [
                z_v + self.type_embed[:, 0:1, :].to(device=z_v.device, dtype=z_v.dtype),
                z_a + self.type_embed[:, 1:2, :].to(device=z_a.device, dtype=z_a.dtype),
            ],
            dim=1,
        )
        tokens = self.encoder(tokens)
        return self.output_norm(self.pool(tokens))


class PairwiseStage4Model(nn.Module):
    def __init__(
        self,
        *,
        world_dim: int = 1408,
        text_dim: int = 4096,
        action_dim: int = 7,
        latent_dim: int = 1024,
        num_latent_tokens: int = 4,
        world_tokens_len: int = 512,
        text_tokens_len: int = 128,
        action_horizon: int = 32,
        tau: float = 0.07,
    ) -> None:
        super().__init__()
        if int(world_dim) != 1408:
            raise ValueError(f"Stage4 expects world_dim=1408, got {world_dim}.")
        if int(text_dim) != 4096:
            raise ValueError(f"Stage4 expects text_dim=4096, got {text_dim}.")
        if int(action_dim) != 7:
            raise ValueError(f"Stage4 expects action_dim=7, got {action_dim}.")
        if int(latent_dim) != 1024:
            raise ValueError(f"Stage4 expects latent_dim=1024, got {latent_dim}.")
        if int(num_latent_tokens) != 4:
            raise ValueError(f"Stage4 expects num_latent_tokens=4, got {num_latent_tokens}.")
        if int(world_tokens_len) != 512:
            raise ValueError(f"Stage4 expects world_tokens_len=512, got {world_tokens_len}.")
        if int(text_tokens_len) != 128:
            raise ValueError(f"Stage4 expects text_tokens_len=128, got {text_tokens_len}.")
        if int(action_horizon) != 32:
            raise ValueError(f"Stage4 expects action_horizon=32, got {action_horizon}.")
        if float(tau) <= 0.0:
            raise ValueError(f"`tau` must be positive, got {tau}.")

        self.world_dim = int(world_dim)
        self.text_dim = int(text_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.num_latent_tokens = int(num_latent_tokens)
        self.world_tokens_len = int(world_tokens_len)
        self.world_token_count = self.world_tokens_len
        self.text_tokens_len = int(text_tokens_len)
        self.action_horizon = int(action_horizon)
        self.tau = float(tau)

        vision_projector_cls = VisionProjectorBase or _FallbackVisionProjector
        fusion_vl_cls = FusionVLBase or _FallbackFusionVL

        self.vision_projector = vision_projector_cls(
            input_dim=self.world_dim,
            token_count=self.world_token_count,
            latent_dim=self.latent_dim,
            num_queries=self.num_latent_tokens,
        )
        self.language_projector = LanguageProjector(
            text_dim=self.text_dim,
            latent_dim=self.latent_dim,
            num_queries=self.num_latent_tokens,
        )
        self.proprio_projector = ProprioProjector(
            latent_dim=self.latent_dim,
            num_tokens=self.num_latent_tokens,
        )
        self.action_encoder = ActionEncoder(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            latent_dim=self.latent_dim,
            num_queries=self.num_latent_tokens,
        )
        self.fusion_vl = fusion_vl_cls(
            latent_dim=self.latent_dim,
            num_tokens=self.num_latent_tokens,
        )
        self.fusion_vlp = FusionVLP(
            latent_dim=self.latent_dim,
            num_tokens=self.num_latent_tokens,
        )
        self.fusion_va = FusionVA(
            latent_dim=self.latent_dim,
            num_tokens=self.num_latent_tokens,
        )
        self.vlp_to_action_head = TextToActionHead(
            latent_dim=self.latent_dim,
            num_tokens=self.num_latent_tokens,
        )
        self.va_to_language_head = TextToActionHead(
            latent_dim=self.latent_dim,
            num_tokens=self.num_latent_tokens,
        )

    def _prepare_world_tokens(self, world_tokens: torch.Tensor) -> torch.Tensor:
        _require_shape(
            world_tokens,
            (None, self.world_tokens_len, self.world_dim),
            name="world_tokens",
        )
        return world_tokens

    def _prepare_text(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _require_shape(
            text_tokens,
            (None, self.text_tokens_len, self.text_dim),
            name="text_tokens",
        )
        if text_mask is None:
            return text_tokens, None
        _require_shape(text_mask, (text_tokens.shape[0], self.text_tokens_len), name="text_mask")
        return text_tokens, text_mask.to(device=text_tokens.device, dtype=torch.bool)

    def _prepare_action(self, action_chunk: torch.Tensor) -> torch.Tensor:
        _require_shape(
            action_chunk,
            (None, self.action_horizon, self.action_dim),
            name="action_chunk",
        )
        return action_chunk

    def _zero_proprio_tokens(self, *, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros((batch_size, self.num_latent_tokens, self.latent_dim))

    def _prepare_proprio_tokens(
        self,
        *,
        batch_size: int,
        reference: torch.Tensor,
        proprio: torch.Tensor | None,
        proprio_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if proprio_tokens is not None:
            _require_shape(
                proprio_tokens,
                (batch_size, self.num_latent_tokens, self.latent_dim),
                name="proprio_tokens",
            )
            return proprio_tokens.to(device=reference.device, dtype=reference.dtype)
        if proprio is not None:
            proprio = proprio.to(device=reference.device, dtype=reference.dtype)
            return self.proprio_projector(proprio)
        return self._zero_proprio_tokens(batch_size=batch_size, reference=reference)

    def forward_train(
        self,
        *,
        world_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        action_chunk: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        proprio_tokens: torch.Tensor | None = None,
        tau: float | None = None,
    ) -> dict[str, torch.Tensor]:
        world_tokens = self._prepare_world_tokens(world_tokens)
        text_tokens, text_mask = self._prepare_text(text_tokens, text_mask)
        action_chunk = self._prepare_action(action_chunk)

        z_v = self.vision_projector(world_tokens)
        z_l = self.language_projector(text_tokens, text_mask=text_mask)
        z_p = self._prepare_proprio_tokens(
            batch_size=int(world_tokens.shape[0]),
            reference=world_tokens,
            proprio=proprio,
            proprio_tokens=proprio_tokens,
        )
        z_a = self.action_encoder(action_chunk)

        z_vl = self.fusion_vl(z_v, z_l)
        vlp_tokens = self.fusion_vlp(z_v=z_v, z_l=z_l, z_p=z_p, residual_tokens=z_vl)
        va_tokens = self.fusion_va(z_v=z_v, z_a=z_a)

        q_a_vlp = self.vlp_to_action_head(vlp_tokens)
        q_l_va = self.va_to_language_head(va_tokens)

        loss_vlp_to_a, retrieval_acc_vlp_to_a = contrastive_loss(
            q_a_vlp,
            z_a,
            tau=float(self.tau if tau is None else tau),
        )
        loss_va_to_l, retrieval_acc_va_to_l = contrastive_loss(
            q_l_va,
            z_l,
            tau=float(self.tau if tau is None else tau),
        )

        z_task_token = q_a_vlp
        z_task = mean_pool_tokens(z_task_token)

        return {
            "z_v": z_v,
            "z_l": z_l,
            "z_p": z_p,
            "z_a": z_a,
            "q_a_vlp": q_a_vlp,
            "q_l_va": q_l_va,
            "z_task": z_task,
            "z_task_token": z_task_token,
            "loss_vlp_to_a": loss_vlp_to_a,
            "loss_va_to_l": loss_va_to_l,
            "retrieval_acc_vlp_to_a": retrieval_acc_vlp_to_a,
            "retrieval_acc_va_to_l": retrieval_acc_va_to_l,
        }

    def forward_infer(
        self,
        *,
        world_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        proprio_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        world_tokens = self._prepare_world_tokens(world_tokens)
        text_tokens, text_mask = self._prepare_text(text_tokens, text_mask)

        z_v = self.vision_projector(world_tokens)
        z_l = self.language_projector(text_tokens, text_mask=text_mask)
        z_p = self._prepare_proprio_tokens(
            batch_size=int(world_tokens.shape[0]),
            reference=world_tokens,
            proprio=proprio,
            proprio_tokens=proprio_tokens,
        )
        z_vl = self.fusion_vl(z_v, z_l)
        vlp_tokens = self.fusion_vlp(z_v=z_v, z_l=z_l, z_p=z_p, residual_tokens=z_vl)
        q_a_vlp = self.vlp_to_action_head(vlp_tokens)
        z_task_token = q_a_vlp
        z_task = mean_pool_tokens(z_task_token)

        return {
            "z_v": z_v,
            "z_l": z_l,
            "z_p": z_p,
            "q_a_vlp": q_a_vlp,
            "z_task": z_task,
            "z_task_token": z_task_token,
        }

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        if "action_chunk" in kwargs and kwargs["action_chunk"] is not None:
            return self.forward_train(*args, **kwargs)
        return self.forward_infer(*args, **kwargs)


def load_stage4_checkpoint(
    model: PairwiseStage4Model,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    if not isinstance(model, PairwiseStage4Model):
        raise TypeError(f"`model` must be PairwiseStage4Model, got {type(model)}.")

    path = Path(checkpoint_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Stage4 checkpoint does not exist: {path}")

    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Stage4 checkpoint must be a dict, got {type(checkpoint)}.")

    full_state = None
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            full_state = value
            break
    if full_state is not None:
        incompatible = model.load_state_dict(full_state, strict=strict)
        return {
            "source": "full_state_dict",
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "loaded_keys_count": len(full_state),
        }

    module_names = (
        "vision_projector",
        "language_projector",
        "proprio_projector",
        "action_encoder",
        "fusion_vl",
        "fusion_vlp",
        "fusion_va",
        "vlp_to_action_head",
        "va_to_language_head",
    )
    stats: dict[str, Any] = {}
    for name in module_names:
        module = getattr(model, name)
        state = checkpoint.get(name)
        if not isinstance(state, dict):
            if strict:
                raise ValueError(f"Stage4 checkpoint missing `{name}` state_dict: {path}")
            stats[name] = {
                "loaded": False,
                "missing": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "loaded_keys_count": 0,
            }
            continue
        incompatible = module.load_state_dict(state, strict=strict)
        stats[name] = {
            "loaded": True,
            "missing": False,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "loaded_keys_count": len(state),
        }
    return stats


__all__ = [
    "FusionVA",
    "FusionVLP",
    "PairwiseStage4Model",
    "ProprioProjector",
    "load_stage4_checkpoint",
    "mean_pool_tokens",
]
