from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ZTaskContextAdapter(nn.Module):
    """Project z_task latents into ActionDiT context tokens.

    This module is intentionally small and low-risk. It does not modify
    ActionDiT, JepaToFastWAMAdapter, or JepaFuturePredictor. It only converts
    a task latent into a context token that can be appended after the JEPA
    adapter output and before ActionDiT.

    Input:
        z_task:
            [B, D_z] or [B, N, D_z].
            By default Stage5 keeps task tokens, so
            [B, 4, 1024] -> [B, 4, context_dim].
            Optional pooling is available for older one-token experiments.

    Output:
        z_task_context_token:
            [B, 1, context_dim] for [B, D_z] input, or [B, N, context_dim]
            for [B, N, D_z] input when pool_tokens=False.
    """

    def __init__(
        self,
        *,
        z_task_dim: int,
        context_dim: int,
        hidden_dim: Optional[int] = None,
        pool_tokens: bool = False,
        gate_init: float = -4.0,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        use_mlp: bool = True,
    ) -> None:
        super().__init__()
        if int(z_task_dim) <= 0:
            raise ValueError(f"`z_task_dim` must be positive, got {z_task_dim}.")
        if int(context_dim) <= 0:
            raise ValueError(f"`context_dim` must be positive, got {context_dim}.")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}.")

        self.z_task_dim = int(z_task_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else max(self.z_task_dim, self.context_dim)
        self.pool_tokens = bool(pool_tokens)
        self.use_layernorm = bool(use_layernorm)
        self.use_mlp = bool(use_mlp)

        self.input_norm = nn.LayerNorm(self.z_task_dim) if self.use_layernorm else nn.Identity()

        if self.use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(self.z_task_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.context_dim),
            )
        else:
            self.proj = nn.Linear(self.z_task_dim, self.context_dim)

        self.output_norm = nn.LayerNorm(self.context_dim) if self.use_layernorm else nn.Identity()
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def gate(self) -> torch.Tensor:
        """Return the scalar sigmoid gate as a tensor."""
        return torch.sigmoid(self.gate_logit)

    def forward(self, z_task: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(z_task):
            raise TypeError(f"`z_task` must be a torch.Tensor, got {type(z_task)}.")

        if z_task.ndim == 2:
            tokens = z_task.unsqueeze(1)
        elif z_task.ndim == 3:
            if int(z_task.shape[1]) <= 0:
                raise ValueError(f"`z_task` token length must be positive, got shape {tuple(z_task.shape)}.")
            tokens = z_task.mean(dim=1, keepdim=True) if self.pool_tokens else z_task
        else:
            raise ValueError(
                "`z_task` must be [B, D_z] or [B, N, D_z], "
                f"got shape {tuple(z_task.shape)}."
            )

        if tokens.shape[-1] != self.z_task_dim:
            raise ValueError(
                "`z_task` last dim mismatch, "
                f"got {tokens.shape[-1]} vs expected {self.z_task_dim}."
            )

        token = self.proj(self.input_norm(tokens))
        token = self.output_norm(token)
        token = token * self.gate().to(device=token.device, dtype=token.dtype)
        return token


def append_z_task_to_context(
    *,
    context: torch.Tensor,
    z_task_context_token: torch.Tensor,
    context_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Append z_task_context_token to an ActionDiT context sequence.

    Args:
        context:
            [B, L, D]
        z_task_context_token:
            [B, 1, D] or [B, K, D]
        context_mask:
            Optional [B, L] bool-like mask. True means valid.

    Returns:
        new_context:
            [B, L + K, D]
        new_context_mask:
            [B, L + K] if context_mask is provided, otherwise None.
    """
    if not torch.is_tensor(context):
        raise TypeError(f"`context` must be a torch.Tensor, got {type(context)}.")
    if not torch.is_tensor(z_task_context_token):
        raise TypeError(
            "`z_task_context_token` must be a torch.Tensor, "
            f"got {type(z_task_context_token)}."
        )

    if context.ndim != 3:
        raise ValueError(f"`context` must be [B, L, D], got shape {tuple(context.shape)}.")
    if z_task_context_token.ndim != 3:
        raise ValueError(
            "`z_task_context_token` must be [B, K, D], "
            f"got shape {tuple(z_task_context_token.shape)}."
        )

    if context.shape[0] != z_task_context_token.shape[0]:
        raise ValueError(
            "Batch size mismatch when appending z_task token: "
            f"context B={context.shape[0]} vs z_task B={z_task_context_token.shape[0]}."
        )
    if context.shape[2] != z_task_context_token.shape[2]:
        raise ValueError(
            "Context dim mismatch when appending z_task token: "
            f"context D={context.shape[2]} vs z_task D={z_task_context_token.shape[2]}."
        )
    if int(z_task_context_token.shape[1]) <= 0:
        raise ValueError(
            "`z_task_context_token` must contain at least one token, "
            f"got shape {tuple(z_task_context_token.shape)}."
        )

    z_task_context_token = z_task_context_token.to(device=context.device, dtype=context.dtype)
    new_context = torch.cat([context, z_task_context_token], dim=1)

    if context_mask is None:
        return new_context, None

    if not torch.is_tensor(context_mask):
        raise TypeError(f"`context_mask` must be a torch.Tensor, got {type(context_mask)}.")
    if context_mask.ndim != 2:
        raise ValueError(f"`context_mask` must be [B, L], got shape {tuple(context_mask.shape)}.")
    if tuple(context_mask.shape) != tuple(context.shape[:2]):
        raise ValueError(
            "`context_mask` must match context [B, L], "
            f"got {tuple(context_mask.shape)} vs {tuple(context.shape[:2])}."
        )

    z_task_mask = torch.ones(
        (context.shape[0], z_task_context_token.shape[1]),
        dtype=torch.bool,
        device=context_mask.device,
    )
    new_context_mask = torch.cat(
        [context_mask.to(device=context_mask.device, dtype=torch.bool), z_task_mask],
        dim=1,
    )
    return new_context, new_context_mask
