from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class JepaToFastWAMAdapter(nn.Module):
    """Convert V-JEPA current/future tokens into ActionDiT context tokens.

    Shape contract:
        current_jepa_tokens: [B, N_c, D_v]
        future_jepa_tokens:  [B, N_f, D_v], optional
        base_context:        [B, L, D_t], optional text/proprio context

    Output:
        context:      [B, L + R_c + R_f, D_t]
        context_mask: [B, L + R_c + R_f], True means valid token
    """

    def __init__(
        self,
        *,
        vjepa_dim: int,
        text_dim: int,
        num_current_context_tokens: int = 64,
        num_future_context_tokens: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vjepa_dim = int(vjepa_dim)
        self.text_dim = int(text_dim)
        self.num_current_context_tokens = int(num_current_context_tokens)
        self.num_future_context_tokens = int(num_future_context_tokens)

        if self.vjepa_dim <= 0:
            raise ValueError(f"`vjepa_dim` must be positive, got {self.vjepa_dim}.")
        if self.text_dim <= 0:
            raise ValueError(f"`text_dim` must be positive, got {self.text_dim}.")
        if self.num_current_context_tokens <= 0:
            raise ValueError(
                "`num_current_context_tokens` must be positive, "
                f"got {self.num_current_context_tokens}."
            )
        if self.num_future_context_tokens < 0:
            raise ValueError(
                "`num_future_context_tokens` must be non-negative, "
                f"got {self.num_future_context_tokens}."
            )

        self.current_projection = nn.Linear(self.vjepa_dim, self.text_dim)
        self.future_projection = nn.Linear(self.vjepa_dim, self.text_dim)
        self.current_norm = nn.LayerNorm(self.text_dim)
        self.future_norm = nn.LayerNorm(self.text_dim)
        self.current_type_embedding = nn.Parameter(torch.zeros(1, 1, self.text_dim))
        self.future_type_embedding = nn.Parameter(torch.zeros(1, 1, self.text_dim))
        nn.init.trunc_normal_(self.current_type_embedding, std=0.02)
        nn.init.trunc_normal_(self.future_type_embedding, std=0.02)
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _resample_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if target_tokens == 0:
            return tokens[:, :0]
        if tokens.shape[1] == target_tokens:
            return tokens
        # Adaptive average pooling over token dimension, preserving feature dim.
        x = tokens.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, target_tokens)
        return x.transpose(1, 2)

    def _validate_jepa_tokens(self, tokens: torch.Tensor, *, name: str) -> None:
        if tokens.ndim != 3:
            raise ValueError(f"`{name}` must be [B, N, D_v], got shape {tuple(tokens.shape)}.")
        if tokens.shape[2] != self.vjepa_dim:
            raise ValueError(
                f"`{name}` last dim must be {self.vjepa_dim}, got {tokens.shape[2]}."
            )

    def forward(
        self,
        *,
        current_jepa_tokens: torch.Tensor,
        future_jepa_tokens: Optional[torch.Tensor] = None,
        base_context: Optional[torch.Tensor] = None,
        base_context_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_jepa_tokens(current_jepa_tokens, name="current_jepa_tokens")
        batch_size = int(current_jepa_tokens.shape[0])
        device = current_jepa_tokens.device

        if future_jepa_tokens is not None:
            self._validate_jepa_tokens(future_jepa_tokens, name="future_jepa_tokens")
            if future_jepa_tokens.shape[0] != batch_size:
                raise ValueError(
                    "`future_jepa_tokens` batch size must match current tokens, "
                    f"got {future_jepa_tokens.shape[0]} vs {batch_size}."
                )

        if base_context_mask is not None and base_context is None:
            raise ValueError("`base_context_mask` was provided without `base_context`.")
        if base_context is not None:
            if base_context.ndim != 3:
                raise ValueError(
                    "`base_context` must be [B, L, D_t], "
                    f"got shape {tuple(base_context.shape)}."
                )
            if base_context.shape[0] != batch_size or base_context.shape[2] != self.text_dim:
                raise ValueError(
                    "`base_context` must match [B, L, text_dim], "
                    f"got {tuple(base_context.shape)} with B={batch_size}, D_t={self.text_dim}."
                )
            base_context = base_context.to(device=device, dtype=current_jepa_tokens.dtype)
            if base_context_mask is None:
                base_context_mask = torch.ones(
                    (batch_size, base_context.shape[1]),
                    dtype=torch.bool,
                    device=device,
                )
            else:
                if base_context_mask.ndim != 2 or tuple(base_context_mask.shape) != tuple(base_context.shape[:2]):
                    raise ValueError(
                        "`base_context_mask` must match base_context [B, L], "
                        f"got {tuple(base_context_mask.shape)} vs {tuple(base_context.shape[:2])}."
                    )
                base_context_mask = base_context_mask.to(device=device, dtype=torch.bool)

        current_context = self.current_projection(current_jepa_tokens)
        current_context = self._resample_tokens(current_context, self.num_current_context_tokens)
        current_context = self.current_norm(current_context + self.current_type_embedding.to(
            device=current_context.device,
            dtype=current_context.dtype,
        ))
        current_context = self.dropout(current_context)
        current_mask = torch.ones(
            (batch_size, current_context.shape[1]),
            dtype=torch.bool,
            device=device,
        )

        context_parts = []
        mask_parts = []
        if base_context is not None:
            context_parts.append(base_context)
            mask_parts.append(base_context_mask)
        context_parts.append(current_context)
        mask_parts.append(current_mask)

        if future_jepa_tokens is not None and self.num_future_context_tokens > 0:
            future_context = self.future_projection(future_jepa_tokens.to(
                device=device,
                dtype=current_jepa_tokens.dtype,
            ))
            future_context = self._resample_tokens(future_context, self.num_future_context_tokens)
            future_context = self.future_norm(future_context + self.future_type_embedding.to(
                device=future_context.device,
                dtype=future_context.dtype,
            ))
            future_context = self.dropout(future_context)
            future_mask = torch.ones(
                (batch_size, future_context.shape[1]),
                dtype=torch.bool,
                device=device,
            )
            context_parts.append(future_context)
            mask_parts.append(future_mask)

        return torch.cat(context_parts, dim=1), torch.cat(mask_parts, dim=1)
