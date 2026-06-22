from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class JepaFuturePredictorBlock(nn.Module):
    """Lightweight JEPA future block.

    Shape contract:
        future tokens:    [B, N_f, D_h]
        current tokens:   [B, N_c, D_h]
        condition tokens: [B, L_c, D_h], optional

    TODO(v2-real-trunk): replace or initialize this block from the official
    V-JEPA2-AC predictor trunk blocks when the checkpoint/key mapping is wired.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_current = nn.LayerNorm(hidden_dim)
        self.current_cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_condition = nn.LayerNorm(hidden_dim)
        self.condition_cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        future_tokens: torch.Tensor,
        *,
        current_tokens: torch.Tensor,
        condition_tokens: Optional[torch.Tensor] = None,
        condition_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_attn_input = self.norm_self(future_tokens)
        self_attn_out, _ = self.self_attn(
            self_attn_input,
            self_attn_input,
            self_attn_input,
            need_weights=False,
        )
        future_tokens = future_tokens + self_attn_out

        current_attn_input = self.norm_current(future_tokens)
        current_attn_out, _ = self.current_cross_attn(
            current_attn_input,
            current_tokens,
            current_tokens,
            need_weights=False,
        )
        future_tokens = future_tokens + current_attn_out

        if condition_tokens is not None:
            condition_attn_input = self.norm_condition(future_tokens)
            condition_attn_out, _ = self.condition_cross_attn(
                condition_attn_input,
                condition_tokens,
                condition_tokens,
                key_padding_mask=condition_key_padding_mask,
                need_weights=False,
            )
            future_tokens = future_tokens + condition_attn_out

        return future_tokens + self.mlp(self.norm_mlp(future_tokens))


class JepaFuturePredictor(nn.Module):
    """Predict future V-JEPA tokens without consuming action/noisy-action tokens.

    Inputs:
        current_jepa_tokens: [B, N_c, D_v]
        condition_context:   [B, L_c, D_t], optional text/proprio context
        condition_mask:      [B, L_c], optional, True means valid token

    Outputs:
        future_hidden_tokens: [B, N_f, D_h]
        pred_future_tokens:   [B, N_f, D_v]
    """

    def __init__(
        self,
        *,
        vjepa_dim: int,
        hidden_dim: int,
        num_future_tokens: int,
        text_dim: Optional[int] = None,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vjepa_dim = int(vjepa_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_future_tokens = int(num_future_tokens)
        self.text_dim = None if text_dim is None else int(text_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        if self.vjepa_dim <= 0:
            raise ValueError(f"`vjepa_dim` must be positive, got {self.vjepa_dim}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {self.hidden_dim}.")
        if self.num_future_tokens <= 0:
            raise ValueError(f"`num_future_tokens` must be positive, got {self.num_future_tokens}.")
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {self.num_layers}.")
        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be positive, got {self.num_heads}.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "`hidden_dim` must be divisible by `num_heads`, "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )
        if ffn_dim is None:
            ffn_dim = self.hidden_dim * 4

        self.predictor_embed = nn.Linear(self.vjepa_dim, self.hidden_dim)
        self.future_query_tokens = nn.Parameter(torch.zeros(self.num_future_tokens, self.hidden_dim))
        nn.init.trunc_normal_(self.future_query_tokens, std=0.02)

        self.condition_projection = (
            nn.Linear(self.text_dim, self.hidden_dim) if self.text_dim is not None else None
        )
        self.predictor_blocks = nn.ModuleList(
            [
                JepaFuturePredictorBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    ffn_dim=int(ffn_dim),
                    dropout=float(dropout),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.predictor_norm = nn.LayerNorm(self.hidden_dim)
        self.predictor_proj = nn.Linear(self.hidden_dim, self.vjepa_dim)

    def _validate_condition(
        self,
        *,
        condition_context: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> None:
        if condition_mask is not None and condition_context is None:
            raise ValueError("`condition_mask` was provided without `condition_context`.")
        if condition_context is None:
            return
        if self.condition_projection is None or self.text_dim is None:
            raise ValueError("`text_dim` must be set to use `condition_context`.")
        if condition_context.ndim != 3:
            raise ValueError(
                "`condition_context` must be [B, L_c, D_t], "
                f"got shape {tuple(condition_context.shape)}."
            )
        if condition_context.shape[0] != batch_size:
            raise ValueError(
                "`condition_context` batch size mismatch, "
                f"got {condition_context.shape[0]} vs {batch_size}."
            )
        if condition_context.shape[2] != self.text_dim:
            raise ValueError(
                "`condition_context` last dim mismatch, "
                f"got {condition_context.shape[2]} vs {self.text_dim}."
            )
        if condition_mask is not None:
            if condition_mask.ndim != 2:
                raise ValueError(
                    "`condition_mask` must be [B, L_c], "
                    f"got shape {tuple(condition_mask.shape)}."
                )
            expected = (batch_size, condition_context.shape[1])
            if tuple(condition_mask.shape) != expected:
                raise ValueError(
                    "`condition_mask` shape must match condition context, "
                    f"got {tuple(condition_mask.shape)} vs {expected}."
                )

    def forward(
        self,
        *,
        current_jepa_tokens: torch.Tensor,
        condition_context: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if current_jepa_tokens.ndim != 3:
            raise ValueError(
                "`current_jepa_tokens` must be [B, N_c, D_v], "
                f"got shape {tuple(current_jepa_tokens.shape)}."
            )
        if current_jepa_tokens.shape[2] != self.vjepa_dim:
            raise ValueError(
                "`current_jepa_tokens` last dim must match `vjepa_dim`, "
                f"got {current_jepa_tokens.shape[2]} vs {self.vjepa_dim}."
            )

        batch_size = int(current_jepa_tokens.shape[0])
        self._validate_condition(
            condition_context=condition_context,
            condition_mask=condition_mask,
            batch_size=batch_size,
        )

        current_tokens = self.predictor_embed(current_jepa_tokens)
        future_tokens = self.future_query_tokens.to(
            device=current_tokens.device,
            dtype=current_tokens.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        condition_tokens = None
        condition_key_padding_mask = None
        if condition_context is not None:
            condition_tokens = self.condition_projection(
                condition_context.to(device=current_tokens.device, dtype=current_tokens.dtype)
            )
            if condition_mask is not None:
                # PyTorch key_padding_mask uses True for ignored positions.
                condition_key_padding_mask = ~condition_mask.to(
                    device=current_tokens.device,
                    dtype=torch.bool,
                )

        for block in self.predictor_blocks:
            future_tokens = block(
                future_tokens,
                current_tokens=current_tokens,
                condition_tokens=condition_tokens,
                condition_key_padding_mask=condition_key_padding_mask,
            )

        future_hidden_tokens = self.predictor_norm(future_tokens)
        pred_future_tokens = self.predictor_proj(future_hidden_tokens)
        return {
            "future_hidden_tokens": future_hidden_tokens,
            "pred_future_tokens": pred_future_tokens,
        }

    def load_vjepa_ac_trunk_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """TODO hook for V-JEPA2-AC trunk reuse.

        The v2 design should map compatible official V-JEPA2-AC predictor trunk
        weights into `predictor_blocks` / `predictor_norm`. This first skeleton
        keeps the hook explicit and avoids importing `external/vjepa2`.
        """
        raise NotImplementedError("V-JEPA2-AC trunk weight mapping is a v2 follow-up task.")
