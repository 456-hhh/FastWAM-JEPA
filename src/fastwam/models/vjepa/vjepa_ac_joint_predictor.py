from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class VJepaACJointBlock(nn.Module):
    """Joint block with self-attention over main tokens and optional condition cross-attention."""

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
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
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
        main_tokens: torch.Tensor,
        condition_tokens: Optional[torch.Tensor] = None,
        condition_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_attn_input = self.norm_self(main_tokens)
        self_attn_out, _ = self.self_attn(
            self_attn_input,
            self_attn_input,
            self_attn_input,
            need_weights=False,
        )
        main_tokens = main_tokens + self_attn_out

        if condition_tokens is not None:
            cross_attn_input = self.norm_cross(main_tokens)
            cross_attn_out, _ = self.cross_attn(
                cross_attn_input,
                condition_tokens,
                condition_tokens,
                key_padding_mask=condition_key_padding_mask,
                need_weights=False,
            )
            main_tokens = main_tokens + cross_attn_out

        main_tokens = main_tokens + self.mlp(self.norm_mlp(main_tokens))
        return main_tokens


class VJepaACJointPredictor(nn.Module):
    """V-JEPA2-AC-style joint predictor skeleton for FastWAM-JEPA v1.

    This module receives action hidden tokens from ActionDiT.pre_dit(), never
    raw noisy actions.

    Main joint tokens:
        visual_joint_tokens: [B, N_v, D_h]
        future_query_tokens: [B, N_f, D_h]
        action_tokens:       [B, T_a, D_h]

    Optional condition context:
        condition_context: [B, L_c, D_t]
        condition_mask:    [B, L_c]
    """

    def __init__(
        self,
        *,
        vjepa_dim: int,
        hidden_dim: int,
        num_future_tokens: int,
        text_dim: Optional[int] = None,
        num_layers: int = 2,
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
            raise ValueError(
                f"`num_future_tokens` must be positive, got {self.num_future_tokens}."
            )
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
        ffn_dim = int(ffn_dim)
        if ffn_dim <= 0:
            raise ValueError(f"`ffn_dim` must be positive, got {ffn_dim}.")

        self.visual_adapter = nn.Linear(self.vjepa_dim, self.hidden_dim)
        self.future_query_tokens = nn.Parameter(
            torch.zeros(self.num_future_tokens, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.future_query_tokens, std=0.02)

        self.blocks = nn.ModuleList(
            [
                VJepaACJointBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    ffn_dim=ffn_dim,
                    dropout=float(dropout),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.condition_projection = (
            nn.Linear(self.text_dim, self.hidden_dim) if self.text_dim is not None else None
        )
        self.future_feature_projection = nn.Linear(self.hidden_dim, self.vjepa_dim)

    def _validate_condition_inputs(
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
        if condition_context.ndim != 3:
            raise ValueError(
                "`condition_context` must be 3D with shape [B, L_c, D_t], "
                f"got shape {tuple(condition_context.shape)}."
            )
        if condition_context.shape[0] != batch_size:
            raise ValueError(
                "`condition_context` batch dimension must match main tokens, "
                f"got {condition_context.shape[0]} vs {batch_size}."
            )
        if self.text_dim is None:
            raise ValueError(
                "`text_dim` must be set at construction to use `condition_context`."
            )
        if condition_context.shape[2] != self.text_dim:
            raise ValueError(
                "`condition_context` last dim must match `text_dim`, "
                f"got {condition_context.shape[2]} vs {self.text_dim}."
            )
        if condition_mask is not None:
            if condition_mask.ndim != 2:
                raise ValueError(
                    "`condition_mask` must be 2D with shape [B, L_c], "
                    f"got shape {tuple(condition_mask.shape)}."
                )
            expected = (batch_size, condition_context.shape[1])
            if tuple(condition_mask.shape) != expected:
                raise ValueError(
                    "`condition_mask` shape must match condition context [B, L_c], "
                    f"got {tuple(condition_mask.shape)} vs {expected}."
                )

    def _build_condition_tokens(
        self,
        *,
        condition_context: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if condition_context is None:
            return None, None

        context = condition_context.to(device=device, dtype=dtype)
        condition_tokens = self.condition_projection(context)
        if condition_mask is None:
            return condition_tokens, None

        # PyTorch MultiheadAttention key_padding_mask uses True for ignored keys.
        # FastWAM condition masks use True/1 for valid condition tokens.
        key_padding_mask = ~condition_mask.to(device=device, dtype=torch.bool)
        return condition_tokens, key_padding_mask

    def forward(
        self,
        *,
        current_visual_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        condition_context: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run shape-compatible joint prediction.

        Args:
            current_visual_tokens: V-JEPA tokens [B, N_v, D_v].
            action_tokens: ActionDiT hidden tokens [B, T_a, D_h].
            condition_context: Optional text/proprio context [B, L_c, D_t].
            condition_mask: Optional valid-token mask [B, L_c].

        Returns:
            Dict containing:
                updated_action_tokens: [B, T_a, D_h]
                future_hidden_tokens:  [B, N_f, D_h]
                pred_future_tokens:    [B, N_f, D_v]
        """
        if current_visual_tokens.ndim != 3:
            raise ValueError(
                "`current_visual_tokens` must be 3D with shape [B, N_v, D_v], "
                f"got shape {tuple(current_visual_tokens.shape)}."
            )
        if current_visual_tokens.shape[2] != self.vjepa_dim:
            raise ValueError(
                "`current_visual_tokens` last dim must match `vjepa_dim`, "
                f"got {current_visual_tokens.shape[2]} vs {self.vjepa_dim}."
            )
        if action_tokens.ndim != 3:
            raise ValueError(
                "`action_tokens` must be 3D with shape [B, T_a, D_h]. "
                "Pass ActionDiT.pre_dit(...)[\"tokens\"], not raw noisy_action. "
                f"Got shape {tuple(action_tokens.shape)}."
            )
        if action_tokens.shape[2] != self.hidden_dim:
            raise ValueError(
                "`action_tokens` last dim must match `hidden_dim`. "
                "This module expects ActionDiT hidden tokens, not raw noisy_action. "
                f"Got {action_tokens.shape[2]} vs {self.hidden_dim}."
            )

        batch_size = int(current_visual_tokens.shape[0])
        if action_tokens.shape[0] != batch_size:
            raise ValueError(
                "`action_tokens` batch dimension must match `current_visual_tokens`, "
                f"got {action_tokens.shape[0]} vs {batch_size}."
            )

        self._validate_condition_inputs(
            condition_context=condition_context,
            condition_mask=condition_mask,
            batch_size=batch_size,
        )

        visual_joint_tokens = self.visual_adapter(current_visual_tokens)
        future_query_tokens = self.future_query_tokens.to(
            device=action_tokens.device,
            dtype=action_tokens.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        # Main tokens are visual + future query + action only. Text/proprio are
        # condition inputs and are not concatenated into this sequence.
        main_tokens = torch.cat(
            [visual_joint_tokens, future_query_tokens, action_tokens],
            dim=1,
        )

        condition_tokens, condition_key_padding_mask = self._build_condition_tokens(
            condition_context=condition_context,
            condition_mask=condition_mask,
            device=main_tokens.device,
            dtype=main_tokens.dtype,
        )
        encoded_tokens = main_tokens
        for block in self.blocks:
            encoded_tokens = block(
                encoded_tokens,
                condition_tokens=condition_tokens,
                condition_key_padding_mask=condition_key_padding_mask,
            )

        num_visual_tokens = int(current_visual_tokens.shape[1])
        num_action_tokens = int(action_tokens.shape[1])
        future_start = num_visual_tokens
        future_end = future_start + self.num_future_tokens
        action_start = future_end
        action_end = action_start + num_action_tokens

        future_hidden_tokens = encoded_tokens[:, future_start:future_end, :]
        updated_action_tokens = encoded_tokens[:, action_start:action_end, :]
        pred_future_tokens = self.future_feature_projection(future_hidden_tokens)

        return {
            "updated_action_tokens": updated_action_tokens,
            "future_hidden_tokens": future_hidden_tokens,
            "pred_future_tokens": pred_future_tokens,
        }
