from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rotary3DSelfAttention(nn.Module):
    """Full-sequence self-attention with simplified 3D RoPE.

    Shape contract:
        tokens: [B, N, D_h]
        coords: [N, 3] with columns [t, y, x]

    Rotary pairs inside each head are assigned to t/y/x axes in a round-robin
    pattern. This keeps the implementation local and checkpoint-free while
    preserving the important V-JEPA2-AC-style idea: latent tokens interact through
    full self-attention with spatial/temporal position encoded in q/k.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.rope_base = float(rope_base)

        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {self.hidden_dim}.")
        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be positive, got {self.num_heads}.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "`hidden_dim` must be divisible by `num_heads`, "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )
        self.head_dim = self.hidden_dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                "RoPE requires an even per-head dim, "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}."
            )

        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _rope_angles(self, coords: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = coords.to(device=device, dtype=torch.float32)
        num_pairs = self.head_dim // 2
        pair_index = torch.arange(num_pairs, device=device, dtype=torch.float32)
        axis_index = torch.arange(num_pairs, device=device) % 3
        selected_pos = coords[:, axis_index]
        inv_freq = self.rope_base ** (-pair_index / max(float(num_pairs), 1.0))
        return (selected_pos * inv_freq.unsqueeze(0)).to(dtype=dtype)

    @staticmethod
    def _apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, D_head], angles: [N, D_head / 2]
        x_pair = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        cos = angles.float().cos().unsqueeze(0).unsqueeze(0)
        sin = angles.float().sin().unsqueeze(0).unsqueeze(0)
        x0 = x_pair[..., 0]
        x1 = x_pair[..., 1]
        rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
        return rotated.flatten(-2).to(dtype=x.dtype)

    def forward(self, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must be [B, N, D_h], got {tuple(tokens.shape)}.")
        batch_size, seq_len, hidden_dim = tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"`tokens` last dim must be {self.hidden_dim}, got {hidden_dim}.")
        if coords.ndim != 2 or tuple(coords.shape) != (seq_len, 3):
            raise ValueError(f"`coords` must be [N, 3], got {tuple(coords.shape)} for N={seq_len}.")

        qkv = self.qkv(tokens)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        angles = self._rope_angles(coords, device=tokens.device, dtype=tokens.dtype)
        q = self._apply_rope(q, angles)
        k = self._apply_rope(k, angles)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        return self.out(attn)


class JepaFuturePredictorBlock(nn.Module):
    """RoPE full-transformer block for JEPA latent dynamics.

    Shape contract:
        latent tokens:    [B, N, D_h]
        condition tokens: [B, L_c, D_h], optional text/proprio context

    The latent path is full sequence self-attention. Text/proprio remain optional
    conditioning via cross-attention and are not action/noisy-action tokens.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = Rotary3DSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            rope_base=rope_base,
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
        tokens: torch.Tensor,
        *,
        coords: torch.Tensor,
        condition_tokens: Optional[torch.Tensor] = None,
        condition_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = tokens + self.self_attn(self.norm_self(tokens), coords)

        if condition_tokens is not None:
            condition_attn_input = self.norm_condition(tokens)
            condition_attn_out, _ = self.condition_cross_attn(
                condition_attn_input,
                condition_tokens,
                condition_tokens,
                key_padding_mask=condition_key_padding_mask,
                need_weights=False,
            )
            tokens = tokens + condition_attn_out

        return tokens + self.mlp(self.norm_mlp(tokens))


class JepaFuturePredictor(nn.Module):
    """Predict future V-JEPA tokens from current V-JEPA tokens.

    This is the v2 RoPE full-transformer variant. It removes learnable future
    query tokens: current_jepa_tokens are the full latent sequence, and the model
    directly updates that sequence into pred_future_tokens.

    Inputs:
        current_jepa_tokens: [B, N, D_v], v2 default N=256, D_v=1408
        condition_context:   [B, L_c, D_t], optional text/proprio context
        condition_mask:      [B, L_c], optional, True means valid token

    Outputs:
        future_hidden_tokens: [B, N, D_h]
        pred_future_tokens:   [B, N, D_v]
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
        rope_base: float = 10000.0,
        use_temporal_embedding: bool = True,
        max_temporal_positions: int = 16,
    ) -> None:
        super().__init__()
        self.vjepa_dim = int(vjepa_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_future_tokens = int(num_future_tokens)
        self.text_dim = None if text_dim is None else int(text_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.use_temporal_embedding = bool(use_temporal_embedding)
        self.max_temporal_positions = int(max_temporal_positions)
        self.init_source = "random_init"

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
        if (self.hidden_dim // self.num_heads) % 2 != 0:
            raise ValueError(
                "RoPE requires an even per-head dim, "
                f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )
        if self.max_temporal_positions <= 0:
            raise ValueError(
                "`max_temporal_positions` must be positive, "
                f"got {self.max_temporal_positions}."
            )
        if ffn_dim is None:
            ffn_dim = self.hidden_dim * 4

        self.predictor_embed = nn.Linear(self.vjepa_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.temporal_embedding = (
            nn.Embedding(self.max_temporal_positions, self.hidden_dim)
            if self.use_temporal_embedding
            else None
        )
        self.condition_projection = (
            nn.Linear(self.text_dim, self.hidden_dim) if self.text_dim is not None else None
        )
        self.condition_early_fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.predictor_blocks = nn.ModuleList(
            [
                JepaFuturePredictorBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    ffn_dim=int(ffn_dim),
                    dropout=float(dropout),
                    rope_base=float(rope_base),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.predictor_norm = nn.LayerNorm(self.hidden_dim)
        self.predictor_proj = nn.Linear(self.hidden_dim, self.vjepa_dim)

    @staticmethod
    def _infer_3d_coords(num_tokens: int, *, device: torch.device) -> torch.Tensor:
        if num_tokens <= 0:
            raise ValueError(f"`num_tokens` must be positive, got {num_tokens}.")
        if num_tokens % 256 == 0:
            temporal = num_tokens // 256
            height = width = 16
        else:
            temporal = 1
            height = int(math.sqrt(num_tokens))
            while height > 1 and num_tokens % height != 0:
                height -= 1
            width = int(math.ceil(num_tokens / height))
        coords = torch.arange(num_tokens, device=device)
        spatial = max(height * width, 1)
        t = torch.div(coords, spatial, rounding_mode="floor").clamp(max=max(temporal - 1, 0))
        rem = coords % spatial
        y = torch.div(rem, width, rounding_mode="floor")
        x = rem % width
        return torch.stack((t, y, x), dim=1)

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

    @staticmethod
    def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cleaned: dict[str, torch.Tensor] = {}
        prefixes = (
            "module.predictor.",
            "module.ac_predictor.",
            "module.vjepa2_ac_predictor.",
            "predictor.",
            "ac_predictor.",
            "vjepa2_ac_predictor.",
            "module.",
        )
        for key, value in state_dict.items():
            new_key = key
            if isinstance(new_key, str):
                changed = True
                while changed:
                    changed = False
                    for prefix in prefixes:
                        if new_key.startswith(prefix):
                            new_key = new_key[len(prefix) :]
                            changed = True
            cleaned[new_key] = value
        return cleaned

    @classmethod
    def _extract_vjepa2ac_predictor_state_dict(cls, payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            for key in ("predictor", "ac_predictor", "vjepa2_ac_predictor", "model", "model_state_dict", "state_dict"):
                value = payload.get(key)
                if isinstance(value, dict):
                    try:
                        return cls._extract_vjepa2ac_predictor_state_dict(value)
                    except ValueError:
                        pass
            tensor_items = {key: value for key, value in payload.items() if isinstance(key, str) and torch.is_tensor(value)}
            if tensor_items:
                return cls._strip_module_prefix(tensor_items)
        raise ValueError("Could not find a tensor state_dict in V-JEPA2-AC checkpoint payload.")

    @staticmethod
    def _is_disallowed_vjepa2ac_key(key: str) -> bool:
        disallowed = (
            "action_encoder.",
            "state_encoder.",
            "extrinsics_encoder.",
            "policy_head.",
            "action_head.",
            "proprio_head.",
            "robot_head.",
            "head.",
            "decoder.",
            "attn_mask",
        )
        return any(key.startswith(prefix) or prefix in key for prefix in disallowed)

    @staticmethod
    def _map_vjepa2ac_key(key: str) -> str | None:
        # Official V-JEPA2-AC predictor trunk keys use ACBlock names:
        #   predictor_blocks.N.norm1 / attn.qkv / attn.proj / norm2 / mlp.fc*
        # This simplified predictor uses:
        #   predictor_blocks.N.norm_self / self_attn.qkv / self_attn.out / norm_mlp / mlp.0|3
        replacements = (
            (".norm1.", ".norm_self."),
            (".attn.qkv.", ".self_attn.qkv."),
            (".attn.proj.", ".self_attn.out."),
            (".norm2.", ".norm_mlp."),
            (".mlp.fc1.", ".mlp.0."),
            (".mlp.fc2.", ".mlp.3."),
        )
        mapped = key
        for src, dst in replacements:
            mapped = mapped.replace(src, dst)
        if ".mlp.fc3." in mapped:
            # Official SwiGLU checkpoints use fc1/fc2/fc3. This predictor uses
            # a standard GELU MLP, so fc3 cannot be mapped safely.
            return None
        if mapped.startswith("norm."):
            return "predictor_norm." + mapped[len("norm.") :]
        if mapped.startswith("proj."):
            return "predictor_proj." + mapped[len("proj.") :]
        return mapped

    def _load_partial_vjepa2ac_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        own_state = self.state_dict()
        mapped_state: dict[str, torch.Tensor] = {}
        loaded_keys: list[str] = []
        skipped_keys: list[str] = []
        loaded_params_count = 0
        skipped_params_count = 0
        shape_mismatch_keys: list[dict[str, Any]] = []

        for raw_key, value in self._strip_module_prefix(state_dict).items():
            if not torch.is_tensor(value):
                skipped_keys.append(raw_key)
                skipped_params_count += int(value.numel()) if torch.is_tensor(value) else 0
                continue
            if self._is_disallowed_vjepa2ac_key(raw_key):
                skipped_keys.append(raw_key)
                skipped_params_count += int(value.numel()) if torch.is_tensor(value) else 0
                continue
            mapped_key = self._map_vjepa2ac_key(raw_key)
            if mapped_key is None or mapped_key not in own_state:
                skipped_keys.append(raw_key)
                skipped_params_count += int(value.numel()) if torch.is_tensor(value) else 0
                continue
            if tuple(own_state[mapped_key].shape) != tuple(value.shape):
                shape_mismatch_keys.append(
                    {
                        "source_key": raw_key,
                        "target_key": mapped_key,
                        "source_shape": tuple(value.shape),
                        "target_shape": tuple(own_state[mapped_key].shape),
                    }
                )
                skipped_keys.append(raw_key)
                skipped_params_count += int(value.numel()) if torch.is_tensor(value) else 0
                continue
            mapped_state[mapped_key] = value.to(dtype=own_state[mapped_key].dtype)
            loaded_keys.append(raw_key)
            loaded_params_count += int(value.numel())

        missing, unexpected = self.load_state_dict(mapped_state, strict=False)
        self.init_source = "vjepa2ac_pretrained" if loaded_keys else "random_init_no_vjepa2ac_match"
        stats = {
            "init_source": self.init_source,
            "loaded_keys_count": len(loaded_keys),
            "skipped_keys_count": len(skipped_keys),
            "loaded_params_count": loaded_params_count,
            "skipped_params_count": skipped_params_count,
            "missing_keys_count": len(missing),
            "unexpected_keys_count": len(unexpected),
            "shape_mismatch_count": len(shape_mismatch_keys),
            "loaded_keys": loaded_keys,
            "skipped_keys": skipped_keys,
            "shape_mismatch_keys": shape_mismatch_keys,
        }
        print(
            " ".join(
                [
                    "V-JEPA2-AC predictor weight load",
                    f"init_source={stats['init_source']}",
                    f"loaded_keys_count={stats['loaded_keys_count']}",
                    f"skipped_keys_count={stats['skipped_keys_count']}",
                    f"shape_mismatch_count={stats['shape_mismatch_count']}",
                    f"loaded_params_count={stats['loaded_params_count']}",
                    f"skipped_params_count={stats['skipped_params_count']}",
                ]
            ),
            flush=True,
        )
        return stats

    def load_vjepa2ac_predictor_weights(self, checkpoint_path: str | Path | None) -> dict[str, Any]:
        """Partially initialize this predictor from a V-JEPA2-AC predictor checkpoint.

        The loader is intentionally conservative: it never imports external/vjepa2,
        skips action/state/extrinsics/policy heads, maps only compatible trunk
        names, and loads only tensors whose shapes exactly match this simplified
        RoPE full-transformer predictor. Missing checkpoints fall back to random
        initialization with a warning so Stage-B predictor training can continue.
        """
        if checkpoint_path is None:
            warnings.warn(
                "No V-JEPA2-AC predictor checkpoint was provided; using random predictor init.",
                RuntimeWarning,
            )
            self.init_source = "random_init_missing_vjepa2ac_checkpoint"
            stats = {
                "init_source": self.init_source,
                "loaded_keys_count": 0,
                "skipped_keys_count": 0,
                "missing_checkpoint": True,
            }
            print(
                "V-JEPA2-AC predictor weight load "
                f"init_source={stats['init_source']} loaded_keys_count=0 skipped_keys_count=0",
                flush=True,
            )
            return stats

        path = Path(checkpoint_path)
        if not path.exists():
            warnings.warn(
                f"V-JEPA2-AC predictor checkpoint does not exist: {path}. Using random predictor init.",
                RuntimeWarning,
            )
            self.init_source = "random_init_missing_vjepa2ac_checkpoint"
            stats = {
                "init_source": self.init_source,
                "loaded_keys_count": 0,
                "skipped_keys_count": 0,
                "missing_checkpoint": True,
                "checkpoint_path": str(path),
            }
            print(
                "V-JEPA2-AC predictor weight load "
                f"init_source={stats['init_source']} loaded_keys_count=0 skipped_keys_count=0",
                flush=True,
            )
            return stats

        payload = torch.load(path, map_location="cpu")
        state_dict = self._extract_vjepa2ac_predictor_state_dict(payload)
        stats = self._load_partial_vjepa2ac_state_dict(state_dict)
        stats["checkpoint_path"] = str(path)
        return stats

    def load_vjepa_ac_trunk_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Load compatible V-JEPA2-AC predictor trunk tensors from a state_dict."""
        return self._load_partial_vjepa2ac_state_dict(state_dict)

    def forward(
        self,
        *,
        current_jepa_tokens: torch.Tensor,
        condition_context: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if current_jepa_tokens.ndim != 3:
            raise ValueError(
                "`current_jepa_tokens` must be [B, N, D_v], "
                f"got shape {tuple(current_jepa_tokens.shape)}."
            )
        if current_jepa_tokens.shape[2] != self.vjepa_dim:
            raise ValueError(
                "`current_jepa_tokens` last dim must match `vjepa_dim`, "
                f"got {current_jepa_tokens.shape[2]} vs {self.vjepa_dim}."
            )
        if current_jepa_tokens.shape[1] != self.num_future_tokens:
            raise ValueError(
                "RoPE full-transformer predictor expects input token count to match output token count, "
                f"got N={current_jepa_tokens.shape[1]} vs num_future_tokens={self.num_future_tokens}."
            )

        batch_size = int(current_jepa_tokens.shape[0])
        self._validate_condition(
            condition_context=condition_context,
            condition_mask=condition_mask,
            batch_size=batch_size,
        )

        tokens = self.input_norm(self.predictor_embed(current_jepa_tokens))
        coords = self._infer_3d_coords(tokens.shape[1], device=tokens.device)
        if self.temporal_embedding is not None:
            temporal_index = coords[:, 0].clamp(max=self.max_temporal_positions - 1).long()
            tokens = tokens + self.temporal_embedding(temporal_index).to(
                device=tokens.device,
                dtype=tokens.dtype,
            ).unsqueeze(0)

        condition_tokens = None
        condition_key_padding_mask = None
        if condition_context is not None:
            condition_tokens = self.condition_projection(
                condition_context.to(device=tokens.device, dtype=tokens.dtype)
            )
            if condition_mask is not None:
                condition_mask_bool = condition_mask.to(
                    device=tokens.device,
                    dtype=torch.bool,
                )
                # PyTorch key_padding_mask uses True for ignored positions.
                condition_key_padding_mask = ~condition_mask_bool
                valid = condition_mask_bool.to(dtype=condition_tokens.dtype).unsqueeze(-1)
                pooled_condition = (condition_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            else:
                pooled_condition = condition_tokens.mean(dim=1)
            # Early fusion: a pooled text/proprio summary biases every latent token.
            # Late fusion still happens inside each block through cross-attention.
            tokens = tokens + self.condition_early_fusion(pooled_condition).unsqueeze(1)

        for block in self.predictor_blocks:
            tokens = block(
                tokens,
                coords=coords,
                condition_tokens=condition_tokens,
                condition_key_padding_mask=condition_key_padding_mask,
            )

        future_hidden_tokens = self.predictor_norm(tokens)
        pred_future_tokens = self.predictor_proj(future_hidden_tokens)
        return {
            "future_hidden_tokens": future_hidden_tokens,
            "pred_future_tokens": pred_future_tokens,
        }
