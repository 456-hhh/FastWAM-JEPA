from __future__ import annotations

import weakref
from typing import Any, Optional

import torch
import torch.nn as nn

from .v5_contract import (
    CAMERA_ORDER,
    SPATIAL_POOL_SIZE,
    TOKENS_PER_CAMERA_GROUP,
    TOKENS_PER_TEMPORAL_GROUP,
    VISUAL_TOKEN_COUNT,
    VJEPA_DIM,
)
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import (
    GateModule,
    RMSNorm,
    flash_attention,
    modulate,
    precompute_freqs_cis,
    rope_apply,
    sinusoidal_embedding_1d,
)


def build_v5_visual_temporal_mask(
    *,
    temporal_groups: int = 3,
    tokens_per_group: int = TOKENS_PER_TEMPORAL_GROUP,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if temporal_groups <= 0 or tokens_per_group <= 0:
        raise ValueError("temporal_groups and tokens_per_group must be positive.")
    time_index = torch.arange(temporal_groups, device=device).repeat_interleave(tokens_per_group)
    mask = time_index[:, None] >= time_index[None, :]
    expected = temporal_groups * tokens_per_group
    if tuple(mask.shape) != (expected, expected):
        raise RuntimeError("V5 visual temporal mask construction failed.")
    return mask


def build_v5_joint_attention_mask(
    *,
    visual_seq_len: int = VISUAL_TOKEN_COUNT,
    action_seq_len: int = 16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if visual_seq_len != VISUAL_TOKEN_COUNT:
        raise ValueError(f"V5 visual sequence length must be {VISUAL_TOKEN_COUNT}.")
    if action_seq_len != 16:
        raise ValueError("V5 action sequence length must be 16.")
    total = visual_seq_len + action_seq_len
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:visual_seq_len, :visual_seq_len] = build_v5_visual_temporal_mask(device=device)
    mask[visual_seq_len:, :visual_seq_len] = True
    mask[visual_seq_len:, visual_seq_len:] = True
    return mask


class SharedVisualContextKV(nn.Module):
    def __init__(self, *, context_dim: int, attention_dim: int, eps: float) -> None:
        super().__init__()
        self.k = nn.Linear(context_dim, attention_dim)
        self.v = nn.Linear(context_dim, attention_dim)
        self.norm_k = RMSNorm(attention_dim, eps=eps)


class VisualCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        eps: float,
        context_kv: SharedVisualContextKV,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attention_hidden_dim = self.num_heads * self.attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        # K/V are owned once by JEPAVisualDiTV5. A weak reference avoids
        # duplicating the same large tensors under every block state_dict key.
        object.__setattr__(self, "_context_kv_ref", weakref.ref(context_kv))

    @property
    def context_kv(self) -> SharedVisualContextKV:
        module = self._context_kv_ref()
        if module is None:
            raise RuntimeError("Shared V5 visual context K/V module is unavailable.")
        return module

    @property
    def k(self) -> nn.Linear:
        return self.context_kv.k

    @property
    def v(self) -> nn.Linear:
        return self.context_kv.v

    @property
    def norm_k(self) -> RMSNorm:
        return self.context_kv.norm_k

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.context_kv.norm_k(self.context_kv.k(context))
        v = self.context_kv.v(context)
        attended = flash_attention(q, k, v, self.num_heads, ctx_mask=ctx_mask)
        return self.o(attended)


class VisualDiTBlockV5(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        context_dim: int,
        ffn_dim: int,
        num_heads: int,
        attn_head_dim: int,
        eps: float,
        context_kv: SharedVisualContextKV,
    ) -> None:
        super().__init__()
        from .wan_video_dit import SelfAttention

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = VisualCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            eps=eps,
            context_kv=context_kv,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        has_sequence_timestep = t_mod.ndim == 4
        if has_sequence_timestep:
            expected_t_mod_shape = (
                int(x.shape[0]),
                int(x.shape[1]),
                6,
                self.hidden_dim,
            )
            chunk_dim = 2
        else:
            expected_t_mod_shape = (int(x.shape[0]), 6, self.hidden_dim)
            chunk_dim = 1
        if tuple(t_mod.shape) != expected_t_mod_shape:
            timestep_kind = "token-wise" if has_sequence_timestep else "scalar"
            raise ValueError(
                f"Visual {timestep_kind} timestep modulation must be "
                f"{expected_t_mod_shape}, got {tuple(t_mod.shape)}."
            )
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(
            6, dim=chunk_dim
        )
        if has_sequence_timestep:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                item.squeeze(2)
                for item in (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            )
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(attn_input, freqs, self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return self.gate(x, gate_mlp, self.ffn(mlp_input))


class JEPAVisualDiTV5(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int = 30,
        hidden_dim: int = 768,
        ffn_dim: int = 3072,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        vjepa_dim: int = VJEPA_DIM,
        text_dim: int = 4096,
        spatial_pool_size: int = SPATIAL_POOL_SIZE,
        freq_dim: int = 256,
        eps: float = 1e-6,
        use_gradient_checkpointing: bool = True,
        infer_shift: float = 5.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or hidden_dim <= 0 or ffn_dim <= 0:
            raise ValueError("Visual DiT layer and hidden dimensions must be positive.")
        if num_heads <= 0 or attn_head_dim <= 0 or attn_head_dim % 2 != 0:
            raise ValueError("Visual DiT requires positive heads and an even head dimension.")
        if spatial_pool_size != SPATIAL_POOL_SIZE:
            raise ValueError(f"V5 spatial_pool_size must be {SPATIAL_POOL_SIZE}.")
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attention_hidden_dim = self.num_heads * self.attn_head_dim
        self.vjepa_dim = int(vjepa_dim)
        self.text_dim = int(text_dim)
        self.spatial_pool_size = int(spatial_pool_size)
        self.freq_dim = int(freq_dim)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.infer_scheduler = WanContinuousFlowMatchScheduler(shift=float(infer_shift))

        self.input_projection = nn.Linear(vjepa_dim, hidden_dim)
        self.camera_embedding = nn.Parameter(torch.zeros(len(CAMERA_ORDER), hidden_dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(3, hidden_dim))
        self.spatial_embedding = nn.Parameter(
            torch.zeros(spatial_pool_size * spatial_pool_size, hidden_dim)
        )
        nn.init.normal_(self.camera_embedding, std=0.02)
        nn.init.normal_(self.temporal_embedding, std=0.02)
        nn.init.normal_(self.spatial_embedding, std=0.02)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        self.context_kv = SharedVisualContextKV(
            context_dim=text_dim,
            attention_dim=num_heads * attn_head_dim,
            eps=eps,
        )
        self.blocks = nn.ModuleList(
            [
                VisualDiTBlockV5(
                    hidden_dim=hidden_dim,
                    context_dim=text_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    attn_head_dim=attn_head_dim,
                    eps=eps,
                    context_kv=self.context_kv,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.output_projection = nn.Linear(hidden_dim, vjepa_dim)
        self.freqs = self._build_visual_rope(attn_head_dim)
        self.register_buffer(
            "visual_attention_mask", build_v5_visual_temporal_mask(), persistent=False
        )

    @staticmethod
    def _build_visual_rope(head_dim: int) -> torch.Tensor:
        spatial_axis = (head_dim // 3) // 2 * 2
        temporal_axis = head_dim - 2 * spatial_axis
        if min(temporal_axis, spatial_axis) <= 0 or temporal_axis % 2:
            raise ValueError(f"Cannot split head_dim={head_dim} into even 3D RoPE axes.")
        temporal_freqs = precompute_freqs_cis(temporal_axis, end=3)
        y_freqs = precompute_freqs_cis(spatial_axis, end=SPATIAL_POOL_SIZE)
        x_freqs = precompute_freqs_cis(spatial_axis, end=SPATIAL_POOL_SIZE)
        coordinates = []
        for temporal_index in range(3):
            for _camera_index in range(len(CAMERA_ORDER)):
                for y in range(SPATIAL_POOL_SIZE):
                    for x in range(SPATIAL_POOL_SIZE):
                        coordinates.append(
                            torch.cat(
                                (temporal_freqs[temporal_index], y_freqs[y], x_freqs[x]), dim=0
                            )
                        )
        freqs = torch.stack(coordinates, dim=0).unsqueeze(1)
        if tuple(freqs.shape) != (VISUAL_TOKEN_COUNT, 1, head_dim // 2):
            raise RuntimeError(
                "V5 visual RoPE shape mismatch: "
                f"expected {(VISUAL_TOKEN_COUNT, 1, head_dim // 2)}, got {tuple(freqs.shape)}."
            )
        return freqs

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _position_embedding(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = []
        for temporal_index in range(3):
            for camera_index in range(len(CAMERA_ORDER)):
                position.append(
                    self.temporal_embedding[temporal_index].unsqueeze(0)
                    + self.camera_embedding[camera_index].unsqueeze(0)
                    + self.spatial_embedding
                )
        return torch.cat(position, dim=0).to(device=device, dtype=dtype)

    def _timestep_modulation(self, timestep: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        if timestep.ndim == 1:
            if int(timestep.shape[0]) not in (1, batch_size):
                raise ValueError("Visual scalar timestep must have length 1 or batch size.")
            if int(timestep.shape[0]) == 1 and batch_size > 1:
                timestep = timestep.expand(batch_size)
            embedded = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            return self.time_projection(embedded).unflatten(1, (6, self.hidden_dim))
        if timestep.ndim != 2 or tuple(timestep.shape) != (batch_size, VISUAL_TOKEN_COUNT):
            raise ValueError(
                "Visual token-wise timestep must be [B,216], "
                f"got {tuple(timestep.shape)}."
            )
        flat = timestep.reshape(-1)
        embedded = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, flat))
        projected = self.time_projection(embedded).reshape(
            batch_size, VISUAL_TOKEN_COUNT, 6, self.hidden_dim
        )
        return projected

    def pre_dit(
        self,
        visual_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if visual_tokens.ndim != 3 or tuple(visual_tokens.shape[1:]) != (
            VISUAL_TOKEN_COUNT,
            self.vjepa_dim,
        ):
            raise ValueError(
                f"Visual tokens must be [B,{VISUAL_TOKEN_COUNT},{self.vjepa_dim}], "
                f"got {tuple(visual_tokens.shape)}."
            )
        batch_size = int(visual_tokens.shape[0])
        if context.ndim != 3 or tuple(context.shape[::2]) != (batch_size, self.text_dim):
            if context.ndim != 3 or int(context.shape[0]) != batch_size or int(context.shape[2]) != self.text_dim:
                raise ValueError(f"Visual context must be [B,L,{self.text_dim}].")
        if context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError("Visual context_mask must match [B,L].")
        context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        if bool((context_mask.sum(dim=1) == 0).any()):
            raise ValueError("Every visual context sample must contain a valid token.")
        tokens = self.input_projection(visual_tokens)
        tokens = tokens + self._position_embedding(device=tokens.device, dtype=tokens.dtype)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, VISUAL_TOKEN_COUNT, -1)
        return {
            "tokens": tokens,
            "freqs": self.freqs.to(device=tokens.device),
            "t_mod": self._timestep_modulation(timestep, batch_size=batch_size),
            "context": context,
            "context_mask": context_attn_mask,
            "attention_mask": self.visual_attention_mask.to(device=tokens.device),
        }

    def run_blocks(self, pre_state: dict[str, Any]) -> torch.Tensor:
        x = pre_state["tokens"]
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                def block_forward(value: torch.Tensor, module: nn.Module = block) -> torch.Tensor:
                    return module(
                        value,
                        pre_state["context"],
                        pre_state["t_mod"],
                        pre_state["freqs"],
                        context_mask=pre_state["context_mask"],
                        self_attn_mask=pre_state["attention_mask"],
                    )

                x = torch.utils.checkpoint.checkpoint(block_forward, x, use_reentrant=False)
            else:
                x = block(
                    x,
                    pre_state["context"],
                    pre_state["t_mod"],
                    pre_state["freqs"],
                    context_mask=pre_state["context_mask"],
                    self_attn_mask=pre_state["attention_mask"],
                )
        return x

    def post_dit(self, tokens: torch.Tensor, *, future_only: bool = True) -> torch.Tensor:
        output = self.output_projection(self.output_norm(tokens))
        return output[:, TOKENS_PER_TEMPORAL_GROUP:] if future_only else output

    def forward_hidden(
        self,
        visual_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.run_blocks(self.pre_dit(visual_tokens, timestep, context, context_mask))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.post_dit(
            self.forward_hidden(visual_tokens, timestep, context, context_mask),
            future_only=True,
        )

    @torch.no_grad()
    def infer_future_jepa(
        self,
        current_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int,
        seed: int,
    ) -> dict[str, Any]:
        if current_tokens.ndim != 3 or tuple(current_tokens.shape[1:]) != (
            TOKENS_PER_TEMPORAL_GROUP,
            self.vjepa_dim,
        ):
            raise ValueError(
                f"current_tokens must be [B,{TOKENS_PER_TEMPORAL_GROUP},{self.vjepa_dim}]."
            )
        if seed <= 0:
            raise ValueError("V5 future inference seed must be > 0.")
        original_current = current_tokens.clone()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        future = torch.randn(
            (
                int(current_tokens.shape[0]),
                2 * TOKENS_PER_TEMPORAL_GROUP,
                self.vjepa_dim,
            ),
            generator=generator,
            dtype=torch.float32,
        ).to(device=current_tokens.device, dtype=current_tokens.dtype)
        timesteps, deltas = self.infer_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=current_tokens.device,
            dtype=current_tokens.dtype,
        )
        for timestep_value, delta in zip(timesteps, deltas):
            batch_size = int(current_tokens.shape[0])
            token_timestep = torch.cat(
                (
                    torch.zeros(
                        (batch_size, TOKENS_PER_TEMPORAL_GROUP),
                        device=current_tokens.device,
                        dtype=current_tokens.dtype,
                    ),
                    timestep_value.expand(batch_size, 2 * TOKENS_PER_TEMPORAL_GROUP),
                ),
                dim=1,
            )
            prediction = self(
                torch.cat((current_tokens, future), dim=1),
                token_timestep,
                context,
                context_mask,
            )
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("V5 detected nonfinite future flow prediction.")
            future = self.infer_scheduler.step(prediction, delta, future)
        if not torch.equal(current_tokens, original_current):
            raise RuntimeError("V5 future inference modified external z0 tokens.")
        z1, z2 = future.split(TOKENS_PER_TEMPORAL_GROUP, dim=1)
        return {
            "future_tokens": future,
            "z1": z1,
            "z2": z2,
            "debug": {
                "num_inference_steps": int(num_inference_steps),
                "seed": int(seed),
                "z0_bitwise_fixed": True,
            },
        }
