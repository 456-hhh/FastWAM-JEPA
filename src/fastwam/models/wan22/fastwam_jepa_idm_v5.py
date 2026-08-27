from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.models.vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper

from .action_dit import ActionDiT
from .jepa_visual_dit_v5 import (
    JEPAVisualDiTV5,
    build_v5_joint_attention_mask,
)
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .v5_contract import (
    ACTION_HORIZON,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CURRENT_TOKEN_COUNT_PER_CAMERA,
    FUTURE_TOKEN_COUNT_PER_CAMERA,
    TOKENS_PER_TEMPORAL_GROUP,
    VISUAL_TOKEN_COUNT,
    VJEPA_DIM,
    build_vjepa_clips,
    pool_dual_camera_vjepa_tokens,
)


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"V5 detected nonfinite values in {name}.")


def _tokenwise_visual_timestep(
    future_timestep: torch.Tensor, *, clean_future: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if future_timestep.ndim != 1:
        raise ValueError("future_timestep must be [B].")
    batch_size = int(future_timestep.shape[0])
    future_t = future_timestep
    if clean_future is not None:
        if clean_future.ndim != 1 or int(clean_future.shape[0]) != batch_size:
            raise ValueError("clean_future must be a [B] boolean tensor.")
        future_t = torch.where(clean_future, torch.zeros_like(future_t), future_t)
    return torch.cat(
        (
            torch.zeros(
                (batch_size, TOKENS_PER_TEMPORAL_GROUP),
                device=future_timestep.device,
                dtype=future_timestep.dtype,
            ),
            future_t[:, None].expand(-1, 2 * TOKENS_PER_TEMPORAL_GROUP),
        ),
        dim=1,
    )


def compute_visual_flow_loss(
    *,
    visual_dit: JEPAVisualDiTV5,
    scheduler: WanContinuousFlowMatchScheduler,
    z0: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    visual_module = visual_dit.module if hasattr(visual_dit, "module") else visual_dit
    future_gt = torch.cat((z1, z2), dim=1)
    expected_current = (z0.shape[0], TOKENS_PER_TEMPORAL_GROUP, visual_module.vjepa_dim)
    expected_future = (z0.shape[0], 2 * TOKENS_PER_TEMPORAL_GROUP, visual_module.vjepa_dim)
    if tuple(z0.shape) != expected_current or tuple(future_gt.shape) != expected_future:
        raise ValueError(
            f"V5 visual latent contract mismatch: z0={tuple(z0.shape)}, future={tuple(future_gt.shape)}."
        )
    noise = torch.randn_like(future_gt)
    timestep = scheduler.sample_training_t(
        batch_size=int(z0.shape[0]), device=z0.device, dtype=z0.dtype
    )
    future_noisy = scheduler.add_noise(future_gt, noise, timestep)
    visual_input = torch.cat((z0, future_noisy), dim=1)
    token_timestep = _tokenwise_visual_timestep(timestep)
    prediction = visual_dit(visual_input, token_timestep, context, context_mask)
    target = scheduler.training_target(future_gt, noise, timestep)
    _require_finite("visual_prediction", prediction)
    loss = F.mse_loss(prediction.float(), target.float())
    _require_finite("loss_visual", loss)
    return loss, {
        "loss_visual": loss.detach(),
        "visual_timestep_mean": timestep.detach().float().mean(),
    }


class FastWAMJEPAIDMV5(nn.Module):
    def __init__(
        self,
        *,
        vjepa_encoder: VJepaEncoderWrapper,
        visual_dit: JEPAVisualDiTV5,
        action_expert: ActionDiT,
        proprio_encoder: nn.Linear,
        visual_train_shift: float = 5.0,
        visual_infer_shift: float = 5.0,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        video_cond_noise_prob: float = 0.5,
        allow_dummy_vjepa: bool = False,
    ) -> None:
        super().__init__()
        if vjepa_encoder.dummy and not allow_dummy_vjepa:
            raise ValueError("Production FastWAMJEPAIDMV5 requires a real V-JEPA encoder.")
        if not vjepa_encoder.freeze:
            raise ValueError("V5 V-JEPA must be frozen.")
        if int(action_expert.action_dim) != 7:
            raise ValueError("V5 ActionDiT must use action_dim=7.")
        if len(action_expert.blocks) != len(visual_dit.blocks):
            raise ValueError("V5 Visual DiT and ActionDiT must have the same layer count.")
        if int(action_expert.num_heads) != int(visual_dit.num_heads):
            raise ValueError("V5 Visual DiT and ActionDiT must have the same num_heads.")
        if int(action_expert.attn_head_dim) != int(visual_dit.attn_head_dim):
            raise ValueError("V5 Visual DiT and ActionDiT must have the same attn_head_dim.")
        if not isinstance(proprio_encoder, nn.Linear):
            raise ValueError("V5 proprio_encoder must be nn.Linear(8,4096).")
        if proprio_encoder.in_features != 8 or proprio_encoder.out_features != 4096:
            raise ValueError("V5 proprio_encoder must be Linear(8,4096).")
        if not 0.0 <= float(video_cond_noise_prob) <= 1.0:
            raise ValueError("video_cond_noise_prob must be in [0,1].")

        self.vjepa_encoder = vjepa_encoder
        self.visual_dit = visual_dit
        self.action_expert = action_expert
        self.proprio_encoder = proprio_encoder
        self.video_cond_noise_prob = float(video_cond_noise_prob)
        self.mot = MoT(
            mixtures={"video": visual_dit, "action": action_expert},
            mot_checkpoint_mixed_attn=True,
        )
        self.visual_train_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps, shift=visual_train_shift
        )
        self.visual_infer_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps, shift=visual_infer_shift
        )
        self.visual_dit.infer_scheduler = self.visual_infer_scheduler
        self.action_train_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps, shift=action_train_shift
        )
        self.action_infer_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps, shift=action_infer_shift
        )
        self.vjepa_encoder.requires_grad_(False)
        self.vjepa_encoder.eval()

    def train(self, mode: bool = True) -> "FastWAMJEPAIDMV5":
        super().train(mode)
        self.vjepa_encoder.eval()
        return self

    def runtime_device(self) -> torch.device:
        return next(self.visual_dit.parameters()).device

    def runtime_dtype(self) -> torch.dtype:
        return next(self.visual_dit.parameters()).dtype

    def set_stage2_trainability(self) -> None:
        self.requires_grad_(False)
        self.visual_dit.requires_grad_(True)
        self.vjepa_encoder.requires_grad_(False)

    def set_stage3_trainability(self) -> None:
        self.requires_grad_(False)
        self.visual_dit.requires_grad_(True)
        self.action_expert.requires_grad_(True)
        self.proprio_encoder.requires_grad_(True)
        self.vjepa_encoder.requires_grad_(False)
        trainable = sum(p.numel() for p in self.action_expert.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.action_expert.parameters())
        if trainable != total:
            raise RuntimeError(f"Stage3 must unfreeze all ActionDiT parameters: {trainable} != {total}.")

    def _encode_clip(self, clip: torch.Tensor, *, expected_tokens: int) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.vjepa_encoder(clip)
        if tuple(tokens.shape[1:]) != (expected_tokens, VJEPA_DIM):
            raise ValueError(
                "Strict V5 V-JEPA output mismatch: "
                f"expected [B,{expected_tokens},{VJEPA_DIM}], got {tuple(tokens.shape)}."
            )
        return tokens

    def encode_current(
        self, agentview: torch.Tensor, wrist: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if wrist is None:
            clips = build_vjepa_clips(agentview)
            agent_clip = clips["agentview_current"]
            wrist_clip = clips["wrist_current"]
        else:
            for name, frame in (("agentview", agentview), ("wrist", wrist)):
                if frame.ndim != 4 or tuple(frame.shape[1:]) != (
                    3,
                    CAMERA_HEIGHT,
                    CAMERA_WIDTH,
                ):
                    raise ValueError(f"{name} current RGB must be [B,3,224,224].")
            if int(agentview.shape[0]) != int(wrist.shape[0]):
                raise ValueError("Current camera batch sizes must match.")
            agent_clip = agentview.unsqueeze(2).repeat(1, 1, 2, 1, 1)
            wrist_clip = wrist.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        agent_tokens = self._encode_clip(agent_clip, expected_tokens=CURRENT_TOKEN_COUNT_PER_CAMERA)
        wrist_tokens = self._encode_clip(wrist_clip, expected_tokens=CURRENT_TOKEN_COUNT_PER_CAMERA)
        pooled = pool_dual_camera_vjepa_tokens(
            agent_tokens, wrist_tokens, temporal_groups=1, vjepa_dim=VJEPA_DIM
        )
        z0 = pooled[:, 0]
        if tuple(z0.shape[1:]) != (TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM):
            raise RuntimeError("V5 z0 pooling contract failed.")
        return z0

    def encode_future_gt(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clips = build_vjepa_clips(video)
        agent_tokens = self._encode_clip(
            clips["agentview_future"], expected_tokens=FUTURE_TOKEN_COUNT_PER_CAMERA
        )
        wrist_tokens = self._encode_clip(
            clips["wrist_future"], expected_tokens=FUTURE_TOKEN_COUNT_PER_CAMERA
        )
        pooled = pool_dual_camera_vjepa_tokens(
            agent_tokens, wrist_tokens, temporal_groups=2, vjepa_dim=VJEPA_DIM
        )
        z1, z2 = pooled[:, 0], pooled[:, 1]
        return z1, z2

    def build_base_context(
        self, context: torch.Tensor, context_mask: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3 or int(context.shape[-1]) != 4096:
            raise ValueError("V5 context must be [B,L,4096].")
        if context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError("V5 requires the real [B,L] context mask.")
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        if proprio.ndim != 2 or tuple(proprio.shape) != (int(context.shape[0]), 8):
            raise ValueError("V5 current proprio must be [B,8].")
        real_mask = context_mask.to(device=context.device, dtype=torch.bool)
        if bool((real_mask.sum(dim=1) == 0).any()):
            raise ValueError("Every V5 context sample must contain a valid token.")
        context = context.masked_fill(~real_mask.unsqueeze(-1), 0.0)
        proprio_token = self.proprio_encoder(proprio.to(dtype=context.dtype)).unsqueeze(1)
        proprio_mask = torch.ones(
            (context.shape[0], 1), dtype=torch.bool, device=context.device
        )
        return torch.cat((context, proprio_token), dim=1), torch.cat(
            (real_mask, proprio_mask), dim=1
        )

    def visual_training_loss(
        self,
        *,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return compute_visual_flow_loss(
            visual_dit=self.visual_dit,
            scheduler=self.visual_train_scheduler,
            z0=z0,
            z1=z1,
            z2=z2,
            context=context,
            context_mask=context_mask,
        )

    def _teacher_forced_visual_condition(
        self, z0: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        future_gt = torch.cat((z1, z2), dim=1)
        batch_size = int(z0.shape[0])
        timestep = self.visual_train_scheduler.sample_training_t(
            batch_size=batch_size, device=z0.device, dtype=z0.dtype
        )
        noise = torch.randn_like(future_gt)
        noisy_future = self.visual_train_scheduler.add_noise(future_gt, noise, timestep)
        noisy_sample = torch.rand((batch_size,), device=z0.device) < self.video_cond_noise_prob
        conditioned_future = torch.where(noisy_sample[:, None, None], noisy_future, future_gt)
        clean_future = ~noisy_sample
        visual_timestep = _tokenwise_visual_timestep(timestep, clean_future=clean_future)
        return torch.cat((z0, conditioned_future), dim=1), visual_timestep, noisy_sample

    def action_training_loss_teacher_forcing(
        self,
        *,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if action.ndim != 3 or tuple(action.shape[1:]) != (ACTION_HORIZON, 7):
            raise ValueError(f"V5 action target must be [B,16,7], got {tuple(action.shape)}.")
        visual_condition, visual_timestep, noisy_condition = self._teacher_forced_visual_condition(
            z0, z1, z2
        )
        action_timestep = self.action_train_scheduler.sample_training_t(
            batch_size=int(action.shape[0]), device=action.device, dtype=action.dtype
        )
        action_noise = torch.randn_like(action)
        noisy_action = self.action_train_scheduler.add_noise(action, action_noise, action_timestep)
        target = self.action_train_scheduler.training_target(action, action_noise, action_timestep)

        visual_pre = self.visual_dit.pre_dit(
            visual_condition, visual_timestep, context, context_mask
        )
        action_pre = self.action_expert.pre_dit(
            noisy_action, action_timestep, context, context_mask
        )
        joint_mask = build_v5_joint_attention_mask(device=action.device)
        outputs = self.mot(
            embeds_all={"video": visual_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=joint_mask,
            freqs_all={"video": visual_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {
                    "context": visual_pre["context"],
                    "mask": visual_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={"video": visual_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        prediction = self.action_expert.post_dit(outputs["action"], action_pre)
        _require_finite("action_prediction", prediction)
        per_token = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=-1)
        if action_is_pad is not None:
            if tuple(action_is_pad.shape) != tuple(per_token.shape):
                raise ValueError("action_is_pad must be [B,16].")
            valid = ~action_is_pad.to(device=action.device, dtype=torch.bool)
            valid_count = valid.sum(dim=1)
            if bool((valid_count == 0).any()):
                raise ValueError("V5 encountered an all-padding action sample.")
            per_sample = (per_token * valid).sum(dim=1) / valid_count
        else:
            per_sample = per_token.mean(dim=1)
        weight = self.action_train_scheduler.training_weight(action_timestep).to(
            device=action.device, dtype=per_sample.dtype
        )
        loss = (per_sample * weight).mean()
        _require_finite("loss_action", loss)
        return loss, {
            "loss_action": loss.detach(),
            "video_condition_noisy_fraction": noisy_condition.float().mean().detach(),
            "action_timestep_mean": action_timestep.float().mean().detach(),
        }

    def training_loss(self, sample: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        required = ("video", "action", "context", "context_mask", "proprio")
        missing = [name for name in required if name not in sample]
        if missing:
            raise ValueError(f"V5 training sample is missing {missing}.")
        z0 = self.encode_current(sample["video"])
        z1, z2 = self.encode_future_gt(sample["video"])
        context, context_mask = self.build_base_context(
            sample["context"], sample["context_mask"], sample["proprio"]
        )
        loss_visual, visual_metrics = self.visual_training_loss(
            z0=z0, z1=z1, z2=z2, context=context, context_mask=context_mask
        )
        loss_action, action_metrics = self.action_training_loss_teacher_forcing(
            z0=z0,
            z1=z1,
            z2=z2,
            action=sample["action"],
            context=context,
            context_mask=context_mask,
            action_is_pad=sample.get("action_is_pad"),
        )
        total = loss_visual + loss_action
        _require_finite("loss_total", total)
        metrics = {**visual_metrics, **action_metrics, "loss_total": total.detach()}
        return total, {key: float(value.float().item()) for key, value in metrics.items()}

    def forward(self, sample: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        return self.training_loss(sample)

    @staticmethod
    def _seeded_noise(
        shape: tuple[int, ...], *, seed: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if seed <= 0:
            raise ValueError("V5 inference seed must be > 0.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=dtype
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
        if tuple(current_tokens.shape[1:]) != (TOKENS_PER_TEMPORAL_GROUP, VJEPA_DIM):
            raise ValueError("current_tokens must be [B,72,1408].")
        return self.visual_dit.infer_future_jepa(
            current_tokens,
            context,
            context_mask,
            num_inference_steps,
            seed,
        )

    def _prefill_visual_cache(
        self, visual_tokens: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        zeros = torch.zeros(
            (int(visual_tokens.shape[0]), VISUAL_TOKEN_COUNT),
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )
        visual_pre = self.visual_dit.pre_dit(visual_tokens, zeros, context, context_mask)
        return self.mot.prefill_video_cache(
            video_tokens=visual_pre["tokens"],
            video_freqs=visual_pre["freqs"],
            video_t_mod=visual_pre["t_mod"],
            video_context_payload={
                "context": visual_pre["context"],
                "mask": visual_pre["context_mask"],
            },
            video_attention_mask=visual_pre["attention_mask"],
        )

    def _predict_action_with_cache(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        cache: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(action, timestep, context, context_mask)
        tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=cache,
            attention_mask=build_v5_joint_attention_mask(device=action.device),
            video_seq_len=VISUAL_TOKEN_COUNT,
        )
        return self.action_expert.post_dit(tokens, action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        visual_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int,
        seed: int,
        initial_action_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tuple(visual_tokens.shape[1:]) != (VISUAL_TOKEN_COUNT, VJEPA_DIM):
            raise ValueError("V5 action inference visual_tokens must be [B,216,1408].")
        cache = self._prefill_visual_cache(visual_tokens, context, context_mask)
        expected_noise_shape = (int(visual_tokens.shape[0]), ACTION_HORIZON, 7)
        if initial_action_noise is None:
            action = self._seeded_noise(
                expected_noise_shape,
                seed=seed,
                device=visual_tokens.device,
                dtype=visual_tokens.dtype,
            )
        else:
            if tuple(initial_action_noise.shape) != expected_noise_shape:
                raise ValueError(f"initial_action_noise must be {expected_noise_shape}.")
            action = initial_action_noise.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
        timesteps, deltas = self.action_infer_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=action.device,
            dtype=action.dtype,
        )
        for timestep_value, delta in zip(timesteps, deltas):
            timestep = timestep_value.expand(int(action.shape[0]))
            prediction = self._predict_action_with_cache(
                action, timestep, context, context_mask, cache
            )
            action = self.action_infer_scheduler.step(prediction, delta, action)
            _require_finite("inferred_action", action)
        return action

    @torch.no_grad()
    def infer_joint(
        self,
        *,
        agentview_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        num_visual_inference_steps: int = 10,
        num_action_inference_steps: int = 10,
        seed: int = 42,
    ) -> dict[str, Any]:
        z0 = self.encode_current(agentview_rgb, wrist_rgb)
        base_context, base_mask = self.build_base_context(context, context_mask, proprio)
        future = self.infer_future_jepa(
            z0,
            base_context,
            base_mask,
            num_inference_steps=num_visual_inference_steps,
            seed=seed,
        )
        visual_tokens = torch.cat((z0, future["z1"], future["z2"]), dim=1)
        action = self.infer_action(
            visual_tokens,
            base_context,
            base_mask,
            num_inference_steps=num_action_inference_steps,
            seed=seed + 1,
        )
        return {
            "action": action,
            "z0": z0,
            "z1": future["z1"],
            "z2": future["z2"],
            "future_debug": future["debug"],
        }
