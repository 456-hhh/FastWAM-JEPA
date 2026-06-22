from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vjepa.jepa_fastwam_adapter import JepaToFastWAMAdapter
from ..vjepa.jepa_future_predictor import JepaFuturePredictor
from ..vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper
from .action_dit import ActionDiT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


class FastWAMJEPAIDM(nn.Module):
    """FastWAM-JEPA-IDM v2 skeleton.

    v2 keeps the action denoising path on the original ActionDiT forward path:

        noisy_action
        -> ActionDiT.pre_dit()
        -> ActionDiT.blocks
        -> ActionDiT.post_dit()

    The Wan video future branch is replaced by V-JEPA feature-space prediction:

        current_video -> V-JEPA encoder -> current_jepa_tokens
        future_video  -> V-JEPA encoder -> target_future_jepa_tokens
        current_jepa_tokens + text/proprio -> JepaFuturePredictor -> pred_future_jepa_tokens

    JepaToFastWAMAdapter converts current/predicted-future V-JEPA tokens into
    ActionDiT context tokens. This file is intentionally separate from v1
    FastWAMJEPAJoint and does not modify original FastWAM/IDM defaults.
    """

    def __init__(
        self,
        *,
        action_expert: ActionDiT,
        vjepa_encoder: Optional[VJepaEncoderWrapper] = None,
        future_predictor: Optional[JepaFuturePredictor] = None,
        jepa_adapter: Optional[JepaToFastWAMAdapter] = None,
        action_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        vjepa_dim: int = 1408,
        num_future_tokens: int = 256,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        lambda_action: float = 1.0,
        lambda_future: float = 0.1,
        current_frame_count: int = 2,
        future_frame_count: int = 2,
        adapter_current_tokens: int = 64,
        adapter_future_tokens: int = 64,
        future_predictor_layers: int = 6,
        future_predictor_heads: int = 8,
        future_source: str = "oracle",
    ) -> None:
        super().__init__()
        if action_expert is None:
            raise ValueError("`action_expert` is required.")

        self.action_expert = action_expert
        self.action_dim = int(action_dim if action_dim is not None else action_expert.action_dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else action_expert.hidden_dim)
        self.vjepa_dim = int(vjepa_dim)
        self.num_future_tokens = int(num_future_tokens)
        self.text_dim = int(text_dim if text_dim is not None else action_expert.text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.device = None if device is None else torch.device(device)
        self.torch_dtype = torch_dtype
        self.lambda_action = float(lambda_action)
        self.lambda_future = float(lambda_future)
        self.current_frame_count = int(current_frame_count)
        self.future_frame_count = int(future_frame_count)
        self.future_source = str(future_source)
        self.last_forward_shapes: dict[str, tuple[int, ...] | str] = {}

        if self.action_dim != int(action_expert.action_dim):
            raise ValueError(
                f"`action_dim` must match action_expert.action_dim, "
                f"got {self.action_dim} vs {action_expert.action_dim}."
            )
        if self.hidden_dim != int(action_expert.hidden_dim):
            raise ValueError(
                f"`hidden_dim` must match action_expert.hidden_dim, "
                f"got {self.hidden_dim} vs {action_expert.hidden_dim}."
            )
        if self.text_dim != int(action_expert.text_dim):
            raise ValueError(
                f"`text_dim` must match action_expert.text_dim, "
                f"got {self.text_dim} vs {action_expert.text_dim}."
            )
        if self.current_frame_count <= 0 or self.future_frame_count <= 0:
            raise ValueError(
                "`current_frame_count` and `future_frame_count` must be positive, "
                f"got {self.current_frame_count} and {self.future_frame_count}."
            )
        if self.future_source not in {"oracle", "predicted", "no_future"}:
            raise ValueError(
                "`future_source` must be one of {'oracle', 'predicted', 'no_future'}, "
                f"got {self.future_source!r}."
            )

        self.vjepa_encoder = vjepa_encoder or VJepaEncoderWrapper(
            dummy=True,
            num_tokens=self.num_future_tokens,
            vjepa_dim=self.vjepa_dim,
            freeze=True,
        )
        self.future_predictor = future_predictor or JepaFuturePredictor(
            vjepa_dim=self.vjepa_dim,
            hidden_dim=self.hidden_dim,
            num_future_tokens=self.num_future_tokens,
            text_dim=self.text_dim,
            num_layers=int(future_predictor_layers),
            num_heads=int(future_predictor_heads),
        )
        self.jepa_adapter = jepa_adapter or JepaToFastWAMAdapter(
            vjepa_dim=self.vjepa_dim,
            text_dim=self.text_dim,
            num_current_context_tokens=int(adapter_current_tokens),
            num_future_context_tokens=int(adapter_future_tokens),
        )

        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        if self.device is not None:
            self.to(device=self.device, dtype=self.torch_dtype)

    def _runtime_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _encode_jepa_video(self, video: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self.vjepa_encoder, "freeze", False)):
            with torch.no_grad():
                return self.vjepa_encoder(video)
        return self.vjepa_encoder(video)

    def _append_proprio_to_context(
        self,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError("`proprio` is required when `proprio_dim` is enabled.")
        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        elif proprio.ndim != 2:
            raise ValueError(
                "`proprio` must be [B, D_p] or [B, T, D_p], "
                f"got shape {tuple(proprio.shape)}."
            )
        if proprio.shape[0] != context.shape[0] or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                "`proprio` must match [B, proprio_dim], "
                f"got {tuple(proprio.shape)} with B={context.shape[0]}, D_p={self.proprio_dim}."
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=context.device, dtype=context.dtype).unsqueeze(1)
        )
        proprio_mask = torch.ones(
            (context_mask.shape[0], 1),
            dtype=torch.bool,
            device=context_mask.device,
        )
        return torch.cat([context, proprio_token], dim=1), torch.cat([context_mask, proprio_mask], dim=1)

    def _build_inputs(self, sample: dict) -> dict[str, torch.Tensor | None]:
        required = ("video", "action", "context", "context_mask")
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"FastWAMJEPAIDM sample is missing keys: {missing}.")

        video = sample["video"]
        action = sample["action"]
        context = sample["context"]
        context_mask = sample["context_mask"]

        if video.ndim != 5:
            raise ValueError(
                "`sample['video']` must be [B, 3, T, H, W], "
                f"got shape {tuple(video.shape)}."
            )
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dim must be 3, got {video.shape[1]}.")
        total_required = self.current_frame_count + self.future_frame_count
        if video.shape[2] < total_required:
            raise ValueError(
                "`sample['video']` does not have enough frames, "
                f"got T={video.shape[2]}, required {total_required}."
            )
        if action.ndim != 3 or action.shape[0] != video.shape[0] or action.shape[2] != self.action_dim:
            raise ValueError(
                "`sample['action']` must be [B, T_a, action_dim], "
                f"got shape {tuple(action.shape)} with B={video.shape[0]}, A={self.action_dim}."
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                "`context/context_mask` must be [B, L, D_t]/[B, L], "
                f"got {tuple(context.shape)} and {tuple(context_mask.shape)}."
            )
        if context.shape[0] != video.shape[0] or tuple(context_mask.shape) != tuple(context.shape[:2]):
            raise ValueError(
                "`context_mask` must match context [B, L], "
                f"got {tuple(context_mask.shape)} vs {tuple(context.shape[:2])}."
            )
        if context.shape[2] != self.text_dim:
            raise ValueError(f"`context` last dim must be {self.text_dim}, got {context.shape[2]}.")

        device = self._runtime_device()
        context = context.to(device=device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        context, context_mask = self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=sample.get("proprio"),
        )

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            if action_is_pad.ndim != 2 or tuple(action_is_pad.shape) != tuple(action.shape[:2]):
                raise ValueError(
                    "`action_is_pad` must be [B, T_a], "
                    f"got {tuple(action_is_pad.shape)} vs {tuple(action.shape[:2])}."
                )
            action_is_pad = action_is_pad.to(device=device, dtype=torch.bool)

        return {
            "video": video.to(device=device, dtype=self.torch_dtype),
            "action": action.to(device=device, dtype=self.torch_dtype),
            "context": context,
            "context_mask": context_mask,
            "action_is_pad": action_is_pad,
        }

    def training_loss(self, sample: dict) -> tuple[torch.Tensor, dict[str, float]]:
        inputs = self._build_inputs(sample)
        video = inputs["video"]
        action = inputs["action"]
        condition_context = inputs["context"]
        condition_mask = inputs["context_mask"]
        action_is_pad = inputs["action_is_pad"]
        batch_size = int(video.shape[0])

        current_video = video[:, :, : self.current_frame_count]
        future_video = video[:, :, self.current_frame_count : self.current_frame_count + self.future_frame_count]

        current_jepa_tokens = self._encode_jepa_video(current_video)
        target_future_jepa_tokens = self._encode_jepa_video(future_video).detach()
        future_out = self.future_predictor(
            current_jepa_tokens=current_jepa_tokens,
            condition_context=condition_context,
            condition_mask=condition_mask,
        )
        pred_future_jepa_tokens = future_out["pred_future_tokens"]
        if pred_future_jepa_tokens.shape != target_future_jepa_tokens.shape:
            raise ValueError(
                "`pred_future_jepa_tokens` shape must match target future tokens, "
                f"got {tuple(pred_future_jepa_tokens.shape)} vs {tuple(target_future_jepa_tokens.shape)}."
            )

        if self.future_source == "oracle":
            adapter_future_tokens = target_future_jepa_tokens
        elif self.future_source == "predicted":
            adapter_future_tokens = pred_future_jepa_tokens
        elif self.future_source == "no_future":
            adapter_future_tokens = None
        else:
            raise RuntimeError(f"Unexpected future_source={self.future_source!r}.")

        action_context, action_context_mask = self.jepa_adapter(
            current_jepa_tokens=current_jepa_tokens,
            future_jepa_tokens=adapter_future_tokens,
            base_context=condition_context,
            base_context_mask=condition_mask,
        )

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self._runtime_device(),
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # Critical v2 contract: call ActionDiT.forward(), which executes
        # pre_dit -> ActionDiT.blocks -> post_dit.
        pred_action = self.action_expert(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=action_context,
            context_mask=action_context_mask,
        )
        self.last_forward_shapes = {
            "future_source": self.future_source,
            "current_jepa_tokens": tuple(current_jepa_tokens.shape),
            "target_future_jepa_tokens": tuple(target_future_jepa_tokens.shape),
            "pred_future_jepa_tokens": tuple(pred_future_jepa_tokens.shape),
            "action_context": tuple(action_context.shape),
            "action_context_mask": tuple(action_context_mask.shape),
            "pred_action": tuple(pred_action.shape),
        }

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_future_jepa = F.smooth_l1_loss(
            pred_future_jepa_tokens.float(),
            target_future_jepa_tokens.float(),
        )
        loss_total = self.lambda_action * loss_action + self.lambda_future * loss_future_jepa
        return loss_total, {
            "loss_total": float(loss_total.detach().item()),
            "loss_action": float(loss_action.detach().item()),
            "loss_future_jepa": float(loss_future_jepa.detach().item()),
        }

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
