from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vjepa.vjepa_ac_joint_predictor import VJepaACJointPredictor
from ..vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper
from .action_dit import ActionDiT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


class FastWAMJEPAJoint(nn.Module):
    """FastWAM-Joint in V-JEPA feature space.

    This v1 skeleton keeps the FastWAM action flow path while replacing the
    video latent branch with V-JEPA feature tokens:

        video [B, 3, T, H_img, W_img]
        -> current/future V-JEPA tokens [B, N, D_v]

        action [B, T_a, A]
        -> noisy_action [B, T_a, A]
        -> ActionDiT.pre_dit(...)
        -> action_tokens [B, T_a, D_h]

        main tokens = visual tokens + future query tokens + action tokens
        condition context = text/proprio tokens, used by cross-attention
    """

    def __init__(
        self,
        *,
        action_expert: ActionDiT,
        vjepa_encoder: Optional[VJepaEncoderWrapper] = None,
        joint_predictor: Optional[VJepaACJointPredictor] = None,
        action_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        vjepa_dim: int = 1408,
        num_future_tokens: int = 256,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
        action_train_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        lambda_future: float = 0.1,
        current_frame_count: int = 2,
        future_frame_count: int = 1,
    ) -> None:
        super().__init__()
        if action_expert is None:
            raise ValueError("`action_expert` is required for FastWAMJEPAJoint.")

        self.action_expert = action_expert
        self.action_dim = int(action_dim if action_dim is not None else action_expert.action_dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else action_expert.hidden_dim)
        self.vjepa_dim = int(vjepa_dim)
        self.num_future_tokens = int(num_future_tokens)
        self.text_dim = int(text_dim if text_dim is not None else action_expert.text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.device = None if device is None else torch.device(device)
        self.torch_dtype = torch_dtype
        self.lambda_future = float(lambda_future)
        self.current_frame_count = int(current_frame_count)
        self.future_frame_count = int(future_frame_count)

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
        if self.vjepa_dim <= 0:
            raise ValueError(f"`vjepa_dim` must be positive, got {self.vjepa_dim}.")
        if self.num_future_tokens <= 0:
            raise ValueError(
                f"`num_future_tokens` must be positive, got {self.num_future_tokens}."
            )
        if self.current_frame_count <= 0:
            raise ValueError(
                f"`current_frame_count` must be positive, got {self.current_frame_count}."
            )
        if self.future_frame_count <= 0:
            raise ValueError(
                f"`future_frame_count` must be positive, got {self.future_frame_count}."
            )

        self.vjepa_encoder = vjepa_encoder or VJepaEncoderWrapper(
            dummy=True,
            num_tokens=self.num_future_tokens,
            vjepa_dim=self.vjepa_dim,
            freeze=True,
        )
        self.joint_predictor = joint_predictor or VJepaACJointPredictor(
            vjepa_dim=self.vjepa_dim,
            hidden_dim=self.hidden_dim,
            num_future_tokens=self.num_future_tokens,
            text_dim=self.text_dim,
        )

        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
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
            raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")

        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        elif proprio.ndim != 2:
            raise ValueError(
                "`proprio` must be 2D [B, D_p] or 3D [B, T, D_p], "
                f"got shape {tuple(proprio.shape)}."
            )
        if proprio.shape[0] != context.shape[0]:
            raise ValueError(
                f"`proprio` batch dimension must match context, got {proprio.shape[0]} vs {context.shape[0]}."
            )
        if proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}."
            )

        proprio_token = self.proprio_encoder(
            proprio.to(device=context.device, dtype=context.dtype).unsqueeze(1)
        )
        proprio_mask = torch.ones(
            (context_mask.shape[0], 1),
            dtype=torch.bool,
            device=context_mask.device,
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _build_inputs(self, sample: dict) -> dict[str, torch.Tensor | None]:
        required = ("video", "action", "context", "context_mask")
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"FastWAMJEPAJoint training sample is missing keys: {missing}.")

        video = sample["video"]
        action = sample["action"]
        context = sample["context"]
        context_mask = sample["context_mask"]

        if video.ndim != 5:
            raise ValueError(
                "`sample['video']` must be 5D [B, 3, T, H_img, W_img], "
                f"got shape {tuple(video.shape)}."
            )
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dim must be 3, got {video.shape[1]}.")
        if video.shape[2] < 2:
            raise ValueError(f"`sample['video']` must contain current and future frames, got T={video.shape[2]}.")
        if action.ndim != 3:
            raise ValueError(
                "`sample['action']` must be 3D [B, T_a, A], "
                f"got shape {tuple(action.shape)}."
            )
        if action.shape[0] != video.shape[0] or action.shape[2] != self.action_dim:
            raise ValueError(
                "`sample['action']` shape mismatch: expected [B, T_a, action_dim] "
                f"with B={video.shape[0]}, action_dim={self.action_dim}; got {tuple(action.shape)}."
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                "`context/context_mask` must be [B, L, D_t]/[B, L], "
                f"got {tuple(context.shape)} and {tuple(context_mask.shape)}."
            )
        if context.shape[0] != video.shape[0] or context_mask.shape[:2] != context.shape[:2]:
            raise ValueError(
                "`context_mask` must match context [B, L], "
                f"got context={tuple(context.shape)}, mask={tuple(context_mask.shape)}."
            )
        if context.shape[2] != self.text_dim:
            raise ValueError(f"`context` last dim must be {self.text_dim}, got {context.shape[2]}.")

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            if action_is_pad.ndim != 2 or tuple(action_is_pad.shape) != tuple(action.shape[:2]):
                raise ValueError(
                    "`sample['action_is_pad']` must be [B, T_a] matching action, "
                    f"got {tuple(action_is_pad.shape)} vs {tuple(action.shape[:2])}."
                )
            action_is_pad = action_is_pad.to(device=self._runtime_device(), dtype=torch.bool)

        device = self._runtime_device()
        context = context.to(device=device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        context, context_mask = self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=sample.get("proprio"),
        )

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

        # FastWAMJEPAJoint v1 defaults to real consecutive two-frame current
        # video and the following one frame as the future target. This does not
        # modify original FastWAM default behavior.
        total_required = self.current_frame_count + self.future_frame_count
        if video.shape[2] < total_required:
            raise ValueError(
                "`sample['video']` does not contain enough frames for FastWAMJEPAJoint, "
                f"got T={video.shape[2]}, required at least {total_required} "
                f"({self.current_frame_count} current + {self.future_frame_count} future)."
            )
        current_video = video[:, :, : self.current_frame_count]
        future_video = video[:, :, self.current_frame_count : total_required]

        current_visual_tokens = self.vjepa_encoder(current_video)
        target_future_tokens = self.vjepa_encoder(future_video).detach()
        if target_future_tokens.shape[1] != self.num_future_tokens:
            raise ValueError(
                "Target future V-JEPA token count must match joint predictor future tokens, "
                f"got {target_future_tokens.shape[1]} vs {self.num_future_tokens}."
            )

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self._runtime_device(),
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action,
            noise_action,
            timestep_action,
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=condition_context,
            context_mask=condition_mask,
        )
        action_tokens = action_pre["tokens"]

        joint_out = self.joint_predictor(
            current_visual_tokens=current_visual_tokens,
            action_tokens=action_tokens,
            condition_context=condition_context,
            condition_mask=condition_mask,
        )
        pred_action_flow = self.action_expert.post_dit(
            joint_out["updated_action_tokens"],
            action_pre,
        )
        pred_future_tokens = joint_out["pred_future_tokens"]
        if pred_future_tokens.shape != target_future_tokens.shape:
            raise ValueError(
                "`pred_future_tokens` shape must match `target_future_tokens`, "
                f"got {tuple(pred_future_tokens.shape)} vs {tuple(target_future_tokens.shape)}."
            )

        action_loss_token = F.mse_loss(
            pred_action_flow.float(),
            target_action.float(),
            reduction="none",
        ).mean(dim=2)
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
        loss_future_vjepa = F.smooth_l1_loss(
            pred_future_tokens.float(),
            target_future_tokens.float(),
        )
        loss_total = loss_action + self.lambda_future * loss_future_vjepa

        loss_dict = {
            "loss_total": float(loss_total.detach().item()),
            "loss_action": float(loss_action.detach().item()),
            "loss_future_vjepa": float(loss_future_vjepa.detach().item()),
        }
        return loss_total, loss_dict

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
