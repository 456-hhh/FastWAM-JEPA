from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from fastwam.models.wan22.pairwise_conditional_latent_v3 import (
    ActionEncoder,
    LanguageProjector,
    TextToActionHead,
    contrastive_loss,
)


def _fit_sequence(
    value: torch.Tensor,
    length: int,
    *,
    dim: int,
    pad_value: bool | float = 0.0,
) -> torch.Tensor:
    """Pad or truncate a tensor along one dimension.

    This mirrors the fixed-length handling in
    tools/train_fastwam_jepa_idm_v3_stage1_text_action.py.
    """
    current = int(value.shape[dim])
    target = int(length)
    if current == target:
        return value
    if current > target:
        return value.narrow(dim, 0, target)

    pad_shape = list(value.shape)
    pad_shape[dim] = target - current
    if value.dtype == torch.bool:
        pad = torch.full(
            pad_shape,
            bool(pad_value),
            dtype=value.dtype,
            device=value.device,
        )
    else:
        pad = torch.full(
            pad_shape,
            float(pad_value),
            dtype=value.dtype,
            device=value.device,
        )
    return torch.cat([value, pad], dim=dim)


def _mean_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tokens):
        raise TypeError(f"`tokens` must be a torch.Tensor, got {type(tokens)}.")
    if tokens.ndim != 3:
        raise ValueError(f"`tokens` must be [B, K, D], got shape {tuple(tokens.shape)}.")
    if int(tokens.shape[1]) <= 0:
        raise ValueError(f"`tokens` must contain at least one token, got shape {tuple(tokens.shape)}.")
    return tokens.mean(dim=1)


class PairwiseStage1TextActionCompatWrapper(nn.Module):
    """Stage1 text-action compatibility wrapper for Stage5/6 integration.

    This wrapper adapts the current Stage1 text-action latent modules to the
    interface needed by later Stage5/6 code.

    Current Stage1 behavior:
        text/context -> z_l
        action_chunk -> z_a
        z_l -> q_a_text
        contrastive(q_a_text, z_a)

    Stage5/6-facing outputs:
        z_task_token = q_a_text        # [B, 4, 1024]
        z_task = mean(q_a_text, dim=1) # [B, 1024]
        loss_vlp_to_a = text->action contrastive loss for now
        loss_va_to_l = 0 placeholder for now

    Important:
        This is not the final Stage1-4 pairwise latent model. It is a
        compatibility layer so Stage5/6 can be written now and later swapped to
        a fuller V/L/P/A implementation with minimal changes.
    """

    def __init__(
        self,
        *,
        text_dim: int = 4096,
        action_dim: int = 7,
        latent_dim: int = 1024,
        num_latent_tokens: int = 4,
        max_text_tokens: int = 128,
        action_horizon: int = 32,
        tau: float = 0.07,
    ) -> None:
        super().__init__()

        if int(text_dim) != 4096:
            raise ValueError(f"Current Stage1 LanguageProjector expects text_dim=4096, got {text_dim}.")
        if int(action_dim) != 7:
            raise ValueError(f"Current Stage1 ActionEncoder expects action_dim=7, got {action_dim}.")
        if int(latent_dim) != 1024:
            raise ValueError(f"Current Stage1 modules expect latent_dim=1024, got {latent_dim}.")
        if int(num_latent_tokens) != 4:
            raise ValueError(f"Current Stage1 modules expect num_latent_tokens=4, got {num_latent_tokens}.")
        if int(max_text_tokens) != 128:
            raise ValueError(f"Current Stage1 LanguageProjector expects max_text_tokens=128, got {max_text_tokens}.")
        if int(action_horizon) != 32:
            raise ValueError(f"Current Stage1 ActionEncoder expects action_horizon=32, got {action_horizon}.")
        if float(tau) <= 0.0:
            raise ValueError(f"`tau` must be positive, got {tau}.")

        self.text_dim = int(text_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.num_latent_tokens = int(num_latent_tokens)
        self.max_text_tokens = int(max_text_tokens)
        self.action_horizon = int(action_horizon)
        self.tau = float(tau)

        # Use the existing Stage1 modules exactly as the current training script does.
        self.language_projector = LanguageProjector()
        self.action_encoder = ActionEncoder()
        self.text_to_action_head = TextToActionHead()

    def _prepare_text(
        self,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(text_tokens):
            raise TypeError(f"`text_tokens` must be a torch.Tensor, got {type(text_tokens)}.")
        if text_tokens.ndim != 3:
            raise ValueError(f"`text_tokens` must be [B, L, D_text], got shape {tuple(text_tokens.shape)}.")
        if int(text_tokens.shape[-1]) != self.text_dim:
            raise ValueError(
                "`text_tokens` last dim mismatch, "
                f"got {text_tokens.shape[-1]} vs expected {self.text_dim}."
            )

        batch_size = int(text_tokens.shape[0])
        text_tokens = _fit_sequence(
            text_tokens,
            self.max_text_tokens,
            dim=1,
            pad_value=0.0,
        )

        if text_mask is None:
            original_len = min(int(text_tokens.shape[1]), self.max_text_tokens)
            text_mask = torch.zeros(
                (batch_size, self.max_text_tokens),
                dtype=torch.bool,
                device=text_tokens.device,
            )
            text_mask[:, :original_len] = True
        else:
            if not torch.is_tensor(text_mask):
                raise TypeError(f"`text_mask` must be a torch.Tensor, got {type(text_mask)}.")
            if text_mask.ndim != 2:
                raise ValueError(f"`text_mask` must be [B, L], got shape {tuple(text_mask.shape)}.")
            if int(text_mask.shape[0]) != batch_size:
                raise ValueError(
                    "`text_mask` batch mismatch, "
                    f"got {text_mask.shape[0]} vs expected {batch_size}."
                )
            text_mask = _fit_sequence(
                text_mask.to(device=text_tokens.device, dtype=torch.bool),
                self.max_text_tokens,
                dim=1,
                pad_value=False,
            )

        if tuple(text_mask.shape) != tuple(text_tokens.shape[:2]):
            raise ValueError(
                "`text_mask` must match text_tokens [B, L], "
                f"got {tuple(text_mask.shape)} vs {tuple(text_tokens.shape[:2])}."
            )
        return text_tokens, text_mask

    def _prepare_action(self, action_chunk: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(action_chunk):
            raise TypeError(f"`action_chunk` must be a torch.Tensor, got {type(action_chunk)}.")
        if action_chunk.ndim != 3:
            raise ValueError(f"`action_chunk` must be [B, T, action_dim], got shape {tuple(action_chunk.shape)}.")
        if int(action_chunk.shape[-1]) != self.action_dim:
            raise ValueError(
                "`action_chunk` last dim mismatch, "
                f"got {action_chunk.shape[-1]} vs expected {self.action_dim}."
            )
        return _fit_sequence(
            action_chunk,
            self.action_horizon,
            dim=1,
            pad_value=0.0,
        )

    def encode_text(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_tokens, text_mask = self._prepare_text(text_tokens, text_mask)
        return self.language_projector(text_tokens, text_mask=text_mask)

    def encode_action(self, *, action_chunk: torch.Tensor) -> torch.Tensor:
        action_chunk = self._prepare_action(action_chunk)
        return self.action_encoder(action_chunk)

    def forward_train(
        self,
        *,
        text_tokens: torch.Tensor,
        action_chunk: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        world_tokens: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Training path.

        world_tokens and proprio are accepted for future Stage2-4 compatibility,
        but this Stage1 wrapper intentionally ignores them.
        """
        del world_tokens, proprio

        z_l = self.encode_text(text_tokens=text_tokens, text_mask=text_mask)
        z_a = self.encode_action(action_chunk=action_chunk)
        q_a_text = self.text_to_action_head(z_l)

        loss_vlp_to_a, retrieval_acc = contrastive_loss(
            q_a_text,
            z_a,
            tau=float(self.tau if tau is None else tau),
        )
        loss_va_to_l = loss_vlp_to_a.new_zeros(())

        z_task_token = q_a_text
        z_task = _mean_pool_tokens(z_task_token)

        if torch.is_tensor(retrieval_acc):
            retrieval_acc_tensor = retrieval_acc.detach()
        else:
            retrieval_acc_tensor = loss_vlp_to_a.new_tensor(float(retrieval_acc))

        return {
            "z_l": z_l,
            "z_a": z_a,
            "q_a_text": q_a_text,
            "z_task": z_task,
            "z_task_token": z_task_token,
            "loss_vlp_to_a": loss_vlp_to_a,
            "loss_va_to_l": loss_va_to_l,
            "retrieval_acc_vlp_to_a": retrieval_acc_tensor,
        }

    def forward_infer(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        world_tokens: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Inference path.

        This method intentionally does not accept action_chunk as a named
        argument. Stage5/6 rollout must not use expert action or future tokens.
        """
        del world_tokens, proprio

        z_l = self.encode_text(text_tokens=text_tokens, text_mask=text_mask)
        q_a_text = self.text_to_action_head(z_l)

        z_task_token = q_a_text
        z_task = _mean_pool_tokens(z_task_token)

        return {
            "z_l": z_l,
            "q_a_text": q_a_text,
            "z_task": z_task,
            "z_task_token": z_task_token,
        }

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        if "action_chunk" in kwargs and kwargs["action_chunk"] is not None:
            return self.forward_train(*args, **kwargs)
        return self.forward_infer(*args, **kwargs)


def load_stage1_text_action_checkpoint(
    wrapper: PairwiseStage1TextActionCompatWrapper,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load the current Stage1 text-action checkpoint format.

    Expected checkpoint keys:
        language_projector
        action_encoder
        text_to_action_head
    """
    if not isinstance(wrapper, PairwiseStage1TextActionCompatWrapper):
        raise TypeError(
            "`wrapper` must be PairwiseStage1TextActionCompatWrapper, "
            f"got {type(wrapper)}."
        )

    path = Path(checkpoint_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Stage1 text-action checkpoint does not exist: {path}")

    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Stage1 checkpoint must be a dict, got {type(checkpoint)}.")

    modules = {
        "language_projector": wrapper.language_projector,
        "action_encoder": wrapper.action_encoder,
        "text_to_action_head": wrapper.text_to_action_head,
    }

    stats: dict[str, Any] = {}
    for key, module in modules.items():
        state = checkpoint.get(key)
        if not isinstance(state, dict):
            if strict:
                raise ValueError(f"Stage1 checkpoint missing `{key}` state_dict: {path}")
            stats[key] = {
                "loaded": False,
                "missing": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "loaded_keys_count": 0,
            }
            continue

        missing_keys, unexpected_keys = module.load_state_dict(state, strict=strict)
        stats[key] = {
            "loaded": True,
            "missing": False,
            "missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys),
            "loaded_keys_count": len(state),
        }

    return stats
