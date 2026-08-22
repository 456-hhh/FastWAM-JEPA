from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from train_fastwam_jepa_idm_v21_stage2_libero import (
    canonicalize_v21_libero_batch,
    predicted_action_forward,
)


class DummyAdapter(nn.Module):
    def forward(
        self,
        *,
        current_jepa_tokens: torch.Tensor,
        future_jepa_tokens: torch.Tensor,
        base_context: torch.Tensor,
        base_context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_scalar = current_jepa_tokens.float().mean(dim=(1, 2), keepdim=True)
        future_scalar = future_jepa_tokens.float().mean(dim=(1, 2), keepdim=True)
        offset = (current_scalar + future_scalar).to(base_context.dtype)
        return base_context + offset, base_context_mask


class DummyActionExpert(nn.Module):
    def forward(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        del context_mask
        condition = context.float().mean(dim=(1, 2), keepdim=True).to(action_tokens.dtype)
        time = timestep.reshape(-1, 1, 1).to(action_tokens.dtype)
        return action_tokens + condition + time


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.jepa_adapter = DummyAdapter()
        self.action_expert = DummyActionExpert()


def main() -> None:
    torch.manual_seed(21)
    batch_size = 2
    video = torch.zeros(batch_size, 3, 33, 8, 8)
    for frame_index in range(33):
        video[:, :, frame_index].fill_(float(frame_index))
    action = torch.arange(32, dtype=torch.float32).view(1, 32, 1).expand(batch_size, -1, 7)
    proprio = torch.arange(33, dtype=torch.float32).view(1, 33, 1).expand(batch_size, -1, 8)
    batch = {
        "video": video,
        "action": action,
        "proprio": proprio,
        "context": torch.randn(batch_size, 128, 32),
        "context_mask": torch.ones(batch_size, 128, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch_size, 32, dtype=torch.bool),
    }
    args = SimpleNamespace(vjepa_img_size=8, use_proprio=True)
    canonical = canonicalize_v21_libero_batch(
        batch,
        args=args,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    current_raw = video[:, :, 0:1]
    current_repeated = canonical["video"]
    future_teacher = canonical["future_video"]
    assert tuple(current_repeated.shape) == (batch_size, 3, 4, 8, 8)
    assert torch.equal(current_repeated[:, :, 0], current_repeated[:, :, 3])
    assert torch.equal(current_repeated[:, :, 0], current_raw[:, :, 0])
    print("CURRENT_REPEAT_PASS")
    expected_future = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(future_teacher[0, 0, :, 0, 0], expected_future)
    assert float(canonical["action"][0, 0, 0]) == 0.0
    assert float(canonical["proprio"][0, 0]) == 0.0
    print("TEMPORAL_ALIGNMENT_PASS")

    current_jepa = torch.randn(batch_size, 512, 1408)
    target_future_jepa = torch.randn(batch_size, 512, 1408)
    pred_future_jepa = current_jepa * 0.5
    condition_context = torch.randn(batch_size, 16, 32)
    condition_mask = torch.ones(batch_size, 16, dtype=torch.bool)
    noisy_action = torch.randn(batch_size, 32, 7)
    timestep = torch.full((batch_size,), 0.5)
    model = DummyModel()
    first_action, action_context, _ = predicted_action_forward(
        model,
        current_jepa_tokens=current_jepa,
        pred_future_jepa_tokens=pred_future_jepa,
        condition_context=condition_context,
        condition_mask=condition_mask,
        noisy_action=noisy_action,
        timestep_action=timestep,
        action_grad_to_predictor=False,
    )
    changed_target_future_jepa = target_future_jepa + 1000.0
    second_action, second_context, _ = predicted_action_forward(
        model,
        current_jepa_tokens=current_jepa,
        pred_future_jepa_tokens=pred_future_jepa,
        condition_context=condition_context,
        condition_mask=condition_mask,
        noisy_action=noisy_action,
        timestep_action=timestep,
        action_grad_to_predictor=False,
    )
    assert not torch.equal(target_future_jepa, changed_target_future_jepa)
    assert torch.equal(action_context, second_context)
    assert torch.equal(first_action, second_action)
    print("FUTURE_LEAKAGE_PASS")
    print(f"current_raw={tuple(current_raw.shape)}")
    print(f"current_repeated={tuple(current_repeated.shape)}")
    print(f"future_teacher={tuple(future_teacher.shape)}")
    print(f"current_jepa={tuple(current_jepa.shape)}")
    print(f"target_future_jepa={tuple(target_future_jepa.shape)}")
    print(f"pred_future_jepa={tuple(pred_future_jepa.shape)}")
    print(f"action_context={tuple(action_context.shape)}")
    print(f"pred_action={tuple(first_action.shape)}")
    print("PASS")


if __name__ == "__main__":
    main()
