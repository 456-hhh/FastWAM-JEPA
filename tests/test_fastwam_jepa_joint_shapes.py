import pytest
import torch
import torch.nn as nn

from fastwam.models.vjepa.vjepa_ac_joint_predictor import VJepaACJointPredictor
from fastwam.models.wan22.fastwam_jepa_joint import FastWAMJEPAJoint


class FakeActionExpert(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, text_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.last_updated_action_tokens_shape = None
        self.last_pred_action_flow_shape = None

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = self.action_encoder(action_tokens)
        return {
            "tokens": tokens,
            "context": context,
            "context_mask": context_mask,
            "timestep": timestep,
        }

    def post_dit(
        self,
        updated_action_tokens: torch.Tensor,
        action_pre: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del action_pre
        pred_action_flow = self.action_head(updated_action_tokens)
        self.last_updated_action_tokens_shape = tuple(updated_action_tokens.shape)
        self.last_pred_action_flow_shape = tuple(pred_action_flow.shape)
        return pred_action_flow


def test_fastwam_jepa_joint_training_loss_dummy_shapes():
    batch_size = 2
    num_frames = 4
    action_horizon = 4
    action_dim = 7
    image_size = 16
    text_dim = 32
    hidden_dim = 64
    vjepa_dim = 48
    num_future_tokens = 8
    text_len = 5
    proprio_dim = 9

    sample = {
        "video": torch.randn(batch_size, 3, num_frames, image_size, image_size),
        "action": torch.randn(batch_size, action_horizon, action_dim),
        "context": torch.randn(batch_size, text_len, text_dim),
        "context_mask": torch.ones(batch_size, text_len, dtype=torch.bool),
        "proprio": torch.randn(batch_size, num_frames, proprio_dim),
        "action_is_pad": torch.zeros(batch_size, action_horizon, dtype=torch.bool),
    }

    fake_action_expert = FakeActionExpert(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        text_dim=text_dim,
    )
    model = FastWAMJEPAJoint(
        action_expert=fake_action_expert,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        vjepa_dim=vjepa_dim,
        num_future_tokens=num_future_tokens,
        text_dim=text_dim,
        proprio_dim=proprio_dim,
        lambda_future=0.1,
    )

    loss_total, loss_dict = model.training_loss(sample)

    assert loss_total.ndim == 0
    assert torch.isfinite(loss_total)
    assert {"loss_total", "loss_action", "loss_future_vjepa"} <= set(loss_dict)
    assert fake_action_expert.last_updated_action_tokens_shape == (
        batch_size,
        action_horizon,
        hidden_dim,
    )
    assert fake_action_expert.last_pred_action_flow_shape == (
        batch_size,
        action_horizon,
        action_dim,
    )


def test_vjepa_ac_joint_predictor_output_shapes():
    batch_size = 2
    num_visual_tokens = 6
    action_horizon = 4
    hidden_dim = 64
    vjepa_dim = 48
    num_future_tokens = 8
    text_len = 5
    text_dim = 32

    predictor = VJepaACJointPredictor(
        vjepa_dim=vjepa_dim,
        hidden_dim=hidden_dim,
        num_future_tokens=num_future_tokens,
        text_dim=text_dim,
    )

    out = predictor(
        current_visual_tokens=torch.randn(batch_size, num_visual_tokens, vjepa_dim),
        action_tokens=torch.randn(batch_size, action_horizon, hidden_dim),
        condition_context=torch.randn(batch_size, text_len, text_dim),
        condition_mask=torch.ones(batch_size, text_len, dtype=torch.bool),
    )

    assert out["updated_action_tokens"].shape == (batch_size, action_horizon, hidden_dim)
    assert out["future_hidden_tokens"].shape == (batch_size, num_future_tokens, hidden_dim)
    assert out["pred_future_tokens"].shape == (batch_size, num_future_tokens, vjepa_dim)


def test_vjepa_ac_joint_predictor_rejects_raw_noisy_action_shape():
    batch_size = 2
    num_visual_tokens = 6
    action_horizon = 4
    action_dim = 7
    hidden_dim = 64
    vjepa_dim = 48
    num_future_tokens = 8
    text_len = 5
    text_dim = 32

    predictor = VJepaACJointPredictor(
        vjepa_dim=vjepa_dim,
        hidden_dim=hidden_dim,
        num_future_tokens=num_future_tokens,
        text_dim=text_dim,
    )

    with pytest.raises(ValueError):
        predictor(
            current_visual_tokens=torch.randn(batch_size, num_visual_tokens, vjepa_dim),
            action_tokens=torch.randn(batch_size, action_horizon, action_dim),
            condition_context=torch.randn(batch_size, text_len, text_dim),
            condition_mask=torch.ones(batch_size, text_len, dtype=torch.bool),
        )
