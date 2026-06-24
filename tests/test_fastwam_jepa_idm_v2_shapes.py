import torch
import torch.nn as nn

from fastwam.models.vjepa.jepa_fastwam_adapter import JepaToFastWAMAdapter
from fastwam.models.vjepa.jepa_future_predictor import JepaFuturePredictor
from fastwam.models.wan22.fastwam_jepa_idm import FastWAMJEPAIDM


class CountingBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + self.proj(x)


class FakeActionExpertWithBlocks(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int,
        text_dim: int,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.text_dim = int(text_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.blocks = nn.ModuleList([CountingBlock(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, action_dim)
        self.forward_calls = 0
        self.pre_dit_calls = 0
        self.post_dit_calls = 0
        self.last_context_shape = None
        self.last_context_mask_shape = None
        self.last_pred_action_shape = None

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.pre_dit_calls += 1
        self.last_context_shape = tuple(context.shape)
        self.last_context_mask_shape = tuple(context_mask.shape)
        return {
            "tokens": self.action_encoder(action_tokens),
            "timestep": timestep,
            "context": context,
            "context_mask": context_mask,
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, torch.Tensor]) -> torch.Tensor:
        del pre_state
        self.post_dit_calls += 1
        pred = self.head(tokens)
        self.last_pred_action_shape = tuple(pred.shape)
        return pred

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls += 1
        pre = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre["tokens"]
        for block in self.blocks:
            x = block(x)
        return self.post_dit(x, pre)


def test_jepa_future_predictor_shapes():
    batch_size = 2
    current_tokens = 8
    future_tokens = 8
    vjepa_dim = 48
    hidden_dim = 64
    text_dim = 32
    text_len = 5

    predictor = JepaFuturePredictor(
        vjepa_dim=vjepa_dim,
        hidden_dim=hidden_dim,
        num_future_tokens=future_tokens,
        text_dim=text_dim,
        num_layers=2,
        num_heads=8,
    )

    out = predictor(
        current_jepa_tokens=torch.randn(batch_size, current_tokens, vjepa_dim),
        condition_context=torch.randn(batch_size, text_len, text_dim),
        condition_mask=torch.ones(batch_size, text_len, dtype=torch.bool),
    )

    assert not hasattr(predictor, "future_query_tokens")
    assert out["future_hidden_tokens"].shape == (batch_size, future_tokens, hidden_dim)
    assert out["pred_future_tokens"].shape == (batch_size, future_tokens, vjepa_dim)

def test_jepa_future_predictor_loads_matching_vjepa2ac_trunk_keys():
    predictor = JepaFuturePredictor(
        vjepa_dim=48,
        hidden_dim=64,
        num_future_tokens=8,
        text_dim=32,
        num_layers=1,
        num_heads=8,
    )
    block = predictor.predictor_blocks[0]
    fake_vjepa2ac_state = {
        "predictor.predictor_embed.weight": torch.randn_like(predictor.predictor_embed.weight),
        "predictor.predictor_embed.bias": torch.randn_like(predictor.predictor_embed.bias),
        "predictor.predictor_blocks.0.norm1.weight": torch.randn_like(block.norm_self.weight),
        "predictor.predictor_blocks.0.norm1.bias": torch.randn_like(block.norm_self.bias),
        "predictor.predictor_blocks.0.attn.qkv.weight": torch.randn_like(block.self_attn.qkv.weight),
        "predictor.predictor_blocks.0.attn.qkv.bias": torch.randn_like(block.self_attn.qkv.bias),
        "predictor.predictor_blocks.0.attn.proj.weight": torch.randn_like(block.self_attn.out.weight),
        "predictor.predictor_blocks.0.attn.proj.bias": torch.randn_like(block.self_attn.out.bias),
        "predictor.predictor_blocks.0.norm2.weight": torch.randn_like(block.norm_mlp.weight),
        "predictor.predictor_blocks.0.norm2.bias": torch.randn_like(block.norm_mlp.bias),
        "predictor.predictor_blocks.0.mlp.fc1.weight": torch.randn_like(block.mlp[0].weight),
        "predictor.predictor_blocks.0.mlp.fc1.bias": torch.randn_like(block.mlp[0].bias),
        "predictor.predictor_blocks.0.mlp.fc2.weight": torch.randn_like(block.mlp[3].weight),
        "predictor.predictor_blocks.0.mlp.fc2.bias": torch.randn_like(block.mlp[3].bias),
        "predictor.predictor_norm.weight": torch.randn_like(predictor.predictor_norm.weight),
        "predictor.predictor_norm.bias": torch.randn_like(predictor.predictor_norm.bias),
        "predictor.predictor_proj.weight": torch.randn_like(predictor.predictor_proj.weight),
        "predictor.predictor_proj.bias": torch.randn_like(predictor.predictor_proj.bias),
        "predictor.action_encoder.weight": torch.randn(64, 7),
    }

    stats = predictor.load_vjepa_ac_trunk_state_dict(fake_vjepa2ac_state)

    assert stats["init_source"] == "vjepa2ac_pretrained"
    assert stats["loaded_keys_count"] >= 18
    assert "action_encoder.weight" in stats["skipped_keys"]
    assert stats["loaded_params_count"] > 0


def test_jepa_to_fastwam_adapter_shapes():
    batch_size = 2
    current_tokens = 8
    future_tokens = 8
    vjepa_dim = 48
    text_dim = 32
    text_len = 5
    current_context_tokens = 4
    future_context_tokens = 3

    adapter = JepaToFastWAMAdapter(
        vjepa_dim=vjepa_dim,
        text_dim=text_dim,
        num_current_context_tokens=current_context_tokens,
        num_future_context_tokens=future_context_tokens,
    )
    context, mask = adapter(
        current_jepa_tokens=torch.randn(batch_size, current_tokens, vjepa_dim),
        future_jepa_tokens=torch.randn(batch_size, future_tokens, vjepa_dim),
        base_context=torch.randn(batch_size, text_len, text_dim),
        base_context_mask=torch.ones(batch_size, text_len, dtype=torch.bool),
    )

    total_len = text_len + current_context_tokens + future_context_tokens
    assert context.shape == (batch_size, total_len, text_dim)
    assert mask.shape == (batch_size, total_len)
    assert mask.dtype == torch.bool


def test_fastwam_jepa_idm_training_loss_uses_action_blocks():
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
    action_layers = 3

    sample = {
        "video": torch.randn(batch_size, 3, num_frames, image_size, image_size),
        "action": torch.randn(batch_size, action_horizon, action_dim),
        "context": torch.randn(batch_size, text_len, text_dim),
        "context_mask": torch.ones(batch_size, text_len, dtype=torch.bool),
        "proprio": torch.randn(batch_size, num_frames, proprio_dim),
        "action_is_pad": torch.zeros(batch_size, action_horizon, dtype=torch.bool),
    }

    action_expert = FakeActionExpertWithBlocks(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        text_dim=text_dim,
        num_layers=action_layers,
    )
    model = FastWAMJEPAIDM(
        action_expert=action_expert,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        vjepa_dim=vjepa_dim,
        num_future_tokens=num_future_tokens,
        text_dim=text_dim,
        proprio_dim=proprio_dim,
        adapter_current_tokens=4,
        adapter_future_tokens=4,
        future_predictor_layers=2,
        future_predictor_heads=8,
    )

    loss_total, loss_dict = model.training_loss(sample)

    assert loss_total.ndim == 0
    assert torch.isfinite(loss_total)
    assert {"loss_total", "loss_action", "loss_future_jepa"} <= set(loss_dict)
    assert action_expert.forward_calls == 1
    assert action_expert.pre_dit_calls == 1
    assert action_expert.post_dit_calls == 1
    assert sum(block.calls for block in action_expert.blocks) == action_layers
    assert action_expert.last_pred_action_shape == (batch_size, action_horizon, action_dim)
    assert action_expert.last_context_shape[0] == batch_size
    assert action_expert.last_context_shape[2] == text_dim
    assert action_expert.last_context_mask_shape == action_expert.last_context_shape[:2]
