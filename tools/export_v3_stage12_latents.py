from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import open_dict
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (SRC_ROOT, PROJECT_ROOT, TOOLS_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import train_fastwam_jepa_idm_v2_stage2_libero as stage2_libero
import train_fastwam_jepa_idm_v3_stage1_text_action as stage1_train
import train_fastwam_jepa_idm_v3_stage2_vl_action as stage2_train
from fastwam.models.wan22.pairwise_conditional_latent_v3 import (
    ActionEncoder,
    FusionVL,
    LanguageProjector,
    TextToActionHead,
    VisionProjector,
)

OPTIONAL_LABEL_KEYS = (
    "proprio", "task_id", "task_index", "episode_id", "timestep",
    "object_pos", "object_state", "robot_state", "eef_pos", "success",
    "distance_to_goal",
)


class Stage1ExportModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_projector = LanguageProjector()
        self.action_encoder = ActionEncoder()
        self.text_to_action_head = TextToActionHead()

    def forward(
        self, context: torch.Tensor, context_mask: torch.Tensor, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        z_l = self.language_projector(context, text_mask=context_mask)
        z_a = self.action_encoder(action)
        return {"z_l": z_l, "q_a_text": self.text_to_action_head(z_l), "z_a": z_a}


class Stage2ExportModel(nn.Module):
    def __init__(self, *, raw_vjepa_tokens: int, vjepa_dim: int) -> None:
        super().__init__()
        self.language_projector = LanguageProjector()
        self.action_encoder = ActionEncoder()
        self.vision_projector = VisionProjector(
            input_dim=int(vjepa_dim), token_count=int(raw_vjepa_tokens)
        )
        self.fusion_vl = FusionVL()

    def forward(
        self,
        current_jepa_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_l = self.language_projector(context, text_mask=context_mask)
        z_v = self.vision_projector(current_jepa_tokens)
        z_a = self.action_encoder(action)
        return {
            "z_l": z_l,
            "z_v": z_v,
            "q_a_vl": self.fusion_vl(z_v, z_l),
            "z_a": z_a,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export v3 Stage1/Stage2 representation latents.")
    parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--split", default="train", choices=("train", "val", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--vjepa-repo", default=stage2_libero.DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-checkpoint", default=stage2_libero.DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", default="-1_1", choices=("-1_1", "0_1"))
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--raw-vjepa-tokens", type=int, default=512)
    parser.add_argument("--current-frame-count", type=int, default=4)
    return parser.parse_args()


def _load_module_states(checkpoint_path: str | Path, modules: dict[str, nn.Module]) -> None:
    path = stage2_libero.require_file(checkpoint_path, name="--checkpoint")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dict, got {type(payload)}.")
    for key, module in modules.items():
        state = payload.get(key)
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint missing {key} state_dict.")
        module.load_state_dict(state, strict=True)
    print(f"checkpoint={path}")


def _configure_split(cfg: Any, split: str) -> None:
    if split == "train":
        return
    candidates = (split, "validation" if split == "val" else split)
    selected = next((name for name in candidates if name in cfg.data), None)
    if selected is None:
        raise ValueError(
            f"Requested split={split!r} is unavailable; cfg.data keys={list(cfg.data.keys())}."
        )
    with open_dict(cfg):
        cfg.data.train = cfg.data[selected]


def _batch_value_to_numpy(value: Any, *, batch_size: int, key: str) -> np.ndarray | None:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), batch_size, axis=0)
    if int(array.shape[0]) != batch_size:
        return None
    if key == "proprio":
        if array.ndim >= 3:
            array = array[:, 0]
        if array.ndim != 2 or int(array.shape[1]) != 8:
            return None
    if array.dtype == object:
        try:
            array = array.astype(str)
        except (TypeError, ValueError):
            return None
    return array


def _append_latents(
    chunks: dict[str, list[np.ndarray]],
    values: dict[str, torch.Tensor],
    take: int,
) -> None:
    for key, value in values.items():
        chunks.setdefault(key, []).append(value[:take].detach().float().cpu().numpy())


def _output_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    if args.max_samples == 0:
        raise ValueError("--max-samples must be positive, or negative for no limit.")
    stage2_libero.seed_everything(int(args.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dtype, autocast_dtype = stage2_libero.precision_to_dtype(args.precision, device)

    cfg = stage2_libero.compose_cfg(args.config_name, args.task)
    _configure_split(cfg, args.split)
    loader, _ = stage2_libero.build_libero_loader(
        cfg, args=args, ddp_enabled=False, rank=0, world_size=1
    )

    if args.stage == "stage1":
        model: nn.Module = Stage1ExportModel()
        _load_module_states(
            args.checkpoint,
            {
                "language_projector": model.language_projector,
                "action_encoder": model.action_encoder,
                "text_to_action_head": model.text_to_action_head,
            },
        )
        vjepa_encoder = None
    else:
        model = Stage2ExportModel(
            raw_vjepa_tokens=int(args.raw_vjepa_tokens), vjepa_dim=int(args.vjepa_dim)
        )
        _load_module_states(
            args.checkpoint,
            {
                "language_projector": model.language_projector,
                "action_encoder": model.action_encoder,
                "vision_projector": model.vision_projector,
                "fusion_vl": model.fusion_vl,
            },
        )
        vjepa_encoder = stage2_train.build_frozen_vjepa_encoder(
            args, device=device, dtype=model_dtype, rank=0
        )

    model = model.to(device=device, dtype=model_dtype).eval()
    model.requires_grad_(False)
    latent_chunks: dict[str, list[np.ndarray]] = {}
    label_chunks: dict[str, list[np.ndarray]] = {
        "action_mean": [], "action_first": [], "action_norm": []
    }
    optional_enabled: set[str] | None = None
    skipped_labels: set[str] = set()
    exported = 0
    sample_limit = None if int(args.max_samples) < 0 else int(args.max_samples)

    with torch.inference_mode():
        for batch in loader:
            if sample_limit is not None and exported >= sample_limit:
                break
            if args.stage == "stage1":
                canonical = stage1_train.canonicalize_text_action_batch(
                    batch, args=args, device=device, dtype=model_dtype
                )
            else:
                canonical = stage2_train.canonicalize_stage2_vl_batch(
                    batch, args=args, device=device, dtype=model_dtype
                )
            batch_size = int(canonical["action"].shape[0])
            take = batch_size if sample_limit is None else min(batch_size, sample_limit - exported)
            if take <= 0:
                break
            autocast_cm = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )
            with autocast_cm:
                if args.stage == "stage1":
                    values = model(
                        canonical["context"], canonical["context_mask"], canonical["action"]
                    )
                else:
                    assert vjepa_encoder is not None
                    tokens = stage2_train.encode_current_jepa_tokens(
                        vjepa_encoder=vjepa_encoder,
                        current_video=canonical["current_video"],
                        args=args,
                    )
                    values = model(
                        tokens,
                        canonical["context"],
                        canonical["context_mask"],
                        canonical["action"],
                    )
            _append_latents(latent_chunks, values, take)

            action_cpu = canonical["action"][:take].detach().float().cpu()
            label_chunks["action_mean"].append(action_cpu.mean(dim=1).numpy())
            label_chunks["action_first"].append(action_cpu[:, 0].numpy())
            label_chunks["action_norm"].append(action_cpu.flatten(1).norm(dim=1).numpy())

            available: dict[str, np.ndarray] = {}
            for key in OPTIONAL_LABEL_KEYS:
                if key in batch:
                    array = _batch_value_to_numpy(batch[key], batch_size=batch_size, key=key)
                    if array is not None:
                        available[key] = array[:take]
            if optional_enabled is None:
                optional_enabled = set(available)
                skipped_labels.update(set(OPTIONAL_LABEL_KEYS) - optional_enabled)
                for key in optional_enabled:
                    label_chunks[key] = []
            else:
                for key in list(optional_enabled - set(available)):
                    optional_enabled.remove(key)
                    label_chunks.pop(key, None)
                    skipped_labels.add(key)
            for key in optional_enabled:
                label_chunks[key].append(available[key])
            exported += take

    if exported == 0:
        raise RuntimeError("No samples were exported.")

    arrays: dict[str, np.ndarray] = {}
    for key, chunks in {**latent_chunks, **label_chunks}.items():
        if chunks:
            arrays[key] = np.concatenate(chunks, axis=0)
    arrays["export_stage"] = np.asarray(args.stage)
    arrays["split"] = np.asarray(args.split)
    arrays["task"] = np.asarray(args.task)

    output = _output_path(args.output)
    np.savez_compressed(output, **arrays)
    print(f"output={output}")
    print(f"num_samples={exported}")
    for key in sorted(arrays):
        print(f"key={key} shape={arrays[key].shape} dtype={arrays[key].dtype}")
    print(f"skipped_labels={','.join(sorted(skipped_labels)) if skipped_labels else 'none'}")


if __name__ == "__main__":
    main()
