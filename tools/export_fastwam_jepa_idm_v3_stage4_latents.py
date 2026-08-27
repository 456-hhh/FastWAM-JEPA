from __future__ import annotations

import argparse
import hashlib
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
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import train_fastwam_jepa_idm_v2_stage2_libero as stage2_libero
import train_fastwam_jepa_idm_v3_stage3_vlp_action as stage3_train
from fastwam.models.wan22.pairwise_stage4 import Stage4VLPVAActionModel


MODULE_KEYS = (
    "language_projector",
    "action_encoder",
    "vision_projector",
    "proprio_projector",
    "fusion_vlp",
    "fusion_va",
)
METADATA_KEYS = ("task_id", "task_index", "episode_id", "timestep")
INSTRUCTION_KEYS = ("instruction", "text", "language")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FastWAM-JEPA v3 Stage4 representation latents.")
    parser.add_argument("--stage4-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="libero_idm_2cam224_1e-4")
    parser.add_argument("--split", default="train", choices=("train", "val", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--proprio-dim", type=int, default=8)
    return parser.parse_args()


def _strip_uniform_module_prefix(state: dict[str, Any], *, name: str) -> dict[str, Any]:
    keys = list(state)
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError(f"{name} state_dict mixes module.-prefixed and unprefixed keys.")
    if prefixed and all(prefixed):
        return {key[len("module."):]: value for key, value in state.items()}
    return state


def load_stage4_checkpoint_strict(
    model: Stage4VLPVAActionModel,
    checkpoint_path: str | Path,
) -> Path:
    path = stage2_libero.require_file(checkpoint_path, name="--stage4-checkpoint")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Stage4 checkpoint must be a dict, got {type(payload)}.")
    checkpoint_stage = payload.get("stage")
    if checkpoint_stage is not None and checkpoint_stage != "stage4_vlp_va":
        raise RuntimeError(
            f"Expected stage='stage4_vlp_va', got {checkpoint_stage!r} in {path}."
        )
    for name in MODULE_KEYS:
        state = payload.get(name)
        if not isinstance(state, dict):
            raise RuntimeError(f"Stage4 checkpoint missing required {name} state_dict: {path}")
        normalized = _strip_uniform_module_prefix(state, name=name)
        try:
            getattr(model, name).load_state_dict(normalized, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"Strict Stage4 load failed for {name}: {exc}") from exc
    return Path(path)


def _configure_split(cfg: Any, split: str) -> str:
    if split == "train":
        return "train"
    candidates = (split, "validation" if split == "val" else split)
    selected = next((name for name in candidates if name in cfg.data), None)
    if selected is None:
        print(f"WARNING split={split} unavailable; using shuffled train split")
        return "train_fallback"
    with open_dict(cfg):
        cfg.data.train = cfg.data[selected]
    return selected


def _batch_array(value: Any, *, batch_size: int) -> np.ndarray | None:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), batch_size)
    if int(array.shape[0]) != batch_size:
        return None
    if array.ndim > 1:
        if int(np.prod(array.shape[1:])) != 1:
            return None
        array = array.reshape(batch_size)
    if array.dtype == object:
        try:
            array = array.astype(str)
        except (TypeError, ValueError):
            return None
    return array


def _instruction_strings(value: Any, *, batch_size: int) -> np.ndarray | None:
    array = _batch_array(value, batch_size=batch_size)
    if array is None:
        return None
    if array.dtype.kind not in ("U", "S"):
        return None
    return array.astype(str)


def _context_hash_ids(
    raw_batch: dict[str, Any],
    *,
    batch_size: int,
    context_tokens: int,
) -> np.ndarray:
    context = raw_batch.get("context")
    mask = raw_batch.get("context_mask")
    if not torch.is_tensor(context) or not torch.is_tensor(mask):
        raise RuntimeError("context_hash requires tensor context/context_mask.")
    context = context[:, : int(context_tokens)].detach().float().cpu()
    mask = mask[:, : int(context_tokens)].detach().bool().cpu()
    if tuple(mask.shape) != tuple(context.shape[:2]) or int(context.shape[0]) != batch_size:
        raise RuntimeError(
            f"context/context_mask shape mismatch for hashing: {tuple(context.shape)} vs {tuple(mask.shape)}."
        )
    ids: list[str] = []
    for index in range(batch_size):
        valid = context[index][mask[index]].contiguous().numpy()
        ids.append(hashlib.sha1(valid.tobytes()).hexdigest())
    return np.asarray(ids)


def _select_instruction_ids(
    raw_batch: dict[str, Any],
    *,
    batch_size: int,
    context_tokens: int,
    required_source: str | None,
) -> tuple[np.ndarray, str, np.ndarray | None]:
    candidates: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for key in INSTRUCTION_KEYS:
        if key in raw_batch:
            strings = _instruction_strings(raw_batch[key], batch_size=batch_size)
            if strings is not None:
                ids = np.asarray(
                    [hashlib.sha1(("instruction\0" + value).encode("utf-8")).hexdigest() for value in strings]
                )
                candidates[key] = (ids, strings)
    for key in ("task_id", "task_index"):
        if key in raw_batch:
            values = _batch_array(raw_batch[key], batch_size=batch_size)
            if values is not None:
                ids = np.asarray([f"{key}:{value}" for value in values.astype(str)])
                candidates[key] = (ids, None)
    candidates["context_hash"] = (
        _context_hash_ids(
            raw_batch,
            batch_size=batch_size,
            context_tokens=context_tokens,
        ),
        None,
    )

    source = required_source or next(
        key for key in (*INSTRUCTION_KEYS, "task_id", "task_index", "context_hash") if key in candidates
    )
    if source not in candidates:
        raise RuntimeError(f"instruction_id source {source!r} disappeared from a later batch.")
    ids, strings = candidates[source]
    return ids, source, strings


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
    if int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be positive.")
    if (
        int(args.context_tokens) != 128
        or int(args.action_horizon) != 32
        or int(args.proprio_dim) != 8
    ):
        raise ValueError("Stage4 export requires context=128, action_horizon=32, proprio_dim=8.")
    if (
        int(args.current_frame_count) != 4
        or int(args.vjepa_img_size) != 256
        or int(args.raw_vjepa_tokens) != 512
        or int(args.vjepa_dim) != 1408
    ):
        raise ValueError("Stage4 export requires video [B,3,4,256,256] and raw tokens [B,512,1408].")

    stage2_libero.seed_everything(int(args.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dtype, autocast_dtype = stage2_libero.precision_to_dtype(str(args.precision), device)

    cfg = stage2_libero.compose_cfg(str(args.config_name), str(args.task))
    actual_split = _configure_split(cfg, str(args.split))
    loader, _ = stage2_libero.build_libero_loader(
        cfg,
        args=args,
        ddp_enabled=False,
        rank=0,
        world_size=1,
    )

    model = Stage4VLPVAActionModel(
        raw_vjepa_tokens=int(args.raw_vjepa_tokens),
        vjepa_dim=int(args.vjepa_dim),
        context_tokens=int(args.context_tokens),
        action_horizon=int(args.action_horizon),
        proprio_dim=int(args.proprio_dim),
    )
    checkpoint_path = load_stage4_checkpoint_strict(model, args.stage4_checkpoint)
    model = model.to(device=device, dtype=model_dtype).eval()
    model.requires_grad_(False)
    vjepa_encoder = stage3_train.build_frozen_vjepa_encoder(
        args,
        device=device,
        dtype=model_dtype,
        rank=0,
    )

    chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in ("q_l", "z_l", "z_task", "z_a", "action", "action_norm", "instruction_id")
    }
    metadata_active: set[str] | None = None
    instruction_source: str | None = None
    instruction_enabled = False
    exported = 0

    with torch.inference_mode():
        for raw_batch in loader:
            if exported >= int(args.max_samples):
                break
            batch = stage3_train.canonicalize_stage3_vlp_batch(
                raw_batch,
                args=args,
                device=device,
                dtype=model_dtype,
            )
            batch_size = int(batch["action"].shape[0])
            take = min(batch_size, int(args.max_samples) - exported)
            ids, source, instructions = _select_instruction_ids(
                raw_batch,
                batch_size=batch_size,
                context_tokens=int(args.context_tokens),
                required_source=instruction_source,
            )
            instruction_source = source
            if exported == 0 and instructions is not None:
                chunks["instruction"] = []
                instruction_enabled = True
            if instruction_enabled and instructions is None:
                raise RuntimeError("Raw instruction text disappeared from a later batch.")

            autocast_context = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )
            current_jepa_tokens = stage3_train.encode_current_jepa_tokens(
                vjepa_encoder=vjepa_encoder,
                current_video=batch["current_video"],
                args=args,
            )
            with autocast_context:
                out = model(
                    current_jepa_tokens=current_jepa_tokens,
                    context=batch["context"],
                    context_mask=batch["context_mask"],
                    proprio=batch["proprio"],
                    action=batch["action"],
                    tau=0.07,
                )
            for key in ("q_l", "z_l", "z_task", "z_a"):
                chunks[key].append(out[key][:take].detach().float().cpu().numpy())
            action = batch["action"][:take].detach().float().cpu()
            chunks["action"].append(action.numpy())
            chunks["action_norm"].append(action.flatten(1).norm(dim=1).numpy())
            chunks["instruction_id"].append(ids[:take])
            if instruction_enabled:
                assert instructions is not None
                chunks["instruction"].append(instructions[:take])

            available_metadata: dict[str, np.ndarray] = {}
            for key in METADATA_KEYS:
                if key in raw_batch:
                    value = _batch_array(raw_batch[key], batch_size=batch_size)
                    if value is not None:
                        available_metadata[key] = value[:take]
            if metadata_active is None:
                metadata_active = set(available_metadata)
                for key in metadata_active:
                    chunks[key] = []
            else:
                for key in list(metadata_active - set(available_metadata)):
                    metadata_active.remove(key)
                    chunks.pop(key, None)
                    print(f"WARNING skipped inconsistent metadata key={key}")
            for key in metadata_active:
                chunks[key].append(available_metadata[key])
            exported += take

    if exported <= 0:
        raise RuntimeError("No Stage4 samples were exported.")
    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items() if value}
    arrays["instruction_id_source"] = np.asarray(instruction_source or "unknown")
    arrays["split"] = np.asarray(actual_split)
    arrays["stage"] = np.asarray("stage4_vlp_va")
    arrays["checkpoint"] = np.asarray(str(checkpoint_path))

    output = _output_path(args.output)
    np.savez_compressed(output, **arrays)
    unique_ids, counts = np.unique(arrays["instruction_id"].astype(str), return_counts=True)
    print(f"output={output}")
    print(f"num_samples={exported}")
    print(f"instruction_id_source={instruction_source}")
    print(f"instruction_categories={len(unique_ids)}")
    print(f"instruction_count_range={int(counts.min())}-{int(counts.max())}")
    for key in sorted(arrays):
        print(f"key={key} shape={arrays[key].shape} dtype={arrays[key].dtype}")


if __name__ == "__main__":
    main()
