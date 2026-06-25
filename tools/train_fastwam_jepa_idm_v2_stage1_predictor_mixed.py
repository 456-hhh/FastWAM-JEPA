from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (SRC_ROOT, PROJECT_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam_jepa_runtime_guard import configure_runtime_stability

DEFAULT_VJEPA_REPO = "/data1/Johnny/challenge/dd/FastWAM_jepa/external/vjepa2"
DEFAULT_VJEPA_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM_jepa/checkpoints/vjepa2/vitg.pt"
)


class InfiniteSourceIterator:
    def __init__(self, *, name: str, dataset, num_workers: int, seed: int) -> None:
        self.name = str(name)
        self.dataset = dataset
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.epoch = 0
        self.iterator = self._new_iterator()

    def _new_iterator(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=lambda batch: batch[0],
            generator=generator,
        )
        self.epoch += 1
        return iter(loader)

    def next(self) -> dict[str, Any]:
        try:
            sample = next(self.iterator)
        except StopIteration:
            self.iterator = self._new_iterator()
            sample = next(self.iterator)
        if not isinstance(sample, dict):
            raise ValueError(
                f"Dataset {self.name!r} returned {type(sample)}, expected dict."
            )
        sample = dict(sample)
        sample["dataset_name"] = self.name
        return sample


class MixedDemoBatcher:
    def __init__(
        self,
        *,
        sources: dict[str, InfiniteSourceIterator],
        weights: dict[str, float],
        batch_size: int,
        video_size: int,
        seed: int,
        use_proprio: bool,
    ) -> None:
        self.sources = sources
        self.weights = normalize_mix(weights)
        self.batch_size = int(batch_size)
        self.video_size = int(video_size)
        self.rng = random.Random(int(seed))
        self.use_proprio = bool(use_proprio)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")

    def next_batch(self) -> dict[str, Any]:
        names = list(self.weights.keys())
        probs = [self.weights[name] for name in names]
        samples = [
            self.sources[self.rng.choices(names, weights=probs, k=1)[0]].next()
            for _ in range(self.batch_size)
        ]
        return collate_stage1_samples(
            samples, video_size=self.video_size, use_proprio=self.use_proprio
        )

class RoboTwin2Stage1Dataset(torch.utils.data.Dataset):
    """Fast Stage-1 RoboTwin loader for LeRobot-style local datasets.

    This avoids constructing LeRobotDataset, which validates every episode file
    during initialization. Only meta/info.json and meta/episodes.jsonl are read
    up front; parquet/mp4 paths are derived and optionally checked for selected
    sampled episodes inside __getitem__.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        camera_key: str,
        current_frame_count: int,
        future_frame_count: int,
        context_len: int,
        text_dim: int,
        text_embedding_cache_dir: str | None = None,
        max_episodes: int | None = None,
        validate_files: str = "selected",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root).resolve()
        self.camera_key = str(camera_key)
        self.current_frame_count = int(current_frame_count)
        self.future_frame_count = int(future_frame_count)
        self.total_frames = self.current_frame_count + self.future_frame_count
        self.context_len = int(context_len)
        self.text_dim = int(text_dim)
        self.text_embedding_cache_dir = resolve_path(text_embedding_cache_dir) if text_embedding_cache_dir is not None else None
        self.max_episodes = None if max_episodes is None else int(max_episodes)
        if self.max_episodes is not None and self.max_episodes <= 0:
            raise ValueError(f"max_episodes must be positive when set, got {self.max_episodes}.")
        self.validate_files = str(validate_files)
        self.rng = random.Random(int(seed))
        self._warned_missing_context = False

        if self.current_frame_count <= 0 or self.future_frame_count <= 0:
            raise ValueError(
                "current_frame_count and future_frame_count must be positive, "
                f"got {self.current_frame_count}/{self.future_frame_count}."
            )
        if self.context_len <= 0 or self.text_dim <= 0:
            raise ValueError(f"context_len/text_dim must be positive, got {self.context_len}/{self.text_dim}.")
        if self.validate_files not in {"none", "selected"}:
            raise ValueError(f"validate_files must be 'none' or 'selected', got {self.validate_files!r}.")
        for relative in ("meta", "data", "videos"):
            path = self.root / relative
            if not path.exists():
                raise FileNotFoundError(f"RoboTwin fast loader expected {relative}/ under root: {path}")

        info_path = self.root / "meta" / "info.json"
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not info_path.exists():
            raise FileNotFoundError(f"RoboTwin meta/info.json not found: {info_path}")
        if not episodes_path.exists():
            raise FileNotFoundError(f"RoboTwin meta/episodes.jsonl not found: {episodes_path}")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.data_path_template = str(self.info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
        video_template = self.info.get("video_path")
        if video_template is None:
            raise ValueError("RoboTwin fast loader requires info['video_path']; got None.")
        self.video_path_template = str(video_template)
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        if self.chunks_size <= 0:
            raise ValueError(f"info['chunks_size'] must be positive, got {self.chunks_size}.")

        episodes: list[dict[str, Any]] = []
        with episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                episodes.append(json.loads(line))
                if self.max_episodes is not None and len(episodes) >= self.max_episodes:
                    break
        if not episodes:
            raise ValueError(f"No episodes found in {episodes_path}")
        self.episodes = episodes
        self.tasks_by_index = self._load_tasks_by_index()

    def __len__(self) -> int:
        return len(self.episodes)

    def _episode_index(self, episode: dict[str, Any]) -> int:
        if "episode_index" not in episode:
            raise ValueError(f"Episode entry is missing episode_index: {episode}")
        return int(episode["episode_index"])

    def _episode_chunk(self, episode_index: int) -> int:
        return int(episode_index) // self.chunks_size

    def data_file_path(self, episode_index: int) -> Path:
        return self.root / self.data_path_template.format(
            episode_chunk=self._episode_chunk(episode_index),
            episode_index=int(episode_index),
        )

    def video_file_path(self, episode_index: int) -> Path:
        return self.root / self.video_path_template.format(
            episode_chunk=self._episode_chunk(episode_index),
            video_key=self.camera_key,
            episode_index=int(episode_index),
        )

    def _validate_selected_files(self, episode_index: int) -> None:
        if self.validate_files == "none":
            return
        data_path = self.data_file_path(episode_index)
        video_path = self.video_file_path(episode_index)
        if not data_path.exists():
            raise FileNotFoundError(f"Selected RoboTwin episode parquet not found: {data_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"Selected RoboTwin episode video not found: {video_path}")

    def _load_tasks_by_index(self) -> dict[int, str]:
        path = self.root / "meta" / "tasks.jsonl"
        if not path.exists():
            return {}
        mapping: dict[int, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "task_index" in item:
                    task = item.get("task") or item.get("instruction") or item.get("name")
                    if task is not None:
                        mapping[int(item["task_index"])] = str(task)
        return mapping

    def _episode_instruction(self, episode: dict[str, Any]) -> str:
        for key in ("instruction", "task", "language_instruction"):
            if key in episode and episode[key] is not None:
                return str(episode[key])
        tasks = episode.get("tasks")
        if isinstance(tasks, list) and tasks:
            return str(tasks[0])
        task_index = episode.get("task_index")
        if task_index is not None and int(task_index) in self.tasks_by_index:
            return self.tasks_by_index[int(task_index)]
        return "robotwin demonstration"

    def _prompt(self, episode: dict[str, Any]) -> str:
        return f"A video recorded from a robot's point of view executing the following instruction: {self._episode_instruction(episode)}"

    def _context_from_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is not None:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            cache_path = self.text_embedding_cache_dir / f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
            if cache_path.exists():
                payload = torch.load(cache_path, map_location="cpu")
                context = payload["context"]
                mask = payload["mask"].bool()
                if context.ndim != 2 or tuple(context.shape) != (self.context_len, self.text_dim):
                    raise ValueError(
                        f"Cached context shape must be {(self.context_len, self.text_dim)}, "
                        f"got {tuple(context.shape)} in {cache_path}"
                    )
                if mask.ndim != 1 or mask.shape[0] != self.context_len:
                    raise ValueError(f"Cached mask shape must be [{self.context_len}], got {tuple(mask.shape)} in {cache_path}")
                context = context.clone()
                context[~mask] = 0.0
                return context, torch.ones_like(mask, dtype=torch.bool)
            if not self._warned_missing_context:
                print(
                    "RoboTwin fast loader warning: missing text embedding cache for at least one prompt; "
                    "using zero context tokens for those episodes.",
                    flush=True,
                )
                self._warned_missing_context = True
        return torch.zeros(self.context_len, self.text_dim, dtype=torch.float32), torch.ones(self.context_len, dtype=torch.bool)

    @staticmethod
    def _table_to_tensors(path: Path) -> dict[str, torch.Tensor]:
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
        result: dict[str, torch.Tensor] = {}
        for col_name in table.column_names:
            col = table[col_name]
            try:
                arr = col.to_numpy(zero_copy_only=True)
            except Exception:
                raw = col.to_numpy()
                try:
                    import numpy as np

                    arr = np.stack(raw) if raw.dtype == object else raw
                except Exception:
                    continue
            if getattr(arr, "dtype", None) is not None and str(arr.dtype) != "object":
                result[col_name] = torch.as_tensor(arr)
        return result

    @staticmethod
    def _first_existing_tensor(data: dict[str, torch.Tensor], candidates: tuple[str, ...]) -> torch.Tensor | None:
        for key in candidates:
            value = data.get(key)
            if torch.is_tensor(value):
                return value.float()
        return None

    def _load_video(self, path: Path) -> torch.Tensor:
        from torchvision.io import read_video

        video, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
        if video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(f"Decoded RoboTwin video must be [T, 3, H, W], got {tuple(video.shape)} from {path}")
        if video.shape[0] <= 0:
            raise ValueError(f"Decoded RoboTwin video has no frames: {path}")
        video = video.float() / 127.5 - 1.0
        return video.permute(1, 0, 2, 3).contiguous()

    def _select_clip(self, video: torch.Tensor) -> torch.Tensor:
        frames = int(video.shape[1])
        if frames >= self.total_frames:
            start = self.rng.randint(0, frames - self.total_frames)
            return video[:, start : start + self.total_frames]
        pad_count = self.total_frames - frames
        pad = video[:, -1:].expand(-1, pad_count, -1, -1)
        return torch.cat([video, pad], dim=1)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode = self.episodes[int(idx) % len(self.episodes)]
        episode_index = self._episode_index(episode)
        self._validate_selected_files(episode_index)
        data_path = self.data_file_path(episode_index)
        video_path = self.video_file_path(episode_index)
        prompt = self._prompt(episode)
        context, context_mask = self._context_from_prompt(prompt)
        video = self._select_clip(self._load_video(video_path))

        parquet_tensors: dict[str, torch.Tensor] = {}
        if data_path.exists():
            parquet_tensors = self._table_to_tensors(data_path)
        action = self._first_existing_tensor(parquet_tensors, ("action", "action.default"))
        proprio = self._first_existing_tensor(
            parquet_tensors,
            ("observation.state", "proprio", "state", "observation.proprio"),
        )
        return {
            "video": video,
            "current_video": video[:, : self.current_frame_count],
            "future_video": video[:, self.current_frame_count : self.total_frames],
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio,
            "action": action,
            "prompt": prompt,
            "dataset_name": "robotwin",
            "source_name": "robotwin",
            "episode_index": episode_index,
            "data_path": str(data_path),
            "video_path": str(video_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 target-domain JEPA latent predictor pretraining for FastWAM-JEPA-IDM v2. "
            "Only trains JepaFuturePredictor and condition modules; V-JEPA2 and ActionDiT stay frozen/unused."
        )
    )
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--libero-task", default="libero_joint_2cam224_1e-4")
    parser.add_argument("--robotwin-task", default="robotwin_joint_3cam_384_1e-4")
    parser.add_argument("--libero-data-root", default=None)
    parser.add_argument("--robotwin-data-root", default=None)
    parser.add_argument("--robotwin-loader", default="fast", choices=["lerobot", "fast"])
    parser.add_argument("--max-robotwin-episodes", type=int, default=None)
    parser.add_argument("--robotwin-camera-key", default="observation.images.cam_high")
    parser.add_argument("--debug-dataset-only", action="store_true", default=False)
    parser.add_argument("--num-debug-samples", type=int, default=4)
    parser.add_argument("--robotwin-validate-files", default="selected", choices=["none", "selected"])
    parser.add_argument("--dataset-mix", default="libero:0.5,robotwin:0.5")
    parser.add_argument("--vjepa-repo", default=DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-checkpoint", default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--vjepa2ac-checkpoint", default=None)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--vjepa-img-size", type=int, default=256)
    parser.add_argument("--vjepa-input-range", default="-1_1", choices=["-1_1", "0_1"])
    parser.add_argument("--vjepa-tubelet-size", type=int, default=2)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--future-predictor-hidden-dim", type=int, default=1408)
    parser.add_argument("--num-future-tokens", type=int, default=256)
    parser.add_argument("--future-predictor-layers", type=int, default=12)
    parser.add_argument("--future-predictor-heads", type=int, default=8)
    parser.add_argument("--adapter-current-tokens", type=int, default=16)
    parser.add_argument("--adapter-future-tokens", type=int, default=16)
    parser.add_argument("--current-frame-count", type=int, default=2)
    parser.add_argument("--future-frame-count", type=int, default=2)
    parser.add_argument(
        "--video-size",
        type=int,
        default=256,
        help="Common pre-batch H/W used for mixed-source collation.",
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-cos", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--output-dir", default="runs/fastwam_jepa_idm_v2_stage1_predictor_mixed"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-proprio", action="store_true", default=False)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-rank0-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disable-wsl-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--runtime-log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--runtime-log-path", default=None)
    parser.add_argument("--runtime-log-max-mb", type=int, default=100)
    return parser.parse_args()


def resolve_path(path_value: str | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def require_dir(path_value: str | None, *, name: str) -> Path:
    path = resolve_path(path_value)
    if path is None:
        raise ValueError(f"{name} is required for the selected dataset mix.")
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {path}")
    return path


def require_file(path_value: str | None, *, name: str) -> Path:
    path = resolve_path(path_value)
    if path is None:
        raise ValueError(f"{name} is required.")
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")
    return path


def compose_cfg(config_name: str, task: str) -> DictConfig:
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir = (PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=[f"task={task}"])


def parse_dataset_mix(mix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in str(mix).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid --dataset-mix item {item!r}; expected name:weight."
            )
        name, value = item.split(":", 1)
        name = name.strip().lower()
        if name not in {"libero", "robotwin"}:
            raise ValueError(
                f"Unsupported dataset source {name!r}; expected libero or robotwin."
            )
        weight = float(value)
        if weight < 0.0:
            raise ValueError(
                f"Dataset weight must be non-negative, got {name}:{weight}."
            )
        if weight > 0.0:
            result[name] = weight
    if not result:
        raise ValueError(
            "--dataset-mix must contain at least one source with positive weight."
        )
    return normalize_mix(result)


def normalize_mix(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError(f"Dataset weights must sum to > 0, got {weights}.")
    return {
        name: float(value) / total
        for name, value in weights.items()
        if float(value) > 0.0
    }


def resolve_dataset_dirs_from_cfg(
    cfg: DictConfig, *, override_root: str | None, source: str
) -> list[str]:
    if override_root is not None:
        root = require_dir(override_root, name=f"--{source}-data-root")
        if source == "libero":
            candidates = sorted(
                path
                for path in root.iterdir()
                if path.is_dir() and path.name.endswith("_lerobot")
            )
            if candidates:
                return [str(path.resolve()) for path in candidates]
        return [str(root.resolve())]

    dataset_dirs = cfg.data.train.get("dataset_dirs")
    if dataset_dirs is None:
        raise ValueError(f"cfg.data.train.dataset_dirs is required for {source}.")
    resolved: list[str] = []
    for dataset_dir in dataset_dirs:
        path = Path(str(dataset_dir))
        abs_path = (
            path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        )
        if not abs_path.exists():
            raise FileNotFoundError(
                f"{source} dataset directory does not exist: {abs_path}. "
                f"Pass --{source}-data-root to override."
            )
        if not abs_path.is_dir():
            raise FileNotFoundError(
                f"{source} dataset path is not a directory: {abs_path}"
            )
        resolved.append(str(abs_path))
    return resolved


def build_source_dataset(*, source: str, args: argparse.Namespace):
    start_time = time.time()
    print(f"build_source_dataset_start source={source}", flush=True)
    task = args.libero_task if source == "libero" else args.robotwin_task
    cfg = compose_cfg(args.config_name, task)
    override_root = (
        args.libero_data_root if source == "libero" else args.robotwin_data_root
    )
    dataset_dirs = resolve_dataset_dirs_from_cfg(
        cfg, override_root=override_root, source=source
    )
    cfg.data.train.dataset_dirs = dataset_dirs
    print(f"{source}_dataset_dirs={dataset_dirs}", flush=True)

    if source == "robotwin" and args.robotwin_loader == "fast":
        if len(dataset_dirs) != 1:
            raise ValueError(
                f"RoboTwin fast loader expects exactly one dataset root, got {dataset_dirs}."
            )
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        if not isinstance(model_cfg, dict):
            raise ValueError(f"cfg.model must resolve to dict, got {type(model_cfg)}.")
        action_cfg = dict(model_cfg["action_dit_config"])
        dataset = RoboTwin2Stage1Dataset(
            root=dataset_dirs[0],
            camera_key=str(args.robotwin_camera_key),
            current_frame_count=int(args.current_frame_count),
            future_frame_count=int(args.future_frame_count),
            context_len=int(cfg.data.train.get("context_len", 128)),
            text_dim=int(action_cfg["text_dim"]),
            text_embedding_cache_dir=cfg.data.train.get("text_embedding_cache_dir"),
            max_episodes=args.max_robotwin_episodes,
            validate_files=str(args.robotwin_validate_files),
            seed=int(getattr(args, "_rank_seed", args.seed)),
        )
    else:
        dataset = instantiate(cfg.data.train)

    elapsed = time.time() - start_time
    print(
        f"build_source_dataset_done source={source} len={len(dataset)} init_time_sec={elapsed:.3f}",
        flush=True,
    )
    print(f"{source}_dataset={type(dataset).__name__} len={len(dataset)}", flush=True)
    return dataset

def print_debug_dataset_samples(datasets: dict[str, Any], *, num_samples: int) -> None:
    for source, dataset in datasets.items():
        print(f"debug_dataset source={source} len={len(dataset)} type={type(dataset).__name__}", flush=True)
        limit = min(int(num_samples), len(dataset))
        for idx in range(limit):
            sample = dataset[idx]
            if not isinstance(sample, dict):
                raise ValueError(f"Debug sample from {source} is {type(sample)}, expected dict.")
            video = sample.get("video")
            current_video = sample.get("current_video")
            future_video = sample.get("future_video")
            source_name = sample.get("source_name", sample.get("dataset_name", source))
            keys = sorted(str(key) for key in sample.keys())
            video_shape = tuple(video.shape) if torch.is_tensor(video) else None
            current_shape = tuple(current_video.shape) if torch.is_tensor(current_video) else None
            future_shape = tuple(future_video.shape) if torch.is_tensor(future_video) else None
            print(
                " ".join(
                    [
                        f"debug_sample source={source}",
                        f"idx={idx}",
                        f"source_name={source_name}",
                        f"keys={keys}",
                        f"video_shape={video_shape}",
                        f"current_video_shape={current_shape}",
                        f"future_video_shape={future_shape}",
                    ]
                ),
                flush=True,
            )


def resize_video(video: torch.Tensor, *, size: int) -> torch.Tensor:
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"video must be [3, T, H, W], got {tuple(video.shape)}.")
    if int(video.shape[-2]) == int(size) and int(video.shape[-1]) == int(size):
        return video
    x = video.permute(1, 0, 2, 3).float()
    x = F.interpolate(
        x, size=(int(size), int(size)), mode="bilinear", align_corners=False
    )
    return x.to(dtype=video.dtype).permute(1, 0, 2, 3)


def maybe_stack_tensors(values: list[Any], *, key: str) -> torch.Tensor | None:
    if not all(torch.is_tensor(value) for value in values):
        return None
    shapes = [tuple(value.shape) for value in values]
    if len(set(shapes)) != 1:
        print(f"batch_{key}_not_stacked_due_to_shape_mismatch={shapes}", flush=True)
        return None
    return torch.stack(values, dim=0)


def collate_stage1_samples(
    samples: list[dict[str, Any]], *, video_size: int, use_proprio: bool
) -> dict[str, Any]:
    videos = []
    future_videos = []
    contexts = []
    context_masks = []
    actions = []
    proprios = []
    dataset_names = []
    prompts = []
    for sample in samples:
        dataset_name = str(sample.get("dataset_name", "unknown"))
        dataset_names.append(dataset_name)
        prompts.append(sample.get("prompt"))

        video = sample.get("video")
        future_video = sample.get("future_video")
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if not torch.is_tensor(video):
            raise ValueError(f"{dataset_name} sample is missing tensor video.")
        if not torch.is_tensor(context) or not torch.is_tensor(context_mask):
            raise ValueError(
                f"{dataset_name} sample is missing context/context_mask tensors."
            )
        videos.append(resize_video(video, size=video_size))
        if torch.is_tensor(future_video):
            future_videos.append(resize_video(future_video, size=video_size))
        contexts.append(context)
        context_masks.append(context_mask.bool())
        actions.append(sample.get("action"))
        proprios.append(sample.get("proprio"))

    video_batch = torch.stack(videos, dim=0)
    future_video_batch = torch.stack(future_videos, dim=0) if len(future_videos) == len(videos) else None
    context_batch = maybe_stack_tensors(contexts, key="context")
    mask_batch = maybe_stack_tensors(context_masks, key="context_mask")
    if context_batch is None or mask_batch is None:
        raise ValueError(
            "Mixed dataset context/context_mask shapes must match exactly; no silent projection is applied."
        )

    action_batch = maybe_stack_tensors(actions, key="action")
    proprio_batch = maybe_stack_tensors(proprios, key="proprio")
    if use_proprio and proprio_batch is None:
        raise ValueError(
            "--use-proprio requires all mixed sources to have matching proprio tensor shapes."
        )

    return {
        "video": video_batch,
        "future_video": future_video_batch,
        "context": context_batch,
        "context_mask": mask_batch,
        "proprio": proprio_batch if use_proprio else None,
        "action": action_batch,
        "dataset_name": dataset_names,
        "source_name": dataset_names,
        "prompt": prompts,
        "source_counts": dict(Counter(dataset_names)),
    }


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def precision_to_dtype(
    precision: str, device: torch.device
) -> tuple[torch.dtype, torch.dtype | None]:
    if precision == "fp32":
        return torch.float32, None
    if precision == "bf16":
        return (
            torch.bfloat16 if device.type == "cuda" else torch.float32,
            torch.bfloat16 if device.type == "cuda" else None,
        )
    if precision == "fp16":
        return (
            torch.float16 if device.type == "cuda" else torch.float32,
            torch.float16 if device.type == "cuda" else None,
        )
    raise ValueError(f"Unsupported precision: {precision}")

def init_distributed_from_env() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP mode requires CUDA because this script uses NCCL.")
        torch.cuda.set_device(local_rank)
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return True, world_size, rank, local_rank, torch.device(f"cuda:{local_rank}")
    return False, world_size, rank, local_rank, resolve_device()


def configure_rank_printing(*, rank: int, log_rank0_only: bool) -> None:
    if log_rank0_only and rank != 0:
        builtins.print = lambda *args, **kwargs: None


def is_rank0(rank: int) -> bool:
    return int(rank) == 0


def unwrap_ddp(module: torch.nn.Module | None) -> torch.nn.Module | None:
    if module is None:
        return None
    return module.module if isinstance(module, DDP) else module


def reduce_mean_tensor(value: torch.Tensor, *, ddp_enabled: bool, world_size: int) -> torch.Tensor:
    reduced = value.detach().float()
    if ddp_enabled:
        reduced = reduced.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced = reduced / float(world_size)
    return reduced


def reduce_source_counts(
    counts: dict[str, int], *, source_names: list[str], device: torch.device, ddp_enabled: bool
) -> dict[str, int]:
    values = torch.tensor([int(counts.get(name, 0)) for name in source_names], device=device, dtype=torch.long)
    if ddp_enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {name: int(values[idx].item()) for idx, name in enumerate(source_names)}


def move_batch(
    batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("video", "context"):
        value = moved.get(key)
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, dtype=dtype, non_blocking=True)
    mask = moved.get("context_mask")
    if torch.is_tensor(mask):
        moved["context_mask"] = mask.to(
            device=device, dtype=torch.bool, non_blocking=True
        )
    proprio = moved.get("proprio")
    if torch.is_tensor(proprio):
        moved["proprio"] = proprio.to(device=device, dtype=dtype, non_blocking=True)
    return moved


def build_condition(
    *,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor | None,
    proprio_encoder: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if proprio_encoder is None or proprio is None:
        return context, context_mask
    if proprio.ndim == 3:
        proprio = proprio[:, 0, :]
    elif proprio.ndim != 2:
        raise ValueError(
            f"proprio must be [B, D_p] or [B, T, D_p], got {tuple(proprio.shape)}."
        )
    proprio_token = proprio_encoder(proprio).unsqueeze(1)
    proprio_mask = torch.ones(
        (context.shape[0], 1), device=context.device, dtype=torch.bool
    )
    return torch.cat([context, proprio_token], dim=1), torch.cat(
        [context_mask, proprio_mask], dim=1
    )


def build_modules(
    *,
    args: argparse.Namespace,
    text_dim: int,
    proprio_dim: int | None,
    device: torch.device,
    dtype: torch.dtype,
):
    from fastwam.models.vjepa.jepa_fastwam_adapter import JepaToFastWAMAdapter
    from fastwam.models.vjepa.jepa_future_predictor import JepaFuturePredictor
    from fastwam.models.vjepa.vjepa_encoder_wrapper import VJepaEncoderWrapper

    vjepa_encoder = VJepaEncoderWrapper(
        dummy=False,
        model_name=str(args.vjepa_model_name),
        external_repo_path=str(require_dir(args.vjepa_repo, name="--vjepa-repo")),
        checkpoint_path=str(
            require_file(args.vjepa_checkpoint, name="--vjepa-checkpoint")
        ),
        pretrained=False,
        vjepa_dim=int(args.vjepa_dim),
        num_tokens=int(args.num_future_tokens),
        freeze=True,
        normalize_tokens=False,
        img_size=int(args.vjepa_img_size),
        input_range=str(args.vjepa_input_range),
        tubelet_size=int(args.vjepa_tubelet_size),
        frame_encoding_mode="clip_or_repeat",
    ).to(device=device, dtype=dtype)
    vjepa_encoder.eval()
    vjepa_encoder.requires_grad_(False)

    predictor = JepaFuturePredictor(
        vjepa_dim=int(args.vjepa_dim),
        hidden_dim=int(args.future_predictor_hidden_dim),
        num_future_tokens=int(args.num_future_tokens),
        text_dim=int(text_dim),
        num_layers=int(args.future_predictor_layers),
        num_heads=int(args.future_predictor_heads),
    ).to(device=device, dtype=dtype)

    adapter = JepaToFastWAMAdapter(
        vjepa_dim=int(args.vjepa_dim),
        text_dim=int(text_dim),
        num_current_context_tokens=int(args.adapter_current_tokens),
        num_future_context_tokens=int(args.adapter_future_tokens),
    ).to(device=device, dtype=dtype)

    proprio_encoder = None
    if args.use_proprio:
        if proprio_dim is None:
            raise ValueError(
                "--use-proprio was set but the first mixed batch has no stacked proprio tensor."
            )
        proprio_encoder = torch.nn.Linear(int(proprio_dim), int(text_dim)).to(
            device=device, dtype=dtype
        )

    load_stats: dict[str, Any] | None = None
    if args.vjepa2ac_checkpoint is not None:
        ckpt_path = require_file(args.vjepa2ac_checkpoint, name="--vjepa2ac-checkpoint")
        load_stats = predictor.load_vjepa2ac_predictor_weights(ckpt_path)
        if int(load_stats.get("loaded_keys_count", 0)) <= 0:
            raise RuntimeError(
                f"V-JEPA2-AC predictor init loaded no compatible keys: {load_stats}"
            )
    else:
        load_stats = {
            "init_source": "random_init_no_vjepa2ac_checkpoint",
            "loaded_keys_count": 0,
            "skipped_keys_count": 0,
            "shape_mismatch_count": 0,
        }
        print("V-JEPA2-AC predictor init skipped: random init", flush=True)

    return vjepa_encoder, predictor, adapter, proprio_encoder, load_stats


def encode_videos(
    vjepa_encoder: torch.nn.Module,
    video: torch.Tensor,
    *,
    current_frames: int,
    future_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"video must be [B, 3, T, H, W], got {tuple(video.shape)}.")
    required = int(current_frames) + int(future_frames)
    if int(video.shape[2]) < required:
        raise ValueError(f"video has T={video.shape[2]}, required at least {required}.")
    current_video = video[:, :, : int(current_frames)]
    future_video = video[:, :, int(current_frames) : required]
    with torch.no_grad():
        current_tokens = vjepa_encoder(current_video)
        target_future_tokens = vjepa_encoder(future_video).detach()
    return current_tokens, target_future_tokens


def future_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()


def source_counts_to_str(counts: dict[str, int]) -> str:
    return ",".join(f"{name}:{counts[name]}" for name in sorted(counts))

def _extract_predictor_block_index(key: str) -> int | None:
    match = re.search(r"(?:^|\.)predictor_blocks\.(\d+)\.", str(key))
    return None if match is None else int(match.group(1))


def predictor_block_load_summary(load_stats: dict[str, Any]) -> dict[str, Any]:
    loaded_keys = [str(key) for key in load_stats.get("loaded_keys", [])]
    skipped_keys = [str(key) for key in load_stats.get("skipped_keys", [])]
    mismatch_keys = [str(item.get("source_key")) for item in load_stats.get("shape_mismatch_keys", [])]
    detected_blocks = sorted(
        {
            idx
            for key in [*loaded_keys, *skipped_keys, *mismatch_keys]
            for idx in [_extract_predictor_block_index(key)]
            if idx is not None
        }
    )
    loaded_blocks = sorted(
        {
            idx
            for key in loaded_keys
            for idx in [_extract_predictor_block_index(key)]
            if idx is not None
        }
    )
    return {
        "checkpoint_predictor_blocks_detected": len(detected_blocks),
        "checkpoint_predictor_block_ids_detected": detected_blocks,
        "loaded_predictor_blocks": loaded_blocks,
        "loaded_predictor_blocks_count": len(loaded_blocks),
    }


def save_checkpoint(
    *,
    output_dir: Path,
    predictor: torch.nn.Module,
    adapter: torch.nn.Module,
    proprio_encoder: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    load_stats: dict[str, Any] | None,
    loss_dict: dict[str, float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_step_{int(step):06d}.pt"
    payload = {
        "future_predictor": unwrap_ddp(predictor).state_dict(),
        "jepa_adapter": unwrap_ddp(adapter).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": vars(args),
        "vjepa2ac_load_stats": load_stats,
        "vjepa2ac_block_summary": predictor_block_load_summary(load_stats or {}),
        "loss_dict": dict(loss_dict),
    }
    if proprio_encoder is not None:
        payload["proprio_encoder"] = unwrap_ddp(proprio_encoder).state_dict()
    torch.save(payload, path)
    return path


def main() -> None:
    args = parse_args()
    ddp_enabled, world_size, rank, local_rank, device = init_distributed_from_env()
    configure_rank_printing(rank=rank, log_rank0_only=bool(args.log_rank0_only))
    args._rank = rank
    args._world_size = world_size
    args._local_rank = local_rank
    args._rank_seed = int(args.seed) + rank * 100003

    runtime_status = configure_runtime_stability(
        disable_wsl_fallback=args.disable_wsl_fallback,
        log_level=args.runtime_log_level,
        log_path=args.runtime_log_path,
        max_log_mb=args.runtime_log_max_mb,
    )
    print(f"runtime_safe_mode={runtime_status['safe_mode']}", flush=True)
    print(
        f"runtime_disable_wsl_fallback={runtime_status['disable_wsl_fallback']}",
        flush=True,
    )
    print(f"runtime_log_level={runtime_status['log_level']}", flush=True)

    if int(args.steps) <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}.")
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if int(args.grad_accum_steps) <= 0:
        raise ValueError(f"--grad-accum-steps must be positive, got {args.grad_accum_steps}.")
    if float(args.lambda_cos) < 0.0:
        raise ValueError(f"--lambda-cos must be non-negative, got {args.lambda_cos}.")

    torch.manual_seed(int(args._rank_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args._rank_seed))
    param_dtype, autocast_dtype = precision_to_dtype(str(args.precision), device)
    print(f"ddp_enabled={ddp_enabled}", flush=True)
    print(f"world_size={world_size}", flush=True)
    print(f"rank={rank}", flush=True)
    print(f"local_rank={local_rank}", flush=True)
    print(f"per_gpu_batch_size={args.batch_size}", flush=True)
    print(f"grad_accum_steps={args.grad_accum_steps}", flush=True)
    print(f"effective_global_batch_size={int(args.batch_size) * int(world_size) * int(args.grad_accum_steps)}", flush=True)
    output_dir = resolve_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir is required.")

    mix = parse_dataset_mix(args.dataset_mix)
    print(f"dataset_sampling_ratio={mix}", flush=True)

    datasets = {
        source: build_source_dataset(source=source, args=args) for source in mix
    }
    if args.debug_dataset_only:
        print_debug_dataset_samples(datasets, num_samples=int(args.num_debug_samples))
        print("debug_dataset_only_done", flush=True)
        return

    sources = {
        source: InfiniteSourceIterator(
            name=source,
            dataset=dataset,
            num_workers=int(args.num_workers),
            seed=int(getattr(args, "_rank_seed", args.seed)) + idx * 1009,
        )
        for idx, (source, dataset) in enumerate(datasets.items())
    }
    batcher = MixedDemoBatcher(
        sources=sources,
        weights=mix,
        batch_size=int(args.batch_size),
        video_size=int(args.video_size),
        seed=int(getattr(args, "_rank_seed", args.seed)),
        use_proprio=bool(args.use_proprio),
    )

    first_batch = batcher.next_batch()
    text_dim = int(first_batch["context"].shape[-1])
    proprio_tensor = first_batch.get("proprio")
    proprio_dim = (
        int(proprio_tensor.shape[-1]) if torch.is_tensor(proprio_tensor) else None
    )
    print(f"inferred_text_dim={text_dim}", flush=True)
    print(f"inferred_proprio_dim={proprio_dim}", flush=True)
    print(f"first_batch_source_counts={first_batch['source_counts']}", flush=True)
    print(f"first_batch_video_shape={tuple(first_batch['video'].shape)}", flush=True)
    print(
        f"first_batch_context_shape={tuple(first_batch['context'].shape)}", flush=True
    )

    vjepa_encoder, predictor, adapter, proprio_encoder, load_stats = build_modules(
        args=args,
        text_dim=text_dim,
        proprio_dim=proprio_dim,
        device=device,
        dtype=param_dtype,
    )
    predictor.train()
    adapter.eval()
    adapter.requires_grad_(False)
    if proprio_encoder is not None:
        proprio_encoder.train()

    if ddp_enabled:
        predictor = DDP(
            predictor,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )
        if proprio_encoder is not None:
            proprio_encoder = DDP(
                proprio_encoder,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=bool(args.ddp_find_unused_parameters),
            )

    train_modules: list[torch.nn.Module] = [predictor]
    if proprio_encoder is not None:
        train_modules.append(proprio_encoder)
    # The adapter is saved for downstream FastWAM context use. Stage 1 does not
    # consume adapter outputs, so it is frozen here; predictor.condition_projection
    # is trained as the active condition adapter.
    trainable_params = [
        param
        for module in train_modules
        for param in module.parameters()
        if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    block_summary = predictor_block_load_summary(load_stats)
    print(
        "vjepa2ac_init_stats "
        f"model_num_layers={args.future_predictor_layers} "
        f"checkpoint_predictor_blocks_detected={block_summary['checkpoint_predictor_blocks_detected']} "
        f"loaded_predictor_blocks={block_summary['loaded_predictor_blocks']} "
        f"loaded_keys_count={load_stats.get('loaded_keys_count')} "
        f"loaded_params_count={load_stats.get('loaded_params_count', 0)} "
        f"skipped_keys_count={load_stats.get('skipped_keys_count')} "
        f"shape_mismatch_count={load_stats.get('shape_mismatch_count')} "
        f"init_source={load_stats.get('init_source')}",
        flush=True,
    )
    print(
        f"trainable_predictor_params={sum(p.numel() for p in predictor.parameters() if p.requires_grad)}",
        flush=True,
    )
    print(
        f"saved_frozen_adapter_params={sum(p.numel() for p in adapter.parameters())}",
        flush=True,
    )
    print(
        f"trainable_proprio_encoder_params={0 if proprio_encoder is None else sum(p.numel() for p in proprio_encoder.parameters() if p.requires_grad)}",
        flush=True,
    )
    print(
        f"frozen_vjepa_params={sum(p.numel() for p in vjepa_encoder.parameters())}",
        flush=True,
    )

    pending_batch = first_batch
    last_loss_dict: dict[str, float] = {}
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    update_step = 0
    micro_step = 0
    grad_accum_steps = int(args.grad_accum_steps)
    source_names = sorted(mix.keys())
    accum_loss_future_l1 = torch.zeros((), device=device, dtype=torch.float32)
    accum_loss_future_cos = torch.zeros((), device=device, dtype=torch.float32)
    accum_loss_total = torch.zeros((), device=device, dtype=torch.float32)
    accum_source_counts: Counter[str] = Counter()

    while update_step < int(args.steps):
        batch = pending_batch if pending_batch is not None else batcher.next_batch()
        pending_batch = None
        batch = move_batch(batch, device=device, dtype=param_dtype)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            current_tokens, target_future_tokens = encode_videos(
                vjepa_encoder,
                batch["video"],
                current_frames=int(args.current_frame_count),
                future_frames=int(args.future_frame_count),
            )
            condition_context, condition_mask = build_condition(
                context=batch["context"],
                context_mask=batch["context_mask"],
                proprio=batch.get("proprio"),
                proprio_encoder=proprio_encoder,
            )
            out = predictor(
                current_jepa_tokens=current_tokens,
                condition_context=condition_context,
                condition_mask=condition_mask,
            )
            pred_future_tokens = out["pred_future_tokens"]
            if tuple(pred_future_tokens.shape) != tuple(target_future_tokens.shape):
                raise ValueError(
                    "Predicted future token shape mismatch, "
                    f"got {tuple(pred_future_tokens.shape)} vs target {tuple(target_future_tokens.shape)}."
                )
            loss_future_l1 = F.l1_loss(pred_future_tokens, target_future_tokens)
            loss_future_cos = future_cosine_loss(pred_future_tokens, target_future_tokens)
            loss_total = loss_future_l1 + float(args.lambda_cos) * loss_future_cos
            scaled_loss = loss_total / float(grad_accum_steps)

        finite_flag = torch.tensor(
            1 if torch.isfinite(loss_total).item() else 0,
            device=device,
            dtype=torch.int32,
        )
        if ddp_enabled:
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
        if int(finite_flag.item()) != 1:
            raise RuntimeError(
                f"Non-finite loss detected at micro_step={micro_step + 1}, "
                f"next_update_step={update_step + 1}."
            )

        scaled_loss.backward()
        micro_step += 1
        accum_loss_future_l1 += loss_future_l1.detach().float()
        accum_loss_future_cos += loss_future_cos.detach().float()
        accum_loss_total += loss_total.detach().float()
        accum_source_counts.update(batch["source_counts"])

        if micro_step % grad_accum_steps != 0:
            continue

        if float(args.max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1

        local_loss_future_l1 = accum_loss_future_l1 / float(grad_accum_steps)
        local_loss_future_cos = accum_loss_future_cos / float(grad_accum_steps)
        local_loss_total = accum_loss_total / float(grad_accum_steps)
        reduced_loss_future_l1 = reduce_mean_tensor(
            local_loss_future_l1,
            ddp_enabled=ddp_enabled,
            world_size=world_size,
        )
        reduced_loss_future_cos = reduce_mean_tensor(
            local_loss_future_cos,
            ddp_enabled=ddp_enabled,
            world_size=world_size,
        )
        reduced_loss_total = reduce_mean_tensor(
            local_loss_total,
            ddp_enabled=ddp_enabled,
            world_size=world_size,
        )
        global_source_counts = reduce_source_counts(
            dict(accum_source_counts),
            source_names=source_names,
            device=device,
            ddp_enabled=ddp_enabled,
        )

        last_loss_dict = {
            "loss_total": float(reduced_loss_total.item()),
            "loss_future_l1": float(reduced_loss_future_l1.item()),
            "loss_future_cos": float(reduced_loss_future_cos.item()),
        }
        if update_step == 1 or update_step % int(args.log_every) == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            samples = update_step * int(args.batch_size) * world_size * grad_accum_steps
            print(
                "step="
                f"{update_step} loss_total={last_loss_dict['loss_total']:.6f} "
                f"loss_future_l1={last_loss_dict['loss_future_l1']:.6f} "
                f"loss_future_cos={last_loss_dict['loss_future_cos']:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.6e} "
                f"batch_source_counts={global_source_counts} "
                f"samples_per_sec={samples / elapsed:.2f}",
                flush=True,
            )

        if is_rank0(rank) and int(args.save_every) > 0 and update_step % int(args.save_every) == 0:
            path = save_checkpoint(
                output_dir=output_dir,
                step=update_step,
                predictor=predictor,
                adapter=adapter,
                proprio_encoder=proprio_encoder,
                optimizer=optimizer,
                args=args,
                loss_dict=last_loss_dict,
                load_stats=load_stats,
            )
            print(f"saved_checkpoint path={path}", flush=True)

        accum_loss_future_l1.zero_()
        accum_loss_future_cos.zero_()
        accum_loss_total.zero_()
        accum_source_counts.clear()

    if is_rank0(rank):
        final_path = save_checkpoint(
            output_dir=output_dir,
            step=int(args.steps),
            predictor=predictor,
            adapter=adapter,
            proprio_encoder=proprio_encoder,
            optimizer=optimizer,
            args=args,
            loss_dict=last_loss_dict,
            load_stats=load_stats,
        )
        print(f"saved_final_checkpoint path={final_path}", flush=True)

        summary_path = output_dir / "stage1_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "final_checkpoint": str(final_path),
                    "last_loss": last_loss_dict,
                    "dataset_sampling_ratio": mix,
                    "vjepa2ac_load_stats": load_stats,
                    "vjepa2ac_block_summary": predictor_block_load_summary(load_stats),
                    "args": vars(args),
                    "ddp": {
                        "enabled": ddp_enabled,
                        "world_size": world_size,
                        "rank": rank,
                        "local_rank": local_rank,
                        "grad_accum_steps": grad_accum_steps,
                        "effective_global_batch_size": int(args.batch_size) * world_size * grad_accum_steps,
                    },
                },
                f,
                indent=2,
            )
        print(f"saved_summary path={summary_path}", flush=True)

    if ddp_enabled:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
