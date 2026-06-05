from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
LIBERO_EXP_ROOT = PROJECT_ROOT / "experiments" / "libero"
for path in (SRC_ROOT, PROJECT_ROOT, LIBERO_EXP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_VJEPA_REPO = "/data1/Johnny/challenge/dd/FastWAM_jepa/external/vjepa2"
DEFAULT_VJEPA_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM_jepa/checkpoints/vjepa2/vitg.pt"
)
DEFAULT_FASTWAM_BASE_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM/runs/libero_joint_2cam224_1e-4/"
    "libero_joint_4gpu_20k_20260524_171732/checkpoints/weights/step_020000.pt"
)
DEFAULT_JEPA_CHECKPOINT = (
    "/data1/Johnny/challenge/dd/FastWAM_jepa/runs/fastwam_jepa_joint_v1_2k/"
    "checkpoints/checkpoint_step_002000.pt"
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class FrameHistory:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = int(frame_count)
        self.frames: list[torch.Tensor] = []

    def append(self, frame: torch.Tensor) -> None:
        if frame.ndim != 4 or frame.shape[0] != 1 or frame.shape[1] != 3:
            raise ValueError(f"Frame must be [1,3,H,W], got {tuple(frame.shape)}.")
        self.frames.append(frame.detach())
        self.frames = self.frames[-self.frame_count :]

    def as_video(self) -> torch.Tensor:
        if not self.frames:
            raise ValueError("FrameHistory is empty.")
        frames = list(self.frames)
        while len(frames) < self.frame_count:
            frames.insert(0, frames[0].clone())
        return torch.stack(frames[-self.frame_count :], dim=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FastWAM-JEPA-Joint v1 on LIBERO.")
    parser.add_argument("--checkpoint", default=DEFAULT_JEPA_CHECKPOINT)
    parser.add_argument("--fastwam-base-checkpoint", default=DEFAULT_FASTWAM_BASE_CHECKPOINT)
    parser.add_argument("--vjepa-checkpoint", default=DEFAULT_VJEPA_CHECKPOINT)
    parser.add_argument("--vjepa-repo", default=DEFAULT_VJEPA_REPO)
    parser.add_argument("--output-dir", default="evaluate_results/fastwam_jepa_joint")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", default="0", help="Task id or `all`.")
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--config-task", default="libero_joint_2cam224_1e-4")
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument("--save-rollout-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--lambda-future", type=float, default=0.1)
    parser.add_argument("--current-frame-count", type=int, default=2)
    parser.add_argument("--future-frame-count", type=int, default=2)
    parser.add_argument("--num-future-tokens", type=int, default=256)
    parser.add_argument("--vjepa-dim", type=int, default=1408)
    parser.add_argument("--vjepa-model-name", default="vjepa2_vit_giant")
    parser.add_argument("--binarize-gripper", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path_value: str | None, *, base: Path = PROJECT_ROOT) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(os.path.expanduser(os.path.expandvars(str(path_value))))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def compose_cfg(args: argparse.Namespace) -> DictConfig:
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir = (PROJECT_ROOT / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=args.config_name, overrides=[f"task={args.config_task}"])
    cfg.EVALUATION.task_suite_name = args.task_suite
    cfg.EVALUATION.num_trials = int(args.num_episodes)
    cfg.EVALUATION.replan_steps = int(args.replan_steps)
    cfg.EVALUATION.num_steps_wait = int(args.num_steps_wait)
    cfg.EVALUATION.num_inference_steps = int(args.num_inference_steps)
    cfg.EVALUATION.sigma_shift = args.sigma_shift
    cfg.EVALUATION.rand_device = args.rand_device
    cfg.EVALUATION.output_dir = str(resolve_path(args.output_dir))
    if args.dataset_stats_path is not None:
        cfg.EVALUATION.dataset_stats_path = str(resolve_path(args.dataset_stats_path))
    cfg.seed = int(args.seed)
    return cfg


def setup_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def extract_action_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload["mot"] if "mot" in payload and isinstance(payload["mot"], dict) else payload
    prefixes = (
        "mixtures.action.",
        "dit.mixtures.action.",
        "mot.mixtures.action.",
        "module.mixtures.action.",
        "module.dit.mixtures.action.",
        "module.mot.mixtures.action.",
        "action_expert.",
        "model.action_expert.",
        "module.action_expert.",
    )
    for prefix in prefixes:
        action_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if action_state:
            return action_state
    raise ValueError("Could not find ActionDiT weights in FastWAM base checkpoint.")


def build_action_expert(
    *,
    action_cfg: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from fastwam.models.wan22.action_dit import ActionDiT

    action_expert = ActionDiT(**action_cfg)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Base checkpoint payload must be dict, got {type(payload)}.")
    action_expert.load_state_dict(extract_action_state_dict(payload), strict=True)
    return action_expert.to(device=device, dtype=dtype)


def build_model(cfg: DictConfig, args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype):
    from fastwam.models.vjepa import VJepaEncoderWrapper
    from fastwam.models.wan22.fastwam_jepa_joint import FastWAMJEPAJoint

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):
        raise ValueError(f"cfg.model must resolve to dict, got {type(model_cfg)}.")
    action_cfg = dict(model_cfg["action_dit_config"])

    action_expert = build_action_expert(
        action_cfg=action_cfg,
        checkpoint_path=resolve_path(args.fastwam_base_checkpoint),
        device=device,
        dtype=dtype,
    )
    vjepa_encoder = VJepaEncoderWrapper(
        dummy=False,
        model_name=args.vjepa_model_name,
        external_repo_path=str(resolve_path(args.vjepa_repo)),
        checkpoint_path=str(resolve_path(args.vjepa_checkpoint)),
        pretrained=False,
        vjepa_dim=int(args.vjepa_dim),
        num_tokens=int(args.num_future_tokens),
        freeze=True,
        normalize_tokens=True,
    ).to(device=device, dtype=dtype)

    action_scheduler_cfg = model_cfg.get("action_scheduler", {})
    proprio_dim = model_cfg.get("proprio_dim")
    model = FastWAMJEPAJoint(
        action_expert=action_expert,
        vjepa_encoder=vjepa_encoder,
        action_dim=int(action_cfg["action_dim"]),
        hidden_dim=int(action_cfg["hidden_dim"]),
        vjepa_dim=int(args.vjepa_dim),
        num_future_tokens=int(args.num_future_tokens),
        text_dim=int(action_cfg["text_dim"]),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=None,
        torch_dtype=dtype,
        action_train_shift=float(action_scheduler_cfg.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler_cfg.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(action_scheduler_cfg.get("num_train_timesteps", 1000)),
        lambda_future=float(args.lambda_future),
        current_frame_count=int(args.current_frame_count),
        future_frame_count=int(args.future_frame_count),
    ).to(device=device, dtype=dtype)

    load_jepa_checkpoint(model, resolve_path(args.checkpoint), device=device)
    model.eval()
    return model


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key[len("module.") :] if isinstance(key, str) and key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_jepa_checkpoint(model: torch.nn.Module, checkpoint_path: Path, *, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"JEPA checkpoint must be dict, got {type(payload)}.")

    state = None
    for key in ("model", "model_state_dict", "state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            state = value
            break
    if state is None and all(torch.is_tensor(value) for value in payload.values()):
        state = payload

    if state is not None:
        missing, unexpected = model.load_state_dict(strip_module_prefix(state), strict=False)
        logging.info("Loaded JEPA checkpoint: %s", checkpoint_path)
        logging.info("Missing keys: %s", list(missing))
        logging.info("Unexpected keys: %s", list(unexpected))
        print(f"Loaded JEPA checkpoint: {checkpoint_path}")
        print(f"Missing keys: {list(missing)}")
        print(f"Unexpected keys: {list(unexpected)}")
        return

    if "joint_predictor" in payload:
        missing, unexpected = model.joint_predictor.load_state_dict(payload["joint_predictor"], strict=False)
        logging.info("Loaded joint_predictor keys from %s", checkpoint_path)
        logging.info("joint_predictor missing=%s unexpected=%s", list(missing), list(unexpected))
        if payload.get("proprio_encoder") is not None and model.proprio_encoder is not None:
            model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=False)
        return

    raise ValueError(
        "Could not find model state in JEPA checkpoint. Expected one of "
        "`model`, `model_state_dict`, `state_dict`, or `joint_predictor`."
    )


def resolve_dataset_stats_path(cfg: DictConfig, args: argparse.Namespace) -> Path:
    candidates: list[Path] = []
    explicit = resolve_path(args.dataset_stats_path) if args.dataset_stats_path else None
    if explicit is not None:
        candidates.append(explicit)
    cfg_explicit = cfg.EVALUATION.get("dataset_stats_path")
    if cfg_explicit is not None:
        candidates.append(resolve_path(str(cfg_explicit)))
    for ckpt_arg in (args.checkpoint, args.fastwam_base_checkpoint):
        ckpt = resolve_path(ckpt_arg)
        if ckpt is not None:
            for parent in list(ckpt.parents)[:5]:
                candidates.append(parent / "dataset_stats.json")
    seen: set[Path] = set()
    for path in candidates:
        if path is None:
            continue
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Pass --dataset-stats-path explicitly."
    )


def load_cached_text_context(prompt: str, cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = cfg.data.train.get("text_embedding_cache_dir")
    if cache_dir is None:
        raise ValueError("cfg.data.train.text_embedding_cache_dir is required.")
    cache_dir = resolve_path(str(cache_dir))
    context_len = int(cfg.data.train.get("context_len", 128))
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{hashed}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing cached text context: {cache_path}. Run scripts/precompute_text_embeds.py first."
        )
    payload = torch.load(cache_path, map_location="cpu")
    context = payload["context"]
    context_mask = payload["mask"].bool()
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask)
    return context, context_mask


def get_task_ids(task_suite, task_id_arg: str) -> list[int]:
    if str(task_id_arg).lower() == "all":
        return list(range(int(task_suite.n_tasks)))
    return [int(task_id_arg)]


def get_max_steps(task_suite_name: str, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_episode(
    *,
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor,
    cfg: DictConfig,
    args: argparse.Namespace,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_horizon: int,
    input_w: int,
    input_h: int,
    episode_idx: int,
    video_dir: Path,
) -> dict[str, Any]:
    from experiments.libero.eval_libero_single import _denormalize_action, _obs_to_model_input
    from experiments.libero.libero_utils import (
        get_libero_dummy_action,
        invert_gripper_action,
        save_rollout_video,
    )

    max_steps = get_max_steps(args.task_suite, args.max_steps)
    if bool(args.dry_run):
        max_steps = min(max_steps, 10)
    frame_history = FrameHistory(frame_count=int(args.current_frame_count))

    env.reset()
    obs = env.set_init_state(initial_state)
    replay_images = []
    pending_actions: list[list[float]] = []
    episode_return = 0.0
    done = False
    length = 0

    for t in range(max_steps + int(args.num_steps_wait)):
        image, proprio, imgs = _obs_to_model_input(
            obs,
            cfg=cfg,
            processor=processor,
            width=input_w,
            height=input_h,
            device=str(model._runtime_device()),
            dtype=model.torch_dtype,
        )
        frame_history.append(image)
        if bool(args.save_rollout_video):
            replay_images.append(imgs.copy())

        if t < int(args.num_steps_wait):
            action_to_env = get_libero_dummy_action()
        else:
            if not pending_actions:
                current_video = frame_history.as_video()
                pred = model.predict_action(
                    current_video=current_video,
                    action_horizon=action_horizon,
                    context=context,
                    context_mask=context_mask,
                    proprio=proprio,
                    num_inference_steps=int(args.num_inference_steps),
                    sigma_shift=args.sigma_shift,
                    seed=None if args.seed is None else int(args.seed) + int(episode_idx),
                    rand_device=str(args.rand_device),
                )
                action = _denormalize_action(pred["action"], processor)[0]
                action[..., -1] = action[..., -1] * 2 - 1
                action = invert_gripper_action(action)
                if bool(args.binarize_gripper):
                    action[..., -1] = np.sign(action[..., -1])
                pending_actions = action[: int(args.replan_steps)].tolist()

            action_to_env = pending_actions.pop(0)

        obs, reward, done, _ = env.step(action_to_env)
        episode_return += float(np.asarray(reward).reshape(-1)[0])
        length += 1
        if done:
            break

    video_path = None
    if bool(args.save_rollout_video):
        video_path = save_rollout_video(
            video_dir,
            replay_images,
            f"episode{episode_idx}",
            success=bool(done),
            task_description=task_description,
        )

    return {
        "episode": int(episode_idx),
        "success": bool(done),
        "return": float(episode_return),
        "length": int(length),
        "video_path": video_path,
    }


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    assert output_dir is not None
    log_path = setup_logging(output_dir)

    from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.utils.pytorch_utils import set_global_seed
    from libero.libero import benchmark

    set_global_seed(int(args.seed), get_worker_init_fn=False)
    cfg = compose_cfg(args)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = resolve_dtype(args.dtype)

    logging.info("Building FastWAMJEPAJoint eval model")
    model = build_model(cfg, args, device=device, dtype=dtype).eval()

    stats_path = resolve_dataset_stats_path(cfg, args)
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", stats_path)

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    action_horizon = int(args.action_horizon) if args.action_horizon else int(cfg.data.train.num_frames) - 1

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[str(args.task_suite)]()
    task_ids = get_task_ids(task_suite, str(args.task_id))
    if bool(args.dry_run):
        args.num_episodes = 1

    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    all_episode_results = []

    start_time = time.time()
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = list(task_suite.get_task_init_states(task_id))
        while len(initial_states) < int(args.num_episodes):
            initial_states.extend(initial_states[: int(args.num_episodes) - len(initial_states)])
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, int(args.seed))
        prompt = DEFAULT_PROMPT.format(task=task_description)
        context, context_mask = load_cached_text_context(prompt, cfg)

        logging.info("Evaluating task_id=%s description=%s", task_id, task_description)
        task_results = []
        for episode_idx in range(int(args.num_episodes)):
            result = run_episode(
                env=env,
                initial_state=initial_states[episode_idx],
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                args=args,
                context=context,
                context_mask=context_mask,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                episode_idx=episode_idx,
                video_dir=video_dir,
            )
            result["task_id"] = int(task_id)
            result["task_description"] = task_description
            task_results.append(result)
            all_episode_results.append(result)
            logging.info(
                "episode task=%s idx=%s success=%s return=%.4f length=%s",
                task_id,
                episode_idx,
                result["success"],
                result["return"],
                result["length"],
            )
            print(
                f"task={task_id} episode={episode_idx} "
                f"success={result['success']} return={result['return']:.4f} "
                f"length={result['length']}"
            )

        if hasattr(env, "close"):
            env.close()

    success_count = sum(1 for item in all_episode_results if item["success"])
    total = len(all_episode_results)
    avg_len = float(np.mean([item["length"] for item in all_episode_results])) if total else 0.0
    results = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "fastwam_base_checkpoint": str(resolve_path(args.fastwam_base_checkpoint)),
        "task_suite": str(args.task_suite),
        "task_id": str(args.task_id),
        "num_episodes": int(args.num_episodes),
        "successes": int(success_count),
        "total_episodes": int(total),
        "success_rate": float(success_count / max(total, 1)),
        "avg_episode_length": avg_len,
        "duration_sec": float(time.time() - start_time),
        "episodes": all_episode_results,
        "log_path": str(log_path),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logging.info("Saved results to %s", results_path)
    print(f"success_rate={results['success_rate']:.4f} avg_episode_length={avg_len:.2f}")
    print(f"Saved results to {results_path}")
    print(f"Saved log to {log_path}")


if __name__ == "__main__":
    main()
