from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_LATENTS = ("z_l", "q_a_text", "z_a", "z_v", "q_a_vl")
DEFAULT_TARGETS = (
    "action_mean",
    "action_first",
    "action_norm",
    "proprio",
    "object_pos",
    "robot_state",
    "eef_pos",
    "distance_to_goal",
    "task_id",
    "task_index",
    "success",
)
CLASSIFICATION_KEYS = {"task_id", "task_index"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v3 latents with frozen linear or MLP probes.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--latent-keys", nargs="+", default=list(DEFAULT_LATENTS))
    parser.add_argument("--target-keys", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--probe", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def mean_pool_latent(value: np.ndarray, key: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3:
        array = array.mean(axis=1)
    if array.ndim != 2:
        raise ValueError(f"{key} must be [N, D] or [N, T, D], got {array.shape}.")
    return array.astype(np.float32, copy=False)


def build_probe(probe_type: str, input_dim: int, output_dim: int) -> nn.Module:
    if probe_type == "linear":
        return nn.Linear(input_dim, output_dim)
    hidden = min(512, max(128, input_dim // 2))
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, output_dim),
    )


def classification_target(value: np.ndarray, key: str) -> tuple[np.ndarray, int] | None:
    flat = np.asarray(value)
    if flat.ndim > 1:
        if int(np.prod(flat.shape[1:])) != 1:
            return None
        flat = flat.reshape(-1)
    if key == "success":
        try:
            numeric = flat.astype(np.float64)
        except (TypeError, ValueError):
            return None
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0 or not set(np.unique(finite)).issubset({0.0, 1.0}):
            return None
    _, encoded = np.unique(flat.astype(str), return_inverse=True)
    return encoded.astype(np.int64), int(encoded.max()) + 1 if encoded.size else 0


def regression_target(value: np.ndarray) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2:
        return None
    return array


def split_indices(count: int, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("At least two valid samples are required.")
    if not 0.0 < ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    indices = np.random.default_rng(seed).permutation(count)
    train_count = min(count - 1, max(1, int(round(count * ratio))))
    return indices[:train_count], indices[train_count:]


def train_probe(
    features: np.ndarray,
    target: np.ndarray,
    *,
    classification: bool,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int | str]:
    finite_x = np.isfinite(features).all(axis=1)
    if classification:
        valid = finite_x
    else:
        valid = finite_x & np.isfinite(target).all(axis=1)
    features = features[valid]
    target = target[valid]
    if len(features) < 2:
        return {"status": "insufficient_samples", "num_samples": int(len(features))}

    train_idx, test_idx = split_indices(len(features), float(args.train_ratio), int(args.seed))
    x_mean = features[train_idx].mean(axis=0, keepdims=True)
    x_std = features[train_idx].std(axis=0, keepdims=True)
    x_std[x_std < 1.0e-6] = 1.0
    x = (features - x_mean) / x_std

    if classification:
        y_train_np = target[train_idx]
        y_test_np = target[test_idx]
        output_dim = int(num_classes)
        y_mean = y_std = None
    else:
        y_mean = target[train_idx].mean(axis=0, keepdims=True)
        y_std = target[train_idx].std(axis=0, keepdims=True)
        y_std[y_std < 1.0e-6] = 1.0
        normalized_y = (target - y_mean) / y_std
        y_train_np = normalized_y[train_idx]
        y_test_np = target[test_idx]
        output_dim = int(target.shape[1])

    x_train = torch.from_numpy(x[train_idx]).float()
    if classification:
        y_train = torch.from_numpy(y_train_np).long()
    else:
        y_train = torch.from_numpy(y_train_np).float()
    generator = torch.Generator().manual_seed(int(args.seed))
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(int(args.batch_size), len(x_train)),
        shuffle=True,
        generator=generator,
    )

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    model = build_probe(args.probe, int(features.shape[1]), output_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    loss_fn: nn.Module = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    model.train()
    for _ in range(int(args.epochs)):
        for x_batch, y_batch in loader:
            prediction = model(x_batch.to(device))
            loss = loss_fn(prediction, y_batch.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x[test_idx]).float().to(device)).cpu().numpy()
    if classification:
        accuracy = float((prediction.argmax(axis=1) == y_test_np).mean())
        return {
            "status": "ok",
            "num_samples": int(len(features)),
            "target_dim": 1,
            "mse": "",
            "r2": "",
            "accuracy": accuracy,
        }

    assert y_mean is not None and y_std is not None
    prediction = prediction * y_std + y_mean
    residual = float(np.square(prediction - y_test_np).sum())
    total = float(np.square(y_test_np - y_test_np.mean(axis=0, keepdims=True)).sum())
    r2 = float("nan") if total <= 1.0e-12 else 1.0 - residual / total
    return {
        "status": "ok",
        "num_samples": int(len(features)),
        "target_dim": int(target.shape[1]),
        "mse": float(np.square(prediction - y_test_np).mean()),
        "r2": r2,
        "accuracy": "",
    }


def empty_result(status: str, *, num_samples: int = 0, target_dim: int | str = "") -> dict[str, Any]:
    return {
        "status": status,
        "num_samples": num_samples,
        "target_dim": target_dim,
        "mse": "",
        "r2": "",
        "accuracy": "",
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    input_path = Path(args.input).resolve()
    output_path = Path(args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    with np.load(input_path, allow_pickle=False) as data:
        for latent_key in args.latent_keys:
            if latent_key not in data:
                for target_key in args.target_keys:
                    result = empty_result("missing_latent")
                    rows.append({"latent_key": latent_key, "target_key": target_key, "probe_type": args.probe, **result})
                continue
            try:
                features = mean_pool_latent(data[latent_key], latent_key)
            except ValueError:
                for target_key in args.target_keys:
                    result = empty_result("invalid_latent")
                    rows.append({"latent_key": latent_key, "target_key": target_key, "probe_type": args.probe, **result})
                continue

            for target_key in args.target_keys:
                base = {
                    "latent_key": latent_key,
                    "target_key": target_key,
                    "probe_type": args.probe,
                }
                if target_key not in data:
                    rows.append({**base, **empty_result("missing_target")})
                    continue
                if int(data[target_key].shape[0]) != int(features.shape[0]):
                    rows.append({**base, **empty_result("sample_count_mismatch")})
                    continue

                is_classification = target_key in CLASSIFICATION_KEYS or target_key == "success"
                if is_classification:
                    prepared = classification_target(data[target_key], target_key)
                    if prepared is None:
                        rows.append({**base, **empty_result("invalid_classification_target")})
                        continue
                    target, num_classes = prepared
                    if num_classes < 2:
                        rows.append(
                            {**base, **empty_result("single_class_target", num_samples=len(target), target_dim=1)}
                        )
                        continue
                else:
                    target = regression_target(data[target_key])
                    if target is None:
                        rows.append({**base, **empty_result("invalid_regression_target")})
                        continue
                    num_classes = 0

                result = train_probe(
                    features,
                    target,
                    classification=is_classification,
                    num_classes=num_classes,
                    args=args,
                    device=device,
                )
                rows.append({**base, **result})
                print(
                    f"latent={latent_key} target={target_key} probe={args.probe} "
                    f"status={result['status']} samples={result.get('num_samples', 0)}"
                )

    fieldnames = (
        "latent_key",
        "target_key",
        "probe_type",
        "num_samples",
        "target_dim",
        "mse",
        "r2",
        "accuracy",
        "status",
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"output_csv={output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
