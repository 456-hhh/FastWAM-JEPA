from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LATENTS = ("z_l", "z_v", "q_a_text", "q_a_vl", "z_a")
DEFAULT_COLORS = ("task_id", "task_index", "episode_id", "timestep", "action_norm", "success")
CATEGORICAL_COLORS = {"task_id", "task_index", "episode_id", "success"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize v3 latents with PCA and optional UMAP.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-keys", nargs="+", default=list(DEFAULT_LATENTS))
    parser.add_argument("--color-keys", nargs="+", default=list(DEFAULT_COLORS))
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-umap", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def mean_pool_latent(value: np.ndarray, key: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3:
        array = array.mean(axis=1)
    if array.ndim != 2:
        raise ValueError(f"{key} must be [N, D] or [N, T, D], got {array.shape}.")
    return array.astype(np.float32, copy=False)


def sample_indices(count: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(count, size=max_points, replace=False))


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < 2:
        raise ValueError("PCA requires at least two samples.")
    centered = features.astype(np.float64, copy=False) - features.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = centered @ vt[:2].T
    variance = np.square(singular_values)
    total = variance.sum()
    explained = np.zeros(2, dtype=np.float64)
    if total > 0:
        explained[: min(2, len(variance))] = variance[:2] / total
    if components.shape[1] == 1:
        components = np.concatenate([components, np.zeros((len(components), 1))], axis=1)
    return components.astype(np.float32), explained


def usable_color(value: np.ndarray, count: int) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim == 0 or int(array.shape[0]) != count:
        return None
    if array.ndim > 1:
        if int(np.prod(array.shape[1:])) != 1:
            return None
        array = array.reshape(-1)
    if array.dtype == object:
        try:
            array = array.astype(str)
        except (TypeError, ValueError):
            return None
    return array


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def write_embedding_csv(
    path: Path,
    embedding: np.ndarray,
    original_indices: np.ndarray,
    colors: dict[str, np.ndarray],
) -> None:
    fieldnames = ["sample_index", "x", "y", *colors.keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(len(embedding)):
            row: dict[str, Any] = {
                "sample_index": int(original_indices[row_index]),
                "x": float(embedding[row_index, 0]),
                "y": float(embedding[row_index, 1]),
            }
            for key, value in colors.items():
                item = value[row_index]
                row[key] = item.item() if hasattr(item, "item") else item
            writer.writerow(row)


def plot_embedding(
    path: Path,
    embedding: np.ndarray,
    *,
    title: str,
    color_key: str | None = None,
    color_value: np.ndarray | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    if color_key is None or color_value is None:
        axis.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.7, linewidths=0)
    elif color_key in CATEGORICAL_COLORS or color_value.dtype.kind in ("U", "S", "b"):
        labels, encoded = np.unique(color_value.astype(str), return_inverse=True)
        points = axis.scatter(
            embedding[:, 0], embedding[:, 1], c=encoded, cmap="tab20", s=10, alpha=0.75, linewidths=0
        )
        if len(labels) <= 20:
            colorbar = figure.colorbar(points, ax=axis)
            colorbar.set_ticks(np.arange(len(labels)))
            colorbar.set_ticklabels(labels)
            colorbar.set_label(color_key)
    else:
        numeric = color_value.astype(np.float64)
        points = axis.scatter(
            embedding[:, 0], embedding[:, 1], c=numeric, cmap="viridis", s=10, alpha=0.75, linewidths=0
        )
        figure.colorbar(points, ax=axis, label=color_key)
    axis.set_title(title)
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def render_method(
    *,
    output_dir: Path,
    latent_key: str,
    method: str,
    embedding: np.ndarray,
    indices: np.ndarray,
    colors: dict[str, np.ndarray],
    subtitle: str,
) -> None:
    latent_name = safe_name(latent_key)
    base = f"{latent_name}_{method}"
    csv_path = output_dir / f"{base}.csv"
    write_embedding_csv(csv_path, embedding, indices, colors)
    plot_embedding(
        output_dir / f"{base}.png",
        embedding,
        title=f"{latent_key} {method.upper()} {subtitle}".strip(),
    )
    for color_key, color_value in colors.items():
        plot_embedding(
            output_dir / f"{base}_{safe_name(color_key)}.png",
            embedding,
            title=f"{latent_key} {method.upper()} colored by {color_key}",
            color_key=color_key,
            color_value=color_value,
        )
    print(f"latent={latent_key} method={method} points={len(embedding)} csv={csv_path.name}")


def main() -> None:
    args = parse_args()
    if args.max_points == 0:
        raise ValueError("--max-points must be positive, or negative for no limit.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    umap_module = None
    if args.run_umap:
        try:
            import umap as umap_module
        except ImportError:
            print("skipped_umap: umap-learn is not installed")

    with np.load(Path(args.input).resolve(), allow_pickle=False) as data:
        for latent_key in args.latent_keys:
            if latent_key not in data:
                print(f"skipped_latent={latent_key} reason=missing")
                continue
            features = mean_pool_latent(data[latent_key], latent_key)
            indices = sample_indices(len(features), int(args.max_points), int(args.seed))
            sampled = features[indices]
            finite = np.isfinite(sampled).all(axis=1)
            sampled = sampled[finite]
            indices = indices[finite]
            if len(sampled) < 2:
                print(f"skipped_latent={latent_key} reason=insufficient_finite_points")
                continue

            colors: dict[str, np.ndarray] = {}
            for color_key in args.color_keys:
                if color_key not in data:
                    print(f"skipped_color={color_key} latent={latent_key} reason=missing")
                    continue
                color = usable_color(data[color_key], len(features))
                if color is None:
                    print(f"skipped_color={color_key} latent={latent_key} reason=invalid_shape")
                    continue
                colors[color_key] = color[indices]

            pca_embedding, explained = pca_2d(sampled)
            subtitle = f"explained={explained[0]:.3f},{explained[1]:.3f}"
            render_method(
                output_dir=output_dir,
                latent_key=latent_key,
                method="pca",
                embedding=pca_embedding,
                indices=indices,
                colors=colors,
                subtitle=subtitle,
            )

            if args.run_umap and umap_module is not None:
                if len(sampled) < 3:
                    print(f"skipped_umap latent={latent_key} reason=insufficient_points")
                    continue
                neighbors = min(15, len(sampled) - 1)
                reducer = umap_module.UMAP(
                    n_components=2,
                    n_neighbors=neighbors,
                    random_state=int(args.seed),
                )
                umap_embedding = reducer.fit_transform(sampled).astype(np.float32, copy=False)
                render_method(
                    output_dir=output_dir,
                    latent_key=latent_key,
                    method="umap",
                    embedding=umap_embedding,
                    indices=indices,
                    colors=colors,
                    subtitle="",
                )


if __name__ == "__main__":
    main()
