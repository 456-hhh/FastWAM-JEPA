from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


LATENT_KEYS = ("q_l", "z_l", "z_task", "z_a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FastWAM-JEPA v3 Stage4 representations.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pca-points", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _pool_and_normalize(value: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value)
    if array.ndim != 3 or tuple(array.shape[1:]) != (4, 1024):
        raise ValueError(f"{name} must be [N,4,1024], got {array.shape}.")
    pooled = array.astype(np.float32, copy=False).mean(axis=1)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    normalized = pooled / np.maximum(norms, 1.0e-12)
    return pooled, normalized.astype(np.float32, copy=False)


def _sample_pairs(count: int, *, max_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("At least two samples are required for pairwise metrics.")
    pair_count = min(int(max_pairs), count * (count - 1))
    first = rng.integers(0, count, size=pair_count, dtype=np.int64)
    second = rng.integers(0, count - 1, size=pair_count, dtype=np.int64)
    second += second >= first
    return first, second


def _effective_rank(pooled: np.ndarray) -> float:
    centered = pooled.astype(np.float64, copy=False) - pooled.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = np.square(singular)
    total = float(variance.sum())
    if total <= 0.0:
        return 0.0
    probability = variance / total
    probability = probability[probability > 0.0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _collapse_metrics(
    pooled: np.ndarray,
    normalized: np.ndarray,
    *,
    rng: np.random.Generator,
) -> dict[str, float]:
    dimension_std = pooled.std(axis=0)
    first, second = _sample_pairs(len(pooled), max_pairs=100000, rng=rng)
    pairwise_cosine = np.sum(normalized[first] * normalized[second], axis=1)
    return {
        "mean_norm": float(np.linalg.norm(pooled, axis=1).mean()),
        "mean_dimension_std": float(dimension_std.mean()),
        "active_dimension_count": float(np.count_nonzero(dimension_std > 1.0e-3)),
        "effective_rank": _effective_rank(pooled),
        "pairwise_cosine_mean": float(pairwise_cosine.mean()),
        "pairwise_cosine_std": float(pairwise_cosine.std()),
        "fraction_pairwise_cosine_gt_0.99": float(np.mean(pairwise_cosine > 0.99)),
    }


def _top_k_indices(similarity: np.ndarray, k: int) -> np.ndarray:
    width = min(int(k), int(similarity.shape[1]))
    return np.argpartition(-similarity, kth=width - 1, axis=1)[:, :width]


def _language_retrieval(
    q_l: np.ndarray,
    z_l: np.ndarray,
    instruction_id: np.ndarray,
) -> dict[str, float]:
    similarity = q_l @ z_l.T
    top1 = similarity.argmax(axis=1)
    top5 = _top_k_indices(similarity, 5)
    labels = np.arange(len(q_l))
    paired = np.sum(q_l * z_l, axis=1)
    _, counts = np.unique(instruction_id, return_counts=True)
    probabilities = counts.astype(np.float64) / float(len(instruction_id))
    return {
        "exact_sample_retrieval_at_1": float(np.mean(top1 == labels)),
        "exact_sample_retrieval_at_5": float(np.mean(np.any(top5 == labels[:, None], axis=1))),
        "same_instruction_retrieval_at_1": float(np.mean(instruction_id[top1] == instruction_id)),
        "same_instruction_retrieval_at_5": float(
            np.mean(np.any(instruction_id[top5] == instruction_id[:, None], axis=1))
        ),
        "paired_q_l_z_l_cosine_mean": float(paired.mean()),
        "paired_q_l_z_l_cosine_std": float(paired.std()),
        "random_same_instruction_at_1": float(np.square(probabilities).sum()),
    }


def _action_metrics(
    z_task: np.ndarray,
    z_a: np.ndarray,
    action: np.ndarray,
    *,
    rng: np.random.Generator,
) -> dict[str, float]:
    similarity = z_task @ z_a.T
    nearest = similarity.argmax(axis=1)
    top5 = _top_k_indices(similarity, 5)
    labels = np.arange(len(z_task))
    paired = np.sum(z_task * z_a, axis=1)

    flat_action = action.astype(np.float32, copy=False).reshape(len(action), -1)
    action_unit = flat_action / np.maximum(
        np.linalg.norm(flat_action, axis=1, keepdims=True),
        1.0e-12,
    )
    random_index = rng.integers(0, len(action) - 1, size=len(action), dtype=np.int64)
    random_index += random_index >= labels

    nearest_l2 = np.linalg.norm(flat_action - flat_action[nearest], axis=1)
    nearest_cosine = np.sum(action_unit * action_unit[nearest], axis=1)
    random_l2 = np.linalg.norm(flat_action - flat_action[random_index], axis=1)
    random_cosine = np.sum(action_unit * action_unit[random_index], axis=1)
    return {
        "exact_action_retrieval_at_1": float(np.mean(nearest == labels)),
        "exact_action_retrieval_at_5": float(np.mean(np.any(top5 == labels[:, None], axis=1))),
        "paired_z_task_z_a_cosine_mean": float(paired.mean()),
        "nearest_neighbor_action_l2": float(nearest_l2.mean()),
        "nearest_neighbor_action_cosine": float(nearest_cosine.mean()),
        "random_neighbor_action_l2": float(random_l2.mean()),
        "random_neighbor_action_cosine": float(random_cosine.mean()),
    }


def _pca_2d(features: np.ndarray) -> np.ndarray:
    if len(features) < 2:
        raise ValueError("PCA requires at least two samples.")
    centered = features.astype(np.float64, copy=False) - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vt[:2].T
    if coordinates.shape[1] == 1:
        coordinates = np.concatenate(
            [coordinates, np.zeros((len(coordinates), 1), dtype=coordinates.dtype)],
            axis=1,
        )
    return coordinates.astype(np.float32)


def _pca_sample_indices(count: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.sort(rng.choice(count, size=int(max_points), replace=False))


def _plot_language(path: Path, coordinates: np.ndarray, instruction_id: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    unique, counts = np.unique(instruction_id, return_counts=True)
    order = np.argsort(-counts)
    top = set(unique[order[:20]].tolist())
    grouped = np.asarray([value if value in top else "other" for value in instruction_id])
    categories = [value for value in unique[order[:20]] if value in set(grouped)]
    if "other" in grouped:
        categories.append("other")
    mapping = {value: index for index, value in enumerate(categories)}
    colors = np.asarray([mapping[value] for value in grouped])

    figure, axis = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=colors,
        cmap="tab20",
        s=10,
        alpha=0.72,
        linewidths=0,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            label=("other" if value == "other" else str(value)[:10]),
            color=plt.cm.tab20(mapping[value] / max(len(categories) - 1, 1)),
        )
        for value in categories
    ]
    axis.legend(handles=handles, title="instruction_id", fontsize=7, loc="best", ncol=2)
    axis.set(title="Stage4 q_l PCA by instruction", xlabel="PC1", ylabel="PC2")
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_action(path: Path, coordinates: np.ndarray, action_norm: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    points = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=action_norm,
        cmap="viridis",
        s=10,
        alpha=0.72,
        linewidths=0,
    )
    figure.colorbar(points, ax=axis, label="action_norm")
    axis.set(title="Stage4 z_task PCA by action norm", xlabel="PC1", ylabel="PC2")
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _metric_rows(
    collapse: dict[str, dict[str, float]],
    language: dict[str, float],
    action: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation, values in collapse.items():
        for metric, value in values.items():
            rows.append(
                {
                    "section": "collapse",
                    "representation": representation,
                    "metric": metric,
                    "value": value,
                }
            )
    for section, values in (("language_retrieval", language), ("action_alignment", action)):
        for metric, value in values.items():
            rows.append(
                {
                    "section": section,
                    "representation": "",
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def _verdict(
    collapse: dict[str, dict[str, float]],
    language: dict[str, float],
    action: dict[str, float],
) -> tuple[str, bool, bool, bool]:
    collapse_ok = all(
        values["active_dimension_count"] >= 8
        and values["effective_rank"] >= 2.0
        and values["fraction_pairwise_cosine_gt_0.99"] < 0.90
        for values in collapse.values()
    )
    baseline = language["random_same_instruction_at_1"]
    language_margin = max(0.02, 0.10 * baseline)
    language_ok = (
        language["same_instruction_retrieval_at_1"] > baseline + language_margin
        and language["same_instruction_retrieval_at_5"] > baseline + language_margin
    )
    action_ok = (
        action["nearest_neighbor_action_l2"] < 0.90 * action["random_neighbor_action_l2"]
        and action["nearest_neighbor_action_cosine"]
        > action["random_neighbor_action_cosine"] + 0.05
    )
    if collapse_ok and language_ok and action_ok:
        verdict = "PASS"
    elif collapse_ok and (language_ok or action_ok):
        verdict = "CONDITIONAL_PASS"
    else:
        verdict = "FAIL"
    return verdict, collapse_ok, language_ok, action_ok


def main() -> None:
    args = parse_args()
    if int(args.max_pca_points) <= 0:
        raise ValueError("--max-pca-points must be positive.")
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    with np.load(input_path, allow_pickle=False) as data:
        required = (*LATENT_KEYS, "action", "action_norm", "instruction_id")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Input NPZ missing required keys: {','.join(missing)}")
        action = np.asarray(data["action"], dtype=np.float32)
        action_norm = np.asarray(data["action_norm"], dtype=np.float32)
        instruction_id = np.asarray(data["instruction_id"]).astype(str)
        instruction_source = (
            str(data["instruction_id_source"].item())
            if "instruction_id_source" in data
            else "unknown"
        )
        pooled: dict[str, np.ndarray] = {}
        normalized: dict[str, np.ndarray] = {}
        for key in LATENT_KEYS:
            pooled[key], normalized[key] = _pool_and_normalize(data[key], name=key)

    sample_count = len(instruction_id)
    if sample_count < 2:
        raise ValueError("At least two samples are required.")
    if action.shape != (sample_count, 32, 7):
        raise ValueError(f"action must be [N,32,7], got {action.shape}.")
    if action_norm.shape != (sample_count,):
        raise ValueError(f"action_norm must be [N], got {action_norm.shape}.")
    if any(len(value) != sample_count for value in pooled.values()):
        raise ValueError("Latent first dimensions do not match instruction_id.")

    collapse = {
        key: _collapse_metrics(
            pooled[key],
            normalized[key],
            rng=np.random.default_rng(int(args.seed) + index),
        )
        for index, key in enumerate(LATENT_KEYS)
    }
    language = _language_retrieval(
        normalized["q_l"],
        normalized["z_l"],
        instruction_id,
    )
    action_metrics = _action_metrics(
        normalized["z_task"],
        normalized["z_a"],
        action,
        rng=np.random.default_rng(int(args.seed)),
    )

    metrics_path = output_dir / "stage4_metrics.csv"
    rows = _metric_rows(collapse, language, action_metrics)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("section", "representation", "metric", "value"),
        )
        writer.writeheader()
        writer.writerows(rows)

    pca_indices = _pca_sample_indices(sample_count, int(args.max_pca_points), rng)
    language_path = output_dir / "stage4_language_pca.png"
    action_path = output_dir / "stage4_action_pca.png"
    _plot_language(
        language_path,
        _pca_2d(normalized["q_l"][pca_indices]),
        instruction_id[pca_indices],
    )
    _plot_action(
        action_path,
        _pca_2d(normalized["z_task"][pca_indices]),
        action_norm[pca_indices],
    )

    unique_ids, counts = np.unique(instruction_id, return_counts=True)
    verdict, collapse_ok, language_ok, action_ok = _verdict(
        collapse,
        language,
        action_metrics,
    )
    summary_path = output_dir / "stage4_summary.md"
    lines = [
        "# Stage4 Representation Summary",
        "",
        "## Samples",
        "",
        f"- Samples: {sample_count}",
        f"- Instruction source: {instruction_source}",
        f"- Instruction categories: {len(unique_ids)}",
        f"- Samples per instruction: {int(counts.min())}-{int(counts.max())}",
        "",
        "## Collapse",
        "",
        "| latent | mean norm | mean dim std | active dims | effective rank | pair cosine mean/std | cosine > 0.99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in LATENT_KEYS:
        value = collapse[key]
        lines.append(
            f"| {key} | {value['mean_norm']:.4f} | {value['mean_dimension_std']:.6f} | "
            f"{int(value['active_dimension_count'])} | {value['effective_rank']:.2f} | "
            f"{value['pairwise_cosine_mean']:.4f}/{value['pairwise_cosine_std']:.4f} | "
            f"{value['fraction_pairwise_cosine_gt_0.99']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## q_l to z_l Language Retrieval",
            "",
            f"- Exact sample @1/@5: {language['exact_sample_retrieval_at_1']:.4f} / {language['exact_sample_retrieval_at_5']:.4f}",
            f"- Same instruction @1/@5: {language['same_instruction_retrieval_at_1']:.4f} / {language['same_instruction_retrieval_at_5']:.4f}",
            f"- Random same-instruction @1: {language['random_same_instruction_at_1']:.4f}",
            f"- Paired q_l/z_l cosine mean/std: {language['paired_q_l_z_l_cosine_mean']:.4f} / {language['paired_q_l_z_l_cosine_std']:.4f}",
            "",
            "## z_task to z_a Action Preservation",
            "",
            f"- Exact action @1/@5: {action_metrics['exact_action_retrieval_at_1']:.4f} / {action_metrics['exact_action_retrieval_at_5']:.4f}",
            f"- Paired z_task/z_a cosine mean: {action_metrics['paired_z_task_z_a_cosine_mean']:.4f}",
            f"- Nearest action L2 vs random: {action_metrics['nearest_neighbor_action_l2']:.4f} vs {action_metrics['random_neighbor_action_l2']:.4f}",
            f"- Nearest action cosine vs random: {action_metrics['nearest_neighbor_action_cosine']:.4f} vs {action_metrics['random_neighbor_action_cosine']:.4f}",
            "",
            "## PCA",
            "",
            f"- Language: {language_path}",
            f"- Action: {action_path}",
            "",
            "## Conclusion",
            "",
            f"**{verdict}**",
            "",
            f"- Collapse check: {'pass' if collapse_ok else 'fail'}",
            f"- Same-instruction retrieval above empirical baseline: {'pass' if language_ok else 'fail'}",
            f"- Retrieved actions improve over random neighbors: {'pass' if action_ok else 'fail'}",
            "- Exact-sample retrieval and PCA appearance are not used alone for the verdict.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"input={input_path}")
    print(f"samples={sample_count} instruction_categories={len(unique_ids)}")
    print(f"metrics={metrics_path}")
    print(f"language_pca={language_path}")
    print(f"action_pca={action_path}")
    print(f"summary={summary_path}")
    print(f"verdict={verdict}")


if __name__ == "__main__":
    main()
