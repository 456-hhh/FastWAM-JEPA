import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def replan_idx_from_name(name: str):
    m = re.search(r"replan(\d+)", name)
    return int(m.group(1)) if m else -1


def load_rgb_files(rgb_dir: Path):
    files = sorted(
        rgb_dir.glob("*replan*_rgb.png"),
        key=lambda p: replan_idx_from_name(p.name),
    )
    return files


def make_overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45):
    h, w = rgb.shape[:2]

    heat = heat.astype(np.float32)
    heat = heat - heat.min()
    heat = heat / max(heat.max(), 1e-12)

    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heat_arr = np.asarray(heat_img).astype(np.float32) / 255.0

    cmap = plt.get_cmap("jet")
    color_heat = cmap(heat_arr)[..., :3]
    rgb_float = rgb.astype(np.float32) / 255.0

    overlay = (1 - alpha) * rgb_float + alpha * color_heat
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--latent_frame", type=int, default=0)
    parser.add_argument("--max_replans", type=int, default=999)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    root = Path(args.dir)
    rgb_dir = root / "replan_rgb"
    spatial_path = root / "spatial_trajectory_summary" / "replan_spatial_heatmaps.npy"

    out_dir = Path(args.out_dir) if args.out_dir else root / "rgb_attention_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rgb_dir.exists():
        raise RuntimeError(f"Missing RGB dir: {rgb_dir}")

    if not spatial_path.exists():
        raise RuntimeError(f"Missing spatial heatmaps: {spatial_path}")

    rgb_files = load_rgb_files(rgb_dir)
    heatmaps = np.load(spatial_path)  # [R, 3, 7, 14]

    n = min(len(rgb_files), heatmaps.shape[0], args.max_replans)
    rgb_files = rgb_files[:n]
    heatmaps = heatmaps[:n, args.latent_frame]  # [R, 7, 14]

    cols = min(args.cols, n)
    rows = int(np.ceil(n / cols))

    # 1. RGB contact sheet
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            rgb = np.asarray(Image.open(rgb_files[i]).convert("RGB"))
            ax.imshow(rgb)
            ax.set_title(f"replan {i}", fontsize=9)

    fig.suptitle("RGB current observation at each replan", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "replan_rgb_contact_sheet.png", dpi=200)
    plt.close(fig)

    # 2. overlay contact sheet
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            rgb = np.asarray(Image.open(rgb_files[i]).convert("RGB"))
            overlay = make_overlay(rgb, heatmaps[i])
            ax.imshow(overlay)
            ax.set_title(f"replan {i}", fontsize=9)

    fig.suptitle(f"RGB + attention overlay, latent frame {args.latent_frame}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"replan_rgb_attention_overlay_frame{args.latent_frame}.png", dpi=200)
    plt.close(fig)

    # 3. side-by-side RGB / heatmap for every replan
    fig, axes = plt.subplots(n, 2, figsize=(9, max(2.2 * n, 4)))
    if n == 1:
        axes = np.array([axes])

    vmin = heatmaps.min()
    vmax = heatmaps.max()

    for i in range(n):
        rgb = np.asarray(Image.open(rgb_files[i]).convert("RGB"))

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"replan {i} RGB")
        axes[i, 0].axis("off")

        im = axes[i, 1].imshow(heatmaps[i], vmin=vmin, vmax=vmax, aspect="auto")
        axes[i, 1].set_title(f"replan {i} attention frame {args.latent_frame}")
        axes[i, 1].axis("off")

    fig.colorbar(im, ax=axes[:, 1], shrink=0.6, label="attention")
    fig.tight_layout()
    fig.savefig(out_dir / f"replan_rgb_and_heatmap_side_by_side_frame{args.latent_frame}.png", dpi=200)
    plt.close(fig)

    print("saved to:", out_dir)


if __name__ == "__main__":
    main()