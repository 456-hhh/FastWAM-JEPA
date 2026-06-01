import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def replan_idx(p: Path):
    m = re.search(r"replan(\d+)", p.name)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    root = Path(args.dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / "trajectory_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    sem_dirs = sorted(root.glob("task*_replan*/semantic_plots"), key=lambda p: replan_idx(p.parent))

    rows = []
    frame_rows = []
    local_rows = []

    for sem in sem_dirs:
        idx = replan_idx(sem.parent)

        rec_path = sem / "semantic_attention_records.csv"
        frame_path = sem / "action_to_frame_by_call.npy"
        local_path = sem / "action_to_action_local_window_mass.csv"

        if not rec_path.exists() or not frame_path.exists():
            continue

        rec = pd.read_csv(rec_path)
        frame = np.load(frame_path)  # [calls, 3]
        frame_mean = frame.mean(axis=0)

        rows.append({
            "replan_idx": idx,
            "video_mass": rec["video_mass"].mean(),
            "action_mass": rec["action_mass"].mean(),
            "frame0_current_mass": frame_mean[0],
            "frame1_future_mass": frame_mean[1],
            "frame2_future_mass": frame_mean[2],
            "future_mass": frame_mean[1:].sum(),
        })

        frame_rows.append(frame_mean)

        if local_path.exists():
            local = pd.read_csv(local_path)
            local["replan_idx"] = idx
            local_rows.append(local)

    df = pd.DataFrame(rows).sort_values("replan_idx")
    df.to_csv(out_dir / "replan_attention_summary.csv", index=False)

    frame_arr = np.stack(frame_rows, axis=0)
    np.save(out_dir / "replan_action_to_frame.npy", frame_arr)

    plt.figure(figsize=(8, 5))
    plt.imshow(frame_arr, aspect="auto")
    plt.colorbar(label="attention mass")
    plt.xlabel("video latent frame")
    plt.ylabel("replan index")
    plt.title("Action → video latent frame over replans")
    plt.tight_layout()
    plt.savefig(out_dir / "replan_action_to_frame_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df["replan_idx"], df["frame0_current_mass"], label="frame0/current")
    plt.plot(df["replan_idx"], df["future_mass"], label="frame1+frame2/future")
    plt.xlabel("replan index")
    plt.ylabel("attention mass")
    plt.title("Current vs future latent attention over episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "replan_current_vs_future_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df["replan_idx"], df["video_mass"], label="video tokens")
    plt.plot(df["replan_idx"], df["action_mass"], label="action tokens")
    plt.xlabel("replan index")
    plt.ylabel("attention mass")
    plt.title("Video/action attention mass over episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "replan_video_vs_action_curve.png", dpi=200)
    plt.close()

    if local_rows:
        local_df = pd.concat(local_rows, ignore_index=True)
        local_df.to_csv(out_dir / "replan_action_local_window_mass.csv", index=False)

        pivot = local_df.pivot(index="replan_idx", columns="window", values="mass")

        plt.figure(figsize=(8, 4))
        for w in [1, 2, 4, 8]:
            if w in pivot.columns:
                plt.plot(pivot.index, pivot[w], label=f"±{w}")
        plt.xlabel("replan index")
        plt.ylabel("local action attention mass")
        plt.title("Action local attention over replans")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "replan_action_local_window_curve.png", dpi=200)
        plt.close()

    print("saved to", out_dir)


if __name__ == "__main__":
    main()