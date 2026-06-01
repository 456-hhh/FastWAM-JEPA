import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_records(attn_dir: Path):
    records_path = attn_dir / "records.jsonl"
    records = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def plot_inventory(records, out_dir: Path):
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "attention_inventory.csv", index=False)

    if len(df) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(df["q_len"], df["k_len"], s=20)
    plt.xlabel("q_len")
    plt.ylabel("k_len")
    plt.title("SDPA calls: q_len vs k_len")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "inventory_qk_scatter.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    df["q_len"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("q_len")
    plt.ylabel("count")
    plt.title("SDPA q_len distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "inventory_q_len_hist.png", dpi=200)
    plt.close()

    # Save likely action-query records.
    likely = df[(df["q_len"] <= 128) & (df["k_len"] > df["q_len"])]
    likely.to_csv(out_dir / "likely_action_attention_records.csv", index=False)


def plot_attention_matrix(matrix_path: Path, out_dir: Path, title: str):
    attn = np.load(matrix_path)

    plt.figure(figsize=(12, 5))
    plt.imshow(attn, aspect="auto")
    plt.colorbar(label="attention")
    plt.xlabel("key token index")
    plt.ylabel("query token index")
    plt.title(title)
    plt.tight_layout()
    save_path = out_dir / f"{matrix_path.stem}_heatmap.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    # key importance: average over query tokens
    key_score = attn.mean(axis=0)

    plt.figure(figsize=(12, 3))
    plt.plot(key_score)
    plt.xlabel("key token index")
    plt.ylabel("mean attention")
    plt.title(title + " | key importance")
    plt.tight_layout()
    plt.savefig(out_dir / f"{matrix_path.stem}_key_importance.png", dpi=200)
    plt.close()

    # query importance: average over key tokens
    query_score = attn.mean(axis=1)

    plt.figure(figsize=(8, 3))
    plt.plot(query_score)
    plt.xlabel("query token index")
    plt.ylabel("mean attention")
    plt.title(title + " | query importance")
    plt.tight_layout()
    plt.savefig(out_dir / f"{matrix_path.stem}_query_importance.png", dpi=200)
    plt.close()


def plot_binned_key_importance(matrix_path: Path, out_dir: Path, bins: int):
    attn = np.load(matrix_path)
    key_score = attn.mean(axis=0)
    n = len(key_score)

    edges = np.linspace(0, n, bins + 1).astype(int)
    values = []
    labels = []
    for i in range(bins):
        l, r = edges[i], edges[i + 1]
        if r <= l:
            values.append(0.0)
        else:
            values.append(float(key_score[l:r].sum()))
        labels.append(f"{l}-{r}")

    plt.figure(figsize=(14, 4))
    plt.bar(np.arange(bins), values)
    plt.xlabel("key-token bins")
    plt.ylabel("attention mass")
    plt.title(f"{matrix_path.stem} | binned key attention mass")
    plt.tight_layout()
    plt.savefig(out_dir / f"{matrix_path.stem}_binned_key_mass.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_dir", type=str, required=True)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--max_plots", type=int, default=50)
    args = parser.parse_args()

    attn_dir = Path(args.attn_dir)
    plot_dir = attn_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(attn_dir)
    plot_inventory(records, plot_dir)

    saved = [r for r in records if r.get("saved_matrix") and r.get("matrix_path")]
    saved = sorted(saved, key=lambda r: (r["q_len"], r["k_len"]))

    for r in saved[: args.max_plots]:
        matrix_path = Path(r["matrix_path"])
        title = (
            f"call={r['call_idx']} "
            f"q={r['q_len']} k={r['k_len']} "
            f"module={r.get('module')}"
        )
        plot_attention_matrix(matrix_path, plot_dir, title)
        plot_binned_key_importance(matrix_path, plot_dir, args.bins)

    print(f"[Plot] saved plots to {plot_dir}")
    print(f"[Plot] saved {len(saved)} attention matrices, plotted {min(len(saved), args.max_plots)}")


if __name__ == "__main__":
    main()