"""FIXED R1 headline figure: predicted interference vs. ACTUAL forgetting.

The original plot_1 plotted softmax temperature (constant at 1.0) against
forgetting -- a correlation over a single x value, i.e. meaningless. This is
your project's central claim, so it has to be right.

This version:
  1. For each (seed, transition) recomputes real per-task forgetting from the
     saved accuracy matrix (metrics_FIXED.per_task_forgetting).
  2. Joins each old class's PREDICTED risk (results/risk_scores.csv, which you
     already save) to the forgetting its task actually suffered.
  3. Plots predicted risk (x) vs actual forgetting (y) and reports Pearson r.

Drop this into figures.py in place of plot_1_predicted_vs_forgetting, or run
it standalone after experiments.py has produced results/matrices/ and
results/risk_scores.csv.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks import generate_task_splits
from metrics import per_task_forgetting, pearson_correlation

FIGURES_DIR = "results/figures"


def build_r1_dataframe(risk_csv="results/risk_scores.csv",
                       matrices_dir="results/matrices") -> pd.DataFrame:
    """Join predicted risk to actual forgetting, one row per (class, transition, seed)."""
    risk = pd.read_csv(risk_csv)
    splits = generate_task_splits()
    # class -> task index
    class_to_task = {c: t for t, info in splits.items() for c in info["classes"]}

    rows = []
    # risk rows come from IAR runs; use those matrices for the matching forgetting
    for (mem, seed), grp in risk.groupby(["memory_size", "seed"]):
        mat_path = os.path.join(matrices_dir, f"iar_N{mem}_seed{seed}.npy")
        if not os.path.exists(mat_path):
            continue
        R = np.load(mat_path)
        forget_by_task = per_task_forgetting(R)
        for _, r in grp.iterrows():
            c = int(r["class_id"])
            t = class_to_task.get(c)
            if t is not None and t in forget_by_task:
                rows.append({
                    "class_id": c,
                    "predicted_risk": float(r["predicted_risk"]),
                    "actual_forgetting": forget_by_task[t],
                    "memory_size": mem,
                    "seed": seed,
                })
    return pd.DataFrame(rows)


def plot_r1(risk_csv="results/risk_scores.csv", matrices_dir="results/matrices"):
    df = build_r1_dataframe(risk_csv, matrices_dir)
    if df.empty:
        print("No R1 data -- run experiments.py first.")
        return

    # average over seeds to reduce noise (keep raw points faint behind)
    x_raw, y_raw = df["predicted_risk"].values, df["actual_forgetting"].values
    agg = df.groupby("class_id")[["predicted_risk", "actual_forgetting"]].mean()
    x, y = agg["predicted_risk"].values, agg["actual_forgetting"].values

    r = pearson_correlation(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_raw, y_raw, alpha=0.12, s=14, color="gray", label="per-seed")
    ax.scatter(x, y, alpha=0.85, s=40, color="C0", label="per-class mean")
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), "r--", label=f"fit r={r:.3f}")
    ax.set_xlabel("Predicted interference (prototype similarity, before training)")
    ax.set_ylabel("Actual forgetting (measured after training)")
    ax.set_title(f"R1: Forgetting is predictable before training  (Pearson r = {r:.3f})")
    ax.legend()
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_predicted_vs_forgetting.png"), dpi=150)
    plt.close(fig)
    print(f"Saved fixed R1 figure.  Pearson r = {r:.3f}  over {len(x)} classes")
    return r


if __name__ == "__main__":
    plot_r1()
