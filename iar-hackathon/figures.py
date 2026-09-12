"""Generate 4 figures from results.csv and saved accuracy matrices."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

FIGURES_DIR = "results/figures"
MATRICES_DIR = "results/matrices"


def plot_1_predicted_vs_forgetting(results_csv: str = "results/results.csv"):
    """IAR temperature sensitivity versus actual forgetting."""
    df = pd.read_csv(results_csv)
    df = df[df["method"] == "iar"]

    x = df["tau"].values
    y = df["avg_forgetting"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.7)

    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", label=f"fit: y={z[0]:.4f}x+{z[1]:.4f}")

    corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    ax.set_xlabel("IAR softmax temperature")
    ax.set_ylabel("Actual Forgetting")
    ax.set_title(f"IAR temperature sensitivity (r={corr:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_predicted_vs_forgetting.png"), dpi=150)
    plt.close(fig)
    print("Saved fig1")


def plot_2_avg_forgetting(results_csv: str = "results/results.csv"):
    """Average forgetting per method (bar chart)."""
    df = pd.read_csv(results_csv)
    summary = df.groupby("method")["avg_forgetting"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(summary["method"], summary["avg_forgetting"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Average Forgetting")
    ax.set_title("Average Forgetting per Method")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_avg_forgetting.png"), dpi=150)
    plt.close(fig)
    print("Saved fig2")


def plot_3_accuracy_vs_budget(results_csv: str = "results/results.csv"):
    """Final accuracy vs memory budget (line, N = 20/50/100)."""
    df = pd.read_csv(results_csv)

    fig, ax = plt.subplots(figsize=(8, 6))
    for method in df["method"].unique():
        sub = (df[df["method"] == method]
               .groupby("memory_size", as_index=False)["final_accuracy"].mean()
               .sort_values("memory_size"))
        ax.plot(sub["memory_size"], sub["final_accuracy"], marker="o", label=method)

    ax.set_xlabel("Memory Budget (N)")
    ax.set_ylabel("Final Accuracy (mean)")
    ax.set_title("Final Accuracy vs Memory Budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_accuracy_vs_budget.png"), dpi=150)
    plt.close(fig)
    print("Saved fig3")


def plot_4_accuracy_matrices(methods: list[str] = ["naive", "iar"]):
    """Accuracy-matrix heatmaps, side by side."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MATRICES_DIR, exist_ok=True)

    matrices = {}
    for method in methods:
        paths = sorted(Path(MATRICES_DIR).glob(f"{method}_*.npy"))
        if paths:
            mats = [np.load(p) for p in paths]
            matrices[method] = np.mean(mats, axis=0)

    if not matrices:
        print("No matrices found for plotting.")
        return

    fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 6))
    if len(matrices) == 1:
        axes = [axes]

    for ax, (name, mat) in zip(axes, matrices.items()):
        im = ax.imshow(mat, cmap="viridis", interpolation="nearest")
        ax.set_title(f"{name}")
        ax.set_xlabel("Task j (evaluated)")
        ax.set_ylabel("Task i (trained through)")
        plt.colorbar(im, ax=ax)

    fig.suptitle("Accuracy Matrix Heatmaps", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_accuracy_matrices.png"), dpi=150)
    plt.close(fig)
    print("Saved fig4")


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MATRICES_DIR, exist_ok=True)
    results_csv = "results/results.csv"
    if not os.path.exists(results_csv):
        print("No results.csv found. Run experiments first.")
    else:
        # FIXED R1: the original plot_1 plotted a constant (tau) against
        # forgetting. Use the corrected version that joins predicted risk to
        # real per-task forgetting. See figure_R1_FIXED.py.
        try:
            from figure_R1_FIXED import plot_r1
            plot_r1()
        except Exception as e:
            print("Fixed R1 figure failed, falling back to old plot_1:", e)
            plot_1_predicted_vs_forgetting(results_csv)
        plot_2_avg_forgetting(results_csv)
        plot_3_accuracy_vs_budget(results_csv)
        plot_4_accuracy_matrices()
