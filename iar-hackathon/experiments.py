"""Run all experiments and write results.csv.

Conditions:
- Naive sequential fine-tuning (no buffer)
- Joint training (all data at once)
- Random ER (buffer capacity N)
- Herding ER (buffer capacity N)
- IAR (buffer capacity N)

Memory sweep: N = 20, 50, 100
Seeds: 42, 43, 44, 45, 46
Ablation 2x2:
- Uniform allocation + random selection
- Uniform allocation + herding selection
- Risk-based allocation + random selection
- Risk-based allocation + herding selection
"""
import numpy as np
import pandas as pd
import torch
from seed import set_seed
from data import load_cifar100
from tasks import generate_task_splits
from continual import run_naive, run_joint_training, run_continual_learning
from metrics import summarize, pearson_correlation
from buffer import RandomBuffer, HerdingBuffer, IARBuffer


def run_single_experiment(embeddings: np.ndarray,
                          labels: np.ndarray,
                          task_splits: dict,
                          method: str,
                          memory_size: int | None = None,
                          seed: int = 42,
                          tau: float = 1.0,
                          epochs: int = 20,
                          device: str | None = None) -> dict:
    """Run one experiment condition."""
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_tasks = len(task_splits)

    if method == "naive":
        R, risk_rows = run_naive(embeddings, labels, task_splits, epochs=epochs, device=device, seed=seed)
    elif method == "joint":
        R, risk_rows = run_joint_training(embeddings, labels, task_splits, epochs=epochs, device=device, seed=seed)
    elif method in {"random", "herding", "iar", "iar_random"}:
        if memory_size is None:
            memory_size = 100
        # Initialize buffer
        if method == "random":
            buffer = RandomBuffer(embeddings, labels, capacity=memory_size)
        elif method == "herding":
            buffer = HerdingBuffer(embeddings, labels, capacity=memory_size)
        else:  # IAR variants
            buffer = IARBuffer(embeddings, labels, capacity=memory_size, temp=tau)
        R, risk_rows = run_continual_learning(
            embeddings, labels, task_splits,
            epochs=epochs,
            device=device,
            replay_buffer=buffer,
            method=method,
            tau=tau,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    summary = summarize(R)
    return {
        "R": R,
        "final_accuracy": summary["final_avg_accuracy"],
        "avg_forgetting": summary["avg_forgetting"],
        "method": method,
        "memory_size": memory_size,
        "seed": seed,
        "tau": tau,
        "risk_rows": risk_rows,
    }


def run_all_experiments(embeddings: np.ndarray,
                        labels: np.ndarray,
                        task_splits: dict,
                        memory_sizes: list[int] = [20, 50, 100],
                        seeds: list[int] = [42, 43, 44, 45, 46],
                        methods: list[str] | None = None,
                        output_path: str = "results.csv") -> pd.DataFrame:
    """Run all conditions, memory sizes, and seeds. Write a tidy results.csv."""
    if methods is None:
        methods = ["naive", "joint", "random", "herding", "iar"]

    rows, risk_records = [], []
    for method in methods:
        for memory_size in memory_sizes:
            for seed in seeds:
                result = run_single_experiment(
                    embeddings=embeddings,
                    labels=labels,
                    task_splits=task_splits,
                    method=method,
                    memory_size=memory_size,
                    seed=seed,
                    epochs=20,
                )
                rows.append({
                    "method": method,
                    "memory_size": memory_size,
                    "seed": seed,
                    "final_accuracy": result["final_accuracy"],
                    "avg_forgetting": result["avg_forgetting"],
                    "tau": result["tau"],
                })
                np.save(f"results/matrices/{method}_N{memory_size}_seed{seed}.npy", result["R"])
                for item in result["risk_rows"]:
                    risk_records.append({**item, "method": method, "memory_size": memory_size, "seed": seed})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    if risk_records:
        pd.DataFrame(risk_records).to_csv("results/risk_scores.csv", index=False)
    return df


def run_ablation_study(embeddings: np.ndarray,
                       labels: np.ndarray,
                       task_splits: dict,
                       memory_size: int = 50,
                       seeds: list[int] = [42],
                       output_path: str = "ablation.csv") -> pd.DataFrame:
    """Run the 2x2 ablation study."""
    set_seed(42)
    rows = []
    # Allocation: uniform vs risk-based
    # Selection: random vs herding
    alloc_methods = ["uniform", "risk"]
    select_methods = ["random", "herding"]

    for alloc in alloc_methods:
        for select in select_methods:
            for seed in seeds:
                if alloc == "uniform" and select == "random":
                    method = "random"  # This is just standard random ER
                elif alloc == "uniform" and select == "herding":
                    method = "herding"  # Standard herding ER
                elif alloc == "risk" and select == "random":
                    method = "iar_random"
                elif alloc == "risk" and select == "herding":
                    method = "iar"  # Full IAR: risk-based + herding
                
                result = run_single_experiment(
                    embeddings=embeddings,
                    labels=labels,
                    task_splits=task_splits,
                    method=method,
                    memory_size=memory_size,
                    seed=seed,
                    epochs=20,
                )
                rows.append({
                    "allocation": alloc,
                    "selection": select,
                    "seed": seed,
                    "final_accuracy": result["final_accuracy"],
                    "avg_forgetting": result["avg_forgetting"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    import os
    set_seed(42)
    _, labels = load_cifar100()
    splits = generate_task_splits()
    if not os.path.exists("embeddings.npy"):
        raise FileNotFoundError("embeddings.npy not found; run embed.py first.")
    os.makedirs("results/matrices", exist_ok=True)
    emb = np.load("embeddings.npy")
    df = run_all_experiments(emb, labels, splits, output_path="results/results.csv")
    print(df.groupby("method")[["final_accuracy", "avg_forgetting"]].mean())
    run_ablation_study(emb, labels, splits, memory_size=50, seeds=[42, 43, 44, 45, 46],
                       output_path="results/ablation.csv")
