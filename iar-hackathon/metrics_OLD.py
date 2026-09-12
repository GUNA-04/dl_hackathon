"""Metrics for continual learning evaluation."""
import numpy as np


def final_avg_accuracy(R: np.ndarray) -> float:
    """A_T = mean over j of R[last][j]."""
    return float(np.nanmean(R[-1]))


def avg_forgetting(R: np.ndarray) -> float:
    """F_T = mean over earlier tasks of (max_i R[i][j] - R[last][j])."""
    last = R[-1]
    per_task = []
    for j in range(R.shape[1]):
        # only rows after task j were evaluated on task j
        history = R[j:, j]
        history = history[~np.isnan(history)]
        if len(history) > 0:
            if not np.isnan(last[j]):
                per_task.append(float(np.max(history) - last[j]))
    return float(np.mean(per_task)) if per_task else 0.0


def per_class_forgetting(R: np.ndarray,
                         task_to_class: dict[int, list[int]]) -> dict[int, float]:
    """Per-class forgetting: max accuracy on that class's task minus final accuracy."""
    result = {}
    for t, classes in task_to_class.items():
        for c in classes:
            acc = float(R[t, t]) if t < R.shape[0] and t < R.shape[1] else None
            if acc is not None:
                result[c] = acc
    return result


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(x) < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def summarize(R: np.ndarray) -> dict:
    """Return final average accuracy and average forgetting."""
    return {
        "final_avg_accuracy": final_avg_accuracy(R),
        "avg_forgetting": avg_forgetting(R),
    }


if __name__ == "__main__":
    # quick demo
    R = np.array([[0.90, -1.0], [0.30, 0.80]])
    print(summarize(R))
