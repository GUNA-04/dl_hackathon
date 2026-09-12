"""Metrics for continual learning evaluation.

FIXED VERSION. The original per_class_forgetting returned INITIAL accuracy
(R[t,t]) and mislabelled it as forgetting. Real forgetting for a class is
(best accuracy ever seen on its task) minus (final accuracy on its task).
That fix is required for the R1 correlation to mean anything.
"""
import numpy as np


def final_avg_accuracy(R: np.ndarray) -> float:
    """A_T = mean over j of R[last][j]."""
    return float(np.nanmean(R[-1]))


def avg_forgetting(R: np.ndarray) -> float:
    """F_T = mean over earlier tasks of (max_i R[i][j] - R[last][j])."""
    last = R[-1]
    per_task = []
    for j in range(R.shape[1]):
        history = R[j:, j]
        history = history[~np.isnan(history)]
        if len(history) > 0 and not np.isnan(last[j]):
            per_task.append(float(np.max(history) - last[j]))
    return float(np.mean(per_task)) if per_task else 0.0


def per_task_forgetting(R: np.ndarray) -> dict[int, float]:
    """FIXED. Real forgetting per task j:
        max over rows i>=j of R[i][j]   minus   R[last][j].

    Returns {task_index: forgetting_value}. Higher = more forgotten.
    This is the quantity R1 must correlate predicted risk against.
    """
    last = R[-1]
    result = {}
    for j in range(R.shape[1]):
        history = R[j:, j]
        history = history[~np.isnan(history)]
        if len(history) > 0 and not np.isnan(last[j]):
            result[j] = float(np.max(history) - last[j])
    return result


def per_class_forgetting(R: np.ndarray,
                         task_to_class: dict[int, list[int]]) -> dict[int, float]:
    """FIXED. Map each class to its task's real forgetting value.

    Every class in task j inherits task j's forgetting (accuracy is measured
    per task, not per class, so this is the honest granularity — say so in the
    report rather than pretending to a finer measure than you have).
    """
    task_forget = per_task_forgetting(R)
    result = {}
    for t, classes in task_to_class.items():
        if t in task_forget:
            for c in classes:
                result[c] = task_forget[t]
    return result


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def summarize(R: np.ndarray) -> dict:
    return {
        "final_avg_accuracy": final_avg_accuracy(R),
        "avg_forgetting": avg_forgetting(R),
    }
