"""Split the 100 CIFAR-100 classes into 20 ordered tasks of 5 classes each,
deterministically under seed 42. Return per task: which classes, and train/val
index lists into the full training set."""
import numpy as np
import torch
from torchvision import datasets
from seed import set_seed


def _indices_per_class(train_labels_np: np.ndarray) -> dict[int, np.ndarray]:
    """Return a dict mapping each class label to the array of global indices
    where that label appears in the training set."""
    idx = {}
    for c in range(100):
        idx[c] = np.where(train_labels_np == c)[0]
    return idx


def generate_task_splits(seed: int = 42,
                         n_tasks: int = 20,
                         n_per_task: int = 5,
                         n_val_per_class: int = 5) -> dict:
    """Generate task splits.

    Returns
    -------
    task_dict : dict
        Keys are task indices 0 .. n_tasks-1.
        Each value is a dict with keys:
            'classes' : list of class ids (length n_per_task)
            'train_idx' : global indices for the training samples of those classes
            'val_idx'   : global indices for the validation samples of those classes
    """
    set_seed(seed)
    # Load the whole training set
    train_ds = datasets.CIFAR100(root="./data", train=True, download=True)
    train_labels = np.array(train_ds.targets, dtype=np.int64)
    train_images = train_ds.data  # (N,32,32,3) uint8

    # Map class -> global indices
    class_to_idx = _indices_per_class(train_labels)

    # Deterministically permute the 100 classes under the seed
    rng = np.random.default_rng(seed)
    perm = rng.permutation(100)

    # Create ordered tasks
    task_dict = {}
    for t in range(n_tasks):
        # pick the next n_per_task classes
        start = t * n_per_task
        task_classes = perm[start:start + n_per_task].tolist()

        # gather train+val indices for those classes
        all_tr = []
        all_va = []
        for c in task_classes:
            cls_idx = class_to_idx[c]
            # randomly choose n_val_per_class as validation (without replacement)
            # we seed the rng per class to be deterministic overall
            # Use a deterministic split: first n_val_per_class go to val
            # but we shuffle cls_idx using a hash of (seed, c)
            cs_rng = np.random.default_rng(seed * 1000 + c)
            cs_rng.shuffle(cls_idx)
            val_take = cls_idx[:n_val_per_class]
            train_take = cls_idx[n_val_per_class:]
            all_va.append(val_take)
            all_tr.append(train_take)

        task_dict[t] = {
            "classes": task_classes,
            "train_idx": np.concatenate(all_tr),
            "val_idx": np.concatenate(all_va),
        }

    return task_dict


if __name__ == "__main__":
    splits = generate_task_splits()
    for t, info in splits.items():
        print(f"Task {t}: classes={info['classes']}, "
              f"train_n={len(info['train_idx'])}, val_n={len(info['val_idx'])}")