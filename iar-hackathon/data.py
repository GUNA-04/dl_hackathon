"""Load CIFAR-100 and return raw images + labels as numpy arrays."""
import numpy as np
import torch
from torchvision import datasets


def load_cifar100(root: str = "./data") -> tuple[np.ndarray, np.ndarray]:
    """Download (if needed) and load CIFAR-100.

    Returns
    -------
    images : (N, 3, 32, 32) uint8 numpy array
    labels  : (N,) int64 numpy array
    """
    ds = datasets.CIFAR100(root=root, train=True, download=True)
    images = ds.data  # (N, 32, 32, 3) uint8
    labels = np.array(ds.targets, dtype=np.int64)
    images = np.transpose(images, (0, 3, 1, 2))  # HWC -> CHW
    return images, labels


if __name__ == "__main__":
    from seed import set_seed

    set_seed(42)
    imgs, lbls = load_cifar100()
    print("images:", imgs.shape, imgs.dtype, "labels:", lbls.shape, lbls.dtype)