"""Freeze a pretrained ImageNet ResNet-18 backbone and store embeddings."""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def build_resnet18_prototype() -> nn.Module:
    """Return a ResNet-18 model with the final classification layer removed,
    so forward passes yield (batch, 512) embeddings."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Identity()
    return model


def compute_embeddings(model: nn.Module, images: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run the model on the supplied images and collect (N, 512) embeddings.

    Parameters
    ----------
    model : nn.Module
        The frozen ResNet-18 prototype network.
    images : np.ndarray
        Shape (N, 3, 32, 32) uint8.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    embeddings : np.ndarray
        Shape (N, 512) float32.
    """
    model.eval()
    model.to(device)
    img_tensor = torch.from_numpy(images).float().to(device)
    with torch.no_grad():
        emb = model(img_tensor)  # (N, 512)
    return emb.cpu().numpy()


def save_embeddings(embeddings: np.ndarray, path: str = "embeddings.npy") -> None:
    """Persist embeddings to disk."""
    np.save(path, embeddings)
    print(f"Saved embeddings to {path}")


if __name__ == "__main__":
    from seed import set_seed
    from data import load_cifar100

    set_seed(42)
    images, _ = load_cifar100()
    print("Loaded", len(images), "images")

    model = build_resnet18_prototype()
    print("Prototype network built")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = compute_embeddings(model, images, device=device)
    print("Embeddings shape:", embeddings.shape)

    save_embeddings(embeddings)
