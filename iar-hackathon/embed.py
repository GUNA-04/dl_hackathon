"""Freeze a pretrained ImageNet ResNet-18 backbone and store embeddings.

FIXED VERSION. The original had two bugs that made every downstream number
meaningless:
  1. Raw uint8 pixels (0-255) were fed straight into ResNet. ImageNet models
     expect inputs scaled to [0,1] then normalised by ImageNet mean/std.
     Feeding 0-255 produces saturated, mostly-dead embeddings.
  2. 32x32 images were never resized to 224x224, so ResNet's pooling produced
     degenerate features.

Both are fixed below. Verify with the linear probe at the bottom before
trusting any downstream result: it must exceed ~50% on 100 classes.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


# ImageNet normalisation constants — these are not optional.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_resnet18_prototype() -> nn.Module:
    """ResNet-18 with the classifier removed -> (batch, 512) embeddings."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Identity()
    return model


def _preprocess_batch(images_uint8: np.ndarray) -> torch.Tensor:
    """images_uint8: (B, 3, 32, 32) uint8  ->  (B, 3, 224, 224) normalised float.

    Steps that were missing before:
      uint8 -> float in [0,1] -> resize to 224 -> ImageNet normalise.
    """
    x = torch.from_numpy(images_uint8).float() / 255.0          # [0,1]
    x = transforms.functional.resize(x, [224, 224], antialias=True)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def compute_embeddings(model: nn.Module, images: np.ndarray,
                       device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """Run the model on (N,3,32,32) uint8 images -> (N,512) float32 embeddings.

    Batched, because after resize to 224 the full tensor is far too big to hold
    at once.
    """
    model.eval()
    model.to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            x = _preprocess_batch(batch).to(device)
            emb = model(x)                      # (B, 512)
            out.append(emb.cpu().numpy())
            if start % (batch_size * 20) == 0:
                print(f"  embedded {start + len(batch)}/{len(images)}")
    return np.concatenate(out).astype(np.float32)


def save_embeddings(embeddings: np.ndarray, path: str = "embeddings.npy") -> None:
    np.save(path, embeddings)
    print(f"Saved embeddings to {path}  shape={embeddings.shape}")


def linear_probe(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Sanity gate: train a linear classifier on ALL 100 classes, report test acc.

    Healthy ImageNet embeddings should give well above 50% here. If this is
    near chance (1%), the embeddings are broken and nothing downstream is valid.
    """
    from numpy.random import default_rng
    rng = default_rng(42)
    idx = rng.permutation(len(embeddings))
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    Xtr = torch.from_numpy(embeddings[tr]).float()
    ytr = torch.from_numpy(labels[tr]).long()
    Xte = torch.from_numpy(embeddings[te]).float()
    yte = torch.from_numpy(labels[te]).long()

    clf = nn.Linear(embeddings.shape[1], int(labels.max()) + 1)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for epoch in range(30):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(clf(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (clf(Xte).argmax(1) == yte).float().mean().item()
    return acc


if __name__ == "__main__":
    from seed import set_seed
    from data import load_cifar100

    set_seed(42)
    images, labels = load_cifar100()
    print("Loaded", len(images), "images")

    model = build_resnet18_prototype()
    print("Backbone built (frozen ResNet-18, ImageNet weights)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    embeddings = compute_embeddings(model, images, device=device)
    print("Embeddings shape:", embeddings.shape)
    print("Embedding stats  min/max/mean/std:",
          embeddings.min(), embeddings.max(), embeddings.mean(), embeddings.std())
    print("Fraction exactly zero:", (embeddings == 0).mean(),
          "(should be small, was 0.74 when broken)")

    save_embeddings(embeddings)

    print("\nRunning linear probe sanity gate...")
    acc = linear_probe(embeddings, labels)
    print(f"LINEAR PROBE ACCURACY: {acc:.3f}")
    if acc < 0.40:
        print("  *** FAIL: embeddings are still broken. Do not proceed. ***")
    else:
        print("  PASS: embeddings carry class information. Safe to proceed.")
