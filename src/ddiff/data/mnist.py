from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNISTDataset(Dataset):
    """MNIST images quantized into integer categorical tokens."""

    def __init__(
        self,
        root: str | Path = "data/mnist",
        split: str = "train",
        num_classes: int = 32,
        download: bool = True,
    ) -> None:
        if split not in {"train", "test", "all"}:
            raise ValueError(f"Unknown split {split!r}; expected train, test, or all.")
        if not 2 <= num_classes <= 256:
            raise ValueError("num_classes must be in [2, 256] for quantized MNIST pixels.")

        self.root = Path(root)
        self.num_classes = int(num_classes)

        if split == "all":
            train = MNIST(root=str(self.root), train=True, download=download)
            test = MNIST(root=str(self.root), train=False, download=download)
            images = torch.cat([train.data, test.data], dim=0)
            labels = torch.cat([train.targets, test.targets], dim=0)
        else:
            dataset = MNIST(root=str(self.root), train=(split == "train"), download=download)
            images = dataset.data
            labels = dataset.targets

        self.x = self._quantize(images)
        self.y = labels.long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[idx], "y": self.y[idx]}

    def _quantize(self, images: torch.Tensor) -> torch.Tensor:
        images = images.long()
        if self.num_classes == 256:
            return images
        return torch.div(images * self.num_classes, 256, rounding_mode="floor").clamp_max(
            self.num_classes - 1
        )
