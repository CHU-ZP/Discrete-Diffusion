from __future__ import annotations

from typing import Any

import torch

from ddiff.data.mnist import MNISTDataset
from ddiff.data.modelnet_voxel import ModelNetVoxelDataset


def build_dataset(cfg: dict[str, Any], split: str = "train"):
    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"]

    if name == "mnist":
        return MNISTDataset(
            root=dataset_cfg.get("root", "data/mnist"),
            split=split,
            num_classes=int(dataset_cfg["num_classes"]),
            download=bool(dataset_cfg.get("download", True)),
        )

    if name == "modelnet10_voxel":
        return ModelNetVoxelDataset(
            cache_path=dataset_cfg["cache_path"],
            split=split,
        )

    raise ValueError(f"Unknown dataset backend {name!r}.")


def collate_samples(samples: list[dict[str, torch.Tensor | None]]) -> dict[str, torch.Tensor | None]:
    x = torch.stack([sample["x"] for sample in samples])
    ys = [sample["y"] for sample in samples]
    y = None if all(item is None for item in ys) else torch.stack([item for item in ys if item is not None])
    return {"x": x, "y": y}
