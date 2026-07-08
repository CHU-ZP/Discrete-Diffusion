from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNetVoxelDataset(Dataset):
    """Cached ModelNet voxel tensors for categorical diffusion.

    The preferred cache is an ``.npz`` with split-specific ``train_x`` /
    ``test_x`` uint8 occupancy arrays and matching ``train_y`` / ``test_y``
    labels. The voxel values are categorical tokens, usually ``0`` for empty
    and ``1`` for occupied.
    """

    def __init__(self, cache_path: str | Path, split: str = "train") -> None:
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"Voxel cache not found at {self.cache_path}. "
                "Run scripts/prepare_modelnet_voxels.py or point "
                "dataset.cache_path at an existing .npz cache."
            )

        data = np.load(self.cache_path, allow_pickle=False)
        self.metadata = {
            key: data[key]
            for key in data.files
            if key.startswith("subtype_")
            or key
            in {
                "class_names",
                "class_counts",
                "resolution",
                "voxel_token_classes",
                "num_model_classes",
                "filled_interiors",
                "surface_dilation",
            }
        }
        if f"{split}_x" in data:
            x = data[f"{split}_x"]
            y = data[f"{split}_y"] if f"{split}_y" in data else None
        elif "x" in data:
            x = data["x"]
            y = data["y"] if "y" in data else None
            if "split" in data:
                split_values = data["split"].astype(str)
                mask = split_values == split
                x = x[mask]
                y = y[mask] if y is not None else None
        else:
            raise ValueError(
                f"{self.cache_path} must contain either x or {split}_x arrays."
            )

        self.x = torch.from_numpy(x.astype(np.int64)).long()
        self.y = torch.from_numpy(y.astype(np.int64)).long() if y is not None else None

        if self.x.ndim != 4:
            raise ValueError(f"Expected voxel data shaped [N, D, H, W], got {self.x.shape}.")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | None]:
        y = None if self.y is None else self.y[idx]
        return {"x": self.x[idx], "y": y}
