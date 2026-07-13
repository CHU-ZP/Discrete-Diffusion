from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_voxel_cache_metadata(cache_path: str | Path) -> dict[str, np.ndarray]:
    """Load the label metadata needed for conditioned voxel sampling."""

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    with np.load(cache_path, allow_pickle=False) as data:
        return {
            key: data[key]
            for key in data.files
            if key.startswith("subtype_") or key == "class_names"
        }


def validate_voxel_conditioning_metadata(
    metadata: Mapping[str, np.ndarray],
    num_labels: int,
) -> None:
    """Ensure every subtype-level metadata array uses the configured label order."""

    label_level_keys = (
        "subtype_names",
        "subtype_class_ids",
        "subtype_local_ids",
        "subtype_counts",
        "subtype_test_counts",
        "subtype_centers",
    )
    for key in label_level_keys:
        if key not in metadata:
            continue
        values = np.asarray(metadata[key])
        if values.ndim == 0 or values.shape[0] != num_labels:
            actual = "scalar" if values.ndim == 0 else str(values.shape[0])
            raise ValueError(
                f"{key} describes {actual} labels, but dataset.num_labels={num_labels}. "
                "The config, voxel cache, and checkpoint must use the same subtype mapping."
            )


def resolve_voxel_label_names(
    metadata: Mapping[str, np.ndarray],
    num_labels: int,
) -> list[str]:
    """Resolve conditioning ids to their human-readable cache labels."""

    validate_voxel_conditioning_metadata(metadata, num_labels)
    if "subtype_names" in metadata:
        return [str(name) for name in np.asarray(metadata["subtype_names"]).reshape(-1).tolist()]

    if "class_names" in metadata:
        class_names = [str(name) for name in np.asarray(metadata["class_names"]).reshape(-1).tolist()]
        if len(class_names) == num_labels:
            return class_names

    raise ValueError(
        "The voxel cache does not contain a label-name table matching "
        f"dataset.num_labels={num_labels}. Expected subtype_names for a subtype cache "
        "or class_names for a class-conditioned cache."
    )


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

        # Keep the cache compact in host memory. Training converts each batch
        # to long only after moving it to the target device.
        self.x = torch.from_numpy(x.astype(np.uint8))
        self.y = torch.from_numpy(y.astype(np.int64)).long() if y is not None else None

        if self.x.ndim != 4:
            raise ValueError(f"Expected voxel data shaped [N, D, H, W], got {self.x.shape}.")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | None]:
        y = None if self.y is None else self.y[idx]
        return {"x": self.x[idx], "y": y}
