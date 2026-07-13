from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_voxel_grid(
    x: torch.Tensor,
    path: str | Path,
    max_items: int | None = 4,
    labels: torch.Tensor | None = None,
    label_names: Sequence[str] | None = None,
    ncols: int = 4,
) -> None:
    """Save a small occupancy visualization for binary voxel samples."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = x.detach().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected voxel data shaped [N, D, H, W], got {tuple(x.shape)}.")
    if labels is not None:
        labels = labels.detach().cpu().reshape(-1)
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive or None.")
    if ncols <= 0:
        raise ValueError("ncols must be positive.")

    count = x.shape[0] if max_items is None else min(max_items, x.shape[0])
    if count == 0:
        raise ValueError("Cannot visualize an empty voxel batch.")
    if labels is not None and labels.shape[0] < count:
        raise ValueError("labels must have at least one item per visualized voxel sample.")

    cols = min(ncols, count)
    rows = (count + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 2.7, rows * 2.7))
    for idx in range(count):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        ax.voxels(x[idx].bool().numpy(), facecolors="#3b82f6", edgecolor="k", linewidth=0.1)
        if labels is not None and idx < labels.shape[0]:
            label_id = int(labels[idx].item())
            if label_names is None:
                title = str(label_id)
            else:
                if label_id < 0 or label_id >= len(label_names):
                    raise ValueError(
                        f"Label {label_id} cannot be resolved with {len(label_names)} label names."
                    )
                title = str(label_names[label_id])
            ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
