from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_voxel_grid(x: torch.Tensor, path: str | Path, max_items: int = 4) -> None:
    """Save a small occupancy visualization for binary voxel samples."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = x.detach().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    count = min(max_items, x.shape[0])

    fig = plt.figure(figsize=(count * 2.5, 2.5))
    for idx in range(count):
        ax = fig.add_subplot(1, count, idx + 1, projection="3d")
        ax.voxels(x[idx].bool().numpy(), facecolors="#3b82f6", edgecolor="k", linewidth=0.1)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
