from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch


def save_image_grid(
    x: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    labels: torch.Tensor | None = None,
    value_range: tuple[int, int] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = x.detach().cpu()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if labels is not None:
        labels = labels.detach().cpu().reshape(-1)
        if labels.shape[0] < x.shape[0]:
            raise ValueError("labels must have at least one item per image.")

    vmin, vmax = _value_range(x, value_range)
    count = x.shape[0]
    cols = min(nrow, count)
    rows = (count + cols - 1) // cols
    cell_height = 1.35 if labels is not None else 1.2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * cell_height), squeeze=False)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < count:
            ax.imshow(x[idx], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            if labels is not None:
                ax.set_title(str(int(labels[idx].item())), fontsize=8)

    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_forward_chain(
    chain: dict[int, torch.Tensor] | Iterable[torch.Tensor],
    path: str | Path,
    value_range: tuple[int, int] | None = None,
) -> None:
    frames = _chain_frames(chain, reverse=False)
    _save_chain(frames, path, value_range=value_range)


def save_reverse_chain(
    chain: dict[int, torch.Tensor] | Iterable[torch.Tensor],
    path: str | Path,
    value_range: tuple[int, int] | None = None,
) -> None:
    frames = _chain_frames(chain, reverse=True)
    _save_chain(frames, path, value_range=value_range)


def _chain_frames(
    chain: dict[int, torch.Tensor] | Iterable[torch.Tensor],
    *,
    reverse: bool,
) -> list[tuple[int | None, torch.Tensor]]:
    if isinstance(chain, dict):
        keys = sorted(chain.keys(), reverse=reverse)
        return [(key, chain[key]) for key in keys]
    return [(None, frame) for frame in chain]


def _save_chain(
    frames: list[tuple[int | None, torch.Tensor]],
    path: str | Path,
    *,
    value_range: tuple[int, int] | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    reference = torch.cat([frame.detach().cpu().reshape(-1) for _, frame in frames])
    vmin, vmax = _value_range(reference, value_range)
    cols = len(frames)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 1.3, 1.5), squeeze=False)
    for ax, (step, frame) in zip(axes.flat, frames):
        image = frame.detach().cpu()
        if image.ndim == 3:
            image = image[0]
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axis("off")
        if step is not None:
            ax.set_title(f"t={step}", fontsize=8)

    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _value_range(x: torch.Tensor, value_range: tuple[int, int] | None) -> tuple[int, int]:
    if value_range is not None:
        return value_range
    if x.numel() == 0:
        return 0, 1
    return 0, max(1, int(x.max().item()))
