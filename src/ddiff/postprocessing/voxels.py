from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ComponentFilterStats:
    components: int
    original_voxels: int
    kept_voxels: int

    @property
    def removed_voxels(self) -> int:
        return self.original_voxels - self.kept_voxels


def filter_voxel_components(
    samples: torch.Tensor,
    *,
    mode: str = "largest",
    connectivity: int = 6,
) -> tuple[torch.Tensor, list[ComponentFilterStats]]:
    """Remove disconnected occupancy fragments from a voxel sample batch.

    ``largest`` keeps only the largest connected occupied component in each
    sample. ``none`` leaves samples unchanged while still reporting component
    statistics. Occupied voxels are all non-zero values.
    """

    if mode not in {"largest", "none"}:
        raise ValueError(f"Unknown voxel component filter mode {mode!r}.")
    if connectivity not in {6, 26}:
        raise ValueError("Voxel connectivity must be 6 or 26.")

    squeeze = samples.ndim == 3
    if squeeze:
        samples = samples.unsqueeze(0)
    if samples.ndim != 4:
        raise ValueError(
            f"Expected voxel samples shaped [B, D, H, W] or [D, H, W], got {tuple(samples.shape)}."
        )

    original_device = samples.device
    original_dtype = samples.dtype
    occupied_batch = samples.detach().cpu().numpy() != 0
    filtered_batch = np.zeros_like(occupied_batch, dtype=np.uint8)
    stats: list[ComponentFilterStats] = []

    for index, occupied in enumerate(occupied_batch):
        components = _connected_components(occupied, connectivity=connectivity)
        original_voxels = int(occupied.sum())
        if mode == "none":
            filtered_batch[index] = occupied
            kept_voxels = original_voxels
        elif components:
            largest = max(components, key=len)
            filtered_batch[index].reshape(-1)[largest] = 1
            kept_voxels = len(largest)
        else:
            kept_voxels = 0

        stats.append(
            ComponentFilterStats(
                components=len(components),
                original_voxels=original_voxels,
                kept_voxels=kept_voxels,
            )
        )

    filtered = torch.from_numpy(filtered_batch).to(device=original_device, dtype=original_dtype)
    if squeeze:
        filtered = filtered[0]
    return filtered, stats


def _connected_components(occupied: np.ndarray, *, connectivity: int) -> list[list[int]]:
    depth, height, width = occupied.shape
    plane = height * width
    occupied_flat = occupied.reshape(-1)
    visited = np.zeros(occupied_flat.shape[0], dtype=bool)
    components: list[list[int]] = []

    for seed_value in np.flatnonzero(occupied_flat):
        seed = int(seed_value)
        if visited[seed]:
            continue

        visited[seed] = True
        queue: deque[int] = deque([seed])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            z, remainder = divmod(current, plane)
            y, x = divmod(remainder, width)
            for neighbor in _neighbor_indices(
                z,
                y,
                x,
                depth=depth,
                height=height,
                width=width,
                plane=plane,
                connectivity=connectivity,
            ):
                if occupied_flat[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(component)

    return components


def _neighbor_indices(
    z: int,
    y: int,
    x: int,
    *,
    depth: int,
    height: int,
    width: int,
    plane: int,
    connectivity: int,
):
    current = z * plane + y * width + x
    if connectivity == 6:
        if z > 0:
            yield current - plane
        if z + 1 < depth:
            yield current + plane
        if y > 0:
            yield current - width
        if y + 1 < height:
            yield current + width
        if x > 0:
            yield current - 1
        if x + 1 < width:
            yield current + 1
        return

    for dz in (-1, 0, 1):
        nz = z + dz
        if not 0 <= nz < depth:
            continue
        for dy in (-1, 0, 1):
            ny = y + dy
            if not 0 <= ny < height:
                continue
            for dx in (-1, 0, 1):
                nx = x + dx
                if dx == dy == dz == 0 or not 0 <= nx < width:
                    continue
                yield nz * plane + ny * width + nx
