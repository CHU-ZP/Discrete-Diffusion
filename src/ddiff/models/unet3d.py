from __future__ import annotations

import torch
from torch import nn


class UNet3D(nn.Module):
    """Minimal 3D categorical model scaffold following the shared interface."""

    def __init__(
        self,
        num_classes: int,
        shape: list[int] | tuple[int, ...],
        timesteps: int,
        base_channels: int = 32,
        dropout: float = 0.1,
        conditional: bool = False,
        num_labels: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.shape = tuple(int(dim) for dim in shape)
        if len(self.shape) != 3:
            raise ValueError(f"UNet3D expects a 3D spatial shape, got {self.shape}.")
        self.conditional = bool(conditional)

        self.token_embedding = nn.Embedding(self.num_classes, base_channels)
        self.time_embedding = nn.Embedding(timesteps + 1, base_channels)
        self.class_embedding = (
            nn.Embedding(int(num_labels), base_channels)
            if self.conditional and num_labels is not None
            else None
        )
        self.net = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, self.num_classes, kernel_size=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_t.dtype != torch.long:
            raise TypeError("x_t must be a LongTensor.")
        if tuple(x_t.shape[1:]) != self.shape:
            raise ValueError(f"Expected spatial shape {self.shape}, got {tuple(x_t.shape[1:])}.")

        h = self.token_embedding(x_t).permute(0, 4, 1, 2, 3).contiguous()
        h = h + self.time_embedding(t).view(t.shape[0], -1, 1, 1, 1)
        if self.class_embedding is not None:
            if y is None:
                raise ValueError("Conditional model requires y labels.")
            h = h + self.class_embedding(y).view(y.shape[0], -1, 1, 1, 1)
        return self.net(h)
