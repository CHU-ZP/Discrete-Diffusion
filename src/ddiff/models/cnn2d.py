from __future__ import annotations

import torch
from torch import nn


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.cond = nn.Linear(channels, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.cond(cond).view(cond.shape[0], -1, 1, 1)
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        return x + h


class CNN2DDenoiser(nn.Module):
    """CNN denoiser for 2D categorical diffusion."""

    def __init__(
        self,
        num_classes: int,
        shape: list[int] | tuple[int, ...],
        timesteps: int,
        base_channels: int = 96,
        num_blocks: int = 8,
        dropout: float = 0.1,
        conditional: bool = False,
        num_labels: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.shape = tuple(int(dim) for dim in shape)
        if len(self.shape) != 2:
            raise ValueError(f"CNN2DDenoiser expects a 2D spatial shape, got {self.shape}.")
        self.conditional = bool(conditional)

        self.token_embedding = nn.Embedding(self.num_classes, base_channels)
        self.time_embedding = nn.Embedding(timesteps + 1, base_channels)
        self.class_embedding = (
            nn.Embedding(int(num_labels), base_channels)
            if self.conditional and num_labels is not None
            else None
        )

        self.input = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock2D(base_channels, dropout=dropout) for _ in range(int(num_blocks))]
        )
        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output = nn.Conv2d(base_channels, self.num_classes, kernel_size=1)

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

        cond = self.time_embedding(t)
        if self.class_embedding is not None:
            if y is None:
                raise ValueError("Conditional model requires y labels.")
            cond = cond + self.class_embedding(y)

        h = self.token_embedding(x_t).permute(0, 3, 1, 2).contiguous()
        h = self.input(h)
        h = h + cond.view(cond.shape[0], -1, 1, 1)
        for block in self.blocks:
            h = block(h, cond)
        return self.output(torch.nn.functional.silu(self.output_norm(h)))


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
