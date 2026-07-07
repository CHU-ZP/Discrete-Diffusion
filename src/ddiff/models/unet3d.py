from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cond = nn.Linear(cond_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(F.silu(cond)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale.view(cond.shape[0], -1, 1, 1, 1))
        h = h + shift.view(cond.shape[0], -1, 1, 1, 1)
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet3D(nn.Module):
    """Residual 3D U-Net denoiser for categorical voxel diffusion."""

    def __init__(
        self,
        num_classes: int,
        shape: list[int] | tuple[int, ...],
        timesteps: int,
        base_channels: int = 32,
        channel_mults: list[int] | tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        conditional: bool = False,
        num_labels: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.shape = tuple(int(dim) for dim in shape)
        if len(self.shape) != 3:
            raise ValueError(f"UNet3D expects a 3D spatial shape, got {self.shape}.")
        if not channel_mults:
            raise ValueError("channel_mults must contain at least one entry.")
        if num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive.")

        down_factor = 2 ** (len(channel_mults) - 1)
        if any(dim % down_factor != 0 for dim in self.shape):
            raise ValueError(
                f"Spatial shape {self.shape} must be divisible by {down_factor} "
                "for the configured number of U-Net levels."
            )

        self.conditional = bool(conditional)
        self.channels = tuple(int(base_channels) * int(mult) for mult in channel_mults)
        cond_dim = int(base_channels) * 4

        self.token_embedding = nn.Embedding(self.num_classes, int(base_channels))
        self.input = nn.Conv3d(int(base_channels), self.channels[0], kernel_size=3, padding=1)
        self.time_embedding = nn.Sequential(
            nn.Embedding(timesteps + 1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.class_embedding = (
            nn.Embedding(int(num_labels), cond_dim)
            if self.conditional and num_labels is not None
            else None
        )

        current_channels = self.channels[0]
        skip_channels: list[int] = []
        self.encoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for level, out_channels in enumerate(self.channels):
            blocks = nn.ModuleList()
            for _ in range(int(num_res_blocks)):
                blocks.append(
                    ResidualBlock3D(
                        current_channels,
                        out_channels,
                        cond_dim=cond_dim,
                        dropout=dropout,
                    )
                )
                current_channels = out_channels
                skip_channels.append(current_channels)
            self.encoder_levels.append(blocks)
            if level != len(self.channels) - 1:
                self.downsamples.append(Downsample3D(current_channels))

        self.middle = nn.ModuleList(
            [
                ResidualBlock3D(current_channels, current_channels, cond_dim, dropout),
                ResidualBlock3D(current_channels, current_channels, cond_dim, dropout),
            ]
        )

        self.decoder_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        skip_stack = list(skip_channels)
        for level, out_channels in reversed(list(enumerate(self.channels))):
            blocks = nn.ModuleList()
            for _ in range(int(num_res_blocks)):
                skip_channels_for_block = skip_stack.pop()
                blocks.append(
                    ResidualBlock3D(
                        current_channels + skip_channels_for_block,
                        out_channels,
                        cond_dim=cond_dim,
                        dropout=dropout,
                    )
                )
                current_channels = out_channels
            self.decoder_levels.append(blocks)
            if level != 0:
                self.upsamples.append(Upsample3D(current_channels))

        self.output_norm = nn.GroupNorm(_group_count(current_channels), current_channels)
        self.output = nn.Conv3d(current_channels, self.num_classes, kernel_size=1)

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
            cond = cond + self._class_condition(y, cond)

        h = self.token_embedding(x_t).permute(0, 4, 1, 2, 3).contiguous()
        h = self.input(h)

        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.encoder_levels):
            for block in blocks:
                h = block(h, cond)
                skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        for block in self.middle:
            h = block(h, cond)

        for level, blocks in enumerate(self.decoder_levels):
            for block in blocks:
                skip = skips.pop()
                if h.shape[2:] != skip.shape[2:]:
                    raise RuntimeError(
                        f"Skip shape mismatch: decoder has {h.shape[2:]}, skip has {skip.shape[2:]}."
                    )
                h = torch.cat([h, skip], dim=1)
                h = block(h, cond)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)

        return self.output(F.silu(self.output_norm(h)))

    def _class_condition(self, y: torch.Tensor | None, cond: torch.Tensor) -> torch.Tensor:
        class_cond = torch.zeros_like(cond)
        if y is None:
            return class_cond

        y = y.to(device=cond.device, dtype=torch.long)
        if y.ndim == 0:
            y = y.expand(cond.shape[0])
        valid = y >= 0
        if torch.any(valid):
            class_cond[valid] = self.class_embedding(y[valid])
        return class_cond


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
