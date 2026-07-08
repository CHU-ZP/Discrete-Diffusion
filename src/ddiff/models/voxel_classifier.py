from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.silu(self.norm(self.conv(x))))


class VoxelClassifier3D(nn.Module):
    """Compact 3D CNN classifier used to learn voxel shape embeddings."""

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 24,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.embedding_dim = int(embedding_dim)

        c1 = self.base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        self.encoder = nn.Sequential(
            ConvBlock3D(1, c1, dropout=dropout),
            ConvBlock3D(c1, c1, dropout=dropout),
            ConvBlock3D(c1, c2, stride=2, dropout=dropout),
            ConvBlock3D(c2, c2, dropout=dropout),
            ConvBlock3D(c2, c3, stride=2, dropout=dropout),
            ConvBlock3D(c3, c3, dropout=dropout),
            ConvBlock3D(c3, c4, stride=2, dropout=dropout),
            ConvBlock3D(c4, c4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.embedding = nn.Sequential(
            nn.Linear(c4, self.embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, x: torch.Tensor, *, return_embedding: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.extract_embedding(x)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f"Expected voxel tensor shaped [B, D, H, W] or [B, 1, D, H, W], got {tuple(x.shape)}.")
        h = self.encoder(x.float())
        h = self.pool(h).flatten(1)
        return self.embedding(h)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
