from __future__ import annotations

import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float,
    beta_end: float,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError("timesteps must be positive.")
    if not 0.0 <= beta_start <= 1.0 or not 0.0 <= beta_end <= 1.0:
        raise ValueError("beta_start and beta_end must be in [0, 1].")
    return torch.linspace(beta_start, beta_end, timesteps, device=device)
