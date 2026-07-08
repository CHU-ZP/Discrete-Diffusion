from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

from ddiff.diffusion.schedules import linear_beta_schedule


class CategoricalDiffusion:
    """Uniform categorical corruption with D3PM-style reverse sampling."""

    def __init__(
        self,
        num_classes: int,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        schedule: str = "linear",
        transition: str = "uniform",
        device: torch.device | str | None = None,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if schedule != "linear":
            raise ValueError(f"Only linear schedule is implemented, got {schedule!r}.")
        if transition != "uniform":
            raise ValueError(f"Only uniform transitions are implemented, got {transition!r}.")

        self.num_classes = int(num_classes)
        self.timesteps = int(timesteps)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        betas = linear_beta_schedule(
            timesteps,
            beta_start,
            beta_end,
            device=self.device,
        )
        self.betas = torch.zeros(timesteps + 1, device=self.device)
        self.betas[1:] = betas
        self.q = self._build_transition_matrices()
        self.qbar = self._build_cumulative_transitions()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], device: torch.device | str | None = None) -> "CategoricalDiffusion":
        diffusion_cfg = cfg["diffusion"]
        return cls(
            num_classes=int(cfg["dataset"]["num_classes"]),
            timesteps=int(diffusion_cfg["timesteps"]),
            beta_start=float(diffusion_cfg["beta_start"]),
            beta_end=float(diffusion_cfg["beta_end"]),
            schedule=diffusion_cfg.get("schedule", "linear"),
            transition=diffusion_cfg.get("transition", "uniform"),
            device=device,
        )

    def to(self, device: torch.device | str) -> "CategoricalDiffusion":
        self.device = torch.device(device)
        self.betas = self.betas.to(self.device)
        self.q = self.q.to(self.device)
        self.qbar = self.qbar.to(self.device)
        return self

    def _build_transition_matrices(self) -> torch.Tensor:
        k = self.num_classes
        eye = torch.eye(k, device=self.device)
        uniform = torch.full((k, k), 1.0 / k, device=self.device)
        q = torch.empty(self.timesteps + 1, k, k, device=self.device)
        q[0] = eye
        for t in range(1, self.timesteps + 1):
            beta = self.betas[t]
            q[t] = (1.0 - beta) * eye + beta * uniform
        return q

    def _build_cumulative_transitions(self) -> torch.Tensor:
        k = self.num_classes
        qbar = torch.empty(self.timesteps + 1, k, k, device=self.device)
        qbar[0] = torch.eye(k, device=self.device)
        for t in range(1, self.timesteps + 1):
            qbar[t] = qbar[t - 1] @ self.q[t]
        return qbar

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0)."""

        if x0.dtype != torch.long:
            raise TypeError("x0 must be a LongTensor of categorical values.")
        if t.ndim != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError("t must have shape [B].")

        self._ensure_device(x0.device)
        original_shape = x0.shape
        batch = x0.shape[0]
        flat = x0.reshape(batch, -1)
        one_hot = F.one_hot(flat, num_classes=self.num_classes).float()
        probs = torch.bmm(one_hot, self.qbar[t])
        sampled = torch.multinomial(
            probs.reshape(-1, self.num_classes),
            num_samples=1,
        ).reshape(flat.shape)
        return sampled.reshape(original_shape)

    def training_loss(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        y: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = x0.shape[0]
        t = torch.randint(
            1,
            self.timesteps + 1,
            (batch,),
            device=x0.device,
            dtype=torch.long,
        )
        x_t = self.q_sample(x0, t)
        logits = model(x_t, t, y)
        return F.cross_entropy(logits, x0, weight=class_weights)

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: Iterable[int],
        y: torch.Tensor | None = None,
        batch_size: int | None = None,
        return_chain: bool = False,
        chain_steps: Iterable[int] | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Sample from p_theta by iterating the learned reverse chain."""

        model_was_training = model.training
        model.eval()
        device = torch.device(device) if device is not None else next(model.parameters()).device
        self._ensure_device(device)

        spatial_shape = tuple(shape)
        if batch_size is None:
            if len(spatial_shape) < 2:
                raise ValueError("shape must include batch and spatial dims when batch_size is None.")
            sample_shape = spatial_shape
        else:
            sample_shape = (int(batch_size), *spatial_shape)

        x_t = torch.randint(
            0,
            self.num_classes,
            sample_shape,
            device=device,
            dtype=torch.long,
        )

        if y is not None:
            y = y.to(device=device, dtype=torch.long)
            if y.ndim == 0:
                y = y.expand(sample_shape[0])

        if chain_steps is None:
            chain_steps = torch.linspace(self.timesteps, 0, 8).round().long().tolist()
        chain_step_set = {int(step) for step in chain_steps}
        chain: dict[int, torch.Tensor] = {}
        if return_chain and self.timesteps in chain_step_set:
            chain[self.timesteps] = x_t.detach().cpu()

        for t_int in range(self.timesteps, 0, -1):
            t = torch.full((sample_shape[0],), t_int, device=device, dtype=torch.long)
            logits = model(x_t, t, y)
            x_t = self._sample_previous_from_logits(x_t, logits, t_int)
            if return_chain and (t_int - 1) in chain_step_set:
                chain[t_int - 1] = x_t.detach().cpu()

        if model_was_training:
            model.train()

        if return_chain:
            return x_t, chain
        return x_t

    def _sample_previous_from_logits(
        self,
        x_t: torch.Tensor,
        logits: torch.Tensor,
        t_int: int,
    ) -> torch.Tensor:
        batch = x_t.shape[0]
        spatial_shape = x_t.shape[1:]
        k = self.num_classes

        x_flat = x_t.reshape(-1)
        probs_x0 = logits.softmax(dim=1).movedim(1, -1).reshape(-1, k)

        qbar_prev = self.qbar[t_int - 1]
        q_t = self.q[t_int]
        qbar_t = self.qbar[t_int]

        q_t_a_b = q_t[:, x_flat].transpose(0, 1)
        denom = qbar_t[:, x_flat].transpose(0, 1).clamp_min(1e-12)
        clean_weight = probs_x0 / denom
        posterior = (clean_weight @ qbar_prev) * q_t_a_b
        posterior = posterior.clamp_min(0.0)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        sampled = torch.multinomial(posterior, num_samples=1).reshape(batch, *spatial_shape)
        return sampled

    def _ensure_device(self, device: torch.device | str) -> None:
        device = torch.device(device)
        if self.device != device:
            self.to(device)
