from __future__ import annotations

from typing import Any

from ddiff.models.cnn2d import CNN2DDenoiser
from ddiff.models.unet3d import UNet3D


def build_model(cfg: dict[str, Any]):
    dataset_cfg = cfg["dataset"]
    diffusion_cfg = cfg["diffusion"]
    model_cfg = cfg["model"]

    common = {
        "num_classes": int(dataset_cfg["num_classes"]),
        "shape": dataset_cfg["shape"],
        "timesteps": int(diffusion_cfg["timesteps"]),
        "conditional": bool(dataset_cfg.get("conditional", False)),
        "num_labels": dataset_cfg.get("num_labels"),
    }

    name = model_cfg["name"]
    if name == "cnn2d":
        return CNN2DDenoiser(
            **common,
            base_channels=int(model_cfg.get("base_channels", 96)),
            num_blocks=int(model_cfg.get("num_blocks", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name == "unet3d":
        return UNet3D(
            **common,
            base_channels=int(model_cfg.get("base_channels", 32)),
            channel_mults=tuple(int(mult) for mult in model_cfg.get("channel_mults", [1, 2, 4])),
            num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )

    raise ValueError(f"Unknown model backend {name!r}.")
