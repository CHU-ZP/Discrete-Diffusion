from __future__ import annotations

from collections.abc import Mapping

import torch


def load_sampling_weights(
    model: torch.nn.Module,
    checkpoint: Mapping,
    selection: str = "auto",
) -> str:
    """Load raw or EMA model weights and return the selected checkpoint key.

    ``auto`` prefers EMA weights produced by current training runs and falls
    back to the raw model weights for checkpoints created by older versions.
    """

    if selection not in {"auto", "ema", "model"}:
        raise ValueError("selection must be one of: auto, ema, model.")

    if selection in {"auto", "ema"} and "ema_model" in checkpoint:
        key = "ema_model"
    elif selection == "ema":
        raise ValueError(
            "This checkpoint does not contain EMA weights. Use --weights model "
            "or --weights auto for an older checkpoint."
        )
    elif "model" in checkpoint:
        key = "model"
    else:
        raise KeyError("Checkpoint does not contain a 'model' state dict.")

    model.load_state_dict(checkpoint[key])
    return key
