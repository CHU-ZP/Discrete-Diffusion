from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ddiff.diffusion.categorical import CategoricalDiffusion
from ddiff.models.registry import build_model
from ddiff.utils.config import ensure_dir, load_config, resolve_device
from ddiff.utils.seed import set_seed
from ddiff.visualization.images import save_image_grid, save_reverse_chain
from ddiff.visualization.voxels import save_voxel_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained categorical diffusion model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples to generate.")
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Comma-separated class labels for conditional sampling. "
            "Use 'all' or omit it to cycle through all labels."
        ),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale. Defaults to sampling.guidance_scale or 1.0.",
    )
    parser.add_argument("--device", default=None, help="Optional device override.")
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["train"]["device"] = args.device
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(cfg["train"].get("device", "auto"))
    sample_dir = ensure_dir(cfg["output"]["sample_dir"])

    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = CategoricalDiffusion.from_config(cfg, device=device)

    spatial_shape = tuple(cfg["dataset"]["shape"])
    labels = _build_conditioning_labels(cfg, args.num_samples, args.labels, device)
    guidance_scale = _resolve_guidance_scale(cfg, args.guidance_scale)
    samples, chain = diffusion.sample(
        model,
        spatial_shape,
        y=labels,
        batch_size=args.num_samples,
        return_chain=True,
        device=device,
        guidance_scale=guidance_scale,
    )

    if cfg["dataset"]["name"] == "mnist":
        value_range = (0, int(cfg["dataset"]["num_classes"]) - 1)
        save_image_grid(
            samples.cpu(),
            sample_dir / "generated_samples.png",
            nrow=8,
            labels=labels,
            value_range=value_range,
        )
        save_reverse_chain(chain, sample_dir / "reverse_chain.png", value_range=value_range)
    elif cfg["dataset"]["name"] == "modelnet10_voxel":
        save_voxel_grid(samples.cpu(), sample_dir / "generated_voxels.png", labels=labels)

    print(f"Saved samples to {Path(sample_dir).resolve()}.")


def _build_conditioning_labels(
    cfg: dict,
    num_samples: int,
    labels_arg: str | None,
    device: torch.device,
) -> torch.Tensor | None:
    conditional = bool(cfg["dataset"].get("conditional", False))
    if labels_arg is not None and not conditional:
        raise ValueError("--labels can only be used when dataset.conditional is true.")
    if not conditional:
        return None

    num_labels = int(cfg["dataset"].get("num_labels", 0))
    if num_labels <= 0:
        raise ValueError("Conditional sampling requires dataset.num_labels > 0.")

    requested = _parse_labels_arg(labels_arg, num_labels)
    repeats = (num_samples + len(requested) - 1) // len(requested)
    labels = (requested * repeats)[:num_samples]
    return torch.tensor(labels, device=device, dtype=torch.long)


def _parse_labels_arg(labels_arg: str | None, num_labels: int) -> list[int]:
    if labels_arg is None or labels_arg.strip().lower() == "all":
        return list(range(num_labels))

    labels: list[int] = []
    for item in labels_arg.split(","):
        item = item.strip()
        if not item:
            continue
        labels.append(int(item))

    if not labels:
        raise ValueError("--labels did not contain any labels.")
    for label in labels:
        if label < 0 or label >= num_labels:
            raise ValueError(f"Label {label} is outside the valid range [0, {num_labels - 1}].")
    return labels


def _resolve_guidance_scale(cfg: dict, guidance_scale: float | None) -> float:
    if guidance_scale is not None:
        return float(guidance_scale)
    return float(cfg.get("sampling", {}).get("guidance_scale", 1.0))


if __name__ == "__main__":
    main()
