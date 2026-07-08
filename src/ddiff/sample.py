from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
            "Comma-separated conditioning labels. Use 'all' to cycle through labels. "
            "For ModelNet subtype caches, omit it to sample subtype labels by their empirical prior."
        ),
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Optional comma-separated ModelNet original class ids or names; samples subtypes within those classes.",
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
    labels = _build_conditioning_labels(cfg, args.num_samples, args.labels, args.classes, device)

    if cfg["dataset"]["name"] == "mnist":
        samples, chain = diffusion.sample(
            model,
            spatial_shape,
            y=labels,
            batch_size=args.num_samples,
            return_chain=True,
            device=device,
        )
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
        samples = diffusion.sample(
            model,
            spatial_shape,
            y=labels,
            batch_size=args.num_samples,
            device=device,
        )
        save_voxel_grid(samples.cpu(), sample_dir / "generated_voxels.png", labels=labels)

    print(f"Saved samples to {Path(sample_dir).resolve()}.")


def _build_conditioning_labels(
    cfg: dict,
    num_samples: int,
    labels_arg: str | None,
    classes_arg: str | None,
    device: torch.device,
) -> torch.Tensor | None:
    conditional = bool(cfg["dataset"].get("conditional", False))
    if (labels_arg is not None or classes_arg is not None) and not conditional:
        raise ValueError("--labels/--classes can only be used when dataset.conditional is true.")
    if not conditional:
        return None

    num_labels = int(cfg["dataset"].get("num_labels", 0))
    if num_labels <= 0:
        raise ValueError("Conditional sampling requires dataset.num_labels > 0.")

    if cfg["dataset"]["name"] == "modelnet10_voxel":
        metadata = _load_voxel_cache_metadata(cfg)
        if classes_arg is not None:
            return _sample_subtypes_for_classes(metadata, classes_arg, num_samples, num_labels, device)
        if labels_arg is None or labels_arg.strip().lower() == "prior":
            prior = metadata.get("subtype_counts")
            if prior is not None:
                return _sample_labels_from_prior(prior, num_samples, num_labels, device)

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


def _load_voxel_cache_metadata(cfg: dict) -> dict[str, np.ndarray]:
    cache_path = Path(cfg["dataset"]["cache_path"])
    if not cache_path.exists():
        return {}
    data = np.load(cache_path, allow_pickle=False)
    return {key: data[key] for key in data.files if key.startswith("subtype_") or key == "class_names"}


def _sample_labels_from_prior(
    counts: np.ndarray,
    num_samples: int,
    num_labels: int,
    device: torch.device,
) -> torch.Tensor:
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    if counts.shape[0] != num_labels:
        raise ValueError(f"subtype_counts has {counts.shape[0]} values, but dataset.num_labels={num_labels}.")
    if np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("subtype_counts must be non-negative and contain at least one positive count.")
    weights = torch.tensor(counts, device=device, dtype=torch.float32)
    return torch.multinomial(weights, num_samples=num_samples, replacement=True).long()


def _sample_subtypes_for_classes(
    metadata: dict[str, np.ndarray],
    classes_arg: str,
    num_samples: int,
    num_labels: int,
    device: torch.device,
) -> torch.Tensor:
    if "subtype_class_ids" not in metadata:
        raise ValueError("--classes requires a subtype cache with subtype_class_ids metadata.")
    subtype_class_ids = np.asarray(metadata["subtype_class_ids"], dtype=np.int64).reshape(-1)
    if subtype_class_ids.shape[0] != num_labels:
        raise ValueError(
            f"subtype_class_ids has {subtype_class_ids.shape[0]} values, but dataset.num_labels={num_labels}."
        )
    class_ids = _parse_class_arg(classes_arg, metadata.get("class_names"))
    eligible = np.flatnonzero(np.isin(subtype_class_ids, class_ids))
    if eligible.size == 0:
        raise ValueError(f"No subtype labels found for requested classes {class_ids}.")

    counts = metadata.get("subtype_counts")
    if counts is None:
        weights = np.ones(eligible.shape[0], dtype=np.float64)
    else:
        counts = np.asarray(counts, dtype=np.float64).reshape(-1)
        weights = counts[eligible]
        if weights.sum() <= 0:
            weights = np.ones_like(weights)

    sampled_local = torch.multinomial(
        torch.tensor(weights, device=device, dtype=torch.float32),
        num_samples=num_samples,
        replacement=True,
    )
    return torch.tensor(eligible, device=device, dtype=torch.long)[sampled_local]


def _parse_class_arg(classes_arg: str, class_names: np.ndarray | None) -> list[int]:
    names = [str(name).lower() for name in class_names.tolist()] if class_names is not None else []
    class_ids: list[int] = []
    max_class_id = len(names) - 1 if names else None
    for item in classes_arg.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lstrip("-").isdigit():
            class_id = int(item)
            if class_id < 0 or (max_class_id is not None and class_id > max_class_id):
                if max_class_id is None:
                    raise ValueError(f"Class id {class_id} must be non-negative.")
                raise ValueError(f"Class id {class_id} is outside the valid range [0, {max_class_id}].")
            class_ids.append(class_id)
            continue
        if item.lower() not in names:
            raise ValueError(f"Unknown class name {item!r}; available names are {names}.")
        class_ids.append(names.index(item.lower()))
    if not class_ids:
        raise ValueError("--classes did not contain any class ids or names.")
    return class_ids


if __name__ == "__main__":
    main()
