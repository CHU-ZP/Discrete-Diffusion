from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ddiff.data.modelnet_voxel import (
    load_voxel_cache_metadata,
    resolve_voxel_label_names,
    validate_voxel_conditioning_metadata,
)
from ddiff.diffusion.categorical import CategoricalDiffusion
from ddiff.models.registry import build_model
from ddiff.postprocessing.voxels import ComponentFilterStats, filter_voxel_components
from ddiff.utils.checkpoints import load_sampling_weights
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
    parser.add_argument(
        "--weights",
        choices=("auto", "ema", "model"),
        default="auto",
        help="Checkpoint weights: auto prefers EMA and supports legacy raw-only checkpoints.",
    )
    parser.add_argument(
        "--voxel-component-filter",
        choices=("largest", "none"),
        default=None,
        help="Override voxel connected-component filtering from the config.",
    )
    parser.add_argument(
        "--voxel-connectivity",
        type=int,
        choices=(6, 26),
        default=None,
        help="Override voxel connectivity used by component filtering.",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.voxel_component_filter is not None:
        cfg.setdefault("sample", {})["voxel_component_filter"] = args.voxel_component_filter
    if args.voxel_connectivity is not None:
        cfg.setdefault("sample", {})["voxel_connectivity"] = args.voxel_connectivity
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(cfg["train"].get("device", "auto"))
    sample_dir = ensure_dir(cfg["output"]["sample_dir"])

    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    _validate_checkpoint_sampling_config(cfg, checkpoint)
    loaded_weights = load_sampling_weights(model, checkpoint, args.weights)
    print(f"Loaded checkpoint weights: {loaded_weights}.")
    model.eval()
    diffusion = CategoricalDiffusion.from_config(cfg, device=device)

    spatial_shape = tuple(cfg["dataset"]["shape"])
    voxel_metadata: dict[str, np.ndarray] | None = None
    voxel_label_names: list[str] | None = None
    if cfg["dataset"]["name"] == "modelnet10_voxel" and bool(
        cfg["dataset"].get("conditional", False)
    ):
        num_labels = int(cfg["dataset"].get("num_labels", 0))
        voxel_metadata = load_voxel_cache_metadata(cfg["dataset"]["cache_path"])
        voxel_label_names = _resolve_sampling_label_names(checkpoint, voxel_metadata, num_labels)

    labels = _build_conditioning_labels(
        cfg,
        args.num_samples,
        args.labels,
        args.classes,
        device,
        voxel_metadata=voxel_metadata,
    )

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
        raw_samples = samples.cpu()
        sample_cfg = cfg.get("sample", {})
        component_filter_mode = str(sample_cfg.get("voxel_component_filter", "largest"))
        component_connectivity = int(sample_cfg.get("voxel_connectivity", 6))
        samples, component_stats = filter_voxel_components(
            raw_samples,
            mode=component_filter_mode,
            connectivity=component_connectivity,
        )
        save_voxel_grid(
            samples,
            sample_dir / "generated_voxels.png",
            max_items=args.num_samples,
            labels=labels,
            label_names=voxel_label_names,
        )
        _save_voxel_samples(
            samples,
            labels,
            voxel_label_names,
            sample_dir / "generated_voxels.npz",
            raw_samples=raw_samples,
            component_stats=component_stats,
            component_filter_mode=component_filter_mode,
            component_connectivity=component_connectivity,
        )

    print(f"Saved samples to {Path(sample_dir).resolve()}.")


def _build_conditioning_labels(
    cfg: dict,
    num_samples: int,
    labels_arg: str | None,
    classes_arg: str | None,
    device: torch.device,
    *,
    voxel_metadata: dict[str, np.ndarray] | None = None,
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
        metadata = voxel_metadata
        if metadata is None:
            metadata = load_voxel_cache_metadata(cfg["dataset"]["cache_path"])
        validate_voxel_conditioning_metadata(metadata, num_labels)
        if classes_arg is not None:
            return _sample_subtypes_for_classes(metadata, classes_arg, num_samples, num_labels, device)
        if labels_arg is None or labels_arg.strip().lower() == "prior":
            prior = metadata.get("subtype_counts")
            if prior is not None:
                return _sample_labels_from_prior(prior, num_samples, num_labels, device)

    requested = _parse_labels_arg(labels_arg, num_labels)
    if num_samples < len(requested):
        raise ValueError(
            f"--num-samples={num_samples} cannot cover all {len(requested)} requested labels. "
            "Increase --num-samples or request fewer labels."
        )
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


def _resolve_sampling_label_names(
    checkpoint: dict,
    metadata: dict[str, np.ndarray],
    num_labels: int,
) -> list[str]:
    checkpoint_names = checkpoint.get("conditioning_label_names")
    if checkpoint_names is not None:
        checkpoint_names = [str(name) for name in checkpoint_names]
        if len(checkpoint_names) != num_labels:
            raise ValueError(
                f"Checkpoint contains {len(checkpoint_names)} conditioning label names, "
                f"but dataset.num_labels={num_labels}."
            )

    cache_names = resolve_voxel_label_names(metadata, num_labels)
    if checkpoint_names is not None and checkpoint_names != cache_names:
        raise ValueError(
            "The checkpoint and voxel cache use different conditioning-label mappings. "
            "Use the same subtype cache that was used for training."
        )
    return checkpoint_names if checkpoint_names is not None else cache_names


def _validate_checkpoint_sampling_config(cfg: dict, checkpoint: dict) -> None:
    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, dict):
        return

    checks = (
        ("dataset", "name"),
        ("dataset", "num_classes"),
        ("dataset", "num_labels"),
        ("dataset", "shape"),
        ("dataset", "conditional"),
        ("diffusion", "timesteps"),
        ("diffusion", "schedule"),
        ("diffusion", "beta_start"),
        ("diffusion", "beta_end"),
        ("diffusion", "transition"),
        ("model", "name"),
        ("model", "base_channels"),
        ("model", "channel_mults"),
        ("model", "num_res_blocks"),
        ("model", "num_blocks"),
        ("model", "dropout"),
    )
    mismatches: list[str] = []
    for section, key in checks:
        current = cfg.get(section, {}).get(key)
        trained = checkpoint_cfg.get(section, {}).get(key)
        if current != trained:
            mismatches.append(f"{section}.{key}: checkpoint={trained!r}, config={current!r}")

    current_cache = cfg.get("dataset", {}).get("cache_path")
    trained_cache = checkpoint_cfg.get("dataset", {}).get("cache_path")
    if current_cache is not None and trained_cache is not None and current_cache != trained_cache:
        mismatches.append(
            f"dataset.cache_path: checkpoint={trained_cache!r}, config={current_cache!r}"
        )

    if mismatches:
        details = "\n  ".join(mismatches)
        raise ValueError(
            "Sampling config does not match the checkpoint training config:\n  "
            f"{details}"
        )


def _save_voxel_samples(
    samples: torch.Tensor,
    labels: torch.Tensor | None,
    label_names: list[str] | None,
    path: Path,
    *,
    raw_samples: torch.Tensor | None = None,
    component_stats: list[ComponentFilterStats] | None = None,
    component_filter_mode: str | None = None,
    component_connectivity: int | None = None,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "samples": samples.numpy().astype(np.uint8),
    }
    if raw_samples is not None:
        if raw_samples.shape != samples.shape:
            raise ValueError("raw_samples and filtered samples must have the same shape.")
        arrays["raw_samples"] = raw_samples.numpy().astype(np.uint8)
    if component_filter_mode is not None:
        arrays["component_filter_mode"] = np.asarray(component_filter_mode, dtype="U")
    if component_connectivity is not None:
        arrays["component_connectivity"] = np.asarray(component_connectivity, dtype=np.int64)
    if component_stats is not None:
        if len(component_stats) != samples.shape[0]:
            raise ValueError("component_stats must contain one entry per generated sample.")
        arrays.update(
            component_counts=np.asarray([stat.components for stat in component_stats], dtype=np.int64),
            original_voxel_counts=np.asarray(
                [stat.original_voxels for stat in component_stats], dtype=np.int64
            ),
            kept_voxel_counts=np.asarray(
                [stat.kept_voxels for stat in component_stats], dtype=np.int64
            ),
            removed_voxel_counts=np.asarray(
                [stat.removed_voxels for stat in component_stats], dtype=np.int64
            ),
        )
    if labels is not None:
        if label_names is None:
            raise ValueError("Conditional voxel samples require human-readable label names.")
        labels_cpu = labels.detach().cpu().reshape(-1)
        if labels_cpu.shape[0] != samples.shape[0]:
            raise ValueError(
                f"Generated {samples.shape[0]} samples but received {labels_cpu.shape[0]} labels."
            )
        sample_names = [label_names[int(label)] for label in labels_cpu.tolist()]
        arrays.update(
            labels=labels_cpu.numpy().astype(np.int64),
            sample_label_names=np.asarray(sample_names, dtype="U"),
            conditioning_label_names=np.asarray(label_names, dtype="U"),
        )
        print("Generated conditioning labels:")
        for index, (label, name) in enumerate(zip(labels_cpu.tolist(), sample_names)):
            cleanup = ""
            if component_stats is not None:
                stat = component_stats[index]
                cleanup = (
                    f", components={stat.components}, removed_voxels={stat.removed_voxels}"
                )
            print(f"  sample {index:03d}: {label} -> {name}{cleanup}")
    np.savez_compressed(path, **arrays)


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
