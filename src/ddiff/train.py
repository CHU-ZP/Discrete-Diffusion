from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ddiff.data.modelnet_voxel import resolve_voxel_label_names
from ddiff.data.registry import build_dataset, collate_samples
from ddiff.diffusion.categorical import CategoricalDiffusion
from ddiff.models.registry import build_model
from ddiff.postprocessing.voxels import filter_voxel_components
from ddiff.utils.config import ensure_dir, load_config, resolve_device
from ddiff.utils.seed import set_seed
from ddiff.visualization.images import save_forward_chain, save_image_grid, save_reverse_chain
from ddiff.visualization.voxels import save_voxel_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a categorical diffusion model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--steps", type=int, default=None, help="Optional training step override.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.device is not None:
        cfg["train"]["device"] = args.device

    train(cfg)


def train(cfg: dict[str, Any]) -> None:
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(cfg["train"].get("device", "auto"))
    run_dir = ensure_dir(cfg["output"]["run_dir"])
    sample_dir = ensure_dir(cfg["output"]["sample_dir"])

    dataset = build_dataset(cfg, split="train")
    _validate_dataset_conditioning(cfg, dataset)
    conditioning_label_names = _conditioning_label_names(cfg, dataset)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        drop_last=True,
        collate_fn=collate_samples,
    )
    batches = cycle(loader)

    model = build_model(cfg).to(device)
    diffusion = CategoricalDiffusion.from_config(cfg, device=device)
    token_loss_weights = _build_token_loss_weights(cfg, dataset, device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    print(f"Training on {device} with {len(dataset)} samples.")
    _save_initial_visuals(
        cfg,
        dataset,
        diffusion,
        sample_dir,
        device,
        conditioning_label_names=conditioning_label_names,
    )

    total_steps = int(cfg["train"]["steps"])
    log_every = int(cfg["train"].get("log_every", 100))
    sample_every = int(cfg["train"].get("sample_every", 500))
    if sample_every <= 0:
        raise ValueError("train.sample_every must be positive.")
    conditional = bool(cfg["dataset"].get("conditional", False))
    spatial_shape = tuple(cfg["dataset"]["shape"])

    progress = tqdm(range(1, total_steps + 1), desc="train", dynamic_ncols=True)
    last_loss = None
    for step in progress:
        batch = next(batches)
        x0 = batch["x"].to(device=device, dtype=torch.long)
        y = batch["y"].to(device=device, dtype=torch.long) if conditional and batch["y"] is not None else None

        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.training_loss(model, x0, y, class_weights=token_loss_weights)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        if step % log_every == 0 or step == 1:
            progress.set_postfix(loss=f"{last_loss:.4f}")
            print(f"step {step:06d} loss {last_loss:.4f}")

        if step % sample_every == 0:
            _save_checkpoint(
                model,
                optimizer,
                cfg,
                step,
                run_dir / "latest.pt",
                conditioning_label_names=conditioning_label_names,
            )
            _save_checkpoint(
                model,
                optimizer,
                cfg,
                step,
                run_dir / f"step_{step:06d}.pt",
                conditioning_label_names=conditioning_label_names,
            )
            _save_samples(
                cfg,
                model,
                diffusion,
                sample_dir,
                spatial_shape,
                device,
                step=step,
                conditioning_label_names=conditioning_label_names,
            )

    _save_checkpoint(
        model,
        optimizer,
        cfg,
        total_steps,
        run_dir / "latest.pt",
        conditioning_label_names=conditioning_label_names,
    )
    if total_steps % sample_every != 0:
        _save_samples(
            cfg,
            model,
            diffusion,
            sample_dir,
            spatial_shape,
            device,
            step=total_steps,
            conditioning_label_names=conditioning_label_names,
        )
    if last_loss is not None:
        print(f"Finished training at step {total_steps} with loss {last_loss:.4f}.")


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    step: int,
    path: Path,
    *,
    conditioning_label_names: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "step": step,
    }
    if conditioning_label_names is not None:
        checkpoint["conditioning_label_names"] = list(conditioning_label_names)
    torch.save(checkpoint, path)


@torch.no_grad()
def _save_initial_visuals(
    cfg: dict[str, Any],
    dataset,
    diffusion: CategoricalDiffusion,
    sample_dir: Path,
    device: torch.device,
    *,
    conditioning_label_names: list[str] | None = None,
) -> None:
    count = min(64, len(dataset))
    examples = [dataset[idx] for idx in range(count)]
    x = torch.stack([example["x"] for example in examples]).long()
    if _is_image_dataset(cfg):
        labels = _example_labels(examples) if bool(cfg["dataset"].get("conditional", False)) else None
        value_range = _sample_value_range(cfg)
        save_image_grid(x, sample_dir / "real_samples.png", nrow=8, labels=labels, value_range=value_range)
        x0 = x[:1].to(device)
        steps = torch.linspace(0, diffusion.timesteps, 8).round().long().tolist()
        chain: dict[int, torch.Tensor] = {}
        for step in steps:
            if step == 0:
                chain[0] = x0.cpu()
            else:
                t = torch.full((1,), int(step), device=device, dtype=torch.long)
                chain[int(step)] = diffusion.q_sample(x0, t).cpu()
        save_forward_chain(chain, sample_dir / "forward_chain.png", value_range=value_range)
    elif cfg["dataset"]["name"] == "modelnet10_voxel":
        labels = _example_labels(examples) if bool(cfg["dataset"].get("conditional", False)) else None
        save_voxel_grid(
            x,
            sample_dir / "real_voxels.png",
            labels=labels,
            label_names=conditioning_label_names,
        )


@torch.no_grad()
def _save_samples(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    diffusion: CategoricalDiffusion,
    sample_dir: Path,
    spatial_shape: tuple[int, ...],
    device: torch.device,
    *,
    step: int,
    conditioning_label_names: list[str] | None = None,
) -> None:
    name = cfg["dataset"]["name"]
    if _is_image_dataset(cfg):
        labels = _balanced_conditioning_labels(cfg, batch_size=64, device=device)
        samples, chain = diffusion.sample(
            model,
            spatial_shape,
            y=labels,
            batch_size=64,
            return_chain=True,
            device=device,
        )
        value_range = _sample_value_range(cfg)
        save_image_grid(samples.cpu(), sample_dir / "generated_samples.png", nrow=8, labels=labels, value_range=value_range)
        save_image_grid(
            samples.cpu(),
            sample_dir / f"generated_samples_step_{step:06d}.png",
            nrow=8,
            labels=labels,
            value_range=value_range,
        )
        save_reverse_chain(chain, sample_dir / "reverse_chain.png", value_range=value_range)
    elif name == "modelnet10_voxel":
        batch_size = int(cfg["train"].get("sample_batch_size", 4))
        labels = _balanced_conditioning_labels(cfg, batch_size=batch_size, device=device)
        samples = diffusion.sample(
            model,
            spatial_shape,
            y=labels,
            batch_size=batch_size,
            device=device,
        )
        sample_cfg = cfg.get("sample", {})
        samples, component_stats = filter_voxel_components(
            samples.cpu(),
            mode=str(sample_cfg.get("voxel_component_filter", "largest")),
            connectivity=int(sample_cfg.get("voxel_connectivity", 6)),
        )
        removed_voxels = sum(stat.removed_voxels for stat in component_stats)
        affected_samples = sum(stat.removed_voxels > 0 for stat in component_stats)
        print(
            f"Voxel component filter removed {removed_voxels} voxels "
            f"from {affected_samples}/{batch_size} preview samples."
        )
        save_voxel_grid(
            samples,
            sample_dir / f"generated_voxels_step_{step:06d}.png",
            max_items=batch_size,
            labels=labels,
            label_names=conditioning_label_names,
        )


def _example_labels(examples: list[dict[str, torch.Tensor | None]]) -> torch.Tensor | None:
    labels = [example["y"] for example in examples]
    if any(label is None for label in labels):
        return None
    return torch.stack([label for label in labels if label is not None]).long()


def _validate_dataset_conditioning(cfg: dict[str, Any], dataset) -> None:
    if not bool(cfg["dataset"].get("conditional", False)):
        return

    num_labels = int(cfg["dataset"].get("num_labels", 0))
    if num_labels <= 0:
        raise ValueError("Conditional training requires dataset.num_labels > 0.")
    labels = getattr(dataset, "y", None)
    if labels is None:
        raise ValueError("Conditional training requires a label for every dataset sample.")

    labels = torch.as_tensor(labels).reshape(-1).long()
    if labels.numel() != len(dataset):
        raise ValueError(f"Dataset has {len(dataset)} samples but {labels.numel()} labels.")
    if labels.numel() == 0:
        raise ValueError("Cannot train on an empty conditional dataset.")
    if int(labels.min()) < 0 or int(labels.max()) >= num_labels:
        raise ValueError(
            f"Dataset labels must be in [0, {num_labels - 1}], got "
            f"[{int(labels.min())}, {int(labels.max())}]."
        )

    present = set(int(label) for label in labels.unique().tolist())
    missing = sorted(set(range(num_labels)) - present)
    if missing:
        raise ValueError(
            f"The training split has no samples for conditioning labels {missing}. "
            "Sampling these labels would use untrained conditions."
        )


def _conditioning_label_names(cfg: dict[str, Any], dataset) -> list[str] | None:
    if cfg["dataset"]["name"] != "modelnet10_voxel":
        return None
    if not bool(cfg["dataset"].get("conditional", False)):
        return None

    num_labels = int(cfg["dataset"]["num_labels"])
    metadata = getattr(dataset, "metadata", {})
    return resolve_voxel_label_names(metadata, num_labels)


def _build_token_loss_weights(
    cfg: dict[str, Any],
    dataset,
    device: torch.device,
) -> torch.Tensor | None:
    spec = cfg["train"].get("token_loss_weights")
    if spec is None or str(spec).lower() in {"none", "false", "off"}:
        return None

    num_classes = int(cfg["dataset"]["num_classes"])
    if isinstance(spec, str):
        if spec.lower() != "auto":
            raise ValueError("train.token_loss_weights must be 'auto', a list of weights, or omitted.")
        counts = _count_dataset_tokens(dataset, num_classes)
        if torch.any(counts == 0):
            raise ValueError(f"Cannot build automatic token weights with empty token classes: {counts.tolist()}.")
        weights = counts.max().float() / counts.float()
        max_weight = cfg["train"].get("max_token_loss_weight")
        if max_weight is not None:
            weights = weights.clamp(max=float(max_weight))
    else:
        weights = torch.tensor(spec, dtype=torch.float32)
        if weights.numel() != num_classes:
            raise ValueError(
                f"train.token_loss_weights must have {num_classes} values, got {weights.numel()}."
            )

    weights = weights.to(device=device, dtype=torch.float32)
    print(f"Using token loss weights: {[round(float(weight), 4) for weight in weights.detach().cpu()]}")
    return weights


def _count_dataset_tokens(dataset, num_classes: int) -> torch.Tensor:
    if hasattr(dataset, "x"):
        x = dataset.x
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        return torch.bincount(x.reshape(-1).long(), minlength=num_classes)[:num_classes].cpu()

    counts = torch.zeros(num_classes, dtype=torch.long)
    for idx in range(len(dataset)):
        x = dataset[idx]["x"]
        counts += torch.bincount(x.reshape(-1).long(), minlength=num_classes)[:num_classes].cpu()
    return counts


def _balanced_conditioning_labels(
    cfg: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not bool(cfg["dataset"].get("conditional", False)):
        return None

    num_labels = int(cfg["dataset"].get("num_labels", 0))
    if num_labels <= 0:
        raise ValueError("Conditional sampling requires dataset.num_labels > 0.")

    labels = torch.arange(batch_size, device=device, dtype=torch.long)
    return labels % num_labels


def _is_image_dataset(cfg: dict[str, Any]) -> bool:
    return cfg["dataset"]["name"] == "mnist"


def _sample_value_range(cfg: dict[str, Any]) -> tuple[int, int]:
    return 0, int(cfg["dataset"]["num_classes"]) - 1


if __name__ == "__main__":
    main()
