from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
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
from ddiff.postprocessing.voxels import ComponentFilterStats, filter_voxel_components
from ddiff.utils.config import ensure_dir, load_config, resolve_device
from ddiff.utils.seed import set_seed
from ddiff.visualization.images import save_forward_chain, save_image_grid, save_reverse_chain
from ddiff.visualization.voxels import save_voxel_grid


@dataclass
class VoxelSampleBatch:
    raw: torch.Tensor
    filtered: torch.Tensor
    labels: torch.Tensor
    component_stats: list[ComponentFilterStats]


@dataclass
class VoxelQualityReference:
    voxels_by_label: dict[int, torch.Tensor]
    occupancy_by_label: torch.Tensor
    surface_by_label: torch.Tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a categorical diffusion model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--steps", type=int, default=None, help="Optional training step override.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume raw/EMA/optimizer state from a training checkpoint.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable generation-quality evaluation, useful for short smoke runs.",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Skip periodic reverse-diffusion previews, useful for short smoke runs.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
        if int(cfg["train"].get("warmup_steps", 0)) >= args.steps:
            cfg["train"]["warmup_steps"] = max(0, args.steps // 10)
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.no_eval:
        cfg.setdefault("evaluation", {})["enabled"] = False
    if args.no_samples:
        cfg["train"]["save_samples"] = False

    train(cfg, resume_path=args.resume)


def train(
    cfg: dict[str, Any],
    *,
    resume_path: str | Path | None = None,
) -> None:
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(cfg["train"].get("device", "auto"))
    run_dir = ensure_dir(cfg["output"]["run_dir"])
    sample_dir = ensure_dir(cfg["output"]["sample_dir"])

    dataset = build_dataset(cfg, split="train")
    _validate_dataset_conditioning(cfg, dataset)
    conditioning_label_names = _conditioning_label_names(cfg, dataset)
    num_workers = int(cfg["train"].get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_samples,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    batches = _repeat_loader(loader)

    model = build_model(cfg).to(device)
    ema_model = _build_ema_model(model)
    diffusion = CategoricalDiffusion.from_config(cfg, device=device)
    token_loss_weights = _build_token_loss_weights(cfg, dataset, device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    total_steps = int(cfg["train"]["steps"])
    ema_decay = float(cfg["train"].get("ema_decay", 0.0))
    ema_start_step = int(cfg["train"].get("ema_start_step", 0))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("train.ema_decay must be in [0, 1).")
    if ema_start_step < 0 or ema_start_step >= total_steps:
        raise ValueError("train.ema_start_step must be in [0, train.steps).")
    _validate_learning_rate_config(cfg, total_steps)

    start_step = 0
    best_quality_score = math.inf
    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        _load_training_checkpoint(
            cfg,
            resume_checkpoint,
            model,
            ema_model,
            optimizer,
            conditioning_label_names=conditioning_label_names,
        )
        start_step = int(resume_checkpoint.get("step", 0))
        best_quality_score = float(resume_checkpoint.get("best_quality_score", math.inf))
        if start_step <= 0:
            raise ValueError("Resume checkpoint must contain a positive training step.")
        if start_step >= total_steps:
            raise ValueError(
                f"Resume checkpoint is at step {start_step}, but train.steps={total_steps}. "
                "Increase --steps or use a later target in the config."
            )

    quality_cfg = cfg.get("evaluation", {})
    quality_enabled = bool(quality_cfg.get("enabled", False))
    quality_reference: VoxelQualityReference | None = None
    quality_every = int(quality_cfg.get("every", cfg["train"].get("sample_every", 500)))
    if quality_enabled:
        if cfg["dataset"]["name"] != "modelnet10_voxel":
            raise ValueError("Generation-quality checkpoint selection currently requires ModelNet voxels.")
        if quality_every <= 0:
            raise ValueError("evaluation.every must be positive.")
        num_labels = int(cfg["dataset"]["num_labels"])
        quality_sample_count = int(cfg["train"].get("sample_batch_size", 4))
        if quality_sample_count < num_labels or quality_sample_count % num_labels != 0:
            raise ValueError(
                "With evaluation enabled, train.sample_batch_size must be a positive "
                "multiple of dataset.num_labels so every label is evaluated equally."
            )
        quality_weights = quality_cfg.get("weights", {})
        if any(float(value) < 0.0 for value in quality_weights.values()):
            raise ValueError("evaluation.weights values must be non-negative.")
        validation_dataset = build_dataset(cfg, split="test")
        _validate_dataset_conditioning(cfg, validation_dataset)
        quality_reference = _build_voxel_quality_reference(
            validation_dataset,
            num_labels=num_labels,
            max_items_per_label=int(quality_cfg.get("reference_samples_per_label", 16)),
        )

    print(f"Training on {device} with {len(dataset)} samples.")
    if resume_checkpoint is None:
        _save_initial_visuals(
            cfg,
            dataset,
            diffusion,
            sample_dir,
            device,
            conditioning_label_names=conditioning_label_names,
        )
    else:
        _restore_checkpoint_rng(resume_checkpoint, device)
        print(f"Resumed from {Path(resume_path).resolve()} at step {start_step}.")

    log_every = int(cfg["train"].get("log_every", 100))
    sample_every = int(cfg["train"].get("sample_every", 500))
    if sample_every <= 0:
        raise ValueError("train.sample_every must be positive.")
    sample_enabled = bool(cfg["train"].get("save_samples", True))
    conditional = bool(cfg["dataset"].get("conditional", False))
    spatial_shape = tuple(cfg["dataset"]["shape"])
    if quality_enabled and not sample_enabled:
        raise ValueError("Generation-quality evaluation requires train.save_samples=true.")
    if quality_enabled and quality_every % sample_every != 0:
        raise ValueError("evaluation.every must be a multiple of train.sample_every.")

    print(
        f"Training schedule: steps={total_steps}, base_lr={float(cfg['train']['lr']):.3g}, "
        f"min_lr={float(cfg['train'].get('min_lr', cfg['train']['lr'])):.3g}, "
        f"warmup={int(cfg['train'].get('warmup_steps', 0))}, "
        f"scheduler={cfg['train'].get('lr_scheduler', 'constant')}, "
        f"ema_decay={ema_decay}, ema_start={ema_start_step}."
    )

    progress = tqdm(
        range(start_step + 1, total_steps + 1),
        desc="train",
        dynamic_ncols=True,
    )
    last_loss = None
    for step in progress:
        learning_rate = _learning_rate_for_step(cfg, step, total_steps)
        _set_optimizer_learning_rate(optimizer, learning_rate)
        batch = next(batches)
        x0 = batch["x"].to(device=device, dtype=torch.long, non_blocking=True)
        y = (
            batch["y"].to(device=device, dtype=torch.long, non_blocking=True)
            if conditional and batch["y"] is not None
            else None
        )

        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.training_loss(model, x0, y, class_weights=token_loss_weights)
        loss.backward()
        optimizer.step()
        _update_ema_model(
            ema_model,
            model,
            ema_decay,
            step=step,
            start_step=ema_start_step,
        )

        last_loss = float(loss.item())
        if step % log_every == 0 or step == 1:
            progress.set_postfix(loss=f"{last_loss:.4f}", lr=f"{learning_rate:.2e}")
            print(f"step {step:06d} loss {last_loss:.4f} lr {learning_rate:.6g}")

        if sample_enabled and step % sample_every == 0:
            sampled_batch = _save_samples(
                cfg,
                ema_model,
                diffusion,
                sample_dir,
                spatial_shape,
                device,
                step=step,
                conditioning_label_names=conditioning_label_names,
                sampling_seed=int(quality_cfg.get("seed", 12345)),
            )
            quality_metrics = None
            is_best = False
            if quality_enabled and step % quality_every == 0:
                if sampled_batch is None or quality_reference is None:
                    raise RuntimeError("Voxel quality evaluation requires a generated voxel batch.")
                quality_metrics = _evaluate_voxel_generation_quality(
                    sampled_batch,
                    quality_reference,
                    quality_cfg,
                )
                score = quality_metrics["score"]
                is_best = score < best_quality_score
                if is_best:
                    best_quality_score = score
                print(
                    f"generation quality step {step:06d}: "
                    f"score={score:.6f}, nearest_iou={quality_metrics['nearest_iou']:.4f}, "
                    f"occupancy_error={quality_metrics['occupancy_error']:.4f}, "
                    f"surface_error={quality_metrics['surface_error']:.4f}, "
                    f"fragment_ratio={quality_metrics['fragment_ratio']:.4f}"
                )
            _save_checkpoint(
                model,
                optimizer,
                cfg,
                step,
                run_dir / "latest.pt",
                conditioning_label_names=conditioning_label_names,
                ema_model=ema_model,
                best_quality_score=best_quality_score,
                quality_metrics=quality_metrics,
                learning_rate=learning_rate,
            )
            _save_checkpoint(
                model,
                optimizer,
                cfg,
                step,
                run_dir / f"step_{step:06d}.pt",
                conditioning_label_names=conditioning_label_names,
                ema_model=ema_model,
                best_quality_score=best_quality_score,
                quality_metrics=quality_metrics,
                learning_rate=learning_rate,
            )
            if is_best:
                _save_checkpoint(
                    model,
                    optimizer,
                    cfg,
                    step,
                    run_dir / "best.pt",
                    conditioning_label_names=conditioning_label_names,
                    ema_model=ema_model,
                    best_quality_score=best_quality_score,
                    quality_metrics=quality_metrics,
                    learning_rate=learning_rate,
                )
                print(f"Saved new best checkpoint: score={best_quality_score:.6f}.")

    if not sample_enabled or total_steps % sample_every != 0:
        learning_rate = _learning_rate_for_step(cfg, total_steps, total_steps)
        sampled_batch = None
        if sample_enabled:
            sampled_batch = _save_samples(
                cfg,
                ema_model,
                diffusion,
                sample_dir,
                spatial_shape,
                device,
                step=total_steps,
                conditioning_label_names=conditioning_label_names,
                sampling_seed=int(quality_cfg.get("seed", 12345)),
            )
        quality_metrics = None
        is_best = False
        if quality_enabled:
            if sampled_batch is None or quality_reference is None:
                raise RuntimeError("Voxel quality evaluation requires a generated voxel batch.")
            quality_metrics = _evaluate_voxel_generation_quality(
                sampled_batch,
                quality_reference,
                quality_cfg,
            )
            is_best = quality_metrics["score"] < best_quality_score
            if is_best:
                best_quality_score = quality_metrics["score"]
        _save_checkpoint(
            model,
            optimizer,
            cfg,
            total_steps,
            run_dir / "latest.pt",
            conditioning_label_names=conditioning_label_names,
            ema_model=ema_model,
            best_quality_score=best_quality_score,
            quality_metrics=quality_metrics,
            learning_rate=learning_rate,
        )
        if is_best:
            _save_checkpoint(
                model,
                optimizer,
                cfg,
                total_steps,
                run_dir / "best.pt",
                conditioning_label_names=conditioning_label_names,
                ema_model=ema_model,
                best_quality_score=best_quality_score,
                quality_metrics=quality_metrics,
                learning_rate=learning_rate,
            )
    if last_loss is not None:
        best_message = (
            f", best generation score {best_quality_score:.6f}"
            if math.isfinite(best_quality_score)
            else ""
        )
        print(f"Finished training at step {total_steps} with loss {last_loss:.4f}{best_message}.")


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    step: int,
    path: Path,
    *,
    conditioning_label_names: list[str] | None = None,
    ema_model: torch.nn.Module | None = None,
    best_quality_score: float | None = None,
    quality_metrics: dict[str, float] | None = None,
    learning_rate: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "step": step,
        "torch_rng_state": torch.get_rng_state(),
    }
    model_device = next(model.parameters()).device
    if model_device.type == "cuda":
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state(model_device)
    if ema_model is not None:
        checkpoint["ema_model"] = ema_model.state_dict()
    if conditioning_label_names is not None:
        checkpoint["conditioning_label_names"] = list(conditioning_label_names)
    if best_quality_score is not None and math.isfinite(best_quality_score):
        checkpoint["best_quality_score"] = float(best_quality_score)
    if quality_metrics is not None:
        checkpoint["quality_metrics"] = dict(quality_metrics)
    if learning_rate is not None:
        checkpoint["learning_rate"] = float(learning_rate)
    torch.save(checkpoint, path)


def _load_training_checkpoint(
    cfg: dict[str, Any],
    checkpoint: dict,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    conditioning_label_names: list[str] | None,
) -> None:
    # Reuse the strict dataset/diffusion/model compatibility checks used by
    # inference. Training-only values such as target steps may change.
    from ddiff.sample import _validate_checkpoint_sampling_config

    _validate_checkpoint_sampling_config(cfg, checkpoint)
    if "model" not in checkpoint or "optimizer" not in checkpoint:
        raise KeyError("Resume checkpoint must contain model and optimizer states.")
    model.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    optimizer.load_state_dict(checkpoint["optimizer"])

    checkpoint_names = checkpoint.get("conditioning_label_names")
    if checkpoint_names is not None and conditioning_label_names is not None:
        if list(checkpoint_names) != list(conditioning_label_names):
            raise ValueError(
                "Resume checkpoint and current dataset use different conditioning-label mappings."
            )


def _restore_checkpoint_rng(checkpoint: dict, device: torch.device) -> None:
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device=device)


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
    sampling_seed: int = 12345,
) -> VoxelSampleBatch | None:
    name = cfg["dataset"]["name"]
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(sampling_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sampling_seed)

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
            save_image_grid(
                samples.cpu(),
                sample_dir / "generated_samples.png",
                nrow=8,
                labels=labels,
                value_range=value_range,
            )
            save_image_grid(
                samples.cpu(),
                sample_dir / f"generated_samples_step_{step:06d}.png",
                nrow=8,
                labels=labels,
                value_range=value_range,
            )
            save_reverse_chain(chain, sample_dir / "reverse_chain.png", value_range=value_range)
            return None

        if name == "modelnet10_voxel":
            batch_size = int(cfg["train"].get("sample_batch_size", 4))
            micro_batch_size = int(cfg["train"].get("sample_micro_batch_size", batch_size))
            if batch_size <= 0 or micro_batch_size <= 0:
                raise ValueError("train.sample_batch_size and sample_micro_batch_size must be positive.")
            labels = _balanced_conditioning_labels(cfg, batch_size=batch_size, device=device)
            if labels is None:
                raise RuntimeError("Conditional voxel previews require labels.")
            raw_samples = _sample_voxels_in_batches(
                model,
                diffusion,
                spatial_shape,
                labels,
                micro_batch_size,
                device,
            )
            sample_cfg = cfg.get("sample", {})
            samples, component_stats = filter_voxel_components(
                raw_samples,
                mode=str(sample_cfg.get("voxel_component_filter", "largest")),
                connectivity=int(sample_cfg.get("voxel_connectivity", 6)),
            )
            removed_voxels = sum(stat.removed_voxels for stat in component_stats)
            affected_samples = sum(stat.removed_voxels > 0 for stat in component_stats)
            print(
                f"Voxel component filter removed {removed_voxels} voxels "
                f"from {affected_samples}/{batch_size} preview samples."
            )
            labels_cpu = labels.detach().cpu()
            save_voxel_grid(
                samples,
                sample_dir / f"generated_voxels_step_{step:06d}.png",
                max_items=batch_size,
                labels=labels_cpu,
                label_names=conditioning_label_names,
            )
            return VoxelSampleBatch(
                raw=raw_samples,
                filtered=samples,
                labels=labels_cpu,
                component_stats=component_stats,
            )

    return None


def _sample_voxels_in_batches(
    model: torch.nn.Module,
    diffusion: CategoricalDiffusion,
    spatial_shape: tuple[int, ...],
    labels: torch.Tensor,
    micro_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    for start in range(0, labels.shape[0], micro_batch_size):
        batch_labels = labels[start : start + micro_batch_size]
        samples = diffusion.sample(
            model,
            spatial_shape,
            y=batch_labels,
            batch_size=batch_labels.shape[0],
            device=device,
        )
        batches.append(samples.cpu())
    return torch.cat(batches, dim=0)


def _build_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    ema_model.requires_grad_(False)
    return ema_model


def _repeat_loader(loader: DataLoader):
    """Repeat a loader without caching a full epoch of large voxel batches."""

    while True:
        yield from loader


@torch.no_grad()
def _update_ema_model(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float,
    *,
    step: int | None = None,
    start_step: int = 0,
) -> None:
    effective_decay = 0.0 if step is not None and step <= start_step else decay
    ema_parameters = dict(ema_model.named_parameters())
    model_parameters = dict(model.named_parameters())
    if ema_parameters.keys() != model_parameters.keys():
        raise ValueError("EMA and online model parameters do not match.")
    for name, ema_parameter in ema_parameters.items():
        ema_parameter.lerp_(model_parameters[name].detach(), 1.0 - effective_decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    if ema_buffers.keys() != model_buffers.keys():
        raise ValueError("EMA and online model buffers do not match.")
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name].detach())


def _validate_learning_rate_config(cfg: dict[str, Any], total_steps: int) -> None:
    train_cfg = cfg["train"]
    scheduler = str(train_cfg.get("lr_scheduler", "constant")).lower()
    if scheduler not in {"constant", "cosine"}:
        raise ValueError("train.lr_scheduler must be 'constant' or 'cosine'.")
    base_lr = float(train_cfg["lr"])
    min_lr = float(train_cfg.get("min_lr", base_lr))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    warmup_start_factor = float(train_cfg.get("warmup_start_factor", 0.05))
    if total_steps <= 0:
        raise ValueError("train.steps must be positive.")
    if base_lr <= 0.0 or min_lr < 0.0 or min_lr > base_lr:
        raise ValueError("Learning rates must satisfy 0 <= min_lr <= lr and lr > 0.")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("train.warmup_steps must be in [0, train.steps).")
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("train.warmup_start_factor must be in (0, 1].")


def _learning_rate_for_step(
    cfg: dict[str, Any],
    step: int,
    total_steps: int,
) -> float:
    train_cfg = cfg["train"]
    base_lr = float(train_cfg["lr"])
    min_lr = float(train_cfg.get("min_lr", base_lr))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    warmup_start_factor = float(train_cfg.get("warmup_start_factor", 0.05))
    scheduler = str(train_cfg.get("lr_scheduler", "constant")).lower()

    if warmup_steps > 0 and step <= warmup_steps:
        progress = step / warmup_steps
        return base_lr * (warmup_start_factor + (1.0 - warmup_start_factor) * progress)
    if scheduler == "constant":
        return base_lr

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _build_voxel_quality_reference(
    dataset,
    *,
    num_labels: int,
    max_items_per_label: int,
) -> VoxelQualityReference:
    if max_items_per_label <= 0:
        raise ValueError("evaluation.reference_samples_per_label must be positive.")
    if not hasattr(dataset, "x") or getattr(dataset, "y", None) is None:
        raise ValueError("Voxel generation-quality evaluation requires labeled validation tensors.")

    x = torch.as_tensor(dataset.x)
    y = torch.as_tensor(dataset.y).reshape(-1).long()
    voxels_by_label: dict[int, torch.Tensor] = {}
    occupancy_by_label = torch.empty(num_labels, dtype=torch.float32)
    surface_by_label = torch.empty(num_labels, dtype=torch.float32)
    for label in range(num_labels):
        indices = torch.nonzero(y == label, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError(f"Validation split has no reference voxels for label {label}.")
        selected = indices[:max_items_per_label]
        references = x.index_select(0, selected).to(dtype=torch.uint8, device="cpu")
        occupancy, surface = _voxel_shape_statistics(references)
        voxels_by_label[label] = references
        occupancy_by_label[label] = occupancy.mean()
        surface_by_label[label] = surface.mean()

    return VoxelQualityReference(
        voxels_by_label=voxels_by_label,
        occupancy_by_label=occupancy_by_label,
        surface_by_label=surface_by_label,
    )


def _evaluate_voxel_generation_quality(
    samples: VoxelSampleBatch,
    reference: VoxelQualityReference,
    quality_cfg: dict[str, Any],
) -> dict[str, float]:
    generated = samples.filtered.to(dtype=torch.uint8, device="cpu")
    labels = samples.labels.to(dtype=torch.long, device="cpu").reshape(-1)
    if generated.shape[0] != labels.shape[0]:
        raise ValueError("Generated samples and quality-evaluation labels must have equal length.")

    occupancy, surface = _voxel_shape_statistics(generated)
    target_occupancy = reference.occupancy_by_label.index_select(0, labels)
    target_surface = reference.surface_by_label.index_select(0, labels)
    occupancy_error = (
        (occupancy - target_occupancy).abs() / target_occupancy.clamp_min(1e-4)
    ).mean()
    surface_error = (
        (surface - target_surface).abs() / target_surface.clamp_min(1e-4)
    ).mean()

    nearest_ious: list[float] = []
    for sample, label in zip(generated, labels.tolist()):
        references = reference.voxels_by_label[int(label)].bool()
        occupied = sample.bool().unsqueeze(0)
        intersection = (references & occupied).flatten(1).sum(dim=1).float()
        union = (references | occupied).flatten(1).sum(dim=1).float()
        iou = intersection / union.clamp_min(1.0)
        nearest_ious.append(float(iou.max().item()))
    nearest_iou = sum(nearest_ious) / max(len(nearest_ious), 1)

    removed = sum(stat.removed_voxels for stat in samples.component_stats)
    original = sum(stat.original_voxels for stat in samples.component_stats)
    fragment_ratio = removed / max(original, 1)

    weights = quality_cfg.get("weights", {})
    iou_weight = float(weights.get("nearest_iou", 1.0))
    occupancy_weight = float(weights.get("occupancy", 0.25))
    surface_weight = float(weights.get("surface", 0.5))
    fragment_weight = float(weights.get("fragments", 0.5))
    score = (
        iou_weight * (1.0 - nearest_iou)
        + occupancy_weight * float(occupancy_error.item())
        + surface_weight * float(surface_error.item())
        + fragment_weight * fragment_ratio
    )
    return {
        "score": float(score),
        "nearest_iou": float(nearest_iou),
        "occupancy_error": float(occupancy_error.item()),
        "surface_error": float(surface_error.item()),
        "fragment_ratio": float(fragment_ratio),
    }


def _voxel_shape_statistics(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if samples.ndim != 4:
        raise ValueError(f"Expected voxel samples shaped [B, D, H, W], got {tuple(samples.shape)}.")
    occupied = samples.bool()
    counts = occupied.flatten(1).sum(dim=1).float()
    total = occupied[0].numel()
    occupancy = counts / total

    faces = torch.zeros(samples.shape[0], dtype=torch.float32, device=samples.device)
    for axis in range(1, 4):
        first = occupied.select(axis, 0).flatten(1).sum(dim=1)
        last = occupied.select(axis, occupied.shape[axis] - 1).flatten(1).sum(dim=1)
        lower = occupied.narrow(axis, 0, occupied.shape[axis] - 1)
        upper = occupied.narrow(axis, 1, occupied.shape[axis] - 1)
        transitions = (lower != upper).flatten(1).sum(dim=1)
        faces += first.float() + last.float() + transitions.float()
    surface_per_voxel = faces / counts.clamp_min(1.0)
    return occupancy, surface_per_voxel


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
