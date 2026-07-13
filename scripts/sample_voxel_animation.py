#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from ddiff.data.modelnet_voxel import load_voxel_cache_metadata
from ddiff.diffusion.categorical import CategoricalDiffusion
from ddiff.models.registry import build_model
from ddiff.postprocessing.voxels import filter_voxel_components
from ddiff.sample import _resolve_sampling_label_names, _validate_checkpoint_sampling_config
from ddiff.utils.config import ensure_dir, load_config, resolve_device
from ddiff.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a voxel GIF showing reverse diffusion, the raw result, "
            "connected-component detection, and the filtered result."
        )
    )
    parser.add_argument("--config", default="configs/voxel_modelnet10.yaml")
    parser.add_argument(
        "--ckpt",
        default="runs/voxel_modelnet10_64_subtypes/latest.pt",
        help="Voxel diffusion checkpoint.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Conditioning subtype id or exact subtype name, for example 7 or bed_1.",
    )
    parser.add_argument("--output", default=None, help="Output GIF path.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth diffusion step.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.5,
        help="How long to hold the raw, detection, and filtered final frames.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20_000,
        help="Maximum rendered surface points per color and frame.",
    )
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=38.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--voxel-component-filter",
        choices=("largest", "none"),
        default=None,
    )
    parser.add_argument("--voxel-connectivity", type=int, choices=(6, 26), default=None)
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.hold_seconds < 0:
        raise ValueError("--hold-seconds must be non-negative.")
    if args.max_points <= 0:
        raise ValueError("--max-points must be positive.")

    cfg = load_config(args.config)
    if cfg["dataset"]["name"] != "modelnet10_voxel":
        raise ValueError("This animation script requires dataset.name=modelnet10_voxel.")
    if not bool(cfg["dataset"].get("conditional", False)):
        raise ValueError("This animation script requires a conditional voxel model.")
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.voxel_component_filter is not None:
        cfg.setdefault("sample", {})["voxel_component_filter"] = args.voxel_component_filter
    if args.voxel_connectivity is not None:
        cfg.setdefault("sample", {})["voxel_connectivity"] = args.voxel_connectivity

    seed = int(cfg.get("seed", 0)) if args.seed is None else args.seed
    set_seed(seed)
    device = resolve_device(cfg["train"].get("device", "auto"))

    checkpoint = torch.load(args.ckpt, map_location=device)
    _validate_checkpoint_sampling_config(cfg, checkpoint)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = CategoricalDiffusion.from_config(cfg, device=device)

    num_labels = int(cfg["dataset"]["num_labels"])
    metadata = load_voxel_cache_metadata(cfg["dataset"]["cache_path"])
    label_names = _resolve_sampling_label_names(checkpoint, metadata, num_labels)
    label_id = _resolve_label(args.label, label_names)
    label_name = label_names[label_id]

    chain_steps = list(range(diffusion.timesteps, -1, -args.frame_stride))
    if chain_steps[-1] != 0:
        chain_steps.append(0)

    print(
        f"Sampling subtype {label_id} -> {label_name!r} on {device}; "
        f"recording {len(chain_steps)} diffusion frames."
    )
    _, chain = diffusion.sample(
        model,
        tuple(cfg["dataset"]["shape"]),
        y=torch.tensor([label_id], device=device, dtype=torch.long),
        batch_size=1,
        return_chain=True,
        chain_steps=chain_steps,
        device=device,
    )

    raw_result = chain[0][0].long()
    sample_cfg = cfg.get("sample", {})
    filter_mode = str(sample_cfg.get("voxel_component_filter", "largest"))
    connectivity = int(sample_cfg.get("voxel_connectivity", 6))
    filtered_batch, stats = filter_voxel_components(
        raw_result.unsqueeze(0),
        mode=filter_mode,
        connectivity=connectivity,
    )
    filtered_result = filtered_batch[0]
    removed = raw_result.bool() & ~filtered_result.bool()
    stat = stats[0]

    output = _output_path(args.output, cfg, label_name)
    frame_duration_ms = max(1, round(1000.0 / args.fps))
    hold_duration_ms = max(frame_duration_ms, round(args.hold_seconds * 1000.0))
    frames: list[Image.Image] = []
    durations: list[int] = []

    ordered_steps = sorted(chain, reverse=True)
    for step in tqdm(ordered_steps, desc="render diffusion", dynamic_ncols=True):
        stage = "initial noise" if step == diffusion.timesteps else "reverse diffusion"
        if step == 0:
            stage = "raw generated result"
        frames.append(
            _render_frame(
                chain[step][0],
                title=f"{label_name} | {stage} | t={step}",
                max_points=args.max_points,
                elev=args.elev,
                azim=args.azim,
            )
        )
        durations.append(hold_duration_ms if step == 0 else frame_duration_ms)

    detection_title = (
        f"{label_name} | connected components={stat.components} | "
        f"red voxels to remove={stat.removed_voxels}"
    )
    frames.append(
        _render_frame(
            filtered_result,
            removed=removed,
            title=detection_title,
            max_points=args.max_points,
            elev=args.elev,
            azim=args.azim,
        )
    )
    durations.append(hold_duration_ms)
    frames.append(
        _render_frame(
            filtered_result,
            title=(
                f"{label_name} | filtered result | kept={stat.kept_voxels}, "
                f"removed={stat.removed_voxels}"
            ),
            max_points=args.max_points,
            elev=args.elev,
            azim=args.azim,
        )
    )
    durations.append(hold_duration_ms)

    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    print(f"Saved animation: {output.resolve()}")
    print(
        f"Component filter: mode={filter_mode}, connectivity={connectivity}, "
        f"components={stat.components}, removed_voxels={stat.removed_voxels}."
    )


def _resolve_label(value: str, label_names: list[str]) -> int:
    value = value.strip()
    if value.lstrip("-").isdigit():
        label = int(value)
        if not 0 <= label < len(label_names):
            raise ValueError(f"Label {label} is outside [0, {len(label_names) - 1}].")
        return label

    normalized = [name.casefold() for name in label_names]
    if value.casefold() not in normalized:
        raise ValueError(f"Unknown subtype {value!r}; available names are {label_names}.")
    return normalized.index(value.casefold())


def _output_path(output_arg: str | None, cfg: dict, label_name: str) -> Path:
    if output_arg is not None:
        output = Path(output_arg)
    else:
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in label_name)
        output = Path(cfg["output"]["sample_dir"]) / f"reverse_diffusion_{safe_name}.gif"
    ensure_dir(output.parent)
    return output


def _render_frame(
    voxels: torch.Tensor,
    *,
    title: str,
    max_points: int,
    elev: float,
    azim: float,
    removed: torch.Tensor | None = None,
) -> Image.Image:
    occupied = voxels.detach().cpu().numpy().astype(bool)
    if occupied.ndim != 3:
        raise ValueError(f"Expected one [D, H, W] voxel sample, got {occupied.shape}.")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _scatter_surface(ax, occupied, color="#2563eb", max_points=max_points, alpha=0.58)
    if removed is not None:
        removed_mask = removed.detach().cpu().numpy().astype(bool)
        _scatter_surface(ax, removed_mask, color="#ef4444", max_points=max_points, alpha=0.9)

    depth, height, width = occupied.shape
    ax.set_xlim(-1, width)
    ax.set_ylim(-1, height)
    ax.set_zlim(-1, depth)
    ax.set_box_aspect((width, height, depth))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=8)
    fig.tight_layout(pad=0.4)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        rendered = image.convert("RGB").copy()
    buffer.close()
    return rendered


def _scatter_surface(
    ax,
    occupied: np.ndarray,
    *,
    color: str,
    max_points: int,
    alpha: float,
) -> None:
    surface = _surface_voxels(occupied)
    coordinates = np.argwhere(surface)
    if coordinates.shape[0] == 0:
        return
    if coordinates.shape[0] > max_points:
        indices = np.linspace(0, coordinates.shape[0] - 1, max_points, dtype=np.int64)
        coordinates = coordinates[indices]
    ax.scatter(
        coordinates[:, 2],
        coordinates[:, 1],
        coordinates[:, 0],
        c=color,
        s=2.0,
        alpha=alpha,
        linewidths=0,
        depthshade=True,
    )


def _surface_voxels(occupied: np.ndarray) -> np.ndarray:
    padded = np.pad(occupied, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    interior = (
        center
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return occupied & ~interior


if __name__ == "__main__":
    main()
