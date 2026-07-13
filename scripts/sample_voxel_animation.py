#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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
        "--render-resolution",
        type=int,
        default=32,
        help="Voxel render resolution for diffusion frames; sampling still runs at the configured resolution.",
    )
    parser.add_argument(
        "--final-render-resolution",
        type=int,
        default=64,
        help="Voxel render resolution for raw/detection/filtered final frames.",
    )
    parser.add_argument("--image-size", type=int, default=640, help="Square GIF frame size in pixels.")
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--azim", type=float, default=-60.0)
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
    if args.render_resolution <= 0 or args.final_render_resolution <= 0:
        raise ValueError("Render resolutions must be positive.")
    if args.image_size < 200:
        raise ValueError("--image-size must be at least 200.")

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
                render_resolution=(
                    args.final_render_resolution if step == 0 else args.render_resolution
                ),
                image_size=args.image_size,
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
            render_resolution=args.final_render_resolution,
            image_size=args.image_size,
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
            render_resolution=args.final_render_resolution,
            image_size=args.image_size,
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
    render_resolution: int,
    image_size: int,
    elev: float,
    azim: float,
    removed: torch.Tensor | None = None,
) -> Image.Image:
    occupied = voxels.detach().cpu().numpy().astype(bool)
    if occupied.ndim != 3:
        raise ValueError(f"Expected one [D, H, W] voxel sample, got {occupied.shape}.")

    values = occupied.astype(np.uint8)
    if removed is not None:
        removed_mask = removed.detach().cpu().numpy().astype(bool)
        if removed_mask.shape != occupied.shape:
            raise ValueError("removed mask must have the same shape as voxels.")
        values[removed_mask] = 2

    values = _downsample_volume(values, render_resolution)
    return _render_voxel_cubes(
        values,
        title=title,
        image_size=image_size,
        elev=elev,
        azim=azim,
    )


def _downsample_volume(values: np.ndarray, max_resolution: int) -> np.ndarray:
    """Nearest-neighbor downsampling that preserves occupancy density in noise frames."""

    if max(values.shape) <= max_resolution:
        return values
    indices = [
        np.linspace(0, size - 1, min(size, max_resolution)).round().astype(np.int64)
        for size in values.shape
    ]
    return values[np.ix_(*indices)]


def _render_voxel_cubes(
    values: np.ndarray,
    *,
    title: str,
    image_size: int,
    elev: float,
    azim: float,
) -> Image.Image:
    """Render opaque voxel cube faces with the same axis order as ``ax.voxels``."""

    occupied = values != 0
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    if not np.any(occupied):
        _draw_title(draw, title, image_size)
        return image

    right, up, view = _camera_basis(elev=elev, azim=azim)
    polygons: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    shades: list[float] = []

    light = np.asarray((0.45, -0.55, 1.0), dtype=np.float64)
    light /= np.linalg.norm(light)
    for axis, sign in itertools.product(range(3), (-1, 1)):
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = sign
        if float(normal @ view) <= 0.0:
            continue
        exposed = _exposed_face_mask(occupied, axis=axis, sign=sign)
        coordinates = np.argwhere(exposed)
        if coordinates.shape[0] == 0:
            continue
        vertices = _face_vertices(coordinates, axis=axis, sign=sign)
        polygons.append(np.stack((vertices @ right, vertices @ up), axis=-1))
        depths.append((vertices @ view).mean(axis=1))
        colors.append(values[tuple(coordinates.T)])
        shades.append(0.58 + 0.42 * max(0.0, float(normal @ light)))

    if not polygons:
        _draw_title(draw, title, image_size)
        return image

    polygon_array = np.concatenate(polygons, axis=0)
    depth_array = np.concatenate(depths, axis=0)
    color_array = np.concatenate(colors, axis=0)
    shade_array = np.concatenate(
        [np.full(len(group), shade) for group, shade in zip(polygons, shades)]
    )
    pixel_polygons = _project_to_pixels(polygon_array, values.shape, right, up, image_size)
    order = np.argsort(depth_array)
    base_colors = {
        1: np.asarray((37, 99, 235), dtype=np.float64),
        2: np.asarray((239, 68, 68), dtype=np.float64),
    }
    for index in order:
        base = base_colors[int(color_array[index])]
        fill = tuple(np.clip(base * shade_array[index], 0, 255).astype(np.uint8).tolist())
        edge = tuple(np.clip(base * shade_array[index] * 0.55, 0, 255).astype(np.uint8).tolist())
        points = [tuple(point) for point in pixel_polygons[index].round().astype(np.int32)]
        draw.polygon(points, fill=fill, outline=edge, width=1)

    _draw_title(draw, title, image_size)
    return image


def _camera_basis(*, elev: float, azim: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elevation = math.radians(elev)
    azimuth = math.radians(azim)
    view = np.asarray(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ),
        dtype=np.float64,
    )
    right = np.asarray((-math.sin(azimuth), math.cos(azimuth), 0.0), dtype=np.float64)
    up = np.cross(view, right)
    return right, up, view


def _exposed_face_mask(occupied: np.ndarray, *, axis: int, sign: int) -> np.ndarray:
    exposed = occupied.copy()
    current = [slice(None)] * 3
    neighbor = [slice(None)] * 3
    if sign > 0:
        current[axis] = slice(0, -1)
        neighbor[axis] = slice(1, None)
    else:
        current[axis] = slice(1, None)
        neighbor[axis] = slice(0, -1)
    exposed[tuple(current)] &= ~occupied[tuple(neighbor)]
    return exposed


def _face_vertices(coordinates: np.ndarray, *, axis: int, sign: int) -> np.ndarray:
    """Build cube faces without reordering the tensor's three spatial axes."""

    coordinates = coordinates.astype(np.float64)
    vertices = np.repeat(coordinates[:, None, :], 4, axis=1)
    other_axes = [candidate for candidate in range(3) if candidate != axis]
    vertices[:, :, axis] += 1.0 if sign > 0 else 0.0
    vertices[:, :, other_axes[0]] += np.asarray((0.0, 1.0, 1.0, 0.0))
    vertices[:, :, other_axes[1]] += np.asarray((0.0, 0.0, 1.0, 1.0))
    return vertices


def _project_to_pixels(
    polygons: np.ndarray,
    shape: tuple[int, ...],
    right: np.ndarray,
    up: np.ndarray,
    image_size: int,
) -> np.ndarray:
    corners = np.asarray(list(itertools.product(*((0.0, float(size)) for size in shape))))
    projected_corners = np.stack((corners @ right, corners @ up), axis=-1)
    minimum = projected_corners.min(axis=0)
    maximum = projected_corners.max(axis=0)
    center = (minimum + maximum) * 0.5
    span = np.maximum(maximum - minimum, 1e-6)
    title_height = 64
    margin = 28
    available_width = image_size - 2 * margin
    available_height = image_size - title_height - 2 * margin
    scale = min(available_width / span[0], available_height / span[1])

    pixels = polygons.copy()
    pixels[..., 0] = (pixels[..., 0] - center[0]) * scale + image_size * 0.5
    pixels[..., 1] = (
        title_height + available_height * 0.5 - (pixels[..., 1] - center[1]) * scale
    )
    return pixels


def _draw_title(draw: ImageDraw.ImageDraw, title: str, image_size: int) -> None:
    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), size=max(12, image_size // 38))
    wrapped = textwrap.fill(title, width=max(36, image_size // 9))
    bounds = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=2)
    width = bounds[2] - bounds[0]
    draw.multiline_text(
        ((image_size - width) * 0.5, 8),
        wrapped,
        fill=(20, 20, 20),
        font=font,
        align="center",
        spacing=2,
    )


if __name__ == "__main__":
    main()
