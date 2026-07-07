#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from tqdm.auto import tqdm


@dataclass
class ClassFiles:
    name: str
    label: int
    count: int
    train_files: list[Path]
    test_files: list[Path]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a fixed-resolution ModelNet10 voxel cache from OFF meshes. "
            "The output stores binary occupancy tokens and remapped class labels."
        )
    )
    parser.add_argument("--input", default="data/ModelNet10", help="Root directory of extracted ModelNet10.")
    parser.add_argument("--output", default="data/modelnet10_voxel_32_top4.npz", help="Output .npz cache path.")
    parser.add_argument("--resolution", type=int, default=32, help="Voxel grid resolution.")
    parser.add_argument(
        "--num-model-classes",
        type=int,
        default=4,
        help="Number of most frequent ModelNet classes to keep.",
    )
    parser.add_argument(
        "--padding-voxels",
        type=float,
        default=2.0,
        help="Padding left around normalized meshes inside the voxel grid.",
    )
    parser.add_argument(
        "--surface-dilation",
        type=int,
        default=0,
        help="Optional 6-neighbor dilation passes before interior fill.",
    )
    parser.add_argument(
        "--surface-only",
        action="store_true",
        help="Do not flood-fill closed interiors; save surface occupancy only.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap per split and class, useful for quick smoke tests.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip meshes that fail loading or voxelization instead of aborting.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output cache.")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_path = Path(args.output)
    if args.resolution <= 1:
        raise ValueError("--resolution must be greater than 1.")
    if args.num_model_classes <= 0:
        raise ValueError("--num-model-classes must be positive.")
    if args.padding_voxels < 0:
        raise ValueError("--padding-voxels must be non-negative.")
    if args.surface_dilation < 0:
        raise ValueError("--surface-dilation must be non-negative.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")

    selected = select_top_classes(input_root, args.num_model_classes, args.limit_per_class)
    print("Selected ModelNet classes:")
    for cls in selected:
        print(
            f"  label={cls.label} name={cls.name} total={cls.count} "
            f"train={len(cls.train_files)} test={len(cls.test_files)}"
        )

    train_x, train_y, train_paths = build_split(
        selected,
        "train",
        input_root=input_root,
        resolution=args.resolution,
        padding_voxels=args.padding_voxels,
        fill_interior=not args.surface_only,
        surface_dilation=args.surface_dilation,
        skip_errors=args.skip_errors,
    )
    test_x, test_y, test_paths = build_split(
        selected,
        "test",
        input_root=input_root,
        resolution=args.resolution,
        padding_voxels=args.padding_voxels,
        fill_interior=not args.surface_only,
        surface_dilation=args.surface_dilation,
        skip_errors=args.skip_errors,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez if args.no_compress else np.savez_compressed
    save(
        output_path,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        train_paths=np.asarray(train_paths, dtype="U"),
        test_paths=np.asarray(test_paths, dtype="U"),
        class_names=np.asarray([cls.name for cls in selected], dtype="U"),
        class_counts=np.asarray([cls.count for cls in selected], dtype=np.int64),
        resolution=np.asarray(args.resolution, dtype=np.int64),
        voxel_token_classes=np.asarray(2, dtype=np.int64),
        num_model_classes=np.asarray(len(selected), dtype=np.int64),
        filled_interiors=np.asarray(not args.surface_only, dtype=np.bool_),
        surface_dilation=np.asarray(args.surface_dilation, dtype=np.int64),
    )

    print(f"Saved {output_path}")
    print(f"  train_x: {train_x.shape} uint8 occupancy, train_y: {train_y.shape}")
    print(f"  test_x:  {test_x.shape} uint8 occupancy, test_y:  {test_y.shape}")


def select_top_classes(
    input_root: Path,
    num_model_classes: int,
    limit_per_class: int | None,
) -> list[ClassFiles]:
    if not input_root.exists():
        raise FileNotFoundError(
            f"ModelNet10 root not found at {input_root}. Expected directories like "
            f"{input_root}/chair/train/*.off and {input_root}/chair/test/*.off."
        )

    candidates: list[tuple[str, int, list[Path], list[Path]]] = []
    for class_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        train_files = find_split_off_files(class_dir, "train")
        test_files = find_split_off_files(class_dir, "test")
        count = len(train_files) + len(test_files)
        if count > 0:
            candidates.append((class_dir.name, count, train_files, test_files))

    if len(candidates) < num_model_classes:
        raise ValueError(
            f"Found {len(candidates)} ModelNet classes with OFF files under {input_root}, "
            f"but --num-model-classes={num_model_classes}."
        )

    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected: list[ClassFiles] = []
    for label, (name, count, train_files, test_files) in enumerate(candidates[:num_model_classes]):
        if limit_per_class is not None:
            train_files = train_files[:limit_per_class]
            test_files = test_files[:limit_per_class]
        selected.append(
            ClassFiles(
                name=name,
                label=label,
                count=count,
                train_files=train_files,
                test_files=test_files,
            )
        )
    return selected


def find_split_off_files(class_dir: Path, split: str) -> list[Path]:
    split_dir = class_dir / split
    if not split_dir.exists():
        return []
    return sorted(path for path in split_dir.rglob("*.off") if path.is_file())


def build_split(
    selected: list[ClassFiles],
    split: str,
    *,
    input_root: Path,
    resolution: int,
    padding_voxels: float,
    fill_interior: bool,
    surface_dilation: int,
    skip_errors: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    paths: list[str] = []
    total = sum(len(getattr(cls, f"{split}_files")) for cls in selected)
    progress = tqdm(total=total, desc=f"voxelize {split}", dynamic_ncols=True)

    for cls in selected:
        files: list[Path] = getattr(cls, f"{split}_files")
        for path in files:
            try:
                voxels = voxelize_off(
                    path,
                    resolution=resolution,
                    padding_voxels=padding_voxels,
                    fill_interior=fill_interior,
                    surface_dilation=surface_dilation,
                )
            except Exception as exc:
                if not skip_errors:
                    raise RuntimeError(f"Failed to voxelize {path}") from exc
                print(f"Skipping {path}: {type(exc).__name__}: {exc}")
                progress.update(1)
                continue

            xs.append(voxels)
            ys.append(cls.label)
            paths.append(str(path.relative_to(input_root)))
            progress.update(1)

    progress.close()
    if not xs:
        empty_x = np.zeros((0, resolution, resolution, resolution), dtype=np.uint8)
        empty_y = np.zeros((0,), dtype=np.int64)
        return empty_x, empty_y, []

    return np.stack(xs, axis=0).astype(np.uint8), np.asarray(ys, dtype=np.int64), paths


def voxelize_off(
    path: Path,
    *,
    resolution: int,
    padding_voxels: float,
    fill_interior: bool,
    surface_dilation: int,
) -> np.ndarray:
    mesh = load_normalized_mesh(path, resolution=resolution, padding_voxels=padding_voxels)
    voxel_grid = mesh.voxelized(pitch=1.0)
    points = np.asarray(voxel_grid.points)
    if points.size == 0:
        raise ValueError("voxelization produced no occupied voxels")

    indices = np.rint(points).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < resolution), axis=1)
    indices = indices[valid]
    if indices.size == 0:
        raise ValueError("all voxelized points fell outside the target grid")

    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)
    occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    for _ in range(surface_dilation):
        occupancy = dilate_6_connected(occupancy)

    if fill_interior:
        occupancy = fill_closed_interior(occupancy)

    return occupancy.astype(np.uint8)


def load_normalized_mesh(path: Path, *, resolution: int, padding_voxels: float) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, file_type="off", force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("scene did not contain any trimesh geometry")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"unsupported mesh type {type(loaded).__name__}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh has no vertices or faces")

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    max_extent = float(extents.max())
    if not np.isfinite(max_extent) or max_extent <= 0.0:
        raise ValueError("mesh has invalid bounds")

    usable = resolution - 1 - 2.0 * padding_voxels
    if usable <= 0:
        raise ValueError("--padding-voxels leaves no usable voxel volume")

    center = (bounds[0] + bounds[1]) * 0.5
    scale = usable / max_extent
    vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center) * scale
    vertices = vertices + (resolution - 1) * 0.5

    normalized = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    normalized.remove_unreferenced_vertices()
    return normalized


def dilate_6_connected(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[1:, :, :] |= grid[:-1, :, :]
    out[:-1, :, :] |= grid[1:, :, :]
    out[:, 1:, :] |= grid[:, :-1, :]
    out[:, :-1, :] |= grid[:, 1:, :]
    out[:, :, 1:] |= grid[:, :, :-1]
    out[:, :, :-1] |= grid[:, :, 1:]
    return out


def fill_closed_interior(surface: np.ndarray) -> np.ndarray:
    empty = ~surface
    exterior = np.zeros_like(surface, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque()
    resolution = surface.shape[0]

    def add_if_exterior(index: tuple[int, int, int]) -> None:
        if empty[index] and not exterior[index]:
            exterior[index] = True
            queue.append(index)

    last = resolution - 1
    for i in range(resolution):
        for j in range(resolution):
            add_if_exterior((0, i, j))
            add_if_exterior((last, i, j))
            add_if_exterior((i, 0, j))
            add_if_exterior((i, last, j))
            add_if_exterior((i, j, 0))
            add_if_exterior((i, j, last))

    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < resolution and 0 <= ny < resolution and 0 <= nz < resolution:
                add_if_exterior((nx, ny, nz))

    return surface | (~exterior)


if __name__ == "__main__":
    main()
