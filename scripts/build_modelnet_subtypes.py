#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ddiff.models.voxel_classifier import VoxelClassifier3D
from ddiff.utils.config import resolve_device
from ddiff.utils.seed import set_seed


class VoxelEmbeddingDataset(Dataset):
    def __init__(self, x: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.uint8))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a subtype-conditioned ModelNet voxel cache from supervised 3D CNN embeddings."
    )
    parser.add_argument("--input", default="data/modelnet10_voxel_64_top4.npz", help="Input top-class voxel cache.")
    parser.add_argument("--classifier", default="runs/voxel_classifier_top4/best.pt", help="Trained classifier checkpoint.")
    parser.add_argument("--output", default="data/modelnet10_voxel_64_top4_subtypes.npz", help="Output subtype cache.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output CSV manifest. Defaults to <output stem>_manifest.csv.",
    )
    parser.add_argument(
        "--subtypes-per-class",
        default="3",
        help="Either one integer for every class or a comma-separated list, e.g. '3,2,2,3'.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-compress", action="store_true", help="Use np.savez instead of np.savez_compressed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest is not None else output_path.with_name(
        f"{output_path.stem}_manifest.csv"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists; pass --overwrite to replace it.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    cache = np.load(input_path, allow_pickle=False)
    for key in ("train_x", "train_y", "test_x", "test_y"):
        if key not in cache:
            raise KeyError(f"{input_path} is missing required array {key}.")

    class_names = cache["class_names"].astype(str) if "class_names" in cache else None
    num_classes = len(class_names) if class_names is not None else int(max(cache["train_y"].max(), cache["test_y"].max()) + 1)
    subtypes_per_class = parse_subtypes_per_class(args.subtypes_per_class, num_classes)

    classifier = load_classifier(args.classifier, device)
    train_embeddings = extract_embeddings(classifier, cache["train_x"], args.batch_size, args.num_workers, device, "train")
    test_embeddings = extract_embeddings(classifier, cache["test_x"], args.batch_size, args.num_workers, device, "test")
    train_features, test_features = normalize_embeddings(train_embeddings, test_embeddings)

    train_class_y = cache["train_y"].astype(np.int64)
    test_class_y = cache["test_y"].astype(np.int64)
    clustering = cluster_by_class(
        train_features=train_features,
        test_features=test_features,
        train_class_y=train_class_y,
        test_class_y=test_class_y,
        class_names=class_names,
        subtypes_per_class=subtypes_per_class,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez if args.no_compress else np.savez_compressed
    arrays = build_output_arrays(cache, clustering, train_class_y, test_class_y, input_path, args.classifier)
    save(output_path, **arrays)
    write_manifest(manifest_path, cache, clustering, class_names)

    print(f"Saved subtype cache: {output_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Subtype names: {clustering['subtype_names'].tolist()}")
    print(f"Train subtype counts: {clustering['subtype_counts'].tolist()}")


def parse_subtypes_per_class(spec: str, num_classes: int) -> list[int]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if len(parts) == 1:
        value = int(parts[0])
        if value <= 0:
            raise ValueError("--subtypes-per-class must be positive.")
        return [value] * num_classes
    if len(parts) != num_classes:
        raise ValueError(f"Expected {num_classes} subtype counts, got {len(parts)} from {spec!r}.")
    values = [int(part) for part in parts]
    if any(value <= 0 for value in values):
        raise ValueError("--subtypes-per-class values must be positive.")
    return values


def load_classifier(path: str | Path, device: torch.device) -> VoxelClassifier3D:
    checkpoint = torch.load(path, map_location=device)
    model_config = checkpoint["model_config"]
    model = VoxelClassifier3D(
        num_classes=int(model_config["num_classes"]),
        base_channels=int(model_config["base_channels"]),
        embedding_dim=int(model_config["embedding_dim"]),
        dropout=float(model_config.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(
    model: VoxelClassifier3D,
    x: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    split: str,
) -> np.ndarray:
    loader = DataLoader(
        VoxelEmbeddingDataset(x),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    embeddings: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"embed {split}", dynamic_ncols=True):
        batch = batch.to(device=device)
        embedding = model.extract_embedding(batch)
        embeddings.append(embedding.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def normalize_embeddings(train_embeddings: np.ndarray, test_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_embeddings.mean(axis=0, keepdims=True)
    std = train_embeddings.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    train = (train_embeddings - mean) / std
    test = (test_embeddings - mean) / std
    train = l2_normalize(train)
    test = l2_normalize(test)
    return train.astype(np.float32), test.astype(np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-6)


def cluster_by_class(
    *,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_class_y: np.ndarray,
    test_class_y: np.ndarray,
    class_names: np.ndarray | None,
    subtypes_per_class: list[int],
    max_iter: int,
    seed: int,
) -> dict[str, np.ndarray]:
    train_y = np.zeros_like(train_class_y)
    test_y = np.zeros_like(test_class_y)
    train_subtype_y = np.zeros_like(train_class_y)
    test_subtype_y = np.zeros_like(test_class_y)
    train_distance = np.zeros(train_class_y.shape[0], dtype=np.float32)
    test_distance = np.zeros(test_class_y.shape[0], dtype=np.float32)
    subtype_names: list[str] = []
    subtype_class_ids: list[int] = []
    subtype_local_ids: list[int] = []
    centers: list[np.ndarray] = []

    offset = 0
    rng = np.random.default_rng(seed)
    for class_id, k in enumerate(subtypes_per_class):
        train_mask = train_class_y == class_id
        test_mask = test_class_y == class_id
        features = train_features[train_mask]
        if features.shape[0] < k:
            raise ValueError(f"Class {class_id} has {features.shape[0]} train samples, fewer than requested k={k}.")

        local_labels, class_centers = kmeans(features, k=k, max_iter=max_iter, rng=rng)
        train_distances = nearest_center_distance(features, class_centers, local_labels)
        train_y[train_mask] = local_labels + offset
        train_subtype_y[train_mask] = local_labels
        train_distance[train_mask] = train_distances

        if np.any(test_mask):
            test_local_labels, test_distances = assign_to_centers(test_features[test_mask], class_centers)
            test_y[test_mask] = test_local_labels + offset
            test_subtype_y[test_mask] = test_local_labels
            test_distance[test_mask] = test_distances

        class_name = str(class_names[class_id]) if class_names is not None else f"class_{class_id}"
        for local_id in range(k):
            subtype_names.append(f"{class_name}_{local_id}")
            subtype_class_ids.append(class_id)
            subtype_local_ids.append(local_id)
            centers.append(class_centers[local_id])
        offset += k

    total_subtypes = offset
    subtype_counts = np.bincount(train_y, minlength=total_subtypes).astype(np.int64)
    subtype_test_counts = np.bincount(test_y, minlength=total_subtypes).astype(np.int64)
    return {
        "train_y": train_y.astype(np.int64),
        "test_y": test_y.astype(np.int64),
        "train_subtype_y": train_subtype_y.astype(np.int64),
        "test_subtype_y": test_subtype_y.astype(np.int64),
        "train_distance": train_distance,
        "test_distance": test_distance,
        "subtype_names": np.asarray(subtype_names, dtype="U"),
        "subtype_class_ids": np.asarray(subtype_class_ids, dtype=np.int64),
        "subtype_local_ids": np.asarray(subtype_local_ids, dtype=np.int64),
        "subtype_counts": subtype_counts,
        "subtype_test_counts": subtype_test_counts,
        "subtype_centers": np.stack(centers, axis=0).astype(np.float32),
    }


def kmeans(features: np.ndarray, *, k: int, max_iter: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    centers = initialize_kmeans_plus_plus(features, k, rng)
    labels = np.zeros(features.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        new_labels, _ = assign_to_centers(features, centers)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        new_centers = centers.copy()
        for cluster_id in range(k):
            mask = labels == cluster_id
            if np.any(mask):
                new_centers[cluster_id] = features[mask].mean(axis=0)
            else:
                _, distances = assign_to_centers(features, centers)
                new_centers[cluster_id] = features[int(np.argmax(distances))]
        centers = l2_normalize(new_centers)
    labels, _ = assign_to_centers(features, centers)
    return labels, centers.astype(np.float32)


def initialize_kmeans_plus_plus(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((k, features.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, features.shape[0]))
    centers[0] = features[first]
    closest = squared_distance_to_center(features, centers[0])
    for idx in range(1, k):
        total = float(closest.sum())
        if total <= 1e-12:
            chosen = int(rng.integers(0, features.shape[0]))
        else:
            probs = closest / total
            chosen = int(rng.choice(features.shape[0], p=probs))
        centers[idx] = features[chosen]
        closest = np.minimum(closest, squared_distance_to_center(features, centers[idx]))
    return centers


def assign_to_centers(features: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1).astype(np.int64)
    return labels, np.sqrt(distances[np.arange(features.shape[0]), labels]).astype(np.float32)


def nearest_center_distance(features: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = ((features - centers[labels]) ** 2).sum(axis=1)
    return np.sqrt(distances).astype(np.float32)


def squared_distance_to_center(features: np.ndarray, center: np.ndarray) -> np.ndarray:
    return ((features - center) ** 2).sum(axis=1)


def build_output_arrays(
    cache,
    clustering: dict[str, np.ndarray],
    train_class_y: np.ndarray,
    test_class_y: np.ndarray,
    input_path: Path,
    classifier_path: str | Path,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "train_x": cache["train_x"],
        "test_x": cache["test_x"],
        "train_y": clustering["train_y"],
        "test_y": clustering["test_y"],
        "train_class_y": train_class_y.astype(np.int64),
        "test_class_y": test_class_y.astype(np.int64),
        "train_subtype_y": clustering["train_subtype_y"],
        "test_subtype_y": clustering["test_subtype_y"],
        "train_cluster_distance": clustering["train_distance"],
        "test_cluster_distance": clustering["test_distance"],
        "subtype_names": clustering["subtype_names"],
        "subtype_class_ids": clustering["subtype_class_ids"],
        "subtype_local_ids": clustering["subtype_local_ids"],
        "subtype_counts": clustering["subtype_counts"],
        "subtype_test_counts": clustering["subtype_test_counts"],
        "subtype_centers": clustering["subtype_centers"],
        "source_cache": np.asarray(str(input_path), dtype="U"),
        "classifier_checkpoint": np.asarray(str(classifier_path), dtype="U"),
    }
    for key in (
        "train_paths",
        "test_paths",
        "class_names",
        "class_counts",
        "resolution",
        "voxel_token_classes",
        "num_model_classes",
        "filled_interiors",
        "surface_dilation",
    ):
        if key in cache:
            arrays[key] = cache[key]
    return arrays


def write_manifest(
    path: Path,
    cache,
    clustering: dict[str, np.ndarray],
    class_names: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "index",
        "path",
        "class_label",
        "class_name",
        "subtype_label",
        "subtype_name",
        "subtype_local_id",
        "cluster_distance",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "test"):
            paths_key = f"{split}_paths"
            paths = cache[paths_key].astype(str) if paths_key in cache else None
            class_y = cache[f"{split}_y"].astype(np.int64)
            subtype_y = clustering[f"{split}_y"]
            subtype_local_y = clustering[f"{split}_subtype_y"]
            distances = clustering[f"{split}_distance"]
            for idx in range(class_y.shape[0]):
                class_label = int(class_y[idx])
                subtype_label = int(subtype_y[idx])
                writer.writerow(
                    {
                        "split": split,
                        "index": idx,
                        "path": str(paths[idx]) if paths is not None else f"{split}:{idx}",
                        "class_label": class_label,
                        "class_name": str(class_names[class_label]) if class_names is not None else str(class_label),
                        "subtype_label": subtype_label,
                        "subtype_name": str(clustering["subtype_names"][subtype_label]),
                        "subtype_local_id": int(subtype_local_y[idx]),
                        "cluster_distance": f"{float(distances[idx]):.6f}",
                    }
                )


if __name__ == "__main__":
    main()
