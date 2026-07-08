#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ddiff.models.voxel_classifier import VoxelClassifier3D
from ddiff.utils.config import resolve_device
from ddiff.utils.seed import set_seed


class VoxelClassificationDataset(Dataset):
    def __init__(self, cache_path: str | Path, split: str) -> None:
        self.cache_path = Path(cache_path)
        data = np.load(self.cache_path, allow_pickle=False)
        x_key = f"{split}_x"
        class_y_key = f"{split}_class_y"
        y_key = class_y_key if class_y_key in data else f"{split}_y"
        if x_key not in data or y_key not in data:
            raise KeyError(f"{self.cache_path} must contain {x_key} and {y_key}.")
        self.x = torch.from_numpy(data[x_key].astype(np.uint8))
        self.y = torch.from_numpy(data[y_key].astype(np.int64)).long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised 3D CNN classifier for ModelNet voxel embeddings.")
    parser.add_argument("--cache", default="data/modelnet10_voxel_64_top4.npz", help="Input voxel .npz cache.")
    parser.add_argument("--output-dir", default="runs/voxel_classifier_top4", help="Directory for classifier checkpoints.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = VoxelClassificationDataset(args.cache, split="train")
    test_set = VoxelClassificationDataset(args.cache, split="test")
    num_classes = int(max(train_set.y.max().item(), test_set.y.max().item()) + 1)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = VoxelClassifier3D(
        num_classes=num_classes,
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Training voxel classifier on {device}: train={len(train_set)} test={len(test_set)} classes={num_classes}")
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer=optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, device, optimizer=None)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "test_acc": test_acc,
            "model_config": {
                "num_classes": num_classes,
                "base_channels": args.base_channels,
                "embedding_dim": args.embedding_dim,
                "dropout": args.dropout,
            },
            "cache_path": str(args.cache),
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(checkpoint, output_dir / "best.pt")


def run_epoch(
    model: VoxelClassifier3D,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total = 0
    progress = tqdm(loader, desc="train" if training else "eval", leave=False, dynamic_ncols=True)
    for x, y in progress:
        x = x.to(device=device)
        y = y.to(device=device, dtype=torch.long)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with nullcontext() if training else torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        if training:
            loss.backward()
            optimizer.step()

        batch = y.shape[0]
        total_loss += float(loss.item()) * batch
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += batch

    return total_loss / max(total, 1), total_correct / max(total, 1)


if __name__ == "__main__":
    main()
