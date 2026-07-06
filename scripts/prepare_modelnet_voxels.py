#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaffold for preparing cached ModelNet10 voxel tensors."
    )
    parser.add_argument("--output", default="data/modelnet10_voxel_16.npz")
    parser.add_argument("--resolution", type=int, default=16)
    args = parser.parse_args()

    print("ModelNet10 voxel preparation is intentionally scaffolded for now.")
    print("Expected cache format:")
    print(f"  {args.output}")
    print(f"  x: uint8/int array [N, {args.resolution}, {args.resolution}, {args.resolution}] with values 0 or 1")
    print("  y: int array [N] with labels 0..9")
    print("  optional split: string array [N] with values train/test")


if __name__ == "__main__":
    main()
