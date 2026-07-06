#!/usr/bin/env bash
set -euo pipefail
uv run python -m ddiff.sample --config configs/mnist_cnn.yaml --ckpt runs/mnist_cnn/latest.pt --labels all
