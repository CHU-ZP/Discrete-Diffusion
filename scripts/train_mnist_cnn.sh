#!/usr/bin/env bash
set -euo pipefail
uv run python -m ddiff.train --config configs/mnist_cnn.yaml
