# discrete-diffusion-demo

A small standalone PyTorch project for categorical diffusion on discrete image data. The main runnable path trains a class-conditional MNIST generator with a CNN denoiser.

MNIST images are quantized into integer categories before diffusion. The default config uses 32 pixel categories, so each image is a `[28, 28]` grid of tokens in `{0, ..., 31}`. The model predicts `p_theta(x_0 | x_t, t, y)` as per-pixel categorical logits.

## How It Works

The diffusion engine uses uniform categorical corruption. For `K` categories, each step uses:

```text
Q_t = (1 - beta_t) I + beta_t U
```

where `U` is the uniform transition matrix. Training samples `x_t` from `q(x_t | x_0)`, then minimizes cross entropy between model logits and the original clean tokens `x_0`. Sampling starts from uniformly random categorical tokens and uses the exact categorical posterior to step from `x_t` to `x_{t-1}`.

The code is organized around:

```text
MNIST dataset -> CNN2D denoiser -> categorical diffusion engine
```

The ModelNet10 voxel path remains scaffolded for later experiments, but it is not the active path.

## Install

```bash
uv sync
```

This creates a local `.venv/` and installs the project in editable mode from `pyproject.toml`. Run commands through `uv run` so you do not need to activate the environment manually.

## Train MNIST CNN

```bash
uv run python -m ddiff.train --config configs/mnist_cnn.yaml
```

Training saves checkpoints to `runs/mnist_cnn/` and images to `outputs/mnist_cnn/`, including real samples, a forward noising chain, generated samples, and a reverse chain.

For a quick smoke run:

```bash
uv run python -m ddiff.train --config configs/mnist_cnn.yaml --steps 500
```

## Sample MNIST

Generate a grid cycling through labels `0..9`:

```bash
uv run python -m ddiff.sample --config configs/mnist_cnn.yaml --ckpt runs/mnist_cnn/latest.pt --labels all --num-samples 100
```

Generate one requested digit, for example `7`:

```bash
uv run python -m ddiff.sample --config configs/mnist_cnn.yaml --ckpt runs/mnist_cnn/latest.pt --labels 7 --num-samples 64
```

This writes:

```text
outputs/mnist_cnn/generated_samples.png
outputs/mnist_cnn/reverse_chain.png
```

You can also use the helper scripts:

```bash
./scripts/train_mnist_cnn.sh
./scripts/sample_mnist_cnn.sh
```

## Extending To ModelNet10 Voxels

The future voxel path is configured in `configs/voxel_modelnet10.yaml`. It expects cached integer voxel tensors shaped `[B, 16, 16, 16]` with values in `{0, 1}` and optional class labels in `{0, ..., 9}`.

The scaffold files are:

```text
src/ddiff/data/modelnet_voxel.py
src/ddiff/models/unet3d.py
src/ddiff/visualization/voxels.py
scripts/prepare_modelnet_voxels.py
```
