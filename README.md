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

## Prepare ModelNet10 Voxels

The voxel path is configured in `configs/voxel_modelnet10.yaml`. The preparation script reads ModelNet10 `.off` meshes, selects the 4 most frequent object classes, normalizes each mesh into a shared cube, voxelizes it at `32x32x32`, flood-fills closed interiors when possible, and writes a cached `.npz`.

```bash
uv run python scripts/prepare_modelnet_voxels.py \
  --input data/ModelNet10 \
  --output data/modelnet10_voxel_32_top4.npz \
  --resolution 32 \
  --num-model-classes 4 \
  --overwrite
```

The output cache contains:

```text
train_x: uint8 [N, 32, 32, 32], values 0 empty / 1 occupied
train_y: int64 [N], labels remapped to 0..3
test_x:  uint8 [N, 32, 32, 32]
test_y:  int64 [N]
class_names: selected ModelNet class names in label order
```

In the voxel config, `dataset.num_classes: 2` means voxel token classes, while `dataset.num_labels: 4` means the selected object classes used for conditional generation.

To inspect the generated voxel cache, open:

```text
notebooks/visualize_voxels.ipynb
```

The notebook prints cache metadata, label counts, occupancy statistics, 3D voxel renderings, orthogonal slices, and examples by class label.

The voxel files are:

```text
scripts/prepare_modelnet_voxels.py
configs/voxel_modelnet10.yaml
notebooks/visualize_voxels.ipynb
src/ddiff/data/modelnet_voxel.py
src/ddiff/models/unet3d.py
src/ddiff/visualization/voxels.py
```

The `unet3d` backbone is a residual encoder-decoder 3D U-Net with downsampling, upsampling, bottleneck blocks, skip connections, and time/class conditioning in each residual block.

Voxel training also enables token-level weighted cross entropy:

```yaml
train:
  token_loss_weights: auto
  max_token_loss_weight: 8.0
```

With `auto`, the training script counts voxel token frequencies in the train split and up-weights rare tokens. For binary occupancy this gives occupied voxels a larger loss weight than empty voxels, which helps counter the strong empty/occupied imbalance without changing the diffusion transition process.
