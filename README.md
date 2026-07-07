# discrete-diffusion-demo

A small standalone PyTorch project for categorical diffusion on discrete image and voxel data. The runnable paths train a class-conditional MNIST generator with a CNN denoiser and a ModelNet10 voxel generator with a 3D U-Net denoiser.

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
ModelNet10 OFF meshes -> 64^3 voxel cache -> UNet3D denoiser -> categorical diffusion engine
```

## Install

The project is managed with `uv`. On Linux, `pyproject.toml` pins `torch` and `torchvision` to the PyTorch CUDA 12.8 wheel index (`cu128`) instead of the default PyPI CUDA 13 packages. This is the intended setup for servers that support CUDA 12.x but not CUDA 13.x.

```bash
uv sync --frozen --no-dev
```

This creates a local `.venv/` and installs the project in editable mode from `pyproject.toml`. Run commands through `uv run` so you do not need to activate the environment manually.

Verify that the installed PyTorch build is CUDA 12.x and that the GPU is visible:

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

If you previously synced an environment that pulled CUDA 13 packages, remove and recreate it:

```bash
rm -rf .venv
uv sync --frozen --no-dev
```

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

## ModelNet10 Server Workflow

The voxel path is configured in `configs/voxel_modelnet10.yaml`. The full workflow is:

```text
download ModelNet10.zip -> extract OFF meshes -> build 64^3 voxel cache -> train UNet3D -> sample and inspect results
```

Download the official Princeton ModelNet10 zip and extract it under `data/`:

```bash
mkdir -p data
wget -O data/ModelNet10.zip \
  http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip -q data/ModelNet10.zip -d data
```

The expected structure is:

```text
data/ModelNet10/chair/train/*.off
data/ModelNet10/chair/test/*.off
data/ModelNet10/sofa/train/*.off
...
```

Check the extracted mesh files:

```bash
find data/ModelNet10 -path "*/train/*.off" | wc -l
find data/ModelNet10 -path "*/test/*.off" | wc -l
find data/ModelNet10 -maxdepth 2 -type d | sort | head
```

Build the 64^3 top-4-class voxel cache. The preparation script reads `.off` meshes, selects the 4 most frequent object classes, normalizes each mesh into a shared cube, voxelizes it at `64x64x64`, flood-fills closed interiors when possible, and writes a cached `.npz`.

```bash
uv run python scripts/prepare_modelnet_voxels.py \
  --input data/ModelNet10 \
  --output data/modelnet10_voxel_64_top4.npz \
  --resolution 64 \
  --num-model-classes 4 \
  --workers 16 \
  --overwrite
```

The output cache contains:

```text
train_x: uint8 [N, 64, 64, 64], values 0 empty / 1 occupied
train_y: int64 [N], labels remapped to 0..3
test_x:  uint8 [N, 64, 64, 64]
test_y:  int64 [N]
class_names: selected ModelNet class names in label order
```

In the voxel config, `dataset.num_classes: 2` means voxel token classes, while `dataset.num_labels: 4` means the selected object classes used for conditional generation.

Start a detachable training session with `tmux`:

```bash
tmux new -s voxel64
```

Inside tmux, train with:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ddiff.train \
  --config configs/voxel_modelnet10.yaml \
  2>&1 | tee train_voxel64.log
```

Detach without stopping training with `Ctrl-b` then `d`. Reattach later with:

```bash
tmux attach -t voxel64
```

Monitor training:

```bash
tail -f train_voxel64.log
watch -n 2 nvidia-smi
ls -lh runs/voxel_modelnet10_64/
ls -lh outputs/voxel_modelnet10_64/
```

Generate samples from the latest checkpoint:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64/latest.pt \
  --labels all \
  --num-samples 4 \
  --guidance-scale 2.0
```

Training and sampling write:

```text
runs/voxel_modelnet10_64/latest.pt
runs/voxel_modelnet10_64/step_*.pt
outputs/voxel_modelnet10_64/real_voxels.png
outputs/voxel_modelnet10_64/generated_voxels_step_*.png
outputs/voxel_modelnet10_64/generated_voxels.png
```

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

Voxel training uses classifier-free guidance support:

```yaml
train:
  class_dropout_prob: 0.15

sampling:
  guidance_scale: 2.0
```

During training, a fraction of class labels are replaced with a null label so the same model learns both conditional and unconditional denoising. During sampling, conditional logits are guided by unconditional logits to strengthen class-specific shapes.
