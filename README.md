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
download ModelNet10.zip -> extract OFF meshes -> build 64^3 voxel cache -> train classifier -> cluster subtypes -> train UNet3D -> sample and inspect results
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

In the base voxel cache, `train_y` / `test_y` are the original top-4 ModelNet class labels. These labels are used to train a supervised 3D CNN classifier whose embedding space is then clustered inside each original class.

Train the voxel classifier:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_voxel_classifier.py \
  --cache data/modelnet10_voxel_64_top4.npz \
  --output-dir runs/voxel_classifier_top4 \
  --epochs 30 \
  --batch-size 16
```

Build the subtype cache from classifier embeddings. By default this creates 3 subtypes per original class, for 12 total subtype conditioning labels:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/build_modelnet_subtypes.py \
  --input data/modelnet10_voxel_64_top4.npz \
  --classifier runs/voxel_classifier_top4/best.pt \
  --output data/modelnet10_voxel_64_top4_subtypes.npz \
  --subtypes-per-class 3 \
  --batch-size 16 \
  --overwrite
```

You can use class-specific subtype counts, for example:

```bash
uv run python scripts/build_modelnet_subtypes.py \
  --input data/modelnet10_voxel_64_top4.npz \
  --classifier runs/voxel_classifier_top4/best.pt \
  --output data/modelnet10_voxel_64_top4_subtypes.npz \
  --subtypes-per-class 4,2,2,3 \
  --overwrite
```

The subtype cache uses:

```text
train_y: subtype labels used for diffusion conditioning
train_class_y: original top-4 ModelNet class labels
train_subtype_y: class-local subtype id
subtype_names: readable subtype names, e.g. chair_0
subtype_counts: empirical train-set subtype counts for prior sampling
```

In the diffusion config, `dataset.num_classes: 2` means voxel token classes, while `dataset.num_labels: 12` means the subtype labels used for conditional generation. Voxel diffusion is trained as a pure subtype-conditioned model without a null-label branch.

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
ls -lh runs/voxel_modelnet10_64_subtypes/
ls -lh outputs/voxel_modelnet10_64_subtypes/
```

Generate samples from the latest checkpoint:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes/latest.pt \
  --num-samples 12
```

When `--labels` is omitted for a ModelNet subtype cache, sampling first draws subtype labels from `subtype_counts`, then generates conditioned on those subtype labels. To inspect every subtype once, use:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes/latest.pt \
  --labels all \
  --num-samples 12
```

The voxel grid titles use the cache's readable `subtype_names` values, such as
`chair_0`, rather than raw numeric ids. `--labels all` requires enough samples
to cover every configured conditioning label. Sampling also writes
`generated_voxels.npz`, preserving every generated voxel tensor together with
its numeric id and readable subtype name.

Generated voxels are post-processed with 6-connected component analysis. By
default only the largest occupied component is retained, removing isolated
floating voxels and fragments. The NPZ keeps cleaned `samples`, unmodified
`raw_samples`, and per-sample component/removal statistics. This behavior is
configured with:

```yaml
sample:
  voxel_component_filter: largest
  voxel_connectivity: 6
```

For an unfiltered diagnostic sample, override it from the command line:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes/latest.pt \
  --labels all \
  --num-samples 12 \
  --voxel-component-filter none
```

### Voxel Reverse-Diffusion GIF

Generate one animation for a conditioning subtype id:

```bash
uv run python scripts/sample_voxel_animation.py \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes/latest.pt \
  --label 7
```

`--label` also accepts the exact subtype name, such as `--label bed_1`. The GIF
shows the initial noise, every recorded reverse-diffusion step, the raw result,
the connected-component detection frame with removable fragments in red, and
the filtered result. By default every diffusion step is recorded; use
`--frame-stride 5` for a smaller and faster GIF or `--output path/to/file.gif`
to select the destination. Frames use opaque voxel cube faces rather than a
point cloud, preserve the tensor axis order used by `save_voxel_grid`, and use
the same default view (`elev=30`, `azim=-60`). Diffusion frames render at 32³
for speed while the final frames render at 64³; these can be changed with
`--render-resolution` and `--final-render-resolution` without changing the
model's actual 64³ sampling resolution.

To sample subtypes belonging to one original class, use the class name or class id:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes/latest.pt \
  --classes monitor \
  --num-samples 8
```

Training and sampling write:

```text
runs/voxel_classifier_top4/best.pt
data/modelnet10_voxel_64_top4_subtypes.npz
data/modelnet10_voxel_64_top4_subtypes_manifest.csv
runs/voxel_modelnet10_64_subtypes/latest.pt
runs/voxel_modelnet10_64_subtypes/step_*.pt
outputs/voxel_modelnet10_64_subtypes/real_voxels.png
outputs/voxel_modelnet10_64_subtypes/generated_voxels_step_*.png
outputs/voxel_modelnet10_64_subtypes/generated_voxels.png
outputs/voxel_modelnet10_64_subtypes/generated_voxels.npz
```

To inspect the generated voxel cache, open:

```text
notebooks/visualize_voxels.ipynb
```

The notebook prints cache metadata, label counts, occupancy statistics, 3D voxel renderings, orthogonal slices, and examples by class label.

The voxel files are:

```text
scripts/prepare_modelnet_voxels.py
scripts/train_voxel_classifier.py
scripts/build_modelnet_subtypes.py
scripts/sample_voxel_animation.py
configs/voxel_modelnet10.yaml
notebooks/visualize_voxels.ipynb
src/ddiff/data/modelnet_voxel.py
src/ddiff/models/voxel_classifier.py
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
