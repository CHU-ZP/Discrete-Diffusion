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

In the diffusion config, `dataset.num_classes: 2` means voxel token classes,
while `dataset.num_labels: 12` means the subtype labels used for conditional
generation. Voxel diffusion is trained as a pure subtype-conditioned model
without a null-label branch.

The default voxel configuration is sized for a 48 GB GPU: a 33.5M-parameter
four-level 3D U-Net, batch size 4, and 100,000 optimization steps. It uses a
moderate `[empty, occupied] = [1, 2]` loss weight instead of the previous
automatic weight of up to 8, preserving thin furniture parts without strongly
biasing uncertain boundary voxels toward occupied. The learning rate warms up
for 2,000 steps, then follows cosine decay from `2e-4` to `2e-5`. An EMA model
with decay `0.9999` starts after the 2,000-step warmup; before that point it
tracks the online model exactly so random initialization does not pollute early
quality evaluations.

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

Before committing to the full run, verify the data/model path with a short run
that skips expensive generation evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ddiff.train \
  --config configs/voxel_modelnet10.yaml \
  --steps 20 \
  --no-eval \
  --no-samples
```

The `--steps` override automatically shortens warmup when needed. The smoke run
still writes into the configured v2 directories, so run it before the full job
or remove only its small smoke-run checkpoints first.

Detach without stopping training with `Ctrl-b` then `d`. Reattach later with:

```bash
tmux attach -t voxel64
```

Monitor training:

```bash
tail -f train_voxel64.log
watch -n 2 nvidia-smi
ls -lh runs/voxel_modelnet10_64_subtypes_v2/
ls -lh outputs/voxel_modelnet10_64_subtypes_v2/
```

If the job is interrupted, resume from the latest checkpoint while keeping the
same final `train.steps` target:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ddiff.train \
  --config configs/voxel_modelnet10.yaml \
  --resume runs/voxel_modelnet10_64_subtypes_v2/latest.pt \
  2>&1 | tee -a train_voxel64.log
```

Resume restores online weights, EMA weights, optimizer state, best score, and
PyTorch random-number state. It deliberately rejects incompatible voxel,
label, diffusion, or model settings. The DataLoader starts a newly shuffled
epoch after resume, so resumed training is statistically equivalent but not
bit-for-bit identical to an uninterrupted run.

Every 5,000 steps, training generates two fixed-noise samples for each subtype
using EMA weights. It compares those samples against held-out test voxels using
nearest same-subtype IoU, occupancy error, surface-to-volume error, and removed
fragment ratio. The lowest combined score is saved as `best.pt`; `latest.pt`
always represents the most recent checkpoint. Both files contain raw `model`
and averaged `ema_model` state dicts.

Generate samples from the best-quality checkpoint:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --num-samples 12
```

Sampling defaults to `--weights auto`, which loads `ema_model` when available
and falls back to `model` for old checkpoints. To compare raw training weights,
pass `--weights model`. `--weights ema` requires an EMA-capable checkpoint.

When `--labels` is omitted for a ModelNet subtype cache, sampling first draws subtype labels from `subtype_counts`, then generates conditioned on those subtype labels. To inspect every subtype once, use:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
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
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --labels all \
  --num-samples 12 \
  --voxel-component-filter none
```

### Voxel Reverse-Diffusion GIF

Generate one animation for a conditioning subtype id:

```bash
uv run python scripts/sample_voxel_animation.py \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
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

GIF frames contain no title or text overlay. To generate one independent GIF
for every subtype and render them in parallel:

```bash
uv run python scripts/sample_voxel_animation.py \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --labels all \
  --num-samples 1 \
  --render-workers 4 \
  --frame-stride 2
```

You can also select several different subtypes by readable name or numeric id:

```bash
uv run python scripts/sample_voxel_animation.py \
  --labels chair_0,sofa_1,bed_2 \
  --num-samples 2 \
  --render-workers 4
```

Here, `--num-samples` means samples **per selected subtype**, so the second
command generates six GIFs. Each sample is conditioned on its own subtype id;
the names are not merely output filename labels. `--label bed_1` remains the
short form for selecting one subtype.

To generate several independent samples for the same subtype:

```bash
uv run python scripts/sample_voxel_animation.py \
  --label bed_1 \
  --num-samples 4 \
  --render-workers 4 \
  --frame-stride 2
```

This writes `reverse_diffusion_bed_1_sample_000.gif` through
`reverse_diffusion_bed_1_sample_003.gif`. Sampling is batched on the selected
device, while GIF rendering runs in separate CPU processes. With
`--render-workers 0` the script automatically uses up to four workers. For
multiple samples or subtypes, a `.gif` passed to `--output` is treated as a
filename prefix; passing a path without the `.gif` suffix treats it as an
output directory. Output filenames always contain the readable subtype name
when multiple subtypes are selected.

To sample subtypes belonging to one original class, use the class name or class id:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --classes monitor \
  --num-samples 8
```

Training and sampling write:

```text
runs/voxel_classifier_top4/best.pt
data/modelnet10_voxel_64_top4_subtypes.npz
data/modelnet10_voxel_64_top4_subtypes_manifest.csv
runs/voxel_modelnet10_64_subtypes_v2/best.pt
runs/voxel_modelnet10_64_subtypes_v2/latest.pt
runs/voxel_modelnet10_64_subtypes_v2/step_*.pt
outputs/voxel_modelnet10_64_subtypes_v2/real_voxels.png
outputs/voxel_modelnet10_64_subtypes_v2/generated_voxels_step_*.png
outputs/voxel_modelnet10_64_subtypes_v2/generated_voxels.png
outputs/voxel_modelnet10_64_subtypes_v2/generated_voxels.npz
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

The boundary-oriented training settings are:

```yaml
train:
  batch_size: 4
  steps: 100000
  lr: 0.0002
  min_lr: 0.00002
  lr_scheduler: cosine
  warmup_steps: 2000
  token_loss_weights: [1.0, 2.0]
  ema_start_step: 2000
  ema_decay: 0.9999
  sample_every: 5000
  sample_batch_size: 24
  sample_micro_batch_size: 4
```

If batch size 4 does not fit because other processes occupy GPU memory, lower
only `train.batch_size` to 2. `sample_micro_batch_size` controls evaluation-time
GPU memory independently and can also be reduced without changing the quality
metric or number of evaluated samples. Changing the model architecture, voxel
cache, label count, or diffusion schedule makes checkpoints incompatible; the
v2 output directory intentionally avoids overwriting the original experiment.
