# Documentation

This document contains the operational guide for the discrete diffusion demo: environment setup, data preparation, training, sampling, animation, and output locations.

For the method, implementation correspondence, and experiment results, return to the [English README](../README.md) or [中文 README](../README.zh-CN.md).

## Repository Paths

```text
configs/                         experiment configurations
src/ddiff/diffusion/             categorical forward and reverse processes
src/ddiff/models/                MNIST CNN and voxel 3D U-Net
src/ddiff/train.py               training, EMA, scheduling, evaluation, checkpoints
src/ddiff/sample.py              standard sampling entry point
scripts/prepare_modelnet_voxels.py
scripts/train_voxel_classifier.py
scripts/build_modelnet_subtypes.py
scripts/sample_voxel_animation.py
results/                         curated images used by the READMEs
```

## Environment

The project uses `uv` and Python 3.10 or newer. On Linux, `pyproject.toml` selects the PyTorch CUDA 12.8 wheels.

```bash
uv sync --frozen --no-dev
```

This creates `.venv/` and installs the project in editable mode. Run project commands through `uv run` without manually activating the environment.

Check the PyTorch and CUDA installation:

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("memory GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
```

If an older environment selected incompatible CUDA packages, recreate it:

```bash
rm -rf .venv
uv sync --frozen --no-dev
```

## MNIST Experiment

The MNIST configuration downloads the dataset through `torchvision`, quantizes pixels into 32 categories, and trains a digit-conditioned 2D CNN.

### Train

```bash
uv run python -m ddiff.train --config configs/mnist_cnn.yaml
```

A short smoke run is available through the step override:

```bash
uv run python -m ddiff.train \
  --config configs/mnist_cnn.yaml \
  --steps 500
```

Helper script:

```bash
./scripts/train_mnist_cnn.sh
```

### Sample

Cycle through labels `0..9`:

```bash
uv run python -m ddiff.sample \
  --config configs/mnist_cnn.yaml \
  --ckpt runs/mnist_cnn/latest.pt \
  --labels all \
  --num-samples 100
```

Generate one requested digit:

```bash
uv run python -m ddiff.sample \
  --config configs/mnist_cnn.yaml \
  --ckpt runs/mnist_cnn/latest.pt \
  --labels 7 \
  --num-samples 64
```

Helper script:

```bash
./scripts/sample_mnist_cnn.sh
```

### MNIST Outputs

```text
runs/mnist_cnn/latest.pt
runs/mnist_cnn/step_*.pt
outputs/mnist_cnn/real_samples.png
outputs/mnist_cnn/forward_chain.png
outputs/mnist_cnn/generated_samples.png
outputs/mnist_cnn/generated_samples_step_*.png
outputs/mnist_cnn/reverse_chain.png
```

## ModelNet10 Experiment

The complete voxel workflow is:

```text
ModelNet10 OFF meshes
  -> normalized 64^3 binary voxel cache
  -> supervised 3D classifier embeddings
  -> per-class subtype clustering
  -> subtype-conditioned 3D U-Net diffusion
  -> sampling and connected-component cleanup
```

### Download ModelNet10

Download and extract the official Princeton archive under `data/`:

```bash
mkdir -p data
wget -O data/ModelNet10.zip \
  http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip -q data/ModelNet10.zip -d data
```

Expected layout:

```text
data/ModelNet10/chair/train/*.off
data/ModelNet10/chair/test/*.off
data/ModelNet10/sofa/train/*.off
...
```

Check the archive:

```bash
find data/ModelNet10 -path "*/train/*.off" | wc -l
find data/ModelNet10 -path "*/test/*.off" | wc -l
find data/ModelNet10 -maxdepth 2 -type d | sort | head
```

### Build the Base Voxel Cache

The preprocessing script selects the four most frequent ModelNet10 classes, normalizes every mesh into a common cube, voxelizes it at `64 x 64 x 64`, and flood-fills closed interiors.

```bash
uv run python scripts/prepare_modelnet_voxels.py \
  --input data/ModelNet10 \
  --output data/modelnet10_voxel_64_top4.npz \
  --resolution 64 \
  --num-model-classes 4 \
  --workers 16 \
  --overwrite
```

The cache contains:

```text
train_x: uint8 [N, 64, 64, 64], 0 empty / 1 occupied
train_y: int64 [N], original class labels
test_x:  uint8 [N, 64, 64, 64]
test_y:  int64 [N]
class_names
```

### Train the Voxel Classifier

The classifier supplies geometric embeddings for subtype discovery.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_voxel_classifier.py \
  --cache data/modelnet10_voxel_64_top4.npz \
  --output-dir runs/voxel_classifier_top4 \
  --epochs 30 \
  --batch-size 16
```

Use `runs/voxel_classifier_top4/best.pt` for subtype construction.

### Build the Subtype Cache

The default recipe creates three clusters inside each of four original classes, for 12 conditioning labels in total.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/build_modelnet_subtypes.py \
  --input data/modelnet10_voxel_64_top4.npz \
  --classifier runs/voxel_classifier_top4/best.pt \
  --output data/modelnet10_voxel_64_top4_subtypes.npz \
  --subtypes-per-class 3 \
  --batch-size 16 \
  --overwrite
```

Custom per-class subtype counts are also supported:

```bash
uv run python scripts/build_modelnet_subtypes.py \
  --input data/modelnet10_voxel_64_top4.npz \
  --classifier runs/voxel_classifier_top4/best.pt \
  --output data/modelnet10_voxel_64_top4_subtypes.npz \
  --subtypes-per-class 4,2,2,3 \
  --overwrite
```

Important arrays in the subtype cache:

```text
train_y / test_y: global subtype labels used for diffusion conditioning
train_class_y / test_class_y: original ModelNet class labels
train_subtype_y / test_subtype_y: class-local cluster ids
subtype_names: readable names such as chair_0
subtype_counts / subtype_test_counts
```

Before training, ensure every subtype has held-out examples for generation-quality evaluation:

```bash
uv run python - <<'PY'
import numpy as np

data = np.load("data/modelnet10_voxel_64_top4_subtypes.npz")
print("names:", data["subtype_names"])
print("train:", data["subtype_counts"])
print("test:", data["subtype_test_counts"])
assert (data["subtype_test_counts"] > 0).all()
PY
```

If a test count is zero, reduce or change the subtype clustering recipe before diffusion training.

### Voxel Training Configuration

The default [`configs/voxel_modelnet10.yaml`](../configs/voxel_modelnet10.yaml) targets a 48 GB training GPU:

```yaml
model:
  base_channels: 48
  channel_mults: [1, 2, 4, 4]
  num_res_blocks: 2

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

The model has approximately 33.5M parameters. Every 5,000 steps, EMA weights generate two fixed-noise samples per subtype. The checkpoint score combines nearest same-subtype IoU, occupancy error, surface-to-volume error, and disconnected-fragment ratio.

`best.pt` is the lowest generation-quality score. `latest.pt` is the most recent training state and should be used for resume.

### Voxel Smoke Test

Run this before a full server job:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ddiff.train \
  --config configs/voxel_modelnet10.yaml \
  --steps 20 \
  --no-eval \
  --no-samples
```

The smoke run checks loading, forward/backward execution, EMA, and checkpoint writing without running a complete reverse chain.

### Train on a Server

Start a detachable session:

```bash
tmux new -s voxel64
```

Inside `tmux`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ddiff.train \
  --config configs/voxel_modelnet10.yaml \
  2>&1 | tee train_voxel64.log
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t voxel64
```

Monitor the job:

```bash
tail -f train_voxel64.log
watch -n 2 nvidia-smi
ls -lh runs/voxel_modelnet10_64_subtypes_v2/
ls -lh outputs/voxel_modelnet10_64_subtypes_v2/
```

If training batch size 4 does not fit, reduce `train.batch_size` to 2. Evaluation memory is controlled independently by `train.sample_micro_batch_size`; reduce it from 4 to 2 if preview sampling runs out of memory. Keep `sample_batch_size: 24` so all 12 subtypes remain equally represented.

### Resume Training

Keep the same final step target and resume from `latest.pt`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ddiff.train \
  --config configs/voxel_modelnet10.yaml \
  --resume runs/voxel_modelnet10_64_subtypes_v2/latest.pt \
  2>&1 | tee -a train_voxel64.log
```

Resume restores online weights, EMA weights, optimizer state, the best quality score, and PyTorch RNG state. The DataLoader starts a newly shuffled epoch, so the resumed run is statistically equivalent but not bit-for-bit identical to an uninterrupted run.

### Inspect a Checkpoint

```bash
uv run python - <<'PY'
import torch

path = "runs/voxel_modelnet10_64_subtypes_v2/best.pt"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
print("step:", checkpoint["step"])
print("learning rate:", checkpoint.get("learning_rate"))
print("best score:", checkpoint.get("best_quality_score"))
print("metrics:", checkpoint.get("quality_metrics"))
print("has EMA:", "ema_model" in checkpoint)
PY
```

## Voxel Sampling

Standard sampling writes both a rendered grid and a compressed NPZ containing raw and cleaned voxel arrays.

### Empirical Subtype Prior

Omitting label arguments draws subtype ids according to `subtype_counts`:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --num-samples 12
```

### Every Subtype

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --labels all \
  --num-samples 12
```

### Subtypes from One Original Class

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --classes monitor \
  --num-samples 8
```

### EMA and Raw Weights

Sampling defaults to `--weights auto`, which prefers `ema_model` and falls back to `model` for legacy checkpoints.

Force EMA weights:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --weights ema \
  --labels all \
  --num-samples 12
```

Use `--weights model` only for a raw-versus-EMA comparison.

### Connected-Component Cleanup

The default sampling configuration keeps the largest 6-connected occupied component:

```yaml
sample:
  voxel_component_filter: largest
  voxel_connectivity: 6
```

For a raw diagnostic run:

```bash
uv run python -m ddiff.sample \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --labels all \
  --num-samples 12 \
  --voxel-component-filter none
```

The NPZ preserves `raw_samples`, cleaned `samples`, numeric labels, readable subtype names, and per-sample component statistics.

## Reverse-Diffusion Animation

The animation flows directly from noise to the silently cleaned result. It does not display a raw/detection/filtered comparison. Timing is fixed for the 100-step voxel model:

```text
2.5 seconds reverse diffusion + 1 second final hold
```

There are no FPS, frame-stride, or hold-duration command-line controls.

### One Subtype

Subtype names and ids are both accepted:

```bash
uv run python scripts/sample_voxel_animation.py \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --label chair_0
```

### Every Subtype

```bash
uv run python scripts/sample_voxel_animation.py \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --labels all \
  --num-samples 1 \
  --render-workers 4
```

### Several Subtypes and Samples

`--num-samples` is the number generated per selected subtype:

```bash
uv run python scripts/sample_voxel_animation.py \
  --config configs/voxel_modelnet10.yaml \
  --ckpt runs/voxel_modelnet10_64_subtypes_v2/best.pt \
  --labels chair_0,sofa_2,bed_0 \
  --num-samples 2 \
  --render-workers 4
```

This command creates six GIFs. Sampling is batched on the selected device; rendering runs in CPU worker processes. `--render-workers 0` automatically selects up to four workers.

Use `--output path/to/name.gif` as a filename prefix, or pass a path without `.gif` as an output directory.

## Voxel Outputs

Training and sampling write:

```text
runs/voxel_classifier_top4/best.pt
data/modelnet10_voxel_64_top4.npz
data/modelnet10_voxel_64_top4_subtypes.npz
data/modelnet10_voxel_64_top4_subtypes_manifest.csv

runs/voxel_modelnet10_64_subtypes_v2/best.pt
runs/voxel_modelnet10_64_subtypes_v2/latest.pt
runs/voxel_modelnet10_64_subtypes_v2/step_*.pt

outputs/voxel_modelnet10_64_subtypes_v2/real_voxels.png
outputs/voxel_modelnet10_64_subtypes_v2/generated_voxels_step_*.png
outputs/voxel_modelnet10_64_subtypes_v2/generated_voxels.png
outputs/voxel_modelnet10_64_subtypes_v2/generated_voxels.npz
outputs/voxel_modelnet10_64_subtypes_v2/reverse_diffusion_*.gif
```

`runs/`, `outputs/`, and generated data are ignored by Git. Curated result images intended for the repository live under `results/`.

## Inspect the Voxel Cache

Open [`notebooks/visualize_voxels.ipynb`](../notebooks/visualize_voxels.ipynb). It displays metadata, label counts, occupancy statistics, 3D views, orthogonal slices, and examples grouped by class.

## Tests

Run the complete suite:

```bash
uv run python -m unittest discover -s tests -v
```

Run only animation tests:

```bash
uv run python -m unittest tests.test_voxel_animation -v
```

## Troubleshooting

### CUDA is unavailable

Confirm that the NVIDIA driver is visible through `nvidia-smi`, then check `torch.version.cuda` and `torch.cuda.is_available()` with the environment command above.

### Voxel training runs out of GPU memory

Reduce `train.batch_size`. If the failure occurs only during periodic generation, reduce `train.sample_micro_batch_size`. Do not reduce the total quality sample count below one sample per subtype.

### A checkpoint does not match the configuration

The sampler validates the dataset, cache, label count, diffusion schedule, and model architecture. Use the exact configuration and subtype cache that produced the checkpoint. The v2 four-level U-Net is intentionally incompatible with original three-level checkpoints.

### Quality evaluation reports a missing subtype

Inspect `subtype_test_counts`. Every conditioning label needs at least one test reference. Rebuild the subtype cache with fewer clusters if necessary.
