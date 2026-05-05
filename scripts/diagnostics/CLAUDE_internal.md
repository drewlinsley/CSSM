# CSSM — Cepstral State Space Models

JAX/Flax vision models using FFT-based spectral convolutions + temporal recurrence.
Replaces attention with O(N log N) spatial mixing and O(log T) parallel associative scan.

## Quick Start

```bash
# Pathfinder (contour integration, best config)
python main.py --arch simple --cssm add_kqv --dataset pathfinder \
    --pathfinder_difficulty 14 --tfrecord_dir /path/to/pathfinder_tfrecords_128 \
    --gate_type factored --embed_dim 32 --seq_len 8 --batch_size 256 --lr 3e-4 \
    --epochs 60 --use_complex32 --stem_layers 1 --pos_embed spatiotemporal \
    --kernel_size 11 --image_size 128 --stem_norm_order post --norm_type batch \
    --pool_type max --no_wandb

# PathTracker (video contour tracking, 32x32 RGB, 64 frames)
# seq_len and image_size auto-set from video (64 frames, 32x32)
python main.py --arch simple --cssm add_kqv --dataset pathtracker \
    --stem_mode pathtracker --embed_dim 32 \
    --gate_type factored --no_wandb

# ImageNet
python main.py --arch vit --cssm hgru_bi --dataset imagenet \
    --imagenet_dir /path/to/ILSVRC2012 \
    --batch_size 256 --seq_len 8 --depth 12 --embed_dim 384 --bf16
```

## Project Layout

```
main.py                          # Training entry point (all datasets)
src/
  models/
    cssm.py                      # Core CSSM variants (GatedCSSM, HGRUBilinearCSSM, TransformerCSSM, AdditiveCSSM, ...)
    simple_cssm.py               # SimpleCSSM wrapper (stem + CSSM blocks + readout)
    cssm_vit.py                  # CSSM-ViT architecture
    baseline_vit.py              # Standard ViT baseline
    math.py                      # Associative scan operations
    goom.py                      # Log-space (GOOM) primitives
    operations.py                # Log-space arithmetic
  pathfinder_data.py             # Pathfinder loader (PNG + TFRecord)
  pathtracker_data.py            # PathTracker loader (.npy video + TFRecord)
  cabc_data.py                   # cABC loader
  data.py                        # Imagenette/ImageNet loaders
scripts/
  convert_pathtracker_tfrecords.py   # .npy -> TFRecord conversion
docs/                            # Architecture docs, training guides
```

## Key Conventions

- **TensorFlow CPU-only**: TF is used only for TFRecord I/O. It MUST be configured CPU-only before import to avoid NCCL conflicts with JAX. This is done at the top of `main.py`.
- **Datasets**: Each dataset has its own `src/*_data.py` module with `get_*_loader()`, `get_*_info()`, and optionally `get_*_tfrecord_loader()`.
- **Model registry**: `CSSM_REGISTRY` in `simple_cssm.py` maps CLI names to classes.
- **Checkpointing**: Use `--checkpointer simple` (pickle, NFS-safe) on network filesystems. Use local scratch (`/local/scratch/`) for orbax.
- **Multi-GPU**: Automatic via `jax.devices()`. Uses `jax.pmap` with `axis_name='batch'`. Batch size must be divisible by device count.
- **First step is slow**: JAX JIT-compiles on the first batch — expect 1-3 min delay, then fast iteration.

## CSSM Variants

| CLI flag | Class | Notes |
|----------|-------|-------|
| `hgru_bi` | HGRUBilinearCSSM | Primary: 3-state (X,Y,Z) with E/I kernels |
| `add_kqv` | AdditiveCSSM | 3-state Q->K->V, triangular 3x3 scan |
| `add_kqv_2` | AdditiveCSSM | 2-state Q->V only |
| `add_kqv_1` | AdditiveCSSM | 1-state scalar scan |
| `transformer` / `kqv` | TransformerCSSM | Q/K/A attention-like |
| `gated` | GatedCSSM | Mamba-style scalar recurrence |

## Stem Modes (SimpleCSSM)

| `--stem_mode` | Behavior |
|---------------|----------|
| `default` | 2x (3x3 conv + maxpool) = 4x spatial downsample |
| `pathtracker` | 1x1 conv projection, no downsample (for 32x32 inputs) |

## Datasets

| Dataset | `--dataset` | Key flags | Input |
|---------|-------------|-----------|-------|
| Pathfinder | `pathfinder` | `--pathfinder_difficulty 9/14/20`, `--tfrecord_dir` | Static PNG, repeated T times |
| PathTracker | `pathtracker` | `--pathtracker_dir`, `--stem_mode pathtracker`, `--image_size 32` | 64-frame .npy video, subsampled to `--seq_len` |
| cABC | `cabc` | `--cabc_difficulty easy/medium/hard` | Static images |
| ImageNet | `imagenet` | `--imagenet_dir` | ILSVRC2012 |
| Imagenette | `imagenette` | `--data_dir` | Default dataset |

## Data Paths (Lab Machines)

- Pathfinder: `/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025`
- PathTracker: `/media/data_cifs/projects/prj_video_datasets/pathtracker` (0/ and 1/ subdirs, .npy)
- PathTracker TFRecords: `/media/data_cifs/projects/prj_video_datasets/pathtracker_tfrecords`
- cABC: `/media/data_cifs_lrs/projects/prj_LRA/cabc`
- ImageNet: `/gpfs/data/shared/imagenet/ILSVRC2012`

## Hyperparameter Sweep Results (Pathfinder 128px, 20 epochs)

Sweep over `stem_norm_order × learned_init × norm_type × pool_type` on Pathfinder difficulty 14, 128px TFRecords, stem_layers=1, embed_dim=32, add_kqv, factored gates.

### Best config (Pathfinder 128px)

```bash
python main.py --arch simple --cssm add_kqv --dataset pathfinder \
    --pathfinder_difficulty 14 --tfrecord_dir /path/to/pathfinder_tfrecords_128 \
    --gate_type factored --embed_dim 32 --seq_len 8 --batch_size 256 --lr 3e-4 \
    --epochs 60 --use_complex32 --stem_layers 1 --pos_embed spatiotemporal \
    --kernel_size 11 --image_size 128 --stem_norm_order post --norm_type batch \
    --pool_type max --no_wandb
```

### Findings (ranked by effect size)

| Hyperparameter | Best | Accuracy | Notes |
|----------------|------|----------|-------|
| **pool_type** | **max** | 0.70 vs 0.50 (mean) | Mean pooling never learns above chance. Dominant factor. |
| **norm_type** | **batch** | 0.71 vs 0.61 (layer) vs 0.50 (instance) | Instance norm never learns. |
| **stem_norm_order** | **post** | 0.55 vs 0.52 (pre) | Small effect. |
| **learned_init** | **off** | 0.54 vs 0.52 (on) | Slightly hurts. |

BN eval mode (running avg vs online batch stats vs calibrated training-set stats) makes no difference — running average is well-calibrated.

### Top results

| Config | Val Acc |
|--------|---------|
| post / noinit / batch / max | **0.7127** |
| pre / noinit / batch / max | 0.6869 |
| post / noinit / layer / max | 0.6149 |
| post / init / layer / max | 0.6122 |
| Everything else | ~0.50 (chance) |

## GDN Norm Sweep Results (PathTracker, 20 epochs)

Sweep over `norm_type` for GDN (depth=1, embed_dim=32, dk=2, factored gates, mrope) on `pathtracker_equal_large_tfrecords`.

| norm_type | Best Val Acc | Reduction axes (5D: B,T,H,W,C) | Stats computed per |
|-----------|-------------|-------------------------------|-------------------|
| **temporal_layer** | **0.6386** | T, C | each (b, h, w) spatial position |
| layer | 0.6360 | C | each (b, t, h, w) position |
| batch | 0.5990 | B, T, H, W (running stats) | each channel |
| instance | 0.5001 (chance) | H, W | each (b, t, c) |
| global_layer | TBD | T, H, W, C | each (b,) sample |

**Use `--norm_type global_layer` for all GDN PathTracker models.** Top 4 norms within noise (~0.635-0.639); global_layer chosen for simplicity (one mean/sd per sample). Batch and instance norms fail on this task (opposite of Pathfinder where batch wins).

## PathTracker Data Generation

```bash
# From /media/data_cifs/projects/prj_video_datasets/pathtracker-data/
# 55k positive samples
conda run -n neocog python cluster_store_...py -n 55000 -NS 0 -nd 14 -pl 64 \
    --path /media/data_cifs/projects/prj_video_datasets/pathtracker/
# 55k negative samples
conda run -n neocog python cluster_store_...py -n 55000 -NS 1 -nd 14 -pl 64 \
    --path /media/data_cifs/projects/prj_video_datasets/pathtracker/
```

Each .npy: `(64, 32, 32, 3)` uint8. Red=target, Blue=endpoint, Green=distractors. All same size (2x2 px).
