# CSSM - Cepstral State Space Models

A JAX/Flax implementation of Cepstral State Space Models for vision tasks. CSSM replaces traditional attention with FFT-based spectral convolutions and temporal recurrence.

## Key Features

- **FFT-based spatial processing**: O(N log N) spatial mixing via spectral convolution
- **Temporal recurrence**: Associative scan for O(log T) parallel temporal processing
- **GOOM numerics**: Log-space computation for numerical stability
- **Three CSSM variants**: GatedCSSM, HGRUBilinearCSSM (primary), KQVCSSM

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd CSSM

# Create conda environment
conda create -n cssm python=3.10
conda activate cssm

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Pathfinder (Contour Integration)

```bash
python main.py --arch simple --cssm hgru_bi --dataset pathfinder \
    --pathfinder_difficulty 14 \
    --tfrecord_dir /path/to/pathfinder_tfrecords/difficulty_14 \
    --batch_size 256 --seq_len 8 --depth 1 --embed_dim 32 \
    --kernel_size 11 --lr 3e-4 --epochs 60 \
    --pos_embed spatiotemporal --bf16
```

### ImageNet

```bash
python main.py --arch vit --cssm hgru_bi --dataset imagenet \
    --imagenet_dir /path/to/imagenet \
    --batch_size 256 --seq_len 8 --depth 12 --embed_dim 384 \
    --kernel_size 11 --lr 1e-4 --epochs 300 --bf16 \
    --checkpoint_dir /local/scratch/checkpoints
```

## Supported Tasks

| Task | Dataset | Architecture | Command |
|------|---------|--------------|---------|
| Contour integration | Pathfinder | `--arch simple` | See [pathfinder.md](docs/pathfinder.md) |
| Letter recognition | cABC | `--arch simple` | See [cabc.md](docs/cabc.md) |
| Image classification | ImageNet | `--arch vit` | See [imagenet.md](docs/imagenet.md) |

## Model Variants

| CSSM Type | Flag | Description |
|-----------|------|-------------|
| HGRUBilinearCSSM | `--cssm hgru_bi` | 3x3 with X, Y, Z states (recommended) |
| KQVCSSM | `--cssm kqv` | Transformer-inspired K*Q gating |
| GatedCSSM | `--cssm gated` | Mamba-style scalar recurrence |

See [docs/models.md](docs/models.md) for detailed descriptions.

## Architecture Options

| Flag | Options | Description |
|------|---------|-------------|
| `--arch` | `simple`, `vit`, `baseline` | Model architecture |
| `--embed_dim` | 32, 192, 384, 768 | Embedding dimension |
| `--depth` | 1, 6, 12 | Number of blocks |
| `--seq_len` | 8, 16, 32 | Temporal recurrence steps |
| `--kernel_size` | 11, 15 | Spatial kernel size |

## Project Structure

```
CSSM/
├── main.py                     # Main training script
├── src/
│   ├── pathfinder_data.py      # Pathfinder dataset
│   ├── cabc_data.py            # cABC dataset
│   ├── models/
│   │   ├── cssm.py             # Core CSSM implementations
│   │   ├── cssm_vit.py         # CSSM-ViT architecture
│   │   ├── simple_cssm.py      # SimpleCSSM for Pathfinder/cABC
│   │   ├── baseline_vit.py     # Standard ViT baseline
│   │   ├── goom.py             # GOOM log-space primitives
│   │   └── math.py             # Associative scan operations
│   └── training/
│       └── train_imagenet.py   # ImageNet training script
├── scripts/
│   ├── convert_pathfinder_tfrecords.py
│   └── convert_cabc_tfrecords.py
└── docs/
    ├── models.md               # CSSM variant documentation
    ├── pathfinder.md           # Pathfinder training guide
    ├── cabc.md                 # cABC training guide
    ├── imagenet.md             # ImageNet training guide
    └── multigpu.md             # Multi-GPU setup
```

## Multi-GPU Training

Multi-GPU training is enabled automatically:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py ...
```

See [docs/multigpu.md](docs/multigpu.md) for details.

## Checkpointing

Use local storage for checkpoints to avoid NFS issues:

```bash
python main.py ... --checkpoint_dir /local/scratch/checkpoints

# Resume training
python main.py ... --resume /local/scratch/checkpoints/run_name/epoch_50
```

## Experimental Features

The following are experimental and may not be fully maintained:

- `src/cotracker/` - CoTracker optical flow (JAX port)
- `src/jepa/` - JEPA self-supervised learning
- `src/training/train_jepa.py` - JEPA training script

## Citation

```bibtex
@article{cssm2026,
  title={Cepstral State Space Models for Vision},
  author={...},
  year={2026}
}
```

## License

MIT License
