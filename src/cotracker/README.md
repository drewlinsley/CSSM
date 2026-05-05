# CSSM-CoTracker: Point Tracking with Cepstral State Space Models

JAX/Flax implementation of video point tracking using CSSM instead of transformer attention.

## Overview

CoTracker uses transformer attention for temporal and spatial reasoning in point tracking. We replace these attention mechanisms with CSSM (Cepstral State Space Models) to achieve:

- **O(log T)** temporal processing via associative scan (vs O(T²) for attention)
- **O(N log N)** spatial processing via FFT (vs O(N²) for attention)
- **Biologically-inspired dynamics** via opponent channel formulation

## Structure

```
src/cotracker/
├── jax/
│   ├── encoder.py      # BasicEncoder and CSSMEncoder
│   ├── correlation.py  # Correlation computation
│   └── cotracker.py    # Full CSSM-CoTracker model
├── data/
│   └── kubric.py       # Kubric dataset loader
└── train.py            # Training script
```

## Quick Start

### 1. Download Kubric Dataset

```bash
# Download (~100-200GB, takes several hours)
python scripts/download_kubric.py --output_dir /path/to/save

# Or use Python directly:
from huggingface_hub import snapshot_download
snapshot_download("facebook/CoTracker3_Kubric", local_dir="./data/kubric")
```

### 2. Train CSSM-CoTracker

```bash
# Basic training with ResNet encoder
python -m src.cotracker.train --cssm_type opponent --no_wandb

# With CSSM-SHViT encoder (can use JEPA weights)
python -m src.cotracker.train --use_cssm_encoder --encoder_size s1

# With pretrained JEPA weights
python -m src.cotracker.train --use_cssm_encoder \
    --jepa_checkpoint /path/to/jepa/checkpoint
```

## Model Usage

```python
from src.cotracker.jax.cotracker import create_cssm_cotracker
import jax.numpy as jnp

# Create model
model = create_cssm_cotracker(
    cssm_type='opponent',      # 'standard', 'opponent', or 'hgru'
    use_cssm_encoder=True,     # Use CSSM-SHViT encoder
)

# Initialize
rng = jax.random.PRNGKey(0)
video = jnp.zeros((1, 24, 384, 384, 3))   # (B, T, H, W, C)
queries = jnp.zeros((1, 100, 3))          # (B, N, 3) - [frame_idx, x, y]

params = model.init(rng, video, queries, training=False)

# Track points
outputs = model.apply(params, video, queries, training=False)
# outputs['coords']: (B, T, N, 2) - predicted coordinates
# outputs['vis']: (B, T, N, 1) - visibility scores
```

## Architecture

### CSSM-CoTracker Pipeline

```
Video (B, T, H, W, 3)
       │
       ▼
┌─────────────────────┐
│  Encoder            │  BasicEncoder (ResNet-style) or
│  (Feature Network)  │  CSSMEncoder (CSSM-SHViT with JEPA weights)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  CorrBlock          │  Multi-scale correlation pyramid
│  (Correlations)     │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  CSSMUpdateBlock    │  CSSM temporal scan + spatial MLP
│  (Iterative Update) │  Replaces transformer attention
└─────────────────────┘
       │
       ▼
Output: coords (B, T, N, 2), visibility (B, T, N, 1)
```

### CSSM Dynamics (Opponent Mode)

```
[X_t]   [alpha  -K_I*mu ] [X_{t-1}]   [U_t]
[Y_t] = [K_E*gamma delta] [Y_{t-1}] + [0  ]
```

- **X**: Excitation channel (tracks points)
- **Y**: Inhibition channel (provides contrast normalization)
- Computed via `jax.lax.associative_scan` in O(log T)

## Training Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 4 | Batch size per GPU |
| `--lr` | 5e-4 | Learning rate |
| `--num_steps` | 50000 | Total training steps |
| `--sequence_length` | 24 | Frames per sample |
| `--num_points` | 256 | Points per sample |
| `--cssm_type` | opponent | CSSM variant |

## References

- [CoTracker3](https://github.com/facebookresearch/co-tracker)
- [Kubric](https://github.com/google-research/kubric)
- [CSSM Paper](https://arxiv.org/abs/TODO)
