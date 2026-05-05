# Spectral State Space Models (SCSSM)

Spatiotemporal state space models with FFT-based spatial mixing for vision. Replaces attention with spectral convolutions that provide global receptive fields in O(N log N) through the convolution theorem: pointwise multiplication in the frequency domain equals circular convolution in the spatial domain.

## Key Finding

A single feedforward pass of `iFFT(C * B * FFT(x))` — with learned, input-dependent per-frequency gates B and C — achieves 86% on Pathfinder contour integration (CL-14) and 83% on PathTracker object tracking (15-distractor) **with no temporal recurrence** (T=1). Removing the FFT (same architecture, same gates, no spectral transform) drops performance to chance. The FFT + input-dependent gating is the critical mechanism.

## Model Variants

**Primary architectures** (spectral SSMs with FFT-based spatial mixing):

| Model | CLI Flag | Description | PF-14 | PathTracker |
|-------|----------|-------------|-------|-------------|
| **GDN-SCSSM** | `--cssm gdn` | Matrix delta rule with spectral kernels, QKV projections, RMSNorm output | **90.8%** | **84.6%** |
| **Mamba-SCSSM** | `--cssm gated` | Mamba-style scalar recurrence with per-frequency B/C/Delta gating | 87.3% | 81.9% |
| **Mamba-SCSSM 1x1** | `--cssm gated --kernel_size 1` | Same as above but with 1x1 spectral kernel — all spatial mixing via FFT gates | 86.6% | 83.4% |
| **SCSSM** | `--cssm no_gate` | Fixed spectral kernel, no input-dependent gates. Needs T>1. | 86.1% | 80.7% |

**Controls** (these fail, demonstrating FFT is essential):

| Model | CLI Flag | Description | PF-14 | PathTracker |
|-------|----------|-------------|-------|-------------|
| Mamba-CSSM (no FFT) | `--cssm no_fft` | Same Mamba architecture, no FFT transform | 50.7% | 50.5% |
| S5-CSSM | `--cssm conv_ssm` | Spatial conv + temporal scan, no FFT | 50.1% | 50.9% |
| Transformer (spatial) | `--cssm spatial_attn` | Single-head self-attention | 55.0% | — |
| Transformer (ST) | `--cssm spatiotemporal_attn` | Spatiotemporal attention | 50.1% | 53.8% |

Human accuracy: Pathfinder 89%, PathTracker 90%.

## How It Works

The core operation at each timestep:

```
1. in_proj(x) -> (x_main, z)           # channel expansion + split
2. x_main = SiLU(x_main)               # nonlinearity
3. X_hat = FFT2(x_main)                # to frequency domain
4. B, C, Delta = gates(spatial_mean(x)) # input-dependent per-frequency weights
5. h_t = A_eff * h_{t-1} + Delta*B*X_hat  # recurrence in spectral domain
6. y = iFFT2(C * h_t)                  # back to spatial domain
7. output = y * SiLU(z)                # multiplicative gating
8. output = out_proj(output)            # channel mixing
```

**At T=1** (no recurrence): reduces to `iFFT(C * B * FFT(SiLU(x)))` — a learned, input-dependent global convolution. The B and C gates project from the spatial mean of x to per-frequency weights, implementing a dynamic matched filter over the full spatial extent.

**At T>1**: the spectral kernel A adds frequency-selective resonance amplification via the geometric series `sum(A^k)`, which converges to `(I-A)^{-1}` — an IIR filter in the frequency domain.

For a complete derivation, see [docs/spectral_spatial_mixing.md](docs/spectral_spatial_mixing.md).

## Installation

```bash
git clone <repo-url>
cd CSSM
pip install -r requirements.txt
```

Requires JAX with CUDA 12+ support. Tested on NVIDIA B200, H100, A100 GPUs.

## Quick Start

### Pathfinder (Contour Integration)

```bash
# GDN-SCSSM (best accuracy)
python main.py --arch simple --cssm gdn --gate_type factored \
    --dataset pathfinder --pathfinder_difficulty 14 --image_size 128 \
    --tfrecord_dir /path/to/pathfinder_tfrecords_128 \
    --embed_dim 64 --kernel_size 11 --qkv_conv_size 5 --delta_key_dim 4 \
    --seq_len 8 --depth 1 --pool_type max --pos_embed mrope \
    --norm_type global_layer --stem_layers 1 --stem_norm_order post \
    --epochs 120 --lr 1e-3 --batch_size 128

# Mamba-SCSSM 1x1 T=1 (fastest, no recurrence needed)
python main.py --arch simple --cssm gated --gate_type factored \
    --dataset pathfinder --pathfinder_difficulty 14 --image_size 128 \
    --tfrecord_dir /path/to/pathfinder_tfrecords_128 \
    --embed_dim 64 --kernel_size 1 --short_conv_spatial_size 0 \
    --seq_len 1 --depth 1 --pool_type max --pos_embed mrope \
    --norm_type global_layer --stem_layers 1 --stem_norm_order post \
    --epochs 120 --lr 1e-3 --batch_size 128
```

### PathTracker (Object Tracking)

```bash
python main.py --arch simple --cssm gdn --gate_type factored \
    --dataset girik --image_size 32 --seq_len 32 \
    --tfrecord_dir /path/to/15_dist_tfrecords \
    --embed_dim 64 --kernel_size 11 --delta_key_dim 4 \
    --stem_mode pathtracker --frame_readout last \
    --depth 1 --epochs 120 --lr 1e-3 --batch_size 128
```

### ImageNet (CSSM-SHViT)

```bash
python src/training/train_imagenet.py \
    --model cssm_shvit --model_size s1 \
    --cssm_type gated --num_timesteps 1 \
    --kernel_size 1 --short_conv_spatial_size 0 \
    --shvit_recipe --epochs 300 --image_size 224 \
    --batch_size 512 --data_loader tfrecord \
    --tfrecord_dir /path/to/imagenet_tfrecords
```

## Supported Tasks

| Task | Dataset | Description |
|------|---------|-------------|
| Contour integration | Pathfinder (CL-9/14/20/25) | Binary: are two dots on the same contour? |
| Object tracking | PathTracker (15-distractor) | Binary: was the target dot at the cued position? |
| Image classification | ImageNet-1k | 1000-class classification via CSSM-SHViT |

## Key Hyperparameters

| Flag | Description | Recommended |
|------|-------------|-------------|
| `--cssm` | Model variant | `gdn` (best accuracy) or `gated` (fastest) |
| `--kernel_size` | Spectral kernel size | `11` (standard) or `1` (pure FFT gating) |
| `--gate_type` | Gate parameterization | `factored` (separable, fewer params) |
| `--seq_len` | Recurrence timesteps | `1` for images, native T for video |
| `--embed_dim` | Channel dimension | `64` (Pathfinder) or `256+` (ImageNet) |
| `--norm_type` | Normalization | `global_layer` (best for recurrent models) |
| `--delta_key_dim` | GDN key dimension | `4` (Pathfinder) or `2` (PathTracker) |

## Project Structure

```
CSSM/
├── main.py                          # Training entry point (Pathfinder, PathTracker)
├── src/
│   ├── models/
│   │   ├── cssm.py                  # All CSSM variants (GatedCSSM, GatedDeltaNetCSSM, NoGateCSSM, etc.)
│   │   ├── simple_cssm.py           # SimpleCSSM wrapper (stem -> CSSM blocks -> readout)
│   │   ├── cssm_shvit.py            # CSSM-SHViT for ImageNet
│   │   ├── shvit.py                 # SHViT baseline (single-head attention)
│   │   ├── goom.py                  # GOOM log-space primitives
│   │   └── math.py                  # Associative scan operations
│   ├── pathfinder_data.py           # Pathfinder data loader (TFRecord + PNG)
│   ├── pathtracker_data.py          # PathTracker data loader
│   ├── girik_data.py                # 15-distractor PathTracker loader
│   ├── data/
│   │   └── imagenet_tfdata.py       # ImageNet TFRecord loader
│   └── training/
│       ├── train_imagenet.py        # ImageNet training script
│       ├── distributed.py           # Multi-GPU utilities
│       └── optimizers.py            # Optimizer factory (AdamW, LAMB, AGC)
├── scripts/
│   ├── sweep_pathfinder.sh          # Pathfinder hyperparameter sweep
│   ├── sweep_15dist.sh              # PathTracker sweep
│   ├── sweep_imagenet_edge.sh       # ImageNet edge model sweep
│   ├── plot_main_3x2.py             # Main results figure
│   ├── plot_timestep_curves_main.py # Timestep ablation figure
│   ├── convert_pathfinder_tfrecords.py
│   └── convert_imagenet_tfrecords.py
└── docs/
    ├── spectral_spatial_mixing.md   # Why FFT + gating = global convolution
    ├── architecture.md              # Architecture deep dive
    ├── imagenet.md                  # ImageNet training guide
    └── pathfinder.md                # Pathfinder guide
```

## Citation

```bibtex
@article{scssm2026,
  title={Spectral State Space Models for Vision},
  author={},
  year={2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
