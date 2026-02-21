# ImageNet Training Guide

Training CSSM models on ImageNet for image classification.

## Recommended Configuration

```bash
python main.py \
    --arch vit \
    --cssm hgru_bi \
    --dataset imagenet \
    --imagenet_dir /path/to/imagenet/ILSVRC2012 \
    --batch_size 256 \
    --seq_len 8 \
    --depth 12 \
    --embed_dim 384 \
    --kernel_size 11 \
    --lr 1e-4 \
    --epochs 300 \
    --pos_embed spatiotemporal \
    --bf16
```

## Architecture Options

### CSSM-ViT (Recommended)

```bash
--arch vit --depth 12 --embed_dim 384
```

- Similar architecture to ViT-Small
- CSSM replaces attention in each block
- Patch or conv stem (`--stem_mode conv` recommended)

### Model Sizes

| Config | Depth | Embed Dim | Params |
|--------|-------|-----------|--------|
| Tiny | 12 | 192 | ~5M |
| Small | 12 | 384 | ~22M |
| Base | 12 | 768 | ~86M |

## Stem Options

### Conv Stem (Default)
```bash
--stem_mode conv --stem_stride 4 --stem_norm layer
```
- Single conv layer with GELU + norm
- Faster than patch embedding

### Patch Stem
```bash
--stem_mode patch --patch_size 16
```
- Standard ViT-style patch embedding

## Training Schedule

Recommended for 300 epochs:
- Warmup: 500 steps
- Peak LR: 1e-4
- Cosine decay to 1e-6
- Weight decay: 1e-4
- Gradient clipping: 1.0

## Multi-GPU Training

ImageNet training benefits significantly from multi-GPU:

```bash
# Automatically uses all available GPUs
python main.py \
    --arch vit \
    --dataset imagenet \
    --batch_size 512 \  # Total batch size, split across GPUs
    ...
```

For 8x GPUs, effective batch size = 512 (64 per GPU).

## Data Directory Structure

Expected ImageNet layout:
```
/path/to/imagenet/ILSVRC2012/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

## Checkpointing

For long ImageNet runs:

```bash
--checkpoint_dir checkpoints \
--save_every 10 \
--checkpointer simple  # NFS-safe
```

To resume training:
```bash
--resume checkpoints/run_name/epoch_100
```

## Performance Optimization

1. **Enable bf16**: `--bf16` (2x faster on A100/H100)
2. **Adjust workers**: `--num_workers 8`
3. **Prefetch batches**: `--prefetch_batches 4`

## Expected Performance

| Model | Epochs | Top-1 Acc |
|-------|--------|-----------|
| CSSM-ViT-Tiny | 300 | ~72% |
| CSSM-ViT-Small | 300 | ~78% |

## Tips

1. **Start with smaller model** (tiny) to validate setup
2. **Use WandB** for tracking long runs
3. **Monitor GPU utilization** - should be >90%
4. **Check NaN** - use `--checkpointer simple` on NFS
