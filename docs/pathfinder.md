# Pathfinder Task Training Guide

The Pathfinder task tests contour integration - the ability to trace a connected path among distractors.

## Dataset

The Pathfinder dataset consists of grayscale images with two markers connected (or not) by a curvy path among distractor paths. The task is binary classification: connected or not connected.

**Difficulties:**
- `9`: Short contours (easier)
- `14`: Medium contours
- `20`: Long contours (harder)

## Recommended Configuration

```bash
python main.py \
    --arch simple \
    --cssm hgru_bi \
    --dataset pathfinder \
    --pathfinder_difficulty 14 \
    --tfrecord_dir /path/to/pathfinder_tfrecords \
    --batch_size 256 \
    --seq_len 8 \
    --depth 1 \
    --embed_dim 32 \
    --kernel_size 11 \
    --lr 3e-4 \
    --epochs 60 \
    --pos_embed spatiotemporal \
    --bf16
```

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--arch` | `simple` | Minimal architecture for this task |
| `--cssm` | `hgru_bi` | E/I dynamics critical for contour integration |
| `--depth` | 1 | Single CSSM block sufficient |
| `--embed_dim` | 32 | Small embedding for efficiency |
| `--kernel_size` | 11-15 | Larger kernels = larger receptive field |
| `--seq_len` | 8 | Number of recurrence steps |
| `--pos_embed` | `spatiotemporal` or `sinusoidal` | Position encoding |

## Position Embedding Options

For Pathfinder, consider:

- `spatiotemporal`: Full VideoRoPE encoding (default)
- `sinusoidal`: Better length generalization
- `spatial_only`: If testing temporal generalization

## Expected Performance

| Difficulty | Expected Accuracy |
|------------|-------------------|
| 9 | 95%+ |
| 14 | 90%+ |
| 20 | 85%+ |

## TFRecord Data Loading

For best I/O performance, use TFRecord format:

```bash
--tfrecord_dir /path/to/difficulty_14
```

The TFRecord loader provides:
- Better throughput than image-based loading
- Automatic shuffling
- Prefetching for GPU utilization

## Tips

1. **Start with difficulty 9** to validate setup
2. **Use `--bf16`** for faster training on modern GPUs
3. **Monitor validation accuracy** - overfitting is common
4. **Use `--save_best_only`** to only save best checkpoint

## Debugging

If training fails:

1. Check data path: `ls /path/to/pathfinder_tfrecords`
2. Verify TFRecord files exist for difficulty level
3. Try smaller batch size if OOM
4. Use `--no_wandb` for quick tests

## Multi-GPU Training

```bash
python main.py \
    --arch simple \
    --cssm hgru_bi \
    --dataset pathfinder \
    --pathfinder_difficulty 14 \
    --batch_size 512 \  # Will be split across GPUs
    ...
```

Multi-GPU is automatic when multiple GPUs are detected.
