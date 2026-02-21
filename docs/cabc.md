# cABC Task Training Guide

The cABC (contour ABC) task tests contour completion with shape discrimination.

## Dataset

Images contain partially occluded contours. The task is to classify which shape the contour belongs to (A, B, or C - 3-way classification).

**Difficulties:**
- `easy`: Minimal occlusion
- `medium`: Moderate occlusion
- `hard`: Heavy occlusion

## Recommended Configuration

```bash
python main.py \
    --arch simple \
    --cssm hgru_bi \
    --dataset cabc \
    --cabc_difficulty medium \
    --tfrecord_dir /path/to/cabc_tfrecords \
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
| `--arch` | `simple` | Minimal architecture |
| `--cssm` | `hgru_bi` | Contour integration requires E/I |
| `--depth` | 1 | Single CSSM block usually sufficient |
| `--embed_dim` | 32 | Small for efficiency |
| `--kernel_size` | 11 | 11-15 works well |

## Differences from Pathfinder

- **3-way classification** instead of binary
- Requires **shape discrimination** not just connectivity
- May benefit from **more recurrence steps** (`--seq_len 12`)

## Expected Performance

| Difficulty | Expected Accuracy |
|------------|-------------------|
| easy | 90%+ |
| medium | 85%+ |
| hard | 75%+ |

## TFRecord Data

```bash
--tfrecord_dir /path/to/cabc_tfrecords
```

Note: cABC uses `test` split for validation (no separate val split).

## Tips

1. **Harder than Pathfinder** - may need longer training
2. **Consider `--depth 2`** for hard difficulty
3. **Larger kernel** (`--kernel_size 15`) may help with occluded contours

## Training Progress

Monitor both:
- Training accuracy (should improve steadily)
- Validation accuracy (watch for overfitting)

If validation accuracy plateaus while training continues improving, consider:
- Stronger regularization (`--weight_decay 1e-3`)
- Earlier stopping
- Data augmentation (if implemented)
