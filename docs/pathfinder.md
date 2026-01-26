# Pathfinder Training Guide

The Pathfinder task tests long-range spatial integration: determining whether two dots are connected by a contour.

## Dataset

Pathfinder consists of binary classification images with:
- Two marker dots
- Background clutter (distractor curves)
- A potential connecting contour

**Difficulties**:
- `9`: Short contours (easiest)
- `14`: Medium contours (default)
- `20`: Long contours (hardest)

## Data Preparation

### Option 1: TFRecords (Recommended)

TFRecords provide faster I/O, especially on networked storage.

```bash
# Convert PNG dataset to TFRecords
python scripts/convert_pathfinder_tfrecords.py \
    --input_dir /path/to/pathfinder/difficulty_14 \
    --output_dir /path/to/pathfinder_tfrecords/difficulty_14
```

### Option 2: PNG Files

Direct loading from PNG files (slower, but no preprocessing needed).

## Training Commands

### Recommended Configuration

```bash
python main.py --arch simple --cssm hgru_bi --dataset pathfinder \
    --pathfinder_difficulty 14 \
    --tfrecord_dir /path/to/pathfinder_tfrecords/difficulty_14 \
    --batch_size 256 --seq_len 8 --depth 1 --embed_dim 32 \
    --kernel_size 11 --lr 3e-4 --epochs 60 \
    --pos_embed spatiotemporal --bf16
```

### Without TFRecords

```bash
python main.py --arch simple --cssm hgru_bi --dataset pathfinder \
    --pathfinder_difficulty 14 \
    --data_dir /path/to/pathfinder \
    --batch_size 256 --seq_len 8 --depth 1 --embed_dim 32 \
    --kernel_size 11 --lr 3e-4 --epochs 60 \
    --pos_embed spatiotemporal
```

## Key Hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--arch` | `simple` | Minimal architecture for this task |
| `--cssm` | `hgru_bi` | 3x3 opponent dynamics |
| `--pos_embed` | `spatiotemporal` | **Essential** - required to learn the task |
| `--depth` | `1` | Single CSSM block sufficient |
| `--embed_dim` | `32` | Small dimension works well |
| `--seq_len` | `8` | Temporal recurrence steps |
| `--kernel_size` | `11` | Spatial kernel size |
| `--batch_size` | `256` | Adjust based on GPU memory |
| `--lr` | `3e-4` | Learning rate |
| `--epochs` | `60` | Training epochs |

## Expected Results

| Difficulty | Expected Accuracy |
|------------|-------------------|
| 9 | ~95%+ |
| 14 | ~90%+ |
| 20 | ~85%+ |

## Troubleshooting

**Out of Memory**: Reduce `--batch_size` or use `--bf16`

**Slow Training**:
- Use TFRecords instead of PNG loading
- Use `--checkpoint_dir /local/path` to avoid NFS issues

**NaN Loss**:
- Try reducing `--lr` to `1e-4`
- Ensure `--bf16` is enabled for better numerical stability
