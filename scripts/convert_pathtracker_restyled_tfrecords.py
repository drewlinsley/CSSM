"""
Convert PathTracker .npy videos to TFRecords with restyled dots:
  - Red/blue targets: unfilled 3x3 squares (hollow center)
  - Green distractors: 1x1 pixels

Usage:
    python scripts/convert_pathtracker_restyled_tfrecords.py \
        --input_dir /media/data_cifs/projects/prj_video_datasets/pathtracker_equal_large \
        --output_dir /media/data_cifs/projects/prj_video_datasets/pathtracker_restyled_32f_tfrecords \
        --num_frames 32
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(video_bytes, label, num_frames, height, width, channels):
    feature = {
        'video': bytes_feature(video_bytes),
        'label': int64_feature(label),
        'num_frames': int64_feature(num_frames),
        'height': int64_feature(height),
        'width': int64_feature(width),
        'channels': int64_feature(channels),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def restyle_frame(frame):
    """Restyle a single frame: targets → unfilled 3x3 squares, distractors → 1px."""
    h, w, _ = frame.shape
    out = np.zeros_like(frame)

    # Detect colored pixels
    red_mask = (frame[:, :, 0] > 200) & (frame[:, :, 1] < 50) & (frame[:, :, 2] < 50)
    blue_mask = (frame[:, :, 2] > 200) & (frame[:, :, 0] < 50) & (frame[:, :, 1] < 50)
    green_mask = (frame[:, :, 1] > 200) & (frame[:, :, 0] < 50) & (frame[:, :, 2] < 50)

    # Green distractors → 1x1 at top-left of each 2x2 dot
    visited = np.zeros((h, w), bool)
    gy, gx = np.where(green_mask)
    for y, x in zip(gy, gx):
        if not visited[y, x]:
            for dy in range(2):
                for dx in range(2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and green_mask[ny, nx]:
                        visited[ny, nx] = True
            out[y, x] = [0, 255, 0]

    # Red target → unfilled 3x3 square
    ry, rx = np.where(red_mask)
    if len(ry) > 0:
        cy, cx = int(ry.mean()), int(rx.mean())
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] = [255, 0, 0]

    # Blue target → unfilled 3x3 square
    by, bx = np.where(blue_mask)
    if len(by) > 0:
        cy, cx = int(by.mean()), int(bx.mean())
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] = [0, 0, 255]

    return out


def restyle_video(video, target_frames=None):
    """Restyle all frames, optionally subsample."""
    if target_frames is not None and target_frames < video.shape[0]:
        indices = np.linspace(0, video.shape[0] - 1, target_frames, dtype=int)
        video = video[indices]

    return np.stack([restyle_frame(f) for f in video])


def collect_files(input_dir):
    input_path = Path(input_dir)
    files = []
    for label in [0, 1]:
        label_dir = input_path / str(label)
        if not label_dir.exists():
            print(f"WARNING: {label_dir} does not exist, skipping")
            continue
        npy_files = sorted(
            label_dir.glob('*.npy'),
            key=lambda p: int(re.search(r'_(\d+)', p.stem).group(1))
        )
        for fpath in npy_files:
            files.append((str(fpath), label))
    return files


def write_shards(files, output_dir, samples_per_shard, split_name, target_frames=None):
    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    sample_idx = 0
    writer = None
    total_written = 0

    for filepath, label in files:
        if sample_idx % samples_per_shard == 0:
            if writer is not None:
                writer.close()
            shard_path = os.path.join(output_dir, f'shard_{shard_idx:04d}.tfrecord')
            writer = tf.io.TFRecordWriter(shard_path)
            shard_idx += 1

        video = np.load(filepath)  # (64, 32, 32, 3) uint8
        video = restyle_video(video, target_frames)

        # Normalize to float32 (ImageNet stats)
        video_f32 = (video.astype(np.float32) / 255.0
                     - np.array([0.485, 0.456, 0.406], dtype=np.float32)) \
                    / np.array([0.229, 0.224, 0.225], dtype=np.float32)

        T, H, W, C = video_f32.shape
        example = make_example(
            video_bytes=video_f32.tobytes(),
            label=label, num_frames=T, height=H, width=W, channels=C,
        )
        writer.write(example.SerializeToString())

        sample_idx += 1
        total_written += 1
        if total_written % 5000 == 0:
            print(f"  {split_name}: {total_written}/{len(files)} written ({shard_idx} shards)")

    if writer is not None:
        writer.close()
    print(f"  {split_name}: {total_written} samples in {shard_idx} shards -> {output_dir}")
    return total_written, shard_idx


def main():
    parser = argparse.ArgumentParser(description='Convert PathTracker .npy to restyled TFRecords')
    parser.add_argument('--input_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_datasets/pathtracker_equal_large')
    parser.add_argument('--output_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_datasets/pathtracker_restyled_32f_tfrecords')
    parser.add_argument('--samples_per_shard', type=int, default=1000)
    parser.add_argument('--train_size', type=int, default=81000)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Subsample to this many frames (e.g. 32). Default: all frames.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")

    files = collect_files(args.input_dir)
    print(f"Found {len(files)} total samples")
    if len(files) == 0:
        print("ERROR: No .npy files found")
        sys.exit(1)

    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)

    train_size = min(args.train_size, len(files))
    test_size = min(args.test_size, len(files) - train_size)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]

    n_val = max(1, train_size // 10)
    val_indices = train_indices[-n_val:]
    train_indices = train_indices[:-n_val]

    splits = {
        'train': [files[i] for i in train_indices],
        'val': [files[i] for i in val_indices],
        'test': [files[i] for i in test_indices],
    }
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    sample_video = np.load(files[0][0])
    orig_frames = sample_video.shape[0]
    out_frames = args.num_frames if args.num_frames else orig_frames
    print(f"Original frames: {orig_frames}, output frames: {out_frames}")
    print(f"Restyling: targets → unfilled 3x3 squares, distractors → 1px")
    print()

    metadata = {
        'total_samples': len(files),
        'image_size': sample_video.shape[1],
        'num_frames': out_frames,
        'channels': sample_video.shape[3],
        'seed': args.seed,
        'style': 'restyled: unfilled 3x3 target squares, 1px distractors',
    }

    for split_name, split_files in splits.items():
        split_dir = os.path.join(args.output_dir, split_name)
        n_written, n_shards = write_shards(
            split_files, split_dir, args.samples_per_shard, split_name,
            target_frames=args.num_frames
        )
        metadata[f'{split_name}_samples'] = n_written
        metadata[f'{split_name}_shards'] = n_shards

    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to {metadata_path}")
    print("Done!")


if __name__ == '__main__':
    main()
