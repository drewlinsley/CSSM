"""
Convert PathTracker .npy videos to TFRecord format for fast I/O.

Each .npy file is a (64, 32, 32, 3) uint8 video. We store each video as a
single TFRecord example with the raw bytes, plus metadata (label, shape).

Usage:
    python scripts/convert_pathtracker_tfrecords.py \
        --input_dir /media/data_cifs/projects/prj_video_datasets/pathtracker \
        --output_dir /media/data_cifs/projects/prj_video_datasets/pathtracker_tfrecords \
        --samples_per_shard 1000

Output structure:
    output_dir/
        train/
            shard_0000.tfrecord
            shard_0001.tfrecord
            ...
        val/
            shard_0000.tfrecord
            ...
        test/
            shard_0000.tfrecord
            ...
        metadata.json
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
    """Create a TFRecord example from a video."""
    feature = {
        'video': bytes_feature(video_bytes),
        'label': int64_feature(label),
        'num_frames': int64_feature(num_frames),
        'height': int64_feature(height),
        'width': int64_feature(width),
        'channels': int64_feature(channels),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def collect_files(input_dir):
    """Collect all .npy files with labels, sorted by sample number."""
    input_path = Path(input_dir)
    files = []

    for label in [0, 1]:
        label_dir = input_path / str(label)
        if not label_dir.exists():
            print(f"WARNING: {label_dir} does not exist, skipping")
            continue
        npy_files = sorted(
            label_dir.glob('*.npy'),
            key=lambda p: int(re.search(r'_sample_(\d+)', p.stem).group(1))
        )
        for fpath in npy_files:
            files.append((str(fpath), label))

    return files


def write_shards(files, output_dir, samples_per_shard, split_name):
    """Write a list of (filepath, label) to TFRecord shards."""
    os.makedirs(output_dir, exist_ok=True)

    shard_idx = 0
    sample_idx = 0
    writer = None
    total_written = 0

    for filepath, label in files:
        # Open new shard if needed
        if sample_idx % samples_per_shard == 0:
            if writer is not None:
                writer.close()
            shard_path = os.path.join(output_dir, f'shard_{shard_idx:04d}.tfrecord')
            writer = tf.io.TFRecordWriter(shard_path)
            shard_idx += 1

        # Load video and normalize to float32
        video = np.load(filepath)  # (64, 32, 32, 3) uint8
        video_f32 = (video.astype(np.float32) / np.float32(255.0) - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

        T, H, W, C = video_f32.shape
        example = make_example(
            video_bytes=video_f32.tobytes(),
            label=label,
            num_frames=T,
            height=H,
            width=W,
            channels=C,
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
    parser = argparse.ArgumentParser(description='Convert PathTracker .npy to TFRecords')
    parser.add_argument('--input_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_datasets/pathtracker',
                        help='PathTracker dataset root (contains 0/ and 1/)')
    parser.add_argument('--output_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_datasets/pathtracker_tfrecords',
                        help='Output directory for TFRecords')
    parser.add_argument('--samples_per_shard', type=int, default=1000,
                        help='Samples per TFRecord shard')
    parser.add_argument('--train_size', type=int, default=100000,
                        help='Number of training samples')
    parser.add_argument('--test_size', type=int, default=10000,
                        help='Number of test samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for split')
    args = parser.parse_args()

    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")

    # Collect all files
    files = collect_files(args.input_dir)
    print(f"Found {len(files)} total samples")

    if len(files) == 0:
        print("ERROR: No .npy files found")
        sys.exit(1)

    # Shuffle and split (same logic as pathtracker_data.py)
    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)

    train_size = min(args.train_size, len(files))
    test_size = min(args.test_size, len(files) - train_size)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]

    # Carve val from train (last 10%)
    n_val = max(1, train_size // 10)
    val_indices = train_indices[-n_val:]
    train_indices = train_indices[:-n_val]

    splits = {
        'train': [files[i] for i in train_indices],
        'val': [files[i] for i in val_indices],
        'test': [files[i] for i in test_indices],
    }

    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Read actual shape from first file
    sample_video = np.load(files[0][0])
    actual_frames, actual_h, actual_w, actual_c = sample_video.shape
    print(f"Video shape: ({actual_frames}, {actual_h}, {actual_w}, {actual_c})")
    print()

    # Write each split
    metadata = {
        'total_samples': len(files),
        'image_size': actual_h,
        'num_frames': actual_frames,
        'channels': actual_c,
        'seed': args.seed,
    }

    for split_name, split_files in splits.items():
        split_dir = os.path.join(args.output_dir, split_name)
        n_written, n_shards = write_shards(
            split_files, split_dir, args.samples_per_shard, split_name
        )
        metadata[f'{split_name}_samples'] = n_written
        metadata[f'{split_name}_shards'] = n_shards

    # Write metadata
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to {metadata_path}")
    print("Done!")


if __name__ == '__main__':
    main()
