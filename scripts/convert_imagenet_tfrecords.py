#!/usr/bin/env python3
"""
Convert ImageNet dataset to TFRecord format for fast sequential reading.

Usage:
    python scripts/convert_imagenet_tfrecords.py \
        --input_dir /gpfs/data/shared/imagenet/ILSVRC2012 \
        --output_dir ~/scratch/imagenet_tfrecords \
        --num_shards 1024

Output structure:
    ~/scratch/imagenet_tfrecords/
    ├── train/
    │   ├── train-00000-of-01024.tfrecord
    │   └── ...
    └── val/
        ├── val-00000-of-00128.tfrecord
        └── ...
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import threading

import tensorflow as tf
from tqdm import tqdm


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Create bytes feature for TFRecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    """Create int64 feature for TFRecord."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(image_path: str, label: int) -> tf.train.Example:
    """Create a TFRecord Example from image path and label."""
    # Read raw JPEG bytes (don't decode - store compressed)
    with open(image_path, 'rb') as f:
        image_data = f.read()

    feature = {
        'image': _bytes_feature(image_data),
        'label': _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def load_ilsvrc_to_synset_mapping(input_dir: Path) -> dict:
    """Load mapping from ILSVRC2012_ID (1-1000) to synset WNID from devkit."""
    meta_paths = [
        input_dir / 'ILSVRC2012_devkit_t12' / 'data' / 'meta.mat',
        input_dir / 'devkit' / 'data' / 'meta.mat',
    ]

    for meta_path in meta_paths:
        if meta_path.exists():
            try:
                import scipy.io as sio
                meta = sio.loadmat(str(meta_path), squeeze_me=True)
                synsets = meta['synsets']
                ilsvrc_to_synset = {}
                for s in synsets:
                    if len(s) >= 2:
                        ilsvrc_id = int(s[0])
                        wnid = str(s[1])
                        if ilsvrc_id <= 1000:
                            ilsvrc_to_synset[ilsvrc_id] = wnid
                if len(ilsvrc_to_synset) == 1000:
                    print(f"  Loaded ILSVRC->synset mapping from {meta_path}")
                    return ilsvrc_to_synset
            except Exception as e:
                print(f"  Warning: Could not load meta.mat: {e}")

    # Try synset_words.txt
    synset_paths = [
        input_dir / 'synset_words.txt',
        input_dir / 'LOC_synset_mapping.txt',
    ]

    for synset_path in synset_paths:
        if synset_path.exists():
            try:
                ilsvrc_to_synset = {}
                with open(synset_path, 'r') as f:
                    for i, line in enumerate(f, 1):
                        wnid = line.strip().split()[0]
                        ilsvrc_to_synset[i] = wnid
                if len(ilsvrc_to_synset) == 1000:
                    print(f"  Loaded ILSVRC->synset mapping from {synset_path}")
                    return ilsvrc_to_synset
            except Exception as e:
                print(f"  Warning: Could not load synset file: {e}")

    return None


def get_image_label_pairs(split_dir: Path, input_dir: Path) -> Tuple[List[Tuple[str, int]], dict]:
    """
    Get list of (image_path, label) pairs from ImageNet directory structure.

    Handles both:
    - Standard structure: split/class_folder/images
    - Flat structure: split/images (with ground truth file)

    Returns:
        samples: List of (path, label) tuples
        class_to_idx: Dict mapping class name to index
    """
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    if len(class_dirs) > 0:
        # Standard structure: split/class_folder/images
        class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

        samples = []
        for class_dir in class_dirs:
            label = class_to_idx[class_dir.name]
            for ext in ['*.JPEG', '*.jpeg', '*.jpg', '*.JPG', '*.png', '*.PNG']:
                for img_path in class_dir.glob(ext):
                    samples.append((str(img_path), label))
    else:
        # Flat structure: split/images (typically val set)
        print(f"  Flat directory structure detected, loading ground truth labels...")

        train_dir = input_dir / 'train'
        if not train_dir.exists():
            raise ValueError(f"Train directory not found for class mapping: {train_dir}")

        train_class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        class_to_idx = {d.name: i for i, d in enumerate(train_class_dirs)}

        # Load ILSVRC_ID -> synset mapping
        ilsvrc_to_synset = load_ilsvrc_to_synset_mapping(input_dir)

        # Look for ground truth labels file
        gt_paths = [
            input_dir / 'ILSVRC2012_devkit_t12' / 'data' / 'ILSVRC2012_validation_ground_truth.txt',
            input_dir / 'devkit' / 'data' / 'ILSVRC2012_validation_ground_truth.txt',
            input_dir / 'val_ground_truth.txt',
            input_dir / 'val_labels.txt',
            input_dir / 'ILSVRC2012_validation_ground_truth.txt',
        ]

        gt_file = None
        for p in gt_paths:
            if p.exists():
                gt_file = p
                break

        if gt_file is None:
            raise ValueError(
                f"Flat val directory found but no ground truth labels file. "
                f"Looked for: {[str(p) for p in gt_paths]}"
            )

        # Load ground truth ILSVRC_IDs
        with open(gt_file, 'r') as f:
            ilsvrc_ids = [int(line.strip()) for line in f.readlines()]

        # Convert ILSVRC_ID -> synset WNID -> our alphabetical class index
        if ilsvrc_to_synset is not None:
            gt_labels = []
            for ilsvrc_id in ilsvrc_ids:
                synset = ilsvrc_to_synset.get(ilsvrc_id)
                if synset and synset in class_to_idx:
                    gt_labels.append(class_to_idx[synset])
                else:
                    gt_labels.append(ilsvrc_id - 1)  # Fallback
        else:
            print("  WARNING: No ILSVRC->synset mapping found, labels may be incorrect!")
            gt_labels = [x - 1 for x in ilsvrc_ids]

        # Get all image files sorted by name
        image_files = []
        for ext in ['*.JPEG', '*.jpeg', '*.jpg', '*.JPG', '*.png', '*.PNG']:
            image_files.extend(split_dir.glob(ext))
        image_files = sorted(image_files, key=lambda x: x.name)

        if len(image_files) != len(gt_labels):
            raise ValueError(
                f"Mismatch: {len(image_files)} images vs {len(gt_labels)} labels"
            )

        samples = [(str(img), label) for img, label in zip(image_files, gt_labels)]

    return samples, class_to_idx


def write_shard(
    shard_id: int,
    num_shards: int,
    samples: List[Tuple[str, int]],
    output_dir: Path,
    split: str,
) -> int:
    """Write a single TFRecord shard."""
    output_path = output_dir / f"{split}-{shard_id:05d}-of-{num_shards:05d}.tfrecord"

    count = 0
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for image_path, label in samples:
            try:
                example = create_example(image_path, label)
                writer.write(example.SerializeToString())
                count += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    return count


def convert_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    num_shards: int,
    num_workers: int = 16,
):
    """Convert a single split (train or val) to TFRecords."""
    split_dir = input_dir / split
    output_split_dir = output_dir / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {split} split...")

    # Get all samples (pass input_dir for flat val directory handling)
    samples, class_to_idx = get_image_label_pairs(split_dir, input_dir)
    print(f"  Found {len(samples)} images in {len(class_to_idx)} classes")

    # Save class mapping
    class_map_path = output_dir / f"{split}_class_mapping.txt"
    with open(class_map_path, 'w') as f:
        for name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{name}\n")
    print(f"  Saved class mapping to {class_map_path}")

    # Shuffle samples for better distribution across shards
    import random
    random.shuffle(samples)

    # Split samples into shards
    samples_per_shard = len(samples) // num_shards
    shards = []
    for i in range(num_shards):
        start = i * samples_per_shard
        if i == num_shards - 1:
            # Last shard gets remainder
            end = len(samples)
        else:
            end = start + samples_per_shard
        shards.append(samples[start:end])

    # Write shards in parallel
    total_written = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(write_shard, i, num_shards, shard_samples, output_split_dir, split): i
            for i, shard_samples in enumerate(shards)
        }

        with tqdm(total=num_shards, desc=f"  Writing {split} shards") as pbar:
            for future in as_completed(futures):
                count = future.result()
                total_written += count
                pbar.update(1)

    print(f"  Wrote {total_written} examples to {num_shards} shards")


def main():
    parser = argparse.ArgumentParser(description='Convert ImageNet to TFRecords')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to ImageNet root (contains train/, val/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for TFRecords')
    parser.add_argument('--num_shards', type=int, default=1024,
                        help='Number of shards for train set (default: 1024)')
    parser.add_argument('--val_shards', type=int, default=128,
                        help='Number of shards for val set (default: 128)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers (default: 16)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'],
                        help='Splits to convert (default: train val)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir).expanduser()

    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting ImageNet to TFRecords")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")

    for split in args.splits:
        num_shards = args.num_shards if split == 'train' else args.val_shards
        convert_split(input_dir, output_dir, split, num_shards, args.num_workers)

    print(f"\nDone! TFRecords saved to {output_dir}")


if __name__ == '__main__':
    main()
