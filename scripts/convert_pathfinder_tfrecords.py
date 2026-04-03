#!/usr/bin/env python3
"""
Convert Pathfinder dataset to TFRecords for fast training.

Pathfinder is a binary classification task with grayscale images stored in
pos/neg folders. This script converts them to TFRecords with train/val/test splits.

Usage:
    python scripts/convert_pathfinder_tfrecords.py \
        --input_dir /media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025 \
        --output_dir /media/data_cifs/projects/prj_video_imagenet/data/pathfinder_tfrecords \
        --difficulty 9 \
        --image_size 224 \
        --num_workers 16

    # Convert all difficulties:
    for d in 9 14 20; do
        python scripts/convert_pathfinder_tfrecords.py --difficulty $d
    done
"""

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from PIL import Image
from tqdm import tqdm

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _bytes_feature(value, tf):
    """Returns a bytes_list from bytes."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value, tf):
    """Returns an int64_list from int."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_and_preprocess_image(img_path: Path, image_size: int) -> np.ndarray:
    """Load, resize, and normalize a single image."""
    # Load grayscale image
    img = Image.open(img_path).convert('L')

    # Resize to target size
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # Convert to numpy and normalize to [0, 1]
    img_np = np.array(img, dtype=np.float32) / 255.0

    # Convert grayscale to RGB by repeating channels
    img_rgb = np.stack([img_np, img_np, img_np], axis=-1)  # (H, W, 3)

    # Apply ImageNet normalization
    img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD

    return img_rgb


def process_shard(args) -> tuple:
    """Process a shard of images into a TFRecord file."""
    samples, output_path, image_size = args

    # Import TF inside subprocess
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    successful = 0
    failed = 0

    try:
        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for img_path, label in samples:
                try:
                    img_path = Path(img_path)
                    image = load_and_preprocess_image(img_path, image_size)

                    # Serialize to bytes
                    image_bytes = image.tobytes()

                    feature = {
                        'image': _bytes_feature(image_bytes, tf),
                        'label': _int64_feature(label, tf),
                        'height': _int64_feature(image_size, tf),
                        'width': _int64_feature(image_size, tf),
                        'channels': _int64_feature(3, tf),
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    successful += 1

                except Exception as e:
                    failed += 1

    except Exception as e:
        return (successful, failed, str(e))

    return (successful, failed, "ok")


def main():
    parser = argparse.ArgumentParser(description='Convert Pathfinder to TFRecords')
    parser.add_argument('--input_dir', type=str,
                        default='/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025',
                        help='Root directory containing difficulty folders')
    parser.add_argument('--output_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_imagenet/data/pathfinder_tfrecords',
                        help='Output directory for TFRecords')
    parser.add_argument('--difficulty', type=str, default='9',
                        choices=['9', '14', '20', '25'],
                        help='Pathfinder difficulty level')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Target image size')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--samples_per_shard', type=int, default=5000,
                        help='Number of samples per TFRecord shard')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    difficulty_dir = input_dir / f'curv_contour_length_{args.difficulty}'
    imgs_dir = difficulty_dir / 'imgs'

    if not imgs_dir.exists():
        print(f"Error: Images directory not found: {imgs_dir}")
        return

    print("=" * 60)
    print("Pathfinder TFRecord Converter")
    print("=" * 60)
    print(f"Input: {difficulty_dir}")
    print(f"Output: {output_dir}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Image size: {args.image_size}")
    print(f"Workers: {args.num_workers}")
    print("=" * 60)

    # Collect all samples
    samples = []

    # Negative examples (label=0)
    neg_dir = imgs_dir / 'neg'
    if neg_dir.exists():
        neg_images = sorted(neg_dir.glob('*.png'))
        for img_path in neg_images:
            samples.append((str(img_path), 0))
        print(f"Found {len(neg_images)} negative samples")

    # Positive examples (label=1)
    pos_dir = imgs_dir / 'pos'
    if pos_dir.exists():
        pos_images = sorted(pos_dir.glob('*.png'))
        for img_path in pos_images:
            samples.append((str(img_path), 1))
        print(f"Found {len(pos_images)} positive samples")

    if len(samples) == 0:
        print("Error: No images found!")
        return

    print(f"Total samples: {len(samples)}")

    # Shuffle and split
    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    n_total = len(samples)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }

    print(f"\nSplits:")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val: {len(val_indices)}")
    print(f"  Test: {len(test_indices)}")

    # Create output directory
    diff_output_dir = output_dir / f'difficulty_{args.difficulty}'
    diff_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    metadata = {
        'difficulty': args.difficulty,
        'image_size': args.image_size,
        'total_samples': n_total,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
    }

    ctx = mp.get_context('spawn')

    for split_name, split_indices in splits.items():
        split_samples = [samples[i] for i in split_indices]
        n_samples = len(split_samples)

        if n_samples == 0:
            continue

        # Create split output directory
        split_dir = diff_output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create shards
        n_shards = (n_samples + args.samples_per_shard - 1) // args.samples_per_shard
        shards = []

        for i in range(n_shards):
            start = i * args.samples_per_shard
            end = min(start + args.samples_per_shard, n_samples)
            shard_samples = split_samples[start:end]
            output_path = split_dir / f'shard_{i:05d}.tfrecord'
            shards.append((shard_samples, str(output_path), args.image_size))

        print(f"\nProcessing {split_name}: {n_samples} samples -> {n_shards} shards")

        # Process shards in parallel
        total_successful = 0
        total_failed = 0

        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
            futures = {executor.submit(process_shard, shard): shard for shard in shards}

            with tqdm(total=n_shards, desc=f"  {split_name}") as pbar:
                for future in as_completed(futures):
                    successful, failed, status = future.result()
                    total_successful += successful
                    total_failed += failed

                    if status != "ok":
                        tqdm.write(f"    Shard error: {status}")

                    pbar.update(1)
                    pbar.set_postfix({'ok': total_successful, 'fail': total_failed})

        metadata[f'{split_name}_samples'] = total_successful
        metadata[f'{split_name}_shards'] = n_shards

        print(f"  Successful: {total_successful}, Failed: {total_failed}")

    # Save metadata
    metadata_path = diff_output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {diff_output_dir}")
    print(f"Metadata: {metadata_path}")

    # Estimate size
    total_size = sum(f.stat().st_size for f in diff_output_dir.rglob('*.tfrecord'))
    print(f"Total size: {total_size / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
