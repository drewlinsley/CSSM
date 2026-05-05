#!/usr/bin/env python3
"""
Convert cABC (contour ABC) dataset to TFRecords for fast training.

cABC is a binary classification task similar to Pathfinder, with images
organized in easy/medium/hard difficulties with train/test splits.

Structure: {difficulty}/images/{train,test}/{pos,neg}/{1,2,...N}/*.png

Usage:
    python scripts/convert_cabc_tfrecords.py \
        --input_dir /media/data_cifs_lrs/projects/prj_LRA/cabc \
        --output_dir /media/data_cifs/projects/prj_video_imagenet/data/cabc_tfrecords \
        --difficulty easy \
        --image_size 224 \
        --num_workers 16

    # Convert all difficulties:
    for d in easy medium hard; do
        python scripts/convert_cabc_tfrecords.py --difficulty $d
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
    # Load image (cABC images may be grayscale or RGB)
    img = Image.open(img_path)

    # Convert to RGB if grayscale
    if img.mode == 'L':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to target size
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # Convert to numpy and normalize to [0, 1]
    img_np = np.array(img, dtype=np.float32) / 255.0

    # Apply ImageNet normalization
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    return img_np


def collect_samples(images_dir: Path) -> list:
    """Collect all image samples from pos/neg subdirectories.

    cABC structure: {pos,neg}/{1,2,...N}/sample_*.png
    """
    samples = []

    for label_name, label in [('neg', 0), ('pos', 1)]:
        label_dir = images_dir / label_name
        if not label_dir.exists():
            print(f"Warning: {label_dir} does not exist")
            continue

        # Find all PNG files in subdirectories
        for subdir in label_dir.iterdir():
            if subdir.is_dir():
                for img_path in subdir.glob('*.png'):
                    samples.append((str(img_path), label))

    return samples


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
    parser = argparse.ArgumentParser(description='Convert cABC to TFRecords')
    parser.add_argument('--input_dir', type=str,
                        default='/media/data_cifs_lrs/projects/prj_LRA/cabc',
                        help='Root directory containing difficulty folders')
    parser.add_argument('--output_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_imagenet/data/cabc_tfrecords',
                        help='Output directory for TFRecords')
    parser.add_argument('--difficulty', type=str, default='easy',
                        choices=['easy', 'medium', 'hard'],
                        help='cABC difficulty level')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Target image size')
    parser.add_argument('--samples_per_shard', type=int, default=5000,
                        help='Number of samples per TFRecord shard')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    difficulty_dir = input_dir / args.difficulty

    if not difficulty_dir.exists():
        print(f"Error: Difficulty directory not found: {difficulty_dir}")
        return

    print("=" * 60)
    print("cABC TFRecord Converter")
    print("=" * 60)
    print(f"Input: {difficulty_dir}")
    print(f"Output: {output_dir}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Image size: {args.image_size}")
    print(f"Workers: {args.num_workers}")
    print("=" * 60)

    # Create output directory
    diff_output_dir = output_dir / args.difficulty
    diff_output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'difficulty': args.difficulty,
        'image_size': args.image_size,
        'seed': args.seed,
    }

    ctx = mp.get_context('spawn')

    # Process train and test splits
    for split in ['train', 'test']:
        images_dir = difficulty_dir / 'images' / split

        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping {split}")
            continue

        print(f"\nCollecting {split} samples...")
        samples = collect_samples(images_dir)

        if len(samples) == 0:
            print(f"No samples found in {images_dir}")
            continue

        # Shuffle samples
        rng = np.random.RandomState(args.seed)
        rng.shuffle(samples)

        n_samples = len(samples)
        n_pos = sum(1 for _, label in samples if label == 1)
        n_neg = n_samples - n_pos

        print(f"Found {n_samples} samples ({n_pos} pos, {n_neg} neg)")

        # Create split output directory
        split_dir = diff_output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create shards
        n_shards = (n_samples + args.samples_per_shard - 1) // args.samples_per_shard
        shards = []

        for i in range(n_shards):
            start = i * args.samples_per_shard
            end = min(start + args.samples_per_shard, n_samples)
            shard_samples = samples[start:end]
            output_path = split_dir / f'shard_{i:05d}.tfrecord'
            shards.append((shard_samples, str(output_path), args.image_size))

        print(f"Processing {split}: {n_samples} samples -> {n_shards} shards")

        # Process shards in parallel
        total_successful = 0
        total_failed = 0

        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
            futures = {executor.submit(process_shard, shard): shard for shard in shards}

            with tqdm(total=n_shards, desc=f"  {split}") as pbar:
                for future in as_completed(futures):
                    successful, failed, status = future.result()
                    total_successful += successful
                    total_failed += failed

                    if status != "ok":
                        tqdm.write(f"    Shard error: {status}")

                    pbar.update(1)
                    pbar.set_postfix({'ok': total_successful, 'fail': total_failed})

        metadata[f'{split}_samples'] = total_successful
        metadata[f'{split}_shards'] = n_shards
        metadata[f'{split}_pos'] = n_pos
        metadata[f'{split}_neg'] = n_neg

        print(f"  Successful: {total_successful}, Failed: {total_failed}")

    # Save metadata
    metadata['total_samples'] = metadata.get('train_samples', 0) + metadata.get('test_samples', 0)
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
