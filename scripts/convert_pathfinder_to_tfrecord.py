#!/usr/bin/env python3
"""
Convert Pathfinder dataset to TFRecord format for fast streaming.

Pre-processes images and stores them as sharded TFRecord files.
This eliminates image loading/resizing overhead during training.

Usage:
    python scripts/convert_pathfinder_to_tfrecord.py \
        --input_dir /media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025 \
        --output_dir /scratch/pathfinder_tfrecord \
        --difficulty 9 \
        --image_size 224 \
        --num_workers 8

Output format:
    Each TFRecord contains serialized examples with:
    - image: preprocessed float32 array (H, W, 3)
    - label: int64 (0 or 1)
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not installed. Install with: pip install tensorflow")

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_and_preprocess_image(img_path: str, image_size: int) -> np.ndarray:
    """Load and preprocess a single image."""
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


def process_image(args) -> Tuple[np.ndarray, int, str]:
    """Process a single image (for parallel execution)."""
    img_path, label, image_size = args
    try:
        img = load_and_preprocess_image(img_path, image_size)
        return img, label, None
    except Exception as e:
        return None, label, str(e)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_example(image: np.ndarray, label: int) -> bytes:
    """Serialize image and label to TFRecord format."""
    feature = {
        'image': _bytes_feature(image.tobytes()),
        'label': _int64_feature(label),
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'channels': _int64_feature(image.shape[2]),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_all_images(input_dir: str, difficulty: str) -> List[Tuple[str, int]]:
    """Get all image paths and labels."""
    difficulty_dir = Path(input_dir) / f'curv_contour_length_{difficulty}' / 'imgs'

    images = []

    # Negative examples (label=0)
    neg_dir = difficulty_dir / 'neg'
    if neg_dir.exists():
        for img_path in sorted(neg_dir.glob('*.png')):
            images.append((str(img_path), 0))

    # Positive examples (label=1)
    pos_dir = difficulty_dir / 'pos'
    if pos_dir.exists():
        for img_path in sorted(pos_dir.glob('*.png')):
            images.append((str(img_path), 1))

    return images


def main():
    parser = argparse.ArgumentParser(description='Convert Pathfinder to TFRecord')
    parser.add_argument('--input_dir', type=str,
                        default='/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025',
                        help='Root directory containing difficulty folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for TFRecord shards')
    parser.add_argument('--difficulty', type=str, default='9',
                        choices=['9', '14', '20'],
                        help='Pathfinder difficulty level')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Target image size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--shard_size', type=int, default=5000,
                        help='Samples per shard')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits')
    args = parser.parse_args()

    if not HAS_TF:
        print("Error: TensorFlow not installed")
        sys.exit(1)

    # Get all images
    print(f"Finding images in {args.input_dir} (difficulty={args.difficulty})...")
    all_images = get_all_images(args.input_dir, args.difficulty)
    print(f"Found {len(all_images)} images")

    if len(all_images) == 0:
        print("Error: No images found!")
        sys.exit(1)

    # Shuffle and split
    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(all_images))
    rng.shuffle(indices)

    n_train = int(len(all_images) * args.train_ratio)
    n_val = int(len(all_images) * args.val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }

    # Create output directory
    output_dir = Path(args.output_dir) / f'difficulty_{args.difficulty}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split_name, split_indices in splits.items():
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split ({len(split_indices)} samples)")
        print(f"{'='*50}")

        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        # Prepare work items
        work_items = [
            (all_images[idx][0], all_images[idx][1], args.image_size)
            for idx in split_indices
        ]

        # Process images in parallel
        processed = []
        failed = 0

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_image, item) for item in work_items]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_name}"):
                img, label, error = future.result()
                if error is None:
                    processed.append((img, label))
                else:
                    failed += 1

        print(f"  Processed: {len(processed)}, Failed: {failed}")

        # Shuffle processed samples
        rng.shuffle(processed)

        # Write TFRecord shards
        shard_idx = 0
        sample_idx = 0
        writer = None

        for img, label in tqdm(processed, desc=f"Writing {split_name} shards"):
            if sample_idx % args.shard_size == 0:
                if writer is not None:
                    writer.close()
                shard_path = split_dir / f'pathfinder_{split_name}_{shard_idx:05d}.tfrecord'
                writer = tf.io.TFRecordWriter(str(shard_path))
                shard_idx += 1

            example = serialize_example(img, label)
            writer.write(example)
            sample_idx += 1

        if writer is not None:
            writer.close()

        print(f"  Wrote {shard_idx} shards to {split_dir}")

    # Write metadata
    metadata = {
        'difficulty': args.difficulty,
        'image_size': args.image_size,
        'train_samples': len(splits['train']),
        'val_samples': len(splits['val']),
        'test_samples': len(splits['test']),
        'shard_size': args.shard_size,
        'normalization': 'imagenet',
    }

    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print("Conversion complete!")
    print(f"  Output: {output_dir}")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
