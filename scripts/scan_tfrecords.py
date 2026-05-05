#!/usr/bin/env python3
"""
Scan TFRecords to find corrupted images.

Usage:
    python scripts/scan_tfrecords.py --tfrecord_dir /path/to/tfrecords/train
    python scripts/scan_tfrecords.py --tfrecord_dir /oscar/scratch/dlinsley/imagenet_tfrecords/train
"""
import argparse
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import json

# Disable GPU for scanning
tf.config.set_visible_devices([], 'GPU')


def scan_tfrecord(tfrecord_path: Path) -> list:
    """Scan a single TFRecord file for corrupted images.

    Returns list of (record_index, error_message) for corrupted records.
    """
    corrupted = []

    try:
        ds = tf.data.TFRecordDataset(str(tfrecord_path))

        for idx, record in enumerate(ds):
            try:
                parsed = tf.io.parse_single_example(record, {
                    'image': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64),
                })
                # Try to decode the image
                image = tf.io.decode_image(parsed['image'], channels=3, expand_animations=False)
                # Force evaluation
                _ = image.numpy()
            except tf.errors.InvalidArgumentError as e:
                corrupted.append((idx, str(e)[:100]))
            except Exception as e:
                corrupted.append((idx, f"Unknown error: {str(e)[:100]}"))

    except Exception as e:
        return [(-1, f"Failed to read file: {str(e)[:100]}")]

    return corrupted


def main():
    parser = argparse.ArgumentParser(description='Scan TFRecords for corrupted images')
    parser.add_argument('--tfrecord_dir', type=str, required=True,
                        help='Directory containing TFRecord files')
    parser.add_argument('--output', type=str, default='corrupted_tfrecords.json',
                        help='Output JSON file with corruption report')
    parser.add_argument('--split', type=str, default=None,
                        help='Subdirectory (train/val) - if not specified, scans tfrecord_dir directly')
    args = parser.parse_args()

    tfrecord_dir = Path(args.tfrecord_dir).expanduser()

    if args.split:
        tfrecord_dir = tfrecord_dir / args.split

    if not tfrecord_dir.exists():
        print(f"Error: Directory not found: {tfrecord_dir}")
        return

    tfrecord_files = sorted(tfrecord_dir.glob('*.tfrecord'))

    if not tfrecord_files:
        print(f"No .tfrecord files found in {tfrecord_dir}")
        return

    print(f"Scanning {len(tfrecord_files)} TFRecord files in {tfrecord_dir}")
    print("=" * 60)

    results = {
        'tfrecord_dir': str(tfrecord_dir),
        'total_files': len(tfrecord_files),
        'corrupted_files': [],
        'total_corrupted_images': 0,
    }

    corrupted_files = []
    total_corrupted = 0

    for tfrecord_file in tqdm(tfrecord_files, desc="Scanning"):
        corrupted = scan_tfrecord(tfrecord_file)

        if corrupted:
            total_corrupted += len(corrupted)
            corrupted_files.append({
                'file': str(tfrecord_file),
                'filename': tfrecord_file.name,
                'num_corrupted': len(corrupted),
                'corrupted_records': corrupted,
            })
            tqdm.write(f"  Found {len(corrupted)} corrupted in: {tfrecord_file.name}")

    results['corrupted_files'] = corrupted_files
    results['total_corrupted_images'] = total_corrupted
    results['num_affected_shards'] = len(corrupted_files)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    print(f"Total TFRecord files scanned: {len(tfrecord_files)}")
    print(f"Files with corrupted images:  {len(corrupted_files)}")
    print(f"Total corrupted images:       {total_corrupted}")
    print(f"Results saved to:             {args.output}")

    if corrupted_files:
        print("\nAffected files:")
        for cf in corrupted_files:
            print(f"  {cf['filename']}: {cf['num_corrupted']} corrupted")

        print(f"\nTo re-encode these shards, you'll need to:")
        print("1. Find the source images for these shards")
        print("2. Re-run the TFRecord conversion for just these shards")
        print("3. Or check if source images themselves are corrupted")
    else:
        print("\nNo corrupted images found!")


if __name__ == '__main__':
    main()
