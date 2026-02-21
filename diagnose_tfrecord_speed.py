#!/usr/bin/env python3
"""
Diagnose TFRecord data loading speed.
Tests raw read speed, parsing speed, and full pipeline throughput.
"""

import os
import sys
import time
import argparse

# Disable GPU for TensorFlow (we just want CPU data loading)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


def test_raw_read_speed(tfrecord_files, num_records=1000):
    """Test raw file read speed (no parsing)."""
    print("\n" + "="*60)
    print("TEST 1: Raw TFRecord Read Speed")
    print("="*60)

    ds = tf.data.TFRecordDataset(tfrecord_files)

    start = time.time()
    count = 0
    total_bytes = 0
    for record in ds:
        total_bytes += len(record.numpy())
        count += 1
        if count >= num_records:
            break
    elapsed = time.time() - start

    mb_read = total_bytes / (1024 * 1024)
    print(f"  Records read: {count}")
    print(f"  Total data: {mb_read:.2f} MB")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {mb_read/elapsed:.2f} MB/s")
    print(f"  Records/sec: {count/elapsed:.1f}")

    return mb_read / elapsed


def test_parse_speed(tfrecord_files, num_records=1000, image_shape=(224, 224, 3)):
    """Test parsing speed (decode + reshape)."""
    print("\n" + "="*60)
    print("TEST 2: Parse Speed (decode + reshape)")
    print("="*60)

    def parse_fn(example):
        features = tf.io.parse_single_example(example, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
        image = tf.io.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, image_shape)
        return image, features['label']

    ds = tf.data.TFRecordDataset(tfrecord_files).map(parse_fn)

    start = time.time()
    count = 0
    for img, lbl in ds:
        count += 1
        if count >= num_records:
            break
    elapsed = time.time() - start

    print(f"  Records parsed: {count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Records/sec: {count/elapsed:.1f}")

    return count / elapsed


def test_pipeline_speed(tfrecord_files, num_records=1000, batch_size=32,
                        image_shape=(224, 224, 3), num_parallel=4):
    """Test full pipeline with batching and prefetch."""
    print("\n" + "="*60)
    print(f"TEST 3: Full Pipeline (batch={batch_size}, parallel={num_parallel})")
    print("="*60)

    def parse_fn(example):
        features = tf.io.parse_single_example(example, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
        image = tf.io.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, image_shape)
        return image, features['label']

    ds = (tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=num_parallel)
          .map(parse_fn, num_parallel_calls=num_parallel)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))

    start = time.time()
    count = 0
    batches = 0
    for img_batch, lbl_batch in ds:
        count += img_batch.shape[0]
        batches += 1
        if count >= num_records:
            break
    elapsed = time.time() - start

    print(f"  Batches: {batches}")
    print(f"  Records: {count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Batches/sec: {batches/elapsed:.1f}")
    print(f"  Records/sec: {count/elapsed:.1f}")

    return count / elapsed


def test_file_system_speed(tfrecord_files):
    """Test basic file system speed with dd-like read."""
    print("\n" + "="*60)
    print("TEST 4: Raw File System Speed")
    print("="*60)

    test_file = tfrecord_files[0]
    file_size = os.path.getsize(test_file)

    # Read file in chunks
    chunk_size = 1024 * 1024  # 1MB chunks
    start = time.time()
    bytes_read = 0
    with open(test_file, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            bytes_read += len(chunk)
    elapsed = time.time() - start

    mb_read = bytes_read / (1024 * 1024)
    print(f"  File: {os.path.basename(test_file)}")
    print(f"  Size: {mb_read:.2f} MB")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {mb_read/elapsed:.2f} MB/s")

    return mb_read / elapsed


def main():
    parser = argparse.ArgumentParser(description='Diagnose TFRecord loading speed')
    parser.add_argument('--tfrecord_dir', type=str,
                        default='/home/dlinsley/pathfinder_tfrecord/',
                        help='Directory containing TFRecord files')
    parser.add_argument('--split', type=str, default='train',
                        help='Which split to test (train/val)')
    parser.add_argument('--num_records', type=int, default=1000,
                        help='Number of records to test')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for pipeline test')
    parser.add_argument('--image_shape', type=str, default='224,224,3',
                        help='Image shape as H,W,C')
    args = parser.parse_args()

    # Parse image shape
    image_shape = tuple(int(x) for x in args.image_shape.split(','))

    # Find TFRecord files
    split_dir = os.path.join(args.tfrecord_dir, args.split)
    if os.path.isdir(split_dir):
        tfrecord_files = sorted(tf.io.gfile.glob(os.path.join(split_dir, '*.tfrecord')))
    else:
        tfrecord_files = sorted(tf.io.gfile.glob(os.path.join(args.tfrecord_dir, '*.tfrecord')))

    if not tfrecord_files:
        print(f"ERROR: No TFRecord files found in {args.tfrecord_dir}")
        print(f"  Tried: {split_dir}")
        print(f"  And: {args.tfrecord_dir}")
        sys.exit(1)

    print("="*60)
    print("TFRECORD SPEED DIAGNOSTIC")
    print("="*60)
    print(f"Directory: {args.tfrecord_dir}")
    print(f"Split: {args.split}")
    print(f"Files found: {len(tfrecord_files)}")
    print(f"First file: {tfrecord_files[0]}")
    print(f"Image shape: {image_shape}")

    # Run tests
    results = {}

    results['raw_fs'] = test_file_system_speed(tfrecord_files)
    results['raw_tfrecord'] = test_raw_read_speed(tfrecord_files, args.num_records)
    results['parse'] = test_parse_speed(tfrecord_files, args.num_records, image_shape)
    results['pipeline'] = test_pipeline_speed(
        tfrecord_files, args.num_records, args.batch_size, image_shape
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Raw file system:    {results['raw_fs']:.1f} MB/s")
    print(f"  Raw TFRecord read:  {results['raw_tfrecord']:.1f} MB/s")
    print(f"  Parse speed:        {results['parse']:.1f} records/s")
    print(f"  Full pipeline:      {results['pipeline']:.1f} records/s")

    # Diagnosis
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if results['raw_fs'] < 50:
        print("  ⚠️  SLOW FILE SYSTEM - NFS/network storage bottleneck")
        print("      Consider copying data to local scratch")
    elif results['raw_fs'] < 200:
        print("  ⚠️  MODERATE FILE SYSTEM - May be NFS, could be faster")
    else:
        print("  ✓  File system speed looks good")

    if results['pipeline'] < 100:
        print("  ⚠️  SLOW PIPELINE - Data loading may bottleneck training")
    elif results['pipeline'] < 500:
        print("  ⚠️  MODERATE PIPELINE - OK for small batch sizes")
    else:
        print("  ✓  Pipeline speed looks good")

    # Expected training throughput
    expected_batches_per_sec = results['pipeline'] / args.batch_size
    print(f"\n  Expected max throughput: ~{expected_batches_per_sec:.1f} batches/sec")
    print(f"  (Actual will be lower due to GPU compute)")


if __name__ == '__main__':
    main()
