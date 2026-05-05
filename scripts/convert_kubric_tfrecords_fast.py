#!/usr/bin/env python3
"""
Optimized Kubric TFRecord converter for fast training.

Key differences from convert_kubric_tfrecords.py:
1. Shards multiple sequences per TFRecord (100 per file)
2. Stores video as raw uint8 tensor (faster decode, ~3x larger)
3. Pre-crops to training resolution
4. Stores fixed sequence length (no variable parsing)

Usage:
    python scripts/convert_kubric_tfrecords_fast.py \
        --input_dir /media/data_cifs/projects/prj_video_imagenet/data/kubric \
        --output_dir /media/data_cifs/projects/prj_video_imagenet/data/kubric_tfrecords_fast \
        --num_workers 32 \
        --sequences_per_shard 100 \
        --crop_size 384 \
        --sequence_length 24
"""
import argparse
import io
import os
import tarfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

# Note: TensorFlow imported inside functions to avoid multiprocessing issues


def _bytes_feature(value, tf):
    """Returns a bytes_list from bytes."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value, tf):
    """Returns a float_list from float array."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def _int64_feature(value, tf):
    """Returns an int64_list from int."""
    if isinstance(value, (int, np.integer)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def load_sequence_from_tar(
    tar_path: Path,
    sequence_length: int,
    crop_size: int,
    num_points: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load and preprocess a single sequence from tar.gz."""
    try:
        import imageio.v3 as imageio
    except ImportError:
        import imageio

    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = tar.getnames()
            seq_name = members[0].split('/')[0]

            # Find frame files
            frame_members = sorted([m for m in members if '/frames/' in m and m.endswith('.png')])

            if len(frame_members) < sequence_length:
                return None

            # Always take first sequence_length frames for consistency
            frame_members = frame_members[:sequence_length]

            # Load frames
            frames = []
            for member_name in frame_members:
                f = tar.extractfile(member_name)
                img = imageio.imread(io.BytesIO(f.read()))
                if img.shape[-1] == 4:  # RGBA -> RGB
                    img = img[..., :3]
                frames.append(img)

            video = np.stack(frames, axis=0)  # (T, H, W, 3)
            T, H, W, C = video.shape

            # Center crop to crop_size
            if H > crop_size:
                top = (H - crop_size) // 2
                video = video[:, top:top + crop_size, :, :]
            if W > crop_size:
                left = (W - crop_size) // 2
                video = video[:, :, left:left + crop_size, :]

            # Load trajectories
            trajs_file = f"{seq_name}/{seq_name}_trajs_2d.npy"
            vis_file = f"{seq_name}/{seq_name}_visibility.npy"

            if trajs_file in members:
                trajs_data = tar.extractfile(trajs_file)
                trajs = np.load(io.BytesIO(trajs_data.read())).astype(np.float32)
                trajs = trajs[:sequence_length]
            else:
                trajs = np.zeros((sequence_length, num_points, 2), dtype=np.float32)

            if vis_file in members:
                vis_data = tar.extractfile(vis_file)
                visibility = np.load(io.BytesIO(vis_data.read())).astype(np.float32)
                visibility = visibility[:sequence_length]
            else:
                visibility = np.ones((sequence_length, trajs.shape[1]), dtype=np.float32)

            # Adjust trajectories for crop
            if H > crop_size:
                trajs[..., 1] -= (H - crop_size) // 2
            if W > crop_size:
                trajs[..., 0] -= (W - crop_size) // 2

            # Sample/pad points to fixed size
            N = trajs.shape[1]
            if N > num_points:
                indices = np.random.choice(N, num_points, replace=False)
                trajs = trajs[:, indices]
                visibility = visibility[:, indices]
            elif N < num_points:
                pad_size = num_points - N
                trajs = np.pad(trajs, ((0, 0), (0, pad_size), (0, 0)), mode='edge')
                visibility = np.pad(visibility, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)

            return video, trajs, visibility

    except Exception as e:
        print(f"Error loading {tar_path}: {e}")
        return None


def process_shard(
    args: Tuple[List[str], str, int, int, int, int, int]
) -> Tuple[int, int, str]:
    """Process a shard of tar files into a single TFRecord."""
    tar_paths, output_path, shard_idx, sequence_length, crop_size, num_points, total_shards = args

    # Convert strings back to Paths
    tar_paths = [Path(p) for p in tar_paths]
    output_path = Path(output_path)

    # Import TF inside subprocess (avoids multiprocessing issues)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    successful = 0
    failed = 0
    errors = []

    try:
        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for tar_path in tar_paths:
                try:
                    result = load_sequence_from_tar(tar_path, sequence_length, crop_size, num_points)

                    if result is None:
                        failed += 1
                        errors.append(f"{tar_path.name}: returned None")
                        continue

                    video, trajs, visibility = result

                    # Store video as raw bytes (much faster to decode than JPEG)
                    video_bytes = video.tobytes()

                    feature = {
                        'video': _bytes_feature(video_bytes, tf),
                        'video_shape': _int64_feature(list(video.shape), tf),
                        'trajs': _float_feature(trajs, tf),
                        'visibility': _float_feature(visibility, tf),
                        'sequence_length': _int64_feature(sequence_length, tf),
                        'crop_size': _int64_feature(crop_size, tf),
                        'num_points': _int64_feature(num_points, tf),
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    successful += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"{tar_path.name}: {str(e)[:50]}")

        error_str = "; ".join(errors[:3]) if errors else "ok"
        return (successful, failed, error_str)

    except Exception as e:
        return (successful, failed, f"shard error: {str(e)[:100]}")


def main():
    parser = argparse.ArgumentParser(description='Convert Kubric to optimized TFRecords')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing Kubric tar.gz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for TFRecords')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of parallel workers')
    parser.add_argument('--sequences_per_shard', type=int, default=100,
                        help='Number of sequences per TFRecord shard')
    parser.add_argument('--sequence_length', type=int, default=24,
                        help='Fixed sequence length')
    parser.add_argument('--crop_size', type=int, default=384,
                        help='Pre-crop to this size')
    parser.add_argument('--num_points', type=int, default=256,
                        help='Fixed number of points per sequence')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of sequences (for testing)')
    parser.add_argument('--test', action='store_true',
                        help='Test loading a single sequence (no multiprocessing)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    # Find all tar.gz files
    tar_files = sorted(input_dir.glob('*.tar.gz'))

    if not tar_files:
        print(f"No tar.gz files found in {input_dir}")
        return

    print(f"Found {len(tar_files)} tar.gz files")

    # Test mode: load single sequence without multiprocessing
    if args.test:
        print("\n=== TEST MODE ===")
        test_tar = tar_files[0]
        print(f"Testing: {test_tar}")

        result = load_sequence_from_tar(
            test_tar, args.sequence_length, args.crop_size, args.num_points
        )

        if result is None:
            print("ERROR: load_sequence_from_tar returned None")
        else:
            video, trajs, visibility = result
            print(f"SUCCESS!")
            print(f"  Video shape: {video.shape} dtype={video.dtype}")
            print(f"  Trajs shape: {trajs.shape} dtype={trajs.dtype}")
            print(f"  Visibility shape: {visibility.shape} dtype={visibility.dtype}")
            print(f"  Video range: [{video.min()}, {video.max()}]")
            print(f"  Trajs range: [{trajs.min():.1f}, {trajs.max():.1f}]")
        return

    if args.limit:
        tar_files = tar_files[:args.limit]
    print(f"Output directory: {output_dir}")
    print(f"Sequences per shard: {args.sequences_per_shard}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Crop size: {args.crop_size}")
    print(f"Num points: {args.num_points}")
    print(f"Workers: {args.num_workers}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split into shards (convert Paths to strings for pickling)
    num_shards = (len(tar_files) + args.sequences_per_shard - 1) // args.sequences_per_shard
    shards = []
    for i in range(num_shards):
        start = i * args.sequences_per_shard
        end = min(start + args.sequences_per_shard, len(tar_files))
        shard_tar_files = [str(p) for p in tar_files[start:end]]
        output_path = str(output_dir / f"shard_{i:05d}.tfrecord")
        shards.append((
            shard_tar_files, output_path, i,
            args.sequence_length, args.crop_size, args.num_points, num_shards
        ))

    print(f"Processing {num_shards} shards...")

    # Process shards in parallel
    total_successful = 0
    total_failed = 0

    # Use spawn to avoid TF issues with fork
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
        futures = {executor.submit(process_shard, shard): shard for shard in shards}

        with tqdm(total=num_shards, desc="Shards") as pbar:
            for future in as_completed(futures):
                successful, failed, status = future.result()
                total_successful += successful
                total_failed += failed

                if status != "ok":
                    tqdm.write(f"  Shard error: {status}")

                pbar.update(1)
                pbar.set_postfix({'ok': total_successful, 'fail': total_failed})

    # Write metadata
    metadata = {
        'num_sequences': total_successful,
        'num_shards': num_shards,
        'sequences_per_shard': args.sequences_per_shard,
        'sequence_length': args.sequence_length,
        'crop_size': args.crop_size,
        'num_points': args.num_points,
        'failed': total_failed,
    }

    metadata_path = output_dir / 'metadata.txt'
    with open(metadata_path, 'w') as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Shards: {num_shards}")
    print(f"Output: {output_dir}")

    # Estimate size
    total_size = sum(f.stat().st_size for f in output_dir.glob('*.tfrecord'))
    print(f"Total size: {total_size / 1e9:.1f} GB")
    if total_successful > 0:
        print(f"Per sequence: {total_size / total_successful / 1e6:.1f} MB")
    else:
        print("WARNING: No sequences were successfully converted!")
        print("Run with --test to debug a single sequence")


if __name__ == '__main__':
    main()
