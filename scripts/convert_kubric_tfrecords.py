#!/usr/bin/env python3
"""
Convert Kubric dataset from tar.gz files to TFRecords for fast training.

Usage:
    python scripts/convert_kubric_tfrecords.py \
        --input_dir /media/data_cifs/projects/prj_video_imagenet/data/kubric \
        --output_dir /media/data_cifs/projects/prj_video_imagenet/data/kubric_tfrecords \
        --num_workers 16
"""
import argparse
import io
import os
import tarfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Disable GPU for conversion
tf.config.set_visible_devices([], 'GPU')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if isinstance(value, (int, np.integer)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def process_tar_file(args: Tuple[Path, Path, int]) -> Tuple[str, int, str]:
    """Process a single tar.gz file and write to TFRecord.

    Returns: (filename, num_frames, status)
    """
    tar_path, output_dir, sequence_idx = args

    try:
        # Import imageio inside the worker
        try:
            import imageio.v3 as imageio
        except ImportError:
            import imageio

        with tarfile.open(tar_path, 'r:gz') as tar:
            members = tar.getnames()

            # Find the sequence directory name
            seq_name = members[0].split('/')[0]

            # Find frame files
            frame_members = sorted([m for m in members if '/frames/' in m and m.endswith('.png')])

            if not frame_members:
                return (tar_path.name, 0, "no frames found")

            # Load all frames
            frames = []
            for member_name in frame_members:
                f = tar.extractfile(member_name)
                img = imageio.imread(io.BytesIO(f.read()))
                if img.shape[-1] == 4:  # RGBA -> RGB
                    img = img[..., :3]
                frames.append(img)

            video = np.stack(frames, axis=0)  # (T, H, W, 3)
            T, H, W, C = video.shape

            # Load trajectory annotations
            trajs_file = f"{seq_name}/{seq_name}_trajs_2d.npy"
            vis_file = f"{seq_name}/{seq_name}_visibility.npy"

            if trajs_file in members:
                trajs_data = tar.extractfile(trajs_file)
                trajs = np.load(io.BytesIO(trajs_data.read())).astype(np.float32)
            else:
                trajs = np.zeros((T, 0, 2), dtype=np.float32)

            if vis_file in members:
                vis_data = tar.extractfile(vis_file)
                visibility = np.load(io.BytesIO(vis_data.read())).astype(np.float32)
            else:
                visibility = np.ones((T, trajs.shape[1]), dtype=np.float32)

        # Write TFRecord
        output_path = output_dir / f"{sequence_idx:06d}.tfrecord"

        with tf.io.TFRecordWriter(str(output_path)) as writer:
            # Encode video frames as JPEG for compression
            encoded_frames = []
            for frame in video:
                encoded = tf.io.encode_jpeg(frame, quality=95)
                encoded_frames.append(encoded.numpy())

            feature = {
                'sequence_id': _int64_feature(sequence_idx),
                'num_frames': _int64_feature(T),
                'height': _int64_feature(H),
                'width': _int64_feature(W),
                'num_points': _int64_feature(trajs.shape[1]),
                'trajs': _float_feature(trajs),  # (T, N, 2) flattened
                'visibility': _float_feature(visibility),  # (T, N) flattened
            }

            # Add each frame separately for efficient random access
            for i, encoded_frame in enumerate(encoded_frames):
                feature[f'frame_{i:03d}'] = _bytes_feature(encoded_frame)

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        return (tar_path.name, T, "ok")

    except Exception as e:
        return (tar_path.name, 0, f"error: {str(e)[:100]}")


def main():
    parser = argparse.ArgumentParser(description='Convert Kubric tar.gz to TFRecords')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing Kubric tar.gz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for TFRecords')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of sequences to convert (for testing)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    # Find all tar.gz files
    tar_files = sorted(input_dir.glob('*.tar.gz'))

    if not tar_files:
        print(f"No tar.gz files found in {input_dir}")
        return

    if args.limit:
        tar_files = tar_files[:args.limit]

    print(f"Found {len(tar_files)} tar.gz files")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.num_workers}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare work items
    work_items = [(tar_path, output_dir, idx) for idx, tar_path in enumerate(tar_files)]

    # Process in parallel
    successful = 0
    failed = 0
    total_frames = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_tar_file, item): item for item in work_items}

        with tqdm(total=len(work_items), desc="Converting") as pbar:
            for future in as_completed(futures):
                filename, num_frames, status = future.result()

                if status == "ok":
                    successful += 1
                    total_frames += num_frames
                else:
                    failed += 1
                    tqdm.write(f"  Failed: {filename} - {status}")

                pbar.update(1)

    # Write metadata
    metadata = {
        'num_sequences': successful,
        'total_frames': total_frames,
        'failed': failed,
    }

    metadata_path = output_dir / 'metadata.txt'
    with open(metadata_path, 'w') as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total frames: {total_frames}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
