#!/usr/bin/env python3
"""
Convert Ego4D videos to TFRecord format for fast streaming.

Pre-extracts video clips and stores them as sharded TFRecord files.
This is the fastest data loading option for training.

Usage:
    python scripts/convert_ego4d_to_tfrecord.py \
        --input_dir /path/to/ego4d/videos \
        --output_dir /scratch/ego4d_tfrecord \
        --num_frames 16 \
        --clips_per_video 5 \
        --resolution 224 \
        --num_workers 8
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not installed. Install with: pip install tensorflow")

try:
    import decord
    from decord import VideoReader, cpu
    decord.bridge.set_bridge('native')
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    print("decord not installed. Install with: pip install decord")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def extract_clips_from_video(
    video_path: str,
    num_frames: int,
    frame_stride: int,
    clips_per_video: int,
    resolution: int,
) -> List[np.ndarray]:
    """Extract multiple clips from a single video."""
    clips = []
    vr = None

    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)

        if total_frames < num_frames:
            del vr
            return clips

        clip_length = num_frames * frame_stride

        if total_frames < clip_length:
            # Video too short - single clip with repetition
            indices = list(range(0, total_frames, max(1, frame_stride)))
            while len(indices) < num_frames:
                indices.append(indices[-1] if indices else 0)
            indices = indices[:num_frames]

            frames = vr.get_batch(indices).asnumpy()
            frames = resize_frames(frames, resolution)
            clips.append(frames)
        else:
            # Sample multiple clips uniformly
            max_start = total_frames - clip_length
            if clips_per_video == 1:
                start_positions = [max_start // 2]
            else:
                start_positions = np.linspace(0, max_start, clips_per_video, dtype=int)

            for start in start_positions:
                try:
                    indices = list(range(start, start + clip_length, frame_stride))
                    frames = vr.get_batch(indices).asnumpy()
                    frames = resize_frames(frames, resolution)
                    clips.append(frames)
                except Exception:
                    pass

    except Exception:
        pass
    finally:
        # Explicitly delete VideoReader to free memory
        if vr is not None:
            del vr

    return clips


def resize_frames(frames: np.ndarray, resolution: int) -> np.ndarray:
    """Resize frames to target resolution with center crop."""
    T, H, W, C = frames.shape

    scale = resolution / min(H, W)
    new_H = int(H * scale)
    new_W = int(W * scale)

    if HAS_CV2:
        resized = np.stack([
            cv2.resize(frames[t], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            for t in range(T)
        ])
    else:
        from PIL import Image
        resized = np.stack([
            np.array(Image.fromarray(frames[t]).resize((new_W, new_H), Image.BILINEAR))
            for t in range(T)
        ])

    # Center crop
    top = (new_H - resolution) // 2
    left = (new_W - resolution) // 2
    cropped = resized[:, top:top+resolution, left:left+resolution, :]

    # Normalize to [0, 1]
    return cropped.astype(np.float32) / 255.0


def process_video(args) -> Tuple[List[np.ndarray], bool]:
    """Process a single video."""
    video_path, num_frames, frame_stride, clips_per_video, resolution = args
    clips = extract_clips_from_video(
        video_path, num_frames, frame_stride, clips_per_video, resolution
    )
    return clips, len(clips) > 0


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_clip(frames: np.ndarray) -> bytes:
    """Serialize a clip to TFRecord format."""
    T, H, W, C = frames.shape
    feature = {
        'frames': _bytes_feature(frames.tobytes()),
        'num_frames': _int64_feature(T),
        'height': _int64_feature(H),
        'width': _int64_feature(W),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def main():
    parser = argparse.ArgumentParser(description='Convert Ego4D to TFRecord')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for TFRecord shards')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Frames per clip')
    parser.add_argument('--frame_stride', type=int, default=4,
                        help='Temporal stride between frames')
    parser.add_argument('--clips_per_video', type=int, default=5,
                        help='Number of clips to extract per video')
    parser.add_argument('--resolution', type=int, default=224,
                        help='Target spatial resolution')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--shard_size', type=int, default=1000,
                        help='Samples per shard')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Max videos to process (for testing)')
    parser.add_argument('--batch_buffer', type=int, default=500,
                        help='Number of clips to buffer before writing (reduce for OOM)')
    args = parser.parse_args()

    if not HAS_TF:
        print("Error: TensorFlow not installed")
        sys.exit(1)
    if not HAS_DECORD:
        print("Error: decord not installed")
        sys.exit(1)

    # Find all videos
    input_dir = Path(args.input_dir)
    video_paths = []
    for ext in ['*.mp4', '*.MP4', '*.avi', '*.mkv', '*.webm']:
        video_paths.extend(input_dir.glob(f"**/{ext}"))
    video_paths = sorted([str(p) for p in video_paths])

    if args.max_videos:
        video_paths = video_paths[:args.max_videos]

    print(f"Found {len(video_paths)} videos")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process videos and write TFRecords in streaming fashion
    # Accumulate clips in batches to reduce memory usage
    batch_clips = []
    batch_size = args.batch_buffer  # Configurable buffer size
    total_success = 0
    total_fail = 0
    total_clips = 0
    shard_idx = 0

    clip_mem_mb = (args.num_frames * args.resolution * args.resolution * 3 * 4) / (1024 * 1024)
    print(f"Extracting clips with {args.num_workers} workers...")
    print(f"Streaming to TFRecords (batch buffer: {batch_size} clips, ~{batch_size * clip_mem_mb:.0f}MB)")

    def write_batch(clips, shard_idx, output_dir, shard_size):
        """Write a batch of clips to TFRecord shards."""
        np.random.shuffle(clips)  # Shuffle within batch
        written = 0
        writer = None

        for clip in clips:
            if written % shard_size == 0:
                if writer is not None:
                    writer.close()
                shard_path = output_dir / f'ego4d_{shard_idx:06d}.tfrecord'
                writer = tf.io.TFRecordWriter(str(shard_path))
                shard_idx += 1

            example = serialize_clip(clip)
            writer.write(example)
            written += 1

        if writer is not None:
            writer.close()

        return shard_idx

    if args.num_workers == 0:
        # Sequential
        for video_path in tqdm(video_paths, desc="Processing"):
            clips, success = process_video(
                (video_path, args.num_frames, args.frame_stride,
                 args.clips_per_video, args.resolution)
            )
            if success:
                total_success += 1
                batch_clips.extend(clips)
                total_clips += len(clips)

                # Write batch when full
                if len(batch_clips) >= batch_size:
                    shard_idx = write_batch(batch_clips, shard_idx, output_dir, args.shard_size)
                    batch_clips = []  # Clear memory
            else:
                total_fail += 1
    else:
        # Parallel with chunked processing to avoid memory leak
        # Process videos in chunks to prevent futures from accumulating
        chunk_size = args.num_workers * 10  # Process 10 videos per worker at a time

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for chunk_start in tqdm(range(0, len(video_paths), chunk_size), desc="Chunks"):
                chunk_end = min(chunk_start + chunk_size, len(video_paths))
                chunk_paths = video_paths[chunk_start:chunk_end]

                work_items = [
                    (vp, args.num_frames, args.frame_stride, args.clips_per_video, args.resolution)
                    for vp in chunk_paths
                ]

                # Submit only this chunk's futures
                futures = [executor.submit(process_video, item) for item in work_items]

                for future in as_completed(futures):
                    clips, success = future.result()
                    # Explicitly delete future to free memory
                    del future

                    if success:
                        total_success += 1
                        batch_clips.extend(clips)
                        total_clips += len(clips)

                        # Write batch when full to free memory
                        if len(batch_clips) >= batch_size:
                            shard_idx = write_batch(batch_clips, shard_idx, output_dir, args.shard_size)
                            batch_clips = []  # Clear memory
                    else:
                        total_fail += 1

                # Clear futures list after each chunk
                del futures
                gc.collect()

    # Write remaining clips
    if batch_clips:
        shard_idx = write_batch(batch_clips, shard_idx, output_dir, args.shard_size)
        batch_clips = []

    print(f"\nExtracted {total_clips} clips from {total_success} videos ({total_fail} failed)")

    # Write metadata
    metadata = {
        'num_clips': total_clips,
        'num_shards': shard_idx,
        'num_frames': args.num_frames,
        'resolution': args.resolution,
        'frame_stride': args.frame_stride,
        'shard_size': args.shard_size,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print("Conversion complete!")
    print(f"  Videos processed: {total_success}")
    print(f"  Videos failed: {total_fail}")
    print(f"  Total clips: {total_clips}")
    print(f"  Shards: {shard_idx}")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
