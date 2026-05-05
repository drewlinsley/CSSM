#!/usr/bin/env python3
"""
Convert Ego4D videos to WebDataset format for fast streaming.

Pre-extracts video clips and stores them as sharded tar files.
This eliminates video decoding overhead during training.

Usage:
    python scripts/convert_ego4d_to_webdataset.py \
        --input_dir /path/to/ego4d/videos \
        --output_dir /scratch/ego4d_webdataset \
        --num_frames 16 \
        --clips_per_video 10 \
        --resolution 224 \
        --num_workers 16

Output format:
    Each sample in the WebDataset contains:
    - __key__: unique sample ID
    - frames.npy: (T, H, W, C) float32 array
    - metadata.json: clip info (video_id, start_frame, etc.)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import io

import numpy as np
from tqdm import tqdm

try:
    import webdataset as wds
    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False
    print("WebDataset not installed. Install with: pip install webdataset")

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
    timeout_seconds: int = 30,
) -> List[Tuple[np.ndarray, dict]]:
    """
    Extract multiple clips from a single video.

    Returns:
        List of (frames, metadata) tuples
    """
    clips = []

    try:
        # Try to open video with timeout protection
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)

        # Skip very short videos
        if total_frames < num_frames:
            print(f"Skipping {video_path}: too short ({total_frames} frames)")
            return clips

        fps = vr.get_avg_fps()

        clip_length = num_frames * frame_stride

        if total_frames < clip_length:
            # Video too short - extract single clip with frame repetition
            indices = list(range(0, total_frames, max(1, frame_stride)))
            while len(indices) < num_frames:
                indices.append(indices[-1] if indices else 0)
            indices = indices[:num_frames]

            frames = vr.get_batch(indices).asnumpy()
            frames = resize_frames(frames, resolution)

            metadata = {
                'video_path': video_path,
                'video_id': Path(video_path).stem,
                'start_frame': 0,
                'end_frame': total_frames - 1,
                'fps': fps,
                'total_frames': total_frames,
            }
            clips.append((frames, metadata))
        else:
            # Sample multiple clips uniformly across video
            max_start = total_frames - clip_length
            if clips_per_video == 1:
                start_positions = [max_start // 2]  # Center clip
            else:
                # Uniform spacing
                start_positions = np.linspace(0, max_start, clips_per_video, dtype=int)

            for clip_idx, start in enumerate(start_positions):
                try:
                    indices = list(range(start, start + clip_length, frame_stride))
                    frames = vr.get_batch(indices).asnumpy()
                    frames = resize_frames(frames, resolution)

                    metadata = {
                        'video_path': video_path,
                        'video_id': Path(video_path).stem,
                        'clip_idx': clip_idx,
                        'start_frame': int(start),
                        'end_frame': int(indices[-1]),
                        'fps': fps,
                        'total_frames': total_frames,
                    }
                    clips.append((frames, metadata))
                except Exception as clip_e:
                    # Skip this clip but continue with others
                    pass

    except Exception as e:
        # Silently skip problematic videos - they will be counted in main()
        pass

    return clips


def resize_frames(frames: np.ndarray, resolution: int) -> np.ndarray:
    """Resize frames to target resolution with center crop."""
    T, H, W, C = frames.shape

    # Scale to have shorter side = resolution
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

    # Normalize to [0, 1] float32
    return cropped.astype(np.float32) / 255.0


def process_video_batch(args) -> Tuple[List[Tuple[np.ndarray, dict]], int, int]:
    """Process a batch of videos (for multiprocessing).

    Returns:
        clips: List of (frames, metadata) tuples
        success_count: Number of videos processed successfully
        fail_count: Number of videos that failed
    """
    video_paths, num_frames, frame_stride, clips_per_video, resolution = args

    all_clips = []
    success_count = 0
    fail_count = 0

    for video_path in video_paths:
        clips = extract_clips_from_video(
            video_path, num_frames, frame_stride, clips_per_video, resolution
        )
        if clips:
            all_clips.extend(clips)
            success_count += 1
        else:
            fail_count += 1

    return all_clips, success_count, fail_count


def process_single_video(args) -> Tuple[List[Tuple[np.ndarray, dict]], bool]:
    """Process a single video (for sequential processing)."""
    video_path, num_frames, frame_stride, clips_per_video, resolution = args

    clips = extract_clips_from_video(
        video_path, num_frames, frame_stride, clips_per_video, resolution
    )
    return clips, len(clips) > 0


def main():
    parser = argparse.ArgumentParser(description='Convert Ego4D to WebDataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for WebDataset shards')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Frames per clip')
    parser.add_argument('--frame_stride', type=int, default=4,
                        help='Temporal stride between frames')
    parser.add_argument('--clips_per_video', type=int, default=10,
                        help='Number of clips to extract per video')
    parser.add_argument('--resolution', type=int, default=224,
                        help='Target spatial resolution')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers (0 for sequential)')
    parser.add_argument('--shard_size', type=int, default=1000,
                        help='Samples per shard')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Max videos to process (for testing)')
    args = parser.parse_args()

    if not HAS_WEBDATASET:
        print("Error: webdataset not installed")
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

    # Process videos and write WebDataset
    shard_pattern = str(output_dir / "ego4d-%06d.tar")
    sink = wds.ShardWriter(shard_pattern, maxcount=args.shard_size)

    sample_idx = 0
    total_success = 0
    total_fail = 0

    if args.num_workers == 0:
        # Sequential processing (more robust for problematic videos)
        print("Processing sequentially (--num_workers 0)...")
        for video_path in tqdm(video_paths, desc="Processing"):
            clips, success = process_single_video(
                (video_path, args.num_frames, args.frame_stride,
                 args.clips_per_video, args.resolution)
            )
            if success:
                total_success += 1
                for frames, metadata in clips:
                    frames_bytes = io.BytesIO()
                    np.save(frames_bytes, frames)
                    frames_bytes.seek(0)
                    sample = {
                        "__key__": f"sample_{sample_idx:08d}",
                        "frames.npy": frames_bytes.read(),
                        "metadata.json": json.dumps(metadata).encode(),
                    }
                    sink.write(sample)
                    sample_idx += 1
            else:
                total_fail += 1
    else:
        # Parallel processing
        batch_size = max(1, len(video_paths) // args.num_workers)
        video_batches = [
            video_paths[i:i + batch_size]
            for i in range(0, len(video_paths), batch_size)
        ]

        print(f"Processing in {len(video_batches)} batches with {args.num_workers} workers")

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for batch in video_batches:
                future = executor.submit(
                    process_video_batch,
                    (batch, args.num_frames, args.frame_stride,
                     args.clips_per_video, args.resolution)
                )
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                clips, success_count, fail_count = future.result()
                total_success += success_count
                total_fail += fail_count

                for frames, metadata in clips:
                    frames_bytes = io.BytesIO()
                    np.save(frames_bytes, frames)
                    frames_bytes.seek(0)
                    sample = {
                        "__key__": f"sample_{sample_idx:08d}",
                        "frames.npy": frames_bytes.read(),
                        "metadata.json": json.dumps(metadata).encode(),
                    }
                    sink.write(sample)
                    sample_idx += 1

    sink.close()

    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"  Videos processed: {total_success}")
    print(f"  Videos failed/skipped: {total_fail}")
    print(f"  Total clips: {sample_idx}")
    print(f"  Output: {output_dir}")
    print(f"  Shard pattern: {shard_pattern}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
