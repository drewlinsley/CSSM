#!/usr/bin/env python3
"""
Download the Kubric MOVI-f dataset for CoTracker training.

Usage:
    python scripts/download_kubric.py --output_dir /path/to/save

The dataset will be downloaded from HuggingFace:
https://huggingface.co/datasets/facebook/CoTracker3_Kubric

Dataset size: ~100-200 GB
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download Kubric dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/data_cifs/projects/prj_video_imagenet/data/kubric",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="facebook/CoTracker3_Kubric",
        help="HuggingFace dataset repository ID",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (if dataset requires authentication)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Kubric dataset from {args.repo_id}")
    print(f"Output directory: {output_dir}")
    print("This may take several hours depending on your connection...")
    print()

    try:
        local_dir = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            token=args.token,
        )
        print(f"\nDownload complete!")
        print(f"Dataset saved to: {local_dir}")

        # Print dataset statistics
        total_size = sum(
            f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file()
        )
        print(f"Total size: {total_size / (1024**3):.2f} GB")

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (need ~200GB)")
        print("3. If the dataset requires authentication, provide --token")
        print("4. Try again later if HuggingFace servers are busy")
        sys.exit(1)


if __name__ == "__main__":
    main()
