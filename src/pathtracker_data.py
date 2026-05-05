"""
PathTracker dataset loader for CSSM models.

PathTracker is a video-based binary classification task: dots move across
64 frames on a 32x32 RGB grid. The task is to determine if the marked
target dot reaches the marked endpoint. Unlike Pathfinder (static image
repeated T times), PathTracker has real temporal dynamics.

Each .npy file contains a (64, 32, 32, 3) uint8 video.
Filenames: 1_sample_N.npy (positive) / 0_sample_N.npy (negative).

Split convention: first 50k per class = train (100k total),
last 5k per class = test (10k total).

Supports:
- Sequential loading (original)
- Parallel loading with ThreadPoolExecutor (--num_workers > 0)
- Frame subsampling (np.linspace over 64 native frames)
"""

import os
import re
from pathlib import Path
from typing import Tuple, Iterator, Optional
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

import numpy as np
from PIL import Image
import jax.numpy as jnp

# ImageNet normalization constants (same as other loaders)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PathTrackerDataset:
    """
    PathTracker video dataset with pos/neg binary classification.

    Attributes:
        root: Path to dataset directory containing 0/ and 1/ subdirectories
        files: List of (filepath, label) tuples
        image_size: Target spatial size (default 32, native resolution)
        num_frames: Number of frames to subsample from total_frames
        total_frames: Total frames in each .npy video (default 64)
    """

    def __init__(
        self,
        root: str,
        image_size: int = 32,
        num_frames: int = 8,
        total_frames: int = 64,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.num_frames = num_frames
        self.total_frames = total_frames

        # Precompute frame indices for subsampling
        self.frame_indices = np.round(np.linspace(0, total_frames - 1, num_frames)).astype(int)

        # Collect .npy files from 0/ (negative) and 1/ (positive) directories
        self.files = []

        for label in [0, 1]:
            label_dir = self.root / str(label)
            if not label_dir.exists():
                continue
            # Sort by sample number for reproducible ordering
            npy_files = sorted(
                label_dir.glob('*.npy'),
                key=lambda p: int(re.search(r'_sample_(\d+)', p.stem).group(1))
            )
            for fpath in npy_files:
                self.files.append((fpath, label))

        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {self.root}/0/ or {self.root}/1/")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Load and preprocess a single video."""
        filepath, label = self.files[idx]

        # Load video: (64, 32, 32, 3) uint8
        video = np.load(filepath)

        # Subsample frames: pick num_frames evenly spaced from total_frames
        video = video[self.frame_indices]  # (num_frames, 32, 32, 3)

        # Resize if needed (native is 32x32)
        if self.image_size != video.shape[1]:
            resized_frames = []
            for t in range(video.shape[0]):
                img = Image.fromarray(video[t])
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                resized_frames.append(np.array(img))
            video = np.stack(resized_frames, axis=0)

        # Normalize: uint8 -> float32 [0,1] -> ImageNet normalization
        video = video.astype(np.float32) / 255.0
        video = (video - IMAGENET_MEAN) / IMAGENET_STD

        return video, label  # (T, H, W, 3) float32, int


class PathTrackerSubset:
    """Subset of PathTrackerDataset using specific indices."""

    def __init__(self, dataset: PathTrackerDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.dataset[self.indices[idx]]


class PathTrackerVideoLoader:
    """
    Data loader for PathTracker that matches Pathfinder's VideoLoader interface.

    Provides __len__ for proper tqdm progress display.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(dataset)
        self._epoch = 0
        self._seed = seed

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self._seed + self._epoch)
            rng.shuffle(indices)
            self._epoch += 1

        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]

            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            videos = []
            labels = []

            for idx in batch_indices:
                video, label = self.dataset[idx]
                videos.append(video)
                labels.append(label)

            videos = np.stack(videos, axis=0)  # (B, T, H, W, 3)
            labels = np.array(labels, dtype=np.int32)

            yield jnp.array(videos), jnp.array(labels)


class PathTrackerVideoLoaderFast:
    """
    Fast parallel data loader for PathTracker with prefetching.

    Uses ThreadPoolExecutor to load batches in parallel and a prefetch
    queue to overlap data loading with GPU compute.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 4,
        prefetch_batches: int = 2,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.n_samples = len(dataset)
        self._epoch = 0
        self._seed = seed

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def _load_single(self, idx: int) -> Tuple[np.ndarray, int]:
        """Load a single sample (called by worker threads)."""
        return self.dataset[idx]

    def _load_batch(self, batch_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Load a batch of samples using thread pool."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._load_single, batch_indices))

        videos = np.stack([r[0] for r in results], axis=0)
        labels = np.array([r[1] for r in results], dtype=np.int32)
        return videos, labels

    def _prefetch_worker(self, batch_indices_list: list, queue: Queue, stop_event: threading.Event):
        """Background worker that prefetches batches into queue."""
        for batch_indices in batch_indices_list:
            if stop_event.is_set():
                break
            try:
                videos, labels = self._load_batch(batch_indices)
                queue.put((jnp.array(videos), jnp.array(labels)))
            except Exception as e:
                print(f"Prefetch error: {e}")
                queue.put(None)
        queue.put(None)  # Signal end of data

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self._seed + self._epoch)
            rng.shuffle(indices)
            self._epoch += 1

        # Create list of batch indices
        batch_indices_list = []
        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            batch_indices_list.append(batch_indices)

        # Start prefetch worker
        queue = Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        worker = threading.Thread(
            target=self._prefetch_worker,
            args=(batch_indices_list, queue, stop_event)
        )
        worker.start()

        # Yield batches from queue
        try:
            while True:
                batch = queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            stop_event.set()
            worker.join(timeout=1.0)


def get_pathtracker_datasets(
    root: str = '/media/data_cifs/projects/prj_video_datasets/pathtracker',
    image_size: int = 32,
    num_frames: int = 8,
    total_frames: int = 64,
    train_size: int = 20000,
    test_size: int = 2000,
    seed: int = 42,
) -> Tuple[PathTrackerSubset, PathTrackerSubset, PathTrackerSubset]:
    """
    Get train/val/test splits for PathTracker dataset.

    Uses index-based splitting: first train_size samples = train,
    next test_size samples = test. Val is carved from train.

    Args:
        root: Root directory containing 0/ and 1/ subdirectories
        image_size: Target spatial size
        num_frames: Number of frames to subsample
        total_frames: Total frames per video
        train_size: Number of training samples
        test_size: Number of test samples
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    full_dataset = PathTrackerDataset(
        root=root,
        image_size=image_size,
        num_frames=num_frames,
        total_frames=total_frames,
    )

    n_total = len(full_dataset)
    rng = np.random.RandomState(seed)

    # Random split (matches TFRecord conversion script):
    # Shuffle all indices, then carve train/test/val.
    indices = np.arange(n_total)
    rng.shuffle(indices)

    train_size = min(train_size, n_total)
    test_size = min(test_size, n_total - train_size)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]

    # Carve val from train (last 10%)
    n_val = max(1, len(train_indices) // 10)
    val_indices = train_indices[-n_val:]
    train_indices = train_indices[:-n_val]

    train_dataset = PathTrackerSubset(full_dataset, train_indices)
    val_dataset = PathTrackerSubset(full_dataset, val_indices)
    test_dataset = PathTrackerSubset(full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def get_pathtracker_loader(
    root: str = '/media/data_cifs/projects/prj_video_datasets/pathtracker',
    batch_size: int = 32,
    num_frames: int = 8,
    image_size: int = 32,
    total_frames: int = 64,
    split: str = 'train',
    train_size: int = 20000,
    test_size: int = 2000,
    seed: int = 42,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_batches: int = 2,
):
    """
    Get a batch iterator for PathTracker dataset.

    Args:
        root: Root directory containing 0/ and 1/ subdirectories
        batch_size: Batch size
        num_frames: Number of frames to subsample
        image_size: Target spatial size
        total_frames: Total frames per video
        split: 'train', 'val', or 'test'
        train_size: Number of training samples
        test_size: Number of test samples
        seed: Random seed
        shuffle: Whether to shuffle data
        num_workers: Number of parallel workers (0=sequential, >0=parallel)
        prefetch_batches: Number of batches to prefetch (only when num_workers>0)

    Returns:
        PathTrackerVideoLoader or PathTrackerVideoLoaderFast yielding (videos, labels)
        videos: (B, T, H, W, 3) float32
        labels: (B,) int32
    """
    train_ds, val_ds, test_ds = get_pathtracker_datasets(
        root=root,
        image_size=image_size,
        num_frames=num_frames,
        total_frames=total_frames,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )

    if split == 'train':
        dataset = train_ds
    elif split == 'val':
        dataset = val_ds
    else:
        dataset = test_ds

    print(f"  Loaded {len(dataset)} {split} samples")

    if num_workers > 0:
        print(f"  Using parallel loader with {num_workers} workers, {prefetch_batches} prefetch batches")
        return PathTrackerVideoLoaderFast(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
        )
    else:
        return PathTrackerVideoLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )


def get_pathtracker_info(
    root: str = '/media/data_cifs/projects/prj_video_datasets/pathtracker',
    train_size: int = 20000,
    test_size: int = 2000,
) -> dict:
    """Get dataset metadata.

    Args:
        root: Root directory containing 0/ and 1/ subdirectories
        train_size: Expected training set size
        test_size: Expected test set size

    Returns:
        Dictionary with dataset info including train_size
    """
    # Count actual files if directory exists
    root_path = Path(root)
    n_pos = len(list((root_path / '1').glob('*.npy'))) if (root_path / '1').exists() else 0
    n_neg = len(list((root_path / '0').glob('*.npy'))) if (root_path / '0').exists() else 0
    total = n_pos + n_neg

    # Use actual count or expected sizes
    actual_train = min(train_size, total) if total > 0 else train_size
    # Val is carved from train (10%)
    n_val = max(1, actual_train // 10)
    actual_train_after_val = actual_train - n_val

    # Read actual frame count from first file
    num_frames = 64  # default
    for label_dir in [root_path / '1', root_path / '0']:
        npys = sorted(label_dir.glob('*.npy')) if label_dir.exists() else []
        if npys:
            num_frames = np.load(npys[0]).shape[0]
            break

    return {
        'num_classes': 2,
        'train_size': actual_train_after_val,
        'total_size': total,
        'image_size': 32,
        'num_frames': num_frames,
        'task': 'binary_classification',
        'description': f'PathTracker ({n_pos} pos + {n_neg} neg = {total} total, {num_frames} frames)',
    }


# =============================================================================
# TFRecord-based loader for maximum I/O performance
# =============================================================================

try:
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError:
        pass
    HAS_TF = True
except ImportError:
    HAS_TF = False


class PathTrackerTFRecordLoader:
    """
    Fast TFRecord-based data loader for PathTracker videos.

    Supports two TFRecord formats:
      1. "Legacy" format (from convert_pathtracker_tfrecords.py):
         - Shards in split subdirs: {tfrecord_dir}/{split}/shard_NNNN.tfrecord
         - Features: video (float32 bytes), label (int64), num_frames, height, width, channels
         - Pre-normalized (ImageNet mean/std already applied)
         - No compression
      2. "New" format (from PathTracker paper data):
         - Flat dir with named shards: {split}-batch_N--NNNNN-of-NNNNN
         - Features: image (uint8 bytes), label (bytes, 1 byte), height (int64), width (int64)
         - Raw uint8, normalized on the fly
         - GZIP compressed
    """

    def __init__(
        self,
        tfrecord_dir: str,
        split: str,
        batch_size: int,
        num_frames: int = 8,
        total_frames: int = 64,
        shuffle: bool = True,
        shuffle_buffer: int = 10000,
        prefetch_batches: int = 2,
    ):
        if not HAS_TF:
            raise ImportError("TensorFlow required for TFRecord loading.")

        self.batch_size = batch_size
        self.num_frames = num_frames
        self.total_frames = total_frames
        self.shuffle = shuffle
        self.image_size = 32

        tfrecord_path = Path(tfrecord_dir)
        self.format = self._detect_format(tfrecord_path, split)
        print(f"  PathTracker TFRecord format: {self.format}")

        if self.format == 'new':
            self._init_new_format(tfrecord_path, split)
        else:
            self._init_legacy_format(tfrecord_path, split)

        if num_frames > self.n_frames_stored:
            raise ValueError(
                f"--seq_len {num_frames} > {self.n_frames_stored} frames stored in TFRecords. "
                f"Use --seq_len {self.n_frames_stored} or less."
            )
        self.frame_indices = np.round(np.linspace(0, self.n_frames_stored - 1, num_frames)).astype(int)

        self.dataset = self._create_dataset(shuffle_buffer, prefetch_batches)

    def _detect_format(self, tfrecord_path: Path, split: str) -> str:
        """Auto-detect TFRecord format based on file structure."""
        # New format: flat dir with files like "train-batch_0--00000-of-00040"
        split_prefix = 'train' if split == 'train' else 'test'
        new_files = sorted(tfrecord_path.glob(f'{split_prefix}-batch_*'))
        if new_files:
            return 'new'

        # Legacy format: split subdirectory with shard_NNNN.tfrecord
        shard_dir = tfrecord_path / split
        if shard_dir.exists() and list(shard_dir.glob('*.tfrecord')):
            return 'legacy'

        # Try new format with 'val' mapped to 'test'
        if split == 'val':
            new_files = sorted(tfrecord_path.glob('test-batch_*'))
            if new_files:
                return 'new'

        raise ValueError(
            f"No TFRecord files found in {tfrecord_path} for split '{split}'. "
            f"Expected either flat files like 'train-batch_*' or subdir '{split}/shard_*.tfrecord'"
        )

    def _init_new_format(self, tfrecord_path: Path, split: str):
        """Initialize for new gzipped format with image/label features."""
        split_prefix = 'train' if split in ('train', 'val') else 'test'
        self.shard_paths = sorted(tfrecord_path.glob(f'{split_prefix}-batch_*'))
        self.compression = 'GZIP'

        # Probe to get frame count and sample count
        ds = tf.data.TFRecordDataset([str(self.shard_paths[0])], compression_type='GZIP')
        raw = next(iter(ds))
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        image_bytes = ex.features.feature['image'].bytes_list.value[0]
        h = ex.features.feature['height'].int64_list.value[0]
        w = ex.features.feature['width'].int64_list.value[0]
        self.image_size = h
        n_pixels = len(image_bytes)  # uint8
        self.n_frames_stored = n_pixels // (h * w * 3)
        print(f"  New format: {self.n_frames_stored} frames, {h}x{w}, {len(self.shard_paths)} shards")

        # Count samples
        total = 0
        for raw in tf.data.TFRecordDataset([str(p) for p in self.shard_paths], compression_type='GZIP'):
            total += 1
        self.n_samples = total

        # For val split, use last 10% of training data
        if split == 'val':
            n_val = max(1, self.n_samples // 10)
            self.n_samples = n_val
            print(f"  Val split: using {n_val} samples from training shards")

    def _init_legacy_format(self, tfrecord_path: Path, split: str):
        """Initialize for legacy format with video/label features."""
        shard_dir = tfrecord_path / split
        metadata_path = tfrecord_path / 'metadata.json'
        self.shard_paths = sorted(shard_dir.glob('*.tfrecord'))
        self.compression = None

        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.n_samples = metadata.get(f'{split}_samples', 0)
            self.image_size = metadata.get('image_size', 32)
            self.n_frames_stored = metadata.get('num_frames', 64)
        else:
            self.n_samples = len(self.shard_paths) * 1000
            self.n_frames_stored = 64

        # Probe first record to verify
        try:
            raw = next(iter(tf.data.TFRecordDataset([str(self.shard_paths[0])])))
            parsed = tf.io.parse_single_example(raw, {
                'video': tf.io.FixedLenFeature([], tf.string),
            })
            n_bytes = len(parsed['video'].numpy())
            pixels_per_frame = self.image_size * self.image_size * 3
            probed_frames = n_bytes // (pixels_per_frame * 4)
            if probed_frames != self.n_frames_stored:
                print(f"  WARNING: metadata says {self.n_frames_stored} frames, "
                      f"actual data has {probed_frames}. Using {probed_frames}.")
                self.n_frames_stored = probed_frames
        except Exception:
            pass

    def _parse_example_new(self, serialized):
        """Parse new format: image (uint8), label (bytes)."""
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(serialized, features)

        image = tf.io.decode_raw(example['image'], tf.uint8)
        image = tf.reshape(image, [self.n_frames_stored, self.image_size, self.image_size, 3])

        # Normalize: uint8 -> float32, ImageNet mean/std
        video = tf.cast(image, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        video = (video - mean) / std

        label = tf.io.decode_raw(example['label'], tf.uint8)
        label = tf.cast(label[0], tf.int32)

        return video, label

    def _parse_example_legacy(self, serialized):
        """Parse legacy format: video (float32), label (int64)."""
        features = {
            'video': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'num_frames': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(serialized, features)

        video = tf.io.decode_raw(example['video'], tf.float32)
        video = tf.reshape(video, [self.n_frames_stored, self.image_size, self.image_size, 3])

        label = tf.cast(example['label'], tf.int32)

        return video, label

    def _create_dataset(self, shuffle_buffer: int, prefetch_batches: int):
        """Create optimized TF dataset pipeline."""
        files = tf.data.Dataset.from_tensor_slices([str(p) for p in self.shard_paths])

        if self.shuffle:
            files = files.shuffle(len(self.shard_paths))

        compression = self.compression
        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(
                x, buffer_size=8 * 1024 * 1024,
                compression_type=compression or '',
            ),
            cycle_length=min(8, len(self.shard_paths)),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        if self.shuffle:
            dataset = dataset.shuffle(shuffle_buffer)

        parse_fn = self._parse_example_new if self.format == 'new' else self._parse_example_legacy
        dataset = dataset.map(
            parse_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def __len__(self):
        return self.n_samples // self.batch_size

    def _tf_to_jax_worker(self, queue: Queue, stop_event: threading.Event):
        """Background thread: converts TF tensors to JAX arrays."""
        try:
            for videos, labels in self.dataset:
                if stop_event.is_set():
                    break
                videos_np = videos.numpy()
                videos_sub = videos_np[:, self.frame_indices]
                queue.put((jnp.array(videos_sub), jnp.array(labels.numpy())))
        except Exception as e:
            print(f"TFRecord prefetch error: {e}")
        queue.put(None)  # Sentinel

    def __iter__(self):
        queue = Queue(maxsize=4)
        stop_event = threading.Event()
        worker = threading.Thread(
            target=self._tf_to_jax_worker,
            args=(queue, stop_event),
            daemon=True,
        )
        worker.start()
        try:
            while True:
                batch = queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            stop_event.set()
            worker.join(timeout=2.0)


def get_pathtracker_tfrecord_loader(
    tfrecord_dir: str,
    batch_size: int = 32,
    num_frames: int = 8,
    total_frames: int = 64,
    split: str = 'train',
    shuffle: bool = True,
    shuffle_buffer: int = 10000,
    prefetch_batches: int = 2,
):
    """
    Get TFRecord-based loader for PathTracker.

    Requires pre-conversion: python scripts/convert_pathtracker_tfrecords.py

    Args:
        tfrecord_dir: Directory containing TFRecord shards
        batch_size: Batch size
        num_frames: Number of frames to subsample
        total_frames: Total frames stored per video
        split: 'train', 'val', or 'test'
        shuffle: Whether to shuffle data
        shuffle_buffer: Size of shuffle buffer
        prefetch_batches: Number of batches to prefetch

    Returns:
        PathTrackerTFRecordLoader yielding (videos, labels)
    """
    loader = PathTrackerTFRecordLoader(
        tfrecord_dir=tfrecord_dir,
        split=split,
        batch_size=batch_size,
        num_frames=num_frames,
        total_frames=total_frames,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        prefetch_batches=prefetch_batches,
    )

    print(f"  TFRecord loader: {len(loader)} batches from {len(loader.shard_paths)} shards")

    return loader
