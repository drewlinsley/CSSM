"""
cABC (contour ABC) dataset loader for CSSM models.

cABC is a binary classification task similar to Pathfinder, testing
contour integration with different difficulty levels (easy, medium, hard).

Supports:
- PNG loading from original directory structure
- TFRecord loading for maximum I/O performance
"""

import os
from pathlib import Path
from typing import Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

import numpy as np
from PIL import Image
import jax.numpy as jnp

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CABCDataset:
    """
    cABC dataset with pos/neg binary classification.

    Structure: {difficulty}/images/{split}/{pos,neg}/{1,2,...N}/sample_*.png
    """

    def __init__(
        self,
        root: str,
        difficulty: str = 'easy',
        split: str = 'train',
        image_size: int = 224,
        num_frames: int = 1,
    ):
        self.root = Path(root)
        self.difficulty = difficulty
        self.split = split
        self.image_size = image_size
        self.num_frames = num_frames

        # Collect images
        self.image_paths = []
        self.labels = []

        images_dir = self.root / difficulty / 'images' / split

        # Collect from pos and neg directories (with subdirectories)
        for label_name, label in [('neg', 0), ('pos', 1)]:
            label_dir = images_dir / label_name
            if not label_dir.exists():
                continue

            # Find all PNG files in subdirectories
            for subdir in label_dir.iterdir():
                if subdir.is_dir():
                    for img_path in sorted(subdir.glob('*.png')):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

        self.labels = np.array(self.labels)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Load and preprocess a single image."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path)

        # Convert to RGB if needed
        if img.mode == 'L':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to target size
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to numpy and normalize to [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Apply ImageNet normalization
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

        # Add temporal dimension: (H, W, 3) -> (T, H, W, 3)
        img_video = np.stack([img_np] * self.num_frames, axis=0)

        return img_video, label


class CABCVideoLoader:
    """Data loader for cABC that matches the VideoDataLoader interface."""

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

            images = []
            labels = []

            for idx in batch_indices:
                img, label = self.dataset[idx]
                images.append(img)
                labels.append(label)

            images = np.stack(images, axis=0)
            labels = np.array(labels, dtype=np.int32)

            yield jnp.array(images), jnp.array(labels)


class CABCVideoLoaderFast:
    """Fast parallel data loader for cABC with prefetching."""

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
        return self.dataset[idx]

    def _load_batch(self, batch_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._load_single, batch_indices))

        images = np.stack([r[0] for r in results], axis=0)
        labels = np.array([r[1] for r in results], dtype=np.int32)
        return images, labels

    def _prefetch_worker(self, batch_indices_list: list, queue: Queue, stop_event: threading.Event):
        for batch_indices in batch_indices_list:
            if stop_event.is_set():
                break
            try:
                images, labels = self._load_batch(batch_indices)
                queue.put((jnp.array(images), jnp.array(labels)))
            except Exception as e:
                print(f"Prefetch error: {e}")
                queue.put(None)
        queue.put(None)

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self._seed + self._epoch)
            rng.shuffle(indices)
            self._epoch += 1

        batch_indices_list = []
        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            batch_indices_list.append(batch_indices)

        queue = Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        worker = threading.Thread(
            target=self._prefetch_worker,
            args=(batch_indices_list, queue, stop_event)
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
            worker.join(timeout=1.0)


def get_cabc_loader(
    root: str = '/media/data_cifs_lrs/projects/prj_LRA/cabc',
    difficulty: str = 'easy',
    batch_size: int = 32,
    image_size: int = 224,
    num_frames: int = 1,
    split: str = 'train',
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_batches: int = 2,
):
    """
    Get a batch iterator for cABC dataset.

    Args:
        root: Root directory containing difficulty folders
        difficulty: 'easy', 'medium', or 'hard'
        batch_size: Batch size
        image_size: Target image size
        num_frames: Number of frames for temporal dimension
        split: 'train' or 'test'
        shuffle: Whether to shuffle data
        num_workers: Number of parallel workers (0=sequential)
        prefetch_batches: Number of batches to prefetch

    Returns:
        CABCVideoLoader yielding (images, labels)
    """
    dataset = CABCDataset(
        root=root,
        difficulty=difficulty,
        split=split,
        image_size=image_size,
        num_frames=num_frames,
    )

    print(f"  Loaded {len(dataset)} {split} samples")

    if num_workers > 0:
        print(f"  Using parallel loader with {num_workers} workers")
        return CABCVideoLoaderFast(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
        )
    else:
        return CABCVideoLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )


def get_cabc_info(
    difficulty: str = 'easy',
    root: str = '/media/data_cifs_lrs/projects/prj_LRA/cabc',
    tfrecord_dir: str = None,
) -> Dict:
    """Get dataset metadata."""
    # Try TFRecord metadata first
    if tfrecord_dir is not None:
        tfrecord_path = Path(tfrecord_dir)

        # Try two locations
        metadata_path = tfrecord_path / 'metadata.json'
        if not metadata_path.exists():
            metadata_path = tfrecord_path / difficulty / 'metadata.json'

        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            return {
                'num_classes': 2,
                'train_size': metadata.get('train_samples', 0),
                'test_size': metadata.get('test_samples', 0),
                'total_size': metadata.get('total_samples', 0),
                'image_size': metadata.get('image_size', 224),
                'difficulty': metadata.get('difficulty', difficulty),
                'task': 'binary_classification',
                'description': f'cABC {difficulty} (TFRecord)',
            }

    # Fall back to counting images
    images_dir = Path(root) / difficulty / 'images' / 'train'
    n_pos = 0
    n_neg = 0

    if images_dir.exists():
        pos_dir = images_dir / 'pos'
        neg_dir = images_dir / 'neg'

        if pos_dir.exists():
            for subdir in pos_dir.iterdir():
                if subdir.is_dir():
                    n_pos += len(list(subdir.glob('*.png')))

        if neg_dir.exists():
            for subdir in neg_dir.iterdir():
                if subdir.is_dir():
                    n_neg += len(list(subdir.glob('*.png')))

    total = n_pos + n_neg

    return {
        'num_classes': 2,
        'train_size': total,
        'total_size': total,
        'image_size': 300,
        'difficulty': difficulty,
        'task': 'binary_classification',
        'description': f'cABC {difficulty}',
    }


# =============================================================================
# TFRecord-based loader
# =============================================================================

try:
    import tensorflow as tf
    # IMPORTANT: Force TensorFlow to use CPU only to avoid NCCL conflicts with JAX
    try:
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError:
        pass  # Devices already initialized
    HAS_TF = True
except ImportError:
    HAS_TF = False


class CABCTFRecordLoader:
    """Fast TFRecord-based data loader for cABC."""

    def __init__(
        self,
        tfrecord_dir: str,
        difficulty: str,
        split: str,
        batch_size: int,
        num_frames: int = 1,
        shuffle: bool = True,
        shuffle_buffer: int = 10000,
        prefetch_batches: int = 2,
    ):
        if not HAS_TF:
            raise ImportError("TensorFlow required for TFRecord loading")

        self.batch_size = batch_size
        self.num_frames = num_frames
        self.shuffle = shuffle

        tfrecord_path = Path(tfrecord_dir)

        # Try two directory structures
        shard_dir = tfrecord_path / difficulty / split
        metadata_path = tfrecord_path / difficulty / 'metadata.json'

        if not shard_dir.exists():
            shard_dir = tfrecord_path / split
            metadata_path = tfrecord_path / 'metadata.json'

        self.shard_paths = sorted(shard_dir.glob('*.tfrecord'))

        if len(self.shard_paths) == 0:
            raise ValueError(f"No TFRecord shards found in {shard_dir}")

        # Load metadata
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.n_samples = metadata.get(f'{split}_samples', 0)
            self.image_size = metadata.get('image_size', 224)
        else:
            self.n_samples = len(self.shard_paths) * 5000
            self.image_size = 224

        self.dataset = self._create_dataset(shuffle_buffer, prefetch_batches)

    def _parse_example(self, serialized):
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(serialized, features)

        image = tf.io.decode_raw(example['image'], tf.float32)
        image = tf.reshape(image, [self.image_size, self.image_size, 3])
        label = tf.cast(example['label'], tf.int32)

        return image, label

    def _create_dataset(self, shuffle_buffer: int, prefetch_batches: int):
        """Create optimized TF dataset pipeline."""
        files = tf.data.Dataset.from_tensor_slices([str(p) for p in self.shard_paths])

        if self.shuffle:
            files = files.shuffle(len(self.shard_paths))

        # Use higher cycle_length and read-ahead buffer for better I/O
        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x, buffer_size=8 * 1024 * 1024),  # 8MB buffer
            cycle_length=min(8, len(self.shard_paths)),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,  # Allow out-of-order for speed
        )

        if self.shuffle:
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.map(
            self._parse_example,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def __len__(self):
        return self.n_samples // self.batch_size

    def __iter__(self):
        for images, labels in self.dataset:
            images_np = images.numpy()
            images_video = np.stack([images_np] * self.num_frames, axis=1)
            yield jnp.array(images_video), jnp.array(labels.numpy())


def get_cabc_tfrecord_loader(
    tfrecord_dir: str,
    difficulty: str = 'easy',
    batch_size: int = 32,
    num_frames: int = 1,
    split: str = 'train',
    shuffle: bool = True,
    shuffle_buffer: int = 10000,
    prefetch_batches: int = 2,
):
    """Get TFRecord-based loader for cABC."""
    loader = CABCTFRecordLoader(
        tfrecord_dir=tfrecord_dir,
        difficulty=difficulty,
        split=split,
        batch_size=batch_size,
        num_frames=num_frames,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        prefetch_batches=prefetch_batches,
    )

    print(f"  TFRecord loader: {len(loader)} batches from {len(loader.shard_paths)} shards")

    return loader
