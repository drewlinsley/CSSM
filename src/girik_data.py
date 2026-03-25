"""Girik PathTracker dataset loader.

Loads gzip-compressed TFRecords with 64-frame 128×128 RGB videos.
Format: flat directory with train-batch_*/ test-batch_* shards.
Features: image (uint8 bytes), label (uint8 byte), height/width (int64).

Usage:
    loader = get_girik_tfrecord_loader(
        tfrecord_dir='/oscar/scratch/dlinsley/girik',
        batch_size=32, num_frames=32, split='train')
"""

from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def get_girik_info(tfrecord_dir: str) -> dict:
    """Get dataset metadata by probing first shard."""
    p = Path(tfrecord_dir)
    train_shards = sorted(p.glob('train-*'))
    test_shards = sorted(p.glob('test-*'))

    # Probe first shard for frame count / image size
    num_frames = 64
    image_size = 128
    samples_per_shard = 500
    if HAS_TF and train_shards:
        try:
            ds = tf.data.TFRecordDataset([str(train_shards[0])], compression_type='GZIP')
            raw = next(iter(ds))
            example = tf.train.Example()
            example.ParseFromString(raw.numpy())
            f = example.features.feature
            h = f['height'].int64_list.value[0]
            w = f['width'].int64_list.value[0]
            n_bytes = len(f['image'].bytes_list.value[0])
            num_frames = n_bytes // (h * w * 3)
            image_size = h
            samples_per_shard = sum(1 for _ in ds)
        except Exception:
            pass

    return {
        'num_classes': 2,
        'train_size': len(train_shards) * samples_per_shard,
        'test_size': len(test_shards) * samples_per_shard,
        'image_size': image_size,
        'num_frames': num_frames,
        'task': 'binary_classification',
    }


class GirikTFRecordLoader:
    """TFRecord loader for Girik PathTracker dataset."""

    def __init__(
        self,
        tfrecord_dir: str,
        split: str,
        batch_size: int,
        num_frames: int = 64,
        image_size: int = None,
        shuffle: bool = True,
        shuffle_buffer: int = 10000,
        prefetch_batches: int = 2,
    ):
        if not HAS_TF:
            raise ImportError("TensorFlow required for TFRecord loading.")

        self.batch_size = batch_size
        self.num_frames = num_frames
        self.shuffle = shuffle

        tfrecord_path = Path(tfrecord_dir)

        # Find shards by prefix
        prefix = 'train-' if split in ('train', 'val') else 'test-'
        all_shards = sorted(tfrecord_path.glob(f'{prefix}*'))

        if not all_shards:
            raise ValueError(f"No shards found with prefix '{prefix}' in {tfrecord_dir}")

        # Split train into train/val (90/10)
        if split == 'val':
            n_val = max(1, len(all_shards) // 10)
            self.shard_paths = all_shards[-n_val:]
            all_train = all_shards[:-n_val]
        elif split == 'train':
            n_val = max(1, len(all_shards) // 10)
            self.shard_paths = all_shards[:-n_val]
        else:
            self.shard_paths = all_shards

        # Probe for stored dimensions
        ds_probe = tf.data.TFRecordDataset([str(self.shard_paths[0])], compression_type='GZIP')
        raw = next(iter(ds_probe))
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        f = example.features.feature
        self.stored_h = f['height'].int64_list.value[0]
        self.stored_w = f['width'].int64_list.value[0]
        n_bytes = len(f['image'].bytes_list.value[0])
        self.stored_frames = n_bytes // (self.stored_h * self.stored_w * 3)
        self.target_size = image_size  # None = use native

        if num_frames > self.stored_frames:
            raise ValueError(
                f"--seq_len {num_frames} > {self.stored_frames} frames stored. "
                f"Use --seq_len {self.stored_frames} or less."
            )

        # Frame subsampling indices
        self.frame_indices = np.round(
            np.linspace(0, self.stored_frames - 1, num_frames)
        ).astype(int)

        # Count samples
        self.n_samples = sum(1 for _ in ds_probe) * len(self.shard_paths)

        self.dataset = self._create_dataset(shuffle_buffer, prefetch_batches)

    def _parse_example(self, serialized):
        features = tf.io.parse_single_example(serialized, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
        })

        h = self.stored_h
        w = self.stored_w
        n_f = self.stored_frames

        # Decode uint8 video
        image = tf.io.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, [n_f, h, w, 3])
        image = tf.cast(image, tf.float32) / 127.5 - 1.0  # normalize to [-1, 1]

        # Subsample frames
        image = tf.gather(image, self.frame_indices)

        # Resize if needed (nearest neighbor to preserve dot structure)
        if self.target_size is not None and self.target_size != h:
            sz = self.target_size
            # Reshape to (T, H, W, 3) -> resize all frames at once
            image = tf.image.resize(image, [sz, sz], method='nearest')

        label = tf.io.decode_raw(features['label'], tf.uint8)
        label = tf.cast(label[0], tf.int32)

        return image, label

    def _create_dataset(self, shuffle_buffer, prefetch_batches):
        shard_paths_str = [str(p) for p in self.shard_paths]

        if self.shuffle:
            files = tf.data.Dataset.from_tensor_slices(shard_paths_str)
            files = files.shuffle(len(shard_paths_str))
            ds = files.interleave(
                lambda f: tf.data.TFRecordDataset(f, compression_type='GZIP'),
                cycle_length=min(8, len(shard_paths_str)),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            ds = ds.shuffle(shuffle_buffer)
        else:
            ds = tf.data.TFRecordDataset(shard_paths_str, compression_type='GZIP')

        ds = ds.map(self._parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(prefetch_batches)
        return ds

    def __iter__(self):
        for videos, labels in self.dataset:
            yield videos.numpy(), labels.numpy()

    def __len__(self):
        return self.n_samples // self.batch_size


def get_girik_tfrecord_loader(
    tfrecord_dir: str,
    batch_size: int,
    num_frames: int = 64,
    image_size: int = None,
    split: str = 'train',
    shuffle: bool = True,
    shuffle_buffer: int = 10000,
    prefetch_batches: int = 2,
):
    """Create a Girik PathTracker TFRecord loader."""
    return GirikTFRecordLoader(
        tfrecord_dir=tfrecord_dir,
        split=split,
        batch_size=batch_size,
        num_frames=num_frames,
        image_size=image_size,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        prefetch_batches=prefetch_batches,
    )
