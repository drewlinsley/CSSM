"""
Data loading utilities for CSSM ConvNeXt.

Provides video-style data loading from Imagenette dataset by repeating
static images across the temporal dimension.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Iterator, Tuple
import numpy as np


# ImageNet normalization constants
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])


def get_imagenette_video_loader(
    batch_size: int,
    sequence_length: int = 8,
    split: str = 'train',
    image_size: int = 224,
    shuffle_buffer: int = 1000,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Load Imagenette dataset and convert to video format.

    Creates "video" samples by repeating each static image T times.
    This simulates video input for models designed for temporal data.

    Args:
        batch_size: Number of samples per batch
        sequence_length: Number of frames (T) to repeat each image
        split: Dataset split ('train' or 'validation')
        image_size: Target image size (square)
        shuffle_buffer: Size of shuffle buffer (only used for training)

    Returns:
        Iterator yielding (video, label) tuples where:
            video: np.ndarray of shape (B, T, H, W, 3)
            label: np.ndarray of shape (B,) with class indices

    Example:
        >>> loader = get_imagenette_video_loader(batch_size=8, sequence_length=4)
        >>> for videos, labels in loader:
        ...     print(videos.shape)  # (8, 4, 224, 224, 3)
        ...     break
    """
    # Load Imagenette dataset
    ds = tfds.load('imagenette/320px-v2', split=split, as_supervised=True)

    def preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Preprocess single image into video format."""
        # Resize to target size
        image = tf.image.resize(image, (image_size, image_size))

        # Convert to float and normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # Apply ImageNet normalization
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        # Repeat to create video: (H, W, C) -> (T, H, W, C)
        video = tf.repeat(tf.expand_dims(image, 0), sequence_length, axis=0)

        return video, label

    # Apply preprocessing
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle for training split
    if split == 'train':
        ds = ds.shuffle(shuffle_buffer)

    # Batch and prefetch
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds.as_numpy_iterator()


def get_dataset_info() -> dict:
    """
    Get information about the Imagenette dataset.

    Returns:
        Dictionary with dataset metadata including:
            - num_classes: Number of classes (10)
            - class_names: List of class names
            - train_size: Number of training samples
            - val_size: Number of validation samples
    """
    info = tfds.builder('imagenette/320px-v2').info

    return {
        'num_classes': info.features['label'].num_classes,
        'class_names': info.features['label'].names,
        'train_size': info.splits['train'].num_examples,
        'val_size': info.splits['validation'].num_examples,
    }


def create_data_loaders(
    batch_size: int,
    sequence_length: int = 8,
    image_size: int = 224,
) -> Tuple[Iterator, Iterator, dict]:
    """
    Create train and validation data loaders.

    Convenience function to create both loaders with consistent settings.

    Args:
        batch_size: Number of samples per batch
        sequence_length: Number of frames per video
        image_size: Target image size

    Returns:
        Tuple of (train_loader, val_loader, dataset_info)
    """
    train_loader = get_imagenette_video_loader(
        batch_size=batch_size,
        sequence_length=sequence_length,
        split='train',
        image_size=image_size,
    )

    val_loader = get_imagenette_video_loader(
        batch_size=batch_size,
        sequence_length=sequence_length,
        split='validation',
        image_size=image_size,
    )

    info = get_dataset_info()

    return train_loader, val_loader, info
