"""
Data loading utilities for CSSM ConvNeXt.

Provides video-style data loading from Imagenette dataset by repeating
static images across the temporal dimension.
"""

from pathlib import Path
from typing import Iterator, Tuple
import numpy as np
from PIL import Image


# Imagenette class folder names (10 classes)
IMAGENETTE_CLASSES = [
    'n01440764',  # tench
    'n02102040',  # English springer
    'n02979186',  # cassette player
    'n03000684',  # chain saw
    'n03028079',  # church
    'n03394916',  # French horn
    'n03417042',  # garbage truck
    'n03425413',  # gas pump
    'n03445777',  # golf ball
    'n03888257',  # parachute
]

# Human-readable class names
CLASS_NAMES = [
    'tench', 'English springer', 'cassette player', 'chain saw',
    'church', 'French horn', 'garbage truck', 'gas pump',
    'golf ball', 'parachute'
]

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_val(path: str, size: int = 224) -> np.ndarray:
    """Load and preprocess a single image for validation (center crop)."""
    img = Image.open(path).convert('RGB')

    # Resize maintaining aspect ratio, then center crop
    w, h = img.size
    scale = size / min(w, h) * 1.15  # Slight over-scale for crop
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    # Convert to array and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    return arr


def load_image_train(path: str, size: int = 224) -> np.ndarray:
    """Load image with training augmentation (random resized crop + flip)."""
    img = Image.open(path).convert('RGB')
    w, h = img.size

    # Random resized crop
    scale = np.random.uniform(0.08, 1.0)
    ratio = np.random.uniform(0.75, 1.333)

    crop_w = int(min(w, h) * np.sqrt(scale * ratio))
    crop_h = int(min(w, h) * np.sqrt(scale / ratio))
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    left = np.random.randint(0, max(1, w - crop_w + 1))
    top = np.random.randint(0, max(1, h - crop_h + 1))

    img = img.crop((left, top, left + crop_w, top + crop_h))
    img = img.resize((size, size), Image.BILINEAR)

    # Random horizontal flip
    if np.random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Convert to array and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    return arr


def load_dataset(
    data_dir: str,
    split: str = 'train',
    image_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load entire dataset into memory (Imagenette is small enough).

    Args:
        data_dir: Path to imagenette2-320 directory
        split: 'train' or 'val'
        image_size: Target image size

    Returns:
        images: (N, H, W, 3)
        labels: (N,)
    """
    split_dir = Path(data_dir) / split

    images = []
    labels = []

    class_to_idx = {c: i for i, c in enumerate(IMAGENETTE_CLASSES)}
    load_fn = load_image_train if split == 'train' else load_image_val

    for class_name in IMAGENETTE_CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        for img_path in class_dir.glob('*.JPEG'):
            try:
                img = load_fn(str(img_path), image_size)
                images.append(img)
                labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

    return np.stack(images), np.array(labels)


class VideoDataLoader:
    """
    Data loader that creates "videos" by repeating static images.

    This simulates video input for models designed for temporal data
    by repeating each image T times along the time dimension.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        sequence_length: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """
        Args:
            images: (N, H, W, 3) image array
            labels: (N,) label array
            batch_size: Number of samples per batch
            sequence_length: Number of frames (T) to repeat each image
            shuffle: Whether to shuffle data
            drop_last: Drop incomplete final batch
        """
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(images)

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            batch_images = self.images[batch_indices]  # (B, H, W, 3)
            batch_labels = self.labels[batch_indices]  # (B,)

            # Repeat to create video: (B, H, W, 3) -> (B, T, H, W, 3)
            batch_videos = np.repeat(
                batch_images[:, np.newaxis, :, :, :],
                self.sequence_length,
                axis=1
            )

            yield batch_videos, batch_labels


def get_imagenette_video_loader(
    data_dir: str,
    batch_size: int,
    sequence_length: int = 8,
    split: str = 'train',
    image_size: int = 224,
) -> VideoDataLoader:
    """
    Load Imagenette dataset and return a video data loader.

    Creates "video" samples by repeating each static image T times.
    This simulates video input for models designed for temporal data.

    Args:
        data_dir: Path to imagenette2-320 directory
        batch_size: Number of samples per batch
        sequence_length: Number of frames (T) to repeat each image
        split: Dataset split ('train' or 'val')
        image_size: Target image size (square)

    Returns:
        VideoDataLoader yielding (video, label) tuples where:
            video: np.ndarray of shape (B, T, H, W, 3)
            label: np.ndarray of shape (B,) with class indices
    """
    print(f"Loading {split} split from {data_dir}...")
    images, labels = load_dataset(data_dir, split, image_size)
    print(f"  Loaded {len(images)} images")

    return VideoDataLoader(
        images=images,
        labels=labels,
        batch_size=batch_size,
        sequence_length=sequence_length,
        shuffle=(split == 'train'),
        drop_last=True,
    )


def get_dataset_info() -> dict:
    """
    Get information about the Imagenette dataset.

    Returns:
        Dictionary with dataset metadata including:
            - num_classes: Number of classes (10)
            - class_names: List of class names
            - train_size: Approximate number of training samples
            - val_size: Approximate number of validation samples
    """
    return {
        'num_classes': 10,
        'class_names': CLASS_NAMES,
        'train_size': 9469,  # Approximate
        'val_size': 3925,    # Approximate
    }


def create_data_loaders(
    data_dir: str,
    batch_size: int,
    sequence_length: int = 8,
    image_size: int = 224,
) -> Tuple[VideoDataLoader, VideoDataLoader, dict]:
    """
    Create train and validation data loaders.

    Convenience function to create both loaders with consistent settings.

    Args:
        data_dir: Path to imagenette2-320 directory
        batch_size: Number of samples per batch
        sequence_length: Number of frames per video
        image_size: Target image size

    Returns:
        Tuple of (train_loader, val_loader, dataset_info)
    """
    train_loader = get_imagenette_video_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        sequence_length=sequence_length,
        split='train',
        image_size=image_size,
    )

    val_loader = get_imagenette_video_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        sequence_length=sequence_length,
        split='val',
        image_size=image_size,
    )

    info = get_dataset_info()

    return train_loader, val_loader, info
