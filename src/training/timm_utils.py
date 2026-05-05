"""
TIMM-style training utilities for JAX/Flax.

Implements:
- RandAugment
- Mixup / CutMix
- Random Erasing
- Label Smoothing
- EMA (Exponential Moving Average)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Dict, Any
from functools import partial


# =============================================================================
# Label Smoothing
# =============================================================================

def smooth_labels(labels: jnp.ndarray, num_classes: int, smoothing: float = 0.1) -> jnp.ndarray:
    """
    Apply label smoothing.

    Args:
        labels: One-hot labels (B, num_classes) or class indices (B,)
        num_classes: Number of classes
        smoothing: Smoothing factor (0.1 = 10% uniform, 90% true label)

    Returns:
        Smoothed one-hot labels
    """
    if labels.ndim == 1:
        labels = jax.nn.one_hot(labels, num_classes)

    return labels * (1.0 - smoothing) + smoothing / num_classes


def cross_entropy_with_smoothing(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    smoothing: float = 0.1
) -> jnp.ndarray:
    """Cross-entropy loss with label smoothing."""
    smooth = smooth_labels(labels, num_classes, smoothing)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(smooth * log_probs, axis=-1)


# =============================================================================
# Mixup / CutMix
# =============================================================================

def mixup(
    images: jnp.ndarray,
    labels: jnp.ndarray,
    rng: jax.Array,
    alpha: float = 0.8,
    num_classes: int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply Mixup augmentation (JIT-compatible).

    Args:
        images: Batch of images (B, H, W, C)
        labels: Class indices (B,) - must be integer labels, not one-hot
        rng: JAX random key
        alpha: Beta distribution parameter (higher = more mixing)
        num_classes: Number of classes

    Returns:
        Mixed images and soft labels (B, num_classes)
    """
    batch_size = images.shape[0]

    # Sample lambda from Beta distribution
    rng, rng_lambda, rng_perm = jax.random.split(rng, 3)
    lam = jax.random.beta(rng_lambda, alpha, alpha)

    # Random permutation for mixing
    perm = jax.random.permutation(rng_perm, batch_size)

    # Mix images
    mixed_images = lam * images + (1 - lam) * images[perm]

    # Convert labels to one-hot (always expects class indices)
    labels_onehot = jax.nn.one_hot(labels, num_classes)

    # Mix labels
    mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[perm]

    return mixed_images, mixed_labels


def cutmix(
    images: jnp.ndarray,
    labels: jnp.ndarray,
    rng: jax.Array,
    alpha: float = 1.0,
    num_classes: int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply CutMix augmentation (JIT-compatible).

    Args:
        images: Batch of images (B, H, W, C)
        labels: Class indices (B,) - must be integer labels, not one-hot
        rng: JAX random key
        alpha: Beta distribution parameter
        num_classes: Number of classes

    Returns:
        CutMixed images and soft labels (B, num_classes)
    """
    batch_size, H, W, C = images.shape

    # Sample lambda from Beta distribution
    rng, rng_lambda, rng_perm, rng_box = jax.random.split(rng, 4)
    lam = jax.random.beta(rng_lambda, alpha, alpha)

    # Random permutation for mixing
    perm = jax.random.permutation(rng_perm, batch_size)

    # Compute box size
    cut_ratio = jnp.sqrt(1 - lam)
    cut_h = (H * cut_ratio).astype(jnp.int32)
    cut_w = (W * cut_ratio).astype(jnp.int32)

    # Random box position
    rng_cx, rng_cy = jax.random.split(rng_box)
    cx = jax.random.randint(rng_cx, (), 0, H)
    cy = jax.random.randint(rng_cy, (), 0, W)

    # Box boundaries (clipped)
    x1 = jnp.clip(cx - cut_h // 2, 0, H)
    x2 = jnp.clip(cx + cut_h // 2, 0, H)
    y1 = jnp.clip(cy - cut_w // 2, 0, W)
    y2 = jnp.clip(cy + cut_w // 2, 0, W)

    # Create mask using coordinate comparison (JIT-compatible)
    # mask is 1 outside the box, 0 inside the box
    rows = jnp.arange(H)[:, None]  # (H, 1)
    cols = jnp.arange(W)[None, :]  # (1, W)
    mask = ~((rows >= x1) & (rows < x2) & (cols >= y1) & (cols < y2))
    mask = mask[:, :, None].astype(jnp.float32)  # (H, W, 1)

    # Apply CutMix
    mixed_images = images * mask + images[perm] * (1 - mask)

    # Actual lambda based on box area
    lam_actual = 1 - ((x2 - x1) * (y2 - y1)) / (H * W)

    # Convert labels to one-hot (always expects class indices)
    labels_onehot = jax.nn.one_hot(labels, num_classes)

    # Mix labels
    mixed_labels = lam_actual * labels_onehot + (1 - lam_actual) * labels_onehot[perm]

    return mixed_images, mixed_labels


def mixup_cutmix(
    images: jnp.ndarray,
    labels: jnp.ndarray,
    rng: jax.Array,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.5,
    cutmix_prob: float = 0.5,
    num_classes: int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply either Mixup or CutMix with given probabilities.

    Args:
        images: Batch of images (B, H, W, C)
        labels: Class indices (B,)
        rng: JAX random key
        mixup_alpha: Mixup beta parameter
        cutmix_alpha: CutMix beta parameter
        mixup_prob: Probability of applying mixup
        cutmix_prob: Probability of applying cutmix
        num_classes: Number of classes

    Returns:
        Augmented images and soft labels
    """
    rng, rng_choice = jax.random.split(rng)
    choice = jax.random.uniform(rng_choice)

    total_prob = mixup_prob + cutmix_prob
    threshold = mixup_prob / total_prob

    # Use jax.lax.cond for JIT-compatible conditional
    return jax.lax.cond(
        choice < threshold,
        lambda args: mixup(args[0], args[1], args[2], mixup_alpha, num_classes),
        lambda args: cutmix(args[0], args[1], args[2], cutmix_alpha, num_classes),
        (images, labels, rng),
    )


# =============================================================================
# Random Erasing
# =============================================================================

def random_erasing(
    images: jnp.ndarray,
    rng: jax.Array,
    probability: float = 0.25,
    min_area: float = 0.02,
    max_area: float = 0.33,
    min_aspect: float = 0.3,
    value: float = 0.0,
) -> jnp.ndarray:
    """
    Apply Random Erasing augmentation (vectorized for speed).

    Args:
        images: Batch of images (B, H, W, C)
        rng: JAX random key
        probability: Probability of applying erasing
        min_area: Minimum erased area ratio
        max_area: Maximum erased area ratio
        min_aspect: Minimum aspect ratio
        value: Fill value (0 = black, can also use mean)

    Returns:
        Images with random patches erased
    """
    batch_size, H, W, C = images.shape

    def erase_single(img, rng):
        rng, rng_prob, rng_area, rng_aspect, rng_pos = jax.random.split(rng, 5)

        # Check if we should erase
        do_erase = jax.random.uniform(rng_prob) < probability

        # Sample area and aspect ratio
        area = jax.random.uniform(rng_area) * (max_area - min_area) + min_area
        aspect = jnp.exp(jax.random.uniform(rng_aspect) * 2 * jnp.log(1/min_aspect) - jnp.log(1/min_aspect))

        # Compute box size
        erase_h = jnp.int32(jnp.sqrt(H * W * area / aspect))
        erase_w = jnp.int32(jnp.sqrt(H * W * area * aspect))
        erase_h = jnp.minimum(erase_h, H)
        erase_w = jnp.minimum(erase_w, W)

        # Random position
        rng_row, rng_col = jax.random.split(rng_pos)
        row_pos = jax.random.randint(rng_row, (), 0, H - erase_h + 1)
        col_pos = jax.random.randint(rng_col, (), 0, W - erase_w + 1)

        # Create mask using coordinate comparison (JIT-compatible)
        # mask is 1 outside the erased region, 0 inside
        rows = jnp.arange(H)[:, None]  # (H, 1)
        cols = jnp.arange(W)[None, :]  # (1, W)
        mask = ~((rows >= row_pos) & (rows < row_pos + erase_h) &
                 (cols >= col_pos) & (cols < col_pos + erase_w))
        mask = mask[:, :, None].astype(jnp.float32)  # (H, W, 1)

        # Apply erasing
        erased = img * mask + value * (1 - mask)

        # Apply conditionally
        return jnp.where(do_erase, erased, img)

    # Apply to batch in parallel using vmap (much faster than scan)
    rngs = jax.random.split(rng, batch_size)
    return jax.vmap(erase_single)(images, rngs)


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class EMAState:
    """State for EMA tracking."""
    def __init__(self, params, decay: float = 0.9999):
        self.ema_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
        self.decay = decay

    def update(self, params):
        """Update EMA parameters."""
        self.ema_params = jax.tree_util.tree_map(
            lambda ema, p: self.decay * ema + (1 - self.decay) * p,
            self.ema_params, params
        )

    def get_params(self):
        """Get EMA parameters for evaluation."""
        return self.ema_params


def create_ema_state(params, decay: float = 0.9999) -> Dict[str, Any]:
    """Create EMA state as a dict for easier handling."""
    return {
        'ema_params': jax.tree_util.tree_map(lambda x: x.copy(), params),
        'decay': decay,
    }


def update_ema(ema_state: Dict, params) -> Dict:
    """Update EMA state with new parameters."""
    decay = ema_state['decay']
    new_ema_params = jax.tree_util.tree_map(
        lambda ema, p: decay * ema + (1 - decay) * p,
        ema_state['ema_params'], params
    )
    return {'ema_params': new_ema_params, 'decay': decay}


# =============================================================================
# RandAugment (NumPy version for data loading)
# =============================================================================

def rand_augment_np(
    image: np.ndarray,
    num_ops: int = 2,
    magnitude: int = 9,
    magnitude_std: float = 0.5,
) -> np.ndarray:
    """
    Apply RandAugment using NumPy/PIL operations.

    Args:
        image: Image as numpy array (H, W, C), values in [0, 255]
        num_ops: Number of operations to apply
        magnitude: Magnitude of operations (0-10)
        magnitude_std: Standard deviation for magnitude sampling

    Returns:
        Augmented image
    """
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter

    # Convert to PIL
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(image)

    # Define augmentation operations
    def autocontrast(img, _):
        return ImageOps.autocontrast(img)

    def equalize(img, _):
        return ImageOps.equalize(img)

    def invert(img, _):
        return ImageOps.invert(img)

    def rotate(img, mag):
        angle = mag * 30 / 10  # max 30 degrees
        if np.random.random() < 0.5:
            angle = -angle
        return img.rotate(angle, fillcolor=(128, 128, 128))

    def posterize(img, mag):
        bits = int(8 - mag * 4 / 10)
        bits = max(1, min(8, bits))
        return ImageOps.posterize(img, bits)

    def solarize(img, mag):
        thresh = int(256 - mag * 256 / 10)
        return ImageOps.solarize(img, thresh)

    def color(img, mag):
        factor = 1.0 + mag * 0.9 / 10
        if np.random.random() < 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Color(img).enhance(factor)

    def contrast(img, mag):
        factor = 1.0 + mag * 0.9 / 10
        if np.random.random() < 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Contrast(img).enhance(factor)

    def brightness(img, mag):
        factor = 1.0 + mag * 0.9 / 10
        if np.random.random() < 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Brightness(img).enhance(factor)

    def sharpness(img, mag):
        factor = 1.0 + mag * 0.9 / 10
        if np.random.random() < 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Sharpness(img).enhance(factor)

    def shear_x(img, mag):
        shear = mag * 0.3 / 10
        if np.random.random() < 0.5:
            shear = -shear
        return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), fillcolor=(128, 128, 128))

    def shear_y(img, mag):
        shear = mag * 0.3 / 10
        if np.random.random() < 0.5:
            shear = -shear
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0), fillcolor=(128, 128, 128))

    def translate_x(img, mag):
        pixels = int(mag * img.size[0] * 0.45 / 10)
        if np.random.random() < 0.5:
            pixels = -pixels
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=(128, 128, 128))

    def translate_y(img, mag):
        pixels = int(mag * img.size[1] * 0.45 / 10)
        if np.random.random() < 0.5:
            pixels = -pixels
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=(128, 128, 128))

    # List of operations
    ops = [
        autocontrast, equalize, rotate, posterize, solarize,
        color, contrast, brightness, sharpness,
        shear_x, shear_y, translate_x, translate_y,
    ]

    # Apply random operations
    for _ in range(num_ops):
        op = ops[np.random.randint(len(ops))]
        mag = magnitude + np.random.randn() * magnitude_std
        mag = np.clip(mag, 0, 10)
        try:
            pil_img = op(pil_img, mag)
        except Exception:
            pass  # Skip if operation fails

    # Convert back to numpy
    return np.array(pil_img)


# =============================================================================
# GPU-Accelerated Augmentations (JAX)
# =============================================================================

def color_jitter_jax(
    images: jnp.ndarray,
    rng: jax.Array,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
) -> jnp.ndarray:
    """
    Apply color jitter on GPU (JAX).

    Args:
        images: Batch of images (B, H, W, C), normalized to [0, 1] or ImageNet normalized
        rng: JAX random key
        brightness, contrast, saturation: Jitter factors

    Returns:
        Augmented images
    """
    rng_b, rng_c, rng_s = jax.random.split(rng, 3)

    # Brightness: multiply by random factor
    b_factor = 1.0 + jax.random.uniform(rng_b, (images.shape[0], 1, 1, 1), minval=-brightness, maxval=brightness)
    images = images * b_factor

    # Contrast: blend with mean
    mean = jnp.mean(images, axis=(1, 2, 3), keepdims=True)
    c_factor = 1.0 + jax.random.uniform(rng_c, (images.shape[0], 1, 1, 1), minval=-contrast, maxval=contrast)
    images = mean + (images - mean) * c_factor

    # Saturation: blend with grayscale
    gray = jnp.mean(images, axis=-1, keepdims=True)
    s_factor = 1.0 + jax.random.uniform(rng_s, (images.shape[0], 1, 1, 1), minval=-saturation, maxval=saturation)
    images = gray + (images - gray) * s_factor

    return images


def solarize_jax(images: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    """Solarize: invert pixels above threshold."""
    # For normalized images, threshold is relative to the range
    return jnp.where(images > threshold, 1.0 - images, images)


def grayscale_jax(images: jnp.ndarray) -> jnp.ndarray:
    """Convert to grayscale (keeping 3 channels)."""
    # Standard luminance weights
    weights = jnp.array([0.2989, 0.5870, 0.1140])
    gray = jnp.sum(images * weights, axis=-1, keepdims=True)
    return jnp.broadcast_to(gray, images.shape)


def gaussian_blur_jax(images: jnp.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> jnp.ndarray:
    """Apply Gaussian blur using separable convolution."""
    # Create 1D Gaussian kernel
    x = jnp.arange(kernel_size) - kernel_size // 2
    kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Reshape for depthwise conv: (kernel_size, 1, 1, 1)
    kernel_h = kernel_1d.reshape(-1, 1, 1, 1)
    kernel_w = kernel_1d.reshape(1, -1, 1, 1)

    # Apply separable convolution per channel
    # JAX conv expects (N, H, W, C) with kernel (H, W, C_in, C_out)
    # Use lax.conv_general_dilated for more control
    from jax import lax

    B, H, W, C = images.shape

    # Pad for 'same' output
    pad = kernel_size // 2

    # Process each channel
    def blur_channel(channel_img):
        # channel_img: (B, H, W)
        img = channel_img[:, :, :, None]  # (B, H, W, 1)

        # Horizontal blur
        kernel_h_full = jnp.tile(kernel_1d.reshape(1, -1, 1, 1), (1, 1, 1, 1))
        img = lax.conv_general_dilated(
            img, kernel_h_full,
            window_strides=(1, 1),
            padding=[(0, 0), (pad, pad)],
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )

        # Vertical blur
        kernel_v_full = jnp.tile(kernel_1d.reshape(-1, 1, 1, 1), (1, 1, 1, 1))
        img = lax.conv_general_dilated(
            img, kernel_v_full,
            window_strides=(1, 1),
            padding=[(pad, pad), (0, 0)],
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )

        return img[:, :, :, 0]

    # Apply to each channel
    channels = [blur_channel(images[:, :, :, c]) for c in range(C)]
    return jnp.stack(channels, axis=-1)


def three_augment_jax(
    images: jnp.ndarray,
    rng: jax.Array,
    color_jitter: float = 0.3,
) -> jnp.ndarray:
    """
    GPU-accelerated 3-Augment (DeiT III style).

    Applies ONE of three transforms per image:
    1. Grayscale
    2. Solarization
    3. Gaussian blur

    Then applies color jitter to all.

    Args:
        images: Batch of images (B, H, W, C)
        rng: JAX random key
        color_jitter: Color jitter factor

    Returns:
        Augmented images
    """
    B = images.shape[0]
    rng, rng_choice, rng_color = jax.random.split(rng, 3)

    # Random choice per image: 0=grayscale, 1=solarize, 2=blur
    choices = jax.random.randint(rng_choice, (B,), 0, 3)

    # Apply all three transforms
    gray_images = grayscale_jax(images)
    solar_images = solarize_jax(images, threshold=0.5)
    blur_images = gaussian_blur_jax(images, kernel_size=5, sigma=1.0)

    # Select based on choice (vectorized)
    # choices: (B,) -> (B, 1, 1, 1) for broadcasting
    choice_expand = choices[:, None, None, None]

    result = jnp.where(choice_expand == 0, gray_images,
             jnp.where(choice_expand == 1, solar_images, blur_images))

    # Apply color jitter
    if color_jitter > 0:
        result = color_jitter_jax(result, rng_color,
                                   brightness=color_jitter,
                                   contrast=color_jitter,
                                   saturation=color_jitter)

    return result


# =============================================================================
# 3-Augment (DeiT III) - NumPy/CPU version
# =============================================================================

def three_augment_np(
    image: np.ndarray,
    color_jitter: float = 0.3,
) -> np.ndarray:
    """
    Apply DeiT III's 3-Augment.

    3-Augment applies ONE of three transforms with equal probability:
    1. Grayscale conversion
    2. Solarization (threshold 128)
    3. Gaussian blur (radius 2)

    Then followed by:
    - Color jitter (brightness, contrast, saturation)
    - Horizontal flip (handled separately in data loader)

    Reference: DeiT III (https://arxiv.org/abs/2204.07118)

    Args:
        image: Image as numpy array (H, W, C), values in [0, 255] uint8
        color_jitter: Color jitter factor (default 0.3)

    Returns:
        Augmented image as numpy array
    """
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter

    # Convert to PIL
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(image)

    # Apply one of three augmentations with equal probability
    choice = np.random.randint(3)

    if choice == 0:
        # Grayscale
        pil_img = ImageOps.grayscale(pil_img).convert('RGB')
    elif choice == 1:
        # Solarization (invert pixels above threshold)
        pil_img = ImageOps.solarize(pil_img, threshold=128)
    else:
        # Gaussian blur
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=2))

    # Color jitter (brightness, contrast, saturation)
    if color_jitter > 0:
        # Brightness
        factor = 1.0 + np.random.uniform(-color_jitter, color_jitter)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)

        # Contrast
        factor = 1.0 + np.random.uniform(-color_jitter, color_jitter)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)

        # Saturation
        factor = 1.0 + np.random.uniform(-color_jitter, color_jitter)
        pil_img = ImageEnhance.Color(pil_img).enhance(factor)

    return np.array(pil_img)


# =============================================================================
# Combined Augmentation Pipeline
# =============================================================================

def apply_train_augmentations_np(
    image: np.ndarray,
    augmentation_type: str = 'randaugment',
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    color_jitter: float = 0.3,
) -> np.ndarray:
    """
    Apply training augmentations in NumPy (before batch creation).

    This should be called in the data loader on individual images.
    Mixup/CutMix are applied in JAX on the GPU after batching.

    Args:
        image: Input image (H, W, C)
        augmentation_type: 'randaugment', '3aug', or 'none'
        randaugment_num_ops: Number of RandAugment operations
        randaugment_magnitude: RandAugment magnitude (0-10)
        color_jitter: Color jitter factor for 3-Augment

    Returns:
        Augmented image
    """
    if augmentation_type == 'randaugment':
        # RandAugment expects [0, 255] uint8
        if image.dtype == np.float32 and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        image = rand_augment_np(
            image,
            num_ops=randaugment_num_ops,
            magnitude=randaugment_magnitude,
        )

        # Convert back to float32 [0, 1]
        image = image.astype(np.float32) / 255.0

    elif augmentation_type == '3aug':
        # 3-Augment (DeiT III)
        if image.dtype == np.float32 and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        image = three_augment_np(image, color_jitter=color_jitter)

        # Convert back to float32 [0, 1]
        image = image.astype(np.float32) / 255.0

    # 'none' or other: no augmentation

    return image
