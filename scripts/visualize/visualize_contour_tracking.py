#!/usr/bin/env python3
"""
Visualize if per-pixel evidence tracks the contours.
Show evidence magnitude overlaid on contour mask.
"""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, '/media/data_cifs/projects/prj_video_imagenet/CSSM')


class ModelWithFeatures(nn.Module):
    num_classes: int = 2
    embed_dim: int = 64
    kernel_size: int = 15
    seq_len: int = 8

    @nn.compact
    def __call__(self, x, training=False):
        from src.models.cssm import TransformerCSSM, apply_rope, apply_sinusoidal_temporal_encoding

        act = nn.softplus

        if x.ndim == 4:
            x = jnp.repeat(x[:, None, ...], self.seq_len, axis=1)

        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C)

        x = nn.Conv(self.embed_dim, (3, 3), padding='SAME', name='conv1')(x)
        x = act(x)
        x = nn.LayerNorm(name='norm1')(x)
        x = nn.max_pool(x, (2, 2), (2, 2), 'SAME')

        x = nn.Conv(self.embed_dim, (3, 3), padding='SAME', name='conv2')(x)
        x = act(x)
        x = nn.LayerNorm(name='norm2')(x)
        x = nn.max_pool(x, (2, 2), (2, 2), 'SAME')

        _, H_new, W_new, _ = x.shape
        x = x.reshape(B, T, H_new, W_new, self.embed_dim)

        x = apply_rope(x, mode='spatial_only')
        x = apply_sinusoidal_temporal_encoding(x)

        cssm = TransformerCSSM(
            channels=self.embed_dim,
            kernel_size=self.kernel_size,
            block_size=1,
            rope_mode='none',
            name='cssm_0'
        )
        x = x + cssm(x)

        features = x

        x_last = x[:, -1]
        x_last = nn.LayerNorm(name='norm_pre')(x_last)
        x_last = act(x_last)
        x_last = x_last.mean(axis=(1, 2))
        x_last = nn.LayerNorm(name='norm_post')(x_last)
        logits = nn.Dense(self.num_classes, name='head')(x_last)

        return logits, features


def apply_perpixel_readout(features, params):
    B, T, H, W, C = features.shape
    act = nn.softplus

    norm_pre_scale = params['norm_pre']['scale']
    norm_pre_bias = params['norm_pre']['bias']
    head_kernel = params['head']['kernel']
    head_bias = params['head']['bias']

    x = features.reshape(-1, C)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-6) * norm_pre_scale + norm_pre_bias
    x = act(x)
    logits = x @ head_kernel + head_bias

    return logits.reshape(B, T, H, W, -1)


def extract_contour_mask(image):
    """Extract contour pixels from pathfinder image."""
    # Pathfinder images have contours as brighter pixels
    gray = image.mean(axis=-1)
    # Threshold to get contour mask
    threshold = gray.mean() + gray.std()
    mask = gray > threshold
    return mask.astype(float)


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    print("Loading checkpoint...")
    with open('checkpoints/KQV_sinusoidal/epoch_15/checkpoint.pkl', 'rb') as f:
        params = pickle.load(f)['params']

    print("Loading data...")
    val_files = sorted(tf.io.gfile.glob('/home/dlinsley/pathfinder_tfrecord/difficulty_14/val/*.tfrecord'))

    def parse_example(example):
        features = tf.io.parse_single_example(example, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
        image = tf.io.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [224, 224, 3])
        return image, features['label']

    ds = tf.data.TFRecordDataset(val_files[:2]).map(parse_example)
    images, labels = [], []
    for img, lbl in ds:
        images.append(img.numpy())
        labels.append(lbl.numpy())
        if len(images) >= 50:
            break

    model = ModelWithFeatures(seq_len=8)

    # Find good examples
    correct_conn, correct_disc = [], []
    for i in range(len(images)):
        x = jnp.array(images[i])[None]
        logits, _ = model.apply({'params': params}, x, training=False)
        pred = int(logits.argmax())
        if pred == labels[i]:
            if labels[i] == 1:
                correct_conn.append(i)
            else:
                correct_disc.append(i)

    print(f"Found {len(correct_conn)} correct connected, {len(correct_disc)} correct disconnected")

    # Analyze contour tracking
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))

    for row, (name, idx_list) in enumerate([('Connected', correct_conn[:2]), ('Disconnected', correct_disc[:2])]):
        for col, idx in enumerate(idx_list):
            image = images[idx]
            label = labels[idx]

            x = jnp.array(image)[None]
            logits, features = model.apply({'params': params}, x, training=False)
            perpixel_logits = apply_perpixel_readout(features, params)
            evidence = np.array(perpixel_logits[0, :, :, :, 1] - perpixel_logits[0, :, :, :, 0])

            # Get contour mask
            contour_mask = extract_contour_mask(image)

            img_display = (image - image.min()) / (image.max() - image.min() + 1e-8)

            # Show original image
            axes[row*2, col*3].imshow(img_display)
            axes[row*2, col*3].set_title(f'{name} (label={label})')
            axes[row*2, col*3].axis('off')

            # Show contour mask
            axes[row*2, col*3+1].imshow(contour_mask, cmap='gray')
            axes[row*2, col*3+1].set_title('Contour mask')
            axes[row*2, col*3+1].axis('off')

            # Show evidence at T=8 overlaid on contour
            ev_final = evidence[-1]  # T=8
            ev_up = cv2.resize(ev_final, (224, 224), interpolation=cv2.INTER_LINEAR)
            vmax = max(abs(ev_up.min()), abs(ev_up.max()))

            axes[row*2, col*3+2].imshow(img_display, alpha=0.3)
            axes[row*2, col*3+2].imshow(ev_up, cmap='RdBu_r', alpha=0.7, vmin=-vmax, vmax=vmax)
            axes[row*2, col*3+2].set_title(f'Evidence T=8\n[{ev_final.min():.1f}, {ev_final.max():.1f}]')
            axes[row*2, col*3+2].axis('off')

            # Correlation analysis: does evidence follow contours?
            ev_up_flat = ev_up.flatten()
            contour_flat = contour_mask.flatten()

            # Evidence on contour vs off contour
            on_contour_ev = ev_up_flat[contour_flat > 0.5].mean() if (contour_flat > 0.5).sum() > 0 else 0
            off_contour_ev = ev_up_flat[contour_flat <= 0.5].mean() if (contour_flat <= 0.5).sum() > 0 else 0

            # Show evidence evolution
            times = np.arange(1, 9)
            global_ev = evidence.mean(axis=(1, 2))

            # Also track evidence specifically on contours
            contour_mask_down = cv2.resize(contour_mask, (56, 56), interpolation=cv2.INTER_NEAREST)
            contour_ev = np.array([evidence[t][contour_mask_down > 0.5].mean() if (contour_mask_down > 0.5).sum() > 0 else 0 for t in range(8)])
            noncontour_ev = np.array([evidence[t][contour_mask_down <= 0.5].mean() if (contour_mask_down <= 0.5).sum() > 0 else 0 for t in range(8)])

            axes[row*2+1, col*3].plot(times, global_ev, 'k-', label='Global', linewidth=2)
            axes[row*2+1, col*3].plot(times, contour_ev, 'r--', label='On contour', linewidth=2)
            axes[row*2+1, col*3].plot(times, noncontour_ev, 'b--', label='Off contour', linewidth=2)
            axes[row*2+1, col*3].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            axes[row*2+1, col*3].legend(fontsize=8)
            axes[row*2+1, col*3].set_xlabel('Timestep')
            axes[row*2+1, col*3].set_ylabel('Mean Evidence')
            axes[row*2+1, col*3].set_title('Evidence by location')

            # Hide unused axes
            axes[row*2+1, col*3+1].axis('off')
            axes[row*2+1, col*3+2].axis('off')

    plt.suptitle("Per-pixel evidence vs contour location\n(Does the model focus on contours?)", fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/contour_tracking_analysis.png', dpi=150)
    plt.close()
    print("Saved: visualizations/contour_tracking_analysis.png")

    # Print summary statistics
    print("\n=== Contour Tracking Analysis ===")
    for name, idx_list in [('Connected', correct_conn[:5]), ('Disconnected', correct_disc[:5])]:
        print(f"\n{name}:")
        for idx in idx_list:
            image = images[idx]
            x = jnp.array(image)[None]
            _, features = model.apply({'params': params}, x, training=False)
            perpixel_logits = apply_perpixel_readout(features, params)
            evidence = np.array(perpixel_logits[0, -1])  # T=8

            contour_mask = extract_contour_mask(image)
            contour_mask_down = cv2.resize(contour_mask, (56, 56), interpolation=cv2.INTER_NEAREST)

            on_ev = evidence[..., 1][contour_mask_down > 0.5].mean() - evidence[..., 0][contour_mask_down > 0.5].mean()
            off_ev = evidence[..., 1][contour_mask_down <= 0.5].mean() - evidence[..., 0][contour_mask_down <= 0.5].mean()

            print(f"  idx={idx}: On-contour={on_ev:.2f}, Off-contour={off_ev:.2f}, Diff={on_ev-off_ev:.2f}")


if __name__ == '__main__':
    main()
