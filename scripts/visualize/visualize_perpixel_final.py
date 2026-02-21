#!/usr/bin/env python3
"""
Final per-pixel visualization following exact user specification:
CSSM output -> norm_pre -> activation -> norm_post -> head (per pixel, skip pool)
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

sys.path.insert(0, '/media/data_cifs/projects/prj_video_imagenet/CSSM')


def load_tfrecord_batch(tfrecord_files, max_examples=None):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    def parse_example(example):
        features = tf.io.parse_single_example(example, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
        image = tf.io.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [224, 224, 3])
        return image, features['label']

    ds = tf.data.TFRecordDataset(tfrecord_files).map(parse_example)
    images, labels = [], []
    for img, lbl in tqdm(ds, desc="Loading"):
        images.append(img.numpy())
        labels.append(lbl.numpy())
        if max_examples and len(images) >= max_examples:
            break
    return np.array(images), np.array(labels)


class ModelWithFeatures(nn.Module):
    """Model that returns features after CSSM for per-pixel analysis."""
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

        features = x  # (B, T, H', W', C)

        # Standard readout for prediction
        x_last = x[:, -1]
        x_last = nn.LayerNorm(name='norm_pre')(x_last)
        x_last = act(x_last)
        x_last = x_last.mean(axis=(1, 2))
        x_last = nn.LayerNorm(name='norm_post')(x_last)
        logits = nn.Dense(self.num_classes, name='head')(x_last)

        return logits, features


def apply_perpixel_readout(features, params):
    """
    Apply exact readout pipeline per-pixel:
    - norm_pre (LayerNorm)
    - activation (softplus)
    - norm_post (LayerNorm)
    - head (Dense projection)
    Skip: spatial pool only
    """
    B, T, H, W, C = features.shape
    act = nn.softplus

    norm_pre_scale = params['norm_pre']['scale']
    norm_pre_bias = params['norm_pre']['bias']
    norm_post_scale = params['norm_post']['scale']
    norm_post_bias = params['norm_post']['bias']
    head_kernel = params['head']['kernel']
    head_bias = params['head']['bias']

    # Process all pixels
    x = features.reshape(-1, C)

    # norm_pre (LayerNorm per pixel)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-6) * norm_pre_scale + norm_pre_bias

    # activation
    x = act(x)

    # norm_post (LayerNorm per pixel)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-6) * norm_post_scale + norm_post_bias

    # head projection
    logits = x @ head_kernel + head_bias

    return logits.reshape(B, T, H, W, -1)


def create_visualization(image, label, params, seq_len, output_path):
    """Create visualization showing per-pixel evidence evolution."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import cv2

    model = ModelWithFeatures(seq_len=seq_len)
    x = jnp.array(image)[None]
    logits, features = model.apply({'params': params}, x, training=False)

    final_pred = int(np.array(logits.argmax()))

    # Per-pixel readout: norm_pre -> act -> norm_post -> head
    perpixel_logits = apply_perpixel_readout(features, params)
    perpixel_logits = np.array(perpixel_logits[0])  # (T, H', W', 2)

    # Evidence: positive = connected, negative = disconnected
    evidence = perpixel_logits[..., 1] - perpixel_logits[..., 0]

    # Global evidence at each timestep
    global_evidence = evidence.mean(axis=(1, 2))

    # Image setup
    img_display = (image - image.min()) / (image.max() - image.min() + 1e-8)
    H_orig, W_orig = img_display.shape[:2]

    # Use symmetric colorscale centered at 0
    vmax = max(abs(evidence.min()), abs(evidence.max()))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    def update(t):
        for ax in axes:
            ax.clear()

        # 1. Input image
        axes[0].imshow(img_display)
        true_str = "Connected" if label == 1 else "Disconnected"
        pred_str = "Connected" if final_pred == 1 else "Disconnected"
        color = 'green' if final_pred == label else 'red'
        axes[0].set_title(f"Input\nTrue: {true_str}\nPred: {pred_str}", color=color, fontsize=10)
        axes[0].axis('off')

        # 2. Per-pixel evidence overlay
        ev_up = cv2.resize(evidence[t], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        axes[1].imshow(img_display, alpha=0.4)
        axes[1].imshow(ev_up, cmap='RdBu_r', alpha=0.7, vmin=-vmax, vmax=vmax)
        axes[1].set_title(f't={t+1}/{seq_len}: Per-pixel evidence\n(red=conn, blue=disc)', fontsize=9)
        axes[1].axis('off')

        # 3. Evidence heatmap only (no image overlay)
        im = axes[2].imshow(ev_up, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2].set_title(f't={t+1}: Evidence heatmap\nrange: [{evidence[t].min():.1f}, {evidence[t].max():.1f}]', fontsize=9)
        axes[2].axis('off')

        # 4. Global evidence timeline
        times = np.arange(1, seq_len + 1)
        axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        axes[3].plot(times[:t+1], global_evidence[:t+1], 'ko-', markersize=6, linewidth=2)
        axes[3].fill_between(times[:t+1], 0, global_evidence[:t+1],
                            where=global_evidence[:t+1] >= 0, alpha=0.3, color='red')
        axes[3].fill_between(times[:t+1], 0, global_evidence[:t+1],
                            where=global_evidence[:t+1] < 0, alpha=0.3, color='blue')
        axes[3].axvline(x=t+1, color='orange', alpha=0.5, linewidth=3)
        axes[3].set_xlim(0.5, seq_len + 0.5)
        yabs = max(abs(global_evidence.min()), abs(global_evidence.max()), 0.5) * 1.3
        axes[3].set_ylim(-yabs, yabs)
        axes[3].set_xlabel('Timestep')
        axes[3].set_ylabel('Mean Evidence')
        axes[3].set_title('Global evidence\n(>0 = Connected vote)', fontsize=10)

        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=600, blit=False)
    ani.save(output_path, writer='pillow', fps=2)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("="*60)
    print("PER-PIXEL VISUALIZATION - WITH NORM_POST")
    print("Pipeline: CSSM -> norm_pre -> act -> norm_post -> head")
    print("="*60)

    with open('checkpoints/KQV_sinusoidal/epoch_15/checkpoint.pkl', 'rb') as f:
        params = pickle.load(f)['params']

    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    val_files = sorted(tf.io.gfile.glob('/home/dlinsley/pathfinder_tfrecord/difficulty_14/val/*.tfrecord'))
    val_images, val_labels = load_tfrecord_batch(val_files, max_examples=200)

    model = ModelWithFeatures(seq_len=8)

    # Find examples
    correct_conn, correct_disc, incorrect = [], [], []
    for i in tqdm(range(min(100, len(val_images))), desc="Finding examples"):
        x = jnp.array(val_images[i])[None]
        logits, _ = model.apply({'params': params}, x, training=False)
        pred = int(logits.argmax())
        lbl = int(val_labels[i])
        if pred == lbl:
            (correct_conn if lbl == 1 else correct_disc).append(i)
        else:
            incorrect.append(i)

    acc = (len(correct_conn) + len(correct_disc)) / 100
    print(f"\nAccuracy: {acc*100:.1f}%")
    print(f"Found: {len(correct_conn)} correct conn, {len(correct_disc)} correct disc, {len(incorrect)} incorrect")

    os.makedirs('visualizations', exist_ok=True)

    # Create visualizations
    for name, idx_list in [('connected', correct_conn), ('disconnected', correct_disc), ('incorrect', incorrect)]:
        if idx_list:
            idx = idx_list[0]
            print(f"\nCreating {name} visualization (idx={idx})...")
            create_visualization(val_images[idx], val_labels[idx], params, 8,
                               f'visualizations/final_{name}_T8.gif')

    # Also create a static multi-timestep view
    print("\nCreating static comparison...")
    create_static_comparison(val_images, val_labels, params, correct_conn, correct_disc, incorrect)

    print("\nDone!")


def create_static_comparison(val_images, val_labels, params, correct_conn, correct_disc, incorrect):
    """Create a static image comparing all three categories at key timesteps."""
    import matplotlib.pyplot as plt
    import cv2

    model = ModelWithFeatures(seq_len=8)
    timesteps = [0, 3, 7]  # t=1, t=4, t=8

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    categories = [
        ('Connected (correct)', correct_conn[0] if correct_conn else None),
        ('Disconnected (correct)', correct_disc[0] if correct_disc else None),
        ('Incorrect', incorrect[0] if incorrect else None),
    ]

    for row, (name, idx) in enumerate(categories):
        if idx is None:
            continue

        image = val_images[idx]
        label = val_labels[idx]
        img_display = (image - image.min()) / (image.max() - image.min() + 1e-8)

        x = jnp.array(image)[None]
        logits, features = model.apply({'params': params}, x, training=False)
        pred = int(logits.argmax())

        perpixel_logits = apply_perpixel_readout(features, params)
        evidence = np.array(perpixel_logits[0, :, :, :, 1] - perpixel_logits[0, :, :, :, 0])

        vmax = max(abs(evidence.min()), abs(evidence.max()))

        # Show input
        axes[row, 0].imshow(img_display)
        color = 'green' if pred == label else 'red'
        lbl_str = "Conn" if label == 1 else "Disc"
        pred_str = "Conn" if pred == 1 else "Disc"
        axes[row, 0].set_title(f"{name}\nTrue: {lbl_str}, Pred: {pred_str}", color=color, fontsize=9)
        axes[row, 0].axis('off')

        # Show evidence at key timesteps
        for col, t in enumerate(timesteps):
            ev_up = cv2.resize(evidence[t], (224, 224), interpolation=cv2.INTER_LINEAR)
            axes[row, col+1].imshow(img_display, alpha=0.3)
            im = axes[row, col+1].imshow(ev_up, cmap='RdBu_r', alpha=0.7, vmin=-vmax, vmax=vmax)
            axes[row, col+1].set_title(f't={t+1}: [{evidence[t].min():.1f}, {evidence[t].max():.1f}]', fontsize=9)
            axes[row, col+1].axis('off')

    plt.suptitle("Per-pixel evidence: CSSM -> norm_pre -> act -> norm_post -> head\n(Red = votes Connected, Blue = votes Disconnected)", fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/final_static_comparison.png', dpi=150)
    plt.close()
    print("Saved: visualizations/final_static_comparison.png")


if __name__ == '__main__':
    main()
