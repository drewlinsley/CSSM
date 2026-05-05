#!/usr/bin/env python3
"""
Visualize per-pixel contribution to the final decision.

Instead of applying head per-pixel, we look at how much each pixel
contributes to the final (correct) class logit after pooling.
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

        # Standard readout
        x_last = x[:, -1]
        x_normed = nn.LayerNorm(name='norm_pre')(x_last)
        x_act = act(x_normed)
        x_pooled = x_act.mean(axis=(1, 2))
        x_final = nn.LayerNorm(name='norm_post')(x_pooled)
        logits = nn.Dense(self.num_classes, name='head')(x_final)

        return logits, features, x_act  # Return pre-pool activated features


def compute_pixel_contribution(x_act, params, target_class):
    """
    Compute how much each pixel contributes to the target class logit.

    After norm_pre and act, the pipeline is:
    1. pool: mean over spatial dims
    2. norm_post: normalize the pooled vector
    3. head: linear projection

    We want to see each pixel's contribution to the final class logit.
    Since pool is mean, each pixel contributes 1/(H*W) to the mean.
    """
    B, H, W, C = x_act.shape

    # For each pixel, compute what its contribution would be if it were the only pixel
    # This shows the "direction" each pixel is pushing the decision

    head_kernel = params['head']['kernel']  # (C, 2)
    head_bias = params['head']['bias']      # (2,)
    norm_post_scale = params['norm_post']['scale']
    norm_post_bias = params['norm_post']['bias']

    # Project each pixel through a simplified head (ignoring norm_post for interpretability)
    # pixel_logits[h,w] = x_act[h,w] @ head_kernel + head_bias
    pixel_logits = jnp.einsum('bhwc,cn->bhwn', x_act, head_kernel) + head_bias

    # Contribution to target class vs other class
    contribution = pixel_logits[..., target_class] - pixel_logits[..., 1 - target_class]

    return contribution  # (B, H, W)


def create_contribution_gif(image, label, params, seq_len, output_path):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import cv2

    model = ModelWithFeatures(seq_len=seq_len)
    x = jnp.array(image)[None]
    logits, features, _ = model.apply({'params': params}, x, training=False)

    final_pred = int(np.array(logits.argmax()))
    features = np.array(features[0])  # (T, H', W', C)

    # For each timestep, compute normalized features and contribution
    act = nn.softplus
    norm_pre_scale = params['norm_pre']['scale']
    norm_pre_bias = params['norm_pre']['bias']

    contributions = []
    for t in range(seq_len):
        feat_t = features[t:t+1]  # (1, H', W', C)

        # Apply norm_pre
        feat_t = jnp.array(feat_t)
        mean = feat_t.mean(axis=-1, keepdims=True)
        var = feat_t.var(axis=-1, keepdims=True)
        feat_normed = (feat_t - mean) / jnp.sqrt(var + 1e-6) * norm_pre_scale + norm_pre_bias

        # Apply activation
        feat_act = act(feat_normed)

        # Compute contribution to predicted class
        contrib = compute_pixel_contribution(feat_act, params, final_pred)
        contributions.append(np.array(contrib[0]))  # (H', W')

    contributions = np.array(contributions)  # (T, H', W')

    # Global prediction confidence over time
    global_contrib = contributions.mean(axis=(1, 2))

    img_display = (image - image.min()) / (image.max() - image.min() + 1e-8)
    H_orig, W_orig = img_display.shape[:2]

    vmax = np.abs(contributions).max()
    if vmax < 0.1:
        vmax = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    def update(t):
        for ax in axes:
            ax.clear()

        # Input
        axes[0].imshow(img_display)
        true_str = "Connected" if label == 1 else "Disconnected"
        pred_str = "Connected" if final_pred == 1 else "Disconnected"
        color = 'green' if final_pred == label else 'red'
        axes[0].set_title(f"Input\nTrue: {true_str}\nPred: {pred_str}", color=color, fontsize=10)
        axes[0].axis('off')

        # Contribution overlay
        contrib_up = cv2.resize(contributions[t], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        axes[1].imshow(img_display, alpha=0.4)
        axes[1].imshow(contrib_up, cmap='RdBu_r', alpha=0.8, vmin=-vmax, vmax=vmax)
        axes[1].set_title(f't={t+1}/{seq_len}: Contribution to "{pred_str}"\n(red=supports, blue=opposes)', fontsize=9)
        axes[1].axis('off')

        # Heatmap
        axes[2].imshow(contrib_up, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2].set_title(f't={t+1}: Contribution heatmap', fontsize=10)
        axes[2].axis('off')

        # Timeline
        times = np.arange(1, seq_len + 1)
        axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        axes[3].plot(times[:t+1], global_contrib[:t+1], 'ko-', markersize=6, linewidth=2)
        axes[3].fill_between(times[:t+1], 0, global_contrib[:t+1],
                            where=global_contrib[:t+1] >= 0, alpha=0.3, color='red')
        axes[3].fill_between(times[:t+1], 0, global_contrib[:t+1],
                            where=global_contrib[:t+1] < 0, alpha=0.3, color='blue')
        axes[3].axvline(x=t+1, color='orange', alpha=0.5, linewidth=3)
        axes[3].set_xlim(0.5, seq_len + 0.5)
        yabs = max(abs(global_contrib.min()), abs(global_contrib.max()), 0.1) * 1.3
        axes[3].set_ylim(-yabs, yabs)
        axes[3].set_xlabel('Timestep')
        axes[3].set_ylabel('Mean Contribution')
        axes[3].set_title(f'Support for "{pred_str}"', fontsize=10)

        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=600, blit=False)
    ani.save(output_path, writer='pillow', fps=2)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("="*60)
    print("PIXEL CONTRIBUTION VISUALIZATION")
    print("="*60)

    with open('checkpoints/KQV_sinusoidal/epoch_15/checkpoint.pkl', 'rb') as f:
        params = pickle.load(f)['params']

    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    val_files = sorted(tf.io.gfile.glob('/home/dlinsley/pathfinder_tfrecord/difficulty_14/val/*.tfrecord'))
    val_images, val_labels = load_tfrecord_batch(val_files, max_examples=200)

    model = ModelWithFeatures(seq_len=8)

    correct_conn, correct_disc, incorrect = [], [], []
    for i in tqdm(range(min(100, len(val_images))), desc="Finding examples"):
        x = jnp.array(val_images[i])[None]
        logits, _, _ = model.apply({'params': params}, x, training=False)
        pred = int(logits.argmax())
        lbl = int(val_labels[i])
        if pred == lbl:
            (correct_conn if lbl == 1 else correct_disc).append(i)
        else:
            incorrect.append(i)

    print(f"Found: {len(correct_conn)} conn, {len(correct_disc)} disc, {len(incorrect)} wrong")

    os.makedirs('visualizations', exist_ok=True)

    for name, idx_list in [('conn', correct_conn), ('disc', correct_disc), ('wrong', incorrect)]:
        if idx_list:
            idx = idx_list[0]
            print(f"Creating {name} (idx={idx})...")
            create_contribution_gif(val_images[idx], val_labels[idx], params, 8,
                                   f'visualizations/contrib_{name}_T8.gif')

    print("Done!")


if __name__ == '__main__':
    main()
