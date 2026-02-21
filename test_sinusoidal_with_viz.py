#!/usr/bin/env python3
"""
Test the model trained with pos_embed='sinusoidal' at different timesteps.
Also create visualization GIFs showing activity propagation over time.

This model uses spatial-only RoPE + sinusoidal temporal embeddings.
Sinusoidal embeddings naturally extrapolate to longer sequences.
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
from src.models.cssm import (
    TransformerCSSM, apply_rope, apply_sinusoidal_temporal_encoding
)


def load_tfrecord_batch(tfrecord_files, max_examples=None):
    """Load images and labels from TFRecord files."""
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

    images = []
    labels = []
    for img, lbl in tqdm(ds, desc="Loading"):
        images.append(img.numpy())
        labels.append(lbl.numpy())
        if max_examples and len(images) >= max_examples:
            break

    return np.array(images), np.array(labels)


class SimpleCSSMWithIntermediates(nn.Module):
    """
    SimpleCSSM that returns intermediate states at each timestep.
    Used for visualization of activity propagation.
    """
    num_classes: int = 2
    embed_dim: int = 64
    depth: int = 1
    kernel_size: int = 15
    seq_len: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        """
        Forward pass that returns both logits and per-timestep features.

        Returns:
            logits: (B, num_classes)
            timestep_logits: (B, T, num_classes) - logits at each timestep
            timestep_features: (B, T, H', W', embed_dim) - CSSM output at each timestep
        """
        act = nn.softplus

        # Handle single image input
        if x.ndim == 4:
            x = jnp.repeat(x[:, None, ...], self.seq_len, axis=1)

        B, T, H, W, C = x.shape

        # === STEM ===
        x = x.reshape(B * T, H, W, C)

        x = nn.Conv(self.embed_dim, kernel_size=(3, 3), padding='SAME', name='conv1')(x)
        x = act(x)
        x = nn.LayerNorm(name='norm1')(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        x = nn.Conv(self.embed_dim, kernel_size=(3, 3), padding='SAME', name='conv2')(x)
        x = act(x)
        x = nn.LayerNorm(name='norm2')(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        _, H_new, W_new, _ = x.shape
        x = x.reshape(B, T, H_new, W_new, self.embed_dim)

        # === POSITION EMBEDDINGS (sinusoidal) ===
        x = apply_rope(x, mode='spatial_only')
        x = apply_sinusoidal_temporal_encoding(x)

        # === CSSM BLOCK ===
        cssm = TransformerCSSM(
            channels=self.embed_dim,
            kernel_size=self.kernel_size,
            block_size=1,
            rope_mode='none',
            name='cssm_0'
        )
        cssm_out = cssm(x)
        x = x + cssm_out  # Residual

        # Store features at each timestep
        timestep_features = x  # (B, T, H', W', embed_dim)

        # === READOUT for each timestep ===
        # Apply the same readout path to each timestep
        def readout_single_frame(frame):
            """Apply readout to a single frame (H', W', embed_dim)."""
            f = nn.LayerNorm(name='norm_pre')(frame)
            f = act(f)
            f = f.mean(axis=(0, 1))  # Spatial pool -> (embed_dim,)
            f = nn.LayerNorm(name='norm_post')(f)
            f = nn.Dense(self.num_classes, name='head')(f)
            return f

        # Apply readout to each timestep
        timestep_logits = jax.vmap(
            lambda t: jax.vmap(readout_single_frame)(x[:, t])
        )(jnp.arange(T))  # (T, B, num_classes)
        timestep_logits = jnp.transpose(timestep_logits, (1, 0, 2))  # (B, T, num_classes)

        # Final logits from last frame
        final_logits = timestep_logits[:, -1, :]  # (B, num_classes)

        return final_logits, timestep_logits, timestep_features


def test_accuracy(images, labels, params, seq_len, batch_size=4):
    """Test accuracy with given seq_len."""
    model = SimpleCSSMWithIntermediates(
        num_classes=2,
        embed_dim=64,
        depth=1,
        kernel_size=15,
        seq_len=seq_len,
    )

    correct = 0
    total = 0

    def predict_batch(x):
        x = jnp.repeat(x[:, None, ...], seq_len, axis=1)
        logits, _, _ = model.apply({'params': params}, x, training=False)
        return logits.argmax(axis=-1)

    predict_jit = jax.jit(predict_batch)

    for i in tqdm(range(0, len(images), batch_size), desc=f"Testing seq_len={seq_len}", leave=False):
        batch_imgs = images[i:i+batch_size]
        batch_lbls = labels[i:i+batch_size]

        x = jnp.array(batch_imgs)
        preds = predict_jit(x)
        preds = np.array(preds)

        correct += (preds == batch_lbls).sum()
        total += len(batch_lbls)

    return correct / total


def create_activity_gif(image, label, params, seq_len, output_path, model=None):
    """
    Create a GIF showing per-timestep classification confidence.

    For each timestep, shows:
    - Original image
    - Heatmap of class confidence (softmax of logits applied to spatial features)
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import matplotlib.colors as mcolors

    if model is None:
        model = SimpleCSSMWithIntermediates(
            num_classes=2,
            embed_dim=64,
            depth=1,
            kernel_size=15,
            seq_len=seq_len,
        )

    # Get per-timestep outputs
    x = jnp.array(image)[None]  # (1, H, W, C)
    x = jnp.repeat(x[:, None, ...], seq_len, axis=1)  # (1, T, H, W, C)

    final_logits, timestep_logits, timestep_features = model.apply(
        {'params': params}, x, training=False
    )

    # Convert to numpy
    timestep_logits = np.array(timestep_logits[0])  # (T, num_classes)
    timestep_features = np.array(timestep_features[0])  # (T, H', W', embed_dim)

    # Get class probabilities over time
    probs = jax.nn.softmax(jnp.array(timestep_logits), axis=-1)
    probs = np.array(probs)  # (T, 2)

    # Get predictions at each timestep
    preds = timestep_logits.argmax(axis=-1)  # (T,)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Normalize image for display
    img_display = (image - image.min()) / (image.max() - image.min() + 1e-8)

    def update(t):
        for ax in axes:
            ax.clear()

        # Left: Original image with prediction
        axes[0].imshow(img_display)
        pred_label = "Connected" if preds[t] == 1 else "Disconnected"
        true_label = "Connected" if label == 1 else "Disconnected"
        color = 'green' if preds[t] == label else 'red'
        axes[0].set_title(f"t={t+1}/{seq_len}\nPred: {pred_label}\nTrue: {true_label}", color=color)
        axes[0].axis('off')

        # Middle: Class probabilities over time
        axes[1].bar(['Disconn.', 'Connected'], probs[t], color=['blue', 'orange'])
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel('Probability')
        axes[1].set_title(f'Class Probabilities at t={t+1}')
        # Add line showing history
        axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # Right: Probability over time (line plot)
        times = np.arange(1, seq_len + 1)
        axes[2].plot(times[:t+1], probs[:t+1, 1], 'o-', color='orange', label='P(Connected)')
        axes[2].plot(times[:t+1], probs[:t+1, 0], 'o-', color='blue', label='P(Disconnected)')
        axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_xlim(0.5, seq_len + 0.5)
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel('Timestep')
        axes[2].set_ylabel('Probability')
        axes[2].set_title('Classification Confidence Over Time')
        axes[2].legend(loc='upper right')

        plt.tight_layout()
        return axes

    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=500, blit=False)
    ani.save(output_path, writer='pillow', fps=2)
    plt.close()
    print(f"Saved: {output_path}")


def create_feature_heatmap_gif(image, label, params, seq_len, output_path):
    """
    Create a GIF showing spatial activity propagation.

    Shows the mean activation magnitude across channels at each spatial location,
    for each timestep.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    model = SimpleCSSMWithIntermediates(
        num_classes=2,
        embed_dim=64,
        depth=1,
        kernel_size=15,
        seq_len=seq_len,
    )

    # Get per-timestep outputs
    x = jnp.array(image)[None]
    x = jnp.repeat(x[:, None, ...], seq_len, axis=1)

    final_logits, timestep_logits, timestep_features = model.apply(
        {'params': params}, x, training=False
    )

    # Convert to numpy
    timestep_features = np.array(timestep_features[0])  # (T, H', W', embed_dim)
    timestep_logits = np.array(timestep_logits[0])  # (T, num_classes)

    # Compute mean activation magnitude at each spatial location
    # (T, H', W')
    activity = np.abs(timestep_features).mean(axis=-1)

    # Normalize for visualization
    vmin, vmax = activity.min(), activity.max()

    # Get predictions
    preds = timestep_logits.argmax(axis=-1)
    probs = jax.nn.softmax(jnp.array(timestep_logits), axis=-1)
    probs = np.array(probs)

    # Normalize image
    img_display = (image - image.min()) / (image.max() - image.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    def update(t):
        for ax in axes:
            ax.clear()

        # Left: Original image
        axes[0].imshow(img_display)
        pred_label = "Connected" if preds[t] == 1 else "Disconnected"
        true_label = "Connected" if label == 1 else "Disconnected"
        color = 'green' if preds[t] == label else 'red'
        axes[0].set_title(f"Input Image\nTrue: {true_label}", fontsize=10)
        axes[0].axis('off')

        # Middle: Activity heatmap
        im = axes[1].imshow(activity[t], cmap='hot', vmin=vmin, vmax=vmax)
        axes[1].set_title(f't={t+1}/{seq_len}: Feature Activity\nPred: {pred_label}',
                         color=color, fontsize=10)
        axes[1].axis('off')

        # Right: Probability timeline
        times = np.arange(1, seq_len + 1)
        axes[2].fill_between(times[:t+1], 0, probs[:t+1, 1], alpha=0.3, color='orange')
        axes[2].fill_between(times[:t+1], 0, probs[:t+1, 0], alpha=0.3, color='blue')
        axes[2].plot(times[:t+1], probs[:t+1, 1], 'o-', color='orange', label='P(Conn)')
        axes[2].plot(times[:t+1], probs[:t+1, 0], 'o-', color='blue', label='P(Disc)')
        axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_xlim(0.5, seq_len + 0.5)
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel('Timestep')
        axes[2].set_ylabel('Probability')
        axes[2].set_title('Confidence Over Time')
        if t == 0:
            axes[2].legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        return axes

    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=500, blit=False)
    ani.save(output_path, writer='pillow', fps=2)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("="*60)
    print("TESTING pos_embed='sinusoidal' MODEL")
    print("(Spatial RoPE + Sinusoidal Temporal Embeddings)")
    print("="*60)

    # Load checkpoint
    checkpoint_path = 'checkpoints/KQV_sinusoidal/epoch_15/checkpoint.pkl'
    print(f"\nLoading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']

    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    TFRECORD_DIR = '/home/dlinsley/pathfinder_tfrecord/difficulty_14'

    print("\n=== Loading validation data ===")
    val_files = sorted(tf.io.gfile.glob(f'{TFRECORD_DIR}/val/*.tfrecord'))
    val_images, val_labels = load_tfrecord_batch(val_files, max_examples=2000)
    print(f"Loaded {len(val_images)} validation examples")

    # Test different timesteps
    timesteps_to_test = [4, 8, 12, 16, 24, 32]

    print("\n" + "="*60)
    print("Testing at different sequence lengths")
    print("="*60)

    results = {}

    for seq_len in timesteps_to_test:
        print(f"\n--- seq_len = {seq_len} ---")
        acc = test_accuracy(val_images, val_labels, params, seq_len)
        results[seq_len] = acc
        print(f"Accuracy: {acc*100:.2f}%")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Accuracy vs Sequence Length")
    print("="*60)
    print(f"\n{'seq_len':>10} | {'Accuracy':>12} | {'vs T=8':>10}")
    print("-" * 40)
    baseline = results.get(8, 0.5)
    for seq_len in timesteps_to_test:
        acc = results[seq_len] * 100
        diff = (results[seq_len] - baseline) * 100
        diff_str = f"{diff:+.2f}%" if seq_len != 8 else "baseline"
        print(f"{seq_len:>10} | {acc:>11.2f}% | {diff_str:>10}")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON WITH OTHER POSITION EMBEDDINGS")
    print("="*60)
    print("""
Spatiotemporal RoPE (KQV_64):
  T=4:  54.60%
  T=8:  89.50%
  T=16: 51.20%
  T=24: 50.85%

Separate (Spatial RoPE + Learned Temporal):
  T=4:  54.45%
  T=8:  84.55%
  T=16: 53.80%
  T=24: 51.85%

Sinusoidal (Spatial RoPE + Sinusoidal Temporal):
""")
    for seq_len in [4, 8, 16, 24]:
        if seq_len in results:
            print(f"  T={seq_len}: {results[seq_len]*100:.2f}%")

    # === VISUALIZATION ===
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    os.makedirs('visualizations', exist_ok=True)

    # Find some interesting examples (correct and incorrect predictions)
    model = SimpleCSSMWithIntermediates(
        num_classes=2,
        embed_dim=64,
        depth=1,
        kernel_size=15,
        seq_len=8,
    )

    # Get predictions for first 100 examples
    print("\nFinding interesting examples...")
    n_viz = 100
    correct_connected = []
    correct_disconnected = []
    incorrect = []

    for i in tqdm(range(min(n_viz, len(val_images))), desc="Scanning"):
        x = jnp.array(val_images[i])[None]
        x = jnp.repeat(x[:, None, ...], 8, axis=1)
        logits, _, _ = model.apply({'params': params}, x, training=False)
        pred = int(logits.argmax())
        label = int(val_labels[i])

        if pred == label:
            if label == 1:
                correct_connected.append(i)
            else:
                correct_disconnected.append(i)
        else:
            incorrect.append(i)

    print(f"Found {len(correct_connected)} correct connected, {len(correct_disconnected)} correct disconnected, {len(incorrect)} incorrect")

    # Create GIFs for a few examples
    examples_to_viz = []
    if correct_connected:
        examples_to_viz.append(('correct_connected', correct_connected[0]))
    if correct_disconnected:
        examples_to_viz.append(('correct_disconnected', correct_disconnected[0]))
    if incorrect:
        examples_to_viz.append(('incorrect', incorrect[0]))

    for name, idx in examples_to_viz:
        print(f"\nCreating visualization for {name} example (idx={idx})...")

        # Activity propagation GIF at T=8
        create_activity_gif(
            val_images[idx], val_labels[idx], params, seq_len=8,
            output_path=f'visualizations/sinusoidal_{name}_T8_probs.gif',
            model=model
        )

        # Feature heatmap GIF
        create_feature_heatmap_gif(
            val_images[idx], val_labels[idx], params, seq_len=8,
            output_path=f'visualizations/sinusoidal_{name}_T8_features.gif'
        )

        # Also create longer sequence version (T=16) to see extrapolation
        create_activity_gif(
            val_images[idx], val_labels[idx], params, seq_len=16,
            output_path=f'visualizations/sinusoidal_{name}_T16_probs.gif'
        )

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nVisualization files saved to visualizations/")
    print("- *_probs.gif: Shows classification probability evolution over time")
    print("- *_features.gif: Shows spatial feature activity heatmap over time")


if __name__ == '__main__':
    main()
