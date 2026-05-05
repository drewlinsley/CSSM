#!/usr/bin/env python3
"""
Test the model trained with pos_embed='separate' at different timesteps.

This model uses spatial-only RoPE + learned temporal embeddings.
The learned temporal embeddings can interpolate for longer sequences.
"""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp

sys.path.insert(0, '/media/data_cifs/projects/prj_video_imagenet/CSSM')
from src.models.simple_cssm import SimpleCSSM


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


def test_accuracy(images, labels, model, params, seq_len, batch_size=4):
    """Test accuracy with given seq_len."""
    correct = 0
    total = 0

    def predict_batch(x):
        # x is (B, H, W, C), need to expand to (B, T, H, W, C)
        x = jnp.repeat(x[:, None, ...], seq_len, axis=1)
        logits = model.apply({'params': params}, x, training=False)
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


def main():
    print("="*60)
    print("TESTING pos_embed='separate' MODEL")
    print("(Spatial RoPE + Learned Temporal Embeddings)")
    print("="*60)

    # Load checkpoint
    checkpoint_path = 'checkpoints/KQV_separate/epoch_10/checkpoint.pkl'
    print(f"\nLoading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']

    # Check the temporal embedding shape
    if 'temporal_embed' in params:
        t_embed_shape = params['temporal_embed'].shape
        print(f"Temporal embedding shape: {t_embed_shape}")
    else:
        print("Looking for temporal_embed in params...")
        for key in params.keys():
            print(f"  {key}")

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

        # Create model with this seq_len
        # Note: max_seq_len and embed_dim should match training
        model = SimpleCSSM(
            num_classes=2,
            embed_dim=64,  # Matches training (temporal_embed shape is (32, 64))
            depth=1,
            cssm_type='transformer',
            kernel_size=15,
            pos_embed='separate',
            seq_len=seq_len,
            max_seq_len=32,  # Should match training
        )

        acc = test_accuracy(val_images, val_labels, model, params, seq_len)
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

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Check generalization
    t8_acc = results.get(8, 0)
    t4_acc = results.get(4, 0)
    t16_acc = results.get(16, 0)
    t24_acc = results.get(24, 0)

    print(f"\nLength generalization:")
    print(f"  Shorter (T=4):  {t4_acc*100:.2f}% (trained at T=8)")
    print(f"  Trained (T=8):  {t8_acc*100:.2f}%")
    print(f"  Longer (T=16):  {t16_acc*100:.2f}%")
    print(f"  Longer (T=24):  {t24_acc*100:.2f}%")

    if t16_acc > 0.6:
        print("\n✓ Model generalizes to longer sequences!")
    elif t16_acc > 0.55:
        print("\n~ Partial generalization to longer sequences")
    else:
        print("\n✗ Model does not generalize to longer sequences")

    # Compare to spatiotemporal RoPE (from earlier test)
    print("\n" + "="*60)
    print("COMPARISON WITH SPATIOTEMPORAL ROPE (earlier results)")
    print("="*60)
    print("""
Spatiotemporal RoPE (KQV_64 checkpoint):
  T=4:  54.60%
  T=8:  89.50%
  T=16: 51.20%
  T=24: 50.85%

Separate (Spatial RoPE + Learned Temporal):
""")
    for seq_len in [4, 8, 16, 24]:
        if seq_len in results:
            print(f"  T={seq_len}: {results[seq_len]*100:.2f}%")


if __name__ == '__main__':
    main()
