#!/usr/bin/env python3
"""
Test model accuracy at different numbers of timesteps.

If the model does incremental grouping, more timesteps should help
(especially for longer contours).
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

    # Create a prediction function for this seq_len
    def predict_batch(x):
        # x is (B, H, W, C), need to expand to (B, T, H, W, C)
        x = jnp.repeat(x[:, None, ...], seq_len, axis=1)
        logits = model.apply({'params': params}, x, training=False)
        return logits.argmax(axis=-1)

    predict_jit = jax.jit(predict_batch)

    for i in tqdm(range(0, len(images), batch_size), desc=f"Testing seq_len={seq_len}"):
        batch_imgs = images[i:i+batch_size]
        batch_lbls = labels[i:i+batch_size]

        x = jnp.array(batch_imgs)
        preds = predict_jit(x)
        preds = np.array(preds)

        correct += (preds == batch_lbls).sum()
        total += len(batch_lbls)

    return correct / total


def main():
    print("Loading checkpoint...")
    checkpoint_path = 'checkpoints/KQV_64/epoch_20/checkpoint.pkl'
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']

    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    TFRECORD_DIR = '/home/dlinsley/pathfinder_tfrecord/difficulty_14'

    print("\n=== Loading validation data ===")
    val_files = sorted(tf.io.gfile.glob(f'{TFRECORD_DIR}/val/*.tfrecord'))
    val_images, val_labels = load_tfrecord_batch(val_files, max_examples=5000)
    print(f"Loaded {len(val_images)} validation examples")

    # Test different timesteps
    # With pos_embed='spatiotemporal': 4->85.58%, 8->89.76%, 16->50.48%, 24->49.66%
    # Now test with pos_embed='none' to see if RoPE is the problem
    timesteps_to_test = [4, 8, 16, 24]

    print("\n" + "="*50)
    print("Testing WITHOUT position embeddings (pos_embed='none')")
    print("="*50)
    results = {}

    for seq_len in timesteps_to_test:
        print(f"\n{'='*50}")
        print(f"Testing with seq_len = {seq_len}")
        print(f"{'='*50}")

        # Create model with this seq_len (NO position embeddings)
        model = SimpleCSSM(
            num_classes=2,
            embed_dim=128,
            depth=1,
            cssm_type='transformer',
            kernel_size=15,
            pos_embed='none',  # Disable RoPE to test generalization
            seq_len=seq_len,
        )

        acc = test_accuracy(val_images, val_labels, model, params, seq_len)
        results[seq_len] = acc
        print(f"Accuracy: {acc*100:.2f}%")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY: Accuracy vs Timesteps")
    print(f"{'='*50}")
    for seq_len in timesteps_to_test:
        print(f"  seq_len={seq_len:2d}: {results[seq_len]*100:.2f}%")

    # Check trend
    accs = [results[t] for t in timesteps_to_test]
    if accs[-1] > accs[0]:
        print(f"\nMore timesteps → HIGHER accuracy (+{(accs[-1]-accs[0])*100:.2f}%)")
    else:
        print(f"\nMore timesteps → LOWER accuracy ({(accs[-1]-accs[0])*100:.2f}%)")


if __name__ == '__main__':
    main()
