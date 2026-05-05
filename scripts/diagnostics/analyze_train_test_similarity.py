#!/usr/bin/env python3
"""
Test if model performance depends on train/test similarity.

If the model is memorizing or relying on similarity to training examples,
accuracy should be high on val examples "close" to training and low on "far" ones.
If it's actually solving the task, accuracy should be similar across both.
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


def compute_embeddings(images, model, params, batch_size=64):
    """Get stem embeddings for images (before CSSM)."""
    embeddings = []

    for i in tqdm(range(0, len(images), batch_size), desc="Computing embeddings"):
        batch = images[i:i+batch_size]
        x = jnp.array(batch)

        # Just compute a simple embedding: flatten and subsample
        # Using raw pixels downsampled as a proxy for similarity
        x_small = jax.image.resize(x, (len(batch), 56, 56, 3), method='bilinear')
        emb = x_small.reshape(len(batch), -1)
        embeddings.append(np.array(emb))

    return np.concatenate(embeddings, axis=0)


def compute_predictions(images, labels, model, params, batch_size=16):
    """Get model predictions."""
    correct = 0
    total = 0
    predictions = []

    @jax.jit
    def predict_batch(x):
        logits = model.apply({'params': params}, x, training=False)
        return logits.argmax(axis=-1)

    for i in tqdm(range(0, len(images), batch_size), desc="Predicting"):
        batch_imgs = images[i:i+batch_size]
        batch_lbls = labels[i:i+batch_size]

        x = jnp.array(batch_imgs)
        preds = predict_batch(x)
        preds = np.array(preds)

        predictions.extend(preds)
        correct += (preds == batch_lbls).sum()
        total += len(batch_lbls)

    return np.array(predictions), correct / total


def main():
    print("Loading model and checkpoint...")
    checkpoint_path = 'checkpoints/KQV_64/epoch_20/checkpoint.pkl'
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']

    model = SimpleCSSM(
        num_classes=2,
        embed_dim=128,
        depth=1,
        cssm_type='transformer',
        kernel_size=15,
        pos_embed='spatiotemporal',
        seq_len=8,
    )

    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    TFRECORD_DIR = '/home/dlinsley/pathfinder_tfrecord/difficulty_14'

    # Load training and validation data
    print("\n=== Loading training data ===")
    train_files = sorted(tf.io.gfile.glob(f'{TFRECORD_DIR}/train/*.tfrecord'))
    print(f"Found {len(train_files)} training TFRecord files")

    # Sample training data (don't need all of it for similarity computation)
    train_images, train_labels = load_tfrecord_batch(train_files[:10], max_examples=5000)
    print(f"Loaded {len(train_images)} training examples")

    print("\n=== Loading validation data ===")
    val_files = sorted(tf.io.gfile.glob(f'{TFRECORD_DIR}/val/*.tfrecord'))
    print(f"Found {len(val_files)} validation TFRecord files")

    val_images, val_labels = load_tfrecord_batch(val_files, max_examples=5000)
    print(f"Loaded {len(val_images)} validation examples")

    # Compute embeddings (simple pixel-based for speed)
    print("\n=== Computing embeddings ===")
    train_emb = compute_embeddings(train_images, model, params)
    val_emb = compute_embeddings(val_images, model, params)

    print(f"Train embeddings: {train_emb.shape}")
    print(f"Val embeddings: {val_emb.shape}")

    # Normalize embeddings
    train_emb = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-8)
    val_emb = val_emb / (np.linalg.norm(val_emb, axis=1, keepdims=True) + 1e-8)

    # Compute distance from each val example to nearest train example
    print("\n=== Computing nearest neighbor distances ===")

    # Use batched computation to avoid memory issues
    batch_size = 500
    min_distances = []

    for i in tqdm(range(0, len(val_emb), batch_size), desc="Computing distances"):
        val_batch = val_emb[i:i+batch_size]
        # Cosine similarity (higher = more similar)
        similarities = val_batch @ train_emb.T  # (batch, n_train)
        max_sim = similarities.max(axis=1)  # Most similar train example
        # Convert to distance (lower = more similar)
        min_dist = 1 - max_sim
        min_distances.extend(min_dist)

    min_distances = np.array(min_distances)

    print(f"\nDistance statistics:")
    print(f"  Min: {min_distances.min():.4f}")
    print(f"  Max: {min_distances.max():.4f}")
    print(f"  Mean: {min_distances.mean():.4f}")
    print(f"  Median: {np.median(min_distances):.4f}")

    # Split into close/far based on median
    median_dist = np.median(min_distances)
    close_mask = min_distances <= median_dist
    far_mask = min_distances > median_dist

    print(f"\nSplit by median distance ({median_dist:.4f}):")
    print(f"  Close to train: {close_mask.sum()} examples")
    print(f"  Far from train: {far_mask.sum()} examples")

    # Get predictions
    print("\n=== Computing predictions ===")
    predictions, overall_acc = compute_predictions(val_images, val_labels, model, params)

    # Compute accuracy on each split
    close_correct = (predictions[close_mask] == val_labels[close_mask]).sum()
    close_total = close_mask.sum()
    close_acc = close_correct / close_total

    far_correct = (predictions[far_mask] == val_labels[far_mask]).sum()
    far_total = far_mask.sum()
    far_acc = far_correct / far_total

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Overall accuracy: {overall_acc*100:.2f}%")
    print(f"Close to train (n={close_total}): {close_acc*100:.2f}%")
    print(f"Far from train (n={far_total}): {far_acc*100:.2f}%")
    print(f"Difference (close - far): {(close_acc - far_acc)*100:.2f}%")

    # Also split into quartiles
    print(f"\n{'='*50}")
    print(f"QUARTILE ANALYSIS")
    print(f"{'='*50}")

    quartiles = np.percentile(min_distances, [25, 50, 75])

    q1_mask = min_distances <= quartiles[0]
    q2_mask = (min_distances > quartiles[0]) & (min_distances <= quartiles[1])
    q3_mask = (min_distances > quartiles[1]) & (min_distances <= quartiles[2])
    q4_mask = min_distances > quartiles[2]

    for name, mask in [("Q1 (closest)", q1_mask), ("Q2", q2_mask),
                       ("Q3", q3_mask), ("Q4 (farthest)", q4_mask)]:
        if mask.sum() > 0:
            acc = (predictions[mask] == val_labels[mask]).mean()
            print(f"  {name}: {acc*100:.2f}% (n={mask.sum()})")

    # Check if there's class imbalance in the splits
    print(f"\n{'='*50}")
    print(f"CLASS BALANCE CHECK")
    print(f"{'='*50}")
    print(f"Close to train - pos rate: {val_labels[close_mask].mean()*100:.1f}%")
    print(f"Far from train - pos rate: {val_labels[far_mask].mean()*100:.1f}%")

    # Per-class accuracy
    print(f"\n{'='*50}")
    print(f"PER-CLASS ACCURACY")
    print(f"{'='*50}")

    for cls, cls_name in [(0, "Disconnected"), (1, "Connected")]:
        cls_mask = val_labels == cls

        close_cls = close_mask & cls_mask
        far_cls = far_mask & cls_mask

        if close_cls.sum() > 0:
            close_cls_acc = (predictions[close_cls] == val_labels[close_cls]).mean()
        else:
            close_cls_acc = 0

        if far_cls.sum() > 0:
            far_cls_acc = (predictions[far_cls] == val_labels[far_cls]).mean()
        else:
            far_cls_acc = 0

        print(f"{cls_name}:")
        print(f"  Close: {close_cls_acc*100:.2f}% (n={close_cls.sum()})")
        print(f"  Far: {far_cls_acc*100:.2f}% (n={far_cls.sum()})")

    # Save results for visualization
    np.savez('visualizations/train_test_similarity.npz',
             min_distances=min_distances,
             predictions=predictions,
             labels=val_labels,
             close_mask=close_mask,
             far_mask=far_mask)
    print(f"\nSaved results to visualizations/train_test_similarity.npz")


if __name__ == '__main__':
    main()
