#!/usr/bin/env python3
"""
Evaluate length generalization of a trained model.
Tests accuracy at different sequence lengths (T).
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


def evaluate_at_seq_len(model_fn, params, images, labels, seq_len, batch_size=32):
    """Evaluate model accuracy at a specific sequence length."""
    correct = 0
    total = 0

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        x = jnp.array(batch_images)
        logits = model_fn(params, x, seq_len)
        preds = jnp.argmax(logits, axis=-1)

        correct += int((preds == batch_labels).sum())
        total += len(batch_labels)

    return correct / total


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--tfrecord_dir', type=str, default='/oscar/scratch/dlinsley/pathfinder_tfrecords/')
    parser.add_argument('--difficulty', type=int, default=14)
    parser.add_argument('--max_examples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=15)
    parser.add_argument('--position_independent_gates', action='store_true')
    args = parser.parse_args()

    print("="*60)
    print("LENGTH GENERALIZATION EVALUATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Position independent gates: {args.position_independent_gates}")

    # Load checkpoint
    print(f"\nLoading checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']

    # Load validation data
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    val_dir = os.path.join(args.tfrecord_dir, f'difficulty_{args.difficulty}', 'val')
    val_files = sorted(tf.io.gfile.glob(os.path.join(val_dir, '*.tfrecord')))
    print(f"Loading validation data from {val_dir}...")
    val_images, val_labels = load_tfrecord_batch(val_files, max_examples=args.max_examples)
    print(f"Loaded {len(val_images)} examples")

    # Create model function that can handle variable seq_len
    def model_fn(params, x, seq_len):
        model = SimpleCSSM(
            num_classes=2,
            embed_dim=args.embed_dim,
            depth=1,
            cssm_type='transformer',
            kernel_size=args.kernel_size,
            pos_embed='sinusoidal',
            seq_len=seq_len,
            position_independent_gates=args.position_independent_gates,
        )
        return model.apply({'params': params}, x, training=False)

    # Evaluate at different sequence lengths
    seq_lens = [2, 4, 6, 8, 10, 12, 16, 24, 32]

    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"{'T':<6} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-"*40)

    results = {}
    for T in seq_lens:
        try:
            acc = evaluate_at_seq_len(model_fn, params, val_images, val_labels, T, args.batch_size)
            results[T] = acc
            n_correct = int(acc * len(val_images))
            print(f"{T:<6} {acc*100:>6.2f}%      {n_correct}/{len(val_images)}")
        except Exception as e:
            print(f"{T:<6} ERROR: {e}")
            results[T] = None

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    train_T = 8
    if train_T in results and results[train_T] is not None:
        print(f"Training T={train_T}: {results[train_T]*100:.2f}%")

    # Shorter than training
    shorter = [T for T in seq_lens if T < train_T and results.get(T) is not None]
    if shorter:
        avg_shorter = np.mean([results[T] for T in shorter])
        print(f"Shorter (T<{train_T}): {avg_shorter*100:.2f}% avg")

    # Longer than training
    longer = [T for T in seq_lens if T > train_T and results.get(T) is not None]
    if longer:
        avg_longer = np.mean([results[T] for T in longer])
        print(f"Longer (T>{train_T}): {avg_longer*100:.2f}% avg")

    print("\nDone!")


if __name__ == '__main__':
    main()
