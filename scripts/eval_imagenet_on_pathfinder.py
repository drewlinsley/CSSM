#!/usr/bin/env python3
"""
Evaluate an ImageNet-trained CSSM-SHViT on Pathfinder via linear probe.

1. Load CSSM-SHViT checkpoint (frozen backbone)
2. Extract features from Pathfinder images (resize to 224x224)
3. Train a linear classifier on the features
4. Report accuracy

Usage:
    source activate.sh && python scripts/eval_imagenet_on_pathfinder.py \
        --checkpoint /oscar/scratch/dlinsley/imagenet_checkpoints/cssm_shvit_s1_gdn_dk2_k1x1/epoch_290_ema \
        --model_type cssm_shvit \
        --cssm_type gdn \
        --tfrecord_dir /oscar/scratch/dlinsley/pathfinder_tfrecords_128 \
        --difficulty 14
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from sklearn.metrics import accuracy_score
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def create_model(args):
    """Create the CSSM-SHViT model matching the checkpoint.

    NOTE: The v1 checkpoint was trained BEFORE pos_conv was added to CSSMSHViTBlock.
    We need to match the exact architecture the checkpoint was trained with.
    Set --v1 flag to use the old architecture (no pos_conv).
    """
    if args.model_type == 'cssm_shvit':
        from src.models.cssm_shvit import cssm_shvit_s1
        model = cssm_shvit_s1(
            num_classes=1000,
            num_timesteps=args.num_timesteps,
            cssm_type=args.cssm_type,
            delta_key_dim=args.delta_key_dim,
            kernel_sizes=(args.kernel_size,) * 4,
            output_norm='rms',
            gate_type='factored',
            short_conv_spatial_size=args.short_conv_spatial_size,
            short_conv_size=args.short_conv_size,
            block_norm='global_layer',
            rope_mode='spatiotemporal',
            use_pos_conv=not args.v1,  # v1 checkpoint has no pos_conv
            head_pool=args.head_pool,
        )
    elif args.model_type == 'shvit':
        from src.models.shvit import shvit_s1
        model = shvit_s1(num_classes=1000)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    return model


def load_checkpoint(model, checkpoint_path, image_size=224):
    """Load checkpoint and return (params, batch_stats)."""
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, image_size, image_size, 3))
    variables = model.init({'params': rng, 'dropout': rng}, dummy, training=True)

    checkpointer = ocp.StandardCheckpointer()

    # Restore without target (unsafe but works for mismatched trees)
    restored = checkpointer.restore(checkpoint_path)

    # Unwrap EMA if present
    if 'ema_params' in restored:
        params = restored['ema_params']
    elif 'params' in restored:
        params = restored['params']
    else:
        params = restored

    batch_stats = variables.get('batch_stats', None)

    return params, batch_stats


def extract_features(model, params, batch_stats, images, image_size=224):
    """Extract penultimate (B, C) features — after LayerNorm + GAP, before classification head.

    Runs full forward pass to get logits, then recovers the 384-d features
    by subtracting the head bias and projecting back through the head weight.
    """
    if images.shape[1] != image_size:
        images = jax.image.resize(images, (images.shape[0], image_size, image_size, 3), method='bilinear')

    variables = {'params': params}
    if batch_stats is not None:
        variables['batch_stats'] = batch_stats

    # Run full forward, then strip the head: logits = features @ W + b
    logits = model.apply(variables, images, training=False)  # (B, num_classes)
    head_w = params['head']['kernel']  # (C, num_classes)
    head_b = params['head']['bias']    # (num_classes,)
    # Least-squares solve: features @ W = logits - b  =>  features = (logits - b) @ W.T @ inv(W @ W.T)
    # But since C < num_classes (384 < 1000), W.T @ W is (C,C) and invertible:
    #   features = (logits - b) @ W.T @ inv(W @ W.T)  -- but simpler: W is full row rank
    # Actually just use: features = (logits - b) @ pinv(W.T)
    # Simpler: since the head is linear, features = (logits - b) @ W.T @ inv(W @ W.T)
    # Even simpler: W is (C, K) with C=384, K=1000. W^T is (K, C).
    # (logits - b) is (B, K). We want (B, C).
    # features = (logits - b) @ W^T @ (W @ W^T)^{-1} ... no.
    # Just use lstsq: features such that features @ W ≈ logits - b
    residual = logits - head_b  # (B, K)
    features = jnp.linalg.lstsq(head_w.T, residual.T)[0].T  # (B, C)
    return features


def load_pathfinder_data(tfrecord_dir, difficulty, split, max_samples=None):
    """Load Pathfinder data."""
    from src.pathfinder_data import get_pathfinder_tfrecord_loader

    loader = get_pathfinder_tfrecord_loader(
        tfrecord_dir=tfrecord_dir,
        difficulty=difficulty,
        split=split,
        batch_size=64,
        image_size=128,
        num_frames=1,
    )

    all_images = []
    all_labels = []
    for batch in tqdm(loader, desc=f"Loading {split}"):
        images, labels = batch
        if images.ndim == 5:
            images = images[:, 0]  # take first frame
        all_images.append(np.array(images))
        all_labels.append(np.array(labels))
        if max_samples and sum(len(l) for l in all_labels) >= max_samples:
            break

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if max_samples:
        images = images[:max_samples]
        labels = labels[:max_samples]
    return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='cssm_shvit', choices=['cssm_shvit', 'shvit'])
    parser.add_argument('--cssm_type', type=str, default='gdn')
    parser.add_argument('--delta_key_dim', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=1)
    parser.add_argument('--num_timesteps', type=int, default=1)
    parser.add_argument('--short_conv_spatial_size', type=int, default=3)
    parser.add_argument('--short_conv_size', type=int, default=4)
    parser.add_argument('--v1', action='store_true', help='Use v1 architecture (no pos_conv)')
    parser.add_argument('--head_pool', type=str, default='mean', choices=['max', 'mean'],
                        help='Head pooling for CSSM-SHViT (SFA/LeJEPA runs used "mean").')
    parser.add_argument('--tfrecord_dir', type=str, default='/oscar/scratch/dlinsley/pathfinder_tfrecords_128')
    parser.add_argument('--difficulty', type=str, default='14')
    parser.add_argument('--max_train', type=int, default=10000)
    parser.add_argument('--max_test', type=int, default=5000)
    args = parser.parse_args()

    print("=" * 60)
    print("ImageNet -> Pathfinder Linear Probe")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_type} ({args.cssm_type})")
    print(f"Pathfinder difficulty: {args.difficulty}")

    # Create model
    print("\nCreating model...")
    model = create_model(args)

    # Load checkpoint
    print("Loading checkpoint...")
    params, batch_stats = load_checkpoint(model, args.checkpoint)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"  Loaded {n_params:,} params")

    # Load Pathfinder data
    print(f"\nLoading Pathfinder CL-{args.difficulty}...")
    train_images, train_labels = load_pathfinder_data(
        args.tfrecord_dir, args.difficulty, 'train', args.max_train)
    test_images, test_labels = load_pathfinder_data(
        args.tfrecord_dir, args.difficulty, 'test', args.max_test)
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")

    # Extract features
    print("\nExtracting features...")
    batch_size = 32

    def extract_batched(images):
        all_feats = []
        for i in tqdm(range(0, len(images), batch_size), desc="  Features"):
            batch = jnp.array(images[i:i+batch_size])
            feats = extract_features(model, params, batch_stats, batch)
            all_feats.append(np.array(feats))
        return np.concatenate(all_feats, axis=0)

    train_feats = extract_batched(train_images)
    test_feats = extract_batched(test_images)
    print(f"  Train features: {train_feats.shape}")
    print(f"  Test features: {test_feats.shape}")

    # XGBoost classifier
    print("\nTraining XGBoost...")
    clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                        use_label_encoder=False, eval_metric='logloss', verbosity=1)
    clf.fit(train_feats, train_labels)

    train_acc = accuracy_score(train_labels, clf.predict(train_feats))
    test_acc = accuracy_score(test_labels, clf.predict(test_feats))

    print(f"\n{'=' * 60}")
    print(f"Results: Pathfinder CL-{args.difficulty}")
    print(f"  Train accuracy: {train_acc*100:.1f}%")
    print(f"  Test accuracy:  {test_acc*100:.1f}%")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
