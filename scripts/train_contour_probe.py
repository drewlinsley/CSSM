#!/usr/bin/env python3
"""
Train a linear probe decoder on post-CSSM features to reconstruct the input.

Collects features at each timestep from the frozen model, trains a linear
decoder (1x1 conv: C -> 3), and visualizes reconstructions over time.
Shows what information the CSSM state encodes at each recurrence step.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_contour_probe.py \
        --checkpoint checkpoints/gdn_d3_pf_no_spectral/epoch_125 \
        --cssm gdn_d2 --embed_dim 64 --kernel_size 11 --gate_type channel \
        --stem_layers 2 --norm_type temporal_layer --use_complex32 \
        --scan_mode pallas --short_conv_size 0 --no_spectral_l2_norm \
        --qkv_conv_size 1 --pos_embed spatiotemporal --pool_type max \
        --image_size 224 --difficulty 14 --seq_len 8 \
        --num_train 500 --num_vis 10 \
        --output_dir visualizations/contour_probe_recon
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_cssm import SimpleCSSM
from src.pathfinder_data import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img):
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def downsample(img_hw3, target_h, target_w):
    """Downsample (H, W, 3) image to (target_h, target_w, 3)."""
    out = np.zeros((target_h, target_w, 3), dtype=np.float32)
    for c in range(3):
        pil = PILImage.fromarray(img_hw3[:, :, c].astype(np.float32), mode='F')
        pil = pil.resize((target_w, target_h), PILImage.BILINEAR)
        out[:, :, c] = np.array(pil)
    return out


def upsample(img_hw3, target_h, target_w):
    """Upsample (H', W', 3) image to (target_h, target_w, 3)."""
    return downsample(img_hw3, target_h, target_w)


def collect_features(model, variables, samples, batch_size=4):
    """Run frozen model, collect post-CSSM features at each timestep.

    Returns:
        features: (N, T, H', W', C)
        images: (N, H, W, 3) normalized input images
        labels: (N,)
    """
    all_features, all_images, all_labels = [], [], []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        videos = np.stack([s[0] for s in batch])
        labels = np.array([s[1] for s in batch])

        x_5d = jnp.array(videos)
        feat = model.apply(variables, x_5d, training=False, return_features=True)
        all_features.append(np.array(feat))
        all_images.append(videos[:, 0])
        all_labels.append(labels)

        if (i // batch_size) % 20 == 0:
            print(f"  {min(i+batch_size, len(samples))}/{len(samples)}")

    return (np.concatenate(all_features),
            np.concatenate(all_images),
            np.concatenate(all_labels))


def train_linear_probe(features, images):
    """Train linear probe (least squares): features @ W + b ≈ target.

    Trains on features from ALL timesteps. The target is the input image
    downsampled to feature resolution. Same probe applied to any timestep —
    reconstruction quality at each t reveals what information is available.

    Returns:
        W: (C, 3) weights
        b: (3,) bias
    """
    N, T, Hp, Wp, C = features.shape
    _, H, W, _ = images.shape

    # Downsample targets to feature resolution
    targets_ds = np.zeros((N, Hp, Wp, 3), dtype=np.float32)
    for i in range(N):
        targets_ds[i] = downsample(images[i], Hp, Wp)

    # Use only T=8 (last timestep) features for training to keep matrix manageable.
    # The same probe is then applied to all timesteps for visualization.
    X = features[:, -1].reshape(-1, C)  # (N*H'*W', C)
    Y = targets_ds.reshape(-1, 3)       # (N*H'*W', 3)

    # Add bias column
    X_b = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)

    print(f"  Solving least squares: X={X_b.shape}, Y={Y.shape}")
    W_full, _, _, _ = np.linalg.lstsq(X_b, Y, rcond=None)
    print(f"  W_full shape: {W_full.shape}")

    W = W_full[:C]              # (C, 3)
    b = W_full[C:].flatten()   # (3,)
    print(f"  Probe: W={W.shape}, b={b.shape}")

    # Per-timestep MSE (apply probe to each timestep's features)
    print(f"  Per-timestep reconstruction MSE:")
    for t in range(T):
        X_t = features[:, t].reshape(-1, C)
        X_t_b = np.concatenate([X_t, np.ones((X_t.shape[0], 1), dtype=np.float32)], axis=1)
        Y_pred_t = (X_t_b @ W_full).reshape(N, Hp, Wp, 3)
        mse_t = np.mean((Y_pred_t - targets_ds) ** 2)
        print(f"    t={t+1}: {mse_t:.6f}")

    return W, b


def make_reconstruction_gif(image, features_per_t, W, b, label, pred,
                            sample_idx, output_dir, frame_duration=600):
    """GIF showing linear reconstruction from features at each timestep.

    Panels: [input] [reconstruction] [error map] [overlay]
    """
    T, Hp, Wp, C = features_per_t.shape
    H_img, W_img = image.shape[:2]

    correct = pred == label
    label_str = 'connected' if label == 1 else 'disconnected'
    pred_str = 'connected' if pred == 1 else 'disconnected'
    status = 'correct' if correct else 'wrong'

    # Downsampled target for error computation
    image_norm = (image - IMAGENET_MEAN) / IMAGENET_STD
    image_ds = downsample(image_norm, Hp, Wp)

    # Precompute all reconstructions for consistent error scale
    recons = []
    errors = []
    for t in range(T):
        feat_flat = features_per_t[t].reshape(-1, C)
        recon_ds = (feat_flat @ W + b).reshape(Hp, Wp, 3)
        recons.append(recon_ds)
        errors.append(np.sqrt(np.mean((recon_ds - image_ds) ** 2, axis=-1)))

    error_max = max(e.max() for e in errors) * 1.1

    frames = []
    for t in range(T):
        recon_denorm = denormalize(recons[t])
        recon_up = upsample(recon_denorm, H_img, W_img)
        error_up = np.array(PILImage.fromarray(errors[t].astype(np.float32), mode='F')
                            .resize((W_img, H_img), PILImage.BILINEAR))
        mse_t = np.mean(errors[t] ** 2)

        fig = plt.figure(figsize=(16, 4))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.05)

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(image)
        ax0.set_title('input', fontsize=12)
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(recon_up)
        ax1.set_title(f't={t+1}/{T}  (MSE={mse_t:.4f})', fontsize=12)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(error_up, cmap='hot', vmin=0, vmax=error_max)
        ax2.set_title('error', fontsize=12)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[3])
        ax3.imshow(image, alpha=0.4)
        ax3.imshow(recon_up, alpha=0.6)
        ax3.set_title('overlay', fontsize=12)
        ax3.axis('off')

        fig.suptitle(
            f'GT: {label_str} | Pred: {pred_str} ({"OK" if correct else "WRONG"})',
            fontsize=12, fontweight='bold',
            color='green' if correct else 'red')

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    gif_path = os.path.join(output_dir, f'sample_{sample_idx:03d}_{label_str}_{status}.gif')
    pil_frames = [PILImage.fromarray(f) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                        duration=frame_duration, loop=0)
    pil_frames[-1].save(gif_path.replace('.gif', '_final.png'))
    return gif_path


def main():
    from scripts.visualize_saliency_video import (
        make_parser, _args_to_model_kwargs, load_checkpoint,
        build_model, load_pathfinder_samples)

    parser = make_parser()
    parser.add_argument('--num_train', type=int, default=500,
                        help='Number of training samples for probe')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='Number of visualization samples')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    output_dir = args.output_dir or 'visualizations/contour_probe_recon'
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading checkpoint...")
    params, batch_stats, saved_args = load_checkpoint(args.checkpoint)

    if saved_args is not None:
        from scripts.visualize_saliency_video import build_model_config
        model_kwargs = build_model_config(saved_args, args, parser)
    elif args.cssm is not None:
        model_kwargs = _args_to_model_kwargs(vars(args))
    else:
        print("ERROR: Need --cssm for legacy checkpoint")
        return

    print("Building model...")
    model = build_model(model_kwargs)
    variables = {'params': params, 'batch_stats': batch_stats}
    seq_len = model_kwargs.get('seq_len', args.seq_len)

    # Load training data
    print(f"Loading {args.num_train} training samples...")
    # Use a fixed seed different from viz to avoid overlap
    np.random.seed(123)
    train_samples = load_pathfinder_samples(
        args.data_dir, args.difficulty, args.image_size,
        args.num_train, seq_len)
    print(f"  Got {len(train_samples)} samples")

    # Collect features
    print("Collecting features...")
    features, images, labels = collect_features(
        model, variables, train_samples, batch_size=args.batch_size)
    print(f"  Features: {features.shape}")

    # Train probe
    print("Training linear probe...")
    W, b = train_linear_probe(features, images)

    # Save probe
    probe_path = os.path.join(output_dir, 'probe.pkl')
    with open(probe_path, 'wb') as f:
        pickle.dump({'W': W, 'b': b}, f)
    print(f"  Saved to {probe_path}")

    # Visualization on separate samples
    print(f"\nGenerating reconstruction GIFs...")
    np.random.seed(42)
    vis_samples = load_pathfinder_samples(
        args.data_dir, args.difficulty, args.image_size,
        args.num_vis // 2, seq_len)

    correct_count = 0
    for i, (img_video, label) in enumerate(vis_samples):
        x_5d = jnp.array(img_video)[None]
        feat = np.array(model.apply(variables, x_5d, training=False, return_features=True)[0])
        logits = np.array(model.apply(variables, x_5d, training=False)[0])
        pred = int(np.argmax(logits))
        correct_count += int(pred == label)

        gif_path = make_reconstruction_gif(
            denormalize(img_video[0]), feat, W, b, label, pred, i,
            output_dir, frame_duration=args.frame_duration)

        status = "OK" if pred == label else "WRONG"
        lbl = "conn" if label == 1 else "disc"
        prd = "conn" if pred == 1 else "disc"
        print(f"  [{i+1}/{len(vis_samples)}] GT={lbl} Pred={prd} ({status}) -> {gif_path}")

    print(f"\nAccuracy: {correct_count}/{len(vis_samples)}")
    print(f"Output: {output_dir}/")


if __name__ == '__main__':
    main()
