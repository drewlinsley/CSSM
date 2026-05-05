#!/usr/bin/env python3
"""
Visualize per-pixel decision evidence over recurrence timesteps.

Projects post-CSSM features onto the readout head's decision direction
(W_head[:,connected] - W_head[:,disconnected]) to show WHERE the model
accumulates evidence for connected vs disconnected at each timestep.

Also shows temporal evolution: how evidence changes between timesteps,
revealing the model's contour-tracing strategy.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_decision_evidence.py \
        --checkpoint checkpoints/gdn_d3_pf_no_spectral/epoch_125 \
        --cssm gdn_d2 --embed_dim 64 --kernel_size 11 --gate_type channel \
        --stem_layers 2 --norm_type temporal_layer --use_complex32 \
        --scan_mode pallas --short_conv_size 0 --no_spectral_l2_norm \
        --qkv_conv_size 1 --pos_embed spatiotemporal --pool_type max \
        --image_size 224 --difficulty 14 --seq_len 8 \
        --num_samples 20 \
        --output_dir visualizations/decision_evidence
"""

import argparse
import os
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
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pathfinder_data import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img):
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def make_evidence_gif(image, features, w_decision, b_decision, label, pred,
                      sample_idx, output_dir, frame_duration=600):
    """GIF showing decision evidence projected per-pixel at each timestep.

    evidence(x,y,t) = features(x,y,t) @ w_decision + b_decision

    Positive = votes connected, Negative = votes disconnected.

    Panels:
      [input]  [evidence heatmap]  [evidence change from t-1]  [top-5 channel activity]
    """
    T, Hp, Wp, C = features.shape
    H_img, W_img = image.shape[:2]

    correct = pred == label
    label_str = 'connected' if label == 1 else 'disconnected'
    pred_str = 'connected' if pred == 1 else 'disconnected'
    status = 'correct' if correct else 'wrong'

    # Compute evidence at every timestep: (T, H', W')
    evidence = np.einsum('thwc,c->thw', features, w_decision) + b_decision

    # Global scale for consistent colormap across timesteps
    ev_abs_max = max(np.abs(evidence).max(), 1e-6)

    # Temporal difference
    evidence_diff = np.zeros_like(evidence)
    evidence_diff[0] = evidence[0]
    evidence_diff[1:] = evidence[1:] - evidence[:-1]
    diff_abs_max = max(np.abs(evidence_diff).max(), 1e-6)

    # Per-channel contribution to evidence at each timestep
    # channel_contrib[t, c] = mean over space of |features[t,:,:,c] * w_decision[c]|
    channel_contrib = np.zeros((T, C))
    for t in range(T):
        for c in range(C):
            channel_contrib[t, c] = np.mean(np.abs(features[t, :, :, c] * w_decision[c]))

    # Top channels by final-timestep contribution
    top_k = 8
    top_channels = np.argsort(channel_contrib[-1])[::-1][:top_k]

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(20, 5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.15)

        # Panel 1: Input image
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(image)
        ax0.set_title('input', fontsize=11)
        ax0.axis('off')

        # Panel 2: Decision evidence map
        ax1 = fig.add_subplot(gs[1])
        ev_up = np.array(PILImage.fromarray(evidence[t].astype(np.float32), mode='F')
                         .resize((W_img, H_img), PILImage.BILINEAR))
        norm1 = TwoSlopeNorm(vmin=-ev_abs_max, vcenter=0, vmax=ev_abs_max)
        im1 = ax1.imshow(ev_up, cmap='RdBu_r', norm=norm1)
        # Overlay input contours
        ax1.imshow(image, alpha=0.2)
        mean_ev = np.mean(evidence[t])
        max_ev = np.max(evidence[t])
        min_ev = np.min(evidence[t])
        ax1.set_title(f't={t+1}/{T}  mean={mean_ev:.2f}\nmax={max_ev:.2f} min={min_ev:.2f}',
                       fontsize=10)
        ax1.axis('off')

        # Panel 3: Evidence change from previous timestep
        ax2 = fig.add_subplot(gs[2])
        diff_up = np.array(PILImage.fromarray(evidence_diff[t].astype(np.float32), mode='F')
                           .resize((W_img, H_img), PILImage.BILINEAR))
        norm2 = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0, vmax=diff_abs_max)
        ax2.imshow(diff_up, cmap='RdBu_r', norm=norm2)
        ax2.imshow(image, alpha=0.2)
        ax2.set_title(f'delta (t={t+1} - t={t})', fontsize=10)
        ax2.axis('off')

        # Panel 4: Top channel contributions over time
        ax3 = fig.add_subplot(gs[3])
        x_pos = np.arange(T)
        for rank, ch in enumerate(top_channels):
            sign = '+' if w_decision[ch] > 0 else '-'
            ax3.plot(x_pos, channel_contrib[:, ch],
                     label=f'ch{ch}({sign})', alpha=0.7, linewidth=1.5)
        ax3.axvline(t, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('timestep')
        ax3.set_ylabel('|feat * w|')
        ax3.set_title('top channel contributions', fontsize=10)
        ax3.legend(fontsize=6, ncol=2, loc='upper left')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([str(i+1) for i in x_pos])

        fig.suptitle(
            f'GT: {label_str} | Pred: {pred_str} ({"OK" if correct else "WRONG"})  '
            f'[red=connected, blue=disconnected]',
            fontsize=11, fontweight='bold',
            color='green' if correct else 'red')

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    gif_path = os.path.join(output_dir, f'evidence_{sample_idx:03d}_{label_str}_{status}.gif')
    pil_frames = [PILImage.fromarray(f) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                        duration=frame_duration, loop=0)
    pil_frames[-1].save(gif_path.replace('.gif', '_final.png'))
    return gif_path


def make_summary_figure(all_evidence, all_labels, all_preds, all_images,
                        w_decision, output_dir):
    """Summary: average evidence maps for connected vs disconnected samples.

    Also shows per-timestep mean/max evidence for each class.
    """
    T = all_evidence[0].shape[0]

    conn_evidence = [ev for ev, lab in zip(all_evidence, all_labels) if lab == 1]
    disc_evidence = [ev for ev, lab in zip(all_evidence, all_labels) if lab == 0]

    if not conn_evidence or not disc_evidence:
        print("  Need both classes for summary")
        return

    # Per-timestep statistics
    conn_means = [np.mean([np.mean(e[t]) for e in conn_evidence]) for t in range(T)]
    disc_means = [np.mean([np.mean(e[t]) for e in disc_evidence]) for t in range(T)]
    conn_maxes = [np.mean([np.max(e[t]) for e in conn_evidence]) for t in range(T)]
    disc_maxes = [np.mean([np.max(e[t]) for e in disc_evidence]) for t in range(T)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Mean evidence per timestep by class
    ax = axes[0]
    ts = np.arange(1, T+1)
    ax.plot(ts, conn_means, 'r-o', label='connected (mean)', linewidth=2)
    ax.plot(ts, disc_means, 'b-o', label='disconnected (mean)', linewidth=2)
    ax.plot(ts, conn_maxes, 'r--s', label='connected (max)', linewidth=1, alpha=0.6)
    ax.plot(ts, disc_maxes, 'b--s', label='disconnected (max)', linewidth=1, alpha=0.6)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('timestep')
    ax.set_ylabel('decision evidence (+ = connected)')
    ax.set_title('Evidence over time by class')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Evidence separation (conn - disc) over time
    ax = axes[1]
    sep_mean = [c - d for c, d in zip(conn_means, disc_means)]
    sep_max = [c - d for c, d in zip(conn_maxes, disc_maxes)]
    ax.plot(ts, sep_mean, 'k-o', label='mean(conn) - mean(disc)', linewidth=2)
    ax.plot(ts, sep_max, 'k--s', label='max(conn) - max(disc)', linewidth=1, alpha=0.6)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('timestep')
    ax.set_ylabel('evidence separation')
    ax.set_title('Class separation over time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Distribution of max evidence at T=8
    ax = axes[2]
    conn_max_t8 = [np.max(e[-1]) for e in conn_evidence]
    disc_max_t8 = [np.max(e[-1]) for e in disc_evidence]
    ax.hist(conn_max_t8, bins=20, alpha=0.5, color='red', label='connected')
    ax.hist(disc_max_t8, bins=20, alpha=0.5, color='blue', label='disconnected')
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('max evidence at T=8')
    ax.set_ylabel('count')
    ax.set_title('Max evidence distribution (T=8)')
    ax.legend()

    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'summary_evidence.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Summary: {summary_path}")

    # Print table
    print(f"\n  Per-timestep evidence (mean over samples):")
    print(f"  {'t':>3} | {'conn_mean':>10} | {'disc_mean':>10} | {'separation':>10} | {'conn_max':>10} | {'disc_max':>10}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for t in range(T):
        print(f"  {t+1:>3} | {conn_means[t]:>10.4f} | {disc_means[t]:>10.4f} | {sep_mean[t]:>10.4f} | {conn_maxes[t]:>10.4f} | {disc_maxes[t]:>10.4f}")


def main():
    from scripts.visualize_saliency_video import (
        make_parser, _args_to_model_kwargs, load_checkpoint,
        build_model, load_pathfinder_samples)

    parser = make_parser()
    args = parser.parse_args()

    output_dir = args.output_dir or 'visualizations/decision_evidence'
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

    # Extract head weights for decision projection
    W_head = np.array(params['head']['kernel'])   # (C, 2)
    b_head = np.array(params['head']['bias'])      # (2,)
    # Decision direction: connected(1) - disconnected(0)
    w_decision = W_head[:, 1] - W_head[:, 0]      # (C,)
    b_decision = b_head[1] - b_head[0]             # scalar
    print(f"  Decision direction: norm={np.linalg.norm(w_decision):.3f}, bias={b_decision:.4f}")

    # Load samples
    print(f"Loading {args.num_samples} samples...")
    np.random.seed(42)
    samples = load_pathfinder_samples(
        args.data_dir, args.difficulty, args.image_size,
        args.num_samples, seq_len)
    print(f"  Got {len(samples)} samples")

    # Process each sample
    all_evidence = []
    all_labels = []
    all_preds = []
    all_images = []
    correct_count = 0

    print(f"\nGenerating decision evidence GIFs...")
    for i, (img_video, label) in enumerate(samples):
        x_5d = jnp.array(img_video)[None]

        # Get features and prediction
        feat = np.array(model.apply(variables, x_5d, training=False, return_features=True)[0])
        logits = np.array(model.apply(variables, x_5d, training=False)[0])
        pred = int(np.argmax(logits))
        correct_count += int(pred == label)

        # Compute evidence: (T, H', W')
        evidence = np.einsum('thwc,c->thw', feat, w_decision) + b_decision
        all_evidence.append(evidence)
        all_labels.append(label)
        all_preds.append(pred)
        all_images.append(denormalize(img_video[0]))

        gif_path = make_evidence_gif(
            denormalize(img_video[0]), feat, w_decision, b_decision,
            label, pred, i, output_dir, frame_duration=args.frame_duration)

        status = "OK" if pred == label else "WRONG"
        lbl = "conn" if label == 1 else "disc"
        prd = "conn" if pred == 1 else "disc"
        mean_ev_t8 = np.mean(evidence[-1])
        max_ev_t8 = np.max(evidence[-1])
        print(f"  [{i+1}/{len(samples)}] GT={lbl} Pred={prd} ({status}) "
              f"ev@T8: mean={mean_ev_t8:.3f} max={max_ev_t8:.3f} -> {gif_path}")

    print(f"\nAccuracy: {correct_count}/{len(samples)}")

    # Summary analysis
    print("\nGenerating summary...")
    make_summary_figure(all_evidence, all_labels, all_preds, all_images,
                        w_decision, output_dir)

    print(f"\nOutput: {output_dir}/")


if __name__ == '__main__':
    main()
