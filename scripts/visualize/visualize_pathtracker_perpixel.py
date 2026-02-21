#!/usr/bin/env python3
"""
Per-pixel evidence visualization for PathTracker models.

Loads a trained SimpleCSSM (AdditiveCSSM + pathtracker stem) checkpoint and
generates animated GIFs showing how per-pixel classification evidence evolves
across video timesteps. Outputs exactly 1 correct-positive and 1 correct-negative GIF.

For each example, produces a 4-panel animation:
  Panel 1: Raw video frame (changes each timestep)
  Panel 2: Per-pixel evidence overlay on video (RdBu_r colormap)
  Panel 3: Evidence heatmap only
  Panel 4: Global evidence timeline (mean evidence vs timestep)

Usage:
    python scripts/visualize/visualize_pathtracker_perpixel.py \
        --checkpoint checkpoints/pt_simple_add_kqv_d12_e32/epoch_65 \
        --pathtracker_dir /media/data_cifs/projects/prj_video_datasets/pathtracker \
        --num_frames 8 --output_dir visualizations/pathtracker
"""

import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp
from flax import linen as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.simple_cssm import SimpleCSSM


def detect_depth(params):
    """Auto-detect number of CSSM blocks from checkpoint param tree."""
    depth = 0
    while f'cssm_{depth}' in params:
        depth += 1
    return depth


class SimpleCSSMWithFeatures(nn.Module):
    """
    SimpleCSSM wrapper that returns both logits and post-CSSM features
    for per-pixel analysis. Auto-detects depth from checkpoint.
    """
    num_classes: int = 2
    embed_dim: int = 32
    depth: int = 1
    kernel_size: int = 11
    seq_len: int = 8
    gate_type: str = 'factored'
    use_complex32: bool = False

    @nn.compact
    def __call__(self, x, training=False):
        from src.models.cssm import AdditiveCSSM

        act = nn.softplus

        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C)

        # Pathtracker stem: 1x1 conv, no spatial downsampling
        x = nn.Conv(self.embed_dim, kernel_size=(1, 1), name='conv1')(x)
        x = act(x)
        x = nn.LayerNorm(name='norm1')(x)

        _, H_new, W_new, _ = x.shape
        x = x.reshape(B, T, H_new, W_new, self.embed_dim)

        # CSSM blocks (spatiotemporal RoPE applied inside each block)
        for i in range(self.depth):
            cssm = AdditiveCSSM(
                channels=self.embed_dim,
                kernel_size=self.kernel_size,
                block_size=1,
                rope_mode='spatiotemporal',
                gate_type=self.gate_type,
                use_complex32=self.use_complex32,
                name=f'cssm_{i}'
            )
            x = x + cssm(x)

        features = x  # (B, T, H, W, embed_dim)

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
    Apply readout pipeline per-pixel, skipping only the spatial pool.
    Pipeline: norm_pre -> softplus -> norm_post -> head
    """
    B, T, H, W, C = features.shape
    act = nn.softplus

    norm_pre_scale = params['norm_pre']['scale']
    norm_pre_bias = params['norm_pre']['bias']
    norm_post_scale = params['norm_post']['scale']
    norm_post_bias = params['norm_post']['bias']
    head_kernel = params['head']['kernel']
    head_bias = params['head']['bias']

    x = features.reshape(-1, C)

    # norm_pre
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-6) * norm_pre_scale + norm_pre_bias

    x = act(x)

    # norm_post
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-6) * norm_post_scale + norm_post_bias

    logits = x @ head_kernel + head_bias

    return logits.reshape(B, T, H, W, -1)


def create_visualization(video, label, params, model, output_path):
    """Create animated GIF showing per-pixel evidence evolution over video."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    seq_len = video.shape[0]
    x = jnp.array(video)[None]
    logits, features = model.apply({'params': params}, x, training=False)

    final_pred = int(np.array(logits.argmax()))

    perpixel_logits = apply_perpixel_readout(features, params)
    perpixel_logits = np.array(perpixel_logits[0])  # (T, H, W, 2)

    evidence = perpixel_logits[..., 1] - perpixel_logits[..., 0]
    global_evidence = evidence.mean(axis=(1, 2))

    # Denormalize video for display
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    video_display = np.array(video) * IMAGENET_STD + IMAGENET_MEAN
    video_display = np.clip(video_display, 0, 1)

    vmax = max(abs(evidence.min()), abs(evidence.max()), 0.01)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    def update(t):
        for ax in axes:
            ax.clear()

        frame = video_display[t]

        # 1. Raw video frame
        axes[0].imshow(frame, interpolation='nearest')
        true_str = "Positive" if label == 1 else "Negative"
        pred_str = "Positive" if final_pred == 1 else "Negative"
        color = 'green' if final_pred == label else 'red'
        axes[0].set_title(f"Input (t={t+1}/{seq_len})\nTrue: {true_str}\nPred: {pred_str}",
                         color=color, fontsize=10)
        axes[0].axis('off')

        # 2. Per-pixel evidence overlay (no resize needed — already 32x32)
        axes[1].imshow(frame, alpha=0.4, interpolation='nearest')
        axes[1].imshow(evidence[t], cmap='RdBu_r', alpha=0.7,
                      vmin=-vmax, vmax=vmax, interpolation='nearest')
        axes[1].set_title(f't={t+1}: Per-pixel evidence\n(red=pos, blue=neg)', fontsize=9)
        axes[1].axis('off')

        # 3. Evidence heatmap only
        axes[2].imshow(evidence[t], cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                      interpolation='nearest')
        axes[2].set_title(f't={t+1}: Evidence heatmap\n[{evidence[t].min():.2f}, {evidence[t].max():.2f}]',
                         fontsize=9)
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
        axes[3].set_title('Global evidence\n(>0 = Positive vote)', fontsize=10)

        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=600, blit=False)
    ani.save(output_path, writer='pillow', fps=2)
    plt.close()
    print(f"  Saved: {output_path}")


def create_gradient_saliency_visualization(video, label, params, model, output_path):
    """Create animated GIF showing input gradient saliency (d logit / d input)."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    seq_len = video.shape[0]
    x = jnp.array(video)[None]  # (1, T, H, W, C)

    # Get prediction
    logits, _ = model.apply({'params': params}, x, training=False)
    final_pred = int(np.array(logits.argmax()))

    # Compute gradient of predicted class logit w.r.t. input
    def logit_fn(x_in):
        logits, _ = model.apply({'params': params}, x_in, training=False)
        return logits[0, final_pred]

    grad = jax.grad(logit_fn)(x)  # (1, T, H, W, C)
    grad = np.array(grad[0])  # (T, H, W, C)

    # Gradient magnitude: L2 norm over channels
    grad_mag = np.sqrt((grad ** 2).sum(axis=-1))  # (T, H, W)
    global_grad = grad_mag.mean(axis=(1, 2))  # (T,)

    # Denormalize video for display
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    video_display = np.array(video) * IMAGENET_STD + IMAGENET_MEAN
    video_display = np.clip(video_display, 0, 1)

    vmax = max(grad_mag.max(), 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    def update(t):
        for ax in axes:
            ax.clear()

        frame = video_display[t]

        # 1. Raw video frame
        axes[0].imshow(frame, interpolation='nearest')
        true_str = "Positive" if label == 1 else "Negative"
        pred_str = "Positive" if final_pred == 1 else "Negative"
        color = 'green' if final_pred == label else 'red'
        axes[0].set_title(f"Input (t={t+1}/{seq_len})\nTrue: {true_str}\nPred: {pred_str}",
                         color=color, fontsize=10)
        axes[0].axis('off')

        # 2. Gradient magnitude overlay on frame
        axes[1].imshow(frame, alpha=0.4, interpolation='nearest')
        axes[1].imshow(grad_mag[t], cmap='hot', alpha=0.7,
                      vmin=0, vmax=vmax, interpolation='nearest')
        axes[1].set_title(f't={t+1}: Gradient saliency\n(bright = high gradient)', fontsize=9)
        axes[1].axis('off')

        # 3. Gradient heatmap only
        axes[2].imshow(grad_mag[t], cmap='hot', vmin=0, vmax=vmax,
                      interpolation='nearest')
        axes[2].set_title(f't={t+1}: |d logit / d input|\n[{grad_mag[t].min():.4f}, {grad_mag[t].max():.4f}]',
                         fontsize=9)
        axes[2].axis('off')

        # 4. Global gradient magnitude timeline
        times = np.arange(1, seq_len + 1)
        axes[3].plot(times[:t+1], global_grad[:t+1], 'ko-', markersize=4, linewidth=1.5)
        axes[3].fill_between(times[:t+1], 0, global_grad[:t+1], alpha=0.3, color='orange')
        axes[3].axvline(x=t+1, color='orange', alpha=0.5, linewidth=3)
        axes[3].set_xlim(0.5, seq_len + 0.5)
        axes[3].set_ylim(0, max(global_grad.max(), 1e-6) * 1.3)
        axes[3].set_xlabel('Timestep')
        axes[3].set_ylabel('Mean |grad|')
        axes[3].set_title('Mean gradient magnitude', fontsize=10)

        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=600, blit=False)
    ani.save(output_path, writer='pillow', fps=2)
    plt.close()
    print(f"  Saved: {output_path}")


def create_vstate_visualization(video, label, params, model, output_path):
    """Create animated GIF showing V-state L2 norm evolution over video.

    Extracts the V component from the Q->K->V associative scan via Flax's
    sow mechanism. V_hat is in frequency domain (B, T, C, H, W_freq);
    we apply IFFT and compute ||V||_2 over channels per pixel.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    seq_len = video.shape[0]
    H, W = video.shape[1], video.shape[2]
    x = jnp.array(video)[None]  # (1, T, H, W, C)

    # Forward pass with mutable intermediates to capture V_hat
    (logits, features), state = model.apply(
        {'params': params}, x, training=False,
        mutable=['intermediates'])
    final_pred = int(np.array(logits.argmax()))

    # Extract V_hat from sowed intermediates: (B, T, C, H, W_freq) complex64
    v_hat = state['intermediates']['cssm_0']['V_hat'][0]

    # IFFT to spatial domain: (B, T, C, H, W)
    v_spatial = jnp.fft.irfft2(v_hat, s=(H, W), axes=(3, 4))
    v_spatial = np.array(v_spatial[0])  # (T, C, H, W)

    # L2 norm over channels: (T, H, W)
    v_norm = np.sqrt((v_spatial ** 2).sum(axis=1))
    global_vnorm = v_norm.mean(axis=(1, 2))  # (T,)

    # Denormalize video for display
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    video_display = np.array(video) * IMAGENET_STD + IMAGENET_MEAN
    video_display = np.clip(video_display, 0, 1)

    vmax = max(v_norm.max(), 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    def update(t):
        for ax in axes:
            ax.clear()

        frame = video_display[t]

        # 1. Raw video frame
        axes[0].imshow(frame, interpolation='nearest')
        true_str = "Positive" if label == 1 else "Negative"
        pred_str = "Positive" if final_pred == 1 else "Negative"
        color = 'green' if final_pred == label else 'red'
        axes[0].set_title(f"Input (t={t+1}/{seq_len})\nTrue: {true_str}\nPred: {pred_str}",
                         color=color, fontsize=10)
        axes[0].axis('off')

        # 2. V-state norm overlay on frame
        axes[1].imshow(frame, alpha=0.4, interpolation='nearest')
        axes[1].imshow(v_norm[t], cmap='magma', alpha=0.7,
                      vmin=0, vmax=vmax, interpolation='nearest')
        axes[1].set_title(f't={t+1}: V-state ||V||_2\n(bright = high activity)', fontsize=9)
        axes[1].axis('off')

        # 3. V-state norm heatmap only
        axes[2].imshow(v_norm[t], cmap='magma', vmin=0, vmax=vmax,
                      interpolation='nearest')
        axes[2].set_title(f't={t+1}: ||V||_2 heatmap\n[{v_norm[t].min():.3f}, {v_norm[t].max():.3f}]',
                         fontsize=9)
        axes[2].axis('off')

        # 4. Global V-state norm timeline
        times = np.arange(1, seq_len + 1)
        axes[3].plot(times[:t+1], global_vnorm[:t+1], 'ko-', markersize=4, linewidth=1.5)
        axes[3].fill_between(times[:t+1], 0, global_vnorm[:t+1], alpha=0.3, color='purple')
        axes[3].axvline(x=t+1, color='purple', alpha=0.5, linewidth=3)
        axes[3].set_xlim(0.5, seq_len + 0.5)
        axes[3].set_ylim(0, max(global_vnorm.max(), 1e-6) * 1.3)
        axes[3].set_xlabel('Timestep')
        axes[3].set_ylabel('Mean ||V||_2')
        axes[3].set_title('Mean V-state norm', fontsize=10)

        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, frames=seq_len, interval=600, blit=False)
    ani.save(output_path, writer='pillow', fps=2)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PathTracker per-pixel evidence visualization')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/pt_simple_add_kqv_d12_e32/epoch_65',
                       help='Path to checkpoint directory containing checkpoint.pkl')
    parser.add_argument('--pathtracker_dir', type=str,
                       default='/media/data_cifs/projects/prj_video_datasets/pathtracker',
                       help='Path to PathTracker data directory')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Number of video frames to subsample')
    parser.add_argument('--embed_dim', type=int, default=32,
                       help='Model embedding dimension')
    parser.add_argument('--kernel_size', type=int, default=11,
                       help='CSSM spatial kernel size')
    parser.add_argument('--gate_type', type=str, default='factored',
                       help='AdditiveCSSM gate type')
    parser.add_argument('--output_dir', type=str, default='visualizations/pathtracker',
                       help='Output directory for GIFs')
    parser.add_argument('--num_examples', type=int, default=100,
                       help='Max examples to scan per class to find correct ones')
    parser.add_argument('--use_complex32', action='store_true',
                       help='Use complex32 (phase-split bf16) scan representation')
    parser.add_argument('--tfrecord_dir', type=str, default=None,
                       help='Path to TFRecord directory (alternative to --pathtracker_dir)')
    args = parser.parse_args()

    print("=" * 60)
    print("PATHTRACKER PER-PIXEL EVIDENCE VISUALIZATION")
    print("Pipeline: CSSM -> norm_pre -> act -> norm_post -> head (per pixel)")
    print("=" * 60)

    # Load checkpoint
    ckpt_path = os.path.join(args.checkpoint, 'checkpoint.pkl')
    print(f"\nLoading checkpoint: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        params = pickle.load(f)['params']

    depth = detect_depth(params)
    print(f"  Detected depth: {depth}")

    # Create model with auto-detected depth
    model = SimpleCSSMWithFeatures(
        embed_dim=args.embed_dim,
        depth=depth,
        kernel_size=args.kernel_size,
        seq_len=args.num_frames,
        gate_type=args.gate_type,
        use_complex32=args.use_complex32,
    )

    # Load data — TFRecord or raw .npy
    if args.tfrecord_dir:
        print(f"\nLoading PathTracker val TFRecords from: {args.tfrecord_dir}")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError:
            pass
        from src.pathtracker_data import get_pathtracker_tfrecord_loader
        loader = get_pathtracker_tfrecord_loader(
            tfrecord_dir=args.tfrecord_dir,
            batch_size=1,
            num_frames=args.num_frames,
            split='val',
            shuffle=False,
        )
        # Scan batches for 1 correct positive + 1 correct negative
        correct_pos = None  # (video, label)
        correct_neg = None
        print(f"\nScanning for 1 correct positive + 1 correct negative...")
        n_scanned = 0
        for videos, labels in tqdm(loader, total=args.num_examples, desc="Scanning"):
            video = np.array(videos[0])   # (T, H, W, C)
            label = int(labels[0])
            logits, _ = model.apply({'params': params}, jnp.array(video)[None], training=False)
            pred = int(logits.argmax())
            if pred == label:
                if label == 1 and correct_pos is None:
                    correct_pos = (video, label)
                    print(f"  Found correct positive at scan #{n_scanned}")
                elif label == 0 and correct_neg is None:
                    correct_neg = (video, label)
                    print(f"  Found correct negative at scan #{n_scanned}")
            if correct_pos is not None and correct_neg is not None:
                break
            n_scanned += 1
            if n_scanned >= args.num_examples:
                break

        if correct_pos is None:
            print("ERROR: No correct positive found. Try increasing --num_examples.")
            return
        if correct_neg is None:
            print("ERROR: No correct negative found. Try increasing --num_examples.")
            return

        os.makedirs(args.output_dir, exist_ok=True)
        for name, (video, label) in [('positive_correct', correct_pos), ('negative_correct', correct_neg)]:
            # Per-pixel evidence
            output_path = os.path.join(args.output_dir, f'pathtracker_{name}_T{args.num_frames}.gif')
            print(f"\nCreating {name} per-pixel evidence (label={label})...")
            create_visualization(video, label, params, model, output_path)
            # Gradient saliency
            grad_path = os.path.join(args.output_dir, f'pathtracker_{name}_gradient_T{args.num_frames}.gif')
            print(f"Creating {name} gradient saliency...")
            create_gradient_saliency_visualization(video, label, params, model, grad_path)
            # V-state L2 norm
            vstate_path = os.path.join(args.output_dir, f'pathtracker_{name}_vstate_T{args.num_frames}.gif')
            print(f"Creating {name} V-state norm...")
            create_vstate_visualization(video, label, params, model, vstate_path)
    else:
        print(f"\nLoading PathTracker data from: {args.pathtracker_dir}")
        from src.pathtracker_data import PathTrackerDataset
        dataset = PathTrackerDataset(
            root=args.pathtracker_dir,
            image_size=32,
            num_frames=args.num_frames,
            total_frames=64,
        )
        print(f"  Total samples: {len(dataset)}")

        # Find class boundary (dataset has all negatives first, then positives)
        n_total = len(dataset)
        neg_count = sum(1 for i in range(min(100, n_total)) if dataset.files[i][1] == 0)
        boundary = n_total // 2 if neg_count == min(100, n_total) else neg_count

        correct_pos_idx = None
        correct_neg_idx = None

        print(f"\nScanning for 1 correct positive + 1 correct negative...")
        for i in tqdm(range(boundary, min(boundary + args.num_examples, n_total)), desc="Pos"):
            video, label = dataset[i]
            logits, _ = model.apply({'params': params}, jnp.array(video)[None], training=False)
            if int(logits.argmax()) == label:
                correct_pos_idx = i
                break

        for i in tqdm(range(min(args.num_examples, boundary)), desc="Neg"):
            video, label = dataset[i]
            logits, _ = model.apply({'params': params}, jnp.array(video)[None], training=False)
            if int(logits.argmax()) == label:
                correct_neg_idx = i
                break

        if correct_pos_idx is None:
            print("ERROR: No correct positive found. Try increasing --num_examples.")
            return
        if correct_neg_idx is None:
            print("ERROR: No correct negative found. Try increasing --num_examples.")
            return

        print(f"  Found correct positive: idx={correct_pos_idx}")
        print(f"  Found correct negative: idx={correct_neg_idx}")

        os.makedirs(args.output_dir, exist_ok=True)
        for name, idx in [('positive_correct', correct_pos_idx), ('negative_correct', correct_neg_idx)]:
            video, label = dataset[idx]
            # Per-pixel evidence
            output_path = os.path.join(args.output_dir, f'pathtracker_{name}_T{args.num_frames}.gif')
            print(f"\nCreating {name} per-pixel evidence (idx={idx}, label={label})...")
            create_visualization(video, label, params, model, output_path)
            # Gradient saliency
            grad_path = os.path.join(args.output_dir, f'pathtracker_{name}_gradient_T{args.num_frames}.gif')
            print(f"Creating {name} gradient saliency...")
            create_gradient_saliency_visualization(video, label, params, model, grad_path)
            # V-state L2 norm
            vstate_path = os.path.join(args.output_dir, f'pathtracker_{name}_vstate_T{args.num_frames}.gif')
            print(f"Creating {name} V-state norm...")
            create_vstate_visualization(video, label, params, model, vstate_path)

    print(f"\nDone! GIFs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
