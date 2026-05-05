#!/usr/bin/env python3
"""
Visualize how channel activations and input saliency evolve over CSSM recurrence steps.

Two outputs:
  1. channel_evolution.png — Top discriminative channels (2,5,38,45,7) spatial maps at t=1..8
     for connected vs disconnected examples.
  2. saliency_evolution.png — Gradient of "virtual logit at timestep t" w.r.t. input,
     showing where the model attends at each recurrence step.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/visualize_temporal_evolution.py
"""

import os
import sys
import pickle
from pathlib import Path

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_cssm import SimpleCSSM
from src.pathfinder_data import IMAGENET_MEAN, IMAGENET_STD

# ---- Config ----
CKPT = 'checkpoints/channel_c32_ST_longer/epoch_85'
IMAGE_SIZE = 128
SEQ_LEN = 8
FEATURE_SIZE = 64
EMBED_DIM = 64
TOP_CHANNELS = [2, 5, 38, 45, 7]
OUT_DIR = 'visualizations/contour_probe'
NUM_PER_CLASS = 5  # examples per class


def load_checkpoint(checkpoint_path):
    with open(os.path.join(checkpoint_path, 'checkpoint.pkl'), 'rb') as f:
        ckpt = pickle.load(f)
    print(f"Loaded epoch {ckpt['epoch']}, step {ckpt['step']}")
    return ckpt['params'], ckpt.get('batch_stats', {})


def infer_model_config(params):
    embed_dim = params['conv1']['kernel'].shape[-1]
    kernel_size = params['cssm_0']['kernel'].shape[1]
    b_q_kernel = params['cssm_0']['B_Q']['kernel']
    gate_type = 'channel' if b_q_kernel.ndim == 4 else 'dense'
    has_k = 'd_K' in params['cssm_0']
    single_state = 'd_Q' not in params['cssm_0']
    if single_state:
        cssm_type = 'add_kqv_1'
    elif not has_k:
        cssm_type = 'add_kqv_2'
    else:
        cssm_type = 'add_kqv'
    stem_layers = sum(1 for k in params.keys() if k.startswith('conv'))
    return dict(embed_dim=embed_dim, kernel_size=kernel_size, gate_type=gate_type,
                cssm_type=cssm_type, stem_layers=stem_layers, norm_type='batch')


def build_model(config):
    return SimpleCSSM(
        num_classes=2, embed_dim=config['embed_dim'], depth=1,
        cssm_type=config['cssm_type'], kernel_size=config['kernel_size'],
        gate_type=config['gate_type'], stem_layers=config['stem_layers'],
        norm_type='batch', pos_embed='spatiotemporal',
        stem_norm_order='post', pool_type='max',
        seq_len=SEQ_LEN, use_complex32=True)


def load_png_samples(num_per_class):
    """Load balanced connected + disconnected samples from PNG directory."""
    img_dir = '/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025/curv_contour_length_14/imgs'

    pos, neg = [], []
    np.random.seed(42)

    for label, subdir in [(1, 'pos'), (0, 'neg')]:
        d = os.path.join(img_dir, subdir)
        files = sorted([f for f in os.listdir(d) if f.endswith('.png')])
        np.random.shuffle(files)
        count = 0
        for f in files:
            img_300 = np.array(Image.open(os.path.join(d, f)).convert('L'))
            img_pil = Image.fromarray(img_300, mode='L')
            img_128 = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            img_np = np.array(img_128, dtype=np.float32) / 255.0
            img_rgb = np.stack([img_np] * 3, axis=-1)
            img_normalized = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
            if label == 1:
                pos.append(img_normalized)
            else:
                neg.append(img_normalized)
            count += 1
            if count >= num_per_class:
                break

    print(f"Loaded {len(pos)} connected + {len(neg)} disconnected")
    return pos, neg


def denormalize(img):
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading checkpoint...")
    params, batch_stats = load_checkpoint(CKPT)
    config = infer_model_config(params)
    print(f"Config: {config}")
    model = build_model(config)
    variables = {'params': params, 'batch_stats': batch_stats}

    # BN params for manual readout preprocessing
    bn_mean = np.array(batch_stats['norm_pre']['mean'])
    bn_var = np.array(batch_stats['norm_pre']['var'])
    bn_scale = np.array(params['norm_pre']['scale'])
    bn_bias = np.array(params['norm_pre']['bias'])

    def apply_bn_softplus(feat):
        normed = (feat - bn_mean) / np.sqrt(bn_var + 1e-5) * bn_scale + bn_bias
        return np.log1p(np.exp(normed))  # softplus

    # Head weights for reference
    head_w = np.array(params['head']['kernel'])  # (64, 2)
    head_diff = head_w[:, 1] - head_w[:, 0]  # positive = evidence for connected

    # Load samples
    print("\nLoading samples...")
    pos_imgs, neg_imgs = load_png_samples(NUM_PER_CLASS)

    # Extract features for all samples
    print("\nExtracting features...")
    all_data = []
    for cls_name, imgs in [('Connected', pos_imgs), ('Disconnected', neg_imgs)]:
        for i, img_normalized in enumerate(imgs):
            img_video = np.stack([img_normalized] * SEQ_LEN, axis=0)
            x_5d = jnp.array(img_video)[None, ...]

            # Get features at all timesteps
            feat = model.apply(variables, x_5d, training=False, return_features=True)
            feat = np.array(feat[0])  # (T, 64, 64, 64)

            # Get logits
            logits = model.apply(variables, x_5d, training=False)
            logits = np.array(logits[0])  # (2,)
            pred = int(np.argmax(logits))

            # Apply BN + softplus to match readout path
            feat_processed = apply_bn_softplus(feat)  # (T, 64, 64, 64)

            display = denormalize(img_normalized)
            label = 1 if cls_name == 'Connected' else 0

            all_data.append({
                'class': cls_name,
                'label': label,
                'pred': pred,
                'display': display,
                'feat': feat_processed,
                'logits': logits,
                'img_normalized': img_normalized,
            })
            print(f"  {cls_name} {i}: pred={'conn' if pred==1 else 'disc'} "
                  f"logits=[{logits[0]:.2f}, {logits[1]:.2f}]")

    # =========================================================================
    # Figure 1: Channel activation evolution over timesteps
    # =========================================================================
    print("\n=== Generating channel evolution figure ===")
    n_examples = len(all_data)
    n_channels = len(TOP_CHANNELS)

    fig, axes = plt.subplots(n_examples, SEQ_LEN + 1, figsize=(2.2 * (SEQ_LEN + 1), 2.5 * n_examples))

    for row, d in enumerate(all_data):
        # First column: input image
        axes[row, 0].imshow(d['display'])
        status = 'OK' if d['pred'] == d['label'] else 'WRONG'
        color = 'green' if d['pred'] == d['label'] else 'red'
        axes[row, 0].set_title(f"{d['class']}\n({status})", fontsize=9, color=color)
        axes[row, 0].axis('off')

        # For each timestep, show top channel as composite
        for t in range(SEQ_LEN):
            feat_t = d['feat'][t]  # (64, 64, 64)

            # Create RGB composite: R=ch2, G=ch5, B=ch38
            ch_vals = [feat_t[:, :, c] for c in TOP_CHANNELS[:3]]
            composite = np.stack(ch_vals, axis=-1)
            # Normalize each channel to [0, 1] with consistent scale
            for c in range(3):
                vmax = max(np.percentile(composite[:, :, c], 99), 1e-8)
                composite[:, :, c] = np.clip(composite[:, :, c] / vmax, 0, 1)

            axes[row, t + 1].imshow(composite)
            if row == 0:
                axes[row, t + 1].set_title(f't={t+1}', fontsize=10)
            axes[row, t + 1].axis('off')

    fig.suptitle(f'Channel evolution: R=ch{TOP_CHANNELS[0]}, G=ch{TOP_CHANNELS[1]}, B=ch{TOP_CHANNELS[2]} '
                 f'(after BN+softplus)', fontsize=12, y=1.01)
    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, 'channel_evolution_rgb.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path1}")

    # ---- Per-channel evolution (separate rows per channel) ----
    print("\n=== Generating per-channel evolution figure ===")
    # Pick 1 connected + 1 disconnected, show all 5 channels x 8 timesteps
    examples = [all_data[0], all_data[NUM_PER_CLASS]]  # first conn, first disc
    n_ch = len(TOP_CHANNELS)

    fig, axes = plt.subplots(n_ch * len(examples), SEQ_LEN + 1,
                             figsize=(2.0 * (SEQ_LEN + 1), 1.8 * n_ch * len(examples)))

    for ex_idx, d in enumerate(examples):
        for ch_idx, ch in enumerate(TOP_CHANNELS):
            row = ex_idx * n_ch + ch_idx

            # Compute global vmax for this channel across all timesteps
            ch_all_t = d['feat'][:, :, :, ch]  # (T, 64, 64)
            vmax = np.percentile(ch_all_t, 99.5)
            vmin = np.percentile(ch_all_t, 0.5)

            # First col: input
            axes[row, 0].imshow(d['display'])
            hw = head_diff[ch]
            axes[row, 0].set_ylabel(f"ch{ch}\nhw={hw:+.2f}", fontsize=8, rotation=0,
                                     labelpad=50, va='center')
            if ch_idx == 0:
                axes[row, 0].set_title(f"{d['class']}", fontsize=10,
                                        color='green' if d['pred'] == d['label'] else 'red')
            axes[row, 0].axis('off')

            # Each timestep
            for t in range(SEQ_LEN):
                act_map = d['feat'][t, :, :, ch]
                im = axes[row, t + 1].imshow(act_map, cmap='inferno', vmin=vmin, vmax=vmax)
                # Mark max-pool location
                max_idx = np.unravel_index(np.argmax(act_map), act_map.shape)
                axes[row, t + 1].plot(max_idx[1], max_idx[0], '*', color='cyan',
                                       markersize=6, markeredgecolor='white', markeredgewidth=0.5)
                if ex_idx == 0 and ch_idx == 0:
                    axes[row, t + 1].set_title(f't={t+1}', fontsize=9)
                axes[row, t + 1].axis('off')

                # Show max value on last timestep
                if t == SEQ_LEN - 1:
                    axes[row, t + 1].text(1.02, 0.5, f'{act_map.max():.1f}',
                                           transform=axes[row, t + 1].transAxes,
                                           fontsize=7, va='center')

    plt.suptitle('Per-channel activation over recurrence (cyan star = max location)', fontsize=11, y=1.0)
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, 'channel_evolution_detail.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path2}")

    # ---- Max-pool value over time (line plots) ----
    print("\n=== Generating max-pool temporal curves ===")
    fig, axes = plt.subplots(1, n_ch, figsize=(3.5 * n_ch, 3.5))
    ts = np.arange(1, SEQ_LEN + 1)

    for ch_idx, ch in enumerate(TOP_CHANNELS):
        ax = axes[ch_idx]
        for d in all_data:
            ch_all_t = d['feat'][:, :, :, ch]  # (T, 64, 64)
            maxpool_t = ch_all_t.max(axis=(1, 2))  # (T,)
            color = 'steelblue' if d['label'] == 1 else 'coral'
            ls = '-' if d['pred'] == d['label'] else '--'
            ax.plot(ts, maxpool_t, ls, color=color, alpha=0.7, linewidth=1.5)

        ax.set_xlabel('Recurrence step t')
        ax.set_ylabel('Max activation')
        hw = head_diff[ch]
        ax.set_title(f'Ch {ch} (hw={hw:+.2f})', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', linewidth=2, label='Connected'),
        Line2D([0], [0], color='coral', linewidth=2, label='Disconnected'),
    ]
    axes[-1].legend(handles=legend_elements, fontsize=8, loc='upper left')

    plt.suptitle('Max-pooled activation over recurrence steps', fontsize=12)
    plt.tight_layout()
    path3 = os.path.join(OUT_DIR, 'maxpool_temporal_curves.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path3}")

    # =========================================================================
    # Figure 2: Grad-CAM + raw saliency per timestep
    # =========================================================================
    print("\n=== Computing Grad-CAM and saliency per timestep ===")

    # Grad-CAM at timestep t:
    #   1. Forward: get post-CSSM features A at timestep t: (H', W', C)
    #   2. Compute logit_diff = logit_conn - logit_disc using features at t
    #   3. Gradient of logit_diff w.r.t. A: (H', W', C)
    #   4. Global-average-pool gradients -> per-channel weights alpha_c
    #   5. CAM = ReLU(sum_c alpha_c * A_c) -> (H', W')
    #   6. Upsample to input resolution

    def compute_gradcam_and_saliency(model, variables, x_5d, t_idx):
        """Compute Grad-CAM heatmap and input saliency at timestep t."""

        # --- Grad-CAM: gradient w.r.t. post-CSSM features ---
        def feat_to_logit_diff(x_5d):
            """Returns (logit_conn - logit_disc) using features at timestep t_idx."""
            spatial_logits = model.apply(variables, x_5d, training=False, return_spatial=True)
            logits_t = spatial_logits[0, t_idx]  # (H, W, 2)
            pooled = logits_t.max(axis=(0, 1))  # (2,)
            return pooled[1] - pooled[0]

        def feat_fn(x_5d):
            """Return post-CSSM features (before BN+act)."""
            return model.apply(variables, x_5d, training=False, return_features=True)

        # Get features and their gradient w.r.t. the logit diff
        feat_all = feat_fn(x_5d)  # (1, T, H', W', C)
        feat_t = np.array(feat_all[0, t_idx])  # (H', W', C) numpy

        # Compute gradient of logit_diff w.r.t. features
        # We need: d(logit_diff)/d(feat) where feat = return_features output
        # Use a custom function that takes features as input
        def logit_from_feat_at_t(feat_val):
            """Given features (1, T, H', W', C), apply BN+softplus+head at timestep t."""
            f_t = feat_val[0, t_idx]  # (H', W', C)
            # Apply BN
            normed = (f_t - jnp.array(bn_mean)) / jnp.sqrt(jnp.array(bn_var) + 1e-5) * jnp.array(bn_scale) + jnp.array(bn_bias)
            activated = jax.nn.softplus(normed)  # (H', W', C)
            # Apply norm_post + head
            # norm_post params
            np_mean = jnp.array(batch_stats['norm_post']['mean'])
            np_var = jnp.array(batch_stats['norm_post']['var'])
            np_scale = jnp.array(params['norm_post']['scale'])
            np_bias = jnp.array(params['norm_post']['bias'])
            # Per-pixel logits: BN over channel dim, then dense
            normed2 = (activated - np_mean) / jnp.sqrt(np_var + 1e-5) * np_scale + np_bias
            hw = jnp.array(params['head']['kernel'])  # (C, 2)
            hb = jnp.array(params['head']['bias'])    # (2,)
            pixel_logits = normed2 @ hw + hb  # (H', W', 2)
            pooled = pixel_logits.max(axis=(0, 1))  # (2,)
            return pooled[1] - pooled[0]

        grad_feat = jax.grad(logit_from_feat_at_t)(feat_all)  # (1, T, H', W', C)
        grad_feat_t = np.array(grad_feat[0, t_idx])  # (H', W', C)

        # Grad-CAM: GAP the gradients -> channel weights, then weighted sum
        alpha = grad_feat_t.mean(axis=(0, 1))  # (C,)
        cam = np.maximum((feat_t * alpha[None, None, :]).sum(axis=-1), 0)  # (H', W') ReLU
        # Normalize to [0, 1]
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Upsample to input resolution
        cam_up = np.array(Image.fromarray(cam.astype(np.float32), mode='F').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))

        # --- Input saliency ---
        grad_input_fn = jax.grad(feat_to_logit_diff)
        grads_input = grad_input_fn(x_5d)
        sal_map = np.abs(np.array(grads_input[0, 0])).sum(axis=-1)  # (H, W)

        return cam_up, sal_map

    saliency_examples = all_data  # Grad-CAM for all examples

    all_gradcams = []
    all_saliencies = []
    for d in saliency_examples:
        img_video = np.stack([d['img_normalized']] * SEQ_LEN, axis=0)
        x_5d = jnp.array(img_video)[None, ...]

        gradcam_per_t = []
        saliency_per_t = []
        for t in range(SEQ_LEN):
            cam_up, sal_map = compute_gradcam_and_saliency(model, variables, x_5d, t)
            gradcam_per_t.append(cam_up)
            saliency_per_t.append(sal_map)
            if t == 0:
                print(f"  {d['class']}: compiled")

        all_gradcams.append(np.stack(gradcam_per_t))    # (T, H, W)
        all_saliencies.append(np.stack(saliency_per_t))  # (T, H, W)
        print(f"  {d['class']}: done")

    # Plot Grad-CAM + saliency evolution (2 rows per example: gradcam, raw saliency)
    n_sal_ex = len(saliency_examples)
    fig, axes = plt.subplots(n_sal_ex * 2, SEQ_LEN + 1,
                             figsize=(2.2 * (SEQ_LEN + 1), 2.5 * n_sal_ex * 2))

    for ex_idx, (d, gcam, sal) in enumerate(zip(saliency_examples, all_gradcams, all_saliencies)):
        vmax_sal = np.percentile(sal, 99.5)
        row_gc = ex_idx * 2
        row_sal = ex_idx * 2 + 1

        status = 'OK' if d['pred'] == d['label'] else 'WRONG'
        color = 'green' if d['pred'] == d['label'] else 'red'

        # Row 1: Grad-CAM
        axes[row_gc, 0].imshow(d['display'])
        axes[row_gc, 0].set_title(f"{d['class']}", fontsize=9, color=color)
        axes[row_gc, 0].set_ylabel('Grad-CAM', fontsize=8)
        axes[row_gc, 0].axis('off')

        for t in range(SEQ_LEN):
            axes[row_gc, t + 1].imshow(d['display'], alpha=0.4)
            cmap_jet = plt.colormaps.get_cmap('jet')
            axes[row_gc, t + 1].imshow(cmap_jet(gcam[t])[:, :, :3], alpha=0.6)
            if ex_idx == 0:
                axes[row_gc, t + 1].set_title(f't={t+1}', fontsize=10)
            axes[row_gc, t + 1].axis('off')

        # Row 2: Raw saliency
        axes[row_sal, 0].imshow(d['display'])
        axes[row_sal, 0].set_ylabel('Input grad', fontsize=8)
        axes[row_sal, 0].axis('off')

        for t in range(SEQ_LEN):
            axes[row_sal, t + 1].imshow(d['display'], alpha=0.3)
            cmap_hot = plt.colormaps.get_cmap('hot')
            sal_norm = np.clip(sal[t] / (vmax_sal + 1e-8), 0, 1)
            axes[row_sal, t + 1].imshow(cmap_hot(sal_norm)[:, :, :3], alpha=0.8)
            axes[row_sal, t + 1].axis('off')

    fig.suptitle('Per-timestep: Grad-CAM vs raw input gradient', fontsize=11, y=1.01)
    plt.tight_layout()
    path4 = os.path.join(OUT_DIR, 'gradcam_vs_saliency.png')
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path4}")

    # ---- Combined Grad-CAM + channel figure ----
    print("\n=== Generating combined evolution figure ===")

    # For each example: row 1 = Grad-CAM, row 2 = channel 2, row 3 = channel 5
    n_rows_per_ex = 3
    n_ex = len(saliency_examples)

    fig, axes = plt.subplots(n_rows_per_ex * n_ex, SEQ_LEN + 1,
                             figsize=(2.0 * (SEQ_LEN + 1), 1.8 * n_rows_per_ex * n_ex))

    for ex_idx, (d, gcam) in enumerate(zip(saliency_examples, all_gradcams)):
        base_row = ex_idx * n_rows_per_ex

        # Row 0: Grad-CAM
        axes[base_row, 0].imshow(d['display'])
        status = 'OK' if d['pred'] == d['label'] else 'WRONG'
        color = 'green' if d['pred'] == d['label'] else 'red'
        axes[base_row, 0].set_title(f"{d['class']}", fontsize=9, color=color)
        axes[base_row, 0].set_ylabel('Grad-CAM', fontsize=8)
        axes[base_row, 0].axis('off')

        for t in range(SEQ_LEN):
            axes[base_row, t + 1].imshow(d['display'], alpha=0.4)
            cmap_jet = plt.colormaps.get_cmap('jet')
            axes[base_row, t + 1].imshow(cmap_jet(gcam[t])[:, :, :3], alpha=0.6)
            if ex_idx == 0:
                axes[base_row, t + 1].set_title(f't={t+1}', fontsize=9)
            axes[base_row, t + 1].axis('off')

        # Rows 1-2: Channel 2 and Channel 5
        for ch_row, ch in enumerate([2, 5]):
            row = base_row + 1 + ch_row
            ch_all_t = d['feat'][:, :, :, ch]
            vmax_ch = np.percentile(ch_all_t, 99.5)
            vmin_ch = np.percentile(ch_all_t, 0.5)

            axes[row, 0].imshow(d['display'])
            hw = head_diff[ch]
            axes[row, 0].set_ylabel(f'Ch {ch}\nhw={hw:+.2f}', fontsize=8, rotation=0,
                                     labelpad=45, va='center')
            axes[row, 0].axis('off')

            for t in range(SEQ_LEN):
                act_map = d['feat'][t, :, :, ch]
                axes[row, t + 1].imshow(act_map, cmap='inferno', vmin=vmin_ch, vmax=vmax_ch)
                max_idx = np.unravel_index(np.argmax(act_map), act_map.shape)
                axes[row, t + 1].plot(max_idx[1], max_idx[0], '*', color='cyan',
                                       markersize=5, markeredgecolor='white', markeredgewidth=0.3)
                axes[row, t + 1].axis('off')

    plt.suptitle('Temporal evolution: Grad-CAM + top channel activations',
                 fontsize=10, y=1.0)
    plt.tight_layout()
    path5 = os.path.join(OUT_DIR, 'temporal_evolution_combined.png')
    plt.savefig(path5, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path5}")

    # =========================================================================
    # GIFs: animated versions for each example
    # =========================================================================
    print("\n=== Generating GIFs ===")
    FRAME_DURATION = 500  # ms per frame

    for ex_idx, d in enumerate(all_data):
        cls_tag = 'conn' if d['label'] == 1 else 'disc'
        status_tag = 'ok' if d['pred'] == d['label'] else 'wrong'

        # --- GIF: channel activations ---
        frames = []
        for t in range(SEQ_LEN):
            fig, axes_gif = plt.subplots(1, len(TOP_CHANNELS) + 1, figsize=(2.5 * (len(TOP_CHANNELS) + 1), 2.8))

            # Input
            axes_gif[0].imshow(d['display'])
            axes_gif[0].set_title('Input', fontsize=10)
            axes_gif[0].axis('off')

            for ch_idx, ch in enumerate(TOP_CHANNELS):
                act_map = d['feat'][t, :, :, ch]
                ch_all_t = d['feat'][:, :, :, ch]
                vmax_ch = np.percentile(ch_all_t, 99.5)
                vmin_ch = np.percentile(ch_all_t, 0.5)

                axes_gif[ch_idx + 1].imshow(act_map, cmap='inferno', vmin=vmin_ch, vmax=vmax_ch)
                max_idx = np.unravel_index(np.argmax(act_map), act_map.shape)
                axes_gif[ch_idx + 1].plot(max_idx[1], max_idx[0], '*', color='cyan',
                                           markersize=8, markeredgecolor='white', markeredgewidth=0.5)
                hw = head_diff[ch]
                axes_gif[ch_idx + 1].set_title(f'Ch {ch} (max={act_map.max():.1f})', fontsize=9)
                axes_gif[ch_idx + 1].axis('off')

            status_str = 'OK' if d['pred'] == d['label'] else 'WRONG'
            color = 'green' if d['pred'] == d['label'] else 'red'
            fig.suptitle(f"{d['class']} ({status_str})  —  t={t+1}/{SEQ_LEN}",
                         fontsize=12, fontweight='bold', color=color)
            plt.tight_layout()

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(buf)
            plt.close(fig)

        gif_path = os.path.join(OUT_DIR, f'channels_{cls_tag}_{ex_idx}_{status_tag}.gif')
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           duration=FRAME_DURATION, loop=0)
        print(f"  {gif_path}")

    # --- GIFs: Grad-CAM + channels evolution ---
    for ex_idx, (d, gcam) in enumerate(zip(saliency_examples, all_gradcams)):
        cls_tag = 'conn' if d['label'] == 1 else 'disc'
        status_tag = 'ok' if d['pred'] == d['label'] else 'wrong'

        frames = []
        for t in range(SEQ_LEN):
            fig, axes_gif = plt.subplots(1, 4, figsize=(12, 3.2))

            # Input
            axes_gif[0].imshow(d['display'])
            axes_gif[0].set_title('Input', fontsize=10)
            axes_gif[0].axis('off')

            # Grad-CAM overlay
            axes_gif[1].imshow(d['display'], alpha=0.4)
            cmap_jet = plt.colormaps.get_cmap('jet')
            axes_gif[1].imshow(cmap_jet(gcam[t])[:, :, :3], alpha=0.6)
            axes_gif[1].set_title('Grad-CAM', fontsize=10)
            axes_gif[1].axis('off')

            # Channel 2
            act2 = d['feat'][t, :, :, 2]
            ch2_all = d['feat'][:, :, :, 2]
            axes_gif[2].imshow(act2, cmap='inferno',
                               vmin=np.percentile(ch2_all, 0.5), vmax=np.percentile(ch2_all, 99.5))
            mi = np.unravel_index(np.argmax(act2), act2.shape)
            axes_gif[2].plot(mi[1], mi[0], '*', color='cyan', markersize=10,
                             markeredgecolor='white', markeredgewidth=0.5)
            axes_gif[2].set_title(f'Ch 2 (max={act2.max():.1f})', fontsize=10)
            axes_gif[2].axis('off')

            # Channel 5
            act5 = d['feat'][t, :, :, 5]
            ch5_all = d['feat'][:, :, :, 5]
            axes_gif[3].imshow(act5, cmap='inferno',
                               vmin=np.percentile(ch5_all, 0.5), vmax=np.percentile(ch5_all, 99.5))
            mi = np.unravel_index(np.argmax(act5), act5.shape)
            axes_gif[3].plot(mi[1], mi[0], '*', color='cyan', markersize=10,
                             markeredgecolor='white', markeredgewidth=0.5)
            axes_gif[3].set_title(f'Ch 5 (max={act5.max():.1f})', fontsize=10)
            axes_gif[3].axis('off')

            status_str = 'OK' if d['pred'] == d['label'] else 'WRONG'
            color = 'green' if d['pred'] == d['label'] else 'red'
            fig.suptitle(f"{d['class']} ({status_str})  —  t={t+1}/{SEQ_LEN}",
                         fontsize=12, fontweight='bold', color=color)
            plt.tight_layout()

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(buf)
            plt.close(fig)

        gif_path = os.path.join(OUT_DIR, f'gradcam_{cls_tag}_{ex_idx}_{status_tag}.gif')
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           duration=FRAME_DURATION, loop=0)
        print(f"  {gif_path}")

    print(f"\nDone. All outputs in {OUT_DIR}/")


if __name__ == '__main__':
    main()
