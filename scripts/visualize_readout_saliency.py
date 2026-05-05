#!/usr/bin/env python3
"""
Readout-weighted channel saliency across recurrence steps.

Like the channel_evolution_detail visualization but:
  - Channels ranked by |head_diff[c]| (most discriminative first)
  - Each channel's activation scaled by |head_diff[c]|
  - Top-k composite uses |head_diff|-weighted sum
  - Inferno colormap on positive magnitudes (high contrast)

Output:
  visualizations/readout_saliency/
    readout_channels_{conn/disc}_{i}_{ok/wrong}.png   — per-example detail
    readout_channels_{conn/disc}_{i}_{ok/wrong}.gif   — animated
    summary_grid.png                                   — all examples compact

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_readout_saliency.py
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_cssm import SimpleCSSM
from src.pathfinder_data import IMAGENET_MEAN, IMAGENET_STD

# ---- Config ----
CKPT = 'checkpoints/channel_c32_ST_longer/epoch_85'
IMAGE_SIZE = 128
SEQ_LEN = 8
TOP_K = 5
OUT_DIR = 'visualizations/readout_saliency'
NUM_PER_CLASS = 3


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
            img_128 = Image.fromarray(img_300, mode='L').resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
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


def apply_readout_bn(feat, bn_mean, bn_var, bn_scale, bn_bias,
                     np_mean, np_var, np_scale, np_bias):
    """norm_pre BN -> softplus -> norm_post BN. Returns (T, H', W', C), all positive after softplus."""
    normed = (feat - bn_mean) / np.sqrt(bn_var + 1e-5) * bn_scale + bn_bias
    activated = np.log1p(np.exp(normed))  # softplus: always positive
    normed2 = (activated - np_mean) / np.sqrt(np_var + 1e-5) * np_scale + np_bias
    return normed2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading checkpoint...")
    params, batch_stats = load_checkpoint(CKPT)
    config = infer_model_config(params)
    print(f"Config: {config}")
    model = build_model(config)
    variables = {'params': params, 'batch_stats': batch_stats}

    # Extract BN and head params
    bn_mean = np.array(batch_stats['norm_pre']['mean'])
    bn_var = np.array(batch_stats['norm_pre']['var'])
    bn_scale = np.array(params['norm_pre']['scale'])
    bn_bias = np.array(params['norm_pre']['bias'])
    np_mean = np.array(batch_stats['norm_post']['mean'])
    np_var = np.array(batch_stats['norm_post']['var'])
    np_scale = np.array(params['norm_post']['scale'])
    np_bias = np.array(params['norm_post']['bias'])
    head_w = np.array(params['head']['kernel'])  # (C, 2)
    head_diff = head_w[:, 1] - head_w[:, 0]       # positive = evidence for connected

    # Rank channels by |head_diff|
    top_k_idx = np.argsort(np.abs(head_diff))[::-1][:TOP_K]
    print(f"\nTop-{TOP_K} channels by |head_diff|:")
    for rank, ch in enumerate(top_k_idx):
        sign = '+conn' if head_diff[ch] > 0 else '-disc'
        print(f"  #{rank+1}: ch {ch}  hw={head_diff[ch]:+.4f} ({sign})")

    # Load samples
    print("\nLoading samples...")
    pos_imgs, neg_imgs = load_png_samples(NUM_PER_CLASS)

    # Process all examples
    print("\nComputing readout-weighted channel maps...")
    all_data = []

    for cls_name, imgs in [('Connected', pos_imgs), ('Disconnected', neg_imgs)]:
        for i, img_normalized in enumerate(imgs):
            img_video = np.stack([img_normalized] * SEQ_LEN, axis=0)
            x_5d = jnp.array(img_video)[None, ...]

            logits = np.array(model.apply(variables, x_5d, training=False)[0])
            pred = int(np.argmax(logits))
            label = 1 if cls_name == 'Connected' else 0

            # Post-CSSM features -> readout BN path
            feat_all = np.array(model.apply(
                variables, x_5d, training=False, return_features=True))
            feat = feat_all[0]  # (T, H', W', C)
            feat_processed = apply_readout_bn(
                feat, bn_mean, bn_var, bn_scale, bn_bias,
                np_mean, np_var, np_scale, np_bias)

            # Readout-weighted magnitudes: |feat(h,w,c) * head_diff[c]|
            # = feat_processed(h,w,c) * |head_diff[c]|  (feat_processed can be neg after norm_post)
            # Use absolute value for clean inferno visualization
            weighted_abs = np.abs(feat_processed) * np.abs(head_diff)[None, None, None, :]

            # Top-k composite: sum of |head_diff|-weighted |activations| for top-k channels
            composite = weighted_abs[:, :, :, top_k_idx].sum(axis=-1)  # (T, H', W')

            all_data.append({
                'class': cls_name, 'label': label, 'pred': pred,
                'logits': logits,
                'display': denormalize(img_normalized),
                'feat_processed': feat_processed,   # (T, H', W', C) — raw after BN
                'weighted_abs': weighted_abs,        # (T, H', W', C) — |f|*|hw|
                'composite': composite,              # (T, H', W')
            })
            print(f"  {cls_name} {i}: pred={'conn' if pred==1 else 'disc'} "
                  f"logits=[{logits[0]:.2f}, {logits[1]:.2f}]")

    # =========================================================================
    # Per-example figure: top-k readout-weighted channels + composite
    # =========================================================================
    print("\n=== Generating per-example figures ===")
    FRAME_DURATION = 600

    for ex_idx, d in enumerate(all_data):
        cls_tag = 'conn' if d['label'] == 1 else 'disc'
        status = 'ok' if d['pred'] == d['label'] else 'wrong'
        status_label = 'correct' if d['pred'] == d['label'] else 'wrong'

        n_rows = TOP_K + 1  # top-k channels + composite
        fig, axes = plt.subplots(n_rows, SEQ_LEN + 1,
                                 figsize=(2.0 * (SEQ_LEN + 1), 1.8 * n_rows))

        # --- Rows 0..TOP_K-1: individual channels (inferno, magnitude) ---
        for ch_rank in range(TOP_K):
            ch = top_k_idx[ch_rank]
            hw = head_diff[ch]

            # Use |head_diff|-weighted magnitude for this channel
            ch_maps = d['weighted_abs'][:, :, :, ch]  # (T, H', W')

            # First column: input
            axes[ch_rank, 0].imshow(d['display'])
            axes[ch_rank, 0].set_ylabel(f'Ch {ch}\nhw={hw:+.2f}',
                                         fontsize=8, rotation=0,
                                         labelpad=50, va='center')
            if ch_rank == 0:
                color = 'green' if d['pred'] == d['label'] else 'red'
                axes[ch_rank, 0].set_title(f"{d['class']} ({status_label})",
                                            fontsize=10, color=color)
            axes[ch_rank, 0].axis('off')

            for t in range(SEQ_LEN):
                act_map = ch_maps[t]
                # Mean-subtract: shows deviation from spatial average
                act_centered = act_map - act_map.mean()
                vabs = np.percentile(np.abs(act_centered), 99.5)
                if vabs < 1e-8:
                    vabs = 1.0
                axes[ch_rank, t + 1].imshow(act_centered, cmap='inferno',
                                             vmin=-vabs, vmax=vabs)
                # Mark max-pool location
                max_idx = np.unravel_index(np.argmax(act_map), act_map.shape)
                axes[ch_rank, t + 1].plot(max_idx[1], max_idx[0], '*',
                                           color='cyan', markersize=6,
                                           markeredgecolor='white',
                                           markeredgewidth=0.5)
                if ch_rank == 0:
                    axes[ch_rank, t + 1].set_title(f't={t+1}', fontsize=9)
                axes[ch_rank, t + 1].axis('off')

                # Show max value on last timestep
                if t == SEQ_LEN - 1:
                    axes[ch_rank, t + 1].text(
                        1.02, 0.5, f'{act_map.max():.1f}',
                        transform=axes[ch_rank, t + 1].transAxes,
                        fontsize=7, va='center')

        # --- Last row: top-k composite ---
        row_comp = TOP_K
        comp = d['composite']

        axes[row_comp, 0].imshow(d['display'])
        axes[row_comp, 0].set_ylabel(f'Top-{TOP_K}\ncomposite',
                                      fontsize=8, rotation=0,
                                      labelpad=50, va='center')
        axes[row_comp, 0].axis('off')

        for t in range(SEQ_LEN):
            comp_centered = comp[t] - comp[t].mean()
            vabs_c = np.percentile(np.abs(comp_centered), 99.5)
            if vabs_c < 1e-8:
                vabs_c = 1.0
            axes[row_comp, t + 1].imshow(comp_centered, cmap='inferno',
                                          vmin=-vabs_c, vmax=vabs_c)
            max_idx = np.unravel_index(np.argmax(comp[t]), comp[t].shape)
            axes[row_comp, t + 1].plot(max_idx[1], max_idx[0], '*',
                                        color='cyan', markersize=6,
                                        markeredgecolor='white',
                                        markeredgewidth=0.5)
            axes[row_comp, t + 1].axis('off')
            if t == SEQ_LEN - 1:
                axes[row_comp, t + 1].text(
                    1.02, 0.5, f'{comp[t].max():.1f}',
                    transform=axes[row_comp, t + 1].transAxes,
                    fontsize=7, va='center')

        plt.suptitle(
            f'Readout-weighted channels (|activation| * |head_weight|) — '
            f'{d["class"]} ({status_label})',
            fontsize=10, y=1.01)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f'readout_channels_{cls_tag}_{ex_idx}_{status}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {path}")

        # --- Animated GIF ---
        frames = []
        for t in range(SEQ_LEN):
            fig_g, axes_g = plt.subplots(1, TOP_K + 2,
                                          figsize=(2.5 * (TOP_K + 2), 2.8))

            axes_g[0].imshow(d['display'])
            axes_g[0].set_title('Input', fontsize=9)
            axes_g[0].axis('off')

            for ch_rank in range(TOP_K):
                ch = top_k_idx[ch_rank]
                hw = head_diff[ch]
                ch_map = d['weighted_abs'][t, :, :, ch]
                ch_centered = ch_map - ch_map.mean()
                vabs_ch = np.percentile(np.abs(ch_centered), 99.5)
                if vabs_ch < 1e-8:
                    vabs_ch = 1.0

                axes_g[ch_rank + 1].imshow(ch_centered, cmap='inferno',
                                            vmin=-vabs_ch, vmax=vabs_ch)
                max_idx = np.unravel_index(np.argmax(ch_map), ch_map.shape)
                axes_g[ch_rank + 1].plot(max_idx[1], max_idx[0], '*',
                                          color='cyan', markersize=8,
                                          markeredgecolor='white',
                                          markeredgewidth=0.5)
                axes_g[ch_rank + 1].set_title(
                    f'Ch{ch} ({hw:+.2f})\nmax={ch_map.max():.1f}', fontsize=8)
                axes_g[ch_rank + 1].axis('off')

            # Composite
            comp_t = d['composite'][t]
            comp_cg = comp_t - comp_t.mean()
            vabs_cg = np.percentile(np.abs(comp_cg), 99.5)
            if vabs_cg < 1e-8:
                vabs_cg = 1.0
            axes_g[TOP_K + 1].imshow(comp_cg, cmap='inferno',
                                      vmin=-vabs_cg, vmax=vabs_cg)
            max_idx = np.unravel_index(np.argmax(comp_t), comp_t.shape)
            axes_g[TOP_K + 1].plot(max_idx[1], max_idx[0], '*',
                                    color='cyan', markersize=8,
                                    markeredgecolor='white',
                                    markeredgewidth=0.5)
            axes_g[TOP_K + 1].set_title(f'Top-{TOP_K}\nmax={comp_t.max():.1f}',
                                         fontsize=8)
            axes_g[TOP_K + 1].axis('off')

            status_str = 'OK' if d['pred'] == d['label'] else 'WRONG'
            color = 'green' if d['pred'] == d['label'] else 'red'
            fig_g.suptitle(f"{d['class']} ({status_str})  t={t+1}/{SEQ_LEN}",
                           fontsize=11, fontweight='bold', color=color)
            plt.tight_layout()

            fig_g.canvas.draw()
            buf = np.asarray(fig_g.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(buf)
            plt.close(fig_g)

        gif_path = os.path.join(OUT_DIR,
                                f'readout_channels_{cls_tag}_{ex_idx}_{status}.gif')
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           duration=FRAME_DURATION, loop=0)
        print(f"  {gif_path}")

    # =========================================================================
    # Summary grid
    # =========================================================================
    print("\n=== Generating summary grid ===")
    KEY_STEPS = [0, 3, 7]
    SHOW_K = 3
    n_ex = len(all_data)
    n_cols = 1 + (SHOW_K + 1) * len(KEY_STEPS)

    fig, axes = plt.subplots(n_ex, n_cols, figsize=(1.8 * n_cols, 2.0 * n_ex))

    for ex_idx, d in enumerate(all_data):
        axes[ex_idx, 0].imshow(d['display'])
        ok = d['pred'] == d['label']
        color = 'green' if ok else 'red'
        axes[ex_idx, 0].set_ylabel(f"{d['class']}\n({'OK' if ok else 'X'})",
                                    fontsize=7, color=color, rotation=0,
                                    labelpad=35, va='center')
        if ex_idx == 0:
            axes[ex_idx, 0].set_title('Input', fontsize=7)
        axes[ex_idx, 0].axis('off')

        for s_idx, t in enumerate(KEY_STEPS):
            for ch_rank in range(SHOW_K):
                ch = top_k_idx[ch_rank]
                col = 1 + s_idx * (SHOW_K + 1) + ch_rank
                ch_map = d['weighted_abs'][t, :, :, ch]
                ch_c = ch_map - ch_map.mean()
                vabs_s = max(np.percentile(np.abs(ch_c), 99.5), 1e-8)

                axes[ex_idx, col].imshow(ch_c, cmap='inferno',
                                          vmin=-vabs_s, vmax=vabs_s)
                if ex_idx == 0:
                    hw = head_diff[ch]
                    axes[ex_idx, col].set_title(f'Ch{ch} t={t+1}', fontsize=6)
                axes[ex_idx, col].axis('off')

            # Composite column
            col = 1 + s_idx * (SHOW_K + 1) + SHOW_K
            comp_t = d['composite'][t]
            comp_cs = comp_t - comp_t.mean()
            vabs_cs = max(np.percentile(np.abs(comp_cs), 99.5), 1e-8)
            axes[ex_idx, col].imshow(comp_cs, cmap='inferno',
                                      vmin=-vabs_cs, vmax=vabs_cs)
            if ex_idx == 0:
                axes[ex_idx, col].set_title(f'Top{SHOW_K} t={t+1}', fontsize=6)
            axes[ex_idx, col].axis('off')

    plt.suptitle(
        'Readout-weighted channels (|act|*|hw|): top-3 + composite at t=1,4,8',
        fontsize=10, y=1.02)
    plt.tight_layout()
    path_s = os.path.join(OUT_DIR, 'summary_grid.png')
    plt.savefig(path_s, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {path_s}")

    print(f"\nDone. All outputs in {OUT_DIR}/")


if __name__ == '__main__':
    main()
