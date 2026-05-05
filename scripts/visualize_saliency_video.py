#!/usr/bin/env python3
"""
Gradient saliency videos for SimpleCSSM Pathfinder models.

For each sample, computes d(logit_predicted_class) / d(Image_t) at each
recurrence timestep t, showing how the model's spatial attention evolves
over the CSSM recurrence. Output as animated GIFs.

The input image is static (repeated T times), but the gradient at each t
reveals what the model attends to at that step of the recurrence — does it
find endpoints first? Trace the contour? Build evidence gradually?

Usage:
    # New checkpoints (with saved training args — zero config):
    CUDA_VISIBLE_DEVICES=2 python scripts/visualize_saliency_video.py \
        --checkpoint path/to/epoch_125 \
        --tfrecord_dir /path/to/pathfinder_tfrecords_128

    # Legacy checkpoints (without saved args — specify model config):
    CUDA_VISIBLE_DEVICES=2 python scripts/visualize_saliency_video.py \
        --checkpoint path/to/epoch_125 \
        --cssm gdn_d2 --embed_dim 64 --kernel_size 11 --gate_type channel \
        --tfrecord_dir /path/to/pathfinder_tfrecords_128
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
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_cssm import SimpleCSSM
from src.pathfinder_data import IMAGENET_MEAN, IMAGENET_STD


# All SimpleCSSM constructor kwargs, mapped from argparse name -> constructor kwarg name.
# Most are 1:1; negated flags and renames are handled in _args_to_model_kwargs().
SIMPLECSSM_KWARGS = [
    'num_classes', 'embed_dim', 'depth', 'cssm_type', 'kernel_size', 'block_size',
    'frame_readout', 'norm_type', 'stem_norm', 'body_norm', 'readout_norm',
    'stem_norm_order', 'pos_embed', 'act_type', 'pool_type', 'seq_len', 'max_seq_len',
    'position_independent_gates', 'use_goom', 'stem_mode', 'stem_layers',
    'shared_kernel', 'additive_kv', 'spectral_clip', 'q_temporal', 'q_spatial',
    'asymmetric_qk', 'no_feedback', 'no_spectral_clip', 'use_layernorm',
    'no_k_state', 'transformer_readout', 'gate_type', 'n_register_tokens',
    'learned_init', 'use_complex32', 'use_complex16', 'use_ssd', 'scan_mode',
    'ssd_chunk_size', 'num_heads', 'mlp_ratio', 'drop_path_rate',
    'short_conv_size', 'output_norm', 'use_input_gates', 'output_gate_act',
    'use_spectral_l2_norm', 'qkv_conv_size', 'qkv_conv_separable',
    'cross_freq_conv_size', 'delta_key_dim', 'no_spectral_clamp',
]


def _args_to_model_kwargs(args_dict):
    """Convert a training args dict (from argparse namespace) to SimpleCSSM kwargs.

    Handles negated flags and arg name differences:
      --cssm -> cssm_type
      --no_goom -> use_goom=False
      --no_input_gates -> use_input_gates=False
      --no_spectral_l2_norm -> use_spectral_l2_norm=False
      --qkv_conv_full -> qkv_conv_separable=False
      --stem_mode -> stem_mode (clamped to 'default'/'pathtracker')
    """
    kwargs = {}

    # Direct 1:1 mappings
    direct = {
        'embed_dim': 'embed_dim',
        'depth': 'depth',
        'kernel_size': 'kernel_size',
        'block_size': 'block_size',
        'frame_readout': 'frame_readout',
        'norm_type': 'norm_type',
        'stem_norm': 'stem_norm',
        'body_norm': 'body_norm',
        'readout_norm': 'readout_norm',
        'stem_norm_order': 'stem_norm_order',
        'pos_embed': 'pos_embed',
        'act_type': 'act_type',
        'pool_type': 'pool_type',
        'seq_len': 'seq_len',
        'max_seq_len': 'max_seq_len',
        'position_independent_gates': 'position_independent_gates',
        'stem_layers': 'stem_layers',
        'shared_kernel': 'shared_kernel',
        'additive_kv': 'additive_kv',
        'spectral_clip': 'spectral_clip',
        'q_temporal': 'q_temporal',
        'q_spatial': 'q_spatial',
        'asymmetric_qk': 'asymmetric_qk',
        'no_feedback': 'no_feedback',
        'no_spectral_clip': 'no_spectral_clip',
        'use_layernorm': 'use_layernorm',
        'no_k_state': 'no_k_state',
        'transformer_readout': 'transformer_readout',
        'gate_type': 'gate_type',
        'n_register_tokens': 'n_register_tokens',
        'learned_init': 'learned_init',
        'use_complex32': 'use_complex32',
        'use_complex16': 'use_complex16',
        'use_ssd': 'use_ssd',
        'scan_mode': 'scan_mode',
        'ssd_chunk_size': 'ssd_chunk_size',
        'num_heads': 'num_heads',
        'mlp_ratio': 'mlp_ratio',
        'drop_path_rate': 'drop_path_rate',
        'short_conv_size': 'short_conv_size',
        'output_norm': 'output_norm',
        'output_gate_act': 'output_gate_act',
        'cross_freq_conv_size': 'cross_freq_conv_size',
        'delta_key_dim': 'delta_key_dim',
        'no_spectral_clamp': 'no_spectral_clamp',
    }

    for arg_name, kwarg_name in direct.items():
        if arg_name in args_dict:
            kwargs[kwarg_name] = args_dict[arg_name]

    # Renamed: --cssm -> cssm_type
    if 'cssm' in args_dict:
        kwargs['cssm_type'] = args_dict['cssm']

    # Negated flags
    if 'no_goom' in args_dict:
        kwargs['use_goom'] = not args_dict['no_goom']
    if 'no_input_gates' in args_dict:
        kwargs['use_input_gates'] = not args_dict['no_input_gates']
    if 'no_spectral_l2_norm' in args_dict:
        kwargs['use_spectral_l2_norm'] = not args_dict['no_spectral_l2_norm']
    if 'qkv_conv_full' in args_dict:
        kwargs['qkv_conv_separable'] = not args_dict['qkv_conv_full']

    # stem_mode: clamp to valid SimpleCSSM values
    if 'stem_mode' in args_dict:
        sm = args_dict['stem_mode']
        kwargs['stem_mode'] = sm if sm in ('default', 'pathtracker') else 'default'

    # Always 2 classes for Pathfinder
    kwargs['num_classes'] = 2

    return kwargs


def load_checkpoint(checkpoint_path):
    """Load pickle checkpoint, return params, batch_stats, and saved training args."""
    with open(os.path.join(checkpoint_path, 'checkpoint.pkl'), 'rb') as f:
        ckpt = pickle.load(f)
    print(f"Loaded epoch {ckpt['epoch']}, step {ckpt['step']}")
    saved_args = ckpt.get('training_args', None)
    if saved_args is not None:
        print(f"  Found saved training args (cssm={saved_args.get('cssm', '?')})")
    else:
        print("  No saved training args (legacy checkpoint)")
    return ckpt['params'], ckpt.get('batch_stats', {}), saved_args


def _legacy_infer_config(params):
    """Infer SimpleCSSM config from checkpoint param shapes (AdditiveCSSM only).

    This is the legacy fallback for old checkpoints that don't have saved
    training args. Only works for AdditiveCSSM variants.
    """
    # embed_dim from conv1 output channels
    embed_dim = params['conv1']['kernel'].shape[-1]

    # kernel_size from cssm kernel
    kernel_size = params['cssm_0']['kernel'].shape[1]

    # Detect gate_type from spatial gate shape
    b_q_kernel = params['cssm_0']['B_Q']['kernel']
    if b_q_kernel.ndim == 4:
        gate_type = 'channel'
    else:
        gate_type = 'dense'

    # Detect n_states from presence of d_K
    has_k = 'd_K' in params['cssm_0']
    single_state = 'd_Q' not in params['cssm_0']

    if single_state:
        cssm_type = 'add_kqv_1'
    elif not has_k:
        cssm_type = 'add_kqv_2'
    else:
        cssm_type = 'add_kqv'

    # stem_layers: count conv layers
    stem_layers = sum(1 for k in params.keys() if k.startswith('conv'))

    norm_type = 'batch'

    config = dict(
        num_classes=2,
        embed_dim=embed_dim,
        kernel_size=kernel_size,
        gate_type=gate_type,
        cssm_type=cssm_type,
        stem_layers=stem_layers,
        norm_type=norm_type,
    )
    print(f"Legacy inferred config: {config}")
    return config


def build_model_config(saved_args, cli_args, parser):
    """Build SimpleCSSM kwargs by merging saved training args with CLI overrides.

    Priority: CLI arg (if explicitly set) > saved training args > CLI defaults.
    """
    defaults = vars(parser.parse_args([]))
    cli_dict = vars(cli_args)

    if saved_args is not None:
        # Start from saved training args
        merged = dict(saved_args)
        # Override with any explicitly-set CLI args
        for key in cli_dict:
            if cli_dict[key] != defaults.get(key):
                merged[key] = cli_dict[key]
        return _args_to_model_kwargs(merged)
    else:
        # No saved args — use CLI args directly
        return _args_to_model_kwargs(cli_dict)


def build_model(model_kwargs):
    """Build SimpleCSSM from kwargs dict."""
    # Filter to only valid SimpleCSSM constructor kwargs
    valid = {k: v for k, v in model_kwargs.items() if k in SIMPLECSSM_KWARGS}
    print(f"Building SimpleCSSM: cssm_type={valid.get('cssm_type')}, "
          f"embed_dim={valid.get('embed_dim')}, kernel_size={valid.get('kernel_size')}, "
          f"gate_type={valid.get('gate_type')}")
    return SimpleCSSM(**valid)


def load_pathfinder_samples(data_dir, difficulty, image_size, num_samples, num_frames):
    """Load Pathfinder samples (PNG-based)."""
    from src.pathfinder_data import get_pathfinder_datasets
    _, _, test_ds = get_pathfinder_datasets(
        root=data_dir, difficulty=str(difficulty),
        image_size=image_size, num_frames=num_frames,
    )
    np.random.seed(42)
    n = min(num_samples * 4, len(test_ds))
    indices = np.random.choice(len(test_ds), n, replace=False)

    # Collect balanced samples
    pos_samples, neg_samples = [], []
    for idx in indices:
        img, label = test_ds[idx]
        if label == 1 and len(pos_samples) < num_samples:
            pos_samples.append((img, label))
        elif label == 0 and len(neg_samples) < num_samples:
            neg_samples.append((img, label))
        if len(pos_samples) >= num_samples and len(neg_samples) >= num_samples:
            break

    return pos_samples + neg_samples


def load_pathfinder_tfrecord_samples(tfrecord_dir, difficulty, image_size,
                                     num_samples, num_frames, split='test'):
    """Load samples from TFRecords, resizing to image_size if needed."""
    import json
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    split_dir = os.path.join(tfrecord_dir, f'difficulty_{difficulty}', split)
    metadata_path = os.path.join(tfrecord_dir, f'difficulty_{difficulty}', 'metadata.json')
    if not os.path.isdir(split_dir):
        split_dir = os.path.join(tfrecord_dir, split)
        metadata_path = os.path.join(tfrecord_dir, 'metadata.json')

    # Determine native size stored in TFRecords
    native_size = image_size
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            native_size = json.load(f).get('image_size', image_size)
    needs_resize = (image_size != native_size)

    split_files = sorted(tf.io.gfile.glob(os.path.join(split_dir, '*.tfrecord')))
    if not split_files:
        raise FileNotFoundError(f"No TFRecords in {split_dir}")
    resize_msg = f", resizing {native_size}→{image_size}" if needs_resize else ""
    print(f"  Loading from {split_dir} ({len(split_files)} shards{resize_msg})")

    def parse(serialized):
        feats = tf.io.parse_single_example(serialized, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
        })
        image = tf.io.decode_raw(feats['image'], tf.float32)
        image = tf.reshape(image, [native_size, native_size, 3])
        if needs_resize:
            image = tf.image.resize(image, [image_size, image_size], method='bilinear')
        return image, feats['label']

    ds = tf.data.TFRecordDataset(split_files).map(parse)

    pos_samples, neg_samples = [], []
    for img_tf, label_tf in ds:
        img = img_tf.numpy()  # already normalized in TFRecords
        label = int(label_tf.numpy())
        # Stack to (T, H, W, 3)
        img_video = np.stack([img] * num_frames, axis=0)
        if label == 1 and len(pos_samples) < num_samples:
            pos_samples.append((img_video, label))
        elif label == 0 and len(neg_samples) < num_samples:
            neg_samples.append((img_video, label))
        if len(pos_samples) >= num_samples and len(neg_samples) >= num_samples:
            break

    return pos_samples + neg_samples


def denormalize(img):
    """Undo ImageNet normalization for display."""
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def compute_saliency_per_timestep(model, params, batch_stats, x_5d,
                                   smooth_grad_n=0, smooth_grad_sigma=0.15):
    """Compute d(decision_logit) / d(x) for 5D input (1, T, H, W, C).

    The decision logit is the argmax dimension of the readout (i.e., the
    predicted class logit). Gradients show what input regions drive the
    model's actual decision at each recurrence step.

    Args:
        smooth_grad_n: Number of SmoothGrad samples (0 = vanilla gradient).
            Averages gradients over N noisy copies to smooth out artifacts.
        smooth_grad_sigma: Noise std as a fraction of (max - min) of input.

    Returns:
        grads: (T, H, W, C) gradient per timestep
        logits: (2,) model output logits
    """
    variables = {'params': params, 'batch_stats': batch_stats}

    # Get prediction to determine decision class
    logits = model.apply(variables, x_5d, training=False)[0]
    decision_class = int(np.argmax(logits))

    def forward(x):
        out = model.apply(variables, x, training=False)
        return out[0, decision_class]

    grad_fn = jax.grad(forward)

    if smooth_grad_n <= 0:
        grads = grad_fn(x_5d)  # (1, T, H, W, C)
        return np.array(grads[0]), np.array(logits)

    # SmoothGrad: average gradients over N noisy copies
    sigma = smooth_grad_sigma * (float(x_5d.max()) - float(x_5d.min()))
    grads_accum = np.zeros(x_5d.shape, dtype=np.float32)
    for i in range(smooth_grad_n):
        noise = jax.random.normal(jax.random.PRNGKey(i), x_5d.shape) * sigma
        g = grad_fn(x_5d + noise)
        grads_accum += np.array(g)
    grads_accum /= smooth_grad_n
    return grads_accum[0], np.array(logits)


def make_saliency_gif(image_t0, grads_per_t, label, pred, sample_idx,
                      output_dir, frame_duration=500, signed=True):
    """Create animated GIF showing saliency evolution over recurrence steps.

    Args:
        image_t0: (H, W, C) denormalized input image
        grads_per_t: (T, H, W, C) gradients at each timestep
        label: ground truth (0=disc, 1=conn)
        pred: prediction (0=disc, 1=conn)
        sample_idx: sample index for filename
        output_dir: output directory
        frame_duration: ms per frame
        signed: if True, show signed gradients (red=positive, blue=negative evidence)
    """
    T = grads_per_t.shape[0]
    correct = pred == label
    label_str = 'connected' if label == 1 else 'disconnected'
    pred_str = 'connected' if pred == 1 else 'disconnected'
    status = 'correct' if correct else 'wrong'

    # Aggregate gradient across channels
    grad_mag = np.abs(grads_per_t).sum(axis=-1)  # (T, H, W)
    grad_signed = grads_per_t.sum(axis=-1)  # (T, H, W) signed

    # Per-timestep normalization so spatial patterns are visible at every step.
    # The gradient energy bar shows relative magnitude across time.
    per_t_energy = np.array([np.linalg.norm(grad_mag[t]) for t in range(T)])
    max_energy = per_t_energy.max() if per_t_energy.max() > 0 else 1.0
    rel_energy = per_t_energy / max_energy  # 0..1 relative to strongest frame

    frames = []
    for t in range(T):

        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 0.06],
                               wspace=0.05, hspace=0.15)

        # Panel 1: Input image
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(image_t0)
        ax0.set_title(f't={t+1}/{T}', fontsize=13)
        ax0.axis('off')

        # Panel 2: Gradient magnitude overlay (per-timestep normalized)
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(image_t0, alpha=0.3)
        vmax_t = grad_mag[t].max() if grad_mag[t].max() > 0 else 1.0
        grad_norm = np.sqrt(grad_mag[t] / (vmax_t + 1e-8))
        cmap_hot = plt.colormaps.get_cmap('hot')
        overlay = cmap_hot(np.clip(grad_norm, 0, 1))[:, :, :3]
        ax1.imshow(overlay, alpha=0.8)
        ax1.set_title('|grad| (magnitude)', fontsize=11)
        ax1.axis('off')

        # Panel 3: Signed gradient overlay (per-timestep normalized)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(image_t0, alpha=0.3)
        vmax_s = np.abs(grad_signed[t]).max() if np.abs(grad_signed[t]).max() > 0 else 1.0
        signed_norm = np.sign(grad_signed[t]) * np.sqrt(np.abs(grad_signed[t]) / (vmax_s + 1e-8))
        cmap_rdbu = plt.colormaps.get_cmap('RdBu_r')
        overlay_signed = cmap_rdbu(np.clip(signed_norm * 0.5 + 0.5, 0, 1))[:, :, :3]
        ax2.imshow(overlay_signed, alpha=0.8)
        ax2.set_title('grad (signed: red=+, blue=-)', fontsize=11)
        ax2.axis('off')

        # Bottom: gradient energy bar spanning all columns
        ax_bar = fig.add_subplot(gs[1, :])
        colors = ['#444444'] * T
        colors[t] = '#e74c3c'  # highlight current timestep
        ax_bar.bar(range(T), rel_energy, color=colors, width=0.8)
        ax_bar.set_xlim(-0.5, T - 0.5)
        ax_bar.set_ylim(0, 1.15)
        ax_bar.set_xticks(range(T))
        ax_bar.set_xticklabels([f't{i+1}' for i in range(T)], fontsize=8)
        ax_bar.set_ylabel('energy', fontsize=8)
        ax_bar.tick_params(axis='y', labelsize=7)
        ax_bar.set_title('gradient energy (relative)', fontsize=9)

        fig.suptitle(
            f'GT: {label_str} | Pred: {pred_str} ({"OK" if correct else "WRONG"})',
            fontsize=12, fontweight='bold',
            color='green' if correct else 'red',
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    # Save GIF
    gif_path = os.path.join(
        output_dir,
        f'sample_{sample_idx:03d}_{label_str}_{status}.gif'
    )
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=frame_duration, loop=0,
    )

    # Also save final frame as PNG
    png_path = gif_path.replace('.gif', '_final.png')
    pil_frames[-1].save(png_path)

    return gif_path


def compute_decision_maps(model, params, batch_stats, x_5d):
    """Compute per-pixel logits at every timestep via return_spatial.

    Applies: norm_pre → act → head (as 1×1 conv) at each (t, h, w).
    Skips: frame selection, pool, and norm_post (which is designed for the
    post-pool vector and gives wrong statistics when applied per-pixel).

    Args:
        x_5d: (1, T, H, W, 3) input

    Returns:
        spatial_logits: (T, H', W', num_classes) per-pixel class logits
        logits: (num_classes,) final pooled prediction logits
    """
    variables = {'params': params, 'batch_stats': batch_stats}

    # Per-pixel logits at all timesteps
    spatial = model.apply(variables, x_5d, training=False, return_spatial=True)
    # spatial: (1, T, H', W', num_classes)

    # Also get normal prediction
    logits = model.apply(variables, x_5d, training=False)

    return np.array(spatial[0]), np.array(logits[0])


def make_decision_gif(image_t0, spatial_logits, label, pred, sample_idx,
                      output_dir, frame_duration=500, image_size=128):
    """Create animated GIF showing per-pixel readout logits at every timestep.

    The readout head (norm_pre → act → head as 1×1 conv) is applied at every
    (t, h, w) position. Shows how per-pixel class evidence evolves over the
    recurrence.

    Panels:
        1. Input image
        2. GT class logit map
        3. Decision (argmax) class logit map
        4. Other class logit map

    All use global normalization with diverging colormap (blue=neg, red=pos).

    Args:
        image_t0: (H, W, C) denormalized input image
        spatial_logits: (T, H', W', num_classes) per-pixel class logits
        label: ground truth (0=disc, 1=conn)
        pred: prediction (0=disc, 1=conn)
        sample_idx: sample index for filename
        output_dir: output directory
        frame_duration: ms per frame
        image_size: original image size for upsampling
    """
    T, Hp, Wp, num_classes = spatial_logits.shape
    correct = pred == label
    label_str = 'connected' if label == 1 else 'disconnected'
    pred_str = 'connected' if pred == 1 else 'disconnected'
    status = 'correct' if correct else 'wrong'
    other_class = 1 - pred
    class_names = {0: 'disc', 1: 'conn'}

    H, W = image_t0.shape[:2]

    from PIL import Image as PILImage
    def upsample(arr_2d, target_h, target_w):
        pil = PILImage.fromarray(arr_2d.astype(np.float32), mode='F')
        pil = pil.resize((target_w, target_h), PILImage.BILINEAR)
        return np.array(pil)

    # Global scale across all classes and timesteps
    vmax = max(np.abs(spatial_logits).max(), 1e-8)
    cmap_rdbu = plt.colormaps.get_cmap('RdBu_r')

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(18, 4.5))
        gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.08)

        # Panel 0: Input image
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(image_t0)
        ax0.set_title(f't={t+1}/{T}', fontsize=13)
        ax0.axis('off')

        panels = [
            (1, label, f'GT: {class_names[label]}'),
            (2, pred, f'Decision: {class_names[pred]}'),
            (3, other_class, f'Other: {class_names[other_class]}'),
        ]
        for panel_idx, cls, title in panels:
            ax = fig.add_subplot(gs[panel_idx])
            ax.imshow(image_t0, alpha=0.3)
            logit_map = upsample(spatial_logits[t, :, :, cls], H, W)
            normed = np.clip(logit_map / vmax * 0.5 + 0.5, 0, 1)
            ax.imshow(cmap_rdbu(normed)[:, :, :3], alpha=0.8)
            ax.set_title(title, fontsize=11)
            ax.axis('off')

        # Colorbar
        ax_cb = fig.add_subplot(gs[4])
        sm = plt.cm.ScalarMappable(cmap=cmap_rdbu,
                                    norm=plt.Normalize(vmin=-vmax, vmax=vmax))
        plt.colorbar(sm, cax=ax_cb, label='logit')

        fig.suptitle(
            f'GT: {label_str} | Pred: {pred_str} ({"OK" if correct else "WRONG"})',
            fontsize=12, fontweight='bold',
            color='green' if correct else 'red',
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    # Save GIF
    gif_path = os.path.join(
        output_dir,
        f'sample_{sample_idx:03d}_{label_str}_{status}.gif'
    )
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=frame_duration, loop=0,
    )

    # Also save final frame as PNG
    png_path = gif_path.replace('.gif', '_final.png')
    pil_frames[-1].save(png_path)

    return gif_path


def make_parser():
    """Create argument parser with all model config args (for legacy checkpoint support)."""
    parser = argparse.ArgumentParser(
        description='Gradient saliency videos for SimpleCSSM Pathfinder models')

    # Required / visualization args
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory (contains checkpoint.pkl)')
    parser.add_argument('--output_dir', type=str, default='visualizations/saliency',
                        help='Output directory for GIFs')
    parser.add_argument('--data_dir', type=str,
                        default='/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025',
                        help='Path to Pathfinder PNG data')
    parser.add_argument('--tfrecord_dir', type=str, default='',
                        help='Path to Pathfinder TFRecords (faster, optional)')
    parser.add_argument('--difficulty', type=int, default=14)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples per class (total = 2x this)')
    parser.add_argument('--mode', type=str, default='decision',
                        choices=['decision', 'gradient'],
                        help='Visualization mode: decision (per-pixel logits) or gradient (saliency)')
    parser.add_argument('--frame_duration', type=int, default=600,
                        help='Milliseconds per frame in GIF')
    parser.add_argument('--smooth_grad_n', type=int, default=50,
                        help='[gradient mode] SmoothGrad samples (0=vanilla, 50=smooth)')
    parser.add_argument('--smooth_grad_sigma', type=float, default=0.15,
                        help='[gradient mode] SmoothGrad noise std as fraction of input range')

    # Model config args — auto-loaded from checkpoint if available, CLI overrides.
    # These mirror main.py's model construction args.
    parser.add_argument('--cssm', type=str, default=None,
                        help='CSSM variant (required for legacy checkpoints without saved args)')
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--block_size', type=int, default=1)
    parser.add_argument('--gate_type', type=str, default='dense')
    parser.add_argument('--stem_layers', type=int, default=2)
    parser.add_argument('--stem_mode', type=str, default='default')
    parser.add_argument('--stem_norm_order', type=str, default='post')
    parser.add_argument('--norm_type', type=str, default='layer')
    parser.add_argument('--stem_norm', type=str, default='')
    parser.add_argument('--body_norm', type=str, default='')
    parser.add_argument('--readout_norm', type=str, default='pre')
    parser.add_argument('--pos_embed', type=str, default='spatiotemporal')
    parser.add_argument('--act_type', type=str, default='softplus')
    parser.add_argument('--pool_type', type=str, default='max')
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=32)
    parser.add_argument('--frame_readout', type=str, default='last')
    parser.add_argument('--use_complex32', action='store_true')
    parser.add_argument('--use_complex16', action='store_true')
    parser.add_argument('--use_ssd', action='store_true')
    parser.add_argument('--scan_mode', type=str, default='associative')
    parser.add_argument('--ssd_chunk_size', type=int, default=8)
    parser.add_argument('--learned_init', action='store_true')
    parser.add_argument('--n_register_tokens', type=int, default=0)
    parser.add_argument('--position_independent_gates', action='store_true')
    parser.add_argument('--no_goom', action='store_true')
    # GDN-specific
    parser.add_argument('--short_conv_size', type=int, default=4)
    parser.add_argument('--output_norm', type=str, default='rms')
    parser.add_argument('--no_input_gates', action='store_true')
    parser.add_argument('--output_gate_act', type=str, default='silu')
    parser.add_argument('--no_spectral_l2_norm', action='store_true')
    parser.add_argument('--qkv_conv_size', type=int, default=1)
    parser.add_argument('--qkv_conv_full', action='store_true')
    parser.add_argument('--cross_freq_conv_size', type=int, default=0)
    parser.add_argument('--delta_key_dim', type=int, default=2)
    parser.add_argument('--no_spectral_clamp', action='store_true')
    # Transformer-specific
    parser.add_argument('--shared_kernel', action='store_true')
    parser.add_argument('--additive_kv', action='store_true')
    parser.add_argument('--spectral_clip', action='store_true')
    parser.add_argument('--q_temporal', type=int, default=3)
    parser.add_argument('--q_spatial', type=int, default=5)
    parser.add_argument('--asymmetric_qk', action='store_true')
    parser.add_argument('--no_feedback', action='store_true')
    parser.add_argument('--no_spectral_clip', action='store_true')
    parser.add_argument('--use_layernorm', action='store_true')
    parser.add_argument('--no_k_state', action='store_true')
    parser.add_argument('--transformer_readout', type=str, default='qka')
    # Attention-specific
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.0)

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    print("Loading checkpoint...")
    params, batch_stats, saved_args = load_checkpoint(args.checkpoint)

    # Build model config
    if saved_args is not None:
        # New checkpoint with saved training args — merge with CLI overrides
        model_kwargs = build_model_config(saved_args, args, parser)
    elif args.cssm is not None:
        # Legacy checkpoint + explicit --cssm flag — use CLI args
        model_kwargs = _args_to_model_kwargs(vars(args))
    else:
        # Legacy checkpoint, no --cssm — try old AdditiveCSSM inference
        print("No saved args and no --cssm flag, trying legacy AdditiveCSSM inference...")
        model_kwargs = _legacy_infer_config(params)

    # Build model
    print("Building model...")
    model = build_model(model_kwargs)

    # Load data
    print("Loading data...")
    seq_len = model_kwargs.get('seq_len', args.seq_len)
    if args.tfrecord_dir:
        samples = load_pathfinder_tfrecord_samples(
            args.tfrecord_dir, args.difficulty, args.image_size,
            args.num_samples, seq_len,
        )
    else:
        samples = load_pathfinder_samples(
            args.data_dir, args.difficulty, args.image_size,
            args.num_samples, seq_len,
        )
    print(f"Loaded {len(samples)} samples ({args.num_samples} per class)")

    print(f"Mode: {args.mode}")
    print("Compiling (first sample will be slow)...")

    correct_count = 0
    for i, (img_video, label) in enumerate(samples):
        # img_video: (T, H, W, 3) normalized
        x_5d = jnp.array(img_video)[None, ...]  # (1, T, H, W, 3)

        # Denormalize for display (use first frame since all identical)
        display_img = denormalize(img_video[0])

        if args.mode == 'decision':
            spatial_logits, logits = compute_decision_maps(
                model, params, batch_stats, x_5d)
            pred = int(np.argmax(logits))
            correct = pred == label
            correct_count += int(correct)
            gif_path = make_decision_gif(
                display_img, spatial_logits, label, pred, i,
                output_dir, frame_duration=args.frame_duration,
                image_size=args.image_size,
            )
        else:  # gradient
            grads, logits = compute_saliency_per_timestep(
                model, params, batch_stats, x_5d,
                smooth_grad_n=args.smooth_grad_n,
                smooth_grad_sigma=args.smooth_grad_sigma)
            pred = int(np.argmax(logits))
            correct = pred == label
            correct_count += int(correct)
            gif_path = make_saliency_gif(
                display_img, grads, label, pred, i,
                output_dir, frame_duration=args.frame_duration,
            )

        status = "OK" if correct else "WRONG"
        label_str = "conn" if label == 1 else "disc"
        pred_str = "conn" if pred == 1 else "disc"
        print(f"  [{i+1}/{len(samples)}] GT={label_str} Pred={pred_str} ({status}) → {gif_path}")

    print(f"\nAccuracy: {correct_count}/{len(samples)} ({100*correct_count/len(samples):.1f}%)")
    print(f"Output: {output_dir}/")


if __name__ == '__main__':
    main()
