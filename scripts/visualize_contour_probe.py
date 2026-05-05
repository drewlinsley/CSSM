#!/usr/bin/env python3
"""
Contour segmentation probe for CSSM temporal propagation.

Trains a linear probe (logistic regression) to detect the target contour
(the connected component linking two endpoint dots) from CSSM features,
then applies the probe at every recurrence timestep t=1..T to visualize
how contour detection builds over time.

Pipeline:
  1. Load raw Pathfinder PNGs (300x300 grayscale), extract target contour masks
  2. Preprocess images (resize 128, ImageNet normalize), extract CSSM features
  3. Train per-pixel LogisticRegression: features at t=T -> contour (0/1)
  4. Apply probe at every t=1..T for connected and disconnected images -> GIFs

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/visualize_contour_probe.py \
        --checkpoint checkpoints/channel_c32_ST_longer/epoch_85 \
        --num_train 500 --num_val 100 --num_vis 10 \
        --output_dir visualizations/contour_probe
"""

import argparse
import os
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
from scipy import ndimage
from collections import deque
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_cssm import SimpleCSSM
from src.pathfinder_data import IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Checkpoint / model helpers (reused from visualize_feature_pca.py)
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path):
    """Load pickle checkpoint, return params and batch_stats."""
    import pickle
    with open(os.path.join(checkpoint_path, 'checkpoint.pkl'), 'rb') as f:
        ckpt = pickle.load(f)
    print(f"Loaded epoch {ckpt['epoch']}, step {ckpt['step']}")
    return ckpt['params'], ckpt.get('batch_stats', {})


def infer_model_config(params):
    """Infer SimpleCSSM config from checkpoint param shapes."""
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
    norm_type = 'batch'
    config = dict(embed_dim=embed_dim, kernel_size=kernel_size, gate_type=gate_type,
                  cssm_type=cssm_type, stem_layers=stem_layers, norm_type=norm_type)
    print(f"Inferred config: {config}")
    return config


def build_model(config, seq_len=8, pos_embed='spatiotemporal',
                stem_norm_order='post', pool_type='max'):
    """Build SimpleCSSM from inferred config."""
    return SimpleCSSM(
        num_classes=2, embed_dim=config['embed_dim'], depth=1,
        cssm_type=config['cssm_type'], kernel_size=config['kernel_size'],
        gate_type=config['gate_type'], stem_layers=config['stem_layers'],
        norm_type=config['norm_type'], pos_embed=pos_embed,
        stem_norm_order=stem_norm_order, pool_type=pool_type,
        seq_len=seq_len, use_complex32=True,
    )


# ---------------------------------------------------------------------------
# Target contour extraction
# ---------------------------------------------------------------------------

def _bfs_path(mask, start, end):
    """BFS shortest path on a binary mask (8-connected).

    Args:
        mask: (H, W) bool array
        start: (y, x) start coordinate
        end: (y, x) end coordinate

    Returns:
        list of (y, x) tuples forming the path, or None if no path exists
    """
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    parent = np.full((H, W, 2), -1, dtype=np.int32)

    queue = deque()
    sy, sx = int(start[0]), int(start[1])
    ey, ex = int(end[0]), int(end[1])
    queue.append((sy, sx))
    visited[sy, sx] = True

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1)]

    found = False
    while queue:
        cy, cx = queue.popleft()
        if cy == ey and cx == ex:
            found = True
            break
        for dy, dx in neighbors:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and mask[ny, nx]:
                visited[ny, nx] = True
                parent[ny, nx] = [cy, cx]
                queue.append((ny, nx))

    if not found:
        return None

    # Trace back
    path = []
    cy, cx = ey, ex
    while cy != sy or cx != sx:
        path.append((cy, cx))
        py, px = parent[cy, cx]
        cy, cx = py, px
    path.append((sy, sx))
    return path[::-1]


def extract_target_contour(img_gray_300, target_size=64):
    """Extract the target contour mask from a 300x300 Pathfinder positive image.

    Algorithm:
      1. Binary mask: img > 0
      2. Dilate (full 3x3, 3 iters) to bridge dashed-contour gaps into solid lines
      3. Erosion (cross 3x3) to kill thin contour lines, keep endpoint dots
      4. Two largest surviving clusters = the two endpoint markers
      5. BFS shortest path between endpoints on the dilated mask
      6. Dilate the path (3 iters) to capture contour width
      7. Intersect with original mask (keep only real contour pixels)
      8. Downsample to target_size via nearest-neighbor

    Args:
        img_gray_300: (H, W) uint8 grayscale image
        target_size: spatial size to downsample mask to

    Returns:
        mask: (target_size, target_size) binary mask, or None if extraction fails
    """
    # 1. Binary mask
    binary = img_gray_300 > 0
    struct_cross = ndimage.generate_binary_structure(2, 1)  # 3x3 cross
    struct_full = np.ones((3, 3))

    # 2. Heavy dilation to bridge dashed gaps into solid connected contours
    dilated = ndimage.binary_dilation(binary, structure=struct_full, iterations=3)

    # 3. Erosion to kill thin contours, keep endpoint dots
    eroded = ndimage.binary_erosion(binary, structure=struct_cross)

    # 4. Find two largest surviving clusters (endpoint markers)
    labeled_eroded, n_clusters = ndimage.label(eroded)
    if n_clusters < 2:
        return None
    cluster_sizes = ndimage.sum(eroded, labeled_eroded, range(1, n_clusters + 1))
    top2_labels = np.argsort(cluster_sizes)[-2:] + 1  # 1-indexed

    d1 = np.array(np.where(labeled_eroded == top2_labels[1])).mean(axis=1).astype(int)
    d2 = np.array(np.where(labeled_eroded == top2_labels[0])).mean(axis=1).astype(int)

    # 5. BFS shortest path between endpoints on the dilated mask
    path_pts = _bfs_path(dilated, d1, d2)
    if path_pts is None:
        return None

    # 6. Create path mask and dilate to capture contour width
    path_mask = np.zeros_like(binary, dtype=bool)
    for y, x in path_pts:
        path_mask[y, x] = True
    path_wide = ndimage.binary_dilation(path_mask, structure=struct_full, iterations=3)

    # 7. Intersect with original mask (keep only real contour pixels)
    target_mask = path_wide & binary

    # 8. Downsample to target_size via nearest-neighbor
    mask_pil = Image.fromarray(target_mask.astype(np.uint8) * 255, mode='L')
    mask_down = mask_pil.resize((target_size, target_size), Image.NEAREST)
    mask_out = np.array(mask_down) > 127

    return mask_out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_pathfinder_images(img_dir, num_images, label, seed=42):
    """Load raw Pathfinder PNG images from pos/ or neg/ directory.

    Args:
        img_dir: Path to pos/ or neg/ directory
        num_images: Number of images to load
        label: 1 for pos, 0 for neg
        seed: Random seed for reproducible selection

    Returns:
        list of (img_gray_300, img_preprocessed_video, label) tuples
        - img_gray_300: (300, 300) uint8 original
        - img_preprocessed_video: (T, 128, 128, 3) float32 preprocessed
    """
    rng = np.random.RandomState(seed)
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    selected = rng.choice(len(all_files), size=min(num_images, len(all_files)),
                          replace=False)
    selected.sort()

    results = []
    for idx in selected:
        path = os.path.join(img_dir, all_files[idx])
        img_gray = Image.open(path).convert('L')
        img_gray_300 = np.array(img_gray)  # (300, 300) uint8

        results.append((img_gray_300, label, path))

    return results


def preprocess_image(img_gray_300, image_size=128, seq_len=8):
    """Preprocess a raw 300x300 grayscale Pathfinder image for the model.

    Matches the training pipeline in pathfinder_data.py.

    Returns:
        img_video: (T, image_size, image_size, 3) float32
    """
    img_pil = Image.fromarray(img_gray_300, mode='L')
    img_resized = img_pil.resize((image_size, image_size), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_rgb = np.stack([img_np] * 3, axis=-1)  # (H, W, 3)
    img_normalized = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
    img_video = np.stack([img_normalized] * seq_len, axis=0)  # (T, H, W, 3)
    return img_video


def denormalize(img):
    """Undo ImageNet normalization for display."""
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(model, variables, x_5d):
    """Extract post-CSSM features at all timesteps.

    Args:
        x_5d: (1, T, H, W, 3) preprocessed input

    Returns:
        features: (1, T, H', W', C) post-CSSM features
        logits: (1, num_classes)
    """
    features = model.apply(variables, x_5d, training=False, return_features=True)
    logits = model.apply(variables, x_5d, training=False)
    return np.array(features), np.array(logits)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def upsample(arr_2d, target_h, target_w):
    """Upsample a 2D float array via Lanczos interpolation."""
    pil = Image.fromarray(arr_2d.astype(np.float32), mode='F')
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    return np.array(pil)


def make_probe_gif(image_t0, features, probe, contour_mask, embed_dim,
                   label, pred, sample_idx, output_dir, image_size=128,
                   frame_duration=600, feature_size=64):
    """Create animated GIF showing probe predictions at each timestep.

    Layout per frame: [Input] [P(contour) heatmap] [Thresholded] [Ground truth]

    Args:
        image_t0: (H, W, C) denormalized input image
        features: (T, H', W', C) post-CSSM features
        probe: trained LogisticRegression
        contour_mask: (H', W') binary ground truth (or None for neg samples)
        embed_dim: feature dimension
        label: 0 or 1
        pred: model prediction
        sample_idx: for filename
        output_dir: output directory
        image_size: display size
        frame_duration: ms per frame
        feature_size: spatial size of features
    """
    T = features.shape[0]
    H, W = image_t0.shape[:2]
    correct = pred == label
    label_str = 'conn' if label == 1 else 'disc'
    status = 'correct' if correct else 'wrong'

    has_gt = contour_mask is not None
    n_cols = 4 if has_gt else 3
    fig_width = 3.5 * n_cols

    cmap_prob = plt.colormaps.get_cmap('inferno')

    frames = []
    for t in range(T):
        feat_t = features[t]  # (H', W', C)
        X_t = feat_t.reshape(-1, embed_dim)
        prob_t = probe.predict_proba(X_t)[:, 1].reshape(feature_size, feature_size)

        fig = plt.figure(figsize=(fig_width, 4))
        gs = gridspec.GridSpec(1, n_cols, wspace=0.08)

        # Panel 0: Input image
        ax = fig.add_subplot(gs[0])
        ax.imshow(image_t0)
        ax.set_title(f'Input  t={t+1}/{T}', fontsize=11)
        ax.axis('off')

        # Panel 1: P(contour) heatmap
        ax = fig.add_subplot(gs[1])
        prob_up = upsample(prob_t, H, W)
        ax.imshow(cmap_prob(prob_up)[:, :, :3])
        ax.set_title(f'P(contour)', fontsize=11)
        ax.axis('off')

        # Panel 2: Thresholded at 0.5
        ax = fig.add_subplot(gs[2])
        thresh_up = (prob_up > 0.5).astype(np.float32)
        # Overlay thresholded on input
        overlay = image_t0.copy() * 0.4
        overlay[:, :, 0] += thresh_up * 0.6  # red channel
        overlay = np.clip(overlay, 0, 1)
        ax.imshow(overlay)
        ax.set_title('Thresholded (0.5)', fontsize=11)
        ax.axis('off')

        # Panel 3: Ground truth (if available)
        if has_gt:
            ax = fig.add_subplot(gs[3])
            gt_up = upsample(contour_mask.astype(np.float32), H, W)
            gt_overlay = image_t0.copy() * 0.4
            gt_overlay[:, :, 1] += gt_up * 0.6  # green channel
            gt_overlay = np.clip(gt_overlay, 0, 1)
            ax.imshow(gt_overlay)
            ax.set_title('Ground truth', fontsize=11)
            ax.axis('off')

        gt_str = 'connected' if label == 1 else 'disconnected'
        pred_str = 'connected' if pred == 1 else 'disconnected'
        fig.suptitle(
            f'GT: {gt_str} | Pred: {pred_str} '
            f'({"OK" if correct else "WRONG"})',
            fontsize=12, fontweight='bold',
            color='green' if correct else 'red',
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    # Save GIF
    gif_path = os.path.join(output_dir, f'probe_{label_str}_{sample_idx:03d}.gif')
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=frame_duration, loop=0)

    # Save strip (all timesteps side-by-side)
    strip_path = gif_path.replace('.gif', '_strip.png')
    strip = np.concatenate(frames, axis=1)
    Image.fromarray(strip).save(strip_path)

    return gif_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Contour segmentation probe for CSSM temporal propagation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--output_dir', type=str,
                        default='visualizations/contour_probe')
    parser.add_argument('--pathfinder_dir', type=str,
                        default='/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025/curv_contour_length_14/imgs')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--num_train', type=int, default=500,
                        help='Number of connected images for probe training')
    parser.add_argument('--num_val', type=int, default=100,
                        help='Number of connected images for validation')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='Number of samples per class for GIF visualization')
    parser.add_argument('--frame_duration', type=int, default=600,
                        help='Milliseconds per GIF frame')
    # Model config overrides
    parser.add_argument('--pos_embed', type=str, default='spatiotemporal')
    parser.add_argument('--stem_norm_order', type=str, default='post')
    parser.add_argument('--pool_type', type=str, default='max')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    print("Loading checkpoint...")
    params, batch_stats = load_checkpoint(args.checkpoint)
    config = infer_model_config(params)
    embed_dim = config['embed_dim']

    print("Building model...")
    model = build_model(
        config, seq_len=args.seq_len, pos_embed=args.pos_embed,
        stem_norm_order=args.stem_norm_order, pool_type=args.pool_type)

    variables = {'params': params, 'batch_stats': batch_stats}

    # Compute spatial size after stem
    downsample_factor = 2 ** config['stem_layers']
    feature_size = args.image_size // downsample_factor

    # --- Load raw images ---
    pos_dir = os.path.join(args.pathfinder_dir, 'pos')
    neg_dir = os.path.join(args.pathfinder_dir, 'neg')

    total_pos_needed = args.num_train + args.num_val + args.num_vis
    total_neg_needed = args.num_vis

    print(f"\nLoading {total_pos_needed} positive images...")
    pos_samples = load_raw_pathfinder_images(pos_dir, total_pos_needed, label=1, seed=42)
    print(f"Loading {total_neg_needed} negative images...")
    neg_samples = load_raw_pathfinder_images(neg_dir, total_neg_needed, label=0, seed=42)

    # --- Extract target contour masks for positive images ---
    print(f"\nExtracting target contour masks (feature_size={feature_size})...")
    pos_with_masks = []
    n_failed = 0
    for img_gray_300, label, path in pos_samples:
        mask = extract_target_contour(img_gray_300, target_size=feature_size)
        if mask is not None:
            pos_with_masks.append((img_gray_300, label, path, mask))
        else:
            n_failed += 1
    print(f"  Extracted {len(pos_with_masks)} masks, {n_failed} failed")

    if len(pos_with_masks) < args.num_train + args.num_val:
        print(f"WARNING: Only {len(pos_with_masks)} valid masks, need "
              f"{args.num_train + args.num_val} for train+val")

    # Split into train / val / vis
    train_data = pos_with_masks[:args.num_train]
    val_data = pos_with_masks[args.num_train:args.num_train + args.num_val]
    vis_pos_data = pos_with_masks[args.num_train + args.num_val:
                                   args.num_train + args.num_val + args.num_vis]
    vis_neg_data = [(g, l, p, None) for g, l, p in neg_samples[:args.num_vis]]

    # Print contour statistics
    if train_data:
        contour_fracs = []
        for _, _, _, mask in train_data[:20]:
            total_fg = mask.sum()
            # Get all-contour mask at feature_size
            img_gray_300 = train_data[0][0]  # just for shape reference
            contour_fracs.append(float(total_fg) / (feature_size * feature_size))
        print(f"  Contour fraction of image: {np.mean(contour_fracs):.1%} "
              f"(range {np.min(contour_fracs):.1%}-{np.max(contour_fracs):.1%})")

    # ===================================================================
    # Phase 1: Extract features for training data
    # ===================================================================
    print(f"\n=== Phase 1: Feature extraction for probe training ===")
    print("Compiling (first sample will be slow due to JIT)...")

    X_train_list = []
    y_train_list = []

    for i, (img_gray_300, label, path, mask) in enumerate(train_data):
        img_video = preprocess_image(img_gray_300, args.image_size, args.seq_len)
        x_5d = jnp.array(img_video)[None, ...]  # (1, T, H, W, 3)
        features, logits = extract_features(model, variables, x_5d)
        pred = int(np.argmax(logits[0]))

        # Use features at last timestep (t=T)
        feat_last = features[0, -1]  # (H', W', C)
        X_train_list.append(feat_last.reshape(-1, embed_dim))
        y_train_list.append(mask.reshape(-1).astype(np.int32))

        if i < 3 or (i + 1) % 50 == 0:
            status = "OK" if pred == label else "WRONG"
            print(f"  [{i+1}/{len(train_data)}] pred={pred} ({status})")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    print(f"  Training set: {X_train.shape[0]} pixels, "
          f"{y_train.sum()} contour ({y_train.mean():.1%})")

    # ===================================================================
    # Phase 2: Extract features for validation data
    # ===================================================================
    print(f"\n=== Phase 2: Validation feature extraction ===")

    X_val_list = []
    y_val_list = []
    val_features_all = []  # cache all-timestep features for Phase 4

    for i, (img_gray_300, label, path, mask) in enumerate(val_data):
        img_video = preprocess_image(img_gray_300, args.image_size, args.seq_len)
        x_5d = jnp.array(img_video)[None, ...]
        features, logits = extract_features(model, variables, x_5d)

        val_features_all.append(features[0])  # (T, H', W', C)
        feat_last = features[0, -1]
        X_val_list.append(feat_last.reshape(-1, embed_dim))
        y_val_list.append(mask.reshape(-1).astype(np.int32))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(val_data)}]")

    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    print(f"  Validation set: {X_val.shape[0]} pixels, "
          f"{y_val.sum()} contour ({y_val.mean():.1%})")

    # ===================================================================
    # Phase 3: Train probe
    # ===================================================================
    print(f"\n=== Phase 3: Training logistic regression probe ===")
    probe = LogisticRegression(max_iter=5000, C=1.0, solver='lbfgs',
                               class_weight='balanced')
    probe.fit(X_train, y_train)

    # Evaluate on training set
    y_train_pred = probe.predict(X_train)
    print(f"  Train accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Train F1 (contour): {f1_score(y_train, y_train_pred):.4f}")

    # Evaluate on validation set
    y_val_pred = probe.predict(X_val)
    print(f"  Val accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"  Val F1 (contour): {f1_score(y_val, y_val_pred):.4f}")
    print(f"\n  Val classification report:")
    print(classification_report(y_val, y_val_pred,
                                target_names=['background', 'contour']))

    # ===================================================================
    # Phase 4: Per-timestep probe accuracy (temporal propagation curve)
    # ===================================================================
    print(f"\n=== Phase 4: Per-timestep probe accuracy ===")
    # Reuse val_features_all cached from Phase 2

    timestep_accs = []
    timestep_f1s = []
    for t in range(args.seq_len):
        X_t_list = [f[t].reshape(-1, embed_dim) for f in val_features_all]
        X_t = np.concatenate(X_t_list, axis=0)
        y_t_pred = probe.predict(X_t)
        acc = accuracy_score(y_val, y_t_pred)
        f1 = f1_score(y_val, y_t_pred)
        timestep_accs.append(acc)
        timestep_f1s.append(f1)
        print(f"  t={t+1}: acc={acc:.4f}  F1={f1:.4f}")

    # Save temporal curve plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ts = range(1, args.seq_len + 1)
    ax1.plot(ts, timestep_accs, 'o-', color='steelblue', linewidth=2)
    ax1.set_xlabel('Recurrence step t')
    ax1.set_ylabel('Probe accuracy')
    ax1.set_title('Contour detection accuracy over time')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, timestep_f1s, 's-', color='coral', linewidth=2)
    ax2.set_xlabel('Recurrence step t')
    ax2.set_ylabel('Probe F1 (contour class)')
    ax2.set_title('Contour detection F1 over time')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(args.output_dir, 'temporal_curve.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Temporal curve saved to {curve_path}")

    # ===================================================================
    # Phase 5: Generate visualization GIFs
    # ===================================================================
    print(f"\n=== Phase 5: Generating visualization GIFs ===")

    vis_samples = vis_pos_data + vis_neg_data
    print(f"  {len(vis_pos_data)} connected + {len(vis_neg_data)} disconnected")

    for i, (img_gray_300, label, path, mask) in enumerate(vis_samples):
        img_video = preprocess_image(img_gray_300, args.image_size, args.seq_len)
        x_5d = jnp.array(img_video)[None, ...]
        features, logits = extract_features(model, variables, x_5d)
        pred = int(np.argmax(logits[0]))
        status = 'OK' if pred == label else 'WRONG'

        display_img = denormalize(img_video[0])

        print(f"  [{i+1}/{len(vis_samples)}] label={label} pred={pred} ({status})")

        gif_path = make_probe_gif(
            display_img, features[0], probe, mask, embed_dim,
            label, pred, i, args.output_dir,
            image_size=args.image_size, frame_duration=args.frame_duration,
            feature_size=feature_size)
        print(f"    -> {gif_path}")

    print(f"\nDone. Output: {args.output_dir}/")


if __name__ == '__main__':
    main()
