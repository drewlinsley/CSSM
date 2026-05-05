#!/usr/bin/env python3
"""
Feature-space visualization of CSSM propagation.

Four complementary analyses of post-stem / post-CSSM / internal CSSM features:
  A. Post-CSSM PCA  — unsupervised: directions of max variance in output features
  B. Post-CSSM PLS  — supervised: directions most predictive of class label
  C. V-state PCA    — PCA on the internal V state (IFFT of spectral V_hat),
                       showing what the recurrence computes at each timestep
  D. d(logit)/d(stem_t)  — how the model uses its input drive at each t
                            (gradients flow back through the full CSSM recurrence)

PCA/PLS reduce the C-dim channel space to K interpretable spatial maps,
animated over T recurrence steps as GIFs.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/visualize_feature_pca.py \
        --checkpoint checkpoints/channel_c32_ST_longer/epoch_110 \
        --tfrecord_dir /media/data_cifs/projects/prj_video_imagenet/pathfinder_tfrecords_128 \
        --output_dir visualizations/feature_pca
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
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simple_cssm import SimpleCSSM
from src.pathfinder_data import IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Checkpoint / model helpers (reused from visualize_saliency_video.py)
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
# Data loading
# ---------------------------------------------------------------------------

def load_tfrecord_samples(tfrecord_dir, difficulty, image_size, num_samples,
                          num_frames, split='test'):
    """Load balanced samples from TFRecords."""
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    split_dir = os.path.join(tfrecord_dir, f'difficulty_{difficulty}', split)
    if not os.path.isdir(split_dir):
        split_dir = os.path.join(tfrecord_dir, split)

    split_files = sorted(tf.io.gfile.glob(os.path.join(split_dir, '*.tfrecord')))
    if not split_files:
        raise FileNotFoundError(f"No TFRecords in {split_dir}")
    print(f"  Loading from {split_dir} ({len(split_files)} shards)")

    def parse(serialized):
        feats = tf.io.parse_single_example(serialized, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
        })
        image = tf.io.decode_raw(feats['image'], tf.float32)
        image = tf.reshape(image, [image_size, image_size, 3])
        return image, feats['label']

    ds = tf.data.TFRecordDataset(split_files).map(parse)

    pos_samples, neg_samples = [], []
    for img_tf, label_tf in ds:
        img = img_tf.numpy()
        label = int(label_tf.numpy())
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


# ---------------------------------------------------------------------------
# Phase 1: Feature extraction
# ---------------------------------------------------------------------------

def _spectral_to_spatial(hat, H_spatial, W_spatial):
    """Convert spectral state (B, T, C, H, W_freq) complex → (B, T, H, W, C) real."""
    spatial_chw = np.array(jnp.fft.irfft2(hat, s=(H_spatial, W_spatial), axes=(3, 4)))
    return spatial_chw.transpose(0, 1, 3, 4, 2)


def extract_features(model, variables, x_5d, H_spatial=64, W_spatial=64):
    """Extract pre-CSSM, post-CSSM, and internal Q/K/V state features via sow.

    Args:
        x_5d: (1, T, H, W, 3)
        H_spatial, W_spatial: spatial dims after stem (for IFFT of spectral states)
    Returns:
        dict with keys:
            pre_cssm:   (1, T, H', W', C)
            post_cssm:  (1, T, H', W', C)
            Q_spatial:  (1, T, H', W', C) — Q state in spatial domain
            K_spatial:  (1, T, H', W', C) — K state in spatial domain (if 3-state)
            V_spatial:  (1, T, H', W', C) — V state in spatial domain
            logits:     (1, num_classes)
    """
    logits, state = model.apply(
        variables, x_5d, training=False, mutable=['intermediates'])
    cssm_state = state['intermediates']['cssm_0']

    result = {
        'pre_cssm': np.array(state['intermediates']['pre_cssm'][0]),
        'post_cssm': np.array(state['intermediates']['post_cssm'][0]),
        'logits': np.array(logits),
    }

    # Convert spectral states to spatial domain
    for name in ['Q_hat', 'K_hat', 'V_hat']:
        if name in cssm_state:
            spatial_name = name.replace('_hat', '_spatial')
            result[spatial_name] = _spectral_to_spatial(
                cssm_state[name][0], H_spatial, W_spatial)

    return result


# ---------------------------------------------------------------------------
# Phase 2: Dimensionality reduction (PCA / PLS)
# ---------------------------------------------------------------------------

def _remove_dc(features):
    """Remove spatial mean per sample per timestep per channel (DC component).

    Args:
        features: (T, H', W', C)
    Returns:
        features with spatial mean subtracted: (T, H', W', C)
    """
    # Mean over spatial dims (H', W'), keep T and C
    spatial_mean = features.mean(axis=(1, 2), keepdims=True)  # (T, 1, 1, C)
    return features - spatial_mean


def fit_pca(features_list, n_components=3, label=''):
    """Fit PCA on stacked features from multiple samples.

    Removes per-sample spatial DC before fitting so PCA captures
    spatial variation rather than position-independent channel means.

    Args:
        features_list: list of (T, H', W', C) arrays
        n_components: number of principal components
        label: descriptive label for printing
    Returns:
        pca: fitted PCA object
    """
    stacked = np.concatenate(
        [_remove_dc(f).reshape(-1, f.shape[-1]) for f in features_list], axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(stacked)
    print(f"  {label}PCA explained variance: {pca.explained_variance_ratio_}")
    return pca


def fit_pls(features_list, labels_list, n_components=3, label=''):
    """Fit PLS on stacked features predicting class labels.

    Removes per-sample spatial DC before fitting.

    Args:
        features_list: list of (T, H', W', C) arrays
        labels_list: list of int labels
        n_components: number of PLS components
        label: descriptive label for printing
    Returns:
        pls: fitted PLSRegression object
    """
    X_parts, Y_parts = [], []
    for feat, lab in zip(features_list, labels_list):
        dc_removed = _remove_dc(feat)
        n_pixels = dc_removed.reshape(-1, feat.shape[-1]).shape[0]
        X_parts.append(dc_removed.reshape(-1, feat.shape[-1]))
        Y_parts.append(np.full((n_pixels, 1), lab, dtype=np.float32))
    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)
    r2 = pls.score(X, Y)
    print(f"  {label}PLS R^2 (train): {r2:.4f}")
    return pls


def project_pca(pca, features):
    """Project features through PCA (after DC removal).

    Args:
        features: (T, H', W', C)
        pca: fitted PCA
    Returns:
        projected: (T, H', W', K) with K = n_components
    """
    T, Hp, Wp, C = features.shape
    flat = _remove_dc(features).reshape(-1, C)
    proj = pca.transform(flat)
    return proj.reshape(T, Hp, Wp, -1)


def project_pls(pls, features):
    """Project features through PLS (after DC removal).

    Args:
        features: (T, H', W', C)
        pls: fitted PLSRegression
    Returns:
        projected: (T, H', W', K) with K = n_components
    """
    T, Hp, Wp, C = features.shape
    flat = _remove_dc(features).reshape(-1, C)
    proj = pls.transform(flat)
    return proj.reshape(T, Hp, Wp, -1)


# ---------------------------------------------------------------------------
# Phase 3: Gradient analysis
# ---------------------------------------------------------------------------

def _softmax_pool_objective(spatial_logits, pred_class, temperature=0.1):
    """Soft-max pool: differentiable approximation to max pool.

    Concentrates gradient weight on the spatial positions the model
    actually uses for its decision, rather than summing uniformly.
    """
    last_frame = spatial_logits[0, -1, :, :, pred_class]  # (H', W')
    weights = jax.nn.softmax(last_frame.ravel() / temperature)
    return jnp.dot(weights, last_frame.ravel())


def grad_wrt_stem(model, variables, x_5d, pred_class):
    """Gradient × input attribution w.r.t. pre-CSSM features.

    Uses soft-max pool (not hard max or uniform sum) to concentrate
    gradient on positions the model reads from. Returns grad * input
    to highlight features that are both active AND influential.

    Returns:
        gxi: (1, T, H', W', C) — gradient × input (signed)
    """
    _, state = model.apply(
        variables, x_5d, training=False, mutable=['intermediates'])
    pre_cssm = state['intermediates']['pre_cssm'][0]
    pre_cssm_jax = jnp.array(pre_cssm)

    def f(feat):
        spatial_logits = model.apply(
            variables, x_5d, training=False,
            injected_features=feat, return_spatial=True)
        return _softmax_pool_objective(spatial_logits, pred_class)

    g = jax.grad(f)(pre_cssm_jax)
    # Gradient × input: highlights features that are both active and influential
    gxi = g * pre_cssm_jax
    return np.array(gxi)


def grad_wrt_post_cssm(model, variables, x_5d, pred_class):
    """Gradient × input attribution w.r.t. post-CSSM features.

    Uses soft-max pool objective. Gradient flows through the readout only
    (norm → act → norm → head), with no FFT in the gradient path.
    This shows which features at each spatial position and timestep
    are decision-relevant.

    Returns:
        gxi: (1, T, H', W', C) — gradient × input (signed)
    """
    _, state = model.apply(
        variables, x_5d, training=False, mutable=['intermediates'])
    post_cssm = state['intermediates']['post_cssm'][0]
    post_cssm_jax = jnp.array(post_cssm)

    def f(feat):
        spatial_logits = model.apply(
            variables, x_5d, training=False,
            injected_post_cssm=feat, return_spatial=True)
        return _softmax_pool_objective(spatial_logits, pred_class)

    g = jax.grad(f)(post_cssm_jax)
    gxi = g * post_cssm_jax
    return np.array(gxi)


# ---------------------------------------------------------------------------
# Phase 3b: Sparse Autoencoder
# ---------------------------------------------------------------------------

def train_sae(features_list, hidden_dim=2048, l1_weight=0.1, n_steps=5000,
              batch_size=4096, lr=3e-4, label=''):
    """Train a temporal sparse autoencoder on features.

    Input at each spatial position is the full temporal trace: (T*C,).
    Each SAE feature captures a spatiotemporal pattern — e.g., "activity
    that builds from t=3 onwards at this channel combination."

    Args:
        features_list: list of (T, H', W', C) arrays
        hidden_dim: overcomplete hidden dimension
        l1_weight: L1 sparsity penalty
        n_steps: training steps
        batch_size: mini-batch size
        lr: learning rate
        label: descriptive label for printing
    Returns:
        sae_params: dict with encoder/decoder weights
        stats: dict with training statistics and normalization params
        sae_forward: JIT-compiled forward function
    """
    import optax

    # Stack: per spatial position, concatenate all T timesteps → (T*C,)
    T = features_list[0].shape[0]
    C = features_list[0].shape[-1]
    all_data = []
    for f in features_list:
        dc = _remove_dc(f)  # (T, H', W', C)
        # Reshape: (H'*W', T*C) — each row is one spatial position's full trace
        Hp, Wp = dc.shape[1], dc.shape[2]
        spatial = dc.transpose(1, 2, 0, 3).reshape(Hp * Wp, T * C)
        all_data.append(spatial)
    all_data = np.concatenate(all_data, axis=0)  # (N*H'*W', T*C)

    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + 1e-8
    all_data = (all_data - mean) / std
    N, D_in = all_data.shape
    print(f"  {label}SAE: {N} spatial positions, input_dim={D_in} (T={T}×C={C}) "
          f"→ hidden={hidden_dim}, l1={l1_weight}")

    data_jax = jnp.array(all_data, dtype=jnp.float32)

    # Initialize params
    rng = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(rng)
    scale_enc = (2.0 / D_in) ** 0.5
    scale_dec = (2.0 / hidden_dim) ** 0.5
    params = {
        'enc_w': jax.random.normal(k1, (D_in, hidden_dim)) * scale_enc,
        'enc_b': jnp.zeros(hidden_dim),
        'dec_w': jax.random.normal(k2, (hidden_dim, D_in)) * scale_dec,
        'dec_b': jnp.zeros(D_in),
    }

    def sae_forward(params, x):
        h = jax.nn.relu(x @ params['enc_w'] + params['enc_b'])
        x_hat = h @ params['dec_w'] + params['dec_b']
        return x_hat, h

    def loss_fn(params, batch):
        x_hat, h = sae_forward(params, batch)
        mse = jnp.mean((batch - x_hat) ** 2)
        l1 = jnp.mean(jnp.abs(h))
        return mse + l1_weight * l1, (mse, l1)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        (loss, (mse, l1)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, mse, l1

    rng = jax.random.PRNGKey(0)
    for i in range(n_steps):
        rng, key = jax.random.split(rng)
        idx = jax.random.randint(key, (batch_size,), 0, N)
        batch = data_jax[idx]
        params, opt_state, loss, mse, l1 = step(params, opt_state, batch)
        if (i + 1) % 500 == 0 or i == 0:
            _, h_sample = sae_forward(params, batch[:1000])
            active_frac = float((h_sample > 0).mean())
            alive = float((h_sample.max(axis=0) > 0).mean())
            print(f"    Step {i+1}/{n_steps}: loss={float(loss):.4f} "
                  f"mse={float(mse):.4f} l1={float(l1):.4f} "
                  f"active={active_frac:.1%} alive={alive:.1%}")

    # Final stats
    _, h_all = sae_forward(params, data_jax[:10000])
    active_frac = float((h_all > 0).mean())
    alive_frac = float((h_all.max(axis=0) > 0).mean())
    print(f"  {label}SAE trained: {active_frac:.1%} active, "
          f"{alive_frac:.1%} alive features ({int(alive_frac * hidden_dim)}/{hidden_dim})")

    stats = {'mean': mean, 'std': std, 'T': T, 'C': C,
             'active_frac': active_frac, 'alive_frac': alive_frac}
    return params, stats, sae_forward


def encode_sae(sae_params, sae_forward, features, stats):
    """Encode features through temporal SAE, returning sparse codes.

    Args:
        sae_params: trained SAE parameters
        sae_forward: forward function
        features: (T, H', W', C)
        stats: normalization stats from training
    Returns:
        codes: (H', W', hidden_dim) — sparse activation per spatial position
    """
    T, Hp, Wp, C = features.shape
    dc = _remove_dc(features)
    # (H'*W', T*C) — same format as training
    spatial = dc.transpose(1, 2, 0, 3).reshape(Hp * Wp, T * C)
    spatial_norm = (spatial - stats['mean']) / stats['std']
    _, codes = sae_forward(sae_params, jnp.array(spatial_norm, dtype=jnp.float32))
    return np.array(codes).reshape(Hp, Wp, -1)


def make_sae_gif(image_t0, codes, sae_params, stats, label, pred, sample_idx,
                 output_dir, prefix='sae', k_show=8, frame_duration=600,
                 image_size=128):
    """Create animated GIF showing top-K SAE feature activations over time.

    Each SAE feature has a spatial activation map (where it fires) and a
    temporal profile (from decoder weights). The animation shows
    code(h,w) * temporal_weight(t) at each timestep — revealing propagation.

    Args:
        image_t0: (H, W, C) denormalized input image
        codes: (H', W', hidden_dim) sparse codes
        sae_params: for extracting decoder temporal profiles
        stats: with T, C for reshaping decoder
        label, pred: ground truth and prediction
    """
    Hp, Wp, D = codes.shape
    T, C = stats['T'], stats['C']
    H, W = image_t0.shape[:2]
    correct = pred == label
    label_str = 'connected' if label == 1 else 'disconnected'
    pred_str = 'connected' if pred == 1 else 'disconnected'
    status = 'correct' if correct else 'wrong'

    # Decoder weights: (hidden_dim, T*C) → reshape to (hidden_dim, T, C)
    dec_w = np.array(sae_params['dec_w']).reshape(D, T, C)
    # Temporal magnitude per feature: sum |decoder| over channels at each t
    temporal_mag = np.abs(dec_w).sum(axis=-1)  # (hidden_dim, T)

    # Find top-K features by total activation (code * temporal sum)
    total_activation = codes.sum(axis=(0, 1))  # (hidden_dim,)
    top_k_idx = np.argsort(total_activation)[::-1][:k_show]

    n_cols = 1 + k_show
    fig_width = 2.0 * n_cols

    # Per-feature, per-frame activation: code(h,w) * temporal_mag(feature, t)
    # → (T, H', W') per feature
    feature_maps = {}
    for idx in top_k_idx:
        code_map = codes[:, :, idx]  # (H', W')
        t_profile = temporal_mag[idx]  # (T,)
        # Normalize temporal profile to [0, 1] for modulation
        t_norm = t_profile / (t_profile.max() + 1e-8)
        maps_t = np.stack([code_map * t_norm[t] for t in range(T)], axis=0)
        feature_maps[idx] = maps_t

    # Global vmax per feature
    vmaxes = {idx: max(float(feature_maps[idx].max()), 1e-8)
              for idx in top_k_idx}

    cmap = plt.colormaps.get_cmap('magma')

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(fig_width, 3.5))
        gs = gridspec.GridSpec(1, n_cols, wspace=0.05)

        ax = fig.add_subplot(gs[0])
        ax.imshow(image_t0)
        ax.set_title(f't={t+1}/{T}', fontsize=10)
        ax.axis('off')

        for col, idx in enumerate(top_k_idx):
            ax = fig.add_subplot(gs[1 + col])
            activation = feature_maps[idx][t]
            act_up = upsample(activation, H, W)
            norm_val = np.clip(act_up / vmaxes[idx], 0, 1)
            ax.imshow(cmap(norm_val)[:, :, :3])
            ax.set_title(f'f{idx}', fontsize=8)
            ax.axis('off')

        fig.suptitle(
            f'{prefix} | GT: {label_str} | Pred: {pred_str} '
            f'({"OK" if correct else "WRONG"})',
            fontsize=10, fontweight='bold',
            color='green' if correct else 'red',
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    gif_path = os.path.join(
        output_dir, f'{prefix}_{sample_idx:03d}_{label_str}_{status}.gif')
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=frame_duration, loop=0)
    pil_frames[-1].save(gif_path.replace('.gif', '_final.png'))
    return gif_path


# ---------------------------------------------------------------------------
# Phase 4: Visualization
# ---------------------------------------------------------------------------

def gaussian_smooth(arr_2d, sigma=0.5):
    """Light Gaussian smoothing to suppress single-pixel noise."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr_2d, sigma=sigma)


def upsample(arr_2d, target_h, target_w):
    """Upsample a 2D float array via Lanczos interpolation (sharper edges)."""
    pil = Image.fromarray(arr_2d.astype(np.float32), mode='F')
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    return np.array(pil)


def make_pca_pls_gif(image_t0, pca_proj, pls_proj, label, pred, sample_idx,
                     output_dir, prefix='pca_pls', row_labels=('PCA', 'PLS'),
                     k_show=3, frame_duration=600, image_size=128):
    """Create animated GIF showing PCA and PLS projections over time.

    Args:
        image_t0: (H, W, C) denormalized input image
        pca_proj: (T, H', W', K) PCA projections
        pls_proj: (T, H', W', K) PLS projections
        label, pred: ground truth and prediction
        sample_idx: for filename
        output_dir: output directory
        prefix: filename prefix
        row_labels: tuple of (pca_label, pls_label)
        k_show: number of components to show
        frame_duration: ms per frame
        image_size: original image size for upsampling
    """
    T = pca_proj.shape[0]
    H, W = image_t0.shape[:2]
    correct = pred == label
    label_str = 'connected' if label == 1 else 'disconnected'
    pred_str = 'connected' if pred == 1 else 'disconnected'
    status = 'correct' if correct else 'wrong'

    n_cols = 1 + k_show + k_show  # input + PCA components + PLS components
    fig_width = 3 * n_cols

    # Global scales for consistent colorbars across timesteps
    pca_vmax = [max(np.percentile(np.abs(pca_proj[:, :, :, k]), 99.5), 1e-8)
                for k in range(k_show)]
    pls_vmax = [max(np.percentile(np.abs(pls_proj[:, :, :, k]), 99.5), 1e-8)
                for k in range(k_show)]

    cmap = plt.colormaps.get_cmap('RdBu_r')

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(fig_width, 4))
        gs = gridspec.GridSpec(1, n_cols, wspace=0.08)

        # Panel 0: Input image
        ax = fig.add_subplot(gs[0])
        ax.imshow(image_t0)
        ax.set_title(f't={t+1}/{T}', fontsize=12)
        ax.axis('off')

        # Panels 1..k_show: PCA
        for k in range(k_show):
            ax = fig.add_subplot(gs[1 + k])
            comp = pca_proj[t, :, :, k]
            comp_up = upsample(comp, H, W)
            norm_val = np.clip(comp_up / pca_vmax[k] * 0.5 + 0.5, 0, 1)
            ax.imshow(cmap(norm_val)[:, :, :3])
            ax.set_title(f'{row_labels[0]}-{k+1}', fontsize=10)
            ax.axis('off')

        # Panels k_show+1..2*k_show: PLS
        for k in range(k_show):
            ax = fig.add_subplot(gs[1 + k_show + k])
            comp = pls_proj[t, :, :, k]
            comp_up = upsample(comp, H, W)
            norm_val = np.clip(comp_up / pls_vmax[k] * 0.5 + 0.5, 0, 1)
            ax.imshow(cmap(norm_val)[:, :, :3])
            ax.set_title(f'{row_labels[1]}-{k+1}', fontsize=10)
            ax.axis('off')

        fig.suptitle(
            f'GT: {label_str} | Pred: {pred_str} ({"OK" if correct else "WRONG"})',
            fontsize=12, fontweight='bold',
            color='green' if correct else 'red',
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    gif_path = os.path.join(
        output_dir, f'{prefix}_{sample_idx:03d}_{label_str}_{status}.gif')
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=frame_duration, loop=0)
    pil_frames[-1].save(gif_path.replace('.gif', '_final.png'))
    return gif_path


def make_logit_lens_gif(image_t0, spatial_logits, label, pred, sample_idx,
                        output_dir, prefix='logit_lens',
                        frame_duration=600, image_size=128):
    """Create animated GIF showing per-pixel class evidence at each timestep.

    The "logit lens" applies the model's own readout head to features at each
    timestep, producing a spatial map of class evidence. This directly shows
    where the model sees "connected" vs "disconnected" evidence accumulating
    over time — no gradients, no FFT artifacts.

    Args:
        image_t0: (H, W, C) denormalized input image
        spatial_logits: (T, H', W', 2) per-pixel logits at each timestep
        label, pred: ground truth and prediction
    """
    T, Hp, Wp, n_classes = spatial_logits.shape
    H, W = image_t0.shape[:2]
    correct = pred == label
    label_str = 'connected' if label == 1 else 'disconnected'
    pred_str = 'connected' if pred == 1 else 'disconnected'
    status = 'correct' if correct else 'wrong'

    # Logit difference: evidence for "connected" (class 1) minus "disconnected" (class 0)
    logit_diff = spatial_logits[:, :, :, 1] - spatial_logits[:, :, :, 0]  # (T, H', W')

    # Spatial anomaly: remove per-timestep spatial mean to show WHERE evidence is
    # (removes the global "vote" — reveals spatial structure of evidence)
    spatial_mean = logit_diff.mean(axis=(1, 2), keepdims=True)  # (T, 1, 1)
    anomaly = logit_diff - spatial_mean  # (T, H', W')

    # Per-timestep vmax for anomaly (each frame at full contrast)
    anom_vmax_per_t = [max(np.percentile(np.abs(anomaly[t]), 99), 1e-8)
                       for t in range(T)]

    # Temporal difference: how anomaly changes between timesteps
    delta_anomaly = np.zeros_like(anomaly)
    delta_anomaly[0] = anomaly[0]  # first frame: anomaly itself
    for t in range(1, T):
        delta_anomaly[t] = anomaly[t] - anomaly[t - 1]

    cmap_div = plt.colormaps.get_cmap('RdBu_r')
    cmap_hot = plt.colormaps.get_cmap('inferno')

    n_cols = 5  # input | logit diff | anomaly (per-t norm) | Δanomaly | overlay
    fig_width = 3.0 * n_cols

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(fig_width, 4))
        gs = gridspec.GridSpec(1, n_cols, wspace=0.08)

        # Panel 0: Input image
        ax = fig.add_subplot(gs[0])
        ax.imshow(image_t0)
        ax.set_title(f't={t+1}/{T}', fontsize=12)
        ax.axis('off')

        # Panel 1: Raw logit difference (connected - disconnected), diverging
        ax = fig.add_subplot(gs[1])
        diff = logit_diff[t]
        diff_up = upsample(diff, H, W)
        # Per-timestep normalization for raw logit diff too
        diff_vmax_t = max(np.percentile(np.abs(diff), 99), 1e-8)
        norm_val = np.clip(diff_up / diff_vmax_t * 0.5 + 0.5, 0, 1)
        ax.imshow(cmap_div(norm_val)[:, :, :3])
        mean_val = float(spatial_mean[t, 0, 0])
        ax.set_title(f'logit diff (μ={mean_val:.1f})', fontsize=8)
        ax.axis('off')

        # Panel 2: Spatial anomaly (per-timestep normalization)
        ax = fig.add_subplot(gs[2])
        anom = anomaly[t]
        anom_up = upsample(anom, H, W)
        vmax_t = anom_vmax_per_t[t]
        norm_val = np.clip(anom_up / vmax_t * 0.5 + 0.5, 0, 1)
        ax.imshow(cmap_div(norm_val)[:, :, :3])
        ax.set_title(f'anomaly (±{vmax_t:.1f})', fontsize=8)
        ax.axis('off')

        # Panel 3: Temporal difference (what changed from t-1 to t)
        ax = fig.add_subplot(gs[3])
        da = delta_anomaly[t]
        da_up = upsample(da, H, W)
        da_vmax = max(np.percentile(np.abs(da), 99), 1e-8)
        norm_val = np.clip(da_up / da_vmax * 0.5 + 0.5, 0, 1)
        ax.imshow(cmap_div(norm_val)[:, :, :3])
        lbl = 'anomaly(t=1)' if t == 0 else f'Δanomaly'
        ax.set_title(lbl, fontsize=9)
        ax.axis('off')

        # Panel 4: Overlay — anomaly on input image
        ax = fig.add_subplot(gs[4])
        anom_mag = np.abs(anom_up)
        anom_mag_norm = np.clip(anom_mag / vmax_t, 0, 1)
        heatmap = cmap_hot(anom_mag_norm)[:, :, :3]
        overlay = image_t0.copy() * 0.4 + heatmap * 0.6
        overlay = np.clip(overlay, 0, 1)
        ax.imshow(overlay)
        ax.set_title('|anomaly| overlay', fontsize=9)
        ax.axis('off')

        fig.suptitle(
            f'Logit Lens | GT: {label_str} | Pred: {pred_str} '
            f'({"OK" if correct else "WRONG"})',
            fontsize=11, fontweight='bold',
            color='green' if correct else 'red',
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    gif_path = os.path.join(
        output_dir, f'{prefix}_{sample_idx:03d}_{label_str}_{status}.gif')
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=frame_duration, loop=0)
    pil_frames[-1].save(gif_path.replace('.gif', '_final.png'))
    return gif_path


def make_gradient_gif(image_t0, grad_map, label, pred, sample_idx,
                      output_dir, prefix='grad_stem',
                      k_show=3, frame_duration=600, image_size=128):
    """Create animated GIF showing gradient maps over time.

    Uses per-sample PCA on the gradient to find principal gradient directions.
    Applies Gaussian smoothing to diffuse max-pool spike artifacts.

    Args:
        image_t0: (H, W, C) denormalized input image
        grad_map: (1, T, H', W', C) or (T, H', W', C) gradients
        label, pred: ground truth and prediction
        sample_idx: for filename
        output_dir: output directory
        prefix: filename prefix
        k_show: number of components to show
        frame_duration: ms per frame
        image_size: original image size for upsampling
    """
    if grad_map.ndim == 5:
        grad_map = grad_map[0]  # (T, H', W', C)
    T, Hp, Wp, C_feat = grad_map.shape
    H, W = image_t0.shape[:2]
    correct = pred == label
    label_str = 'connected' if label == 1 else 'disconnected'
    pred_str = 'connected' if pred == 1 else 'disconnected'
    status = 'correct' if correct else 'wrong'

    # Per-sample PCA on the gradient (not activation PCA — different structure)
    flat = grad_map.reshape(-1, C_feat)
    sample_pca = PCA(n_components=k_show)
    proj = sample_pca.fit_transform(flat).reshape(T, Hp, Wp, k_show)

    k_actual = min(k_show, proj.shape[-1])
    n_cols = 1 + k_actual + 1  # input + K components + magnitude
    fig_width = 3 * n_cols

    proj_vmax = [max(np.percentile(np.abs(proj[:, :, :, k]), 97), 1e-8)
                 for k in range(k_actual)]
    # Magnitude: sum of absolute grad×input across channels
    grad_mag = np.abs(grad_map).sum(axis=-1)  # (T, H', W')
    mag_vmax = max(np.percentile(grad_mag, 97), 1e-8)

    cmap_div = plt.colormaps.get_cmap('RdBu_r')
    cmap_hot = plt.colormaps.get_cmap('hot')

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(fig_width, 4))
        gs = gridspec.GridSpec(1, n_cols, wspace=0.08)

        # Panel 0: Input
        ax = fig.add_subplot(gs[0])
        ax.imshow(image_t0)
        ax.set_title(f't={t+1}/{T}', fontsize=12)
        ax.axis('off')

        # Panels 1..K: gradient PCs (diverging)
        for k in range(k_actual):
            ax = fig.add_subplot(gs[1 + k])
            comp = proj[t, :, :, k]
            comp_up = upsample(comp, H, W)
            norm_val = np.clip(comp_up / proj_vmax[k] * 0.5 + 0.5, 0, 1)
            ax.imshow(cmap_div(norm_val)[:, :, :3])
            ax.set_title(f'{prefix} PC{k+1}', fontsize=9)
            ax.axis('off')

        # Last panel: magnitude
        ax = fig.add_subplot(gs[1 + k_actual])
        mag = grad_mag[t]
        mag_up = upsample(mag, H, W)
        norm_val = np.clip(mag_up / mag_vmax, 0, 1)
        ax.imshow(cmap_hot(norm_val)[:, :, :3])
        ax.set_title(f'{prefix} |grad|', fontsize=9)
        ax.axis('off')

        fig.suptitle(
            f'GT: {label_str} | Pred: {pred_str} ({"OK" if correct else "WRONG"})',
            fontsize=12, fontweight='bold',
            color='green' if correct else 'red',
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    gif_path = os.path.join(
        output_dir, f'{prefix}_{sample_idx:03d}_{label_str}_{status}.gif')
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=frame_duration, loop=0)
    pil_frames[-1].save(gif_path.replace('.gif', '_final.png'))
    return gif_path




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Feature-space PCA/PLS visualization of CSSM propagation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--tfrecord_dir', type=str, required=True,
                        help='Path to Pathfinder TFRecords')
    parser.add_argument('--output_dir', type=str,
                        default='visualizations/feature_pca')
    parser.add_argument('--difficulty', type=int, default=14)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--num_pca_samples', type=int, default=30,
                        help='Samples for fitting PCA/PLS (balanced)')
    parser.add_argument('--num_vis_samples', type=int, default=5,
                        help='Samples per class for GIF visualization')
    parser.add_argument('--k_show', type=int, default=3,
                        help='Number of PCA/PLS components to display')
    parser.add_argument('--frame_duration', type=int, default=600,
                        help='Milliseconds per GIF frame')
    # Model config overrides
    parser.add_argument('--pos_embed', type=str, default='spatiotemporal')
    parser.add_argument('--stem_norm_order', type=str, default='post')
    parser.add_argument('--pool_type', type=str, default='max')
    parser.add_argument('--logit_lens_only', action='store_true',
                        help='Only run logit lens (skip PCA/PLS/SAE/gradients)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    print("Loading checkpoint...")
    params, batch_stats = load_checkpoint(args.checkpoint)
    config = infer_model_config(params)

    print("Building model...")
    model = build_model(
        config, seq_len=args.seq_len, pos_embed=args.pos_embed,
        stem_norm_order=args.stem_norm_order, pool_type=args.pool_type)

    variables = {'params': params, 'batch_stats': batch_stats}

    # --- Load data ---
    print("Loading data...")
    num_load = args.num_vis_samples if args.logit_lens_only else args.num_pca_samples
    samples = load_tfrecord_samples(
        args.tfrecord_dir, args.difficulty, args.image_size,
        num_load, args.seq_len)
    print(f"Loaded {len(samples)} samples")

    # ===================================================================
    # Logit-lens-only fast path
    # ===================================================================
    if args.logit_lens_only:
        print("\n=== Logit Lens Only Mode ===")
        print("Compiling (first sample will be slow due to JIT)...")
        vis_pos = [(s, l) for s, l in samples if l == 1][:args.num_vis_samples]
        vis_neg = [(s, l) for s, l in samples if l == 0][:args.num_vis_samples]
        vis_samples_raw = vis_pos + vis_neg

        for i, (img_video, label) in enumerate(vis_samples_raw):
            x_5d = jnp.array(img_video)[None, ...]
            logits = np.array(model.apply(
                variables, x_5d, training=False))
            pred = int(np.argmax(logits[0]))
            status = 'OK' if pred == label else 'WRONG'
            print(f"\n  Sample {i+1}/{len(vis_samples_raw)}  "
                  f"(label={label}, pred={pred}, {status})")

            display_img = denormalize(img_video[0])

            # Logit lens: apply readout at each timestep
            spatial_logits = np.array(model.apply(
                variables, x_5d, training=False, return_spatial=True))
            gif = make_logit_lens_gif(
                display_img, spatial_logits[0], label, pred, i,
                args.output_dir, prefix='logit_lens',
                frame_duration=args.frame_duration, image_size=args.image_size)
            print(f"    Logit lens → {gif}")

        print(f"\nDone. Output: {args.output_dir}/")
        return

    # ===================================================================
    # Phase 1: Extract features from all PCA samples
    # ===================================================================
    print("\n=== Phase 1: Feature extraction ===")
    print("Compiling (first sample will be slow due to JIT)...")
    all_feats = {'post_cssm': [], 'Q_spatial': [], 'K_spatial': [], 'V_spatial': []}
    all_labels = []
    all_preds = []
    for i, (img_video, label) in enumerate(samples):
        x_5d = jnp.array(img_video)[None, ...]  # (1, T, H, W, 3)
        feat = extract_features(model, variables, x_5d)
        pred = int(np.argmax(feat['logits'][0]))
        for key in all_feats:
            if key in feat:
                all_feats[key].append(feat[key][0])  # (T, H', W', C)
        all_labels.append(label)
        all_preds.append(pred)
        status = "OK" if pred == label else "WRONG"
        if i < 3 or (i + 1) % 10 == 0:
            shapes = {k: feat[k].shape for k in ['post_cssm', 'Q_spatial', 'V_spatial'] if k in feat}
            print(f"  [{i+1}/{len(samples)}] label={label} pred={pred} ({status})  {shapes}")

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"  Accuracy on PCA samples: {acc:.1%}")

    # Determine which states are available
    state_names = [k for k in ['Q_spatial', 'K_spatial', 'V_spatial'] if all_feats[k]]
    print(f"  Available states: {state_names}")

    # ===================================================================
    # Phase 2: Fit PCA and PLS on all feature types
    # ===================================================================
    print("\n=== Phase 2: Dimensionality reduction ===")

    pca_fits = {}
    pls_fits = {}

    print("Post-CSSM features:")
    pca_fits['post_cssm'] = fit_pca(all_feats['post_cssm'], n_components=args.k_show,
                                     label='post-CSSM ')
    pls_fits['post_cssm'] = fit_pls(all_feats['post_cssm'], all_labels,
                                     n_components=args.k_show, label='post-CSSM ')

    for sname in state_names:
        short = sname.replace('_spatial', '')
        print(f"{short}-state features (internal CSSM recurrence):")
        pca_fits[sname] = fit_pca(all_feats[sname], n_components=args.k_show,
                                   label=f'{short}-state ')
        pls_fits[sname] = fit_pls(all_feats[sname], all_labels,
                                   n_components=args.k_show, label=f'{short}-state ')

    # ===================================================================
    # Phase 2b: Train temporal SAEs on all feature types
    # ===================================================================
    print("\n=== Phase 2b: Sparse Autoencoders (temporal) ===")
    sae_fits = {}

    print("Post-CSSM SAE:")
    sae_fits['post_cssm'] = train_sae(all_feats['post_cssm'], label='post-CSSM ')

    for sname in state_names:
        short = sname.replace('_spatial', '')
        print(f"{short}-state SAE:")
        sae_fits[sname] = train_sae(all_feats[sname], label=f'{short}-state ')

    # ===================================================================
    # Phase 3 & 4: Per-sample visualization + gradients + SAE
    # ===================================================================
    vis_pos = [(s, l, p) for s, l, p in zip(
        [s[0] for s in samples], all_labels, all_preds) if l == 1][:args.num_vis_samples]
    vis_neg = [(s, l, p) for s, l, p in zip(
        [s[0] for s in samples], all_labels, all_preds) if l == 0][:args.num_vis_samples]
    vis_samples = vis_pos + vis_neg
    print(f"\n=== Phase 3 & 4: Visualization ({len(vis_samples)} samples) ===")

    for i, (img_video, label, pred) in enumerate(vis_samples):
        print(f"\n  Sample {i+1}/{len(vis_samples)}  "
              f"(label={label}, pred={pred}, {'OK' if pred==label else 'WRONG'})")
        x_5d = jnp.array(img_video)[None, ...]
        display_img = denormalize(img_video[0])

        feat = extract_features(model, variables, x_5d)

        # Logit lens: apply readout at each timestep → per-pixel class evidence
        spatial_logits = np.array(model.apply(
            variables, x_5d, training=False, return_spatial=True))  # (1, T, H', W', 2)
        gif = make_logit_lens_gif(
            display_img, spatial_logits[0], label, pred, i,
            args.output_dir, prefix='logit_lens',
            frame_duration=args.frame_duration, image_size=args.image_size)
        print(f"    Logit lens → {gif}")

        # Post-CSSM PCA + PLS
        post_feat = feat['post_cssm'][0]
        pca_proj = project_pca(pca_fits['post_cssm'], post_feat)
        pls_proj = project_pls(pls_fits['post_cssm'], post_feat)
        gif = make_pca_pls_gif(
            display_img, pca_proj, pls_proj, label, pred, i,
            args.output_dir, prefix='post_cssm',
            row_labels=('PCA', 'PLS'), k_show=args.k_show,
            frame_duration=args.frame_duration, image_size=args.image_size)
        print(f"    Post-CSSM PCA+PLS → {gif}")

        # Post-CSSM SAE
        sae_p, sae_s, sae_f = sae_fits['post_cssm']
        codes = encode_sae(sae_p, sae_f, post_feat, sae_s)
        gif = make_sae_gif(
            display_img, codes, sae_p, sae_s, label, pred, i,
            args.output_dir, prefix='sae_post_cssm',
            frame_duration=args.frame_duration, image_size=args.image_size)
        print(f"    Post-CSSM SAE → {gif}")

        # Q, K, V state PCA + PLS + SAE
        for sname in state_names:
            short = sname.replace('_spatial', '')
            s_feat = feat[sname][0]
            spca = project_pca(pca_fits[sname], s_feat)
            spls = project_pls(pls_fits[sname], s_feat)
            gif = make_pca_pls_gif(
                display_img, spca, spls, label, pred, i,
                args.output_dir, prefix=f'{short}_state',
                row_labels=(f'{short}-PCA', f'{short}-PLS'),
                k_show=args.k_show,
                frame_duration=args.frame_duration, image_size=args.image_size)
            print(f"    {short}-state PCA+PLS → {gif}")

            # SAE
            sae_p, sae_s, sae_f = sae_fits[sname]
            codes = encode_sae(sae_p, sae_f, s_feat, sae_s)
            gif = make_sae_gif(
                display_img, codes, sae_p, sae_s, label, pred, i,
                args.output_dir, prefix=f'sae_{short}_state',
                frame_duration=args.frame_duration, image_size=args.image_size)
            print(f"    {short}-state SAE → {gif}")

        # d(logit)/d(stem) — gradient through full recurrence
        print("    Computing d(logit)/d(stem)...")
        g_stem = grad_wrt_stem(model, variables, x_5d, pred)
        gif = make_gradient_gif(
            display_img, g_stem, label, pred, i,
            args.output_dir, prefix='grad_stem',
            k_show=args.k_show, frame_duration=args.frame_duration,
            image_size=args.image_size)
        print(f"    d/d(stem) → {gif}")

        # d(logit)/d(post_cssm) — gradient through readout only
        print("    Computing d(logit)/d(post_cssm)...")
        g_post = grad_wrt_post_cssm(model, variables, x_5d, pred)
        gif = make_gradient_gif(
            display_img, g_post, label, pred, i,
            args.output_dir, prefix='grad_post_cssm',
            k_show=args.k_show, frame_duration=args.frame_duration,
            image_size=args.image_size)
        print(f"    d/d(post_cssm) → {gif}")

    print(f"\nDone. Output: {args.output_dir}/")


if __name__ == '__main__':
    main()
