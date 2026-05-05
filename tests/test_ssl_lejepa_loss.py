"""
Tests for recurrent-LeJEPA SSL on CSSM-SHViT: SIGReg sanity + pairing parity,
un-normalized sow, stop-grad probe.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.cssm_shvit import cssm_shvit_s1
from src.training.distributed import (
    _collect_sown,
    _epps_pulley_gaussian_loss,
    _pairwise_predictive,
    _sigreg,
    _ssl_lejepa_loss,
)


def _build_model(num_timesteps=4, ssl_proj_dim=64, linear_probe=True):
    return cssm_shvit_s1(
        num_classes=10,
        num_timesteps=num_timesteps,
        cssm_type='gdn',
        delta_key_dim=2,
        gate_type='factored',
        kernel_sizes=(3, 3, 3, 3),
        short_conv_size=0,
        short_conv_spatial_size=0,
        block_norm='global_layer',
        rope_mode='none',
        use_input_gates=False,
        static_image_fast_path=False,
        ssl_proj_dim=ssl_proj_dim,
        linear_probe=linear_probe,
    )


def test_sigreg_zero_on_gaussian():
    """SIGReg should be near zero when embeddings are exact N(0, I)."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, (4096, 64))
    loss = float(_sigreg(z, num_slices=512, num_points=17, key=k2))
    assert loss < 5e-3, f'SIGReg on N(0,I) should be near zero, got {loss}'


def test_sigreg_large_on_collapsed():
    """SIGReg on collapsed z=0 should be substantially larger than on Gaussian."""
    key = jax.random.PRNGKey(1)
    k_gauss, k1, k2 = jax.random.split(key, 3)
    z_gauss = jax.random.normal(k_gauss, (4096, 64))
    z_zero = jnp.zeros((4096, 64))
    gauss_loss = float(_sigreg(z_gauss, num_slices=512, num_points=17, key=k1))
    zero_loss = float(_sigreg(z_zero, num_slices=512, num_points=17, key=k2))
    assert zero_loss > 100 * gauss_loss, (
        f'SIGReg should distinguish collapse (zero={zero_loss:.4f}) from '
        f'Gaussian ({gauss_loss:.5f})')


def test_epps_pulley_flags_nongaussian_1d():
    """The 1D Epps-Pulley loss should be small on N(0,1), large on uniform.

    (Avoids the CLT problem where high-dim non-Gaussian projects to near-Gaussian:
    SIGReg via slicing may NOT flag high-dim Rademacher, but the 1D core test must
    flag clearly non-Gaussian 1D samples.)
    """
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    z_gauss = jax.random.normal(k1, (8192,))
    # Uniform(-√3, √3) has var=1 but is bounded / non-Gaussian.
    z_unif = (jax.random.uniform(k2, (8192,)) - 0.5) * 2 * math.sqrt(3.0)
    gauss_loss = float(_epps_pulley_gaussian_loss(z_gauss, num_points=17))
    unif_loss = float(_epps_pulley_gaussian_loss(z_unif, num_points=17))
    assert unif_loss > 20 * gauss_loss, (
        f'Epps-Pulley should flag uniform ({unif_loss:.5f}) vs Gaussian '
        f'({gauss_loss:.5f})')


def test_pairwise_successive_parity():
    """For z[:, t] = t·v, successive distances = ‖v‖² each, averaged over T-1."""
    key = jax.random.PRNGKey(3)
    B, T, P = 2, 8, 16
    v = jax.random.normal(key, (B, P))
    t_idx = jnp.arange(T, dtype=jnp.float32)
    z = t_idx[None, :, None] * v[:, None, :]                   # (B, T, P)
    loss = float(_pairwise_predictive(z, 'successive'))
    expected = float((v ** 2).sum(-1).mean())
    assert math.isclose(loss, expected, rel_tol=1e-5), (loss, expected)


def test_pairwise_all_parity():
    """For z[:, t] = t·v, all-pair avg dist² = mean_{i<j}((j-i)²) · ‖v‖²."""
    key = jax.random.PRNGKey(4)
    B, T, P = 2, 8, 16
    v = jax.random.normal(key, (B, P))
    t_idx = jnp.arange(T, dtype=jnp.float32)
    z = t_idx[None, :, None] * v[:, None, :]
    loss = float(_pairwise_predictive(z, 'all'))
    diffs_sq = np.array([(j - i) ** 2 for i in range(T) for j in range(i + 1, T)])
    expected = float(diffs_sq.mean() * (v ** 2).sum(-1).mean())
    assert math.isclose(loss, expected, rel_tol=1e-5), (loss, expected)


def test_pairwise_to_mean_parity():
    """to_mean = mean_t ‖z_t - z̄‖². For z[:, t] = t·v: variance over T of t, times ‖v‖²."""
    key = jax.random.PRNGKey(5)
    B, T, P = 2, 8, 16
    v = jax.random.normal(key, (B, P))
    t_idx = jnp.arange(T, dtype=jnp.float32)
    z = t_idx[None, :, None] * v[:, None, :]
    loss = float(_pairwise_predictive(z, 'to_mean'))
    t_var = float(t_idx.var())
    expected = float(t_var * (v ** 2).sum(-1).mean())
    assert math.isclose(loss, expected, rel_tol=1e-5), (loss, expected)


def test_sown_is_unnormalized():
    """Sowing must return un-normalized (B, T, P) — SIGReg depends on this."""
    model = _build_model(num_timesteps=4, ssl_proj_dim=32)
    key = jax.random.PRNGKey(6)
    x = jax.random.normal(key, (2, 224, 224, 3))
    # Coerce to 5D (video) so the sow site runs.
    x = jnp.broadcast_to(x[:, None], (2, 4, 224, 224, 3))
    params = model.init(key, x, training=False)
    out, state = model.apply(params, x, training=False, mutable=['intermediates'])
    zs = _collect_sown(state['intermediates'], 'cssm_temporal_proj')
    assert len(zs) == 3, f'expected 3 CSSM blocks for s1, got {len(zs)}'
    for z in zs:
        assert z.shape == (2, 4, 32), z.shape
        norms = jnp.linalg.norm(z, axis=-1)
        std = float(norms.std())
        # L2-normalized would give std ≈ 0; un-normalized features have per-sample
        # norm variation from the Dense output.
        assert std > 1e-3, f'sown z looks L2-normalized (norms std={std:.5f})'


def test_stop_grad_probe():
    """Gradient of probe_ce wrt backbone params must be zero."""
    model = _build_model(num_timesteps=4, ssl_proj_dim=32)
    key = jax.random.PRNGKey(7)
    x = jnp.broadcast_to(
        jax.random.normal(key, (2, 224, 224, 3))[:, None],
        (2, 4, 224, 224, 3))
    params = model.init(key, x, training=False)
    labels = jax.nn.one_hot(jnp.array([0, 1]), 10)

    def probe_ce_fn(p):
        logits, probe_logits = model.apply(p, x, training=False)
        log_probs = jax.nn.log_softmax(probe_logits, axis=-1)
        return -(labels * log_probs).sum(-1).mean()

    grads = jax.grad(probe_ce_fn)(params)
    # Probe head itself should have nonzero grad; everything else zero.
    probe_grad = grads['params']['linear_probe_head']['kernel']
    assert float(jnp.abs(probe_grad).sum()) > 0, 'probe head has no gradient'

    def max_abs(tree):
        return jax.tree_util.tree_reduce(
            lambda a, b: jnp.maximum(a, jnp.abs(b).max()),
            tree, jnp.float32(0.0))

    backbone = {k: v for k, v in grads['params'].items() if k != 'linear_probe_head'}
    assert float(max_abs(backbone)) == 0.0, 'stop-grad leak into backbone'


def test_ssl_lejepa_loss_shapes():
    """End-to-end: _ssl_lejepa_loss returns three scalars."""
    key = jax.random.PRNGKey(8)
    z1 = jax.random.normal(jax.random.fold_in(key, 0), (4, 8, 16))
    z2 = jax.random.normal(jax.random.fold_in(key, 1), (4, 8, 16))
    pred, sig, min_var = _ssl_lejepa_loss(
        [z1, z2], pair_mode='all',
        num_slices=64, num_points=17, rng=key)
    assert pred.shape == ()
    assert sig.shape == ()
    assert min_var.shape == ()
    assert float(pred) > 0
    assert float(sig) > 0
    assert float(min_var) > 0
