"""
Tests for the temporal contrastive SSL loss on CSSM-SHViT.

These cover the model-side wiring (sown intermediates, tuple return, stop-grad
probe) and the loss-side math (collapse floor, smooth-trajectory zero, random
nonzero, variance regularizer sign), plus the structural-merge resume helper
and the weight-decay mask used to keep no-gradient heads from decaying.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.cssm_shvit import cssm_shvit_s1
from src.training.distributed import (
    _collect_sown,
    _ssl_triplet_and_var_loss,
)
from src.training.optimizers import (
    create_optimizer,
    make_weight_decay_mask,
    make_param_labels,
)


def _build_model(num_timesteps=8, ssl_proj_dim=64, linear_probe=True):
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


def _init(model, rng_seed=0):
    rng = jax.random.PRNGKey(rng_seed)
    dummy = jnp.ones((2, 32, 32, 3))
    return model.init({'params': rng, 'dropout': rng}, dummy, training=True)


def test_forward_returns_tuple():
    model = _build_model()
    variables = _init(model)
    out = model.apply(
        variables, jnp.ones((2, 32, 32, 3)), training=False
    )
    assert isinstance(out, tuple), "Expected (logits, probe_logits) tuple"
    logits, probe_logits = out
    assert logits.shape == (2, 10)
    assert probe_logits.shape == (2, 10)


def test_sown_intermediate_count_and_shape():
    """s1 has 3 CSSM blocks (depths=(1,2,2,1), use_cssm_stages=(F,F,T,T) ->
    2 in stage 2 + 1 in stage 3). Each contributes one sown projection."""
    model = _build_model(num_timesteps=8, ssl_proj_dim=64)
    variables = _init(model)
    rng = jax.random.PRNGKey(1)
    # Use random input — an all-ones constant image collapses through the
    # LayerNorms to zero activations, which then makes the L2-normalized
    # projection numerically meaningless (norm ≈ eps after the divide).
    images = jax.random.normal(jax.random.PRNGKey(2), (2, 32, 32, 3))
    # The model has BatchNorm in patch_embed → mutable must include batch_stats.
    out, mut = model.apply(
        variables, images,
        training=True, rngs={'dropout': rng},
        mutable=['intermediates', 'batch_stats'],
    )
    intermediates = mut['intermediates']
    block_zs = _collect_sown(intermediates, 'cssm_temporal_proj')
    assert len(block_zs) == 3, f"Expected 3 CSSM blocks, got {len(block_zs)}"
    for z in block_zs:
        assert z.shape == (2, 8, 64), f"Unexpected sown shape {z.shape}"
        # L2-normalized: per-row norm ≈ 1
        norms = jnp.linalg.norm(z, axis=-1)
        assert jnp.allclose(norms, jnp.ones_like(norms), atol=1e-3)


def test_collapse_floor():
    """Identical features across t -> loss = 2 * margin (the collapse floor)."""
    margin = 0.5
    z_collapsed = jnp.broadcast_to(
        jnp.array([[1.0, 0.0, 0.0]])[:, None, :], (4, 8, 3)
    )
    ssl_loss, var_loss, min_z_var = _ssl_triplet_and_var_loss(
        [z_collapsed], margin=margin
    )
    assert jnp.allclose(ssl_loss, 2.0 * margin), \
        f"Collapse floor expected {2.0 * margin}, got {ssl_loss}"
    assert jnp.allclose(var_loss, 0.0), \
        f"Collapsed features should give zero variance loss, got {var_loss}"
    assert jnp.allclose(min_z_var, 0.0)


def test_smooth_trajectory_beats_collapse():
    """A smooth trajectory on the unit sphere should always beat the collapse
    floor — it does not need to hit zero because the margin can still be active
    when adjacent distances are small, only that d_adj < d_skip + margin."""
    margin = 0.5
    # 8 evenly spaced points on a wide great-circle arc (pi radians) so that
    # d_skip > d_adj + margin and the hinge deactivates.
    angles = jnp.linspace(0.0, jnp.pi, 8)
    z = jnp.stack([jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)], axis=-1)
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    z = z[None].repeat(4, axis=0)  # (4, 8, 3)
    ssl_loss, _, _ = _ssl_triplet_and_var_loss([z], margin=margin)
    collapse_floor = 2.0 * margin
    assert ssl_loss < collapse_floor, \
        f"Smooth trajectory loss {ssl_loss} should beat collapse floor {collapse_floor}"


def test_random_loss_finite_and_nonzero():
    """Random L2-normalized features should give a finite, nonzero triplet loss
    (sanity: catches a regression where the loss is silently always 0)."""
    rng = np.random.default_rng(42)
    z = rng.standard_normal((4, 8, 16)).astype(np.float32)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    z = jnp.array(z)
    ssl_loss, var_loss, min_z_var = _ssl_triplet_and_var_loss([z], margin=0.5)
    assert jnp.isfinite(ssl_loss)
    assert ssl_loss > 0.0
    assert jnp.isfinite(var_loss)
    assert var_loss < 0.0  # variance is positive, var_loss = -mean(var)
    assert min_z_var > 0.0


def test_variance_regularizer_sign():
    """Collapsed -> var_loss = 0; spread -> var_loss < 0 (we negate so the
    optimizer minimizes by spreading)."""
    z_collapsed = jnp.zeros((2, 8, 4))
    _, var_collapsed, _ = _ssl_triplet_and_var_loss([z_collapsed], margin=0.1)
    z_spread = jnp.eye(8)[None, :, :].repeat(2, axis=0)  # (2, 8, 8)
    _, var_spread, _ = _ssl_triplet_and_var_loss([z_spread], margin=0.1)
    assert jnp.isclose(var_collapsed, 0.0)
    assert var_spread < 0.0


def test_stop_gradient_on_probe_head():
    """Gradient of probe CE w.r.t. backbone params must be exactly zero — the
    probe head sees stop_gradient(pooled) so its loss never updates the backbone.
    """
    model = _build_model()
    variables = _init(model)
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)
    images = jnp.ones((2, 32, 32, 3))
    labels = jnp.array([0, 1])

    def probe_only_loss(p):
        v = {'params': p}
        if batch_stats is not None:
            v['batch_stats'] = batch_stats
        out = model.apply(v, images, training=False)
        _, probe_logits = out
        one_hot = jax.nn.one_hot(labels, 10)
        log_probs = jax.nn.log_softmax(probe_logits, axis=-1)
        return -jnp.sum(one_hot * log_probs, axis=-1).mean()

    grads = jax.grad(probe_only_loss)(params)

    # Walk the gradient tree; everything except 'linear_probe_head/*' must be 0.
    def check(node, path=()):
        if isinstance(node, dict):
            for k, v in node.items():
                check(v, path + (k,))
            return
        in_probe = any(p == 'linear_probe_head' for p in path)
        if in_probe:
            return  # probe head SHOULD have nonzero gradients
        max_abs = float(jnp.max(jnp.abs(node)))
        assert max_abs == 0.0, \
            f"Backbone leaf at {'/'.join(path)} got nonzero grad {max_abs} from probe loss"

    check(grads)


def test_weight_decay_mask_excludes_heads():
    """The make_weight_decay_mask helper should return False under any subtree
    whose path contains 'head' or 'linear_probe_head'."""
    fake_params = {
        'head': {'kernel': jnp.ones((4, 4)), 'bias': jnp.zeros((4,))},
        'linear_probe_head': {'kernel': jnp.ones((4, 4))},
        'patch_embed': {'kernel': jnp.ones((4, 4))},
        'stage2_block0': {'cssm': {'qkv_proj': {'kernel': jnp.ones((4, 4))}}},
    }
    mask = make_weight_decay_mask(fake_params, ('head', 'linear_probe_head'))
    assert mask['head']['kernel'] is False
    assert mask['head']['bias'] is False
    assert mask['linear_probe_head']['kernel'] is False
    assert mask['patch_embed']['kernel'] is True
    assert mask['stage2_block0']['cssm']['qkv_proj']['kernel'] is True


def test_param_labels_routes_probe_head():
    fake_params = {
        'head': {'kernel': jnp.ones((4, 4))},
        'linear_probe_head': {'kernel': jnp.ones((4, 4))},
        'patch_embed': {'kernel': jnp.ones((4, 4))},
    }
    labels = make_param_labels(fake_params, ('linear_probe_head',))
    assert labels['head']['kernel'] == 'backbone'
    assert labels['linear_probe_head']['kernel'] == 'probe'
    assert labels['patch_embed']['kernel'] == 'backbone'


def test_structural_merge_resume_helper():
    """Replicates the merge logic in train_imagenet.py: take checkpoint values
    where keys exist, keep freshly-initialized values for new keys."""
    import flax
    target = {
        'head': {'kernel': jnp.zeros((4, 4))},
        'linear_probe_head': {'kernel': jnp.ones((4, 4)) * 9.0},  # fresh init
        'cssm_block': {
            'qkv_proj': {'kernel': jnp.zeros((4, 4))},
            'ssl_proj_in': {'kernel': jnp.ones((4, 4)) * 7.0},  # fresh init
        },
    }
    source = {
        'head': {'kernel': jnp.ones((4, 4)) * 3.0},
        'cssm_block': {
            'qkv_proj': {'kernel': jnp.ones((4, 4)) * 5.0},
        },
    }

    def _merge(target, source, path=()):
        if isinstance(target, (dict, flax.core.FrozenDict)):
            tgt = (flax.core.unfreeze(target)
                   if isinstance(target, flax.core.FrozenDict) else target)
            src = (flax.core.unfreeze(source)
                   if isinstance(source, flax.core.FrozenDict)
                   else source) if source is not None else {}
            out = {}
            for k, v in tgt.items():
                if k in src:
                    out[k] = _merge(v, src[k], path + (k,))
                else:
                    out[k] = v
            return out
        if hasattr(target, 'shape') and hasattr(source, 'shape'):
            if target.shape == source.shape:
                return source
            return target
        return source

    merged = _merge(target, source)
    # Backbone params restored from checkpoint
    assert jnp.allclose(merged['head']['kernel'], 3.0)
    assert jnp.allclose(merged['cssm_block']['qkv_proj']['kernel'], 5.0)
    # New SSL/probe params kept at fresh init
    assert jnp.allclose(merged['linear_probe_head']['kernel'], 9.0)
    assert jnp.allclose(merged['cssm_block']['ssl_proj_in']['kernel'], 7.0)
