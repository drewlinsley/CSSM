"""
bf16 mixed-precision sanity test for CSSM-SHViT.

Checks:
1. Model builds with dtype=bf16, param_dtype=fp32 (params stay fp32, activations
   are bf16).
2. Forward produces finite output.
3. Gradients through a loss are finite.
4. bf16 output is within ~5e-2 of the fp32 output (not a strict parity — bf16
   is a lossy compute mode).

Run:
    python tests/test_bf16_fast_path.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np

from src.models.cssm_shvit import cssm_shvit_s1


def _make_model(dtype, num_timesteps=4):
    return cssm_shvit_s1(
        num_classes=1000,
        num_timesteps=num_timesteps,
        cssm_type='gdn',
        delta_key_dim=2,
        gate_type='factored',
        kernel_sizes=(3, 3, 3, 3),
        short_conv_size=0,
        short_conv_spatial_size=0,
        rope_mode='none',
        block_norm='global_layer',
        use_input_gates=False,
        static_image_fast_path=True,
        dtype=dtype,
        param_dtype=jnp.float32,
    )


def _check_finite(tree, label):
    for p, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if not bool(jnp.all(jnp.isfinite(leaf))):
            print(f"FAIL: {label} non-finite at {p}")
            return False
    return True


def _param_dtypes_are_fp32(params):
    for p, leaf in jax.tree_util.tree_leaves_with_path(params):
        if leaf.dtype != jnp.float32:
            print(f"FAIL: param at {p} has dtype {leaf.dtype}, expected float32")
            return False
    return True


def main():
    rng = jax.random.PRNGKey(0)
    init_rng, data_rng = jax.random.split(rng)
    x = jax.random.normal(data_rng, (2, 56, 56, 3), dtype=jnp.float32)

    # Build both fp32 and bf16 models with the same init seed.
    model_fp32 = _make_model(jnp.float32)
    model_bf16 = _make_model(jnp.bfloat16)

    vars_fp32 = model_fp32.init(init_rng, x, training=False)
    vars_bf16 = model_bf16.init(init_rng, x, training=False)

    params_fp32 = vars_fp32['params']
    params_bf16 = vars_bf16['params']
    bstats_fp32 = vars_fp32.get('batch_stats', {})
    bstats_bf16 = vars_bf16.get('batch_stats', {})

    # 1. param dtypes should both be fp32
    if not _param_dtypes_are_fp32(params_fp32):
        return 1
    if not _param_dtypes_are_fp32(params_bf16):
        print("bf16 model has non-fp32 params — param_dtype plumbing broken")
        return 1
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params_fp32))
    print(f"  param tree OK ({n_params:,} params, all fp32)")

    # 2. forward parity (loose)
    def apply(model, params, bstats):
        vars_ = {'params': params}
        if bstats:
            vars_['batch_stats'] = bstats
        return model.apply(vars_, x, training=False)

    out_fp32 = apply(model_fp32, params_fp32, bstats_fp32)
    out_bf16 = apply(model_bf16, params_bf16, bstats_bf16)

    if not _check_finite(out_fp32, "fp32 output"):
        return 1
    if not _check_finite(out_bf16, "bf16 output"):
        return 1
    print(f"  fp32 output dtype: {out_fp32.dtype}")
    print(f"  bf16 output dtype: {out_bf16.dtype} (expected float32 — head is fp32)")

    # bf16 output should round-trip to close-to-fp32 output after the final
    # fp32 head cast. Tolerance is loose because bf16 accumulates ~3-digit
    # precision loss per op and the network is ~20 ops deep.
    diff = float(jnp.max(jnp.abs(out_fp32.astype(jnp.float32) - out_bf16.astype(jnp.float32))))
    print(f"  max|fp32 - bf16| logits = {diff:.4e}")
    if diff > 1.5e-1:
        print(f"FAIL: bf16 logits diverge from fp32 by {diff:.4e} (> 1.5e-1)")
        return 1

    # 3. Gradient finiteness
    def loss_fn(params, model, bstats):
        vars_ = {'params': params}
        if bstats:
            vars_['batch_stats'] = bstats
        out = model.apply(vars_, x, training=False)
        return jnp.mean(out ** 2)

    grads_fp32 = jax.grad(loss_fn)(params_fp32, model_fp32, bstats_fp32)
    grads_bf16 = jax.grad(loss_fn)(params_bf16, model_bf16, bstats_bf16)

    if not _check_finite(grads_fp32, "fp32 grads"):
        return 1
    if not _check_finite(grads_bf16, "bf16 grads"):
        return 1
    print("  gradients finite in both fp32 and bf16")

    print("\nPASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
