"""Bisect the tog9av97 -> current regression.

Compares 4 model variants on identical init + identical (random) batch:
  A: bias=0, pos_conv=OFF   (matches tog9av97 75% run init)
  B: bias=1, pos_conv=OFF   (isolates gate_proj bias change)
  C: bias=0, pos_conv=ON    (isolates pos_conv placement change)
  D: bias=1, pos_conv=ON    (current code)

Metrics per variant (identical PRNG key for init, identical data):
  - init loss (CE on uniform labels)
  - grad norm (flat L2 over params)
  - loss after N SGD steps at lr=1e-3 on a fixed random batch
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import linen as nn

from src.models.cssm_shvit import cssm_shvit_s1

NUM_CLASSES = 1000
BATCH = 64
H = W = 224
N_STEPS = 100
LR = 1e-3
SEED = 0
# Match real training
WEIGHT_DECAY = 0.05
USE_ADAMW = True


def make_model(bias_init: float, use_pos_conv: bool, T: int = 1):
    return cssm_shvit_s1(
        num_classes=NUM_CLASSES,
        num_timesteps=T,
        cssm_type='gdn',
        delta_key_dim=2,
        output_norm='rms',
        gate_type='factored',
        block_norm='global_layer',
        kernel_sizes=(1, 1, 1, 1),
        rope_mode='spatiotemporal',
        use_pos_conv=use_pos_conv,
        gate_proj_bias_init=bias_init,
    )


def run_variant(name, bias_init, use_pos_conv, x, y, T):
    model = make_model(bias_init, use_pos_conv, T=T)
    key = jax.random.PRNGKey(SEED)
    init_vars = model.init({'params': key, 'dropout': jax.random.PRNGKey(SEED + 1)},
                            x, training=True)
    params = init_vars['params']
    batch_stats = init_vars.get('batch_stats', {})
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

    drop_key = jax.random.PRNGKey(SEED + 1)

    def loss_fn(p, bs, xx, yy):
        out, new_state = model.apply(
            {'params': p, 'batch_stats': bs}, xx, training=True,
            mutable=['batch_stats'], rngs={'dropout': drop_key},
        )
        logits = out
        loss = optax.softmax_cross_entropy(
            logits, jax.nn.one_hot(yy, NUM_CLASSES)
        ).mean()
        return loss, (logits, new_state['batch_stats'])

    # Initial forward: loss + per-block output stats
    (loss0, (logits0, new_bs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, batch_stats, x, y
    )
    gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))

    # Capture activation stats on the classifier head input
    logits_std = float(jnp.std(logits0))
    logits_max = float(jnp.max(jnp.abs(logits0)))

    # Short training loop — matches real training optimizer (AdamW + WD)
    if USE_ADAMW:
        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=LR, weight_decay=WEIGHT_DECAY, b1=0.9, b2=0.999, eps=1e-6),
        )
    else:
        opt = optax.sgd(learning_rate=LR, momentum=0.9)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, bs, ostate, xx, yy):
        (l, (_, new_bs)), g = jax.value_and_grad(loss_fn, has_aux=True)(p, bs, xx, yy)
        updates, new_ostate = opt.update(g, ostate, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_bs, new_ostate, l

    losses = [float(loss0)]
    t0 = time.time()
    for i in range(N_STEPS):
        params, batch_stats, opt_state, l = step(params, batch_stats, opt_state, x, y)
        losses.append(float(l))
    dt = time.time() - t0

    print(f"\n=== {name} (bias_init={bias_init}, use_pos_conv={use_pos_conv}, T={T}) ===")
    print(f"  params           : {n_params:>11,}")
    print(f"  init loss        : {losses[0]:.4f}  (uniform = ln(1000) = 6.908)")
    print(f"  grad L2 at init  : {float(gnorm):.4e}")
    print(f"  logits std/max   : {logits_std:.3e} / {logits_max:.3e}")
    print(f"  loss @ step 1    : {losses[1]:.4f}")
    print(f"  loss @ step 20   : {losses[20]:.4f}")
    print(f"  loss @ step 50   : {losses[50]:.4f}")
    print(f"  loss @ step 100  : {losses[-1]:.4f}")
    print(f"  wall time        : {dt:.2f}s")

    return losses


def main():
    print(f"JAX platform: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    # Run both T=1 and T=8 so we can see T-dependent effects
    for T in (1, 8):
        print(f"\n{'#' * 70}\n# T={T}\n{'#' * 70}")
        # Fixed random batch — same across all 4 variants at this T
        key = jax.random.PRNGKey(SEED + T)
        x = jax.random.normal(key, (BATCH, H, W, 3), dtype=jnp.float32)
        y = jax.random.randint(key, (BATCH,), 0, NUM_CLASSES)

        results = {}
        results['A_bias0_pos0'] = run_variant('A_bias0_pos0', 0.0, False, x, y, T)
        results['B_bias1_pos0'] = run_variant('B_bias1_pos0', 1.0, False, x, y, T)
        results['C_bias0_pos1'] = run_variant('C_bias0_pos1', 0.0, True,  x, y, T)
        results['D_bias1_pos1'] = run_variant('D_bias1_pos1', 1.0, True,  x, y, T)

        # Summary table
        print(f"\n--- Summary T={T} (loss @ step N, lower = faster learning) ---")
        print(f"{'variant':18s} {'step 0':>8s} {'step 10':>8s} {'step 25':>8s} {'step 50':>8s}")
        for k, v in results.items():
            print(f"{k:18s} {v[0]:>8.3f} {v[10]:>8.3f} {v[25]:>8.3f} {v[-1]:>8.3f}")


if __name__ == '__main__':
    main()
