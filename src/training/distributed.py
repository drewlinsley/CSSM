"""
Multi-GPU training utilities for JAX.

Uses pmap for data parallelism with proper gradient synchronization.
Designed for H200 multi-GPU training.
"""

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training import train_state
from typing import Callable, Any, Tuple
import optax


def _collect_sown(tree, name: str):
    """Walk a Flax intermediates dict and collect every leaf whose key matches
    `name`. Each `nn.Module.sow(..., name, value)` returns a 1-tuple per call
    site (default mode='append'); we unwrap to the underlying value.

    Returns a Python list whose length is fixed at trace time (= number of
    sow call sites in the module hierarchy).
    """
    out = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == name:
                    if isinstance(v, tuple):
                        out.append(v[0])
                    else:
                        out.append(v)
                else:
                    walk(v)

    walk(tree)
    return out


def _epps_pulley_gaussian_loss(slices_1d, num_points: int = 17):
    """Epps-Pulley characteristic-function test vs. N(0, 1).

    slices_1d: (N,) — 1D projection of embeddings.
    Returns 0 in the infinite-N limit iff samples ~ N(0, 1). Penalizes deviations
    of both moments (mean, variance) and shape (skew, kurt) simultaneously.
    """
    t = jnp.linspace(-3.0, 3.0, num_points)                    # (K,)
    ut = slices_1d[:, None] * t[None, :]                       # (N, K)
    ecf_real = jnp.cos(ut).mean(axis=0)                        # (K,)
    ecf_imag = jnp.sin(ut).mean(axis=0)
    cf_target_real = jnp.exp(-0.5 * t ** 2)                    # Gaussian CF is real
    return jnp.mean((ecf_real - cf_target_real) ** 2 + ecf_imag ** 2)


def _sigreg(z_flat, num_slices: int, num_points: int, key):
    """Sketched Isotropic Gaussian Regularization.

    z_flat: (N, P) — un-normalized embeddings (e.g. (B*T, P) pooled across the
    timestep axis). Projects to `num_slices` random unit directions and applies
    the Epps-Pulley test on each, averaging. Linear time/memory in N.

    NOTE: no per-slice standardization. We want SIGReg to see collapse at z=0
    (would produce ecf=1 for all t, large CF mismatch) and non-unit variance;
    standardizing would mask both.
    """
    N, P = z_flat.shape
    u = jax.random.normal(key, (num_slices, P))                # (S, P)
    u = u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + 1e-6)
    proj = z_flat @ u.T                                        # (N, S)
    return jax.vmap(lambda s: _epps_pulley_gaussian_loss(s, num_points),
                    in_axes=1)(proj).mean()


def _pairwise_predictive(z, mode: str):
    """z: (B, T, P). Return scalar predictive loss over timestep pairs.
    mode ∈ {'all', 'successive', 'to_mean'}.
    """
    if mode == 'successive':
        d = jnp.sum((z[:, 1:] - z[:, :-1]) ** 2, axis=-1)      # (B, T-1)
        return d.mean()
    if mode == 'to_mean':
        zbar = z.mean(axis=1, keepdims=True)                   # (B, 1, P)
        return jnp.sum((z - zbar) ** 2, axis=-1).mean()
    # 'all': every pair, upper triangle (i<j), normalized by num pairs.
    T = z.shape[1]
    zi = z[:, :, None, :]                                      # (B, T, 1, P)
    zj = z[:, None, :, :]                                      # (B, 1, T, P)
    d = jnp.sum((zi - zj) ** 2, axis=-1)                       # (B, T, T)
    mask = jnp.triu(jnp.ones((T, T), dtype=d.dtype), k=1)      # (T, T)
    return (d * mask).sum(axis=(1, 2)).mean() / mask.sum()


def _ssl_lejepa_loss(block_zs, pair_mode: str, num_slices: int,
                     num_points: int, rng):
    """Recurrent LeJEPA loss: predictive pairing + SIGReg, per CSSM block.

    Returns (pred_loss, sigreg_loss, min_z_var). `min_z_var` is a collapse
    watchdog (smallest mean trajectory variance across blocks).
    """
    if len(block_zs) == 0:
        z = jnp.zeros(())
        return z, z, z

    pred_terms, sig_terms, var_terms = [], [], []
    keys = jax.random.split(rng, len(block_zs))
    for z, k in zip(block_zs, keys):                           # (B, T, P) each
        pred_terms.append(_pairwise_predictive(z, pair_mode))
        B, T, P = z.shape
        sig_terms.append(_sigreg(z.reshape(B * T, P), num_slices, num_points, k))
        var_terms.append(jnp.mean(jnp.var(z, axis=1)))
    pred = jnp.mean(jnp.stack(pred_terms))
    sig = jnp.mean(jnp.stack(sig_terms))
    min_var = jnp.min(jnp.stack(var_terms))
    return pred, sig, min_var


def replicate_state(state: train_state.TrainState) -> train_state.TrainState:
    """
    Replicate train state across all devices.

    Args:
        state: Single-device train state

    Returns:
        Replicated train state (one copy per device)
    """
    return jax_utils.replicate(state)


def unreplicate_state(state: train_state.TrainState) -> train_state.TrainState:
    """
    Get single-device state from replicated state.

    Args:
        state: Replicated train state

    Returns:
        Single-device train state (from first device)
    """
    return jax_utils.unreplicate(state)


def shard_batch(batch: Tuple, num_devices: int = None) -> Tuple:
    """
    Shard batch across devices.

    Reshapes (B, ...) to (num_devices, B // num_devices, ...).
    Batch size must be divisible by num_devices.

    Args:
        batch: Tuple of arrays (images, labels)
        num_devices: Number of devices (defaults to jax.device_count())

    Returns:
        Sharded batch with leading device dimension
    """
    if num_devices is None:
        num_devices = jax.device_count()

    def _shard(x):
        # Reshape first dimension to (devices, batch_per_device, ...)
        batch_size = x.shape[0]
        assert batch_size % num_devices == 0, \
            f"Batch size {batch_size} not divisible by num_devices {num_devices}"
        return x.reshape((num_devices, batch_size // num_devices) + x.shape[1:])

    return jax.tree_util.tree_map(_shard, batch)


def create_parallel_train_step(
    train_step_fn: Callable,
    axis_name: str = 'batch'
) -> Callable:
    """
    Create parallelized training step with gradient synchronization.

    The train_step_fn should internally use jax.lax.pmean for gradient sync.

    Args:
        train_step_fn: Training step function with signature
            (state, batch, rng, axis_name) -> (state, metrics)
        axis_name: Axis name for pmean gradient sync

    Returns:
        pmap-ed training step function
    """
    def parallel_train_step(state, batch, rng):
        return train_step_fn(state, batch, rng, axis_name=axis_name)

    return jax.pmap(
        parallel_train_step,
        axis_name=axis_name,
        donate_argnums=(0,)  # Donate state buffer for efficiency
    )


def create_parallel_eval_step(
    eval_step_fn: Callable,
    axis_name: str = 'batch'
) -> Callable:
    """
    Create parallelized evaluation step.

    Args:
        eval_step_fn: Evaluation step function
        axis_name: Axis name for pmean metric sync

    Returns:
        pmap-ed evaluation step function
    """
    return jax.pmap(eval_step_fn, axis_name=axis_name)


def split_rng_for_devices(rng: jax.Array) -> jax.Array:
    """
    Split RNG key for each device.

    Args:
        rng: Single RNG key

    Returns:
        Array of RNG keys, one per device
    """
    return jax.random.split(rng, jax.device_count())


def get_global_batch_size(local_batch_size: int) -> int:
    """
    Get effective global batch size across all devices.

    Args:
        local_batch_size: Batch size per device

    Returns:
        Total batch size across all devices
    """
    return local_batch_size * jax.device_count()


def make_train_step(model, num_classes: int):
    """
    Create a training step function with proper gradient synchronization.
    Supports models with BatchNorm (batch_stats collection).

    Args:
        model: Flax model
        num_classes: Number of output classes

    Returns:
        Training step function compatible with pmap
    """
    def train_step(state, batch, rng, batch_stats=None, axis_name='batch'):
        videos, labels = batch

        def loss_fn(params):
            variables = {'params': params}
            if batch_stats is not None:
                variables['batch_stats'] = batch_stats

            output = state.apply_fn(
                variables,
                videos,
                training=True,
                rngs={'dropout': rng},
                mutable=['batch_stats'] if batch_stats is not None else False,
            )

            if batch_stats is not None:
                logits, new_state = output
                new_batch_stats = new_state.get('batch_stats', None)
            else:
                logits = output
                new_batch_stats = None

            one_hot = jax.nn.one_hot(labels, num_classes)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss, (logits, new_batch_stats)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)

        # Synchronize gradients across devices
        grads = jax.lax.pmean(grads, axis_name=axis_name)
        loss = jax.lax.pmean(loss, axis_name=axis_name)

        # Sync batch_stats across devices
        if new_batch_stats is not None:
            new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name=axis_name)

        # Update state
        state = state.apply_gradients(grads=grads)

        # Compute accuracy
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        acc = jax.lax.pmean(acc, axis_name=axis_name)

        return state, {'loss': loss, 'acc': acc}, new_batch_stats

    return train_step


def make_eval_step(model, num_classes: int, linear_probe: bool = False):
    """
    Create an evaluation step function.
    Supports models with BatchNorm (batch_stats collection).

    When linear_probe=True, the model returns (logits, probe_logits) and the
    eval step computes both top1_main (frozen classifier) and top1_probe
    (rising signal from the SSL-trained backbone features).

    Args:
        model: Flax model
        num_classes: Number of output classes
        linear_probe: Whether the model returns a (logits, probe_logits) tuple

    Returns:
        Evaluation step function compatible with pmap
    """
    def eval_step(state, batch, batch_stats=None, axis_name='batch'):
        videos, labels = batch

        variables = {'params': state.params}
        if batch_stats is not None:
            variables['batch_stats'] = batch_stats

        out = state.apply_fn(
            variables,
            videos,
            training=False,
        )
        if linear_probe:
            logits, probe_logits = out
        else:
            logits = out
            probe_logits = None

        # Loss (main head)
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        loss = jax.lax.pmean(loss, axis_name=axis_name)

        # Accuracy (main head)
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        acc = jax.lax.pmean(acc, axis_name=axis_name)

        metrics = {'loss': loss, 'acc': acc}
        if linear_probe:
            probe_loss = optax.softmax_cross_entropy(probe_logits, one_hot).mean()
            probe_preds = jnp.argmax(probe_logits, axis=-1)
            probe_acc = jnp.mean(probe_preds == labels)
            metrics['probe_loss'] = jax.lax.pmean(probe_loss, axis_name=axis_name)
            metrics['probe_acc'] = jax.lax.pmean(probe_acc, axis_name=axis_name)

        return metrics

    return eval_step


def make_train_step_mixup(
    model,
    num_classes: int,
    label_smoothing: float = 0.0,
    ssl_temporal_loss: bool = False,
    ssl_loss_weight: float = 1.0,
    ssl_sigreg_weight: float = 1.0,
    ssl_pair_mode: str = 'all',
    ssl_num_slices: int = 1024,
    ssl_num_points: int = 17,
):
    """
    Create a training step function that supports soft labels (mixup/cutmix).

    When ssl_temporal_loss=True, the model returns (logits, probe_logits) and
    sows per-block 'cssm_temporal_proj' features. Loss is recurrent LeJEPA:
        total = ssl_loss_weight * pairwise_predictive(z)
              + ssl_sigreg_weight * SIGReg(z)
              + probe_ce
    No predictor, no stop-grad on backbone, no teacher. Probe head input is
    stop-grad so its CE never updates the backbone.
    """
    def train_step(state, batch, rng, batch_stats=None, axis_name='batch'):
        videos, soft_labels = batch  # soft_labels is (B, num_classes)

        def loss_fn(params):
            variables = {'params': params}
            if batch_stats is not None:
                variables['batch_stats'] = batch_stats

            mut = []
            if batch_stats is not None:
                mut.append('batch_stats')
            if ssl_temporal_loss:
                mut.append('intermediates')

            output = state.apply_fn(
                variables,
                videos,
                training=True,
                rngs={'dropout': rng},
                mutable=mut if mut else False,
            )

            if mut:
                model_out, new_state = output
                new_batch_stats = new_state.get('batch_stats', None)
                intermediates = new_state.get('intermediates', {})
            else:
                model_out = output
                new_batch_stats = None
                intermediates = {}

            if ssl_temporal_loss:
                logits, probe_logits = model_out
            else:
                logits = model_out
                probe_logits = None

            # Apply label smoothing to soft labels
            if label_smoothing > 0:
                soft_labels_smooth = soft_labels * (1.0 - label_smoothing) + label_smoothing / num_classes
            else:
                soft_labels_smooth = soft_labels

            if ssl_temporal_loss:
                # SSL-only mode: drop main CE, train backbone via LeJEPA only.
                block_zs = _collect_sown(intermediates, 'cssm_temporal_proj')
                lejepa_rng = jax.random.fold_in(rng, state.step)
                pred_loss, sig_loss, min_z_var = _ssl_lejepa_loss(
                    block_zs, pair_mode=ssl_pair_mode,
                    num_slices=ssl_num_slices, num_points=ssl_num_points,
                    rng=lejepa_rng,
                )

                # Probe CE flows only into linear_probe_head (stop-grad on input).
                probe_log_probs = jax.nn.log_softmax(probe_logits, axis=-1)
                probe_ce = -jnp.sum(soft_labels_smooth * probe_log_probs, axis=-1).mean()

                loss = (ssl_loss_weight * pred_loss
                        + ssl_sigreg_weight * sig_loss
                        + probe_ce)
                aux = (logits, probe_logits, new_batch_stats,
                       pred_loss, sig_loss, probe_ce, min_z_var)
            else:
                # Existing supervised path
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                loss = -jnp.sum(soft_labels_smooth * log_probs, axis=-1).mean()
                aux = (logits, None, new_batch_stats, None, None, None, None)

            return loss, aux

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)
        (logits, probe_logits, new_batch_stats,
         pred_loss, sig_loss, probe_ce, min_z_var) = aux

        # Synchronize gradients across devices
        grads = jax.lax.pmean(grads, axis_name=axis_name)
        loss = jax.lax.pmean(loss, axis_name=axis_name)

        # Sync batch_stats across devices
        if new_batch_stats is not None:
            new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name=axis_name)

        # Update state
        state = state.apply_gradients(grads=grads)

        # Compute accuracy (use argmax of soft labels as ground truth for mixup)
        true_labels = jnp.argmax(soft_labels, axis=-1)
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == true_labels)
        acc = jax.lax.pmean(acc, axis_name=axis_name)

        metrics = {'loss': loss, 'acc': acc}
        if ssl_temporal_loss:
            probe_preds = jnp.argmax(probe_logits, axis=-1)
            probe_acc = jnp.mean(probe_preds == true_labels)
            metrics['probe_acc'] = jax.lax.pmean(probe_acc, axis_name=axis_name)
            metrics['pred_loss'] = jax.lax.pmean(pred_loss, axis_name=axis_name)
            metrics['sigreg_loss'] = jax.lax.pmean(sig_loss, axis_name=axis_name)
            metrics['probe_ce'] = jax.lax.pmean(probe_ce, axis_name=axis_name)
            metrics['min_z_var'] = jax.lax.pmin(min_z_var, axis_name=axis_name)

        return state, metrics, new_batch_stats

    return train_step


def make_train_step_bce(model, num_classes: int):
    """
    Create a training step function using Binary Cross Entropy loss.

    DeiT III uses BCE instead of softmax cross-entropy, treating each
    class as an independent binary classification problem.

    Args:
        model: Flax model
        num_classes: Number of output classes

    Returns:
        Training step function compatible with pmap
    """
    def train_step(state, batch, rng, batch_stats=None, axis_name='batch'):
        videos, soft_labels = batch  # soft_labels is (B, num_classes)

        def loss_fn(params):
            variables = {'params': params}
            if batch_stats is not None:
                variables['batch_stats'] = batch_stats

            output = state.apply_fn(
                variables,
                videos,
                training=True,
                rngs={'dropout': rng},
                mutable=['batch_stats'] if batch_stats is not None else False,
            )

            if batch_stats is not None:
                logits, new_state = output
                new_batch_stats = new_state.get('batch_stats', None)
            else:
                logits = output
                new_batch_stats = None

            # Binary Cross Entropy with logits
            # BCE treats each class independently
            # loss = -y * log(sigmoid(x)) - (1-y) * log(1 - sigmoid(x))
            loss = optax.sigmoid_binary_cross_entropy(logits, soft_labels).mean()
            return loss, (logits, new_batch_stats)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)

        # Synchronize gradients across devices
        grads = jax.lax.pmean(grads, axis_name=axis_name)
        loss = jax.lax.pmean(loss, axis_name=axis_name)

        # Sync batch_stats across devices
        if new_batch_stats is not None:
            new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name=axis_name)

        # Update state
        state = state.apply_gradients(grads=grads)

        # Compute accuracy (use argmax of soft labels as ground truth)
        preds = jnp.argmax(logits, axis=-1)
        true_labels = jnp.argmax(soft_labels, axis=-1)
        acc = jnp.mean(preds == true_labels)
        acc = jax.lax.pmean(acc, axis_name=axis_name)

        return state, {'loss': loss, 'acc': acc}, new_batch_stats

    return train_step


def make_eval_step_with_params(model, num_classes: int):
    """
    Create an evaluation step function that takes explicit params.
    Useful for EMA evaluation where params differ from state.params.

    Args:
        model: Flax model
        num_classes: Number of output classes

    Returns:
        Evaluation step function compatible with pmap
    """
    def eval_step(params, batch, batch_stats=None, axis_name='batch', apply_fn=None):
        videos, labels = batch

        variables = {'params': params}
        if batch_stats is not None:
            variables['batch_stats'] = batch_stats

        logits = apply_fn(
            variables,
            videos,
            training=False,
        )

        # Loss
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        loss = jax.lax.pmean(loss, axis_name=axis_name)

        # Accuracy
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        acc = jax.lax.pmean(acc, axis_name=axis_name)

        return {'loss': loss, 'acc': acc}

    return eval_step
