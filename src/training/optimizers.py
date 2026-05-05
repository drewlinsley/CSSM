"""
Optimizer factory for JAX/Flax training.

Supports:
- AdamW (default, good for most cases)
- LAMB (used by DeiT III for large-batch training)
- SGD with momentum (TIMM default)

Uses official optax implementations.
"""

import jax
import optax
from typing import Callable, Tuple


def make_weight_decay_mask(params, excluded_names: Tuple[str, ...]):
    """Return a same-shape boolean pytree: True where weight decay applies,
    False under any subtree whose path includes a name in `excluded_names`.
    """
    def _walk(node, path):
        if isinstance(node, dict):
            return {k: _walk(v, path + (k,)) for k, v in node.items()}
        # leaf
        excluded = any(p in excluded_names for p in path)
        return not excluded

    return _walk(params, ())


def make_param_labels(params, probe_names: Tuple[str, ...]):
    """Return a same-shape pytree of string labels: 'probe' under any subtree
    whose path includes a name in `probe_names`, 'backbone' elsewhere.

    Used as the label function for optax.multi_transform when applying a
    different (constant) LR to the linear probe head.
    """
    def _walk(node, path):
        if isinstance(node, dict):
            return {k: _walk(v, path + (k,)) for k, v in node.items()}
        return 'probe' if any(p in probe_names for p in path) else 'backbone'

    return _walk(params, ())


def create_optimizer(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.05,
    total_steps: int = 1000,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
    grad_clip_mode: str = 'norm',
    agc_clip_factor: float = 0.02,
    min_lr: float = None,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    no_decay_names: Tuple[str, ...] = (),
    probe_lr: float = None,
    probe_names: Tuple[str, ...] = ('linear_probe_head',),
) -> optax.GradientTransformation:
    """
    Factory function to create optimizers with learning rate schedule.

    Args:
        optimizer_name: 'adamw', 'lamb', or 'sgd'
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        total_steps: Total training steps
        warmup_steps: Warmup steps
        grad_clip: Gradient clipping norm
        b1: Beta1 for Adam/LAMB (default: 0.9)
        b2: Beta2 for Adam/LAMB (default: 0.999)
        eps: Epsilon for numerical stability

    Returns:
        optax.GradientTransformation

    Example DeiT III settings:
        create_optimizer('lamb', lr=3e-3, weight_decay=0.05, ...)

    Example TIMM settings:
        create_optimizer('adamw', lr=1e-3, weight_decay=0.05, ...)
    """
    # Learning rate schedule: warmup + cosine decay
    end_value = min_lr if min_lr is not None else learning_rate * 0.01
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=end_value,
    )

    # Gradient clipping
    if grad_clip_mode == 'agc':
        clip_fn = optax.adaptive_grad_clip(clipping=agc_clip_factor)
    else:
        clip_fn = optax.clip_by_global_norm(grad_clip)

    if optimizer_name == 'adamw':
        # AdamW: Adam with decoupled weight decay
        # Good default for most vision transformers.
        # If no_decay_names is provided, build a weight_decay mask that
        # excludes those subtrees (used to keep classification heads from
        # silently decaying to zero when they receive no gradient signal).
        if no_decay_names:
            wd_mask_fn = lambda params: make_weight_decay_mask(params, no_decay_names)
        else:
            wd_mask_fn = None

        backbone_adamw = optax.adamw(
            learning_rate=schedule,
            weight_decay=weight_decay,
            b1=b1, b2=b2, eps=eps,
            mask=wd_mask_fn,
        )

        if probe_lr is not None:
            # Multi-transform: constant LR for the linear_probe_head (no warmup),
            # standard cosine schedule for everything else. Both optimizers run
            # under the same gradient clip applied first.
            probe_adamw = optax.adamw(
                learning_rate=probe_lr,
                weight_decay=0.0,
                b1=b1, b2=b2, eps=eps,
            )
            label_fn = lambda params: make_param_labels(params, probe_names)
            return optax.chain(
                clip_fn,
                optax.multi_transform(
                    {'backbone': backbone_adamw, 'probe': probe_adamw},
                    param_labels=label_fn,
                ),
            )

        return optax.chain(clip_fn, backbone_adamw)

    elif optimizer_name == 'lamb':
        # LAMB: Layer-wise Adaptive Moments for Batch training
        # Used by DeiT III for large-batch training with high LR
        # Reference: https://arxiv.org/abs/1904.00962
        return optax.chain(
            clip_fn,
            optax.lamb(
                learning_rate=schedule,
                weight_decay=weight_decay,
                b1=b1,
                b2=b2,
                eps=eps,
            ),
        )

    elif optimizer_name == 'sgd':
        # SGD with momentum and Nesterov
        # TIMM's default optimizer
        return optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.sgd(learning_rate=schedule, momentum=0.9, nesterov=True),
            optax.add_decayed_weights(weight_decay),
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from: adamw, lamb, sgd")


def create_deit3_optimizer(
    learning_rate: float = 3e-3,
    weight_decay: float = 0.05,
    total_steps: int = 1000,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
) -> optax.GradientTransformation:
    """
    Create optimizer with DeiT III settings.

    DeiT III uses LAMB optimizer with:
    - Learning rate: 3e-3 to 4e-3
    - Weight decay: 0.05
    - 800 epochs
    - Warmup: 5 epochs

    Reference: https://arxiv.org/abs/2204.07118
    """
    return create_optimizer(
        optimizer_name='lamb',
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        grad_clip=grad_clip,
        b1=0.9,
        b2=0.999,
    )


def create_timm_optimizer(
    learning_rate: float = 1e-3,
    weight_decay: float = 0.05,
    total_steps: int = 1000,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
) -> optax.GradientTransformation:
    """
    Create optimizer with TIMM-style settings.

    TIMM typically uses AdamW with:
    - Learning rate: 1e-3
    - Weight decay: 0.05
    - 300 epochs
    - Warmup: 5 epochs
    """
    return create_optimizer(
        optimizer_name='adamw',
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        grad_clip=grad_clip,
        b1=0.9,
        b2=0.999,
    )
