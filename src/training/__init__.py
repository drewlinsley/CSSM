"""Training utilities for multi-GPU training."""

from .distributed import (
    replicate_state,
    unreplicate_state,
    shard_batch,
    create_parallel_train_step,
    create_parallel_eval_step,
    split_rng_for_devices,
    get_global_batch_size,
)

__all__ = [
    'replicate_state',
    'unreplicate_state',
    'shard_batch',
    'create_parallel_train_step',
    'create_parallel_eval_step',
    'split_rng_for_devices',
    'get_global_batch_size',
]
