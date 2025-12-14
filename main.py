"""
Main training script for CSSM ConvNeXt models.

Supports all 8 model variants with:
- Cosine learning rate scheduling
- Gradient clipping
- Validation evaluation
- Checkpointing (orbax)
- Weights & Biases logging
"""

import argparse
import os
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm
import orbax.checkpoint as ocp

from src.models.convnext import ModelFactory
from src.data import get_imagenette_video_loader, get_dataset_info


class TrainState(train_state.TrainState):
    """Extended train state with additional tracking."""
    epoch: int = 0


def create_train_state(
    rng: jax.Array,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
) -> Tuple[TrainState, optax.GradientTransformation]:
    """
    Initialize training state with optimizer and LR schedule.

    Args:
        rng: Random key for initialization
        model: The model to train
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        total_steps: Total training steps for LR schedule
        warmup_steps: Number of warmup steps
        grad_clip: Maximum gradient norm

    Returns:
        Tuple of (train_state, optimizer)
    """
    # Create dummy input for initialization
    dummy_input = jnp.ones((1, 8, 224, 224, 3))
    variables = model.init({'params': rng, 'dropout': rng}, dummy_input, training=False)
    params = variables['params']

    # Learning rate schedule: warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=learning_rate * 0.01,
    )

    # Optimizer: AdamW with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        epoch=0,
    )

    return state, tx


@partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.Array,
    num_classes: int,
) -> Tuple[TrainState, Dict[str, float]]:
    """
    Single training step.

    Args:
        state: Current training state
        batch: Tuple of (videos, labels)
        rng: Random key for dropout
        num_classes: Number of output classes

    Returns:
        Tuple of (updated_state, metrics_dict)
    """
    videos, labels = batch

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            videos,
            training=True,
            rngs={'dropout': rng},
        )
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    # Update state
    state = state.apply_gradients(grads=grads)

    # Compute metrics
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)

    metrics = {
        'train_loss': loss,
        'train_acc': acc,
    }

    return state, metrics


@partial(jax.jit, static_argnums=(2,))
def eval_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    num_classes: int,
) -> Dict[str, float]:
    """
    Single evaluation step.

    Args:
        state: Current training state
        batch: Tuple of (videos, labels)
        num_classes: Number of output classes

    Returns:
        Dictionary of metrics
    """
    videos, labels = batch

    logits = state.apply_fn(
        {'params': state.params},
        videos,
        training=False,
    )

    one_hot = jax.nn.one_hot(labels, num_classes)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)

    return {
        'val_loss': loss,
        'val_acc': acc,
    }


def evaluate(
    state: TrainState,
    val_loader,
    num_classes: int,
    num_batches: int = None,
) -> Dict[str, float]:
    """
    Run full validation evaluation.

    Args:
        state: Current training state
        val_loader: Validation data iterator
        num_classes: Number of output classes
        num_batches: Max batches to evaluate (None = all)

    Returns:
        Dictionary of averaged metrics
    """
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if num_batches is not None and i >= num_batches:
            break

        metrics = eval_step(state, batch, num_classes)
        total_loss += float(metrics['val_loss'])
        total_acc += float(metrics['val_acc'])
        count += 1

    return {
        'val_loss': total_loss / max(count, 1),
        'val_acc': total_acc / max(count, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Train CSSM ConvNeXt models')

    # Model configuration
    parser.add_argument('--mode', type=str, choices=['pure', 'hybrid'], required=True,
                        help='Model mode: pure (CSSM replaces conv) or hybrid (ConvNeXt + CSSM)')
    parser.add_argument('--cssm', type=str, choices=['standard', 'opponent'], required=True,
                        help='CSSM type: standard or opponent (gated)')
    parser.add_argument('--mixing', type=str, choices=['dense', 'depthwise'], required=True,
                        help='Mixing type: dense (multi-head) or depthwise')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per device')
    parser.add_argument('--seq_len', type=int, default=8,
                        help='Sequence length (number of frames)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay coefficient')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')

    # Logging and checkpointing
    parser.add_argument('--project', type=str, default='cssm-convnext',
                        help='Wandb project name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log metrics every N steps')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    args = parser.parse_args()

    # Generate run name from config
    run_name = f"{args.mode}_{args.cssm}_{args.mixing}"
    print(f"\n{'='*60}")
    print(f"Running Configuration: {run_name}")
    print(f"{'='*60}\n")

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
        )

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # Get dataset info
    dataset_info = get_dataset_info()
    num_classes = dataset_info['num_classes']
    train_size = dataset_info['train_size']
    steps_per_epoch = train_size // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    print(f"Dataset: Imagenette")
    print(f"  Train samples: {train_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}\n")

    # Create model
    model = ModelFactory(
        mode=args.mode,
        cssm_type=args.cssm,
        mixing=args.mixing,
        num_classes=num_classes,
    )

    # Initialize training state
    state, _ = create_train_state(
        rng=init_rng,
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        grad_clip=args.grad_clip,
    )

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {num_params:,}\n")

    if not args.no_wandb:
        wandb.config.update({'num_params': num_params})

    # Setup checkpointing
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()
    checkpoint_path = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

    # Training loop
    global_step = 0
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        state = state.replace(epoch=epoch)

        # Create fresh data loaders each epoch
        train_loader = get_imagenette_video_loader(
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
            split='train',
        )

        # Training
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                rng, step_rng = jax.random.split(rng)
                state, metrics = train_step(state, batch, step_rng, num_classes)

                epoch_loss += float(metrics['train_loss'])
                epoch_acc += float(metrics['train_acc'])
                num_batches += 1
                global_step += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{metrics['train_loss']:.4f}",
                    'acc': f"{metrics['train_acc']:.4f}",
                })

                # Log to wandb
                if not args.no_wandb and global_step % args.log_every == 0:
                    # Get current learning rate from optimizer state
                    lr = args.lr  # Simplified - actual LR from schedule
                    wandb.log({
                        'train/loss': float(metrics['train_loss']),
                        'train/acc': float(metrics['train_acc']),
                        'train/learning_rate': lr,
                        'train/step': global_step,
                    }, step=global_step)

        # Epoch summary
        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_train_acc = epoch_acc / max(num_batches, 1)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")

        # Validation
        if (epoch + 1) % args.eval_every == 0:
            val_loader = get_imagenette_video_loader(
                batch_size=args.batch_size,
                sequence_length=args.seq_len,
                split='validation',
            )
            val_metrics = evaluate(state, val_loader, num_classes)

            print(f"Epoch {epoch+1} - Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}")

            if not args.no_wandb:
                wandb.log({
                    'val/loss': val_metrics['val_loss'],
                    'val/acc': val_metrics['val_acc'],
                    'epoch': epoch + 1,
                }, step=global_step)

            # Track best model
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                print(f"  New best validation accuracy: {best_val_acc:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}')
            checkpointer.save(ckpt_path, state)
            print(f"  Saved checkpoint to {ckpt_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}\n")

    if not args.no_wandb:
        wandb.log({'best_val_acc': best_val_acc})
        wandb.finish()


if __name__ == '__main__':
    main()
