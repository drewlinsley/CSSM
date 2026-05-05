"""
Benchmark timing comparison: Standard CSSM vs Opponent CSSM vs Baseline ViT

Clean benchmark with identical model dimensions, no checkpoints, averaged over N steps.
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from src.models.cssm_vit import CSSMViT
from src.models.baseline_vit import BaselineViT


def create_train_state(rng, model, learning_rate=1e-4):
    """Create a simple train state for benchmarking."""
    # Dummy input: (B, T, H, W, C)
    dummy_input = jnp.ones((1, 8, 224, 224, 3))

    variables = model.init({'params': rng, 'dropout': rng}, dummy_input, training=False)
    params = variables['params']

    tx = optax.adamw(learning_rate)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def make_train_step(num_classes):
    """Create a JIT-compiled training step."""
    @jax.jit
    def train_step(state, batch, rng):
        images, labels = batch

        def loss_fn(params):
            logits = state.apply_fn(
                {'params': params},
                images,
                training=True,
                rngs={'dropout': rng},
            )
            one_hot = jax.nn.one_hot(labels, num_classes)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step


def benchmark_model(model, model_name, num_steps, batch_size, seq_len, num_classes, rng):
    """Benchmark a single model and return timing statistics."""
    print(f"\nBenchmarking {model_name}...")

    # Create train state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model)

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"  Parameters: {num_params:,}")

    # Create training step
    train_step = make_train_step(num_classes)

    # Generate random batches
    rng, data_rng = jax.random.split(rng)
    images = jax.random.normal(data_rng, (batch_size, seq_len, 224, 224, 3))
    labels = jax.random.randint(data_rng, (batch_size,), 0, num_classes)
    batch = (images, labels)

    # Warmup (compile JIT)
    print("  Warming up (JIT compilation)...")
    rng, step_rng = jax.random.split(rng)
    state, _ = train_step(state, batch, step_rng)
    jax.block_until_ready(state)

    # Benchmark
    print(f"  Running {num_steps} steps...")
    step_times = []

    for i in range(num_steps):
        rng, step_rng = jax.random.split(rng)

        start = time.perf_counter()
        state, loss = train_step(state, batch, step_rng)
        jax.block_until_ready(state)
        elapsed = time.perf_counter() - start

        step_times.append(elapsed * 1000)  # Convert to ms

        if (i + 1) % 20 == 0:
            print(f"    Step {i+1}/{num_steps}: {elapsed*1000:.1f}ms")

    step_times = np.array(step_times)

    stats = {
        'name': model_name,
        'params': num_params,
        'mean': np.mean(step_times),
        'std': np.std(step_times),
        'min': np.min(step_times),
        'max': np.max(step_times),
        'median': np.median(step_times),
        'times': step_times,
    }

    print(f"  Results: {stats['mean']:.1f} +/- {stats['std']:.1f} ms/step")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Benchmark CSSM vs ViT timing')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps to benchmark')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=8, help='Sequence length (timesteps)')
    parser.add_argument('--embed_dim', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=12, help='Number of layers')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='benchmark_timing.png', help='Output plot path')
    args = parser.parse_args()

    print("=" * 60)
    print("CSSM vs ViT Timing Benchmark")
    print("=" * 60)
    print(f"Config:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Depth: {args.depth}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Num steps: {args.num_steps}")

    rng = jax.random.PRNGKey(args.seed)

    # Create models with identical dimensions
    models = [
        (
            "Standard CSSM",
            CSSMViT(
                num_classes=args.num_classes,
                embed_dim=args.embed_dim,
                depth=args.depth,
                patch_size=args.patch_size,
                cssm_type='standard',
            )
        ),
        (
            "Opponent CSSM",
            CSSMViT(
                num_classes=args.num_classes,
                embed_dim=args.embed_dim,
                depth=args.depth,
                patch_size=args.patch_size,
                cssm_type='opponent',
            )
        ),
        (
            "Baseline ViT",
            BaselineViT(
                num_classes=args.num_classes,
                embed_dim=args.embed_dim,
                depth=args.depth,
                patch_size=args.patch_size,
                num_heads=args.embed_dim // 64,  # Standard head dim of 64
            )
        ),
    ]

    # Benchmark each model
    all_stats = []
    for name, model in models:
        rng, bench_rng = jax.random.split(rng)
        stats = benchmark_model(
            model, name, args.num_steps, args.batch_size,
            args.seq_len, args.num_classes, bench_rng
        )
        all_stats.append(stats)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<20} {'Params':>12} {'Mean (ms)':>12} {'Std (ms)':>10} {'Median (ms)':>12}")
    print("-" * 60)
    for stats in all_stats:
        print(f"{stats['name']:<20} {stats['params']:>12,} {stats['mean']:>12.1f} {stats['std']:>10.1f} {stats['median']:>12.1f}")

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot with error bars
    ax1 = axes[0]
    names = [s['name'] for s in all_stats]
    means = [s['mean'] for s in all_stats]
    stds = [s['std'] for s in all_stats]
    colors = ['#e74c3c', '#DC4BDC', '#3498db']  # red, pink/purple, blue

    bars = ax1.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Time per step (ms)')
    ax1.set_title(f'Training Step Time\n(batch={args.batch_size}, seq_len={args.seq_len}, depth={args.depth}, dim={args.embed_dim})')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Parameters vs Timing scatter plot
    ax2 = axes[1]
    for i, stats in enumerate(all_stats):
        ax2.scatter(stats['params'] / 1e6, stats['mean'],
                   s=150, c=colors[i], edgecolor='black', linewidth=1.2,
                   label=stats['name'], zorder=3)
        # Add error bars
        ax2.errorbar(stats['params'] / 1e6, stats['mean'], yerr=stats['std'],
                    fmt='none', c=colors[i], capsize=5, zorder=2)

    ax2.set_xlabel('Parameters (millions)')
    ax2.set_ylabel('Time per step (ms)')
    ax2.set_title('Parameters vs Timing')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {args.output}")

    # Also save raw data
    data_path = args.output.replace('.png', '_data.npz')
    np.savez(data_path,
             names=names,
             means=means,
             stds=stds,
             **{f"{s['name'].replace(' ', '_')}_times": s['times'] for s in all_stats})
    print(f"Raw data saved to: {data_path}")


if __name__ == '__main__':
    main()
