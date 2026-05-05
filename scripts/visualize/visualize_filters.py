"""
Visualize learned filters from CSSM models.

Usage:
    python visualize_filters.py --checkpoint checkpoints/run_name/epoch_X
    python visualize_filters.py --checkpoint checkpoints/run_name/epoch_X --layer 0
"""

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from flax.training import train_state


def load_params(checkpoint_path: str):
    """Load parameters from checkpoint."""
    # Orbax requires absolute paths
    checkpoint_path = os.path.abspath(checkpoint_path)

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(checkpoint_path)

    # The state is saved as a dict with 'params', 'opt_state', etc.
    if isinstance(restored, dict) and 'params' in restored:
        return restored['params']
    elif hasattr(restored, 'params'):
        return restored.params
    else:
        return restored


def find_cssm_params(params, prefix=''):
    """Recursively find all CSSM-related parameters."""
    cssm_params = {}

    for key, value in params.items():
        full_key = f"{prefix}/{key}" if prefix else key

        if isinstance(value, dict):
            # Recurse into nested dicts
            cssm_params.update(find_cssm_params(value, full_key))
        else:
            # Check if this is a CSSM parameter
            if any(name in key for name in ['kernel', 'k_exc', 'k_inh', 'decay']):
                cssm_params[full_key] = np.array(value)

    return cssm_params


def plot_spatial_kernels(kernels: np.ndarray, title: str, max_show: int = 16):
    """Plot spatial kernels as a grid of images."""
    n_kernels = min(kernels.shape[0], max_show)
    n_cols = min(4, n_kernels)
    n_rows = (n_kernels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    vmin, vmax = kernels[:n_kernels].min(), kernels[:n_kernels].max()
    vabs = max(abs(vmin), abs(vmax))

    for i in range(n_kernels):
        row, col = i // n_cols, i % n_cols
        im = axes[row, col].imshow(kernels[i], cmap='RdBu_r', vmin=-vabs, vmax=vabs)
        axes[row, col].set_title(f'Ch {i}')
        axes[row, col].axis('off')

    # Hide unused subplots
    for i in range(n_kernels, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label('Weight')
    return fig


def plot_spectral_magnitude(kernels: np.ndarray, title: str, max_show: int = 16):
    """Plot FFT magnitude of kernels."""
    n_kernels = min(kernels.shape[0], max_show)
    n_cols = min(4, n_kernels)
    n_rows = (n_kernels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_kernels):
        row, col = i // n_cols, i % n_cols
        # Compute FFT and shift zero-freq to center
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(kernels[i])))
        im = axes[row, col].imshow(np.log1p(fft_mag), cmap='viridis')
        axes[row, col].set_title(f'Ch {i}')
        axes[row, col].axis('off')

    # Hide unused subplots
    for i in range(n_kernels, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    fig.suptitle(f'{title} - Spectral Magnitude', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label('ln(1 + |FFT|)')
    return fig


def plot_kernel_stats(cssm_params: dict):
    """Plot statistics of all kernels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Collect only spatial kernels (k_exc, k_inh), not Dense layer weights
    all_kernels = []
    labels = []
    for name, param in cssm_params.items():
        if 'k_exc' in name or 'k_inh' in name:
            all_kernels.append(param.flatten())
            # e.g., "block0/cssm/k_exc" -> "b0_k_exc"
            parts = name.split('/')
            block_part = parts[0].replace('block', 'b') if parts else ''
            short_name = f"{block_part}_{parts[-1]}" if len(parts) >= 2 else name
            labels.append(short_name)

    if all_kernels:
        # Histogram of values
        for i, (kernel, label) in enumerate(zip(all_kernels, labels)):
            axes[0].hist(kernel, bins=50, alpha=0.5, label=label)
        axes[0].set_xlabel('Kernel Value')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Kernel Value Distribution')
        if len(labels) <= 10:
            axes[0].legend()
        else:
            axes[0].legend(fontsize=6, ncol=2)

        # Box plot
        axes[1].boxplot(all_kernels, tick_labels=labels)
        axes[1].set_ylabel('Value')
        axes[1].set_title('Kernel Value Range')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize CSSM filters')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer index to visualize (default: all)')
    parser.add_argument('--output_dir', type=str, default='filter_viz',
                        help='Directory to save visualizations')
    parser.add_argument('--max_channels', type=int, default=16,
                        help='Maximum channels to show per plot')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load parameters
    print(f"Loading checkpoint from {args.checkpoint}")
    params = load_params(args.checkpoint)

    # Find CSSM parameters
    cssm_params = find_cssm_params(params)

    print(f"\nFound {len(cssm_params)} CSSM parameters:")
    for name, param in cssm_params.items():
        print(f"  {name}: {param.shape}")

    # Filter by layer if specified
    if args.layer is not None:
        layer_str = f'block{args.layer}' if 'block' in str(list(cssm_params.keys())[0]) else f'stage.*block{args.layer}'
        import re
        cssm_params = {k: v for k, v in cssm_params.items()
                       if re.search(layer_str, k) or f'_{args.layer}' in k}
        print(f"\nFiltered to layer {args.layer}: {len(cssm_params)} parameters")

    # Plot each kernel
    for name, param in cssm_params.items():
        if param.ndim == 3:  # (C, H, W) spatial kernel
            # Spatial visualization
            fig = plot_spatial_kernels(param, name, args.max_channels)
            save_path = os.path.join(args.output_dir, f"{name.replace('/', '_')}_spatial.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close(fig)

            # Spectral visualization
            fig = plot_spectral_magnitude(param, name, args.max_channels)
            save_path = os.path.join(args.output_dir, f"{name.replace('/', '_')}_spectral.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close(fig)

    # Overall statistics
    fig = plot_kernel_stats(cssm_params)
    save_path = os.path.join(args.output_dir, 'kernel_stats.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
