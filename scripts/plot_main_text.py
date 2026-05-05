#!/usr/bin/env python3
"""
Main-text scatter plots for Pathfinder CL-14 and 15-dist PathTracker.

Color scheme by family:
  SCSSM family (blues):     SCSSM, Mamba-SCSSM, GDN-SCSSM
  No-FFT controls (greens): Mamba-CSSM, S5-CSSM
  Transformers (reds):       Transformer-Spatial, Transformer-Spatiotemporal
  1D SSMs (purples):         Mamba2, GDN

Marker shape: circle = dim=64, square = dim=32

Usage:
    source activate.sh && python scripts/plot_main_text.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

try:
    import wandb
except ImportError:
    raise ImportError("pip install wandb")


# ── Model matching ───────────────────────────────────────────────────────────
MODELS = {
    'GDN-SCSSM': {
        'match': lambda r: r.config.get('cssm') == 'gdn' and 'int' not in r.name and r.config.get('delta_key_dim') == 4 and r.config.get('qkv_conv_size') in [1, 5],
    },
    'Mamba-SCSSM': {
        'match': lambda r: r.config.get('cssm') == 'gated' and r.config.get('kernel_size', 0) >= 11 and r.config.get('short_conv_spatial_size', 3) != 0,
    },
    'SCSSM': {
        'match': lambda r: r.config.get('cssm') == 'no_gate' and r.config.get('kernel_size', 0) >= 11 and r.config.get('num_params', 0) > 5000,
    },
    'Mamba-CSSM': {
        'match': lambda r: r.config.get('cssm') == 'no_fft',
    },
    'S5-CSSM': {
        'match': lambda r: r.config.get('cssm') == 'conv_ssm',
    },
    'Transformer-Spatial': {
        'match': lambda r: r.config.get('cssm') == 'spatial_attn',
    },
    'Transformer-Spatiotemporal': {
        'match': lambda r: r.config.get('cssm') == 'spatiotemporal_attn',
    },
    'Mamba-2': {
        'match': lambda r: r.config.get('cssm') == 'mamba2_seq',
    },
    'GDN': {
        'match': lambda r: r.config.get('cssm') == 'gdn_seq',
    },
}

# ── Colors by family ─────────────────────────────────────────────────────────
COLORS = {
    'GDN-SCSSM':                '#1565C0',
    'Mamba-SCSSM':              '#42A5F5',
    'SCSSM':                    '#90CAF9',
    'Mamba-CSSM':               '#2E7D32',
    'S5-CSSM':                  '#81C784',
    'Transformer-Spatial':      '#C62828',
    'Transformer-Spatiotemporal': '#EF9A9A',
    'Mamba-2':                   '#6A1B9A',
    'GDN':                      '#CE93D8',
}

# Marker: circle for dim=64, square for dim=32
DIM_MARKERS = {64: 'o', 32: 's'}

GROUPS = {
    'Spectral SSM': ['GDN-SCSSM', 'Mamba-SCSSM', 'SCSSM'],
    'No-FFT Control': ['Mamba-CSSM', 'S5-CSSM'],
    'Transformer': ['Transformer-Spatial', 'Transformer-Spatiotemporal'],
    '1D Sequence SSM': ['Mamba-2', 'GDN'],
}


def fetch_all_runs(project: str):
    """Fetch best run per (model_family, embed_dim) from W&B."""
    api = wandb.Api()
    runs = api.runs(f'serrelab/{project}')

    # Collect best per (model, dim)
    results = {}
    for r in runs:
        acc = r.summary.get('best_val_acc', r.summary.get('val_acc', None))
        if acc is None or acc < 0.49:
            continue

        ed = r.config.get('embed_dim', None)
        if ed not in [32, 64]:
            continue

        for model_name, spec in MODELS.items():
            if spec['match'](r):
                key = (model_name, ed)
                params = r.config.get('num_params', None)
                if key not in results or acc > results[key]['acc']:
                    results[key] = {'acc': acc, 'params': params, 'dim': ed}
                break

    return results


def plot_scatter(ax, results, title, show_ylabel=True, show_legend=False):
    """Plot accuracy vs params scatter for one dataset."""
    plotted_models = set()

    for (model_name, dim), data in results.items():
        if data['params'] is None or data['params'] == 0:
            continue
        marker = DIM_MARKERS.get(dim, 'o')
        ax.scatter(
            data['params'], data['acc'] * 100,
            color=COLORS[model_name],
            marker=marker,
            s=100 if dim == 64 else 80,
            edgecolors='white',
            linewidths=0.7,
            zorder=5,
        )
        plotted_models.add(model_name)

    # Chance line
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xscale('log')
    ax.set_xlabel('Parameters', fontsize=12)
    if show_ylabel:
        ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(45, 95)
    ax.grid(True, alpha=0.15, which='both')
    ax.tick_params(labelsize=10)

    if show_legend:
        handles = []
        for group_name, model_names in GROUPS.items():
            handles.append(mpatches.Patch(color='none', label=f'$\\bf{{{group_name}}}$'))
            for mn in model_names:
                if mn in plotted_models:
                    # Use circle marker in legend (representative)
                    handles.append(mlines.Line2D([0], [0],
                        marker='o', color='w',
                        markerfacecolor=COLORS[mn],
                        markeredgecolor='white',
                        markersize=9, label=mn, linewidth=0))

        # Add dim legend
        handles.append(mpatches.Patch(color='none', label=''))
        handles.append(mpatches.Patch(color='none', label='$\\bf{Embed\\ Dim}$'))
        handles.append(mlines.Line2D([0], [0], marker='o', color='w',
            markerfacecolor='gray', markeredgecolor='white',
            markersize=9, label='dim=64', linewidth=0))
        handles.append(mlines.Line2D([0], [0], marker='s', color='w',
            markerfacecolor='gray', markeredgecolor='white',
            markersize=8, label='dim=32', linewidth=0))

        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, frameon=True, fancybox=True, shadow=False,
                  handletextpad=0.5, labelspacing=0.5)


def main():
    print("Fetching Pathfinder CL-14...")
    pf_results = fetch_all_runs('CSSM_pathfinder')
    print(f"  Found {len(pf_results)} (model, dim) pairs")
    for (name, dim), data in sorted(pf_results.items(), key=lambda x: -x[1]['acc']):
        print(f"    {name:30s} dim={dim} acc={data['acc']:.4f} params={data['params']:>8}")

    print("\nFetching 15-dist PathTracker...")
    pt_results = fetch_all_runs('CSSM_15dist')
    print(f"  Found {len(pt_results)} (model, dim) pairs")
    for (name, dim), data in sorted(pt_results.items(), key=lambda x: -x[1]['acc']):
        print(f"    {name:30s} dim={dim} acc={data['acc']:.4f} params={data['params']:>8}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    plot_scatter(ax1, pf_results, 'Pathfinder CL-14', show_ylabel=True, show_legend=False)
    plot_scatter(ax2, pt_results, '15-dist PathTracker', show_ylabel=False, show_legend=True)

    plt.tight_layout()
    plt.savefig('main_text_scatter.png', dpi=200, bbox_inches='tight')
    print("\nSaved to main_text_scatter.png")
    plt.close()


if __name__ == '__main__':
    main()
