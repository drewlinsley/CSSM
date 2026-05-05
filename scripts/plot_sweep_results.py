#!/usr/bin/env python3
"""Sweep result plots in tics_alignment style.

Pulls results from W&B. One figure per dataset with 2 panels:
  Left: accuracy vs step time scatter
  Right: legend

Style: Raleway/Arial font, seaborn ticks, no grid, despine, 600 dpi PDF.

Usage:
    source activate.sh && python scripts/plot_sweep_results.py
"""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D
import seaborn as sns
import wandb

# ============================================================================
# tics_alignment style
# ============================================================================
plt.style.use('default')
sns.set_style("ticks")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,
    'axes.linewidth': 1.0,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.0,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
})

# ============================================================================
# Param counts (computed via model.init on PathTracker 32x32x64f)
# ============================================================================
PARAM_COUNTS = {
    'st_d1_e32': 17474, 'st_d1_e64': 67714,
    'st_d3_e32': 51650, 'st_d3_e64': 201602,
    'gdn_d1_e32_qkv1_dk1': 16540, 'gdn_d1_e32_qkv1_dk2': 15572, 'gdn_d1_e32_qkv1_dk4': 14604,
    'gdn_d1_e32_qkv5_dk1': 17372, 'gdn_d1_e32_qkv5_dk2': 16404, 'gdn_d1_e32_qkv5_dk4': 15436,
    'gdn_d1_e64_qkv1_dk1': 43092, 'gdn_d1_e64_qkv1_dk2': 41156, 'gdn_d1_e64_qkv1_dk4': 39220,
    'gdn_d1_e64_qkv5_dk1': 44756, 'gdn_d1_e64_qkv5_dk2': 42820, 'gdn_d1_e64_qkv5_dk4': 40884,
    'gdnint_d1_e32_dk1': 17591, 'gdnint_d1_e32_dk2': 15655, 'gdnint_d1_e32_dk4': 14687,
    'gdnint_d1_e64_dk1': 45143, 'gdnint_d1_e64_dk2': 41271, 'gdnint_d1_e64_dk4': 39335,
    'gdnint_elem_d1_e32_dk2': 15606, 'gdnint_elem_d1_e32_dk4': 14638,
    'gdnint_elem_d1_e64_dk2': 41222, 'gdnint_elem_d1_e64_dk4': 39286,
    'gdnint_qk_d1_e32_dk2': 16629, 'gdnint_qk_d1_e32_dk4': 15661,
    'gdnint_qk_d1_e64_dk2': 42245, 'gdnint_qk_d1_e64_dk4': 41301,
    'cssm_d1_e32': 61762, 'cssm_d1_e64': 128034,
    'sa_d1_e32': 17474, 'sa_d1_e64': 67714,
}

# Compute param counts for new models dynamically at import time
def _compute_new_params():
    """Compute param counts for mamba2_seq, gdn_seq, conv_ssm, cssm_full."""
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    try:
        import jax
        import jax.numpy as jnp
        import sys
        sys.path.insert(0, '.')
        from src.models.simple_cssm import SimpleCSSM

        rng = jax.random.PRNGKey(0)
        configs = [
            # Pathfinder-like: stem_layers=1, 128px → 64×64 latent, seq_len=1 for seq models
            ('m2seq_d1_e32_n16', dict(cssm_type='mamba2_seq', embed_dim=32, state_dim=16, seq_len=1, expand_factor=2, ssd_chunk_size=8, short_conv_size=4, flatten_mode='temporal_spatial', stem_layers=1)),
            ('m2seq_d1_e32_n64', dict(cssm_type='mamba2_seq', embed_dim=32, state_dim=64, seq_len=1, expand_factor=2, ssd_chunk_size=8, short_conv_size=4, flatten_mode='temporal_spatial', stem_layers=1)),
            ('m2seq_d1_e64_n16', dict(cssm_type='mamba2_seq', embed_dim=64, state_dim=16, seq_len=1, expand_factor=2, ssd_chunk_size=8, short_conv_size=4, flatten_mode='temporal_spatial', stem_layers=1)),
            ('m2seq_d1_e64_n64', dict(cssm_type='mamba2_seq', embed_dim=64, state_dim=64, seq_len=1, expand_factor=2, ssd_chunk_size=8, short_conv_size=4, flatten_mode='temporal_spatial', stem_layers=1)),
            ('gdnseq_d1_e32_dk2', dict(cssm_type='gdn_seq', embed_dim=32, delta_key_dim=2, short_conv_size=4, flatten_mode='temporal_spatial', stem_layers=1)),
            ('gdnseq_d1_e32_dk4', dict(cssm_type='gdn_seq', embed_dim=32, delta_key_dim=4, short_conv_size=4, flatten_mode='temporal_spatial', stem_layers=1)),
            ('gdnseq_d1_e64_dk2', dict(cssm_type='gdn_seq', embed_dim=64, delta_key_dim=2, short_conv_size=4, flatten_mode='temporal_spatial', stem_layers=1)),
            ('gdnseq_d1_e64_dk4', dict(cssm_type='gdn_seq', embed_dim=64, delta_key_dim=4, short_conv_size=4, flatten_mode='temporal_spatial', stem_layers=1)),
            ('convssm_d1_e32_ks3', dict(cssm_type='conv_ssm', embed_dim=32, kernel_size=3, stem_layers=1)),
            ('convssm_d1_e32_ks5', dict(cssm_type='conv_ssm', embed_dim=32, kernel_size=5, stem_layers=1)),
            ('convssm_d1_e64_ks3', dict(cssm_type='conv_ssm', embed_dim=64, kernel_size=3, stem_layers=1)),
            ('convssm_d1_e64_ks5', dict(cssm_type='conv_ssm', embed_dim=64, kernel_size=5, stem_layers=1)),
            ('cssm_full_d1_e32', dict(cssm_type='gated', embed_dim=32, kernel_size=64, stem_layers=1)),
            ('cssm_full_d1_e64', dict(cssm_type='gated', embed_dim=64, kernel_size=64, stem_layers=1)),
        ]
        x = jax.random.normal(rng, (1, 1, 32, 32, 3))  # dummy input
        counts = {}
        for name, kw in configs:
            model = SimpleCSSM(num_classes=2, depth=1, pool_type='max', **kw)
            params = model.init(rng, x, training=True)
            n = sum(p.size for p in jax.tree.leaves(params))
            counts[name] = n
        return counts
    except Exception as e:
        print(f'Warning: could not compute new param counts: {e}')
        return {}

PARAM_COUNTS.update(_compute_new_params())

def get_params(name):
    n = name
    for prefix in ('pf_', '32f_', 'rs32f_', '15dist_'):
        if n.startswith(prefix):
            n = n[len(prefix):]
    n_nolog = n.replace('gdn_nolog_', 'gdn_')
    # Strip lr/head suffixes for transformers
    for base in ('sa_d1_e32', 'sa_d1_e64', 'st_d1_e32', 'st_d1_e64', 'st_d3_e32', 'st_d3_e64'):
        if n.startswith(base):
            return PARAM_COUNTS.get(base)
    for key in (n, n_nolog):
        if key in PARAM_COUNTS:
            return PARAM_COUNTS[key]
    return None

# ============================================================================
# Color palette (tics_alignment two-tone style)
# ============================================================================
COLORS = {
    'gdn':          '#E34234',   # red (like CNN)
    'gdn_nolog':    '#EE9888',   # light red
    'gdnint':       '#9C27B0',   # purple (like RNN)
    'gdnint_elem':  '#CE93D8',   # light purple
    'gdnint_qk':    '#2E8540',   # green (like ConvViT)
    'transformer':  '#1E88E5',   # blue (like ViT)
    'transformer_sa': '#90CAF9', # light blue (spatial-only)
    'cssm':         '#fbbc05',   # yellow (like adversarial)
    'cssm_full':    '#F9A825',   # dark yellow (full-kernel CSSM)
    'cssm_1x1':     '#FFD54F',   # pale yellow (1x1 kernel CSSM)
    'mamba2_seq':   '#00ACC1',   # teal (sequence model)
    'gdn_seq':      '#FF7043',   # deep orange (sequence delta rule)
    'conv_ssm':     '#66BB6A',   # green (NVlabs ConvSSM)
    's4nd':         '#26A69A',   # dark teal (S4ND)
    'convs5':       '#8D6E63',   # brown (ConvS5)
    'nofft':        '#AD1457',   # magenta (no-FFT control)
    'nogate':       '#455A64',   # blue-grey (no-gate control)
}

LABELS = {
    'gdn':            'GDN (log)',
    'gdn_nolog':      'GDN (no-log)',
    'gdnint':         'GDN-InT',
    'gdnint_elem':    'GDN-InT elem',
    'gdnint_qk':      'GDN-InT qk',
    'transformer':    'Transformer (ST)',
    'transformer_sa': 'Transformer (S)',
    'cssm':           'Spectral Mamba',
    'cssm_full':      'Spectral Mamba (full)',
    'cssm_1x1':       'Spectral Mamba (1x1)',
    'mamba2_seq':     'Mamba-2 Seq',
    'gdn_seq':        'GDN Seq',
    'conv_ssm':       'ConvSSM',
    's4nd':           'S4ND',
    'convs5':         'ConvS5',
    'nofft':          'No-FFT (control)',
    'nogate':         'No-Gate (control)',
}

# Marker: square=e64, circle=e32 (like vision vs VLM)
MARKERS = {32: 'o', 64: 's'}

# ============================================================================
# Pull data from W&B
# ============================================================================
api = wandb.Api()

def pull_results(project):
    try:
        runs = api.runs(project)
    except Exception:
        return {}
    seen = {}
    run_objects = {}  # keep run objects for curve pulling
    for r in runs:
        if r.state != 'finished':
            continue
        s = r.summary
        best = s.get('best_val_acc', None)
        if best is None:
            continue
        if r.name not in seen or best > seen[r.name]['acc']:
            seen[r.name] = {
                'acc': best,
                'min_step_ms': s.get('timing/epoch_min_step_ms', None),
                'epoch': s.get('early_stopped_epoch', None) or s.get('epoch', None),
            }
            run_objects[r.name] = r
    return seen, run_objects


def pull_efficiency(run_objects, results, threshold=0.95):
    """Compute epoch to reach threshold * peak_acc for each run.

    Returns dict: name → {epoch_to_95: int, total_epochs: int, frac: float, peak_acc: float}
    """
    efficiency = {}
    for name, r in run_objects.items():
        info = results.get(name)
        if info is None:
            continue
        peak = info['acc']
        target = threshold * peak
        try:
            h = r.history(keys=['val/acc', 'epoch'], samples=5000, pandas=True)
            val_rows = h.dropna(subset=['val/acc']).sort_values('epoch')
            if val_rows.empty:
                continue
            total_ep = int(val_rows['epoch'].max())
            # Find first epoch where val/acc >= target
            hit = val_rows[val_rows['val/acc'] >= target]
            if hit.empty:
                epoch_to = total_ep  # never reached
            else:
                epoch_to = int(hit.iloc[0]['epoch'])
            efficiency[name] = {
                'epoch_to_95': epoch_to,
                'total_epochs': total_ep,
                'frac': epoch_to / max(total_ep, 1),
                'peak_acc': peak,
            }
        except Exception as e:
            print(f'  Warning: could not pull curve for {name}: {e}')
            continue
    return efficiency

pt_results, pt_runs = pull_results('CSSM_pathtracker')
pf_results, pf_runs = pull_results('CSSM_pathfinder')
pt32_results, pt32_runs = pull_results('CSSM_pathtracker_32f')
rs32_results, rs32_runs = pull_results('CSSM_pathtracker_restyled_32f')
dist15_results, dist15_runs = pull_results('CSSM_15dist')

# ============================================================================
# Classify runs
# ============================================================================
_DEPTH_RE = re.compile(r'_d(\d+)(?:_|$)')
_DIM_RE = re.compile(r'_e(\d+)(?:_|$)')


def _extract_dim_depth(n):
    """Extract embed_dim and depth from a run-name suffix. Tolerant to
    extra tokens (ks11, 1x1, fac, t1, v2, etc.) between them."""
    dm = _DIM_RE.search('_' + n)  # prefix _ so the regex works when the name starts with eNN
    dp = _DEPTH_RE.search('_' + n)
    dim = int(dm.group(1)) if dm else None
    depth = int(dp.group(1)) if dp else None
    return dim, depth


def classify_run(name):
    n = name
    for prefix in ('pf_', '32f_', 'rs32f_', '15dist_'):
        if n.startswith(prefix):
            n = n[len(prefix):]

    # Specific family prefixes — longest-first so sub-variants win
    if n.startswith('cssm_full_'):
        family = 'cssm_full'
    elif n.startswith('cssm_1x1_'):
        family = 'cssm_1x1'
    elif n.startswith('cssm_'):
        family = 'cssm'
    elif n.startswith('m2seq_'):
        family = 'mamba2_seq'
    elif n.startswith('gdnseq_'):
        family = 'gdn_seq'
    elif n.startswith('convssm_'):
        family = 'conv_ssm'
    elif n.startswith('convs5_'):
        family = 'convs5'
    elif n.startswith('s4nd_'):
        family = 's4nd'
    elif n.startswith('nofft_'):
        family = 'nofft'
    elif n.startswith('nogate_'):
        family = 'nogate'
    elif n.startswith('gdn_nolog_'):
        family = 'gdn_nolog'
    elif n.startswith('gdn_d') or n.startswith('gdn_'):
        # Only match bare GDN if not a sub-family
        if n.startswith('gdnint_elem_'):
            family = 'gdnint_elem'
        elif n.startswith('gdnint_qk_'):
            family = 'gdnint_qk'
        elif n.startswith('gdnint_'):
            family = 'gdnint'
        else:
            family = 'gdn'
    elif n.startswith('gdnint_elem_'):
        family = 'gdnint_elem'
    elif n.startswith('gdnint_qk_'):
        family = 'gdnint_qk'
    elif n.startswith('gdnint_'):
        family = 'gdnint'
    elif n.startswith('sa_'):
        family = 'transformer_sa'
    elif n.startswith('st_'):
        family = 'transformer'
    else:
        return 'other', None, None

    dim, depth = _extract_dim_depth(n)
    return family, dim, depth


FAMILIES = ['transformer', 'transformer_sa', 'cssm', 'cssm_full', 'cssm_1x1',
            'gdn', 'gdn_nolog', 'gdnint', 'gdnint_elem', 'gdnint_qk',
            'mamba2_seq', 'gdn_seq', 'conv_ssm', 's4nd', 'convs5',
            'nofft', 'nogate']


def human_label(name):
    """Convert coded run name to human-readable label.

    Examples:
        gdn_d1_e64_qkv5_dk4  → GDN dim=64 conv=5 dk=4
        cssm_full_d1_e32      → Spectral Mamba (full) dim=32
        m2seq_d1_e64_n16      → Mamba-2 Seq dim=64 N=16
        st_d3_e64_dp01_mlp40  → Transformer (ST) depth=3 dim=64 dp=0.1
    """
    n = name
    for pfx in ('pf_', '32f_', 'rs32f_', '15dist_'):
        if n.startswith(pfx):
            n = n[len(pfx):]

    # Extract common fields via regex
    depth_m = re.search(r'_d(\d+)(?:_|$)', n)
    dim_m = re.search(r'_e(\d+)', n)
    depth = depth_m.group(1) if depth_m else None
    dim = dim_m.group(1) if dim_m else None

    parts = []

    # Model family name
    if n.startswith('cssm_full_'):
        parts.append('Spectral Mamba (full)')
    elif n.startswith('cssm_1x1_'):
        parts.append('Spectral Mamba (1x1)')
    elif n.startswith('cssm_'):
        parts.append('Spectral Mamba')
    elif n.startswith('m2seq_'):
        parts.append('Mamba-2 Seq')
    elif n.startswith('gdnseq_'):
        parts.append('GDN Seq')
    elif n.startswith('convssm_'):
        parts.append('ConvSSM')
    elif n.startswith('convs5_'):
        parts.append('ConvS5')
    elif n.startswith('s4nd_'):
        parts.append('S4ND')
    elif n.startswith('nofft_'):
        parts.append('No-FFT')
    elif n.startswith('nogate_'):
        parts.append('No-Gate')
    elif n.startswith('gdn_nolog_'):
        parts.append('GDN (no-log)')
    elif n.startswith('gdnint_elem_'):
        parts.append('GDN-InT elem')
    elif n.startswith('gdnint_qk_'):
        parts.append('GDN-InT qk')
    elif n.startswith('gdnint_'):
        parts.append('GDN-InT')
    elif n.startswith('gdn_d') or n.startswith('gdn_'):
        parts.append('GDN')
    elif n.startswith('sa_'):
        parts.append('Transformer (S)')
    elif n.startswith('st_'):
        parts.append('Transformer (ST)')
    else:
        return n  # fallback

    # Append hyperparameters
    if depth and depth != '1':
        parts.append(f'depth={depth}')
    if dim:
        parts.append(f'dim={dim}')

    # Model-specific HPs
    qkv_m = re.search(r'_qkv(\d+)', n)
    dk_m = re.search(r'_dk(\d+)', n)
    ks_m = re.search(r'_ks(\d+)', n)
    sd_m = re.search(r'_n(\d+)', n)
    lr_m = re.search(r'_lr([0-9e]+)', n)
    heads_m = re.search(r'_h(\d+)$', n)
    dp_m = re.search(r'_dp(\d+)', n)

    if qkv_m:
        parts.append(f'conv={qkv_m.group(1)}')
    if dk_m:
        parts.append(f'dk={dk_m.group(1)}')
    if ks_m:
        parts.append(f'ks={ks_m.group(1)}')
    if sd_m:
        parts.append(f'N={sd_m.group(1)}')
    if lr_m:
        lr_val = lr_m.group(1)
        parts.append(f'lr={lr_val}')
    if heads_m:
        parts.append(f'heads={heads_m.group(1)}')
    if dp_m:
        dp_val = dp_m.group(1)
        if dp_val != '00':
            parts.append(f'dp=0.{dp_val.lstrip("0")}')

    return ' '.join(parts)


def get_plot_data(results):
    data = []
    for name, info in results.items():
        family, dim, depth = classify_run(name)
        if family not in FAMILIES:
            continue
        params = get_params(name)
        data.append({
            'name': name, 'acc': info['acc'],
            'min_ms': info.get('min_step_ms'),
            'epoch': info.get('epoch') or 60,
            'params': params,
            'family': family, 'dim': dim, 'depth': depth,
        })
    return data


# ============================================================================
# Plotting
# ============================================================================
def make_figure(data, title, filename, x_key='min_ms', x_label='Step Time (ms, post-JIT)'):
    """Create a tics_alignment style figure: main scatter + legend panel."""

    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    # Plot points
    for e in data:
        if e.get(x_key) is None:
            continue
        color = COLORS[e['family']]
        marker = MARKERS.get(e['dim'], 'o')
        ax.scatter(e[x_key], e['acc'], c=color, marker=marker, s=80,
                   edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5)

    # Annotate top-3 and all CSSM points
    valid = [e for e in data if e.get(x_key) is not None]
    sorted_by_acc = sorted(valid, key=lambda x: x['acc'], reverse=True)
    annotated = set()

    def _annotate(e):
        if e['name'] in annotated:
            return
        label = human_label(e['name'])
        ax.annotate(label, (e[x_key], e['acc']),
                    fontsize=8, ha='left', va='bottom',
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='#CCCCCC',
                              boxstyle='round,pad=0.2', linewidth=0.5),
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        annotated.add(e['name'])

    for e in sorted_by_acc[:3]:
        _annotate(e)
    for e in data:
        if e['family'] in ('cssm', 'cssm_full', 'mamba2_seq', 'gdn_seq', 'conv_ssm'):
            _annotate(e)

    # Chance line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1.0, zorder=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Best Validation Accuracy')
    ax.set_title(title, fontweight='bold')
    sns.despine(ax=ax)

    # ── Legend panel ──
    ax_leg.axis('off')
    handles = []
    for fam in FAMILIES:
        if fam not in COLORS:
            continue
        if not any(e['family'] == fam for e in data):
            continue
        handles.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=COLORS[fam], markeredgecolor='black',
                              markeredgewidth=0.5, markersize=9, label=LABELS[fam]))

    # Spacer + shape legend
    handles.append(Line2D([0], [0], color='w', label=''))
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                          markeredgecolor='black', markeredgewidth=0.5,
                          markersize=7, label='dim = 32'))
    handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                          markeredgecolor='black', markeredgewidth=0.5,
                          markersize=7, label='dim = 64'))

    legend = ax_leg.legend(handles=handles, loc='center left',
                           title='Models', fontsize=12, title_fontsize=14,
                           frameon=True, framealpha=0.9, edgecolor='#DDDDDD',
                           borderpad=0.5, handletextpad=0.5, markerscale=1.5)

    plt.tight_layout()

    # Save PNG and PDF
    for ext in ['png', 'pdf']:
        out = f'{filename}.{ext}'
        fig.savefig(out, dpi=600 if ext == 'png' else None, bbox_inches='tight')
    print(f'Saved {filename}.png and {filename}.pdf')
    plt.close(fig)


def make_params_figure(data, title, filename):
    """Accuracy vs params scatter."""
    make_figure(data, title, filename, x_key='params', x_label='Parameters')


def make_efficiency_figure(data, efficiency, title, filename):
    """Sample efficiency: epoch_to_95%_peak (x) vs peak accuracy (y).

    Models that learn fast AND well appear in the top-left (best quadrant).
    """
    # Merge efficiency data into plot data
    eff_data = []
    for e in data:
        eff = efficiency.get(e['name'])
        if eff is None:
            continue
        eff_data.append({**e, **eff})

    if not eff_data:
        return

    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    # Plot points
    for e in eff_data:
        color = COLORS.get(e['family'], '#999')
        marker = MARKERS.get(e['dim'], 'o')
        ax.scatter(e['epoch_to_95'], e['peak_acc'], c=color, marker=marker, s=80,
                   edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5)

    # Annotate top-3 by accuracy and all new model families
    sorted_by_acc = sorted(eff_data, key=lambda x: x['peak_acc'], reverse=True)
    annotated = set()

    def _annotate(e):
        if e['name'] in annotated:
            return
        label = human_label(e['name'])
        ax.annotate(label, (e['epoch_to_95'], e['peak_acc']),
                    fontsize=8, ha='left', va='bottom',
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='#CCCCCC',
                              boxstyle='round,pad=0.2', linewidth=0.5),
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        annotated.add(e['name'])

    for e in sorted_by_acc[:3]:
        _annotate(e)
    # Fastest learners (lowest epoch_to_95 among above-chance)
    sorted_by_speed = sorted(eff_data, key=lambda x: x['epoch_to_95'])
    for e in sorted_by_speed[:3]:
        _annotate(e)
    for e in eff_data:
        if e['family'] in ('cssm', 'cssm_full', 'mamba2_seq', 'gdn_seq', 'conv_ssm'):
            _annotate(e)

    # "Best" quadrant indicator
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1.0, zorder=0)
    ax.set_xlabel('Epochs to 95% of Peak Accuracy')
    ax.set_ylabel('Peak Validation Accuracy')
    ax.set_title(title, fontweight='bold')
    # Arrow pointing to ideal corner (use data coords for log scale)
    xlo = min(e['epoch_to_95'] for e in eff_data)
    yhi = max(e['peak_acc'] for e in eff_data)
    ax.annotate('fast + accurate', xy=(xlo, yhi),
                fontsize=9, color='green', alpha=0.6, style='italic',
                ha='left', va='top', xytext=(5, -5), textcoords='offset points')
    sns.despine(ax=ax)

    # Legend panel
    ax_leg.axis('off')
    handles = []
    for fam in FAMILIES:
        if fam not in COLORS:
            continue
        if not any(e['family'] == fam for e in eff_data):
            continue
        handles.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=COLORS[fam], markeredgecolor='black',
                              markeredgewidth=0.5, markersize=9, label=LABELS[fam]))
    handles.append(Line2D([0], [0], color='w', label=''))
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                          markeredgecolor='black', markeredgewidth=0.5,
                          markersize=7, label='dim = 32'))
    handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                          markeredgecolor='black', markeredgewidth=0.5,
                          markersize=7, label='dim = 64'))
    ax_leg.legend(handles=handles, loc='center left',
                  title='Models', fontsize=12, title_fontsize=14,
                  frameon=True, framealpha=0.9, edgecolor='#DDDDDD',
                  borderpad=0.5, handletextpad=0.5, markerscale=1.5)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        out = f'{filename}.{ext}'
        fig.savefig(out, dpi=600 if ext == 'png' else None, bbox_inches='tight')
    print(f'Saved {filename}.png and {filename}.pdf')
    plt.close(fig)


# ============================================================================
# Generate all plots
# ============================================================================
pt_data = get_plot_data(pt_results)
pf_data = get_plot_data(pf_results)
pt32_data = get_plot_data(pt32_results)
rs32_data = get_plot_data(rs32_results)
dist15_data = get_plot_data(dist15_results)

OUT_DIR = 'PF_res'
import os
os.makedirs(OUT_DIR, exist_ok=True)

# Pull efficiency curves for PF, RS32f, and 15dist
print('Pulling epoch curves for Pathfinder...')
pf_efficiency = pull_efficiency(pf_runs, pf_results)
print(f'  Got {len(pf_efficiency)} efficiency curves')
print('Pulling epoch curves for PathTracker Restyled 32f...')
rs32_efficiency = pull_efficiency(rs32_runs, rs32_results)
print(f'  Got {len(rs32_efficiency)} efficiency curves')
print('Pulling epoch curves for 15_dist...')
dist15_efficiency = pull_efficiency(dist15_runs, dist15_results)
print(f'  Got {len(dist15_efficiency)} efficiency curves')

for name, data, label, efficiency in [
    ('pathtracker64f', pt_data, 'PathTracker 64-frame', None),
    ('pathfinder', pf_data, 'Pathfinder 128px', pf_efficiency),
    ('pathtracker32f', pt32_data, 'PathTracker 32-frame', None),
    ('pathtracker_restyled_32f', rs32_data, 'PathTracker Restyled 32-frame', rs32_efficiency),
    ('15dist', dist15_data, '15-Distractor PathTracker', dist15_efficiency),
]:
    if not data:
        print(f'No data for {name}, skipping')
        continue

    # Step time plot
    step_data = [d for d in data if d.get('min_ms') is not None]
    if step_data:
        make_figure(step_data, f'{label}: Accuracy vs Step Time',
                    f'{OUT_DIR}/{name}_steptime')

    # Params plot
    params_data = [d for d in data if d.get('params') is not None]
    if params_data:
        make_params_figure(params_data, f'{label}: Accuracy vs Parameters',
                           f'{OUT_DIR}/{name}_params')

    # Training cost plot (step_ms * epochs)
    cost_data = [d for d in data if d.get('min_ms') is not None]
    for d in cost_data:
        d['train_cost'] = d['min_ms'] * (d['epoch'] or 60) / 1000.0
    if cost_data:
        make_figure(cost_data, f'{label}: Accuracy vs Training Cost',
                    f'{OUT_DIR}/{name}_traincost', x_key='train_cost',
                    x_label='Training Cost (step_ms $\\times$ epochs / 1000)')

    # Sample efficiency plot (epochs to 95% of peak vs peak accuracy)
    if efficiency:
        make_efficiency_figure(data, efficiency,
                               f'{label}: Learning Speed vs Peak Accuracy',
                               f'{OUT_DIR}/{name}_efficiency')

    # Ranked bar chart
    sorted_data = sorted(data, key=lambda x: x['acc'], reverse=True)
    fig, ax = plt.subplots(figsize=(7, max(len(sorted_data) * 0.32, 4)))
    names = [human_label(d['name']) for d in sorted_data]
    accs = [d['acc'] for d in sorted_data]
    colors = [COLORS.get(d['family'], '#999') for d in sorted_data]

    bars = ax.barh(range(len(names)), accs, color=colors, alpha=0.85, height=0.7,
                   edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Best Validation Accuracy')
    ax.set_title(f'{label}: Model Ranking', fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1.0)
    sns.despine(ax=ax)

    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(acc + 0.003, i, f'{acc:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f'{OUT_DIR}/{name}_ranked.{ext}', dpi=300 if ext == 'png' else None, bbox_inches='tight')
    print(f'Saved {OUT_DIR}/{name}_ranked.png and {OUT_DIR}/{name}_ranked.pdf')
    plt.close(fig)
