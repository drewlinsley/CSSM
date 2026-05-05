#!/usr/bin/env python3
"""ImageNet size-vs-performance teaser.

Single panel, log-x parameter count, linear-y ImageNet-1k top-1 accuracy.
Combines:
  - Published SHViT-S1 / SHViT-S2 numbers from
    https://github.com/ysj9909/SHViT (the original paper).
  - Our jax/flax SHViT-S1 reproduction (W&B project `cssm-imagenet-edge`).
  - Three CSSM-SHViT-S1 variants with `cssm='gdn'` at T=1, 4, 8 from the
    same project (param counts measured locally; accuracy from W&B).

T=4 is still training as of this script's creation; its bar is drawn open
(no fill) and explicitly tagged "(running)" so it isn't mistaken for a
final number.

Style is shared via `scripts/_plot_style.py` (Roboto, sans, despine).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import apply_style  # noqa: E402

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import seaborn as sns

apply_style()


# (label, params_M, acc_pct, group, status)
#   group ∈ {'paper', 'ours_shvit', 'ours_cssm'}
#   status ∈ {'final', 'running'}  — 'running' = open marker + italic label
DATA = [
    # (label, params_M, acc_pct, group, status, T)  — T only used for cssm dots
    ('SHViT-S1 (paper)',              6.3,   72.8,  'paper',      'final',   None),
    ('SHViT-S2 (paper)',              11.4,  75.2,  'paper',      'final',   None),
    # Our flax port of SHViT-S1 (4-stage Conv+Attn hybrid, no partial_dim).
    ('Hybrid\nSHViT-S1',               5.412, 73.78, 'ours_shvit', 'final',   None),
    # GDN-sCSSM-SHViT-S1 family. Three T values share one architecture
    # — params are reported uniformly at the canonical 5.785M so the
    # three dots stack at one x position (the sub-1% spread between
    # configs is from learnable per-T parameters, not the backbone).
    ('GDN-sCSSM-SHViT-S1 T=1',        5.785, 74.62, 'ours_cssm',  'final',   1),
    # T=4 from `cssm_gdn_T4_k3_v5cfg` (300ep finished, EMA-best).
    ('GDN-sCSSM-SHViT-S1 T=4',        5.785, 74.81, 'ours_cssm',  'final',   4),
    ('GDN-sCSSM-SHViT-S1 T=8',        5.785, 75.03, 'ours_cssm',  'final',   8),
]

# Light → dark gradient for the T=1/T=4/T=8 dots (Material Blue palette).
# All cssm dots share the same dark-blue *outline*; only the *fill*
# changes to encode T.
T_FILL = {
    1: '#90CAF9',   # light
    4: '#42A5F5',   # medium
    8: '#1565C0',   # deep (matches outline)
}

GROUP_STYLE = {
    'paper':      dict(color='#888888', marker='o', size=720),
    'ours_shvit': dict(color='#2E7D32', marker='o', size=800),
    'ours_cssm':  dict(color='#1565C0', marker='o', size=950),
}


def main():
    # Square data axes + outsized teaser-style typography. Smaller figsize
    # with large fontsizes makes the text/markers read prominently relative
    # to the data region.
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 10.0))
    ax.tick_params(axis='both', which='major', labelsize=40)

    # 1. Faint connecting line for the published SHViT curve (S1 → S2).
    paper_pts = sorted(((p, a) for l, p, a, g, s, t in DATA if g == 'paper'),
                       key=lambda t: t[0])
    if len(paper_pts) >= 2:
        xs = [p[0] for p in paper_pts]
        ys = [p[1] for p in paper_pts]
        ax.plot(xs, ys, color='#aaaaaa', linewidth=1.8, linestyle='--',
                alpha=0.75, zorder=1)

    # 2. Faint connecting line for the cssm T-sweep (T=1 → 4 → 8).
    cssm_pts = [(p, a) for l, p, a, g, s, t in DATA if g == 'ours_cssm']
    cssm_pts.sort(key=lambda t: t[0])
    if len(cssm_pts) >= 2:
        ax.plot([p[0] for p in cssm_pts], [p[1] for p in cssm_pts],
                color='#1565C0', linewidth=1.8, linestyle=':',
                alpha=0.7, zorder=1)

    # 3. Dots. cssm dots take a graduated fill colour from T_FILL keyed by
    # their T value; non-cssm dots keep their group fill.
    for label, params, acc, group, status, T in DATA:
        style = GROUP_STYLE[group]
        if group == 'ours_cssm' and T in T_FILL:
            fill = T_FILL[T]
        else:
            fill = style['color']
        ax.scatter(params, acc,
                   s=style['size'],
                   marker=style['marker'],
                   facecolor=fill,
                   edgecolor=style['color'],
                   linewidths=2.0,
                   alpha=0.95, zorder=10)

    # 4. Per-dot labels. The three cssm dots are stacked at ~5.8M, so we
    #    place each label to the right at its own y so they don't overlap.
    label_offsets = {
        # (dx_factor in log-x, dy in pp, halign)
        'SHViT-S1 (paper)':              (1.012, -0.22, 'left'),
        # SHViT-S2 is at x=11.4 with xlim right edge 12.5 — place the
        # label to the LEFT of the dot (ha='right') so the text box
        # stays inside the axes.
        'SHViT-S2 (paper)':              (0.988, +0.22, 'right'),
        # Pushed further right of the green dot for clear separation. y=73.78
        # is below the GDN-sCSSM stack (y≈74.6-75) and above the SHViT-S1
        # paper dot (y=72.8), so the label lands in vertical empty space.
        'Hybrid\nSHViT-S1':               (1.06, 0.00, 'left'),
        # GDN-sCSSM dots are stacked at ~5.78M; right-align for clean read.
        'GDN-sCSSM-SHViT-S1 T=1':        (1.05, -0.07, 'left'),
        'GDN-sCSSM-SHViT-S1 T=4':        (1.05, +0.0,  'left'),
        'GDN-sCSSM-SHViT-S1 T=8':        (1.05, +0.07, 'left'),
    }
    for label, params, acc, group, status, T in DATA:
        style = GROUP_STYLE[group]
        dx_factor, dy, ha = label_offsets.get(label, (1.04, 0.0, 'left'))
        ax.text(params * dx_factor, acc + dy, label,
                fontsize=23,
                fontweight='bold' if group == 'ours_cssm' else 'normal',
                color=style['color'],
                style='italic' if status == 'running' else 'normal',
                ha=ha, va='center', zorder=11)

    # Linear x — the data only spans 5.4 → 11.4 M, so a log axis is
    # unnecessary and misleading at this scale.
    ax.set_xlabel('Model Parameters (M)', fontsize=44)
    ax.set_ylabel('ImageNet-1k top-1 (%)', fontsize=44)
    ax.set_box_aspect(1.0)  # square data axes — same as teaser
    sns.despine(ax=ax)

    ax.set_xlim(5.0, 12.0)
    ax.set_ylim(72.3, 75.6)

    ax.xaxis.set_major_locator(FixedLocator([5, 6, 7, 8, 9, 10, 11, 12]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{int(round(v))}'))
    ax.xaxis.set_minor_formatter(NullFormatter())

    plt.tight_layout()
    plt.savefig('imagenet_size_perf.png', dpi=200, bbox_inches='tight')
    plt.savefig('imagenet_size_perf.pdf', dpi=600, bbox_inches='tight')
    print('Saved imagenet_size_perf.png / .pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
