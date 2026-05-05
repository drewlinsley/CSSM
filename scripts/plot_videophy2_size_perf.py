#!/usr/bin/env python3
"""VideoPhy-2 size-vs-joint-score teaser.

Single panel, linear-x parameter count, linear-y VideoPhy-2 joint score
(PC>=4 AND SA>=4 per prompt; mPLUG-Owl autograder).

Compares CogVideoX-5B LoRA finetunes on the OpenVid-1M curated subset:
  - Base CogVideoX-5B (no LoRA): horizontal reference line.
  - Standard LoRA (rank 16, attn1 q/k/v/o targets): grey baseline.
  - Transformer LoRA (rank 16, transformer adapter): red baseline.
  - Our GDN-sCSSM LoRA at k=1 and k=5: blue, with kernel-size gradient fill.

Style is shared via `scripts/_plot_style.py` (Raleway, sans, despine).
Approach mirrors `scripts/plot_imagenet_size_perf.py`.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import apply_style  # noqa: E402

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import seaborn as sns

apply_style()


RESULTS_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'benchmarks', 'results', 'openvid_videophy2_results.json',
)


# Group → fill/edge color, marker, scatter size.
GROUP_STYLE = {
    'base':         dict(color='#888888', marker='o', size=720),
    'baseline_std': dict(color='#888888', marker='o', size=800),  # grey   — std LoRA
    'baseline_tx':  dict(color='#C62828', marker='o', size=800),  # red    — transformer LoRA
    'ours_cssm':    dict(color='#1565C0', marker='o', size=950),  # blue   — GDN-sCSSM LoRA
}

# Light → dark gradient for our k=1/k=5 dots (Material Blue).
K_FILL = {
    1: '#90CAF9',   # light
    5: '#1565C0',   # dark (matches outline)
}


def load_data():
    with open(RESULTS_JSON) as f:
        d = json.load(f)
    v = d['variants']
    # (key, label, params_M, joint_pct, group, k)
    return [
        ('base',
         'CogVideoX-5B base',
         None,                                v['base']['joint_pct'],
         'base',         None),
        ('standard_r16',
         'Standard\nLoRA',
         v['standard_r16']['params_M'],       v['standard_r16']['joint_pct'],
         'baseline_std', None),
        ('transformer_r16',
         'Transformer\nLoRA',
         v['transformer_r16']['params_M'],    v['transformer_r16']['joint_pct'],
         'baseline_tx',  None),
        ('gdn_cssm_r16_k1',
         'GDN-sCSSM\nLoRA',
         v['gdn_cssm_r16_k1']['params_M'],    v['gdn_cssm_r16_k1']['joint_pct'],
         'ours_cssm',    1),
    ]


def main():
    rows = load_data()

    # Square data axes + outsized teaser-style typography. Smaller figsize
    # with large fontsizes makes the text/markers read prominently relative
    # to the data region.
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 10.0))
    ax.tick_params(axis='both', which='major', labelsize=40)

    # 1. Horizontal reference line for the no-LoRA base score.
    base_acc = next(r[3] for r in rows if r[4] == 'base')
    ax.axhline(y=base_acc, color='#888888', linestyle='--',
               linewidth=1.5, alpha=0.7, zorder=1)

    # 2. Faint connecting line for our k-sweep (k=1 → k=5). Both share the
    #    same x, so the line is vertical and reads as a "kernel ramp" guide.
    cssm_pts = sorted(
        ((r[2], r[3]) for r in rows if r[4] == 'ours_cssm'),
        key=lambda t: t[0])
    if len(cssm_pts) >= 2:
        ax.plot([p[0] for p in cssm_pts], [p[1] for p in cssm_pts],
                color='#1565C0', linewidth=1.8, linestyle=':',
                alpha=0.7, zorder=1)

    # 3. Dots. cssm dots take a graduated fill colour from K_FILL keyed by k.
    for key, label, params, joint, group, k in rows:
        if params is None:
            continue
        style = GROUP_STYLE[group]
        if group == 'ours_cssm' and k in K_FILL:
            fill = K_FILL[k]
        else:
            fill = style['color']
        ax.scatter(params, joint,
                   s=style['size'], marker=style['marker'],
                   facecolor=fill, edgecolor=style['color'],
                   linewidths=2.0, alpha=0.95, zorder=10)

    # 4. Per-dot labels. Positions chosen so no label overlaps another label,
    #    a dot, or the dashed base reference line at y=24.027.
    label_offsets = {
        # (dx_factor, dy_pp, halign)
        # Standard/Transformer dots have *identical* joint scores (23.858) and
        # near-identical x (16.515 vs 16.644). Place the labels on opposite
        # sides — Standard to the LEFT of its grey dot (ha='right'),
        # Transformer to the RIGHT of its red dot (ha='left'). Both sit at
        # the dots' own y (23.858), which is already below the dashed base
        # line at y=24.027.
        'Standard\nLoRA':         (0.992, 0.00, 'right'),
        'Transformer\nLoRA':      (1.008, 0.00, 'left'),
        # GDN-sCSSM dot at ~14.51M — label to the *right* of the dot.
        'GDN-sCSSM\nLoRA':  (1.012, +0.00, 'left'),
    }
    for key, label, params, joint, group, k in rows:
        if params is None:
            continue
        style = GROUP_STYLE[group]
        dx_factor, dy, ha = label_offsets.get(label, (1.02, 0.0, 'left'))
        ax.text(params * dx_factor, joint + dy, label,
                fontsize=23,
                fontweight='bold' if group == 'ours_cssm' else 'normal',
                color=style['color'],
                ha=ha, va='center', zorder=11)

    # Base reference annotation. Placed mid-axis on the dashed line, with a
    # white bbox so the line reads "broken" through the text rather than
    # crossing the glyphs. x=15.05 is the empty region between the GDN-sCSSM
    # stack (x=14.51) and the LoRA-baseline pair (x≈16.5).
    ax.text(15.05, base_acc, 'CogVideoX-5B base',
            fontsize=19, color='#888888', style='italic',
            ha='center', va='center', zorder=2,
            bbox=dict(facecolor='white', edgecolor='none', pad=2))

    ax.set_xlabel('LoRA Parameters (M)', fontsize=44)
    ax.set_ylabel('VideoPhy-2 joint score (%)', fontsize=44)
    ax.set_box_aspect(1.0)
    sns.despine(ax=ax)

    ax.set_xlim(13.9, 17.6)
    ax.set_ylim(23.7, 25.2)

    ax.xaxis.set_major_locator(FixedLocator([14, 15, 16, 17]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{int(round(v))}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:.1f}'))

    plt.tight_layout()
    plt.savefig('videophy2_size_perf.png', dpi=200, bbox_inches='tight')
    plt.savefig('videophy2_size_perf.pdf', dpi=600, bbox_inches='tight')
    print('Saved videophy2_size_perf.png / .pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
