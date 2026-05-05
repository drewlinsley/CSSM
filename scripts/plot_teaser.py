#!/usr/bin/env python3
"""Single-panel paper teaser: best accuracy vs. step time on PathTracker.

Plots one dot per model (best run across the W&B project), draws a smooth
Pareto frontier through the baselines, and highlights the sCSSM family
*above* the baseline frontier with an annotation arrow — the visual claim
being that the spectral approach "breaks the tradeoff" between fast scans
(lower-left baselines) and accurate sequence mixers (upper-right baselines).

Uses the shared paper-figure style (`scripts/_plot_style.py`).

Usage:
    source activate.sh && python scripts/plot_teaser.py
    source activate.sh && python scripts/plot_teaser.py --project CSSM_pathfinder
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import (apply_style, COLORS, FAMILY_COLORS, MODEL_NAMES,
                         FAMILIES, should_skip_run, text_color_for)  # noqa: E402

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
import seaborn as sns
import wandb

apply_style()


# Models whose best operating point we highlight as our method.
OURS = set(FAMILIES['Spectral SSM'])  # GDN-sCSSM, Mamba-sCSSM, sCSSM

# Dots shown on the figure. Default = every model in every family,
# minus models the user explicitly asked us to suppress.
HIDDEN_MODELS = {
    'CSSM',         # cssm_type='conv_ssm' — chance-level on these tasks
    'S4ND-Diag',    # cssm_type='s4nd' — non-canonical S4D-per-axis simplification
                    #                    (paper-faithful S4ND lives in 's4nd_full')
    # 'Mamba-ConvS5' fully removed from MODEL_NAMES/COLORS/FAMILIES — no longer
    # needs to be explicitly hidden.
}
VISIBLE_MODELS = ({m for fam in FAMILIES.values() for m in fam}
                  - HIDDEN_MODELS)


def _epochs_to(target_acc, run):
    """Approximate epochs-to-target by walking history (slow). None if not reached."""
    try:
        hist = run.history(keys=['epoch', 'val/acc'], samples=300, pandas=True)
    except Exception:
        return None
    for _, row in hist.iterrows():
        e = row.get('epoch')
        a = row.get('val/acc')
        if e is not None and a is not None and a >= target_acc:
            return int(e)
    return None


_TEASER_SPECTRAL_TYPES = {'gated', 'gdn', 'no_gate', 'transformer'}


def _is_fcssm_control(r):
    """Spectral sCSSM control ablation: kernel_size==1 (no spatial mixing) or
    seq_len==1 (no temporal recurrence). Excluded from the teaser entirely
    — controls live in the 3×2 figure, not in the headline teaser."""
    cfg = r.config
    if cfg.get('cssm') not in _TEASER_SPECTRAL_TYPES:
        return False
    return cfg.get('kernel_size') == 1 or cfg.get('seq_len') == 1


def fetch_runs(project: str, difficulty=None, fetch_epochs=False):
    """Return a list of all runs with the fields we need for the teaser:
    model name, accuracy, step_ms, params, optional epochs_to_95.
    """
    api = wandb.Api()
    runs = api.runs(f'serrelab/{project}')
    out = []
    for r in runs:
        if should_skip_run(r):
            continue
        if _is_fcssm_control(r):
            continue
        acc = r.summary.get('best_val_acc', r.summary.get('val_acc', None))
        if acc is None or acc < 0.49:
            continue
        if difficulty is not None:
            d = r.config.get('pathfinder_difficulty', '?')
            if str(d) != str(difficulty):
                continue
        cssm = r.config.get('cssm', '?')
        name = MODEL_NAMES.get(cssm)
        if name is None:
            continue
        if cssm == 'gdn' and 'int' in r.name:
            continue
        step_ms = r.summary.get('timing/step_ms',
                                r.summary.get('timing/epoch_avg_step_ms', None))
        params = r.config.get('num_params')
        record = {
            'model': name, 'acc': acc * 100,
            'step_ms': step_ms,
            'params': params,
            'epochs_to_95': None,
        }
        if fetch_epochs:
            record['epochs_to_95'] = _epochs_to(0.95 * acc, r)
        out.append(record)
    return out


def best_per_model(records):
    best = {}
    for r in records:
        m = r['model']
        if m not in best or r['acc'] > best[m]['acc']:
            best[m] = r
    return list(best.values())


def baseline_pareto(records, x_key, include=None, exclude=None, min_acc=0.0):
    """Pareto-efficient frontier (minimize x_key, maximize acc).

    Walks the cloud left→right keeping only points that strictly improve y
    over the running max — the standard Pareto definition. Then extends the
    front horizontally to the rightmost x in the pool so the curve spans
    the full step-time range and visually reads as the achievable ceiling
    rather than a single dominant point.

    `include` keeps only records whose model is in this set (None = keep all).
    `exclude` drops records whose model is in this set.
    `min_acc` drops sub-threshold runs before computing the front.
    """
    exclude = exclude or set()
    pool = sorted([(r[x_key], r['acc']) for r in records
                   if (include is None or r['model'] in include)
                   and r['model'] not in exclude
                   and r.get(x_key) not in (None, 0)
                   and r['acc'] >= min_acc],
                  key=lambda p: (p[0], -p[1]))
    if not pool:
        return []
    upper = []
    best_y = -float('inf')
    for x, y in pool:
        if y > best_y:
            upper.append((x, y))
            best_y = y
    # Horizontal extension to the right edge of the pool ("ceiling").
    max_x = pool[-1][0]
    if upper and upper[-1][0] < max_x:
        upper.append((max_x, best_y))
    return upper


def _ours_ellipse(ours_pts, pad_log_x=0.22, pad_y=2.5):
    """Smooth ellipse encompassing the sCSSM-family dots in (log10 x, y) space."""
    if len(ours_pts) < 2:
        return None
    log_xs = np.log10([p[0] for p in ours_pts])
    ys = np.array([p[1] for p in ours_pts])

    cx_log = 0.5 * (log_xs.min() + log_xs.max())
    cy = 0.5 * (ys.min() + ys.max())
    half_w_log = 0.5 * (log_xs.max() - log_xs.min()) + pad_log_x
    half_h = 0.5 * (ys.max() - ys.min()) + pad_y

    theta = np.linspace(0, 2 * np.pi, 200)
    ex_log = cx_log + half_w_log * np.cos(theta)
    ey = cy + half_h * np.sin(theta)
    return np.column_stack([10 ** ex_log, ey]), (10 ** cx_log, cy)


X_KEY_LABELS = {
    'step_ms':      'Step time (ms, log scale)',
    'params':       'Parameters (log scale)',
    'epochs_to_95': 'Epochs to 95% peak (log scale)',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', default='CSSM_15dist',
                    help='W&B project (default: PathTracker 15-dist).')
    ap.add_argument('--difficulty', default=None,
                    help='Optional pathfinder_difficulty filter (e.g. "14").')
    ap.add_argument('--task-label', default='PathTracker (32-frame, 15-distractor)',
                    help='Title text for the figure.')
    ap.add_argument('--human-acc', type=float, default=None,
                    help='Human accuracy (%) — drawn as a horizontal reference line.')
    ap.add_argument('--pareto-front', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='Draw the baseline (non-Ours) Pareto front as a grey curve. '
                         '(default: on; pass --no-pareto-front to disable.)')
    ap.add_argument('--x-key', choices=list(X_KEY_LABELS.keys()),
                    default='step_ms',
                    help='Variable on the x-axis.')
    ap.add_argument('--all-runs', action='store_true',
                    help='Scatter every run (faint dots) in addition to '
                         'best-per-model. Mirrors the main 3x2 figure style.')
    ap.add_argument('--out', default='teaser.png')
    args = ap.parse_args()

    fetch_epochs = (args.x_key == 'epochs_to_95')
    print(f'Fetching {args.project}{"  + epochs" if fetch_epochs else ""}...')
    all_records = fetch_runs(args.project, args.difficulty, fetch_epochs=fetch_epochs)
    # Drop rows with no x-key value (otherwise nothing to plot for them).
    records = [r for r in all_records if r.get(args.x_key) not in (None, 0)]
    # Display only transformers + mambas + our spectral family. Conv-SSM
    # baselines are kept in `records` (so the Pareto front sees them) but
    # excluded from `visible` (so they don't appear as dots).
    visible = [r for r in records if r['model'] in VISIBLE_MODELS]
    best = best_per_model(visible)
    print(f'  {len(records)} runs / {len(visible)} visible / '
          f'{len(best)} models shown with x={args.x_key}')
    x_key = args.x_key
    for r in sorted(best, key=lambda x: x[x_key]):
        print(f"    {r['model']:18s}  {x_key}={r[x_key]}  acc={r['acc']:.2f}%")

    # ── Plot ────────────────────────────────────────────────────────────────
    # Faint scatter for every visible run, two Pareto fronts, legend
    # listing every model type. No big markers, no per-dot annotations.
    # Square data axes (`set_box_aspect(1)`) below so log-x and linear-y
    # render in matched pixel extent regardless of figure aspect.
    fig, ax = plt.subplots(1, 1, figsize=(14.0, 11.0))
    # Big text for the teaser; rcParams are global so override in-place
    # rather than mutating the shared style. Sizes are ~2× the previous
    # paper-figure defaults so the teaser reads at slide / poster scale.
    ax.tick_params(axis='both', which='major', labelsize=36)

    # Note: per-run skip logic (spectral-family ks=1 / seq_len=1 ablations,
    # crashed runs) already happened in fetch_runs via `should_skip_run`.
    # No additional model-level acc filter here — every legitimate
    # architecture, including failed-on-this-task ones like Transformer,
    # appears in the figure and legend.
    surviving_models = {r['model'] for r in visible}

    # 1. All-runs scatter (faint dots, like the 2x3 figure).
    if args.all_runs:
        for r in visible:
            ax.scatter(r[x_key], r['acc'],
                       s=260, color=COLORS[r['model']],
                       edgecolors='black', linewidths=0.5,
                       alpha=0.5, zorder=3)

    # 4. Chance + human reference lines.
    ax.axhline(y=50, color='gray', linestyle='--',
               alpha=0.4, linewidth=0.8, zorder=1)
    if args.human_acc is not None:
        ax.axhline(y=args.human_acc, color='black', linestyle='--',
                   alpha=0.6, linewidth=1.0, zorder=2)

    # 5. Two Pareto fronts:
    #    (a) grey curve through every non-Ours run (the baseline frontier)
    #    (b) blue curve through every sCSSM-family run (our frontier)
    def _draw_front(pts, color, lw, alpha, zorder, label=None):
        if len(pts) < 2:
            return
        fx, fy = zip(*pts)
        fx = np.array(fx); fy = np.array(fy)
        log_fx = np.log10(np.maximum(fx, 1e-12))
        # PCHIP needs strictly increasing x. Deduplicate / nudge any
        # repeats so the interpolator does not raise.
        if (np.diff(log_fx) <= 0).any():
            keep = np.concatenate([[True], np.diff(log_fx) > 0])
            log_fx = log_fx[keep]
            fy = fy[keep]
        kw = dict(color=color, linewidth=lw, alpha=alpha, linestyle='-',
                  zorder=zorder, solid_capstyle='round',
                  solid_joinstyle='round', label=label)
        if len(log_fx) >= 3:
            interp = PchipInterpolator(log_fx, fy, extrapolate=False)
            xs = np.linspace(log_fx[0], log_fx[-1], 800)
            ax.plot(10 ** xs, interp(xs), **kw)
        else:
            ax.plot(10 ** log_fx, fy, **kw)

    if args.pareto_front:
        # Pareto pool: visible runs only. HIDDEN_MODELS (e.g. S4ND-Diag,
        # CSSM) are excluded from BOTH the scatter and the Pareto so the
        # grey curve never reaches a dot that isn't drawn. This keeps the
        # visible scatter and the visible Pareto consistent. The blue
        # sCSSM curve still gets `min_acc=70` to avoid a near-vertical
        # takeoff from a single low-acc spectral run at very small x.
        pareto_pool = [r for r in records if r['model'] in surviving_models]
        baseline_front = baseline_pareto(pareto_pool, x_key,
                                         exclude=OURS, min_acc=0.0)
        ours_front = baseline_pareto(pareto_pool, x_key,
                                     include=OURS, min_acc=70.0)
        print(f'  baseline Pareto: {len(baseline_front)} vertices')
        print(f'  ours Pareto:     {len(ours_front)} vertices')
        _draw_front(baseline_front, color='#555555', lw=4.5, alpha=0.9,
                    zorder=4, label='Baseline Pareto')
        _draw_front(ours_front, color=FAMILY_COLORS['Spectral SSM'],
                    lw=5.0, alpha=0.95, zorder=5, label='sCSSM Pareto')

    ax.set_xscale('log')
    ax.set_xlabel(X_KEY_LABELS[x_key], fontsize=40)
    ax.set_ylabel('Accuracy (%)', fontsize=40)
    ax.set_title(args.task_label, fontsize=46, fontweight='bold', pad=24)
    ax.set_box_aspect(1.0)  # square data axes — visual aspect locked to 1:1
    # Common y-range across both teasers (Pathfinder + PathTracker) so the
    # two figures sit at the same scale when shown side-by-side.
    ax.set_ylim(47, 93)
    sns.despine(ax=ax)

    # 6. Right-edge "chance" / "human" text annotations.
    x_lo, x_hi = ax.get_xlim()
    ax.text(x_hi * 0.98, 50.9, 'chance',
            ha='right', va='bottom', fontsize=28, color='#888',
            style='italic', zorder=2)
    if args.human_acc is not None:
        ax.text(x_hi * 0.98, args.human_acc + 0.8, 'Human',
                ha='right', va='bottom', fontsize=28, color='black',
                alpha=0.75, zorder=3)

    # 7. Legend on the right: every visible model grouped by family,
    #    plus the two Pareto fronts at the bottom.
    legend_handles = []
    seen = set()
    for fam_name, fam_models in FAMILIES.items():
        for m in fam_models:
            if m in surviving_models and m not in seen:
                seen.add(m)
                legend_handles.append(mlines.Line2D(
                    [], [], color=COLORS[m], marker='o',
                    linestyle='None', markersize=20,
                    markeredgecolor='black', markeredgewidth=0.6,
                    label=m))
    if args.pareto_front:
        legend_handles.append(mlines.Line2D(
            [], [], color=FAMILY_COLORS['Spectral SSM'], linewidth=5.0,
            solid_capstyle='round', label='sCSSM Pareto'))
        legend_handles.append(mlines.Line2D(
            [], [], color='#555555', linewidth=4.5,
            solid_capstyle='round', label='Baseline Pareto'))
    ax.legend(handles=legend_handles, loc='center left',
              bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=32, handletextpad=0.8, borderaxespad=0.0,
              labelspacing=0.65)

    plt.tight_layout()
    base, _ = os.path.splitext(args.out)
    plt.savefig(base + '.png', dpi=200, bbox_inches='tight')
    plt.savefig(base + '.pdf', dpi=600, bbox_inches='tight')
    print(f'Saved {base}.png and {base}.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
