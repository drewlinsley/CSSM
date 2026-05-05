#!/usr/bin/env python3
"""
Main-text 3x2 figure: step time, efficiency, params x {Pathfinder, PathTracker}.

Shows ALL run variations as small transparent dots.
Best-of-each-FAMILY annotated with labels (adjustText for no overlap).
Pareto fronts drawn per family (only families with best acc > threshold).

Usage:
    source activate.sh && python scripts/plot_main_3x2.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import apply_style, TEXT_COLORS, should_skip_run  # noqa: E402

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from scipy.interpolate import PchipInterpolator
import seaborn as sns
import wandb
from adjustText import adjust_text
from collections import Counter

apply_style()

# ── Model matching ───────────────────────────────────────────────────────────
# Naming conventions (matched to benchmarks/bench_image_timing.py):
#   - "sCSSM" = Fourier Conv SSM (parallel conv RNN). Variants prepended:
#     Mamba-sCSSM (gated), GDN-sCSSM (gated DeltaNet).
#   - "Transformer" = self-attention (the "spatial" qualifier is dropped now
#     that we plot the spatiotemporal variant as Transformer-ST).
#   - "GDN" = Gated DeltaNet on flattened HW (the sequence baseline; the
#     spectral version is GDN-sCSSM).
# Spectral sCSSM family cssm_types and their two structural ablation axes:
#   ks=1       → spatial mixer ablated  ("NoSpace")
#   seq_len=1  → temporal scan ablated  ("NoTime")
# `no_gate` is *not* a control — it's the vanilla / un-gated sCSSM that
# `gated` and `gdn` extend with gating mechanisms.
_SPECTRAL_TYPES = {'gated', 'gdn', 'no_gate', 'transformer'}

def _is_fcssm_nospace(r):
    """ks=1 spatial-mixer ablation, restricted to `cssm='no_gate'`.

    Gated variants (Mamba-sCSSM, GDN-sCSSM) at ks=1 still solve the task
    via per-pixel channel mixing in their `in_proj`, so they aren't true
    "no spatial mixing" controls — only the un-gated `no_gate` variant
    cleanly removes spatial information when ks=1.

    Doubly-ablated runs (no_gate ks=1 AND seq_len=1) fall here too — ks=1
    is the more fundamental cut."""
    return (r.config.get('cssm') == 'no_gate'
            and r.config.get('kernel_size') == 1)

def _is_fcssm_notime(r):
    """seq_len=1 temporal-scan ablation. Restricted to `cssm='no_gate'` so
    the control is genuinely crippled (gated variants still have per-pixel
    channel mixing that masks the no-recurrence ablation). Excludes the
    ks=1 case so it isn't double-counted as NoSpace."""
    return (r.config.get('cssm') == 'no_gate'
            and r.config.get('seq_len') == 1
            and r.config.get('kernel_size') != 1)

def _is_fcssm_control(r):
    return _is_fcssm_nospace(r) or _is_fcssm_notime(r)


MODELS = {
    'GDN-sCSSM':       lambda r: r.config.get('cssm') == 'gdn' and 'int' not in r.name and not _is_fcssm_control(r),
    'Mamba-sCSSM':     lambda r: r.config.get('cssm') == 'gated' and r.config.get('short_conv_spatial_size', 3) != 0 and not _is_fcssm_control(r),
    'sCSSM':           lambda r: r.config.get('cssm') == 'no_gate' and r.config.get('kernel_size', 0) >= 3 and r.config.get('num_params', 0) > 5000 and not _is_fcssm_control(r),
    # Two structural ablations of the spectral kernel — each gets its own
    # series so the failure modes are visible separately.
    'sCSSM-NoSpace':   lambda r: _is_fcssm_nospace(r),
    'sCSSM-NoTime':    lambda r: _is_fcssm_notime(r),
    # 'Mamba-ConvS5' (cssm='no_fft') intentionally omitted — user request.
    # 'CSSM' (cssm='conv_ssm') intentionally omitted — user request.
    'ConvS5':          lambda r: r.config.get('cssm') == 'convs5',
    # S4ND: paper-faithful HiPPO-LegS DPLR per axis (sweep_s4nd_full.sh).
    # The `s4nd` cssm_type (S4D-per-axis simplification) is intentionally not
    # plotted — it's a non-canonical baseline.
    'S4ND':            lambda r: r.config.get('cssm') == 's4nd_full',
    'Transformer':     lambda r: r.config.get('cssm') == 'spatial_attn',
    'Transformer-ST':  lambda r: r.config.get('cssm') == 'spatiotemporal_attn',
    'Mamba-2':          lambda r: r.config.get('cssm') == 'mamba2_seq',
    'GDN':             lambda r: r.config.get('cssm') == 'gdn_seq',
}

COLORS = {
    'GDN-sCSSM':       '#1565C0',
    'Mamba-sCSSM':     '#42A5F5',
    'sCSSM':           '#90CAF9',  # vanilla / un-gated spectral CSSM
    # Cyan/teal — blue-adjacent but distinct from the three Spectral SSM
    # blues, so the ablation clouds are visually grouped with our family
    # without being confused for a "real" sCSSM variant.
    'sCSSM-NoSpace':   '#F57F17',  # dark yellow (Material Yellow 900) — ks=1 (no spatial mixing)
    'sCSSM-NoTime':    '#FBBC04',  # Google brand yellow — seq_len=1 (no temporal scan)
    # Conv SSM family — three greens, ConvS5 → CSSM → S4ND dark→light.
    # S4ND lives here (not in Sequence SSM) because it's a 2D ConvSSM-style
    # model (HiPPO-LegS DPLR per spatial axis).
    'ConvS5':          '#2E7D32',  # deep green (Material Green 800)
    'S4ND':            '#A5D6A7',  # light green (Material Green 200)
    'Transformer':     '#C62828',
    'Transformer-ST':  '#EF9A9A',
    # Sequence SSM family — three purples, Mamba-2 → GDN → S4ND-Diag dark→light.
    'Mamba-2':         '#4A148C',  # darkest purple
    'GDN':             '#7B1FA2',  # mid-dark purple
}

# Families for Pareto fronts + legend grouping.
# Family name is "Conv SSM" (with a space) to avoid colliding with the
# member-model name "CSSM" (cssm_type='conv_ssm').
FAMILIES = {
    'Spectral SSM':  ['GDN-sCSSM', 'Mamba-sCSSM', 'sCSSM',
                      'sCSSM-NoSpace', 'sCSSM-NoTime'],  # last two are controls
    'Conv SSM':      ['ConvS5', 'S4ND'],
    'Attention':     ['Transformer', 'Transformer-ST'],
    'Sequence SSM':  ['Mamba-2', 'GDN'],
}

FAMILY_COLORS = {
    'Spectral SSM': '#1565C0',
    'Conv SSM':     '#2E7D32',
    'Attention':    '#C62828',
    'Sequence SSM': '#6A1B9A',
}

# Only draw Pareto front / annotate if family best acc > this
PARETO_ACC_THRESHOLD = 0.55


def fetch_all_runs(project: str):
    """Fetch ALL runs with acc > chance, returning every variation."""
    api = wandb.Api()
    runs = api.runs(f'serrelab/{project}')

    all_runs = []
    for r in runs:
        # control_min_acc=0 disables the spectral-sCSSM ks=1 / seq_len=1 acc
        # filter — we WANT to display those controls in the 3×2 figure as
        # 'sCSSM-NoSpace' / 'sCSSM-NoTime' series. Crashed runs are still
        # skipped.
        if should_skip_run(r, control_min_acc=0.0):
            continue
        acc = r.summary.get('best_val_acc', r.summary.get('val_acc', None))
        if acc is None or acc < 0.49:
            continue
        ed = r.config.get('embed_dim', None)
        if ed not in [32, 64]:
            continue

        step_ms = r.summary.get('timing/step_ms', r.summary.get('timing/epoch_avg_step_ms', None))
        params = r.config.get('num_params', None)

        peak = acc
        target = 0.95 * peak
        epochs_to_95 = None
        try:
            hist = r.history(keys=['epoch', 'val/acc'], samples=500, pandas=True)
            for _, row in hist.iterrows():
                e = row.get('epoch', None)
                a = row.get('val/acc', None)
                if e is not None and a is not None and a >= target:
                    epochs_to_95 = int(e)
                    break
        except:
            pass

        for model_name, match_fn in MODELS.items():
            if match_fn(r):
                all_runs.append({
                    'model': model_name,
                    'acc': acc,
                    'params': params,
                    'step_ms': step_ms,
                    'epochs_to_95': epochs_to_95,
                    'dim': ed,
                    'name': r.name,
                })
                break

    return all_runs


def compute_pareto_front(points):
    """Pareto front for minimize-x, maximize-y. Returns sorted (x, y) list."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: p[0])
    front = []
    best_y = -float('inf')
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return front


def _douglas_peucker(points, epsilon):
    """Simplify a polyline by removing vertices closer than `epsilon` to the
    line through their neighbours. Operates on (x, y) tuples already in the
    target plotting space (i.e. log-x where applicable)."""
    if len(points) < 3:
        return list(points)

    def perp_dist(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 == 0:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        # Distance from p to the line through a,b.
        cross = abs(dx * (ay - py) - (ax - px) * dy)
        return cross / (seg2 ** 0.5)

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        # Find the index k in (i, j) farthest from the line points[i]–points[j].
        max_d, max_k = 0.0, -1
        for k in range(i + 1, j):
            d = perp_dist(points[k], points[i], points[j])
            if d > max_d:
                max_d, max_k = d, k
        if max_d > epsilon and max_k > 0:
            keep[max_k] = True
            stack.append((i, max_k))
            stack.append((max_k, j))
    return [p for p, k in zip(points, keep) if k]


def compute_pareto_convex_hull(points, log_x=True, simplify_eps=0.6):
    """Pareto-efficient upper hull through `points` (minimize x, maximize y).

    Uses Andrew's monotone chain in (log10 x, y) space (matches what's
    visually convex on a log-x axis), trims to the Pareto-efficient arc,
    then Douglas–Peucker simplifies the polyline so the smoother sees a
    sparse, kink-free control set. ``simplify_eps`` is in (log10-x, y%)
    units; ~0.6 means drop vertices closer than 0.6 percentage points of
    accuracy from the line through their neighbours, which is well below
    what's visually meaningful.

    Returns a list of (x, y) tuples in original (linear) space.
    """
    if not points or len(points) < 2:
        return list(points)
    if len(points) < 3:
        return compute_pareto_front(points)

    arr = np.array(points, dtype=float)
    xs = arr[:, 0]
    ys = arr[:, 1]
    tx = np.log10(np.maximum(xs, 1e-12)) if log_x else xs

    # 1. FULL CONVEX HULL of all points (in plotting space).
    try:
        from scipy.spatial import ConvexHull
        hull_pts = np.column_stack([tx, ys])
        hull = ConvexHull(hull_pts)
        verts = [(float(tx[i]), float(ys[i])) for i in hull.vertices]
    except Exception:
        verts = list(zip(tx.tolist(), ys.tolist()))

    # 2. STRICTLY-INCREASING-Y SUBSET, sorted by x. Walk the hull vertices
    #    left-to-right; keep a vertex only if its y is strictly above the
    #    running max. Anything else (including same-y plateaus) is dropped.
    verts.sort(key=lambda p: p[0])
    upper = []
    best_y = -float('inf')
    for x, y in verts:
        if y > best_y:
            upper.append((x, y))
            best_y = y

    # Douglas–Peucker simplification only if explicitly requested.
    if simplify_eps and len(upper) > 2:
        upper = _douglas_peucker(upper, simplify_eps)

    if log_x:
        return [(float(10 ** p[0]), float(p[1])) for p in upper]
    return [(float(p[0]), float(p[1])) for p in upper]


def _best_per_family(all_runs):
    """Return dict: family_name -> best run (highest acc) across all members."""
    best = {}
    for family_name, members in FAMILIES.items():
        family_runs = [r for r in all_runs if r['model'] in members]
        if family_runs:
            best[family_name] = max(family_runs, key=lambda r: r['acc'])
    return best


def _best_per_model(all_runs):
    """Return dict: model_name -> best run (highest acc) for that model."""
    best = {}
    for run in all_runs:
        m = run['model']
        if m not in best or run['acc'] > best[m]['acc']:
            best[m] = run
    return best


def _finite_num(v):
    """True iff v is a finite *positive* real number.

    Positive-only, because every panel uses log-x and log(0) = -inf trips
    adjustText/scipy.KDTree downstream (epochs_to_95==0 sneaks through
    otherwise).
    """
    if v is None:
        return False
    try:
        return bool(np.isfinite(v)) and float(v) > 0
    except (TypeError, ValueError):
        return False


def plot_panel(ax, all_runs, x_key, x_label, x_log=False, show_ylabel=True, human_acc=None):
    """Plot one scatter panel with all runs, best-per-MODEL annotations, Pareto front."""
    plotted_models = set()
    best_models = _best_per_model(all_runs)

    # ── Scatter all runs (small, transparent) ────────────────────────────
    for run in all_runs:
        x_val = run.get(x_key)
        if not _finite_num(x_val) or not _finite_num(run['acc']):
            continue
        ax.scatter(
            x_val, run['acc'] * 100,
            color=COLORS[run['model']], marker='o', s=260,
            edgecolors='black', linewidths=0.5,
            alpha=0.5, zorder=3,
        )
        plotted_models.add(run['model'])

    # ── Best-of-each-MODEL: big dot + label ──────────────────────────────
    texts = []
    big_dots = []
    for model_name, run in best_models.items():
        x_val = run.get(x_key)
        if not _finite_num(x_val) or not _finite_num(run['acc']):
            continue
        color = COLORS[model_name]
        text_color = TEXT_COLORS.get(model_name, color)
        big_dots.append(ax.scatter(
            x_val, run['acc'] * 100,
            color=color, marker='o', s=540,
            edgecolors='black', linewidths=0.8,
            alpha=0.95, zorder=10,
        ))
        texts.append(ax.text(
            x_val, run['acc'] * 100, model_name,
            fontsize=28, fontweight='bold', color=text_color,
            zorder=11,
        ))

    # ── Single global Pareto frontier (grey, convex-hull anchored) ───────
    # No accuracy threshold so the bottom-left of the hull can reach the
    # best point of even chance-level models (the user wants those visible
    # at the left edge of the curve). The convex-hull simplification still
    # absorbs noisy runs without producing visible kinks.
    global_points = []
    for run in all_runs:
        x_val = run.get(x_key)
        if _finite_num(x_val) and _finite_num(run['acc']):
            global_points.append((x_val, run['acc'] * 100))
    # 1. Convex hull of all (filtered) points (computed in log-x space).
    # 2. Take strictly-increasing-y subset (Pareto-efficient hull vertices).
    # 3. PCHIP through those vertices: monotone-preserving, so the curve
    #    cannot dip below or overshoot above any vertex — gentle, kink-free.
    front = compute_pareto_convex_hull(global_points, simplify_eps=0)
    if len(front) >= 2:
        fx, fy = zip(*front)
        fx, fy = np.array(fx), np.array(fy)
        log_fx = np.log10(np.maximum(fx, 1e-12))
        # PCHIP requires strictly increasing x; dropping the acc threshold
        # can leave duplicate-x vertices in the hull. Filter those out the
        # same way `plot_teaser._draw_front` does.
        if len(log_fx) >= 2 and (np.diff(log_fx) <= 0).any():
            keep = np.concatenate([[True], np.diff(log_fx) > 0])
            log_fx = log_fx[keep]
            fy = fy[keep]
            fx = fx[keep]
        # Match teaser Pareto styling: thicker, slightly darker grey, with
        # rounded caps and joins so the curve has the same visual weight as
        # the teaser's baseline frontier.
        front_kw = dict(color='#555555', linewidth=4.5, alpha=0.9,
                        linestyle='-', zorder=4,
                        solid_capstyle='round', solid_joinstyle='round',
                        label='_nolegend_')
        if len(fx) >= 3:
            interp = PchipInterpolator(log_fx, fy, extrapolate=False)
            log_xs = np.linspace(log_fx[0], log_fx[-1], 400)
            ax.plot(10 ** log_xs, interp(log_xs), **front_kw)
        else:
            ax.plot(fx, fy, **front_kw)

    # ── Reference lines ──────────────────────────────────────────────────
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4, linewidth=0.8, zorder=1)
    if human_acc is not None:
        ax.axhline(y=human_acc, color='black', linestyle='--', alpha=0.6, linewidth=1.0, zorder=2)
        ax.text(0.98, human_acc + 0.8, 'Human', transform=ax.get_yaxis_transform(),
                fontsize=28, color='black', alpha=0.6, ha='right', va='bottom')

    if x_log:
        ax.set_xscale('log')
    ax.set_xlabel(x_label, fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=36)
    if show_ylabel:
        ax.set_ylabel('Accuracy (%)', fontsize=40)
    ax.set_ylim(44, 100)
    sns.despine(ax=ax)

    # ── adjustText to prevent overlap ────────────────────────────────────
    # Pass `big_dots` as `objects=` so adjustText pushes labels off any of
    # the big best-per-model markers (text-text repulsion alone wasn't
    # enough — labels were landing on top of dots).
    # Debug-print the texts/positions before adjust_text — adjustText's
    # KDTree was crashing on non-finite display-coord bboxes that I
    # couldn't trace from the raw data alone.
    if texts:
        if os.environ.get('DEBUG_3X2'):
            import sys
            ax.figure.canvas.draw()  # force layout so get_window_extent works
            for t in texts:
                bb = t.get_window_extent()
                print(f"  text {t.get_text()!r:30s} pos={t.get_position()}"
                      f" bb=({bb.x0:.2f},{bb.y0:.2f},{bb.x1:.2f},{bb.y1:.2f})",
                      file=sys.stderr)
        # Force a draw first so log-axis transforms are initialised before
        # adjustText's internal explode pass. The display-coord bbox of the
        # `big_dots` PathCollections, when run through `ax.transData.inverted()`
        # under a log-x axis, can produce a non-finite Bbox row when a marker
        # sits very close to the lower-x clip — this is what was crashing
        # scipy.KDTree at fontsize≥82. Drop `objects=` and rely on plain
        # text-text repulsion (the dot positions are still pulled toward
        # via the implicit anchor in original_coords).
        ax.figure.canvas.draw()
        try:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                        expand=(1.4, 1.6),
                        force_text=(0.8, 1.0),
                        ensure_inside_axes=False,
                        iter_lim=300)
        except ValueError as e:
            # Last-resort fallback: render labels at their initial positions
            # without any de-overlap pass.
            import warnings
            warnings.warn(
                f"adjust_text bailed on panel '{x_label}': {e}. "
                f"Rendering labels without overlap-avoidance.")

    return plotted_models


def main():
    print("Fetching Pathfinder CL-14 (all runs)...")
    pf = fetch_all_runs('CSSM_pathfinder')
    print(f"  {len(pf)} total runs")

    print("Fetching 15-dist PathTracker (all runs)...")
    pt = fetch_all_runs('CSSM_15dist')
    print(f"  {len(pt)} total runs")

    for label, data in [('PF-14', pf), ('15dist', pt)]:
        print(f"\n{label}:")
        counts = Counter(r['model'] for r in data)
        for m, c in counts.most_common():
            best = max((r['acc'] for r in data if r['model'] == m), default=0)
            print(f"  {m:25s} runs={c:3d}  best_acc={best:.4f}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(44, 28))

    plotted = set()
    for col, (xk, xl) in enumerate([
        ('step_ms', 'Step Time (ms)'),
        ('epochs_to_95', 'Epochs to 95% Peak'),
        ('params', 'Parameters'),
    ]):
        p = plot_panel(axes[0, col], pf, xk, xl, x_log=True,
                       show_ylabel=(col == 0), human_acc=89)
        plotted |= p
        p = plot_panel(axes[1, col], pt, xk, xl, x_log=True,
                       show_ylabel=(col == 0), human_acc=90)
        plotted |= p

    # Row titles ("PathFinder ...", "PathTracker ...") get extra `pad` so
    # they sit higher above their panels — leaves a clear gap between the
    # row title text and the panel's top edge / x-axis ticks.
    axes[0, 0].set_title('PathFinder 14-dash contours',
                         fontsize=46, fontweight='bold', loc='left', pad=40)
    axes[1, 0].set_title('PathTracker 32-frame 15-distractors',
                         fontsize=46, fontweight='bold', loc='left', pad=40)

    # Column titles (Step Time / Efficiency / Parameters) — primary headers,
    # sized between the row titles (46) and double that. y picked so there's
    # a small gap, but not a yawning one, to the row title beneath.
    for ax, title in zip(axes[0], ['Step Time', 'Efficiency', 'Parameters']):
        ax.annotate(title, xy=(0.5, 1.16), xycoords='axes fraction',
                    fontsize=64, fontweight='bold', ha='center', va='bottom')

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = []
    for family_name, member_models in FAMILIES.items():
        label_tex = family_name.replace(" ", "\\ ")
        handles.append(mpatches.Patch(color='none', label=f'$\\bf{{{label_tex}}}$'))
        for mn in member_models:
            if mn in plotted:
                handles.append(mlines.Line2D([0], [0],
                    marker='o', color='w',
                    markerfacecolor=COLORS[mn],
                    markeredgecolor='black', markeredgewidth=0.4,
                    markersize=20, label=mn, linewidth=0, alpha=0.9))

    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.01, 0.5),
               fontsize=32, frameon=False,
               handletextpad=0.4, labelspacing=0.35, borderpad=0.6)

    plt.subplots_adjust(right=0.82, hspace=0.35, wspace=0.25)
    plt.savefig('main_text_3x2.png', dpi=200, bbox_inches='tight')
    plt.savefig('main_text_3x2.pdf', dpi=600, bbox_inches='tight')
    print("\nSaved to main_text_3x2.png / .pdf")
    plt.close()


if __name__ == '__main__':
    main()
