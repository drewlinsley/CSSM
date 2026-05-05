#!/usr/bin/env python3
"""Image-input timing benchmark for SSM/transformer mixers.

Three log-log panels of step time vs. spatial sequence length N = H*W
(T = 1, single image) with the canonical paper-figure style from
scripts/plot_main_3x2.py.

Panel A: spectral CSSM family vs. attention / Mamba / S5 / S4ND / etc.
Panel B: three spectral circuit variants (Mamba, GDN, Transformer Q/K/A trio).
Panel C: Mamba-SCSSM with embed_dim sweep {64, 128, 256, 512}.

Usage:
    source activate.sh && python benchmarks/bench_image_timing.py \
        --hs 8,16,32,64,128,256 \
        --csv benchmarks/results/bench_image_timing.csv \
        --fig benchmarks/results/bench_image_timing.pdf

    # Re-render the figure from an existing CSV without re-running:
    python benchmarks/bench_image_timing.py \
        --from-csv benchmarks/results/bench_image_timing.csv \
        --fig benchmarks/results/bench_image_timing.pdf
"""

import argparse
import csv
import gc
import os
import sys
import time
import traceback

# Make benchmarks/ + repo root importable when invoked as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import numpy as np

from src.models.simple_cssm import SimpleCSSM
from bench_scan_modes import benchmark_fn


# ---------------------------------------------------------------------------
# Style (verbatim from scripts/plot_main_3x2.py:24-77, plus extensions)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from _plot_style import (apply_style, COLORS,                # noqa: E402
                          MODEL_NAMES as CSSM_TYPE_TO_NAME)

# Channel-sweep palette for Panel C (light → dark).
# C=256 is anchored to Panel B's Mamba-sCSSM blue (#42A5F5) so the same
# data point looks the same colour across both panels.
CHANNEL_SHADES = {
    64:   '#BBDEFB',  # Material Blue 100
    128:  '#64B5F6',  # Blue 300
    256:  '#42A5F5',  # Blue 400  ← matches Panel B Mamba-sCSSM
    512:  '#1976D2',  # Blue 700
    1024: '#0D47A1',  # Blue 900
}

# ---------------------------------------------------------------------------
# Panel specs
# ---------------------------------------------------------------------------

PANEL_A = [
    # Family-grouped, dark → light within each family (matches the ordering
    # convention shared by `_plot_style.FAMILIES`).
    'no_gate',       # Spectral SSM: sCSSM (light blue)
    's4nd_full',     # Conv SSM: S4ND (light green) — moved here from Sequence SSM
    'convs5',        # Conv SSM: ConvS5 (deep green)
    'spatial_attn',  # Attention: Transformer (deep red)
    'mamba2_seq',    # Sequence SSM: Mamba2 (darkest purple)
    'gdn_seq',       # Sequence SSM: GDN (mid purple)
]
PANEL_B = ['no_gate', 'gated', 'gdn']  # ungated, Mamba-style gated, DGM
PANEL_C_FAMILY = 'gated'
PANEL_C_DIMS = [64, 128, 256, 512, 1024]

# Panel T (theory validation) — three CSSM variants timed against video
# length T at fixed (H, C, K). Demonstrates the sequential / parallel /
# spectral scaling regimes empirically.
PANEL_THEORY = ['no_gate_realspace_par', 'no_gate_realspace_seq', 'no_gate']
# Push T high enough for the sequential / parallel / spectral wall-clock
# gaps to actually become visible. At H=32 C=256 the per-step spatial conv
# dominates and the scan-structure differences wash out; at H=16 C=256 the
# per-step work is 4× cheaper, exposing the scan asymptote. T=4096 makes
# the slope-1 sequential reference unambiguous; only feasible on a high-
# memory GPU (B200-class) where the parallel scan's bwd state stays in HBM.
THEORY_TS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
THEORY_H = 16        # fixed spatial side
THEORY_C = 256       # fixed channels (matches Panel B default → exact overlap)
THEORY_K = 11        # fixed kernel size (matches Panel B default → exact overlap)

THEORY_COLORS = {
    # Two non-FFT variants in greys (Spatial = no-FFT parallel = worst,
    # Sequential = K×K real-space sequential = middle); Spectral CSSM is
    # the only saturated colour — the FFT winner.
    'no_gate_realspace_par': '#000000',          # black — real-space conv + parallel scan (no FFT, worst)
    'no_gate_realspace_seq': '#909090',          # mid grey — real-space K×K conv + sequential scan
    'no_gate':               COLORS['sCSSM'],    # canonical Spectral CSSM blue
}
THEORY_NAMES = {
    'no_gate_realspace_par': 'Spatial CSSM',
    'no_gate_realspace_seq': 'Sequential CSSM',
    'no_gate':               'sCSSM',
}


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cssm_type, embed_dim, kernel_size, depth, seq_len=1):
    """Build a SimpleCSSM with minimal stem and one mixer block.

    All per-family extras (state_dim, num_heads, etc.) come from SimpleCSSM
    defaults — we only override the structural knobs. `seq_len` defaults to
    1 (single-image bench); the theory panel sweeps it.
    """
    return SimpleCSSM(
        num_classes=1,
        embed_dim=embed_dim,
        depth=depth,
        cssm_type=cssm_type,
        kernel_size=kernel_size,
        seq_len=seq_len,
        frame_readout='last',
        pos_embed='none',
        stem_mode='pathtracker',
        stem_layers=0,
        norm_type='layer',
    )


# ---------------------------------------------------------------------------
# Per-cell measurement
# ---------------------------------------------------------------------------

def _is_oom(err: BaseException) -> bool:
    """Detect OOM — direct or surfaced via XLA autotuning failures."""
    msg = str(err).lower()
    return any(tag in msg for tag in (
        'out of memory', 'resource_exhausted', 'oom', 'memoryerror',
        'autotuning failed', 'no valid config',
    ))


def run_one_cell(cssm_type, H, embed_dim, kernel_size, depth,
                 batch, warmup, repeats, dtype, T=1, retries=1):
    """Measure fwd and fwd+bwd timings for a single (cssm_type, H, T, C) cell.

    Each retry independently calls benchmark_fn (warmup + repeats); we report
    min/median/max across the per-retry medians for ribbon error bars.
    """
    label = f'[{cssm_type:21s} H={H:3d} T={T:3d} C={embed_dim:3d}]'
    name = (CSSM_TYPE_TO_NAME.get(cssm_type)
            or THEORY_NAMES.get(cssm_type) or cssm_type)
    out = {
        'cssm_type': cssm_type,
        'name': name,
        'H': H, 'N': H * H, 'T_video': T, 'embed_dim': embed_dim,
        'depth': depth, 'kernel_size': kernel_size,
        'batch': batch, 'fwd_ms': float('nan'),
        'fwd_bwd_ms': float('nan'),
        'fwd_ms_min': float('nan'), 'fwd_ms_max': float('nan'),
        'fwd_bwd_ms_min': float('nan'), 'fwd_bwd_ms_max': float('nan'),
        'params': float('nan'),
        'status': 'ok',
    }

    try:
        model = build_model(cssm_type, embed_dim, kernel_size, depth, seq_len=T)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((batch, H, H, 3), dtype=dtype)

        params = model.init(rng, x, training=False)
        n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
        out['params'] = float(n_params)

        @jax.jit
        def fwd(p, x):
            return model.apply(p, x, training=False)

        @jax.jit
        def fwd_bwd(p, x):
            def loss_fn(p_):
                y = model.apply(p_, x, training=False)
                return jnp.mean(jnp.square(y))
            return jax.grad(loss_fn)(p)

        # `retries` independent calls to benchmark_fn — each repeats the
        # full warmup→time loop so jitter from JIT cache state, GPU clock
        # transitions and tile-path crossings shows up as ribbon spread.
        fwd_meds, bwd_meds = [], []
        for _ in range(max(1, retries)):
            t_fwd = benchmark_fn(lambda: fwd(params, x),
                                 warmup=warmup, repeats=repeats)
            t_fwd_bwd = benchmark_fn(lambda: fwd_bwd(params, x),
                                     warmup=warmup, repeats=repeats)
            fwd_meds.append(t_fwd['median_ms'])
            bwd_meds.append(t_fwd_bwd['median_ms'])

        out['fwd_ms'] = float(np.median(fwd_meds))
        out['fwd_bwd_ms'] = float(np.median(bwd_meds))
        out['fwd_ms_min'] = float(np.min(fwd_meds))
        out['fwd_ms_max'] = float(np.max(fwd_meds))
        out['fwd_bwd_ms_min'] = float(np.min(bwd_meds))
        out['fwd_bwd_ms_max'] = float(np.max(bwd_meds))

        del params, fwd, fwd_bwd, model, x

    except (RuntimeError, MemoryError, Exception) as e:  # noqa: BLE001
        if _is_oom(e):
            out['status'] = 'oom'
            print(f'{label}  OOM — skipped', flush=True)
        else:
            out['status'] = 'error'
            print(f'{label}  ERROR: {type(e).__name__}: '
                  f'{str(e)[:200]}', flush=True)
            traceback.print_exc(limit=2)
    finally:
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:  # noqa: BLE001
            pass

    if out['status'] == 'ok':
        print(f'{label}  fwd={out["fwd_ms"]:7.2f}ms  '
              f'fwd+bwd={out["fwd_bwd_ms"]:7.2f}ms  '
              f'params={int(out["params"]):>9d}', flush=True)
    else:
        sys.stdout.flush()

    return out


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def panel_specs(panels, hs):
    """Yield (panel_id, cssm_type, embed_dim, H, T) tuples.

    Panels A/B/T all sweep T_video at fixed H=THEORY_H, embed_dim=THEORY_C
    (kernel_size = K=11 globally). Panel C sweeps H (per `hs`) at T=1 and is
    rendered as the appendix width-scaling figure.
    """
    if 'A' in panels:
        for cssm_type in PANEL_A:
            for T_v in THEORY_TS:
                yield 'A', cssm_type, THEORY_C, THEORY_H, T_v
    if 'B' in panels:
        for cssm_type in PANEL_B:
            for T_v in THEORY_TS:
                yield 'B', cssm_type, THEORY_C, THEORY_H, T_v
    if 'C' in panels:
        for ed in PANEL_C_DIMS:
            for H in hs:
                yield 'C', PANEL_C_FAMILY, ed, H, 1
    if 'T' in panels:
        for cssm_type in PANEL_THEORY:
            for T_v in THEORY_TS:
                yield 'T', cssm_type, THEORY_C, THEORY_H, T_v


def run_sweep(args):
    rows = []
    hs = [int(h) for h in args.hs.split(',')]
    dtype = jnp.float32 if args.dtype == 'fp32' else jnp.bfloat16
    # Cache key includes kernel_size: the theory panel uses K=5 while
    # Panels A/B/C use the CLI default (K=11). Without K in the key, a
    # K=5 theory cell will silently reuse a K=11 row from a prior run.
    seen = set()  # (cssm_type, embed_dim, H, T, K)
    cells = list(panel_specs(args.panels, hs))

    # Optional: load prior CSV and skip cells with status='ok' to enable resume.
    prior_ok = {}
    if args.resume and os.path.exists(args.csv):
        try:
            for r in read_csv(args.csv):
                if r.get('status') == 'ok':
                    key = (r['cssm_type'], r['embed_dim'], r['H'],
                           r.get('T_video', 1), r.get('kernel_size', 0))
                    prior_ok[key] = r
            print(f'Resume: {len(prior_ok)} prior ok cells loaded from {args.csv}',
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f'Resume: failed to load {args.csv}: {e}', flush=True)

    print(f'GPU devices: {jax.devices()}')
    print(f'{len(cells)} (panel × cssm_type × H × T) cells')

    for panel_id, cssm_type, embed_dim, H, T_v in cells:
        kernel_size = THEORY_K if panel_id == 'T' else args.kernel_size
        key = (cssm_type, embed_dim, H, T_v, kernel_size)
        if key in seen:
            prior = next(r for r in rows
                         if (r['cssm_type'], r['embed_dim'], r['H'],
                             r.get('T_video', 1),
                             r.get('kernel_size', 0)) == key)
            row = dict(prior)
            row['panel'] = panel_id
            rows.append(row)
            continue
        seen.add(key)
        if key in prior_ok:
            row = dict(prior_ok[key])
            row['panel'] = panel_id
            rows.append(row)
            print(f'[{cssm_type:21s} H={H:3d} T={T_v:3d} C={embed_dim:3d} '
                  f'K={kernel_size:2d}]  cached', flush=True)
            continue
        r = run_one_cell(
            cssm_type=cssm_type, H=H, embed_dim=embed_dim,
            kernel_size=kernel_size, depth=args.depth,
            batch=args.batch, warmup=args.warmup,
            repeats=args.repeats, dtype=dtype, T=T_v,
            retries=args.retries,
        )
        r['panel'] = panel_id
        rows.append(r)

    return rows


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

CSV_COLS = ['run_id', 'panel', 'cssm_type', 'name', 'H', 'N', 'T_video',
            'embed_dim', 'depth', 'kernel_size', 'batch',
            'fwd_ms', 'fwd_bwd_ms',
            'fwd_ms_min', 'fwd_ms_max',
            'fwd_bwd_ms_min', 'fwd_bwd_ms_max',
            'params', 'status']


def write_csv(rows, path, append=False):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    mode = 'a' if append and os.path.exists(path) else 'w'
    with open(path, mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if mode == 'w':
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in CSV_COLS})
    verb = 'Appended' if mode == 'a' else 'Wrote'
    print(f'{verb} {len(rows)} rows to {path}')


def read_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ('H', 'N', 'embed_dim', 'depth', 'kernel_size', 'batch'):
                if r.get(k):
                    r[k] = int(r[k])
            for k in ('fwd_ms', 'fwd_bwd_ms', 'params',
                      'fwd_ms_min', 'fwd_ms_max',
                      'fwd_bwd_ms_min', 'fwd_bwd_ms_max'):
                if r.get(k) in ('', None):
                    r[k] = float('nan')
                else:
                    r[k] = float(r[k])
            # run_id and T_video are optional — older CSVs lack them.
            if r.get('run_id') in ('', None):
                r['run_id'] = 1
            else:
                try:
                    r['run_id'] = int(r['run_id'])
                except ValueError:
                    r['run_id'] = 1
            if r.get('T_video') in ('', None):
                r['T_video'] = 1
            else:
                try:
                    r['T_video'] = int(r['T_video'])
                except ValueError:
                    r['T_video'] = 1
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _aggregate(by_key):
    """Given dict[N] -> list of (med, lo, hi) triples or scalar fwd_bwd_ms,
    return sorted points (N, median, ribbon_lo, ribbon_hi).

    Triples carry per-cell ribbon info (min/max across retries). Bare scalars
    fall back to the legacy "ribbon = min/max across rows" behavior.
    """
    pts = []
    for N in sorted(by_key):
        meds, los, his = [], [], []
        for v in by_key[N]:
            if isinstance(v, tuple):
                m, lo, hi = v
            else:
                m = lo = hi = v
            if any(np.isnan(z) or np.isinf(z) for z in (m, lo, hi)):
                continue
            meds.append(m); los.append(lo); his.append(hi)
        if not meds:
            continue
        pts.append((N, float(np.median(meds)),
                    float(np.min(los)), float(np.max(his))))
    return pts


def _row_triple(r, key='fwd_bwd_ms'):
    """Extract (median, lo, hi) for a row. Falls back to (med, med, med) when
    per-cell min/max columns are missing or NaN (legacy single-retry CSV)."""
    med = r.get(key, float('nan'))
    lo = r.get(f'{key}_min', float('nan'))
    hi = r.get(f'{key}_max', float('nan'))
    if isinstance(lo, float) and (np.isnan(lo) or np.isinf(lo)):
        lo = med
    if isinstance(hi, float) and (np.isnan(hi) or np.isinf(hi)):
        hi = med
    return (med, lo, hi)


def _oom_xs_for(rows, panel_id, cssm_type, x_key):
    """Return sorted list of x-values (e.g. T_video) where this cell OOMed."""
    out = set()
    for r in rows:
        if r.get('panel') != panel_id:
            continue
        if r.get('cssm_type') != cssm_type:
            continue
        if r.get('status') == 'oom':
            out.add(r[x_key])
    return sorted(out)


def _series_for_panel_a(rows):
    """Baselines (taxonomy) panel — x = T_video at fixed H=THEORY_H, K=11."""
    by_type = {}
    for r in rows:
        if r.get('panel') != 'A':
            continue
        # H filter relaxed: panels A/B may use baselines measured at a
        # different H than the current --theory_h (which scopes panel T).
        # Take whatever H the panel data was measured at.
        ct = r['cssm_type']
        by_type.setdefault(ct, {}).setdefault(
            r['T_video'], []).append(_row_triple(r, 'fwd_bwd_ms'))
    series = []
    for cssm_type in PANEL_A:
        if cssm_type not in by_type:
            continue
        name = CSSM_TYPE_TO_NAME[cssm_type]
        pts = _aggregate(by_type[cssm_type])
        oom_xs = _oom_xs_for(rows, 'A', cssm_type, 'T_video')
        if pts or oom_xs:
            series.append((name, COLORS[name], pts, oom_xs))
    return series


def _series_for_panel_b(rows):
    """sCSSMs panel — x = T_video at fixed H=THEORY_H, K=11."""
    by_type = {}
    for r in rows:
        if r.get('panel') != 'B':
            continue
        # H filter relaxed: panels A/B may use baselines measured at a
        # different H than the current --theory_h (which scopes panel T).
        # Take whatever H the panel data was measured at.
        ct = r['cssm_type']
        by_type.setdefault(ct, {}).setdefault(
            r['T_video'], []).append(_row_triple(r, 'fwd_bwd_ms'))
    series = []
    for cssm_type in PANEL_B:
        if cssm_type not in by_type:
            continue
        name = CSSM_TYPE_TO_NAME[cssm_type]
        pts = _aggregate(by_type[cssm_type])
        oom_xs = _oom_xs_for(rows, 'B', cssm_type, 'T_video')
        if pts or oom_xs:
            series.append((name, COLORS[name], pts, oom_xs))
    return series


def _series_for_panel_c(rows):
    """Width-scaling appendix panel — x = N at fixed T=1."""
    by_dim = {}
    for r in rows:
        if r.get('panel') != 'C':
            continue
        if r.get('T_video', 1) != 1:
            continue
        ed = r['embed_dim']
        by_dim.setdefault(ed, {}).setdefault(r['N'], []).append(_row_triple(r, 'fwd_bwd_ms'))
    series = []
    for ed in PANEL_C_DIMS:
        if ed not in by_dim:
            continue
        pts = _aggregate(by_dim[ed])
        oom_xs = []  # Not yet wired up for panel C; OOMs there manifest as
                     # truncated curves rather than markers.
        if pts:
            label = f'Mamba-sCSSM, C={ed}'
            series.append((label, CHANNEL_SHADES[ed], pts, oom_xs))
    return series


def _series_for_theory(rows):
    """Theory panel — x-axis is T_video at fixed (H, C, K)."""
    by_type = {}
    for r in rows:
        if r.get('panel') != 'T':
            continue
        ct = r['cssm_type']
        by_type.setdefault(ct, {}).setdefault(
            r['T_video'], []).append(_row_triple(r, 'fwd_bwd_ms'))
    series = []
    for cssm_type in PANEL_THEORY:
        if cssm_type not in by_type:
            continue
        name = THEORY_NAMES[cssm_type]
        color = THEORY_COLORS[cssm_type]
        pts = _aggregate(by_type[cssm_type])
        oom_xs = _oom_xs_for(rows, 'T', cssm_type, 'T_video')
        if pts or oom_xs:
            series.append((name, color, pts, oom_xs))
    return series


def _add_slope_guides(ax, xs, var_name='N'):
    """Reference lines for O(log x) and O(x) — bracket the data range.

    `xs` is the array of x-axis values; for the H-sweep panels these are
    `N = H*W`, for the theory panel they are video lengths T. `var_name` is
    the symbol to render in the slope-label LaTeX ('N' or 'T').
    """
    xs = np.array(sorted(set(xs)), dtype=float)
    if len(xs) < 2:
        return
    y_lo, y_hi = ax.get_ylim()
    if y_lo <= 0:
        y_lo = 1e-2
    x0 = xs[0]
    # Anchor low so guides sit at the bottom of the visible range.
    anchor = y_lo * 1.3
    # log_curve and lin both pass through (x0, anchor).
    log_curve = anchor * (np.log2(np.maximum(xs, x0)) / np.log2(max(x0, 2.0)))
    log_curve = np.where(log_curve <= 0, anchor, log_curve)
    lin = anchor * (xs / x0)

    ax.plot(xs, log_curve, color='#888', linestyle=':', linewidth=1.4,
            alpha=0.55, zorder=1, label='_nolegend_', clip_on=True)
    ax.plot(xs, lin, color='#888', linestyle='--', linewidth=1.4,
            alpha=0.55, zorder=1, label='_nolegend_', clip_on=True)

    ax.set_ylim(y_lo, y_hi)

    def _label(curve, txt):
        in_range = curve < y_hi
        if not in_range.any():
            return
        i = int(np.where(in_range)[0][-1])
        ax.text(xs[i] * 0.92, curve[i] * 1.15, txt,
                fontsize=40, color='#555', alpha=0.85, ha='right',
                va='bottom', zorder=2)

    _label(log_curve, rf'$\mathcal{{O}}(\log {var_name})$')
    _label(lin, rf'$\mathcal{{O}}({var_name})$')


def _draw_panel(ax, series, title, hs, show_ylabel, legend_ncol=1,
                attn_crossover_N=None, show_attn_labels=True,
                show_symbol_legend=True, ylim=None, xlim=None,
                x_label='Spatial tokens (N = H*W)', x_values=None,
                slope_var='N'):
    """Render one log-log panel.

    `x_values` is the array of x-axis values used for slope-guide construction.
    Defaults to `[h*h for h in hs]` for the H-sweep panels (N = H*W spatial
    tokens); the theory panel passes the video-length sweep directly.
    """
    import seaborn as sns
    has_data = False
    # Sort series slowest → fastest by average fwd+bwd runtime so the legend
    # reads top-to-bottom in the same visual order the curves stack on the
    # right edge of the panel. Series with no data sink to the bottom.
    def _avg_runtime(entry):
        pts = entry[2]
        if not pts:
            return -1.0  # below everything → ends up at bottom
        return float(np.mean([p[1] for p in pts]))
    series = sorted(series, key=_avg_runtime, reverse=True)

    # One OOM marker per series, placed at the last valid (x, y) on the
    # curve — i.e. the rightmost step before it ran out of memory.
    oom_records = []  # list of (color, x_last, y_last)
    for entry in series:
        if len(entry) == 4:
            label, color, pts, oom_xs = entry
        else:
            label, color, pts = entry
            oom_xs = []
        if pts:
            has_data = True
            xs = [p[0] for p in pts]
            meds = [p[1] for p in pts]
            los = [p[2] for p in pts]
            his = [p[3] for p in pts]
            if any(hi > lo for lo, hi in zip(los, his)):
                ax.fill_between(xs, los, his, color=color, alpha=0.18,
                                edgecolor='none', zorder=4)
            ax.plot(xs, meds, marker='o', linestyle='-', linewidth=6.5,
                    markersize=18, color=color, label=label, alpha=0.95,
                    markeredgecolor='black', markeredgewidth=0.8, zorder=5,
                    solid_capstyle='round', solid_joinstyle='round')
            # Mark the last good point if any later T_video OOMed for this
            # series (the curve "dies" at that step).
            if oom_xs and pts:
                first_oom = min(oom_xs)
                # Largest valid x strictly less than the first OOM x.
                pre_oom = [(x, y) for (x, y, _, _) in pts if x < first_oom]
                if pre_oom:
                    pre_oom.sort()
                    x_last, y_last = pre_oom[-1]
                    oom_records.append((color, x_last, y_last))
    ax.set_xscale('log')
    ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(x_label, fontsize=40)
    if show_ylabel:
        ax.set_ylabel('Step time (ms, fwd + bwd)', fontsize=40)
    ax.set_title(title, fontsize=46, fontweight='bold', pad=24)
    ax.tick_params(axis='both', which='major', labelsize=36)
    ax.grid(which='both', linestyle=':', linewidth=0.5,
            color='#cccccc', alpha=0.4)
    if has_data:
        # Slope guides need final y-range, so add them BEFORE legend (which
        # uses autoscale_view).
        if x_values is None:
            x_values = [h * h for h in hs]
        _add_slope_guides(ax, x_values, var_name=slope_var)
        if attn_crossover_N is not None:
            _add_attn_crossover(ax, attn_crossover_N,
                                show_attn_labels=show_attn_labels,
                                show_symbol_legend=show_symbol_legend)
        ax.legend(loc='upper left', frameon=True, fancybox=False,
                  borderpad=0.5, handletextpad=0.6, labelspacing=0.35,
                  ncol=legend_ncol, columnspacing=0.8, fontsize=42)
    # Draw the OOM end-of-line marker (one per series): a large open circle
    # with × inside, placed at the last successful (x, y) on the curve so it
    # caps the line at the step before OOM.
    if oom_records:
        y_lo_now, y_hi_now = ax.get_ylim()
        for color, x_last, y_last in oom_records:
            ax.scatter([x_last], [y_last], s=1500, marker='o',
                       facecolors='white', edgecolors=color,
                       linewidths=5.5, zorder=12, clip_on=False)
            ax.scatter([x_last], [y_last], s=600, marker='x',
                       color=color, linewidths=6.5, zorder=13,
                       clip_on=False)
        ax.set_ylim(y_lo_now, y_hi_now)
    sns.despine(ax=ax)


def _add_attn_crossover(ax, N_thresh, show_attn_labels=True,
                        show_symbol_legend=True):
    """Shade the linear/quadratic regimes at N = C.

    Args:
        N_thresh: x-axis position of the threshold (typically C).
        show_attn_labels: include "attn ∝ N·C²" / "attn ∝ N²·C" boxes.
        show_symbol_legend: include the small "N = H·W tokens, C = 256 ch." note.
    """
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    # Subtle background tints — left = channel-mix-dominated (light grey),
    # right = token-mix-dominated (white, no fill).
    ax.axvspan(x_lo, N_thresh, color='#E8E8E8', alpha=0.7, zorder=0)
    # Right band stays white (no axvspan — keep default white background).
    # Threshold line.
    ax.axvline(N_thresh, color='#888', linestyle='--', linewidth=0.8,
               alpha=0.7, zorder=1)

    if show_attn_labels:
        y_label = 10 ** (np.log10(y_lo) + 0.78 * (np.log10(y_hi) - np.log10(y_lo)))
        x_left = 10 ** (np.log10(x_lo) + 0.40 * (np.log10(N_thresh) - np.log10(x_lo)))
        x_right = 10 ** (np.log10(N_thresh) + 0.30 * (np.log10(x_hi) - np.log10(N_thresh)))
        ax.text(x_left, y_label, r'attn $\propto N\!\cdot\!C^{2}$',
                fontsize=28, color='#424242', alpha=0.95,
                ha='center', va='center', zorder=4,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                          edgecolor='#616161', linewidth=0.4, alpha=0.9))
        ax.text(x_right, y_label, r'attn $\propto N^{2}\!\cdot\!C$',
                fontsize=28, color='#212121', alpha=0.95,
                ha='center', va='center', zorder=4,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                          edgecolor='#212121', linewidth=0.4, alpha=0.9))

    # N = C threshold marker near the top.
    y_thresh_label = 10 ** (np.log10(y_lo) + 0.92 * (np.log10(y_hi) - np.log10(y_lo)))
    ax.text(N_thresh, y_thresh_label, r'$N=C$',
            fontsize=28, color='#444', alpha=0.9,
            ha='center', va='center', zorder=4,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#888', linewidth=0.4, alpha=0.85))

    if show_symbol_legend:
        y_sym = 10 ** (np.log10(y_lo) + 0.05 * (np.log10(y_hi) - np.log10(y_lo)))
        x_sym = 10 ** (np.log10(x_lo) + 0.97 * (np.log10(x_hi) - np.log10(x_lo)))
        ax.text(x_sym, y_sym,
                r'$N{=}H\!\cdot\!W$ tokens, $C{=}256$ ch.',
                fontsize=24, color='#444', alpha=0.85,
                ha='right', va='bottom', zorder=4,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#bbb', linewidth=0.4, alpha=0.85))
    # Restore limits if axvspan/text nudged them.
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)


def _global_y_range(rows):
    """Return (ymin, ymax) across all valid fwd_bwd_ms with log-friendly padding."""
    vals = [r['fwd_bwd_ms'] for r in rows
            if isinstance(r.get('fwd_bwd_ms'), float)
            and not (np.isnan(r['fwd_bwd_ms']) or np.isinf(r['fwd_bwd_ms']))]
    if not vals:
        return 0.1, 1000.0
    lo = min(vals) / 2.0
    hi = max(vals) * 2.0
    return lo, hi


def render_figure(rows, fig_path, hs, layout='main'):
    """Render the bench figure.

    `layout='main'` (default) draws the 3-panel main-text figure:
        Theory validation | sCSSMs | Baselines
    `layout='width'` draws the appendix width-scaling panel only.

    T-sweep panels filter to T_video values present in `THEORY_TS` so that
    points dropped from that grid (e.g. T=4 below the JIT-overhead floor) are
    excluded cleanly rather than left dangling at the edge of the x-range.
    """
    apply_style()
    import matplotlib.pyplot as plt
    if layout != 'width':
        rows = [r for r in rows
                if r.get('panel') == 'C' or r.get('T_video') in THEORY_TS]

    y_lo, y_hi = _global_y_range(rows)
    Ns = [h * h for h in hs]
    x_lo_N, x_hi_N = min(Ns) / 1.5, max(Ns) * 1.5
    x_lo_T, x_hi_T = min(THEORY_TS) / 1.5, max(THEORY_TS) * 1.5

    if layout == 'width':
        # Appendix: width-scaling panel only — x = N (spatial tokens).
        fig, ax = plt.subplots(1, 1, figsize=(13.5, 13))
        _draw_panel(ax, _series_for_panel_c(rows),
                    'Width scaling', hs,
                    show_ylabel=True, legend_ncol=1,
                    ylim=(y_lo, y_hi), xlim=(x_lo_N, x_hi_N))
    else:
        # Main figure — all three panels share x = video length T at fixed
        # H=THEORY_H, C=THEORY_C, K=11.
        fig, (axTheory, axB, axA) = plt.subplots(1, 3, figsize=(40, 13))
        common_kw = dict(
            hs=THEORY_TS, x_label='Video length (T frames)',
            x_values=THEORY_TS, slope_var='T',
            ylim=(y_lo, y_hi), xlim=(x_lo_T, x_hi_T),
        )
        # (a) Mechanism Validation — algorithmic implementations of the same SSM.
        _draw_panel(axTheory, _series_for_theory(rows),
                    'Mechanism Validation',
                    show_ylabel=True, legend_ncol=1, **common_kw)
        # (b) Modularity — three sCSSM gating circuits, same scaling backbone.
        _draw_panel(axB, _series_for_panel_b(rows),
                    'Modularity',
                    show_ylabel=False, legend_ncol=1, **common_kw)
        # (c) Scalability Landscape — sCSSM family vs. baseline mixers.
        _draw_panel(axA, _series_for_panel_a(rows),
                    'Scalability Landscape',
                    show_ylabel=False, legend_ncol=2, **common_kw)

    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.90,
                        wspace=0.22)

    fig_path = os.path.abspath(fig_path)
    os.makedirs(os.path.dirname(fig_path) or '.', exist_ok=True)
    base, _ = os.path.splitext(fig_path)
    plt.savefig(base + '.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(base + '.png', dpi=200, bbox_inches='tight')
    print(f'Saved figure to {base}.pdf and {base}.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hs', default='8,16,32,64,128,256',
                   help='Comma-separated spatial side lengths (square H=W).')
    p.add_argument('--embed_dim', type=int, default=256,
                   help='Channel dim for panels A and B.')
    p.add_argument('--depth', type=int, default=1)
    p.add_argument('--kernel_size', type=int, default=11)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--retries', type=int, default=5,
                   help='Independent benchmark_fn calls per cell — '
                        'min/median/max across retries become the ribbon.')
    p.add_argument('--panels', default='A,B,T',
                   help='Comma-separated panel ids to run (subset of A,B,C,T).'
                        ' A=baselines, B=fcssms, C=width scaling (appendix),'
                        ' T=theory validation (video-length sweep).')
    p.add_argument('--layout', choices=('main', 'width'), default='main',
                   help="'main' (default) renders Theory|sCSSMs|Baselines."
                        " 'width' renders only the appendix width-scaling panel.")
    p.add_argument('--theory_h', type=int, default=THEORY_H,
                   help='Spatial side for the T-sweep panels (overrides THEORY_H).')
    p.add_argument('--theory_c', type=int, default=THEORY_C,
                   help='Channel dim for the T-sweep panels (overrides THEORY_C).')
    p.add_argument('--theory_ts', default=','.join(str(t) for t in THEORY_TS),
                   help='Comma-separated T_video values for the T-sweep panels.')
    p.add_argument('--csv', default='benchmarks/results/bench_image_timing.csv')
    p.add_argument('--fig', default='benchmarks/results/bench_image_timing.pdf')
    p.add_argument('--dtype', choices=('fp32', 'bf16'), default='fp32')
    p.add_argument('--from-csv', default=None,
                   help='Skip the sweep and re-render from this CSV.')
    p.add_argument('--resume', action='store_true',
                   help='Skip cells already present with status="ok" in --csv.')
    p.add_argument('--run-id', type=int, default=1,
                   help='Tag every emitted row with this integer run id.')
    p.add_argument('--append', action='store_true',
                   help='Append to --csv instead of overwriting.')
    return p.parse_args()


def main():
    args = parse_args()
    hs = [int(h) for h in args.hs.split(',')]
    # Apply variant overrides — these are referenced as module globals
    # by panel_specs, render_figure, and the panel-series helpers.
    global THEORY_H, THEORY_C, THEORY_TS
    THEORY_H = args.theory_h
    THEORY_C = args.theory_c
    THEORY_TS = [int(t) for t in args.theory_ts.split(',')]

    if args.from_csv:
        rows = read_csv(args.from_csv)
        print(f'Loaded {len(rows)} rows from {args.from_csv}')
    else:
        t0 = time.time()
        rows = run_sweep(args)
        for r in rows:
            r['run_id'] = args.run_id
        print(f'Sweep completed in {time.time() - t0:.1f}s')
        write_csv(rows, args.csv, append=args.append)
        # When appending, reload everything for plotting.
        if args.append and os.path.exists(args.csv):
            rows = read_csv(args.csv)

    render_figure(rows, args.fig, hs, layout=args.layout)


if __name__ == '__main__':
    main()
