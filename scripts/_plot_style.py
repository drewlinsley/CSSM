"""Shared paper-figure plotting style.

Centralises:
  - Roboto font registration (from ~/.local/share/fonts/roboto)
  - matplotlib rcParams (sans-serif Roboto, Arial fallback, seaborn ticks)
  - canonical model names (cssm_type → display name) and colour palette
  - family groupings used for legends and Pareto fronts

Usage:
    from _plot_style import apply_style, COLORS, FAMILIES, MODEL_NAMES
    apply_style()
    ...
"""
import os


# ---------------------------------------------------------------------------
# Font + rcParams
# ---------------------------------------------------------------------------

RALEWAY_DIR = os.path.expanduser('~/.local/share/fonts/raleway')
SANS_FALLBACK = ['Raleway', 'Arial', 'Helvetica', 'DejaVu Sans']


def _register_raleway():
    if not os.path.isdir(RALEWAY_DIR):
        return
    import matplotlib.font_manager as fm
    for f in os.listdir(RALEWAY_DIR):
        if f.lower().endswith('.ttf'):
            try:
                fm.fontManager.addfont(os.path.join(RALEWAY_DIR, f))
            except Exception:
                pass


def apply_style():
    """Register Raleway, set Agg backend, apply seaborn ticks + rcParams."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    _register_raleway()
    plt.style.use('default')
    sns.set_style('ticks')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': SANS_FALLBACK,
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 7.5,
    })


# ---------------------------------------------------------------------------
# Canonical naming + colour palette
# ---------------------------------------------------------------------------

# cssm_type → display name. Keep this in sync with bench_image_timing.py.
MODEL_NAMES = {
    'gated':         'Mamba-sCSSM',
    'gdn':           'GDN-sCSSM',
    'no_gate':       'sCSSM',
    'transformer':   'Transformer-sCSSM',
    # 'no_fft' (Mamba-ConvS5) intentionally omitted — dropped from all figures.
    'conv_ssm':      'CSSM',
    'convs5':        'ConvS5',
    's4nd':          'S4ND-Diag',
    's4nd_full':     'S4ND',
    'spatial_attn':  'Transformer',
    'spatiotemporal_attn': 'Transformer-ST',
    'mamba2_seq':    'Mamba-2',
    'gdn_seq':       'GDN',
}

COLORS = {
    # Spectral SSM family. GDN→deep blue, Mamba→cyan (distinct hue, not just
    # a darker blue, so the two are unambiguously separable), sCSSM→light blue.
    'GDN-sCSSM':         '#1565C0',  # Material Blue 800 — deep blue
    'Mamba-sCSSM':       '#00ACC1',  # Material Cyan 600 — cyan-teal
    'sCSSM':             '#90CAF9',  # Material Blue 200 — light blue
    'Transformer-sCSSM': '#0D47A1',  # darkest blue (rarely shown)
    # Conv SSM family — three greens, ConvS5 → CSSM → S4ND dark→light.
    # S4ND lives here (not with Sequence SSM) because it's a 2D ConvSSM-style
    # model (HiPPO-LegS DPLR per spatial axis), grouped with the convolutional
    # baselines.
    'ConvS5':            '#2E7D32',  # deep green (Material Green 800)
    'CSSM':              '#66BB6A',  # mid green (Material Green 400)
    'S4ND':              '#A5D6A7',  # light green (Material Green 200)
    'Transformer':       '#C62828',  # Attention family — deep red
    'Transformer-ST':    '#EF9A9A',  # light red
    # Sequence SSM family — three purples, Mamba-2 → GDN → S4ND-Diag dark→light.
    'Mamba-2':           '#4A148C',  # darkest purple (Material Purple 900)
    'GDN':               '#7B1FA2',  # mid-dark purple (Material Purple 700)
    'S4ND-Diag':         '#CE93D8',  # light purple (Material Purple 200)
}

# Override per-model text colour where the dot colour is too light to read.
# The Spectral SSM family uses a *darkening* gradient anchored on sCSSM's
# already-overridden colour so the family reads light → dark in the text:
#   sCSSM (#1976D2)  →  Mamba-sCSSM (#1565C0)  →  GDN-sCSSM (#0D47A1)
# Dot colours are unchanged; only the on-figure text labels darken.
TEXT_COLORS = {
    'sCSSM':       '#1976D2',  # Material Blue 700
    'Mamba-sCSSM': '#00838F',  # Material Cyan 800 — darker cyan for legibility
    'GDN-sCSSM':   '#0D47A1',  # Material Blue 900
}

FAMILIES = {
    'Spectral SSM': ['GDN-sCSSM', 'Mamba-sCSSM', 'sCSSM'],
    'Conv SSM':     ['ConvS5', 'CSSM', 'S4ND'],
    'Attention':    ['Transformer', 'Transformer-ST'],
    'Sequence SSM': ['Mamba-2', 'GDN', 'S4ND-Diag'],
}

FAMILY_COLORS = {
    'Spectral SSM': '#1565C0',
    'Conv SSM':     '#2E7D32',
    'Attention':    '#C62828',
    'Sequence SSM': '#6A1B9A',
}


def text_color_for(model_name):
    """Return the legible text colour for a model — darker override if any."""
    return TEXT_COLORS.get(model_name, COLORS.get(model_name, 'black'))


# ---------------------------------------------------------------------------
# Shared run-inclusion rule for paper plots.
# ---------------------------------------------------------------------------

SPECTRAL_CSSM_TYPES = {'gated', 'gdn', 'no_gate', 'transformer'}


def should_skip_run(run, control_min_acc=0.75):
    """Shared rule for excluding runs from paper figures.

    Skips:
      - Crashed runs (W&B state == 'crashed').
      - Spectral-sCSSM 'control' ablations — kernel_size==1 (no spatial
        mixing) or seq_len==1 (no temporal recurrence) — whose best
        accuracy is below `control_min_acc`. These are deliberate
        ablations expected to fail.

    The control filter is gated to the spectral sCSSM family only (cssm
    type in {gated, gdn, no_gate, transformer}). Sequence baselines like
    `gdn_seq` / `mamba2_seq` legitimately use seq_len=1 on single-frame
    tasks; those are not controls and must not be dropped.
    """
    if getattr(run, 'state', None) == 'crashed':
        return True
    cfg = run.config
    cssm = cfg.get('cssm', '')
    if cssm not in SPECTRAL_CSSM_TYPES:
        return False
    ks = cfg.get('kernel_size', None)
    sl = cfg.get('seq_len', None)
    is_control = (ks == 1) or (sl == 1)
    if is_control:
        acc = (run.summary.get('best_val_acc',
                               run.summary.get('val_acc', 0)) or 0)
        if acc < control_min_acc:
            return True
    return False
