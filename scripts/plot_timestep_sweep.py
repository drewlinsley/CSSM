#!/usr/bin/env python3
"""
Plot CSSM 1×1 accuracy as a function of timesteps (T=1..10) on Pathfinder.

Usage:
    python scripts/plot_timestep_sweep.py [--project serrelab/CSSM_pathfinder]
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
except ImportError:
    raise ImportError("pip install wandb")


def fetch_timestep_data(project: str, prefix: str = "pf_cssm_1x1_fac_t"):
    """Fetch best val accuracy and step time for each timestep from W&B."""
    api = wandb.Api()
    runs = api.runs(project)

    results = {}  # T -> {'acc': float, 'step_time': float}
    for r in runs:
        name = r.name
        if not name.startswith(prefix):
            continue
        # Extract T from name like "pf_cssm_1x1_t3_d1_e64"
        try:
            t_str = name[len(prefix):].split("_")[0]
            T = int(t_str)
        except (ValueError, IndexError):
            continue

        acc = r.summary.get("best_val_acc", r.summary.get("val_acc", None))
        step_time = r.summary.get("train/ms_per_step", r.summary.get("ms_per_step",
                    r.summary.get("step_time", None)))
        if acc is not None:
            if T not in results or acc > results[T]['acc']:
                results[T] = {'acc': acc, 'step_time': step_time}

    # Also check for the original T=8 run (pf_cssm_1x1_d1_e64)
    if 8 not in results:
        for r in runs:
            if r.name == "pf_cssm_1x1_d1_e64":
                acc = r.summary.get("best_val_acc", r.summary.get("val_acc", None))
                step_time = r.summary.get("train/ms_per_step", r.summary.get("ms_per_step",
                            r.summary.get("step_time", None)))
                if acc is not None:
                    results[8] = {'acc': acc, 'step_time': step_time}
                break

    return results


def plot_timestep_curve(results: dict, output_path: str = "timestep_sweep.png"):
    """Plot accuracy (left axis) and step time (right axis) vs timesteps."""
    if not results:
        print("No data found!")
        return

    timesteps = sorted(results.keys())
    accs = [results[t]['acc'] for t in timesteps]
    step_times = [results[t]['step_time'] for t in timesteps]
    has_step_times = any(st is not None for st in step_times)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # Left axis: accuracy
    color_acc = '#2196F3'
    ax1.plot(timesteps, accs, 'o-', color=color_acc, linewidth=2, markersize=8,
             label='Val Accuracy')
    ax1.set_xlabel('Timesteps (T)', fontsize=13)
    ax1.set_ylabel('Best Val Accuracy', fontsize=13, color=color_acc)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_xticks(range(1, max(timesteps) + 1))
    ax1.set_ylim(0.45, 1.0)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, label='Chance')
    ax1.grid(True, alpha=0.2)

    # Best accuracy annotation
    best_t = timesteps[np.argmax(accs)]
    best_acc = max(accs)
    ax1.annotate(f'T={best_t}: {best_acc:.1%}',
                 xy=(best_t, best_acc), xytext=(best_t + 0.5, best_acc - 0.03),
                 fontsize=10, ha='left',
                 arrowprops=dict(arrowstyle='->', color=color_acc, lw=1))

    # Right axis: step time
    if has_step_times:
        color_time = '#F44336'
        ax2 = ax1.twinx()
        valid_t = [t for t, st in zip(timesteps, step_times) if st is not None]
        valid_st = [st for st in step_times if st is not None]
        ax2.plot(valid_t, valid_st, 's--', color=color_time, linewidth=1.5, markersize=6,
                 alpha=0.8, label='Step Time')
        ax2.set_ylabel('ms / step', fontsize=13, color=color_time)
        ax2.tick_params(axis='y', labelcolor=color_time)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)
    else:
        ax1.legend(loc='lower right', fontsize=10)

    ax1.set_title('CSSM 1×1: Accuracy & Speed vs Temporal Recurrence Steps\n'
                  '(Static image repeated T times, Pathfinder CL-14, 128px)',
                  fontsize=12)

    # Theory annotation
    ax1.text(0.02, 0.98,
             'T=1: y = C·B·x  (FIR filter)\n'
             'T→∞: y = C·(I−A)⁻¹·B·x  (IIR resonance)',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             fontfamily='monospace', alpha=0.6,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='serrelab/CSSM_pathfinder')
    parser.add_argument('--output', default='timestep_sweep.png')
    args = parser.parse_args()

    print(f"Fetching from {args.project}...")
    results = fetch_timestep_data(args.project)

    if results:
        print(f"Found {len(results)} timesteps: {sorted(results.keys())}")
        for t in sorted(results.keys()):
            acc = results[t]['acc']
            st = results[t]['step_time']
            st_str = f"{st:.0f} ms/step" if st else "n/a"
            print(f"  T={t:2d}: acc={acc:.4f}  {st_str}")
    else:
        print("No matching runs found yet.")
        return

    plot_timestep_curve(results, args.output)


if __name__ == '__main__':
    main()
