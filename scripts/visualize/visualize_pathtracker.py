"""Visualize PathTracker samples — raw videos and model-view (subsampled + normalized)."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def load_raw(path):
    """Load a single .npy video: (64, 32, 32, 3) uint8."""
    return np.load(path)


def get_samples(root, n=4, seed=42):
    """Get n positive and n negative sample paths."""
    root = Path(root)
    rng = np.random.RandomState(seed)

    out = []
    for label in [1, 0]:
        label_dir = root / str(label)
        paths = sorted(label_dir.glob('*.npy'))
        chosen = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
        for i in chosen:
            out.append((paths[i], label))
    return out


def plot_filmstrip(video, label, num_frames=8, ax_row=None, title=None):
    """Plot evenly-spaced frames from a video as a filmstrip."""
    indices = np.linspace(0, video.shape[0] - 1, num_frames, dtype=int)
    frames = video[indices]

    if ax_row is None:
        _, ax_row = plt.subplots(1, num_frames, figsize=(2 * num_frames, 2))

    for i, (ax, t) in enumerate(zip(ax_row, indices)):
        ax.imshow(frames[i])
        ax.set_title(f't={t}', fontsize=8)
        ax.axis('off')

    if title:
        ax_row[0].set_ylabel(title, fontsize=9, rotation=0, labelpad=50, va='center')


def cmd_filmstrip(args):
    """Show filmstrips of positive and negative samples."""
    samples = get_samples(args.root, n=args.n, seed=args.seed)
    num_frames = args.num_frames
    n_rows = len(samples)

    fig, axes = plt.subplots(n_rows, num_frames, figsize=(2 * num_frames, 1.8 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, (path, label) in enumerate(samples):
        video = load_raw(path)
        tag = 'pos' if label == 1 else 'neg'
        plot_filmstrip(video, label, num_frames, axes[row],
                       title=f'{tag}\n{path.name}')

    fig.suptitle(f'PathTracker — {num_frames} evenly-spaced frames from 64', fontsize=12)
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()


def cmd_animate(args):
    """Animate a single sample as a video."""
    if args.file:
        path = Path(args.file)
        label = int(path.parent.name) if path.parent.name in ('0', '1') else -1
    else:
        samples = get_samples(args.root, n=1, seed=args.seed)
        path, label = samples[0]

    video = load_raw(path)
    tag = {1: 'positive', 0: 'negative'}.get(label, 'unknown')

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(video[0])
    ax.axis('off')
    title = ax.set_title(f'{tag} — frame 0/63\n{path.name}')

    def update(t):
        im.set_data(video[t])
        title.set_text(f'{tag} — frame {t}/63\n{path.name}')
        return [im, title]

    anim = FuncAnimation(fig, update, frames=video.shape[0], interval=80, blit=True)

    if args.save:
        anim.save(args.save, writer='pillow', fps=12)
        print(f"Saved to {args.save}")
    else:
        plt.show()


def cmd_stats(args):
    """Print dataset statistics."""
    root = Path(args.root)
    for label in [0, 1]:
        label_dir = root / str(label)
        paths = list(label_dir.glob('*.npy'))
        tag = 'positive' if label == 1 else 'negative'
        print(f"{tag:>8}: {len(paths):,} samples in {label_dir}")

        if paths:
            sample = np.load(paths[0])
            print(f"         shape={sample.shape}, dtype={sample.dtype}, "
                  f"range=[{sample.min()}, {sample.max()}]")

    total_pos = len(list((root / '1').glob('*.npy')))
    total_neg = len(list((root / '0').glob('*.npy')))
    print(f"\n   total: {total_pos + total_neg:,} ({total_pos:,} pos + {total_neg:,} neg)")


def main():
    parser = argparse.ArgumentParser(description='Visualize PathTracker dataset')
    parser.add_argument('--root', type=str,
                        default='/media/data_cifs/projects/prj_video_datasets/pathtracker',
                        help='PathTracker dataset root')
    sub = parser.add_subparsers(dest='cmd')

    # filmstrip
    fs = sub.add_parser('filmstrip', help='Show filmstrips of sampled videos')
    fs.add_argument('-n', type=int, default=3, help='Samples per class')
    fs.add_argument('--num_frames', type=int, default=8, help='Frames to show')
    fs.add_argument('--seed', type=int, default=42)
    fs.add_argument('--save', type=str, default=None, help='Save to file instead of showing')

    # animate
    an = sub.add_parser('animate', help='Animate a single sample')
    an.add_argument('--file', type=str, default=None, help='Specific .npy file (else random)')
    an.add_argument('--seed', type=int, default=42)
    an.add_argument('--save', type=str, default=None, help='Save as .gif')

    # stats
    sub.add_parser('stats', help='Print dataset statistics')

    args = parser.parse_args()

    if args.cmd == 'filmstrip':
        cmd_filmstrip(args)
    elif args.cmd == 'animate':
        cmd_animate(args)
    elif args.cmd == 'stats':
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
