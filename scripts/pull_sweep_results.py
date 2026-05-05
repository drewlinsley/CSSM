#!/usr/bin/env python3
"""Pull sweep results from W&B for the /34 transformer+GDN sweep.

Usage:
    python scripts/pull_sweep_results.py [--project CSSM_pathtracker] [--all]

By default only shows runs matching the /34 sweep naming (st_d* and gdn_d*).
Use --all to show all runs in the project.
"""
import argparse
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='serrelab/CSSM_pathtracker')
    parser.add_argument('--all', action='store_true', help='Show all runs, not just sweep runs')
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(args.project)

    # Collect best result per run_name (handles duplicate runs from restarts)
    results = {}
    for r in runs:
        name = r.config.get('run_name', r.name)

        if not args.all:
            if not (name.startswith('st_d') or name.startswith('gdn_d')):
                continue

        best_acc = r.summary.get('best_val_acc', None)
        state = r.state  # 'finished', 'running', 'crashed', etc.
        epoch = r.summary.get('epoch', '?')
        created = r.created_at[:10] if r.created_at else '?'

        # Keep the best accuracy across duplicate runs
        if name not in results or (best_acc is not None and (results[name]['acc'] is None or best_acc > results[name]['acc'])):
            results[name] = {
                'acc': best_acc,
                'state': state,
                'epoch': epoch,
                'created': created,
                'id': r.id,
            }

    if not results:
        print("No matching runs found.")
        return

    # Sort by accuracy (descending), None last
    sorted_results = sorted(results.items(), key=lambda x: (x[1]['acc'] is not None, x[1]['acc'] or 0), reverse=True)

    # Print table
    print(f"{'Run':<40} {'Val Acc':>8} {'Epoch':>6} {'State':>10} {'Date':>12} {'Type':<12}")
    print('-' * 92)
    for name, info in sorted_results:
        typ = 'GDN' if name.startswith('gdn') else 'Transformer' if name.startswith('st_') else '?'
        acc_str = f"{info['acc']:.4f}" if info['acc'] is not None else 'N/A'
        print(f"{name:<40} {acc_str:>8} {str(info['epoch']):>6} {info['state']:>10} {info['created']:>12} {typ:<12}")

    # Summary
    gdn = [v['acc'] for k, v in results.items() if k.startswith('gdn_d') and v['acc'] is not None]
    st = [v['acc'] for k, v in results.items() if k.startswith('st_d') and v['acc'] is not None]
    print()
    if gdn:
        print(f"GDN:         {len(gdn)} finished, best={max(gdn):.4f}, mean={sum(gdn)/len(gdn):.4f}")
    if st:
        print(f"Transformer: {len(st)} finished, best={max(st):.4f}, mean={sum(st)/len(st):.4f}")
    total = len(results)
    finished = sum(1 for v in results.values() if v['acc'] is not None)
    print(f"Total: {finished}/{total} with results")


if __name__ == '__main__':
    main()
