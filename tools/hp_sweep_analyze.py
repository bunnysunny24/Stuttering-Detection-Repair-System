#!/usr/bin/env python3
"""Analyze `output/hp_sweep_results.json` and optionally rerun the best trial.

Usage:
  python tools/hp_sweep_analyze.py --top 5
  python tools/hp_sweep_analyze.py --rerun --epochs 10 --num-workers 0
"""
import argparse
import json
import sys
import subprocess
from pathlib import Path

OUT = Path('output')
RESULTS = OUT / 'hp_sweep_results.json'


def load_results():
    if not RESULTS.exists():
        print('No hp_sweep_results.json found at', RESULTS)
        sys.exit(1)
    try:
        data = json.load(open(RESULTS))
    except Exception as e:
        print('Failed to load results:', e)
        sys.exit(1)
    return data


def print_top(data, top=5):
    scored = [d for d in data if d.get('val_f1') is not None]
    scored.sort(key=lambda x: x.get('val_f1', -1), reverse=True)
    print(f'Found {len(data)} trials ({len(scored)} with val_f1). Top {top}:')
    for i, r in enumerate(scored[:top]):
        lr = r.get('lr')
        wd = r.get('wd')
        lr_s = f"{lr:.6g}" if lr is not None else "None"
        wd_s = f"{wd:.6g}" if wd is not None else "None"
        print(f"#{i+1}: trial={r.get('trial')} val_f1={r.get('val_f1')} lr={lr_s} wd={wd_s} batch={r.get('batch')} metrics={r.get('metrics_file')} log={r.get('log_file')}")
    return scored


def rerun_best(best, epochs=10, num_workers=0, extra_args=None):
    if best is None:
        print('No best trial to rerun')
        return 1
    lr = best.get('lr')
    wd = best.get('wd')
    batch = best.get('batch')
    cmd = [sys.executable, 'Models/train_90plus_final.py', '--epochs', str(epochs), '--batch-size', str(batch), '--lr', str(lr), '--weight-decay', str(wd), '--num-workers', str(num_workers), '--use-ema', '--verbose']
    if extra_args:
        cmd += extra_args.split()
    print('Rerunning best with command:', ' '.join(cmd))
    rc = subprocess.run(cmd)
    print('Rerun finished with rc=', rc.returncode)
    return rc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top', type=int, default=5, help='Show top N trials')
    p.add_argument('--rerun', action='store_true', help='Rerun the best trial')
    p.add_argument('--epochs', type=int, default=10, help='Epochs for rerun')
    p.add_argument('--num-workers', type=int, default=0, help='num workers for rerun')
    p.add_argument('--extra-args', type=str, default=None, help='Extra args to forward to training')
    args = p.parse_args()

    data = load_results()
    scored = print_top(data, top=args.top)
    if args.rerun:
        best = scored[0] if scored else None
        rc = rerun_best(best, epochs=args.epochs, num_workers=args.num_workers, extra_args=args.extra_args)
        # save selected info
        sel = OUT / 'hp_sweep_selected.json'
        sel.write_text(json.dumps({'best': best, 'rc': rc}, indent=2))
        print('Wrote selected info to', sel)


if __name__ == '__main__':
    main()
