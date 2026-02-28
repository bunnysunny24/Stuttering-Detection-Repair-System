"""
Compute label counts and BCE pos_weight (per-class) from feature files.
Usage: python tools/compute_class_weights.py --data-dir datasets/features
"""
import argparse
import numpy as np
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='datasets/features', help='Directory with .npz/.npy feature files')
    args = parser.parse_args()

    p = Path(args.data_dir)
    files = sorted([str(x) for x in p.glob('**/*.npz')]) + sorted([str(x) for x in p.glob('**/*.npy')])
    counts = None
    total = 0
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            lbl = None
            if 'labels' in d:
                lbl = d['labels']
            elif isinstance(d, np.ndarray) and d.dtype == object:
                # improbable, but try
                continue
            if lbl is None:
                continue
            lbl = np.asarray(lbl)
            if lbl.ndim == 1 and lbl.size > 1:
                # likely binary presence vector
                v = (lbl > 0).astype(np.int64)
            else:
                v = (lbl > 0).astype(np.int64)
            if counts is None:
                counts = np.zeros_like(v, dtype=np.int64)
            if v.size == counts.size:
                counts += v
            total += 1
        except Exception:
            continue

    if counts is None:
        print('No label arrays found in', args.data_dir)
        return

    print('Files scanned:', total)
    print('Per-class positive counts:', counts.tolist())
    counts_f = counts.astype(np.float64)
    counts_f = np.maximum(counts_f, 1.0)
    inv = (np.sum(counts_f) / counts_f)
    inv = inv / np.mean(inv)
    print('Suggested pos_weight (normalized inv-freq):', inv.tolist())


if __name__ == '__main__':
    main()
