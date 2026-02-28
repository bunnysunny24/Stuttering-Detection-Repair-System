import numpy as np
from pathlib import Path

def load_npz(p):
    d = np.load(p)
    if 'labels' in d:
        return d['labels']
    return None

def main():
    val_dir = Path('datasets/features') / 'val'
    files = sorted(val_dir.glob('**/*.npz'))
    counts = None
    total = 0
    for p in files:
        lbl = load_npz(p)
        if lbl is None:
            continue
        lbl = np.asarray(lbl)
        if counts is None:
            counts = np.zeros_like(lbl, dtype=int)
        counts += (lbl > 0.5).astype(int)
        total += 1
    print('Files with labels:', total)
    if counts is None:
        print('No labels found')
        return
    print('Per-class positive counts:', counts.tolist())
    print('Per-class positive rates:', (counts / total).tolist())

if __name__ == '__main__':
    main()
