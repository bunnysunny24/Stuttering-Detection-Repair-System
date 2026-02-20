import numpy as np
from pathlib import Path

def scan(dirpath):
    p = Path(dirpath)
    files = list(p.glob('*.npz'))
    sums = None
    any_pos = 0
    for f in files:
        try:
            d = np.load(f)
            labels = d['labels'] if 'labels' in d else np.zeros(5)
            labels = np.asarray(labels).astype(int)
            if sums is None:
                sums = np.zeros_like(labels)
            sums += labels
            if labels.sum() > 0:
                any_pos += 1
        except Exception as e:
            print('ERR', f, e)
    return len(files), sums if sums is not None else np.zeros(5, int), any_pos

for split in ('train','val'):
    total, sums, any_pos = scan(f'datasets/features/{split}')
    print(f"{split.upper()}: files={total}, positive_counts_per_class={sums.tolist()}, files_with_any_positive={any_pos}")
