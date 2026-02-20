import numpy as np
from pathlib import Path
p = Path('datasets/features')
files = list((p/'train').glob('*.npz')) + list((p/'val').glob('*.npz'))
print('Total npz files:', len(files))
high = []
extreme = []
for f in files:
    try:
        d = np.load(f)
        spec = d['spectrogram']
        ch_vars = np.var(spec, axis=1)
        silent = np.sum(ch_vars < 1e-8)
        ratio = silent / spec.shape[0]
        if ratio > 0.8:
            high.append((f.name, ratio))
            if ratio > 0.95:
                extreme.append((f.name, ratio))
    except Exception:
        continue
print('High-silence (>80%) count:', len(high))
print('Extreme-silence (>95%) count:', len(extreme))
print('\nFirst 20 high-silence samples:')
for name, r in high[:20]:
    print(f"{name}: {r*100:.1f}%")
