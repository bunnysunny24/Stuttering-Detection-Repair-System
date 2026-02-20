import numpy as np
from pathlib import Path
import os

# Load mapping via the project's loader
from Models.extract_features_90plus import FeatureExtractionManager

mgr = FeatureExtractionManager()
if len(mgr.mappings) == 0:
    print('No label mappings found; aborting')
    exit(1)

for split in ('train','val'):
    p = Path(f'datasets/features/{split}')
    files = list(p.glob('*.npz'))
    updated = 0
    for f in files:
        name = f.stem
        if name in mgr.mappings:
            labels = mgr.mappings[name]
            try:
                npz = dict(np.load(f))
                npz['labels'] = labels.astype(np.float32)
                np.savez_compressed(f, **npz)
                updated += 1
            except Exception as e:
                print('ERR', f, e)
    print(f"{split}: updated {updated}/{len(files)} files")
