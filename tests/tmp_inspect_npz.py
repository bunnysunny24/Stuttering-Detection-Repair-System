import numpy as np
from pathlib import Path
p = Path('datasets/features/train').glob('*.npz')
for i,f in enumerate(p):
    if i>0: break
    print('file:', f)
    d = np.load(f)
    print('keys:', list(d.keys()))
    if 'labels' in d:
        labels = d['labels']
        print('labels type:', type(labels), 'shape:', getattr(labels,'shape',None))
        try:
            arr = np.asarray(labels)
            print('labels array dtype:', arr.dtype, 'contents:', arr)
        except Exception as e:
            print('labels -> failed to convert:', e)
    else:
        print('no labels key')
    if 'spectrogram' in d:
        spec = d['spectrogram']
        print('spec dtype:', spec.dtype, 'shape:', spec.shape)
    else:
        print('no spectrogram')
