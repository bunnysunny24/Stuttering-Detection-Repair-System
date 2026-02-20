import numpy as np
from pathlib import Path
p = Path('datasets/features/train')
files = list(p.glob('*.npz'))
NUM=5
counts = np.zeros(NUM,dtype=int)
total=0
for f in files:
    d=np.load(f)
    lbl=d.get('labels')
    if lbl is None:
        continue
    arr = np.asarray(lbl)
    if arr.size!=NUM:
        arr = (arr>0).astype(int)
        tmp = np.zeros((NUM,),dtype=int)
        tmp[:min(len(arr),NUM)] = arr[:min(len(arr),NUM)]
        arr = tmp
    else:
        arr = (arr>0).astype(int)
    counts += arr
    total += 1
print('files:', total)
print('positive counts per class:', counts)
print('positive fractions:', counts/float(total))
