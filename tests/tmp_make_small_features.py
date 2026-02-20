import shutil
from pathlib import Path
p = Path('datasets/features')
out = Path('datasets/features_small')
train_out = out / 'train'
val_out = out / 'val'
train_out.mkdir(parents=True, exist_ok=True)
val_out.mkdir(parents=True, exist_ok=True)
train_files = sorted((p/'train').glob('*.npz'))
val_files = sorted((p/'val').glob('*.npz'))
print('Available train files:', len(train_files))
print('Available val files:', len(val_files))
for f in train_files[:200]:
    shutil.copy(str(f), str(train_out / f.name))
for f in val_files[:50]:
    shutil.copy(str(f), str(val_out / f.name))
print('Copied')
