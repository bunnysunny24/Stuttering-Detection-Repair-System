import numpy as np
import glob, sys

paths = glob.glob('datasets/features/train/*.npz')
if not paths:
    paths = glob.glob('datasets/features/**/*.npz', recursive=True)
if not paths:
    print('No NPZs found under datasets/features'); sys.exit(2)

fp = paths[0]
print('Using:', fp)
npz = np.load(fp)
if 'spectrogram' not in npz:
    print('Missing key spectrogram in', fp); sys.exit(2)
spec = npz['spectrogram']
labels = npz.get('labels', None)
print('spectrogram shape, dtype:', spec.shape, spec.dtype)
print('labels:', None if labels is None else (labels.shape, labels.dtype))

# basic sanity checks
if spec.ndim != 2 or spec.shape[0] != 123:
    print('Unexpected spectrogram shape; expected (123, T)'); sys.exit(2)

# dummy numpy forward pass: mean-pool over time -> linear -> sigmoid
x = spec.mean(axis=1)  # (123,)
W = np.random.RandomState(0).normal(size=(123, (labels.size if labels is not None else 5)))
logits = x.dot(W)
probs = 1.0 / (1.0 + np.exp(-logits))
print('probs shape:', probs.shape)

if labels is not None:
    lab = labels.astype(float)
    lab = lab.flatten()[:probs.size]
    loss = - (lab * np.log(probs + 1e-9) + (1 - lab) * np.log(1 - probs + 1e-9)).mean()
    print('dummy BCE loss:', float(loss))

print('SMOKE (no-torch) OK')
