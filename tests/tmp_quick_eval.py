import torch
import numpy as np
from pathlib import Path
import json

# Try to import models
from Models.model_cnn_bilstm import CNNBiLSTM
from Models.model_improved_90plus import ImprovedStutteringCNN

CKPT = Path('Models/checkpoints/training_20260219_015326/cnn_bilstm_best.pth')
VAL_DIR = Path('datasets/features/val')
MAX_SAMPLES = 1000
BATCH = 64

# Load threshold if available
thresh_path = Path('output/thresholds.json')
temperature = 1.0
if thresh_path.exists():
    try:
        tdata = json.loads(thresh_path.read_text())
        # Support several possible formats written by calibration script
        if isinstance(tdata, dict):
            # List under 'thresholds'
            if 'thresholds' in tdata and isinstance(tdata['thresholds'], (list, tuple)):
                thresholds = np.array(tdata['thresholds'], dtype=float)
            # Dict mapping class names -> value
            elif all(k in tdata for k in ['Prolongation','Block','Sound Repetition','Word Repetition','Interjection']):
                thresholds = np.array([tdata.get(k, 0.5) for k in ['Prolongation','Block','Sound Repetition','Word Repetition','Interjection']], dtype=float)
            else:
                # Fallback: try to extract numeric values from top-level dict
                vals = [v for v in tdata.values() if isinstance(v, (int, float))]
                if len(vals) >= 5:
                    thresholds = np.array(vals[:5], dtype=float)
                else:
                    thresholds = np.ones(5) * 0.5
            # Temperature scaling support
            if 'temperature' in tdata:
                try:
                    temperature = float(tdata['temperature'])
                except Exception:
                    temperature = 1.0
        else:
            thresholds = np.ones(5) * 0.5
    except Exception:
        thresholds = np.ones(5) * 0.5
else:
    thresholds = np.ones(5)*0.5

print('Using thresholds:', thresholds)

# Load checkpoint state_dict
state = torch.load(str(CKPT), map_location='cpu')
# Attempt to detect model shape
loaded_ok = False
for ModelCls, kwargs in [(CNNBiLSTM, {'in_channels':123,'n_classes':5}), (ImprovedStutteringCNN, {'n_channels':123,'n_classes':5})]:
    try:
        model = ModelCls(**{k:v for k,v in kwargs.items() if k in ModelCls.__init__.__code__.co_varnames})
        model.load_state_dict(state)
        print('Loaded checkpoint into', ModelCls.__name__)
        loaded_ok = True
        break
    except Exception as e:
        # try state dict prefixes removal
        try:
            sd = {k.replace('module.',''):v for k,v in state.items()}
            model = ModelCls(**{k:v for k,v in kwargs.items() if k in ModelCls.__init__.__code__.co_varnames})
            model.load_state_dict(sd)
            print('Loaded checkpoint into', ModelCls.__name__, '(after prefix strip)')
            loaded_ok = True
            break
        except Exception as e2:
            # continue
            continue

if not loaded_ok:
    print('Failed to load checkpoint into known models; exiting')
    raise SystemExit(1)

model.eval()

# Collect val files
files = list(VAL_DIR.glob('*.npz'))
if len(files) == 0:
    print('No val files found in', VAL_DIR)
    raise SystemExit(1)
files = files[:min(len(files), MAX_SAMPLES)]

all_true = []
all_probs = []

with torch.no_grad():
    for i in range(0, len(files), BATCH):
        batch_files = files[i:i+BATCH]
        specs = []
        trues = []
        for f in batch_files:
            d = np.load(f)
            spec = d['spectrogram']
            lbl = d.get('labels')
            if lbl is None:
                lbl = np.zeros(5,dtype=np.float32)
            else:
                arr = np.asarray(lbl)
                if arr.size!=5:
                    arr = (arr>0).astype(np.float32)
                    tmp = np.zeros(5,dtype=np.float32)
                    tmp[:min(len(arr),5)] = arr[:min(len(arr),5)]
                    arr = tmp
                else:
                    arr = (arr>0).astype(np.float32)
                lbl = arr
            # ensure shape (channels, time)
            if spec.ndim==1:
                spec = spec[np.newaxis,:]
            if spec.shape[0] != 123:
                if spec.shape[0] < 123:
                    pad = 123 - spec.shape[0]
                    spec = np.pad(spec, ((0,pad),(0,0)), mode='constant')
                else:
                    spec = spec[:123,:]
            specs.append(spec)
            trues.append(lbl)
        # Pad variable-length spectrograms in this batch to the same time dimension
        max_time = max(s.shape[1] for s in specs)
        padded = []
        for s in specs:
            if s.shape[1] < max_time:
                pad_width = max_time - s.shape[1]
                s = np.pad(s, ((0,0),(0,pad_width)), mode='constant')
            padded.append(s)
        X = torch.from_numpy(np.stack(padded, axis=0)).float()
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_true.append(np.stack(trues,axis=0))

all_probs = np.vstack(all_probs)
all_true = np.vstack(all_true)

print('Shape true, probs:', all_true.shape, all_probs.shape)

# Per-class stats
pos_rate = (all_probs > thresholds).mean(axis=0)
mean_prob = all_probs.mean(axis=0)
true_pos_frac = all_true.mean(axis=0)

print('positive prediction rate (at thresholds):', pos_rate)
print('mean prob per class:', mean_prob)
print('true positive fraction per class:', true_pos_frac)

# Compute ROC AUC per class via rank method
auc = []
for c in range(all_true.shape[1]):
    y_true = all_true[:,c]
    y_score = all_probs[:,c]
    if len(np.unique(y_true))>1:
        # ranks
        ranks = np.argsort(np.argsort(y_score)) + 1
        n_pos = int((y_true==1).sum())
        n_neg = len(y_true)-n_pos
        sum_ranks_pos = ranks[y_true==1].sum()
        auc_c = (sum_ranks_pos - n_pos*(n_pos+1)/2.0) / (n_pos*n_neg)
    else:
        auc_c = 0.5
    auc.append(float(auc_c))

print('AUC per class:', auc)

# Save quick report
report = {
    'files_evaluated': len(files),
    'thresholds': thresholds.tolist(),
    'pos_rate': pos_rate.tolist(),
    'mean_prob': mean_prob.tolist(),
    'true_pos_frac': true_pos_frac.tolist(),
    'auc': auc
}
Path('output').mkdir(exist_ok=True)
Path('output/quick_eval').mkdir(exist_ok=True)
Path('output/quick_eval/report.json').write_text(json.dumps(report,indent=2))
print('Report saved to output/quick_eval/report.json')
