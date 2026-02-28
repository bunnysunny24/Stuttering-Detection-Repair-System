import numpy as np
import torch
from pathlib import Path
from model_improved_90plus import ImprovedStutteringCNN
from constants import TOTAL_CHANNELS, NUM_CLASSES

def load_npz(path: Path):
    d = np.load(path)
    if 'spectrogram' in d and 'labels' in d:
        return d['spectrogram'].astype(np.float32), d['labels'].astype(np.float32)
    return None, None

def pad_to_length(x, length):
    if x.shape[1] >= length:
        return x[:, :length]
    return np.pad(x, ((0,0),(0,length-x.shape[1])), mode='constant')

def main():
    ckpt = Path('Models/checkpoints/training_20260217_011229/improved_90plus_best.pth')
    model = ImprovedStutteringCNN(n_channels=TOTAL_CHANNELS, n_classes=NUM_CLASSES)
    state = torch.load(str(ckpt), map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict({k.replace('module.',''):v for k,v in state.items()})
    model.eval()

    files = sorted(Path('datasets/features').joinpath('val').glob('**/*.npz'))
    max_len = 0
    for p in files:
        s,_ = load_npz(p)
        if s is None: continue
        max_len = max(max_len, s.shape[1])

    probs = []
    labels = []
    with torch.no_grad():
        for p in files:
            s, l = load_npz(p)
            if s is None: continue
            if s.shape[0] != TOTAL_CHANNELS:
                if s.shape[0] < TOTAL_CHANNELS:
                    s = np.pad(s, ((0,TOTAL_CHANNELS-s.shape[0]),(0,0)), mode='constant')
                else:
                    s = s[:TOTAL_CHANNELS,:]
            s = pad_to_length(s, max_len)
            x = torch.from_numpy(s).unsqueeze(0)
            out = torch.sigmoid(model(x)).numpy()[0]
            probs.append(out)
            labels.append(l)

    probs = np.vstack(probs)
    labels = np.vstack(labels)

    print('probs shape', probs.shape)
    print('labels shape', labels.shape)
    for i in range(labels.shape[1]):
        col = probs[:,i]
        lab = labels[:,i]
        print(i, 'min,max,mean,std,nan_count,unique_labels:', float(col.min()), float(col.max()), float(col.mean()), float(col.std()), int(np.isnan(col).sum()), np.unique(lab).tolist())

    # compute binary metrics treating any label>0 as positive
    from sklearn.metrics import roc_auc_score, average_precision_score
    aucs = []
    aps = []
    binary = (labels > 0).astype(int)
    for i in range(labels.shape[1]):
        try:
            aucs.append(roc_auc_score(binary[:,i], probs[:,i]))
        except Exception:
            aucs.append(float('nan'))
        try:
            aps.append(average_precision_score(binary[:,i], probs[:,i]))
        except Exception:
            aps.append(float('nan'))
    print('Binary per-class AUC:', aucs)
    print('Binary per-class AP :', aps)

if __name__ == '__main__':
    main()
