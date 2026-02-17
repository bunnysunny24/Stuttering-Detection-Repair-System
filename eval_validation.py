import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_improved_90plus import ImprovedStutteringCNN
from constants import TOTAL_CHANNELS, NUM_CLASSES


def load_npz(path: Path):
    data = np.load(path)
    if 'spectrogram' in data:
        spec = data['spectrogram']
    else:
        return None, None
    if 'labels' in data:
        labels = data['labels']
    else:
        return None, None
    return spec.astype(np.float32), labels.astype(np.float32)


def pad_to_length(x, length):
    if x.shape[1] >= length:
        return x[:, :length]
    pad = length - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode='constant')


def main(args):
    val_dir = Path(args.data_dir) / 'val'
    ckpt = Path(args.checkpoint)
    out_dir = Path('output') / f'eval_{ckpt.stem}'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    model = ImprovedStutteringCNN(n_channels=TOTAL_CHANNELS, n_classes=NUM_CLASSES)
    state = torch.load(str(ckpt), map_location='cpu')
    # Support both state dict and wrapped dict
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    try:
        model.load_state_dict(state)
    except Exception:
        # try tolerant load (strip prefix)
        new = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(new)

    model.to(device)
    model.eval()

    files = sorted(val_dir.glob('**/*.npz'))
    if len(files) == 0:
        print('No validation NPZs found in', val_dir)
        return

    all_probs = []
    all_labels = []
    max_len = 0
    # first pass compute max length to pad
    for p in files:
        spec, labels = load_npz(p)
        if spec is None:
            continue
        max_len = max(max_len, spec.shape[1])

    with torch.no_grad():
        for p in tqdm(files, desc='Eval'):
            spec, labels = load_npz(p)
            if spec is None:
                continue
            # ensure channel count
            if spec.shape[0] != TOTAL_CHANNELS:
                if spec.shape[0] < TOTAL_CHANNELS:
                    spec = np.pad(spec, ((0, TOTAL_CHANNELS - spec.shape[0]), (0, 0)), mode='constant')
                else:
                    spec = spec[:TOTAL_CHANNELS, :]
            spec = pad_to_length(spec, max_len)
            x = torch.from_numpy(spec).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # compute metrics - treat any label > 0 as positive (binary conversion)
    from sklearn.metrics import roc_auc_score, average_precision_score

    bin_labels = (all_labels > 0).astype(int)
    aucs = []
    aps = []
    for i in range(bin_labels.shape[1]):
        y_true = bin_labels[:, i]
        y_score = all_probs[:, i]
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = float('nan')
        try:
            ap = average_precision_score(y_true, y_score)
        except Exception:
            ap = float('nan')
        aucs.append(auc)
        aps.append(ap)

    print('Per-class AUC (binary labels):', aucs)
    print('Per-class AP  (binary labels):', aps)

    classes = ['Prolongation', 'Block', 'Sound Repetition', 'Word Repetition', 'Interjection']
    # save histograms
    for i, cname in enumerate(classes):
        plt.figure(figsize=(6, 4))
        plt.hist(all_probs[:, i], bins=50, alpha=0.7, label='probs')
        plt.title(f'{cname} — AUC={aucs[i]:.4f} AP={aps[i]:.4f}')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'hist_{i}_{cname.replace(" ","_")}.png')
        plt.close()

    # save ROC and PR curves
    from sklearn.metrics import roc_curve, precision_recall_curve

    for i, cname in enumerate(classes):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]
        plt.figure(figsize=(6, 6))
        # ROC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.plot(fpr, tpr, label=f'ROC (AUC={aucs[i]:.3f})')
        except Exception:
            pass
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC — {cname}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'roc_{i}_{cname.replace(" ","_")}.png')
        plt.close()

        # PR
        plt.figure(figsize=(6, 6))
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt.plot(recall, precision, label=f'PR (AP={aps[i]:.3f})')
        except Exception:
            pass
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall — {cname}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f'pr_{i}_{cname.replace(" ","_")}.png')
        plt.close()

    # Save numeric results
    import json
    json.dump({'auc': aucs, 'ap': aps}, open(out_dir / 'metrics.json', 'w'))
    print('Saved plots and metrics to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/features', help='Features root dir')
    parser.add_argument('--checkpoint', default='Models/checkpoints/training_20260217_011229/improved_90plus_best.pth')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    main(args)
